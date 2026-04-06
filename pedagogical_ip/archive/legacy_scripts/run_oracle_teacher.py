"""
Demo script: Run episodes with the Oracle Teacher Policy.

Usage:
    python scripts/run_oracle_teacher.py
"""

from __future__ import annotations

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np

from src.envs.map_generator import generate_default_map
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_teacher import OracleTeacherPolicy
from src.teachers.interventions import InterventionType
from src.logging.episode_logger import EpisodeLogger
from src.logging.visualize import plot_belief_heatmap
from src.metrics.online_metrics import epistemic_gain, frustration_score


def load_config(config_dir: str) -> dict:
    """Load all YAML configs and merge."""
    cfg = {}
    for name in ["env", "agent", "teacher", "experiment"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg[name] = yaml.safe_load(f)
    return cfg


def run_experiment():
    config_dir = os.path.join(PROJECT_ROOT, "configs")
    cfg = load_config(config_dir)

    exp_cfg = cfg.get("experiment", {})
    num_episodes = exp_cfg.get("num_episodes", 20)
    seed = exp_cfg.get("seed", 42)
    log_dir = os.path.join(PROJECT_ROOT, exp_cfg.get("log_dir", "output/logs"))
    viz_dir = os.path.join(PROJECT_ROOT, exp_cfg.get("viz_dir", "output/viz"))
    save_npz = exp_cfg.get("save_npz", True)

    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})
    teacher_cfg = cfg.get("teacher", {})

    # Create map
    grid_map = generate_default_map()

    # Create environment
    env = PedagogicalGridEnv(
        grid_map=grid_map,
        max_steps=env_cfg.get("episode", {}).get("max_steps", 60),
        initial_risk_budget=env_cfg.get("episode", {}).get("initial_risk_budget", 1.0),
        shield_duration=env_cfg.get("episode", {}).get("shield_duration", 5),
        risk_trigger_prob=env_cfg.get("terrain", {}).get("risk_trigger_prob", 0.3),
        risk_trigger_prob_shield=env_cfg.get("terrain", {}).get("risk_trigger_prob_shield", 0.02),
        # Agent params
        prior_cost_mean=agent_cfg.get("belief", {}).get("prior_cost_mean", 1.5),
        prior_cost_var=agent_cfg.get("belief", {}).get("prior_cost_var", 4.0),
        prior_risk_mean=agent_cfg.get("belief", {}).get("prior_risk_mean", 0.1),
        prior_risk_var=agent_cfg.get("belief", {}).get("prior_risk_var", 0.25),
        self_noise_var=agent_cfg.get("observation", {}).get("self_cell_noise_var", 0.001),
        neighbor_noise_var=agent_cfg.get("observation", {}).get("neighbor_noise_var", 1.0),
        neighbor_radius=agent_cfg.get("observation", {}).get("neighbor_radius", 1),
        search_budget=agent_cfg.get("planner", {}).get("search_budget", 30),
        lambda_risk=agent_cfg.get("planner", {}).get("lambda_risk", 3.0),
        lambda_uncertainty=agent_cfg.get("planner", {}).get("lambda_uncertainty", 0.5),
        seed=seed,
        render_mode="ansi",
    )

    # Create teacher
    util_w = teacher_cfg.get("utility_weights", {})
    int_costs = teacher_cfg.get("intervention_costs", {})
    teacher = OracleTeacherPolicy(
        w_success=util_w.get("w_success", 1.0),
        w_learn=util_w.get("w_learn", 0.3),
        w_cost=util_w.get("w_cost", 0.2),
        w_takeover=util_w.get("w_takeover", 0.1),
        intervention_costs=int_costs,
        rollout_budget=teacher_cfg.get("rollout", {}).get("depth", 30),
    )

    # Create logger
    logger = EpisodeLogger(log_dir=log_dir, save_npz=save_npz)

    # --- Run episodes ---
    rng = np.random.default_rng(seed)
    results: list[dict] = []

    print(f"{'='*60}")
    print(f"  Pedagogical GridWorld v0 — Oracle Teacher Demo")
    print(f"  Episodes: {num_episodes}  |  Grid: {grid_map.height}×{grid_map.width}")
    print(f"{'='*60}\n")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        logger.start_episode(ep)

        ep_reward = 0.0
        ep_interventions: dict[str, int] = {}
        terminated = truncated = False

        while not terminated and not truncated:
            belief_before = env.agent.belief.copy()

            # Teacher selects action
            passable = env._passable_mask()
            intervention, teacher_info = teacher.select_action(
                agent=env.agent,
                true_cost=env._true_cost_dynamic,
                true_risk=env.grid_map.true_risk,
                goal=env._current_goal(),
                time_left=env.max_steps - env.step_count,
                risk_budget_left=env.risk_budget_left,
                passable_mask=passable,
                door_positions=env.grid_map.door_positions,
                locked_doors=env.locked_doors,
            )

            # Map intervention to action index
            action_idx = {
                InterventionType.WAIT: 0,
                InterventionType.WARN: 1,
                InterventionType.UNLOCK_DOOR: 2,
                InterventionType.DROP_SHIELD: 3,
            }[intervention.type]

            # Set warn message if needed
            if intervention.type == InterventionType.WARN:
                env.warn_message = intervention.param

            # Step environment
            obs, reward, terminated, truncated, step_info = env.step(action_idx)
            ep_reward += reward

            # Compute metrics
            eg = epistemic_gain(belief_before, env.agent.belief)
            fs = frustration_score(
                env.agent.replan_count,
                env.max_steps - env.step_count,
                env.max_steps,
                env.risk_budget_left,
                env.initial_risk_budget,
            )

            # Track interventions
            itype = intervention.type.value
            ep_interventions[itype] = ep_interventions.get(itype, 0) + 1

            # Log
            logger.log_step(
                episode_id=ep,
                step=step_info["step"],
                robot_action=intervention.to_dict(),
                agent_action=step_info.get("agent_action", "STAY"),
                agent_pos_before=step_info.get("agent_pos_before", [0, 0]),
                agent_pos_after=step_info.get("agent_pos_after", [0, 0]),
                true_cost=step_info.get("true_cost", 1.0),
                true_risk=step_info.get("true_risk", 0.0),
                time_left=step_info.get("time_left", 0),
                risk_budget_left=step_info.get("risk_budget_left", 0.0),
                has_object=step_info.get("object_picked", False),
                has_shield=env.agent.has_shield,
                terminated=terminated,
                truncated=truncated,
                reward=reward,
                epistemic_gain=eg,
                frustration_score=fs,
                teacher_info=teacher_info,
                belief_snapshot=step_info.get("belief_snapshot"),
                true_cost_map=env._true_cost_dynamic,
                true_risk_map=env.grid_map.true_risk,
            )

        logger.end_episode()

        # Episode summary
        outcome = "SUCCESS" if step_info.get("object_delivered") else (
            "DEATH" if terminated else "TIMEOUT"
        )
        print(
            f"  Episode {ep:3d}  |  {outcome:8s}  |  "
            f"Steps: {env.step_count:3d}  |  "
            f"Reward: {ep_reward:7.2f}  |  "
            f"Interventions: {ep_interventions}"
        )

        results.append({
            "episode": ep,
            "outcome": outcome,
            "steps": env.step_count,
            "reward": ep_reward,
            "interventions": ep_interventions,
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    successes = sum(1 for r in results if r["outcome"] == "SUCCESS")
    deaths = sum(1 for r in results if r["outcome"] == "DEATH")
    timeouts = sum(1 for r in results if r["outcome"] == "TIMEOUT")
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"  Results: {successes} success / {deaths} death / {timeouts} timeout")
    print(f"  Avg reward: {avg_reward:.2f}  |  Avg steps: {avg_steps:.1f}")
    print(f"  Logs saved to: {log_dir}")

    # --- Save one visualization ---
    # Use the last episode's last step
    last_ep = num_episodes - 1
    npz_dir = os.path.join(log_dir, f"episode_{last_ep:04d}_npz")
    if os.path.isdir(npz_dir):
        npz_files = sorted(
            [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
        )
        if npz_files:
            last_npz = os.path.join(npz_dir, npz_files[-1])
            data = np.load(last_npz)
            viz_path = os.path.join(viz_dir, f"belief_final_ep{last_ep}.png")
            plot_belief_heatmap(
                belief_cost_mean=data["belief_cost_mean"],
                belief_cost_var=data["belief_cost_var"],
                belief_risk_mean=data["belief_risk_mean"],
                belief_risk_var=data["belief_risk_var"],
                true_cost=data.get("true_cost_map"),
                true_risk=data.get("true_risk_map"),
                agent_pos=tuple(env.agent.pos),
                step=env.step_count,
                save_path=viz_path,
            )
            print(f"  Visualization saved to: {viz_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    run_experiment()
