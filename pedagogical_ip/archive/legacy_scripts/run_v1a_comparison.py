"""
v1a Comparison Script.

Runs 4 baselines × N episodes each:
1. No Teacher    — agent navigates alone (all WAIT)
2. Always Help   — WARN at every step
3. Oracle        — Oracle teacher (v0, reads belief)
4. Particle      — SIPS-lite particle teacher (v1a, inference-based)

Prints comparison table + saves per-episode metrics.
"""

from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
from pathlib import Path

import numpy as np

from src.envs.map_generator import generate_default_map, generate_random_map
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_teacher import OracleTeacherPolicy
from src.teachers.particle_teacher import ParticleTeacherPolicy
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.rsa_warning import select_best_warning
from src.agents.belief import log_det_risk_var
from src.metrics.eval_v1 import (
    zero_shot_success_rate,
    epistemic_cost_efficiency,
    counterfactual_frustration_avoidance,
    tom_estimation_mse,
)


# ── Run one episode ──────────────────────────────────────────────────
def run_episode(
    env: PedagogicalGridEnv,
    teacher_mode: str,
    oracle_teacher: OracleTeacherPolicy | None = None,
    particle_teacher: ParticleTeacherPolicy | None = None,
    seed: int = 0,
) -> dict:
    """Run one episode, return summary dict."""
    obs, info = env.reset(seed=seed)
    belief_initial = env.agent.belief.copy()

    total_reward = 0.0
    total_cost = 0.0
    total_int_cost = 0.0
    interventions: dict[str, int] = {}
    frustration_wait: list[float] = []
    frustration_actual: list[float] = []
    tom_mse_steps: list[float] = []
    last_intervention: Intervention | None = None

    terminated = truncated = False

    while not terminated and not truncated:
        # Select action based on teacher mode
        if teacher_mode == "no_teacher":
            intervention = Intervention.wait()
            teacher_info = {}

        elif teacher_mode == "always_help":
            # Warn with best RSA utterance every step
            best_utt, _ = select_best_warning(
                env.agent.belief.risk_mean, env.agent.belief.risk_var,
                env.grid_map.true_risk, env.agent.pos,
                alpha=5.0, beta=0.1, tau=1.0,
            )
            intervention = Intervention.warn(best_utt)
            teacher_info = {}

        elif teacher_mode == "oracle":
            passable = env._passable_mask()
            intervention, teacher_info = oracle_teacher.select_action(
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

        elif teacher_mode == "particle":
            # Particle teacher: first update with last observed action
            if env.step_count > 0:
                passable = env._passable_mask()
                particle_teacher.update(
                    observed_action=env.agent.last_action,
                    agent_pos=env.agent.pos,
                    goal=env._current_goal(),
                    passable_mask=passable,
                    true_cost=env._true_cost_dynamic,
                    true_risk=env.grid_map.true_risk,
                    last_robot_action=last_intervention,
                )

            passable = env._passable_mask()
            intervention, teacher_info = particle_teacher.select_action(
                agent_pos=env.agent.pos,
                goal=env._current_goal(),
                true_risk=env.grid_map.true_risk,
                true_cost=env._true_cost_dynamic,
                time_left=env.max_steps - env.step_count,
                risk_budget_left=env.risk_budget_left,
                passable_mask=passable,
                locked_doors=env.locked_doors,
            )

            # ToM-MSE: compare particle estimate to oracle
            est = particle_teacher.get_estimated_belief()
            tom_mse_steps.append(
                tom_estimation_mse(est.risk_mean, env.agent.belief.risk_mean)
            )
        else:
            raise ValueError(f"Unknown teacher_mode: {teacher_mode}")

        # Map to action index
        action_idx = {
            InterventionType.WAIT: 0,
            InterventionType.WARN: 1,
            InterventionType.UNLOCK_DOOR: 2,
            InterventionType.DROP_SHIELD: 3,
        }[intervention.type]

        if intervention.type == InterventionType.WARN:
            env.warn_message = intervention.param

        # Track intervention
        itype = intervention.type.value
        interventions[itype] = interventions.get(itype, 0) + 1
        total_int_cost += {
            "WAIT": 0.0, "WARN": 0.1, "UNLOCK_DOOR": 0.3, "DROP_SHIELD": 0.5,
        }.get(itype, 0.0)

        last_intervention = intervention

        # Step
        obs, reward, terminated, truncated, step_info = env.step(action_idx)
        total_reward += reward
        total_cost += step_info.get("true_cost", 1.0) * 0.01

    belief_final = env.agent.belief.copy()

    outcome = "SUCCESS" if step_info.get("object_delivered") else (
        "DEATH" if terminated else "TIMEOUT"
    )

    # Compute metrics
    ece = epistemic_cost_efficiency(
        belief_initial, belief_final, total_cost, total_int_cost
    )

    return {
        "outcome": outcome,
        "steps": env.step_count,
        "reward": total_reward,
        "interventions": interventions,
        "ece": ece,
        "tom_mse": float(np.mean(tom_mse_steps)) if tom_mse_steps else None,
    }


# ── Main comparison ──────────────────────────────────────────────────
def run_comparison(
    num_episodes: int = 20,
    base_seed: int = 42,
):
    grid_map = generate_default_map()
    rng = np.random.default_rng(base_seed)

    # Create teachers
    oracle_teacher = OracleTeacherPolicy()
    particle_teacher = ParticleTeacherPolicy(
        height=grid_map.height, width=grid_map.width,
        n_particles=16,
        rng=np.random.default_rng(base_seed),
    )

    baselines = ["no_teacher", "always_help", "oracle", "particle"]
    all_results: dict[str, list[dict]] = {b: [] for b in baselines}

    print(f"{'=' * 72}")
    print(f"  v1a Comparison — {num_episodes} episodes × 4 baselines")
    print(f"  Grid: {grid_map.height}×{grid_map.width}")
    print(f"{'=' * 72}\n")

    for mode in baselines:
        print(f"  ── {mode.upper():20s} ──")
        for ep in range(num_episodes):
            ep_seed = int(rng.integers(0, 2**31))

            env = PedagogicalGridEnv(
                grid_map=grid_map,
                max_steps=60,
                seed=ep_seed,
                render_mode=None,
            )

            if mode == "particle":
                particle_teacher.rng = np.random.default_rng(ep_seed + 1)
                particle_teacher.reset()

            result = run_episode(
                env, mode,
                oracle_teacher=oracle_teacher if mode == "oracle" else None,
                particle_teacher=particle_teacher if mode == "particle" else None,
                seed=ep_seed,
            )
            all_results[mode].append(result)

            print(
                f"    Ep {ep:3d}: {result['outcome']:8s}  "
                f"steps={result['steps']:3d}  "
                f"reward={result['reward']:7.2f}  "
                f"ECE={result['ece']:6.3f}"
                + (f"  ToM-MSE={result['tom_mse']:.4f}" if result['tom_mse'] is not None else "")
            )

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  {'Baseline':<16s}  {'Success%':>8s}  {'AvgSteps':>8s}  "
          f"{'AvgReward':>9s}  {'AvgECE':>8s}  {'AvgToM-MSE':>10s}")
    print(f"  {'-' * 16}  {'-' * 8}  {'-' * 8}  {'-' * 9}  {'-' * 8}  {'-' * 10}")

    for mode in baselines:
        results = all_results[mode]
        n = len(results)
        success = sum(1 for r in results if r["outcome"] == "SUCCESS") / n * 100
        avg_steps = np.mean([r["steps"] for r in results])
        avg_reward = np.mean([r["reward"] for r in results])
        avg_ece = np.mean([r["ece"] for r in results])
        tom_vals = [r["tom_mse"] for r in results if r["tom_mse"] is not None]
        avg_tom = np.mean(tom_vals) if tom_vals else float("nan")

        print(
            f"  {mode:<16s}  {success:7.1f}%  {avg_steps:8.1f}  "
            f"{avg_reward:9.2f}  {avg_ece:8.3f}  {avg_tom:10.4f}"
        )

    print(f"{'=' * 72}")

    # Save results
    out_dir = Path(PROJECT_ROOT) / "output" / "v1a_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in baselines:
        out_file = out_dir / f"{mode}_results.json"
        serializable = []
        for r in all_results[mode]:
            sr = dict(r)
            sr["interventions"] = dict(sr["interventions"])
            serializable.append(sr)
        with open(out_file, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    run_comparison()
