"""
v1b Transfer Suite — Robot-free evaluation on unseen maps.

After interaction, freeze learner belief, remove teacher.
Test on 30 episodes with unseen layout seeds from the same family.
"""

from __future__ import annotations

import csv
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.envs.benchmark_generator import generate_benchmark_map, generate_transfer_map, DIFFICULTIES
from src.envs.map_families import FAMILY_NAMES, FamilyConfig
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_teacher import OracleTeacherPolicy
from src.teachers.particle_teacher import ParticleTeacherPolicy
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.rsa_warning import select_best_warning
from src.agents.belief import BeliefMap


BASELINES = ["no_teacher", "always_help", "oracle", "particle"]
SEEDS_PER_FAMILY = 5
INTERACTION_EPS = 10
TRANSFER_EPS = 30


def _make_env(grid_map, cfg: FamilyConfig, ep_seed: int) -> PedagogicalGridEnv:
    return PedagogicalGridEnv(
        grid_map=grid_map,
        max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget,
        seed=ep_seed,
    )


def run_interaction_phase(
    grid_map, cfg, mode, rng, oracle_teacher, base_seed,
) -> BeliefMap:
    """Run interaction episodes and return the learner's final belief."""
    pt = None
    if mode == "particle":
        pt = ParticleTeacherPolicy(
            height=grid_map.height, width=grid_map.width,
            n_particles=16,
            prior_risk_mean=cfg.prior_risk_mean,
            prior_risk_var=cfg.prior_risk_var,
            rng=np.random.default_rng(base_seed + 7),
        )

    final_belief = None
    last_int: Intervention | None = None

    for ep in range(INTERACTION_EPS):
        ep_seed = int(rng.integers(0, 2**31))
        env = _make_env(grid_map, cfg, ep_seed)
        obs, info = env.reset()

        if pt is not None:
            pt.rng = np.random.default_rng(ep_seed + 1)
            pt.reset()

        terminated = truncated = False
        while not terminated and not truncated:
            if mode == "no_teacher":
                intervention = Intervention.wait()
            elif mode == "always_help":
                best_utt, _ = select_best_warning(
                    env.agent.belief.risk_mean, env.agent.belief.risk_var,
                    env.grid_map.true_risk, env.agent.pos,
                )
                intervention = Intervention.warn(best_utt)
            elif mode == "oracle":
                passable = env._passable_mask()
                intervention, _ = oracle_teacher.select_action(
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
            elif mode == "particle":
                if env.step_count > 0:
                    passable = env._passable_mask()
                    pt.update(
                        observed_action=env.agent.last_action,
                        agent_pos=env.agent.pos,
                        goal=env._current_goal(),
                        passable_mask=passable,
                        true_cost=env._true_cost_dynamic,
                        true_risk=env.grid_map.true_risk,
                        last_robot_action=last_int,
                    )
                passable = env._passable_mask()
                intervention, _ = pt.select_action(
                    agent_pos=env.agent.pos,
                    goal=env._current_goal(),
                    true_risk=env.grid_map.true_risk,
                    true_cost=env._true_cost_dynamic,
                    time_left=env.max_steps - env.step_count,
                    risk_budget_left=env.risk_budget_left,
                    passable_mask=passable,
                    locked_doors=env.locked_doors,
                )
            else:
                intervention = Intervention.wait()

            action_idx = {
                InterventionType.WAIT: 0, InterventionType.WARN: 1,
                InterventionType.UNLOCK_DOOR: 2, InterventionType.DROP_SHIELD: 3,
            }[intervention.type]
            if intervention.type == InterventionType.WARN:
                env.warn_message = intervention.param
            last_int = intervention

            obs, reward, terminated, truncated, step_info = env.step(action_idx)

        final_belief = env.agent.belief.copy()

    return final_belief


def run_transfer_episode(
    grid_map, cfg: FamilyConfig, frozen_belief: BeliefMap, ep_seed: int,
) -> dict:
    """
    Run one transfer episode: frozen belief, no teacher (all WAIT).
    """
    env = _make_env(grid_map, cfg, ep_seed)
    obs, info = env.reset()

    # Inject frozen belief (transfer: use what was learned)
    env.agent.belief = frozen_belief.copy()

    terminated = truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        # No teacher: all WAIT
        obs, reward, terminated, truncated, step_info = env.step(0)
        total_reward += reward

    success = bool(step_info.get("object_delivered", False))
    return {
        "transfer_success": int(success),
        "steps": env.step_count,
        "reward": round(total_reward, 4),
    }


def run_transfer_suite(out_dir: str = "output/v1b_benchmark"):
    """Run the full transfer benchmark."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    oracle_teacher = OracleTeacherPolicy()

    all_rows: list[dict] = []

    print(f"{'=' * 80}")
    print(f"  v1b Transfer Suite — Robot-Free Evaluation")
    print(f"  {len(BASELINES)} baselines × {len(FAMILY_NAMES)} families × "
          f"{len(DIFFICULTIES)} difficulties × {SEEDS_PER_FAMILY} seeds")
    print(f"  Interaction: {INTERACTION_EPS} eps → Transfer: {TRANSFER_EPS} eps")
    print(f"{'=' * 80}\n")

    for family in FAMILY_NAMES:
        print(f"  ── Family: {family.upper()} ──")

        for diff in DIFFICULTIES:
            for seed_idx in range(SEEDS_PER_FAMILY):
                base_seed = seed_idx * 1000 + hash(family) % 10000

                # Generate interaction map
                grid_map, cfg = generate_benchmark_map(family, base_seed, diff)

                for mode in BASELINES:
                    # Phase 1: Interact → get frozen belief
                    interact_rng = np.random.default_rng(base_seed + 42)
                    frozen_belief = run_interaction_phase(
                        grid_map, cfg, mode, interact_rng,
                        oracle_teacher, base_seed,
                    )

                    # Phase 2: Transfer on unseen maps
                    n_succ = 0
                    for tep in range(TRANSFER_EPS):
                        transfer_seed = base_seed + 10000 + tep
                        t_map, t_cfg = generate_transfer_map(
                            family, transfer_seed, diff,
                        )
                        ep_seed = int(rng.integers(0, 2**31))
                        result = run_transfer_episode(
                            t_map, t_cfg, frozen_belief, ep_seed,
                        )
                        row = {
                            "family": family,
                            "difficulty": diff,
                            "seed": base_seed,
                            "transfer_ep": tep,
                            "baseline": mode,
                            **result,
                        }
                        all_rows.append(row)
                        n_succ += result["transfer_success"]

                    tsr = n_succ / TRANSFER_EPS * 100
                    print(f"    {diff:6s} seed={seed_idx} {mode:15s} TSR={tsr:5.1f}%")

    # Save CSV
    csv_path = out_path / "transfer_results.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  {'Family':<20s} {'Baseline':<16s} {'TSR%':>6s} {'AvgSteps':>8s}")
    print(f"  {'-' * 20} {'-' * 16} {'-' * 6} {'-' * 8}")
    for family in FAMILY_NAMES:
        for mode in BASELINES:
            frows = [r for r in all_rows
                     if r["family"] == family and r["baseline"] == mode]
            if not frows:
                continue
            tsr = sum(r["transfer_success"] for r in frows) / len(frows) * 100
            steps = np.mean([r["steps"] for r in frows])
            print(f"  {family:<20s} {mode:<16s} {tsr:5.1f}% {steps:8.1f}")
        print()
    print(f"{'=' * 70}")
    print(f"  Transfer results saved to: {csv_path}")

    return all_rows


if __name__ == "__main__":
    run_transfer_suite()
