"""
v1e Benchmark Suite — Interaction Phase.

Runs 6 baselines × 3 families × 3 difficulties × 5 seeds × 10 episodes.
Families: SemanticTrap, PlanningTrap, ExplorationUseful (skip Mixed).
Baselines: no_teacher, always_help, oracle, oracle_cause, particle.
Tracks cause diagnostics (CauseAcc, WarningPrecision, UnlockUseful, WaitSafety).
Saves per-family CSVs + aggregate summary.
"""

from __future__ import annotations

import csv
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.envs.benchmark_generator import generate_benchmark_map, DIFFICULTIES
from src.envs.map_families import FAMILY_NAMES, FamilyConfig
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_teacher import OracleTeacherPolicy
from src.teachers.oracle_cause_teacher import OracleCauseTeacherPolicy
from src.teachers.particle_teacher import ParticleTeacherPolicy
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.rsa_warning import select_best_warning
from src.agents.belief import log_det_risk_var
from src.metrics.eval_v1 import tom_estimation_mse


# v1e: skip Mixed, focus on A/B/C
V1E_FAMILIES = ["semantic_trap", "planning_trap", "exploration_useful"]
BASELINES = ["no_teacher", "always_help", "oracle", "oracle_cause", "particle"]
SEEDS_PER_FAMILY = 5
EPISODES_PER_SEED = 10


def _make_env(grid_map, cfg: FamilyConfig, ep_seed: int) -> PedagogicalGridEnv:
    """Create env with family-specific overrides."""
    return PedagogicalGridEnv(
        grid_map=grid_map,
        max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget,
        seed=ep_seed,
    )


def run_interaction_episode(
    env: PedagogicalGridEnv,
    mode: str,
    oracle_teacher: OracleTeacherPolicy,
    particle_teacher: ParticleTeacherPolicy | None,
    oracle_cause_teacher: OracleCauseTeacherPolicy | None = None,
    oracle_for_match: OracleTeacherPolicy | None = None,
    family_name: str = "",
) -> dict:
    """
    Run one interaction episode. Returns metrics dict.

    For 'particle' mode, also runs oracle in parallel for policy matching.
    """
    obs, info = env.reset()
    belief_init = env.agent.belief.copy()

    total_reward = 0.0
    total_cost = 0.0
    total_int_cost = 0.0
    interventions: dict[str, int] = {"WAIT": 0, "WARN": 0, "UNLOCK_DOOR": 0, "DROP_SHIELD": 0}
    tom_mse_steps: list[float] = []
    policy_matches = 0
    policy_total = 0
    cause_matches = 0
    cause_total = 0
    warn_on_risky = 0
    warn_total = 0
    unlock_useful_count = 0
    unlock_total = 0
    wait_safe_count = 0
    wait_total = 0
    last_int: Intervention | None = None

    terminated = truncated = False

    while not terminated and not truncated:
        # ── Select intervention ──
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
        elif mode == "oracle_cause":
            passable = env._passable_mask()
            allow_w = family_name != "planning_trap"
            allow_u = family_name not in ("semantic_trap", "exploration_useful")
            allow_s = False
            intervention, oc_info = oracle_cause_teacher.select_action(
                agent=env.agent,
                true_cost=env._true_cost_dynamic,
                true_risk=env.grid_map.true_risk,
                goal=env._current_goal(),
                time_left=env.max_steps - env.step_count,
                risk_budget_left=env.risk_budget_left,
                passable_mask=passable,
                door_positions=env.grid_map.door_positions,
                locked_doors=env.locked_doors,
                allow_warn=allow_w,
                allow_unlock=allow_u,
                allow_shield=allow_s,
            )
        elif mode == "particle":
            # Update particles with last observed action
            if env.step_count > 0:
                passable = env._passable_mask()
                particle_teacher.update(
                    observed_action=env.agent.last_action,
                    agent_pos=env.agent.pos,
                    goal=env._current_goal(),
                    passable_mask=passable,
                    true_cost=env._true_cost_dynamic,
                    true_risk=env.grid_map.true_risk,
                    last_robot_action=last_int,
                )
            passable = env._passable_mask()

            # v1d: per-family modality restrictions
            allow_w = family_name != "planning_trap"
            allow_u = family_name not in ("semantic_trap", "exploration_useful")
            allow_s = family_name == "mixed"

            intervention, pt_info = particle_teacher.select_action(
                agent_pos=env.agent.pos,
                goal=env._current_goal(),
                true_risk=env.grid_map.true_risk,
                true_cost=env._true_cost_dynamic,
                time_left=env.max_steps - env.step_count,
                risk_budget_left=env.risk_budget_left,
                passable_mask=passable,
                locked_doors=env.locked_doors,
                door_positions=env.grid_map.door_positions,
                allow_warn=allow_w,
                allow_unlock=allow_u,
                allow_shield=allow_s,
            )
            # ToM-MSE
            est = particle_teacher.get_estimated_belief()
            tom_mse_steps.append(
                tom_estimation_mse(est.risk_mean, env.agent.belief.risk_mean)
            )
            # Cause agreement with oracle_cause
            if oracle_cause_teacher is not None:
                oc_int, oc_info = oracle_cause_teacher.select_action(
                    agent=env.agent,
                    true_cost=env._true_cost_dynamic,
                    true_risk=env.grid_map.true_risk,
                    goal=env._current_goal(),
                    time_left=env.max_steps - env.step_count,
                    risk_budget_left=env.risk_budget_left,
                    passable_mask=passable,
                    door_positions=env.grid_map.door_positions,
                    locked_doors=env.locked_doors,
                    allow_warn=allow_w,
                    allow_unlock=allow_u,
                    allow_shield=allow_s,
                )
                cause_total += 1
                pt_cause = pt_info.get("dominant_cause", "")
                oc_cause = oc_info.get("dominant_cause", "")
                if pt_cause == oc_cause:
                    cause_matches += 1
                # Policy match
                policy_total += 1
                if oc_int.type == intervention.type:
                    policy_matches += 1

            # v1e diagnostic: WarningPrecision
            if intervention.type == InterventionType.WARN:
                warn_total += 1
                wp = pt_info.get("warn_precision", 1.0)
                warn_on_risky += int(wp >= 0.5)
            # v1e diagnostic: WaitSafety (approx: no failure within H steps)
            if intervention.type == InterventionType.WAIT:
                wait_total += 1
                # checked post-hoc below
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ── Execute ──
        action_idx = {
            InterventionType.WAIT: 0, InterventionType.WARN: 1,
            InterventionType.UNLOCK_DOOR: 2, InterventionType.DROP_SHIELD: 3,
        }[intervention.type]

        if intervention.type == InterventionType.WARN:
            env.warn_message = intervention.param

        itype = intervention.type.value
        interventions[itype] += 1
        int_costs = {"WAIT": 0.0, "WARN": 0.1, "UNLOCK_DOOR": 0.3, "DROP_SHIELD": 0.5}
        total_int_cost += int_costs.get(itype, 0.0)
        last_int = intervention

        obs, reward, terminated, truncated, step_info = env.step(action_idx)
        total_reward += reward
        total_cost += step_info.get("true_cost", 1.0) * 0.01

    belief_final = env.agent.belief.copy()

    # Constrained success
    success = bool(step_info.get("object_delivered", False))
    within_time = env.step_count <= env.max_steps
    within_risk = env.risk_budget_left > 0
    constrained_success = success and within_time and within_risk

    # ECE
    ld_0 = log_det_risk_var(belief_init)
    ld_T = log_det_risk_var(belief_final)
    ece = (ld_0 - ld_T) / (total_cost + 0.5 * total_int_cost + 1e-6)

    # v1e: WaitSafety — if last action was WAIT and episode failed, mark unsafe
    if mode == "particle" and last_int and last_int.type == InterventionType.WAIT:
        if constrained_success:
            wait_safe_count += 1

    return {
        "constrained_success": int(constrained_success),
        "success": int(success),
        "steps": env.step_count,
        "reward": round(total_reward, 4),
        "ece": round(ece, 4),
        "tom_mse": round(float(np.mean(tom_mse_steps)), 6) if tom_mse_steps else None,
        "policy_match": round(policy_matches / max(policy_total, 1), 4),
        "policy_total": policy_total,
        "cause_acc": round(cause_matches / max(cause_total, 1), 4) if cause_total > 0 else None,
        "warn_precision": round(warn_on_risky / max(warn_total, 1), 4) if warn_total > 0 else None,
        "wait_safety": round(wait_safe_count / max(wait_total, 1), 4) if wait_total > 0 else None,
        "n_wait": interventions["WAIT"],
        "n_warn": interventions["WARN"],
        "n_unlock": interventions["UNLOCK_DOOR"],
        "n_shield": interventions["DROP_SHIELD"],
    }


def run_benchmark(out_dir: str = "output/v1e_benchmark"):
    """Run the full interaction benchmark (v1e: A/B/C only)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    oracle_teacher = OracleTeacherPolicy()
    oc_teacher = OracleCauseTeacherPolicy()
    all_rows: list[dict] = []

    families = V1E_FAMILIES
    print(f"{'=' * 80}")
    print(f"  v1e Benchmark Suite — Interaction Phase")
    print(f"  {len(BASELINES)} baselines × {len(families)} families × "
          f"{len(DIFFICULTIES)} difficulties × {SEEDS_PER_FAMILY} seeds × "
          f"{EPISODES_PER_SEED} episodes")
    print(f"{'=' * 80}\n")

    for family in families:
        family_rows: list[dict] = []
        print(f"\n  ── Family: {family.upper()} ──")

        for diff in DIFFICULTIES:
            for seed_idx in range(SEEDS_PER_FAMILY):
                base_seed = seed_idx * 1000 + hash(family) % 10000
                grid_map, cfg = generate_benchmark_map(family, base_seed, diff)

                for mode in BASELINES:
                    # Create particle teacher for this condition
                    pt = None
                    if mode == "particle":
                        pt = ParticleTeacherPolicy(
                            height=grid_map.height,
                            width=grid_map.width,
                            n_particles=16,
                            prior_risk_mean=cfg.prior_risk_mean,
                            prior_risk_var=cfg.prior_risk_var,
                            rng=np.random.default_rng(base_seed + 7),
                        )

                    n_succ = 0
                    for ep in range(EPISODES_PER_SEED):
                        ep_seed = int(rng.integers(0, 2**31))
                        env = _make_env(grid_map, cfg, ep_seed)

                        if pt is not None:
                            pt.rng = np.random.default_rng(ep_seed + 1)
                            pt.reset()

                        result = run_interaction_episode(
                            env, mode,
                            oracle_teacher=oracle_teacher,
                            particle_teacher=pt,
                            oracle_cause_teacher=oc_teacher,
                            family_name=family,
                        )

                        row = {
                            "family": family,
                            "difficulty": diff,
                            "seed": base_seed,
                            "episode": ep,
                            "baseline": mode,
                            **result,
                        }
                        family_rows.append(row)
                        all_rows.append(row)
                        n_succ += result["constrained_success"]

                    sr = n_succ / EPISODES_PER_SEED * 100
                    tag = f"  {diff:6s} seed={seed_idx} {mode:15s} SR={sr:5.1f}%"
                    if mode == "particle" and family_rows:
                        last = family_rows[-1]
                        tag += f"  PMA={last.get('policy_match', 0):.2f}"
                    print(tag)

        # Save per-family CSV
        csv_path = out_path / f"{family}_results.csv"
        _save_csv(csv_path, family_rows)
        print(f"    → Saved: {csv_path}")

    # Save aggregate
    agg_path = out_path / "aggregate_results.csv"
    _save_csv(agg_path, all_rows)

    # Print summary table
    _print_summary(all_rows)

    print(f"\n  All results saved to: {out_path}")
    return all_rows


def _save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows: list[dict]):
    """Print summary table grouped by family × baseline (v1e)."""
    print(f"\n{'=' * 110}")
    print(f"  {'Family':<20s} {'Baseline':<16s} {'CSR%':>6s} {'Steps':>6s} "
          f"{'CauseAcc':>8s} {'WarnP':>6s} {'WaitS':>6s} {'ToM':>8s}")
    print(f"  {'-' * 20} {'-' * 16} {'-' * 6} {'-' * 6} "
          f"{'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8}")

    for family in V1E_FAMILIES:
        for mode in BASELINES:
            frows = [r for r in rows
                     if r["family"] == family and r["baseline"] == mode]
            if not frows:
                continue
            n = len(frows)
            csr = sum(r["constrained_success"] for r in frows) / n * 100
            steps = np.mean([r["steps"] for r in frows])
            toms = [r["tom_mse"] for r in frows if r.get("tom_mse") is not None]
            avg_tom = np.mean(toms) if toms else float("nan")
            ca_vals = [r["cause_acc"] for r in frows if r.get("cause_acc") is not None]
            avg_ca = np.mean(ca_vals) if ca_vals else float("nan")
            wp_vals = [r["warn_precision"] for r in frows if r.get("warn_precision") is not None]
            avg_wp = np.mean(wp_vals) if wp_vals else float("nan")
            ws_vals = [r["wait_safety"] for r in frows if r.get("wait_safety") is not None]
            avg_ws = np.mean(ws_vals) if ws_vals else float("nan")

            print(f"  {family:<20s} {mode:<16s} {csr:5.1f}% {steps:6.1f} "
                  f"{avg_ca:8.2f} {avg_wp:6.2f} {avg_ws:6.2f} {avg_tom:8.4f}")
        print()
    print(f"{'=' * 110}")


if __name__ == "__main__":
    run_benchmark()
