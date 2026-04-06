"""DTMB-L Hard Calibration Sweep (Exp B).

Sweeps `belt_risk`, `terminal_belt_fraction`, and `deadline_ratio` to find
a HARD_v2 regime where:
  0.10 ≤ Surv_canonical ≤ 0.35
  0.00 ≤ Surv_no_tutor ≤ 0.10
  Surv_canonical - Surv_no_item > 0

Uses composite objective J_hard for automatic selection.

Usage:
    python scripts/sweep_dtmb_hard_calibration.py [--seeds 20] [--output-file ...]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict
import multiprocessing as mp

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.dtmb_helpers import DTMBDispatchConfig

FAMILY = "deep_tree_mixed_bottleneck_lattice"

POLICIES = {
    "canonical": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": None,
    },
    "no_tutor": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": False,
        "intervention_family_mode": False,
        "item_drop_enabled": False,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 0,
        "allowed_interventions": None,
    },
    "no_item_drop": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": False,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "WARN", "UNLOCK"}),
    },
    "no_unlock": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "WARN", "ITEM_DROP"}),
    },
}


def run_episode_worker(args):
    """Worker function for multiprocessing."""
    seed, belt_risk, belt_fraction, deadline_ratio, policy_name, policy_cfg = args
    runner = LatticeV2Runner()
    user_cfg = {
        "belt_risk": belt_risk,
        "terminal_belt_fraction": belt_fraction,
        "deadline_ratio": deadline_ratio,
    }
    try:
        state = runner.reset(
            seed=seed, difficulty="hard",
            scenario_family=FAMILY, user_cfg=user_cfg, **policy_cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        metrics["success"] = True
    except Exception as e:
        metrics = {"success": False, "error": str(e),
                   "survived": False, "reached_goal": False, "steps": 0}

    return (belt_risk, belt_fraction, deadline_ratio, policy_name, seed, metrics)


def compute_j_hard(surv_can, surv_nt, surv_ni, target_surv=0.20):
    """Composite objective for HARD calibration.

    J = α₁·Surv_can - α₂·Surv_nt + α₃·(Surv_can - Surv_ni) - α₄·|Surv_can - τ|
    """
    a1, a2, a3, a4 = 1.0, 0.5, 0.5, 2.0
    return (a1 * surv_can
            - a2 * surv_nt
            + a3 * (surv_can - surv_ni)
            - a4 * abs(surv_can - target_surv))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--output-file", default="results/dtmb_hard_sweep.txt")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    workers = args.workers or min(8, mp.cpu_count())

    belt_risks = [0.40, 0.45, 0.50]
    belt_fractions = [0.30, 0.40, 0.50]
    deadline_ratios = [1.08, 1.12, 1.16]

    tasks = []
    for br in belt_risks:
        for bf in belt_fractions:
            for dr in deadline_ratios:
                for policy_name, policy_cfg in POLICIES.items():
                    for seed in range(args.seeds):
                        tasks.append((seed, br, bf, dr, policy_name, policy_cfg))

    print(f"DTMB-L Hard Calibration Sweep")
    print(f"  belt_risk ∈ {belt_risks}")
    print(f"  terminal_belt_fraction ∈ {belt_fractions}")
    print(f"  deadline_ratio ∈ {deadline_ratios}")
    print(f"  {args.seeds} seeds, {len(tasks)} total episodes")
    print(f"  Using {workers} workers")

    results_map = defaultdict(lambda: defaultdict(list))

    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_episode_worker, tasks)):
            br, bf, dr, policy_name, seed, m = res
            cfg_key = (br, bf, dr)
            if m["success"]:
                results_map[cfg_key][policy_name].append(m)
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1} / {len(tasks)}")

    lines = []
    lines.append("DTMB-L Hard Calibration Sweep Results")
    lines.append(f"  seeds={args.seeds}")
    lines.append("=" * 100)

    j_scores = []

    for br in sorted(belt_risks):
        for bf in sorted(belt_fractions):
            for dr in sorted(deadline_ratios):
                cfg_key = (br, bf, dr)
                lines.append(f"\n--- belt_risk={br:.2f}, belt_fraction={bf:.2f}, deadline_ratio={dr:.2f} ---")

                survs = {}
                for policy_name in ["canonical", "no_tutor", "no_item_drop", "no_unlock"]:
                    m_list = results_map[cfg_key][policy_name]
                    if not m_list:
                        lines.append(f"  {policy_name:14s}: NO VALID RUNS")
                        survs[policy_name] = 0.0
                        continue
                    surv = np.mean([m["survived"] for m in m_list])
                    goal = np.mean([m["reached_goal"] for m in m_list])
                    steps = np.mean([m["steps"] for m in m_list])
                    warn_avg = np.mean([m.get("warn_count", 0) for m in m_list])
                    unlock_avg = np.mean([m.get("unlock_count", 0) for m in m_list])
                    n = len(m_list)
                    lines.append(
                        f"  {policy_name:14s}: surv={surv:.3f} goal={goal:.3f} "
                        f"steps={steps:.1f} warn={warn_avg:.1f} unlock={unlock_avg:.1f} "
                        f"(n={n})")
                    survs[policy_name] = surv

                j = compute_j_hard(
                    survs.get("canonical", 0),
                    survs.get("no_tutor", 0),
                    survs.get("no_item_drop", 0),
                )
                lines.append(f"  J_hard = {j:.3f}")
                j_scores.append((j, br, bf, dr, survs))

    # Top-3 configs
    j_scores.sort(key=lambda x: x[0], reverse=True)
    lines.append(f"\n{'='*100}")
    lines.append("TOP-3 CONFIGS BY J_hard")
    lines.append("=" * 100)
    for rank, (j, br, bf, dr, survs) in enumerate(j_scores[:3], 1):
        lines.append(
            f"  #{rank}: J={j:.3f} | belt_risk={br:.2f} belt_frac={bf:.2f} "
            f"deadline={dr:.2f} | Surv_can={survs.get('canonical',0):.3f} "
            f"Surv_nt={survs.get('no_tutor',0):.3f} "
            f"Surv_ni={survs.get('no_item_drop',0):.3f}")

    lines.append("\n" + "=" * 100)
    lines.append("Done.")

    output = "\n".join(lines)
    print(output)

    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
