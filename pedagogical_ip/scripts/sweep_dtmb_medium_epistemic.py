"""DTMB-L Medium Local Robustness Sweep (Exp C).

Sweeps a local neighborhood around the current working medium config to verify
that Δ_warn > 0, Δ_unlock > 0, Δ_item > 0 are stable, not a one-point fluke.

Usage:
    python scripts/sweep_dtmb_medium_epistemic.py [--seeds 20] [--output-file ...]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict
import multiprocessing as mp

from src.envs.lattice_v2_runner import LatticeV2Runner

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
    "no_warn": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "UNLOCK", "ITEM_DROP"}),
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
    },
}


def run_episode_worker(args):
    """Worker function for multiprocessing."""
    seed, cue_rel, mis_frac, lure_str, policy_name, policy_cfg = args
    runner = LatticeV2Runner()
    user_cfg = {
        "stage1_cue_reliability": cue_rel,
        "misleading_fraction": mis_frac,
        "lure_strength": lure_str,
    }
    try:
        state = runner.reset(
            seed=seed, difficulty="medium",
            scenario_family=FAMILY, user_cfg=user_cfg, **policy_cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        metrics["success"] = True
    except Exception as e:
        metrics = {"success": False, "error": str(e),
                   "survived": False, "reached_goal": False, "steps": 0}

    return (cue_rel, mis_frac, lure_str, policy_name, seed, metrics)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--output-file", default="results/dtmb_medium_sweep.txt")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    workers = args.workers or min(8, mp.cpu_count())

    # Local neighborhood around current working config:
    # Current: cue_rel=0.65, mis_frac=0.15, lure_str=0.55
    cue_rels = [0.55, 0.65, 0.75]
    mis_fracs = [0.10, 0.15, 0.25]
    lure_strs = [0.40, 0.55, 0.70]

    tasks = []
    for cr in cue_rels:
        for mf in mis_fracs:
            for ls in lure_strs:
                for policy_name, policy_cfg in POLICIES.items():
                    for seed in range(args.seeds):
                        tasks.append((seed, cr, mf, ls, policy_name, policy_cfg))

    print(f"DTMB-L Medium Local Robustness Sweep (Exp C)")
    print(f"  stage1_cue_reliability ∈ {cue_rels}")
    print(f"  misleading_fraction ∈ {mis_fracs}")
    print(f"  lure_strength ∈ {lure_strs}")
    print(f"  {args.seeds} seeds, {len(tasks)} total episodes")
    print(f"  Using {workers} workers")

    results_map = defaultdict(lambda: defaultdict(list))

    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_episode_worker, tasks)):
            cr, mf, ls, policy_name, seed, m = res
            cfg_key = (cr, mf, ls)
            if m["success"]:
                results_map[cfg_key][policy_name].append(m)
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i + 1} / {len(tasks)}")

    lines = []
    lines.append("DTMB-L Medium Local Robustness Sweep Results (Exp C)")
    lines.append(f"  seeds={args.seeds}")
    lines.append("=" * 100)

    n_total = 0
    n_stable = 0

    for cr in sorted(cue_rels):
        for mf in sorted(mis_fracs):
            for ls in sorted(lure_strs):
                cfg_key = (cr, mf, ls)
                m_can = results_map[cfg_key]["canonical"]
                m_nw = results_map[cfg_key]["no_warn"]
                m_nu = results_map[cfg_key]["no_unlock"]
                m_ni = results_map[cfg_key]["no_item_drop"]
                m_nt = results_map[cfg_key]["no_tutor"]

                if not m_can:
                    continue

                n_total += 1
                s_can = np.mean([m["survived"] for m in m_can])
                s_nw = np.mean([m["survived"] for m in m_nw]) if m_nw else 0
                s_nu = np.mean([m["survived"] for m in m_nu]) if m_nu else 0
                s_ni = np.mean([m["survived"] for m in m_ni]) if m_ni else 0
                s_nt = np.mean([m["survived"] for m in m_nt]) if m_nt else 0

                delta_warn = s_can - s_nw
                delta_unlock = s_can - s_nu
                delta_item = s_can - s_ni
                delta_total = s_can - s_nt

                all_positive = delta_warn > 0 and delta_unlock > 0 and delta_item > 0
                if all_positive:
                    n_stable += 1

                lines.append(f"\n--- cue={cr:.2f}, mis={mf:.2f}, lure={ls:.2f} ---")
                lines.append(f"  canonical    : surv={s_can:.3f}")
                lines.append(f"  no_warn      : surv={s_nw:.3f} (Δ={delta_warn:+.3f})")
                lines.append(f"  no_unlock    : surv={s_nu:.3f} (Δ={delta_unlock:+.3f})")
                lines.append(f"  no_item_drop : surv={s_ni:.3f} (Δ={delta_item:+.3f})")
                lines.append(f"  no_tutor     : surv={s_nt:.3f} (Δ_total={delta_total:+.3f})")
                if all_positive:
                    lines.append("  >>> ALL Δ > 0 — STABLE <<<")

    lines.append(f"\n{'='*100}")
    lines.append("STABILITY SUMMARY")
    lines.append(f"  Configs with all Δ > 0: {n_stable} / {n_total}")
    if n_total > 0:
        pct = 100.0 * n_stable / n_total
        lines.append(f"  Stability rate: {pct:.1f}%")
        if pct >= 60.0:
            lines.append("  >>> PASSES 60% threshold — Medium is STABLE <<<")
        else:
            lines.append("  >>> BELOW 60% threshold — Medium needs further tuning <<<")

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
