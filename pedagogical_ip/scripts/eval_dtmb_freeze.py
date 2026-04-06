"""DTMB-L v1 Benchmark Freeze — Final Archival Evaluation.

Runs all policies on all difficulties with 50 seeds and exports:
  results/dtmb_v1_easy.jsonl
  results/dtmb_v1_medium.jsonl
  results/dtmb_v1_hard.jsonl
  results/dtmb_v1_summary.md

Usage:
    python scripts/eval_dtmb_freeze.py [--seeds 50]
"""
import sys
sys.path.insert(0, ".")

import argparse
import json
import os
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.dtmb_helpers import DTMBDispatchConfig

FAMILY = "deep_tree_mixed_bottleneck_lattice"
OUTDIR = "results"


def _base(**overrides):
    cfg = dict(
        tutor_mode="none", warning_mode="none",
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=True, belief_planning_mode=True,
        latent_mode=True, patch_radius=2, prefix_horizon=5,
        allowed_interventions=None,
    )
    cfg.update(overrides)
    return cfg


POLICIES = {
    "canonical": _base(),
    "no_warn": _base(
        allowed_interventions=frozenset({"WAIT", "UNLOCK", "ITEM_DROP"})),
    "no_unlock": _base(
        allowed_interventions=frozenset({"WAIT", "WARN", "ITEM_DROP"})),
    "no_item_drop": _base(
        item_drop_enabled=False,
        allowed_interventions=frozenset({"WAIT", "WARN", "UNLOCK"})),
    "no_tutor": _base(
        robot_belief_mode=False, intervention_family_mode=False,
        item_drop_enabled=False, prefix_horizon=0),
    "oracle_O1": dict(
        tutor_mode="dtmb_oracle", warning_mode="none",
        robot_belief_mode=False, intervention_family_mode=False,
        item_drop_enabled=False, belief_planning_mode=True,
        latent_mode=True, patch_radius=2, prefix_horizon=0),
    "oracle_O2": dict(
        tutor_mode="dtmb_oracle", warning_mode="none",
        robot_belief_mode=False, intervention_family_mode=False,
        item_drop_enabled=False, belief_planning_mode=True,
        latent_mode=True, patch_radius=2, prefix_horizon=0),
}

ORACLE_DISPATCH = {
    "oracle_O1": DTMBDispatchConfig(oracle_variant="O1"),
    "oracle_O2": DTMBDispatchConfig(oracle_variant="O2"),
}


def run_episode(runner, seed, difficulty, policy_name, policy_cfg):
    dcfg = ORACLE_DISPATCH.get(policy_name)
    try:
        s = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            dtmb_dispatch_cfg=dcfg,
            **policy_cfg)
        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)

        # Extract GT bottlenecks from meta
        gt_bottlenecks = []
        if hasattr(s.meta, "dominant_bottleneck_gt_by_stage"):
            gt_bottlenecks = s.meta.dominant_bottleneck_gt_by_stage

        return {
            "seed": seed,
            "difficulty": difficulty,
            "policy": policy_name,
            "warn_variant": "W1",
            "oracle_variant": dcfg.oracle_variant if dcfg else "none",
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "steps": int(m["steps"]),
            "risky_entered": int(m.get("risky_entered", 0)),
            "warn_count": int(m.get("warn_count", 0)),
            "unlock_count": int(m.get("unlock_count", 0)),
            "route_count": int(getattr(s.meta, "route_count", 0)),
            "gt_bottlenecks": gt_bottlenecks,
            "success": True,
        }
    except Exception as e:
        return {
            "seed": seed, "difficulty": difficulty,
            "policy": policy_name, "survived": False,
            "reached_goal": False, "steps": 0, "success": False,
            "error": str(e),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    runner = LatticeV2Runner()
    os.makedirs(OUTDIR, exist_ok=True)

    summary_lines = ["# DTMB-L v1 Benchmark — Final Results\n"]
    summary_lines.append(f"Seeds: {args.seeds}\n")
    summary_lines.append("Frozen WARN variant: W1\n")
    summary_lines.append("Oracle: dtmb_oracle (O1, O2)\n")
    summary_lines.append("---\n")

    for diff in ["easy", "medium", "hard"]:
        jsonl_path = os.path.join(OUTDIR, f"dtmb_v1_{diff}.jsonl")
        results_by_policy = defaultdict(list)
        all_records = []

        print(f"=== {diff.upper()} ===")
        for policy_name, policy_cfg in POLICIES.items():
            for seed in range(args.seeds):
                rec = run_episode(runner, seed, diff, policy_name, policy_cfg)
                all_records.append(rec)
                if rec["success"]:
                    results_by_policy[policy_name].append(rec)

            n = len(results_by_policy[policy_name])
            if n > 0:
                surv = np.mean([r["survived"] for r in results_by_policy[policy_name]])
                goal = np.mean([r["reached_goal"] for r in results_by_policy[policy_name]])
                warn = np.mean([r.get("warn_count", 0) for r in results_by_policy[policy_name]])
                print(f"  {policy_name:14s}: surv={surv:.3f} goal={goal:.3f} warn={warn:.1f} (n={n})")
            else:
                print(f"  {policy_name:14s}: ALL FAILED")

        # Write JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  → Saved {len(all_records)} records to {jsonl_path}")

        # Summary table
        summary_lines.append(f"\n## {diff.upper()}\n")
        summary_lines.append("| Policy | Surv | Goal | Warn | Unlock | n |")
        summary_lines.append("|--------|------|------|------|--------|---|")
        for pn in ["canonical", "no_warn", "no_unlock", "no_item_drop",
                    "no_tutor", "oracle_O1", "oracle_O2"]:
            recs = results_by_policy[pn]
            if recs:
                surv = np.mean([r["survived"] for r in recs])
                goal = np.mean([r["reached_goal"] for r in recs])
                warn = np.mean([r.get("warn_count", 0) for r in recs])
                unlock = np.mean([r.get("unlock_count", 0) for r in recs])
                n = len(recs)
                summary_lines.append(
                    f"| {pn} | {surv:.3f} | {goal:.3f} | {warn:.1f} | {unlock:.1f} | {n} |")
            else:
                summary_lines.append(f"| {pn} | FAILED | — | — | — | 0 |")

        # Deltas
        can = results_by_policy["canonical"]
        if can:
            s_can = np.mean([r["survived"] for r in can])
            for ablation, label in [("no_warn", "Δ_warn"), ("no_unlock", "Δ_unlock"),
                                    ("no_item_drop", "Δ_item"), ("no_tutor", "Δ_total")]:
                abl = results_by_policy[ablation]
                if abl:
                    s_abl = np.mean([r["survived"] for r in abl])
                    summary_lines.append(f"\n{label} = {s_can - s_abl:+.3f}")

    summary_path = os.path.join(OUTDIR, "dtmb_v1_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n→ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
