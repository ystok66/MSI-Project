"""DTMB-L WARN Helper Redundancy Audit (Exp A).

Compares three WARN target scoring variants:
  W1: GT risk + distance + door suppression (current)
  W2: Risk-only (door presence only)
  W3: Risk + commit proximity

Runs 50 seeds × medium × 5 policies × 3 variants.
Outputs Δ_warn, Δ_unlock, Δ_item per variant.

Usage:
    python scripts/eval_dtmb_warn_ablation.py [--seeds 50] [--output-file ...]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

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

WARN_VARIANTS = ["W1", "W2", "W3"]


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L WARN redundancy audit")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--output-file", default="results/dtmb_warn_ablation.txt")
    return p.parse_args()


def run_episode(runner, seed, difficulty, policy_cfg, warn_variant):
    """Run a single DTMB episode with a specific WARN variant."""
    dcfg = DTMBDispatchConfig(warn_variant=warn_variant)
    try:
        state = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            dtmb_dispatch_cfg=dcfg,
            **policy_cfg,
        )
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        metrics["success"] = True
        return metrics
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "survived": False, "reached_goal": False, "steps": 0,
        }


def main():
    args = parse_args()
    runner = LatticeV2Runner()
    diff = args.difficulty

    lines = []
    lines.append(f"DTMB-L WARN Redundancy Audit (Exp A)")
    lines.append(f"  difficulty={diff}, seeds={args.seeds}")
    lines.append(f"  warn_variants={WARN_VARIANTS}")
    lines.append(f"  policies={list(POLICIES.keys())}")
    lines.append("=" * 80)

    # Structure: results[variant][policy] = list of metrics dicts
    all_results = {}

    for wv in WARN_VARIANTS:
        lines.append(f"\n{'='*30} WARN VARIANT: {wv} {'='*30}")
        variant_results = {}

        for policy_name, policy_cfg in POLICIES.items():
            results = defaultdict(list)
            fail_count = 0

            for seed in range(args.seeds):
                m = run_episode(runner, seed, diff, policy_cfg, wv)
                if m["success"]:
                    results["survived"].append(m["survived"])
                    results["reached_goal"].append(m["reached_goal"])
                    results["steps"].append(m["steps"])
                    results["risky_entered"].append(m.get("risky_entered", 0))
                    results["warn_count"].append(m.get("warn_count", 0))
                    results["unlock_count"].append(m.get("unlock_count", 0))
                else:
                    fail_count += 1

            n = len(results["survived"])
            if n > 0:
                surv = np.mean(results["survived"])
                goal = np.mean(results["reached_goal"])
                steps_avg = np.mean(results["steps"])
                warn_avg = np.mean(results["warn_count"])
                unlock_avg = np.mean(results["unlock_count"])
                lines.append(
                    f"  {policy_name:14s}: surv={surv:.3f} goal={goal:.3f} "
                    f"steps={steps_avg:.1f} warn={warn_avg:.1f} "
                    f"unlock={unlock_avg:.1f} ({n} ok, {fail_count} fail)")
            else:
                lines.append(f"  {policy_name:14s}: ALL FAILED ({fail_count} errors)")

            variant_results[policy_name] = results

        all_results[wv] = variant_results

        # Compute deltas for this variant
        can_surv = np.mean(variant_results["canonical"]["survived"]) if variant_results["canonical"]["survived"] else 0
        nw_surv = np.mean(variant_results["no_warn"]["survived"]) if variant_results["no_warn"]["survived"] else 0
        nu_surv = np.mean(variant_results["no_unlock"]["survived"]) if variant_results["no_unlock"]["survived"] else 0
        ni_surv = np.mean(variant_results["no_item_drop"]["survived"]) if variant_results["no_item_drop"]["survived"] else 0
        nt_surv = np.mean(variant_results["no_tutor"]["survived"]) if variant_results["no_tutor"]["survived"] else 0

        delta_warn = can_surv - nw_surv
        delta_unlock = can_surv - nu_surv
        delta_item = can_surv - ni_surv
        delta_total = can_surv - nt_surv

        lines.append(f"\n  --- Deltas for {wv} ---")
        lines.append(f"  Δ_warn   = {delta_warn:+.3f}")
        lines.append(f"  Δ_unlock = {delta_unlock:+.3f}")
        lines.append(f"  Δ_item   = {delta_item:+.3f}")
        lines.append(f"  Δ_total  = {delta_total:+.3f} (canonical - no_tutor)")

    # === Cross-variant comparison ===
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-VARIANT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Variant':>8s} | {'Surv_can':>9s} | {'Δ_warn':>8s} | {'Δ_unlock':>9s} | {'Δ_item':>8s} | {'Δ_total':>8s}")
    lines.append("-" * 65)

    for wv in WARN_VARIANTS:
        vr = all_results[wv]
        s_can = np.mean(vr["canonical"]["survived"]) if vr["canonical"]["survived"] else 0
        s_nw = np.mean(vr["no_warn"]["survived"]) if vr["no_warn"]["survived"] else 0
        s_nu = np.mean(vr["no_unlock"]["survived"]) if vr["no_unlock"]["survived"] else 0
        s_ni = np.mean(vr["no_item_drop"]["survived"]) if vr["no_item_drop"]["survived"] else 0
        s_nt = np.mean(vr["no_tutor"]["survived"]) if vr["no_tutor"]["survived"] else 0

        lines.append(
            f"{wv:>8s} | {s_can:>9.3f} | {s_can-s_nw:>+8.3f} | "
            f"{s_can-s_nu:>+9.3f} | {s_can-s_ni:>+8.3f} | {s_can-s_nt:>+8.3f}")

    # Decision recommendation
    lines.append("\n--- RECOMMENDATION ---")
    # Compare W1 vs W2/W3 on Δ_warn
    deltas = {}
    for wv in WARN_VARIANTS:
        vr = all_results[wv]
        s_can = np.mean(vr["canonical"]["survived"]) if vr["canonical"]["survived"] else 0
        s_nw = np.mean(vr["no_warn"]["survived"]) if vr["no_warn"]["survived"] else 0
        deltas[wv] = s_can - s_nw

    best_wv = max(deltas, key=deltas.get)
    simplest_ok = None
    for wv in ["W2", "W3", "W1"]:  # simplest first
        if abs(deltas[wv] - deltas[best_wv]) <= 0.03:
            simplest_ok = wv
            break
    if simplest_ok:
        lines.append(f"  Simplest effective variant: {simplest_ok} (Δ_warn={deltas[simplest_ok]:+.3f})")
        lines.append(f"  Best variant: {best_wv} (Δ_warn={deltas[best_wv]:+.3f})")
        if simplest_ok != best_wv:
            lines.append(f"  → {simplest_ok} is within 0.03 of {best_wv}; recommend {simplest_ok} for simplicity")
        else:
            lines.append(f"  → {best_wv} is both simplest and best")
    else:
        lines.append(f"  Best variant: {best_wv} (Δ_warn={deltas[best_wv]:+.3f})")

    lines.append("\n" + "=" * 80)
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
