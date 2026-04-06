"""
Phase 1B — Boredom Shadow Evaluation.

E1: Offline trace replay — compare Q_WAIT_old vs Q_WAIT_new on existing traces.
E2: Shadow closed-loop — run with boredom_weight > 0 and measure FP_wait reduction.
E3: Family regression.

Usage:
    python scripts/phase1b_boredom_shadow_eval.py [--seeds 20] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.intervention_policy import InterventionConfig

FAMILIES = [
    "deep_tree_mixed_bottleneck_lattice",
    "goal_preference_temptation_entanglement_lattice",
]
FAMILY_SHORT = {
    "deep_tree_mixed_bottleneck_lattice": "DTMB",
    "goal_preference_temptation_entanglement_lattice": "GTET",
}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1B: boredom shadow eval")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="3 seeds")
    p.add_argument("--output-dir", default="results/phase1b")
    return p.parse_args()


def run_episode(runner, seed, family, boredom_weight=0.0):
    """Run one episode with given boredom_weight, record diagnostics."""
    cfg = dict(
        tutor_mode="none",
        warning_mode="none",
        latent_mode=True,
        patch_radius=2,
        prefix_horizon=5,
        belief_planning_mode=True,
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
        difficulty="medium",
        scenario_family=family,
    )
    try:
        state = runner.reset(seed=seed, **cfg)
        state.audit_mode = True  # Phase 0 trace
        state.boredom_weight = boredom_weight  # Phase 1B

        while not state.done:
            state = runner.step(state)

        metrics = runner.get_metrics(state)

        # Extract boredom diagnostics from intervention decisions
        boredom_penalties = []
        wait_decisions = 0
        fp_wait = 0
        total_interventions = 0
        intervention_actions = defaultdict(int)

        for trace in state.audit_trace:
            act = trace.get("action", "NONE")
            intervention_actions[act] += 1
            if act == "WAIT":
                wait_decisions += 1

        # Count FP_wait from audit_trace (same logic as Q4)
        for i in range(1, len(state.audit_trace)):
            prev = state.audit_trace[i - 1]
            curr = state.audit_trace[i]
            IG = prev.get("U_t", 0) - curr.get("U_t", 0)
            if not np.isfinite(IG):
                IG = 0.0
            if prev.get("action") == "WAIT" and IG <= 0:
                fp_wait += 1

        fp_wait_rate = fp_wait / max(wait_decisions, 1) if wait_decisions > 0 else 0

        # Compute bore_ratio from trace
        stall_cost = 0.0
        total_cost = 0.0
        for i in range(1, len(state.audit_trace)):
            prev = state.audit_trace[i - 1]
            curr = state.audit_trace[i]
            IG = prev.get("U_t", 0) - curr.get("U_t", 0)
            if not np.isfinite(IG):
                IG = 0.0
            FC = 1.0 + max(0.0, curr.get("cost", 1.0) - 1.0)
            if not np.isfinite(FC):
                FC = 1.0
            total_cost += FC
            if IG <= 0:
                stall_cost += FC

        bore_ratio = stall_cost / max(total_cost, 1e-6) if total_cost > 0 else 0

        return {
            "survived": metrics["survived"],
            "reached_goal": metrics["reached_goal"],
            "steps": metrics["steps"],
            "warn_count": metrics["warn_count"],
            "fp_wait_rate": fp_wait_rate,
            "bore_ratio": bore_ratio,
            "wait_decisions": wait_decisions,
            "intervention_mix": dict(intervention_actions),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "survived": False,
                "reached_goal": False, "fp_wait_rate": 0, "bore_ratio": 0}


def main():
    args = parse_args()
    seeds = 3 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    # Test β_bore values: 0.0 (canonical), 0.3, 0.5, 1.0
    bore_weights = [0.0, 0.3, 0.5, 1.0]

    lines = []
    lines.append("Phase 1B - Boredom Shadow Evaluation")
    lines.append(f"  seeds={seeds}")
    lines.append("=" * 80)

    all_results = {}

    for family in FAMILIES:
        fshort = FAMILY_SHORT[family]

        for bw in bore_weights:
            key = (fshort, bw)
            lines.append(f"\n--- {fshort} x bore_weight={bw:.1f} ---")

            results = []
            for seed in range(seeds):
                m = run_episode(runner, seed, family, boredom_weight=bw)
                results.append(m)
                if not m["success"]:
                    lines.append(f"  seed={seed} FAILED: {m.get('error', '?')}")

            all_results[key] = results
            ok = [r for r in results if r["success"]]
            if ok:
                surv = np.mean([r["survived"] for r in ok])
                goal = np.mean([r["reached_goal"] for r in ok])
                fp = np.mean([r["fp_wait_rate"] for r in ok])
                br = np.mean([r["bore_ratio"] for r in ok])
                steps = np.mean([r["steps"] for r in ok])
                lines.append(f"  n={len(ok)}, surv={surv:.3f}, goal={goal:.3f}, "
                             f"steps={steps:.1f}")
                lines.append(f"  FP_wait={fp:.3f}, BoreRatio={br:.3f}")

    # Summary table
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY TABLE")
    lines.append("=" * 80)
    hdr = f"{'Family':>6s} {'BoreWt':>7s} | {'Surv':>6s} | {'Goal':>6s} | {'Steps':>6s} | {'FP_wait':>8s} | {'BoreRatio':>10s} | {'N':>4s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for family in FAMILIES:
        fshort = FAMILY_SHORT[family]
        for bw in bore_weights:
            key = (fshort, bw)
            ok = [r for r in all_results.get(key, []) if r.get("success")]
            if ok:
                surv = np.mean([r["survived"] for r in ok])
                goal = np.mean([r["reached_goal"] for r in ok])
                steps = np.mean([r["steps"] for r in ok])
                fp = np.mean([r["fp_wait_rate"] for r in ok])
                br = np.mean([r["bore_ratio"] for r in ok])
                n = len(ok)
                lines.append(f"{fshort:>6s} {bw:>7.1f} | {surv:>6.3f} | {goal:>6.3f} | "
                             f"{steps:>6.1f} | {fp:>8.3f} | {br:>10.3f} | {n:>4d}")

    # Verdict
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Check: Does any bore_weight reduce GTET FP_wait from 0.80 to ≤ 0.45?
    gtet_baseline_fp = 0.0
    best_gtet_bore_fp = 1.0
    best_bw = 0.0
    best_gtet_surv = 0.0
    baseline_gtet_surv = 0.0

    for bw in bore_weights:
        ok = [r for r in all_results.get(("GTET", bw), []) if r.get("success")]
        if ok:
            fp = np.mean([r["fp_wait_rate"] for r in ok])
            surv = np.mean([r["survived"] for r in ok])
            if bw == 0.0:
                gtet_baseline_fp = fp
                baseline_gtet_surv = surv
            if fp < best_gtet_bore_fp:
                best_gtet_bore_fp = fp
                best_bw = bw
                best_gtet_surv = surv

    fp_reduced = gtet_baseline_fp - best_gtet_bore_fp
    surv_drop = baseline_gtet_surv - best_gtet_surv

    if best_gtet_bore_fp <= 0.45 and surv_drop <= 0.05:
        lines.append(f"VERDICT A: Boredom penalty effective.")
        lines.append(f"  Best beta_bore={best_bw}, FP_wait: {gtet_baseline_fp:.3f} -> {best_gtet_bore_fp:.3f}")
        lines.append(f"  Surv drop: {surv_drop:.3f} (within tolerance)")
        lines.append(f"  -> Promote to canonical with beta_bore={best_bw}")
    elif fp_reduced > 0.10:
        lines.append(f"VERDICT B: Partial improvement.")
        lines.append(f"  Best beta_bore={best_bw}, FP_wait: {gtet_baseline_fp:.3f} -> {best_gtet_bore_fp:.3f}")
        lines.append(f"  Surv drop: {surv_drop:.3f}")
        lines.append(f"  -> Needs calibration. Consider beta_bore sweep or formula adjustment.")
    else:
        lines.append(f"VERDICT C: No significant improvement.")
        lines.append(f"  Baseline FP_wait={gtet_baseline_fp:.3f}, best={best_gtet_bore_fp:.3f}")
        lines.append(f"  -> Boredom formula needs rethinking.")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "e1_boredom_shadow.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
