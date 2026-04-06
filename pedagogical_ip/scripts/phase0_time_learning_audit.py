"""
Phase 0 — Q4: Time-Learning Closure Audit.

Measures whether current Q_WAIT already covers boredom/frustration,
or if there's a significant gap where WAIT is chosen but no learning happens.

Computes per-step U_t, IG_t, FC_t and derives StallCost, BoreRatio, FP_wait.

Usage:
    python scripts/phase0_time_learning_audit.py [--seeds 20] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILIES = [
    "deep_tree_mixed_bottleneck_lattice",
    "goal_preference_temptation_entanglement_lattice",
]

FAMILY_SHORT = {
    "deep_tree_mixed_bottleneck_lattice": "DTMB",
    "goal_preference_temptation_entanglement_lattice": "GTET",
}

# Config for each tutor mode
def get_config(family, tutor_mode):
    """Build runner kwargs for a tutor mode."""
    base = dict(
        tutor_mode="none",
        warning_mode="none",
        latent_mode=True,
        patch_radius=2,
        prefix_horizon=5,
        belief_planning_mode=True,
        difficulty="medium",
        scenario_family=family,
    )

    if tutor_mode == "selective":
        base.update(
            robot_belief_mode=True,
            intervention_family_mode=True,
            item_drop_enabled=True,
        )
    elif tutor_mode == "always_warn":
        base.update(
            robot_belief_mode=True,
            intervention_family_mode=True,
            item_drop_enabled=True,
        )
    elif tutor_mode == "no_tutor":
        base.update(
            robot_belief_mode=False,
            intervention_family_mode=False,
            item_drop_enabled=False,
        )

    return base


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 Q4: Time-learning closure audit")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="Smoke: 3 seeds")
    p.add_argument("--output-dir", default="results/phase0")
    return p.parse_args()


def compute_time_learning_metrics(audit_trace):
    """Compute StallCost, BoreRatio, FP_wait from per-step audit trace."""
    if len(audit_trace) < 2:
        return {"stall_cost": 0, "total_cost": 0, "bore_ratio": 0, "fp_wait_rate": 0,
                "n_steps": len(audit_trace)}

    stall_cost = 0.0
    total_cost = 0.0
    fp_wait_count = 0
    wait_count = 0

    for i in range(1, len(audit_trace)):
        prev = audit_trace[i - 1]
        curr = audit_trace[i]

        IG = prev["U_t"] - curr["U_t"]
        # Guard against non-finite uncertainty values
        if not np.isfinite(IG):
            IG = 0.0
        FC = 1.0 + max(0.0, curr.get("cost", 1.0) - 1.0)
        if not np.isfinite(FC):
            FC = 1.0
        is_stall = (IG <= 0)

        total_cost += FC
        if is_stall:
            stall_cost += FC

        # FP_wait: WAIT at previous step, but no info gain at current step
        if prev["action"] == "WAIT" and is_stall and FC > 0:
            fp_wait_count += 1
        if prev["action"] == "WAIT":
            wait_count += 1

    bore_ratio = stall_cost / max(total_cost, 1e-6) if total_cost > 0 else 0
    fp_wait_rate = fp_wait_count / max(wait_count, 1) if wait_count > 0 else 0

    return {
        "stall_cost": stall_cost,
        "total_cost": total_cost,
        "bore_ratio": bore_ratio,
        "fp_wait_rate": fp_wait_rate,
        "n_steps": len(audit_trace),
        "n_stall_steps": sum(1 for i in range(1, len(audit_trace))
                             if audit_trace[i-1]["U_t"] - audit_trace[i]["U_t"] <= 0),
        "n_wait": wait_count,
        "n_fp_wait": fp_wait_count,
    }


def run_episode_with_trace(runner, seed, family, tutor_mode):
    """Run one episode with audit_mode=True, return metrics + time-learning metrics."""
    cfg = get_config(family, tutor_mode)
    try:
        state = runner.reset(seed=seed, **cfg)
        state.audit_mode = True  # Enable per-step trace AFTER reset
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        tl_metrics = compute_time_learning_metrics(state.audit_trace)
        return metrics, tl_metrics
    except Exception as e:
        return {"survived": False, "error": str(e)}, {"stall_cost": 0, "total_cost": 0,
                "bore_ratio": 0, "fp_wait_rate": 0, "n_steps": 0}


def main():
    args = parse_args()
    seeds = 3 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    tutor_modes = ["selective", "always_warn", "no_tutor"]

    lines = []
    lines.append("Phase 0 — Q4: Time-Learning Closure Audit")
    lines.append(f"  seeds={seeds}, families={[FAMILY_SHORT[f] for f in FAMILIES]}")
    lines.append("=" * 80)

    all_results = {}  # (family_short, tutor_mode) → list of tl_metrics

    for family in FAMILIES:
        fshort = FAMILY_SHORT[family]
        for tutor_mode in tutor_modes:
            key = (fshort, tutor_mode)
            lines.append(f"\n--- {fshort} × {tutor_mode} ---")

            episode_metrics = []
            tl_list = []

            for seed in range(seeds):
                try:
                    m, tl = run_episode_with_trace(runner, seed, family, tutor_mode)
                    episode_metrics.append(m)
                    tl_list.append(tl)
                except Exception as e:
                    lines.append(f"  seed={seed} FAILED: {e}")

            all_results[key] = tl_list

            n_ok = len(tl_list)
            if n_ok > 0:
                surv = np.mean([m.get("survived", False) for m in episode_metrics])
                avg_bore = np.mean([tl["bore_ratio"] for tl in tl_list])
                avg_fp = np.mean([tl["fp_wait_rate"] for tl in tl_list])
                avg_stall = np.mean([tl["stall_cost"] for tl in tl_list])
                lines.append(f"  n={n_ok}, surv={surv:.3f}, "
                             f"BoreRatio={avg_bore:.3f}, FP_wait={avg_fp:.3f}, "
                             f"StallCost={avg_stall:.1f}")

    # === Summary table ===
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY TABLE")
    lines.append("=" * 80)

    header = f"{'Family':>6s} {'TutorMode':>12s} | {'BoreRatio':>10s} | {'FP_wait':>8s} | {'StallCost':>10s} | {'N':>4s}"
    lines.append(header)
    lines.append("-" * len(header))

    for family in FAMILIES:
        fshort = FAMILY_SHORT[family]
        for tm in tutor_modes:
            key = (fshort, tm)
            tl_list = all_results.get(key, [])
            if tl_list:
                br = np.mean([t["bore_ratio"] for t in tl_list])
                fp = np.mean([t["fp_wait_rate"] for t in tl_list])
                sc = np.mean([t["stall_cost"] for t in tl_list])
                n = len(tl_list)
                lines.append(f"{fshort:>6s} {tm:>12s} | {br:>10.3f} | {fp:>8.3f} | {sc:>10.1f} | {n:>4d}")
            else:
                lines.append(f"{fshort:>6s} {tm:>12s} | NO DATA")

    # === Verdict ===
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Aggregate across families for selective tutor
    selective_fp = []
    selective_bore = []
    for family in FAMILIES:
        fshort = FAMILY_SHORT[family]
        key = (fshort, "selective")
        tl_list = all_results.get(key, [])
        if tl_list:
            fp_vals = [t["fp_wait_rate"] for t in tl_list if np.isfinite(t["fp_wait_rate"])]
            bore_vals = [t["bore_ratio"] for t in tl_list if np.isfinite(t["bore_ratio"])]
            selective_fp.extend(fp_vals)
            selective_bore.extend(bore_vals)

    if selective_fp:
        avg_fp_sel = np.mean(selective_fp)
        avg_bore_sel = np.mean(selective_bore) if selective_bore else 0.0

        # Gap is real if FP_wait rate is non-trivial (> 10%)
        # AND BoreRatio is non-trivial (> 15%)
        gap_real = avg_fp_sel > 0.10 and avg_bore_sel > 0.15

        if gap_real:
            lines.append("VERDICT A: Gap 真实存在。")
            lines.append(f"  selective FP_wait={avg_fp_sel:.3f}, BoreRatio={avg_bore_sel:.3f}")
            lines.append("  → 当前 canonical utility 还未闭环。")
            lines.append("  → Phase 1 需要加 boredom/frustration 项到 Q_WAIT。")
        else:
            lines.append("VERDICT B: Gap 不显著。")
            lines.append(f"  selective FP_wait={avg_fp_sel:.3f}, BoreRatio={avg_bore_sel:.3f}")
            lines.append("  → LearningGain + catastrophe + deadline 已足够覆盖主要现象。")
            lines.append("  → 此任务可在现阶段关闭，不必额外加 boredom 项。")
    else:
        lines.append("VERDICT: 数据不足。")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "q4_time_learning_audit.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
