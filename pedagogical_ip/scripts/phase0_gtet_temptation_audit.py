"""
Phase 0 — Q3: GTET Temptation Posterior Audit.

Confirms whether z (temptation latent) contributes to posterior updates
in canonical GTET. Compares FULL, G_THETA, G_Z, THETA_Z factor modes.

Records per-step Δ_z_mass, Δ_g_mass, Δ_θ_mass and task performance.

Usage:
    python scripts/phase0_gtet_temptation_audit.py [--seeds 20] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "goal_preference_temptation_entanglement_lattice"

# Standard tutor-on config
BASE_CFG = dict(
    tutor_mode="none",
    warning_mode="none",
    robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=True,
    belief_planning_mode=True,
    latent_mode=True,
    patch_radius=2,
    prefix_horizon=5,
    difficulty="medium",
    scenario_family=FAMILY,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 Q3: GTET temptation audit")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="Smoke: 3 seeds")
    p.add_argument("--output-dir", default="results/phase0")
    return p.parse_args()


def run_episode_with_posterior_trace(runner, seed, factor_mode):
    """Run one GTET episode, tracking posterior update mass per step."""
    cfg = dict(BASE_CFG, factor_mode=factor_mode)
    state = runner.reset(seed=seed, **cfg)

    posterior_traces = []
    prev_g = None
    prev_theta = None
    prev_z = None

    while not state.done:
        runner.observe(state)
        runner.apply_tutor(state)

        # Capture posterior snapshot BEFORE plan_and_move
        jgpp = state.gtet_posterior
        if jgpp is not None:
            cur_g = dict(jgpp.marginal_goal()) if hasattr(jgpp, 'marginal_goal') else None
            cur_theta = dict(jgpp.marginal_pref()) if hasattr(jgpp, 'marginal_pref') else None
            cur_z = dict(jgpp.marginal_tempt()) if hasattr(jgpp, 'marginal_tempt') else None

            def _dict_l1(d1, d2):
                if d1 is None or d2 is None:
                    return 0.0
                keys = set(d1.keys()) | set(d2.keys())
                return sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in keys)

            delta_g = _dict_l1(cur_g, prev_g) if prev_g is not None else 0.0
            delta_theta = _dict_l1(cur_theta, prev_theta) if prev_theta is not None else 0.0
            delta_z = _dict_l1(cur_z, prev_z) if prev_z is not None else 0.0

            posterior_traces.append({
                "t": state.t,
                "delta_g": delta_g,
                "delta_theta": delta_theta,
                "delta_z": delta_z,
            })

            prev_g = cur_g
            prev_theta = cur_theta
            prev_z = cur_z

        runner.plan_and_move(state)

    metrics = runner.get_metrics(state)
    return metrics, posterior_traces


def main():
    args = parse_args()
    seeds = 3 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z"]

    lines = []
    lines.append("Phase 0 — Q3: GTET Temptation Posterior Audit")
    lines.append(f"  seeds={seeds}, family={FAMILY}, difficulty=medium")
    lines.append("=" * 80)

    all_metrics = defaultdict(list)
    all_traces = defaultdict(list)

    for fm in factor_modes:
        lines.append(f"\n--- Factor Mode: {fm} ---")
        for seed in range(seeds):
            try:
                m, traces = run_episode_with_posterior_trace(runner, seed, fm)
                all_metrics[fm].append(m)
                all_traces[fm].extend(traces)
            except Exception as e:
                lines.append(f"  seed={seed} FAILED: {e}")

        n_ok = len(all_metrics[fm])
        surv = np.mean([m.get("survived", False) for m in all_metrics[fm]]) if n_ok else 0
        goal = np.mean([m.get("reached_goal", False) for m in all_metrics[fm]]) if n_ok else 0
        lines.append(f"  episodes={n_ok}, surv={surv:.3f}, goal={goal:.3f}")

        if all_traces[fm]:
            avg_dg = np.mean([t["delta_g"] for t in all_traces[fm]])
            avg_dt = np.mean([t["delta_theta"] for t in all_traces[fm]])
            avg_dz = np.mean([t["delta_z"] for t in all_traces[fm]])
            lines.append(f"  mean Δg={avg_dg:.4f}, Δθ={avg_dt:.4f}, Δz={avg_dz:.4f}")

    # === Cross-mode comparison ===
    lines.append(f"\n{'='*80}")
    lines.append("POSTERIOR UPDATE MASS COMPARISON")
    lines.append("=" * 80)

    header = f"{'Mode':>10s} | {'Surv':>6s} | {'Goal':>6s} | {'Δg':>8s} | {'Δθ':>8s} | {'Δz':>8s} | {'Δz/Δg':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for fm in factor_modes:
        n = len(all_metrics[fm])
        if n == 0:
            lines.append(f"{fm:>10s} | ALL FAILED")
            continue
        surv = np.mean([m.get("survived", False) for m in all_metrics[fm]])
        goal = np.mean([m.get("reached_goal", False) for m in all_metrics[fm]])

        avg_dg = np.mean([t["delta_g"] for t in all_traces[fm]]) if all_traces[fm] else 0
        avg_dt = np.mean([t["delta_theta"] for t in all_traces[fm]]) if all_traces[fm] else 0
        avg_dz = np.mean([t["delta_z"] for t in all_traces[fm]]) if all_traces[fm] else 0
        ratio = avg_dz / max(avg_dg, 1e-6) if avg_dg > 0 else 0

        lines.append(f"{fm:>10s} | {surv:>6.3f} | {goal:>6.3f} | {avg_dg:>8.4f} | "
                     f"{avg_dt:>8.4f} | {avg_dz:>8.4f} | {ratio:>8.4f}")

    # === Verdict ===
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Check 3 conditions for demotion:
    # 1. G_THETA not significantly worse than FULL
    # 2. Δz << Δg + Δθ
    # 3. No family-level evidence z changes decisions

    surv_full = np.mean([m.get("survived", False) for m in all_metrics["FULL"]]) if all_metrics["FULL"] else 0
    surv_gt = np.mean([m.get("survived", False) for m in all_metrics["G_THETA"]]) if all_metrics["G_THETA"] else 0

    avg_dz_full = np.mean([t["delta_z"] for t in all_traces["FULL"]]) if all_traces["FULL"] else 0
    avg_dg_full = np.mean([t["delta_g"] for t in all_traces["FULL"]]) if all_traces["FULL"] else 0
    avg_dt_full = np.mean([t["delta_theta"] for t in all_traces["FULL"]]) if all_traces["FULL"] else 0

    gt_not_worse = surv_gt >= surv_full - 0.05
    z_negligible = avg_dz_full < 0.3 * (avg_dg_full + avg_dt_full + 1e-6)

    if gt_not_worse and z_negligible:
        lines.append("VERDICT: z 从 canonical posterior 中降级为 optional plugin。")
        lines.append(f"  G_THETA surv={surv_gt:.3f} ≈ FULL surv={surv_full:.3f} (差<0.05)")
        lines.append(f"  Δz={avg_dz_full:.4f} << Δg+Δθ={avg_dg_full+avg_dt_full:.4f}")
        lines.append("  → 默认 canonical: q(g,θ) only。z 仅在专门 family 中打开。")
    elif not gt_not_worse:
        lines.append("VERDICT: G_THETA 性能不足，z 仍有潜在贡献。")
        lines.append(f"  G_THETA surv={surv_gt:.3f} vs FULL surv={surv_full:.3f}")
        lines.append("  → 暂不降级 z。需进一步调查 z 的具体贡献机制。")
    else:
        lines.append("VERDICT: z 更新量不可忽略，但 G_THETA 表现可接受。边界情况。")
        lines.append(f"  Δz={avg_dz_full:.4f}, Δg+Δθ={avg_dg_full+avg_dt_full:.4f}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "q3_gtet_temptation_audit.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
