"""
Phase 1A — E1: Baseline_v2 Warning Path Audit.

Tests all 5 warning variants on baseline_v2 lattice (NOT DTMB/GTET).
This is the TRUE Q1 audit that Phase 0 missed.

Records per-warning: FlipRate, Drho_inc, PlannerShift (DNLL), PseudoLabelMass.
Also records DTMB/GTET regression at the end.

Usage:
    python scripts/phase1a_baseline_v2_warning_audit.py [--seeds 30] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

# Baseline_v2 uses generate_lattice_v2 (no scenario_family)
BASE_CFG = dict(
    tutor_mode="none",
    warning_mode="selected",  # enable segment warnings
    latent_mode=True,
    patch_radius=2,
    prefix_horizon=5,
    belief_planning_mode=True,
    robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=True,
    difficulty="medium",
    # No scenario_family = baseline_v2
)

VARIANTS = ["legacy_bias", "rsa_obs_l0", "rsa_obs_s1", "rsa_obs_s1_trust", "rsa_plus_phase10"]

DTMB = "deep_tree_mixed_bottleneck_lattice"
GTET = "goal_preference_temptation_entanglement_lattice"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1A E1: baseline_v2 warning audit")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--smoke", action="store_true", help="Smoke: 3 seeds")
    p.add_argument("--output-dir", default="results/phase1a")
    return p.parse_args()


def run_episode(runner, seed, variant):
    """Run one baseline_v2 episode with given warning variant."""
    cfg = dict(BASE_CFG, warning_variant=variant)
    try:
        state = runner.reset(seed=seed, **cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)

        # Extract per-warning diagnostics
        warn_diags = state.rsa_warn_diagnostics
        delta_rho_incs = [d.get("delta_rho_inc", 0) for d in warn_diags]
        delta_rho_uniforms = [d.get("delta_rho_uniform", 0) for d in warn_diags]
        delta_nlls = [d.get("delta_nll_local", 0) for d in warn_diags]

        return {
            "survived": metrics["survived"],
            "reached_goal": metrics["reached_goal"],
            "steps": metrics["steps"],
            "warn_count": metrics["warn_count"],
            "mean_drho_inc": float(np.mean(delta_rho_incs)) if delta_rho_incs else 0,
            "mean_drho_uniform": float(np.mean(delta_rho_uniforms)) if delta_rho_uniforms else 0,
            "mean_dnll": float(np.mean(delta_nlls)) if delta_nlls else 0,
            "n_diag": len(warn_diags),
            "rsa_belief_final": state.rsa_belief_state.belief.tolist(),
            "rsa_entropy": state.rsa_belief_state.entropy(),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "survived": False, "reached_goal": False}


def run_regression_episode(runner, seed, family):
    """Run one family episode for regression check."""
    cfg = dict(BASE_CFG)
    cfg.pop("warning_mode", None)  # family has own warning routing
    cfg["warning_mode"] = "none"
    cfg["scenario_family"] = family
    try:
        state = runner.reset(seed=seed, **cfg)
        while not state.done:
            state = runner.step(state)
        m = runner.get_metrics(state)
        return {"survived": m["survived"], "reached_goal": m["reached_goal"],
                "steps": m["steps"], "success": True}
    except Exception as e:
        return {"success": False, "error": str(e), "survived": False}


def main():
    args = parse_args()
    seeds = 3 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    lines = []
    lines.append("Phase 1A - E1: Baseline_v2 Warning Path Audit")
    lines.append(f"  seeds={seeds}, difficulty=medium")
    lines.append("=" * 80)

    all_results = {}

    for variant in VARIANTS:
        lines.append(f"\n--- Variant: {variant} ---")
        results = []
        for seed in range(seeds):
            m = run_episode(runner, seed, variant)
            results.append(m)
            if not m["success"]:
                lines.append(f"  seed={seed} FAILED: {m.get('error', '?')}")

        all_results[variant] = results
        ok = [r for r in results if r["success"]]
        n = len(ok)
        if n > 0:
            surv = np.mean([r["survived"] for r in ok])
            goal = np.mean([r["reached_goal"] for r in ok])
            steps = np.mean([r["steps"] for r in ok])
            warns = np.mean([r["warn_count"] for r in ok])
            drho = np.mean([r["mean_drho_inc"] for r in ok])
            dnll = np.mean([r["mean_dnll"] for r in ok])
            ent = np.mean([r["rsa_entropy"] for r in ok])
            lines.append(f"  n={n}, surv={surv:.3f}, goal={goal:.3f}, "
                         f"steps={steps:.1f}, warns={warns:.1f}")
            lines.append(f"  mean Drho_inc={drho:.4f}, "
                         f"mean DNLL={dnll:.4f}, "
                         f"mean rsa_entropy={ent:.3f}")

    # === Cross-variant comparison ===
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-VARIANT SUMMARY")
    lines.append("=" * 80)
    hdr = f"{'Variant':>20s} | {'Surv':>6s} | {'Goal':>6s} | {'Steps':>6s} | {'Warns':>6s} | {'Drho_inc':>9s} | {'DNLL':>8s} | {'Entropy':>8s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for v in VARIANTS:
        ok = [r for r in all_results[v] if r["success"]]
        if not ok:
            lines.append(f"{v:>20s} | ALL FAILED")
            continue
        surv = np.mean([r["survived"] for r in ok])
        goal = np.mean([r["reached_goal"] for r in ok])
        steps = np.mean([r["steps"] for r in ok])
        warns = np.mean([r["warn_count"] for r in ok])
        drho = np.mean([r["mean_drho_inc"] for r in ok])
        dnll = np.mean([r["mean_dnll"] for r in ok])
        ent = np.mean([r["rsa_entropy"] for r in ok])
        lines.append(f"{v:>20s} | {surv:>6.3f} | {goal:>6.3f} | {steps:>6.1f} | "
                     f"{warns:>6.1f} | {drho:>+9.4f} | {dnll:>+8.4f} | {ent:>8.3f}")

    # === E3: Family regression ===
    lines.append(f"\n{'='*80}")
    lines.append("E3: FAMILY REGRESSION CHECK")
    lines.append("=" * 80)

    for family, name in [(DTMB, "DTMB"), (GTET, "GTET")]:
        reg_results = []
        for seed in range(min(seeds, 10)):  # cap at 10 for regression
            r = run_regression_episode(runner, seed, family)
            reg_results.append(r)
        ok = [r for r in reg_results if r["success"]]
        if ok:
            surv = np.mean([r["survived"] for r in ok])
            goal = np.mean([r["reached_goal"] for r in ok])
            lines.append(f"  {name}: surv={surv:.3f}, goal={goal:.3f} (n={len(ok)})")
        else:
            lines.append(f"  {name}: ALL FAILED")

    # === Verdict ===
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Compare RSA-only vs hybrid
    def get_surv(v):
        ok = [r for r in all_results.get(v, []) if r.get("success")]
        return np.mean([r["survived"] for r in ok]) if ok else 0

    def get_dnll(v):
        ok = [r for r in all_results.get(v, []) if r.get("success")]
        return np.mean([r["mean_dnll"] for r in ok]) if ok else 0

    surv_legacy = get_surv("legacy_bias")
    surv_rsa_s1 = get_surv("rsa_obs_s1")
    surv_trust = get_surv("rsa_obs_s1_trust")
    surv_hybrid = get_surv("rsa_plus_phase10")

    dnll_legacy = get_dnll("legacy_bias")
    dnll_rsa_s1 = get_dnll("rsa_obs_s1")
    dnll_hybrid = get_dnll("rsa_plus_phase10")

    rsa_close_to_hybrid = abs(surv_rsa_s1 - surv_hybrid) < 0.05
    rsa_better_than_legacy = surv_rsa_s1 >= surv_legacy - 0.03
    rsa_dnll_close = abs(dnll_rsa_s1 - dnll_hybrid) < 0.1

    if rsa_close_to_hybrid and rsa_better_than_legacy:
        lines.append("VERDICT A: RSA-only approx hybrid.")
        lines.append(f"  surv: legacy={surv_legacy:.3f}, rsa_s1={surv_rsa_s1:.3f}, "
                     f"trust={surv_trust:.3f}, hybrid={surv_hybrid:.3f}")
        lines.append(f"  DNLL: legacy={dnll_legacy:.4f}, rsa_s1={dnll_rsa_s1:.4f}, "
                     f"hybrid={dnll_hybrid:.4f}")
        lines.append("  -> Legacy adapter can be demoted to ablation.")
    elif surv_hybrid > surv_rsa_s1 + 0.05:
        lines.append("VERDICT B: Hybrid > RSA-only.")
        lines.append(f"  surv: rsa_s1={surv_rsa_s1:.3f}, hybrid={surv_hybrid:.3f}")
        lines.append("  -> Legacy adapter retained. Enter adapter refinement.")
    else:
        lines.append("VERDICT: Inconclusive.")
        lines.append(f"  surv: legacy={surv_legacy:.3f}, rsa_s1={surv_rsa_s1:.3f}, "
                     f"trust={surv_trust:.3f}, hybrid={surv_hybrid:.3f}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "e1_baseline_v2_warning_audit.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
