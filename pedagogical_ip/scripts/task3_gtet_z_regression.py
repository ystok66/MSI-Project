"""
Task 3 — GTET z Demotion Regression Test.

E3.1: Default regression (G_THETA vs FULL) on GTET + DTMB + baseline_v2.
E3.2: with-z vs no-z ablation comparison.

Usage:
    python scripts/task3_gtet_z_regression.py [--seeds 20] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner


def parse_args():
    p = argparse.ArgumentParser(description="Task 3: GTET z demotion regression")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="3 seeds")
    p.add_argument("--output-dir", default="results/task3")
    return p.parse_args()


def run_episode(runner, seed, family, factor_mode="G_THETA"):
    """Run one episode."""
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
        factor_mode=factor_mode,
    )
    try:
        state = runner.reset(seed=seed, **cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        return {
            "survived": metrics["survived"],
            "reached_goal": metrics["reached_goal"],
            "steps": metrics["steps"],
            "warn_count": metrics["warn_count"],
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "survived": False, "reached_goal": False}


def main():
    args = parse_args()
    seeds = 3 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    families = {
        "GTET": "goal_preference_temptation_entanglement_lattice",
        "DTMB": "deep_tree_mixed_bottleneck_lattice",
        "BL_V2": None,  # baseline_v2
    }
    modes = ["G_THETA", "FULL"]

    lines = []
    lines.append("Task 3 - GTET z Demotion Regression Test")
    lines.append(f"  seeds={seeds}")
    lines.append("=" * 80)

    all_results = {}

    for fname, family in families.items():
        for fm in modes:
            key = (fname, fm)
            lines.append(f"\n--- {fname} x factor_mode={fm} ---")
            results = []
            for seed in range(seeds):
                m = run_episode(runner, seed, family, factor_mode=fm)
                results.append(m)
                if not m["success"]:
                    lines.append(f"  seed={seed} FAILED: {m.get('error', '?')}")
            all_results[key] = results
            ok = [r for r in results if r["success"]]
            if ok:
                surv = np.mean([r["survived"] for r in ok])
                goal = np.mean([r["reached_goal"] for r in ok])
                steps = np.mean([r["steps"] for r in ok])
                lines.append(f"  n={len(ok)}, surv={surv:.3f}, goal={goal:.3f}, steps={steps:.1f}")

    # Summary
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY TABLE")
    lines.append("=" * 80)
    hdr = f"{'Family':>6s} {'Mode':>10s} | {'Surv':>6s} | {'Goal':>6s} | {'Steps':>6s} | {'N':>4s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for fname in families:
        for fm in modes:
            key = (fname, fm)
            ok = [r for r in all_results.get(key, []) if r.get("success")]
            if ok:
                surv = np.mean([r["survived"] for r in ok])
                goal = np.mean([r["reached_goal"] for r in ok])
                steps = np.mean([r["steps"] for r in ok])
                n = len(ok)
                lines.append(f"{fname:>6s} {fm:>10s} | {surv:>6.3f} | {goal:>6.3f} | {steps:>6.1f} | {n:>4d}")

    # Verdict
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Check GTET regression
    gtet_noz = [r for r in all_results.get(("GTET", "G_THETA"), []) if r.get("success")]
    gtet_full = [r for r in all_results.get(("GTET", "FULL"), []) if r.get("success")]
    dtmb_noz = [r for r in all_results.get(("DTMB", "G_THETA"), []) if r.get("success")]
    dtmb_full = [r for r in all_results.get(("DTMB", "FULL"), []) if r.get("success")]

    if gtet_noz and gtet_full:
        s_noz = np.mean([r["survived"] for r in gtet_noz])
        s_full = np.mean([r["survived"] for r in gtet_full])
        g_noz = np.mean([r["reached_goal"] for r in gtet_noz])
        g_full = np.mean([r["reached_goal"] for r in gtet_full])
        surv_drop = s_full - s_noz
        goal_drop = g_full - g_noz

        lines.append(f"GTET surv: FULL={s_full:.3f}, G_THETA={s_noz:.3f}, drop={surv_drop:.3f}")
        lines.append(f"GTET goal: FULL={g_full:.3f}, G_THETA={g_noz:.3f}, drop={goal_drop:.3f}")

        if abs(surv_drop) <= 0.02 and abs(goal_drop) <= 0.02:
            lines.append("→ GTET regression PASS: drop ≤ 0.02")
        elif abs(surv_drop) <= 0.05 and abs(goal_drop) <= 0.05:
            lines.append("→ GTET regression MARGINAL: drop ≤ 0.05")
        else:
            lines.append("→ GTET regression FAIL: drop > 0.05")

    if dtmb_noz and dtmb_full:
        s_noz = np.mean([r["survived"] for r in dtmb_noz])
        s_full = np.mean([r["survived"] for r in dtmb_full])
        surv_drop = s_full - s_noz
        lines.append(f"DTMB surv: FULL={s_full:.3f}, G_THETA={s_noz:.3f}, drop={surv_drop:.3f}")
        if abs(surv_drop) <= 0.03:
            lines.append("→ DTMB regression PASS")
        else:
            lines.append("→ DTMB regression FLAG")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "z_demotion_regression.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
