"""GTET-L Step 4c — z-diagnosis & predictor audit (12 workers).

Exp 1: Predictor comparison P1/P2/P3/P4 × factor modes
Exp 2: z calibration (E[z], MAP_z, entropy)
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

FAMILY = "goal_preference_temptation_entanglement_lattice"
SEEDS = 30
N_WORKERS = min(12, os.cpu_count() or 4)


def _run_episode(args):
    """Worker: run single GTET episode."""
    seed, diff, factor_mode, predictor_mode, no_tutor, ucfg = args
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()

    try:
        if no_tutor:
            s = runner.reset(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=False, intervention_family_mode=False,
                item_drop_enabled=False, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                user_cfg=ucfg)
        else:
            s = runner.reset(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=True, intervention_family_mode=True,
                item_drop_enabled=True, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                factor_mode=factor_mode, predictor_mode=predictor_mode,
                user_cfg=ucfg)

        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        return {
            "seed": seed, "factor_mode": factor_mode,
            "predictor": predictor_mode,
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "warn": int(m.get("warnings", 0)),
        }
    except Exception as e:
        return {
            "seed": seed, "factor_mode": factor_mode,
            "predictor": predictor_mode,
            "survived": False, "reached_goal": False,
            "warn": 0, "error": str(e),
        }


def run_batch(jobs):
    """Run batch of jobs in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_episode, j): j for j in jobs}
        for f in as_completed(futures):
            results.append(f.result())
    return results


def report(label, recs, ref_surv=None):
    n = len(recs)
    surv = np.mean([r["survived"] for r in recs])
    goal = np.mean([r["reached_goal"] for r in recs])
    delta = f"Δ={surv - ref_surv:+.3f}" if ref_surv is not None else ""
    print(f"  {label:30s}: surv={surv:.3f} goal={goal:.3f} n={n} {delta}")
    return surv


# ═══════════════════════════════════════════════════════════════
# Exp 1: Predictor comparison
# ═══════════════════════════════════════════════════════════════

def exp1_predictor():
    print("=" * 70)
    print(f"Exp 1: Predictor P1/P2/P3/P4 × Factor Modes (hard, {SEEDS}s, {N_WORKERS}w)")
    print("=" * 70)

    predictors = ["P1", "P2", "P3", "P4"]
    factor_modes = ["FULL", "G_THETA", "G_Z"]

    # Build all jobs
    jobs = []
    for pred in predictors:
        for fm in factor_modes:
            for seed in range(SEEDS):
                jobs.append((seed, "hard", fm, pred, False, None))

    # Also baselines (no_tutor)
    for seed in range(SEEDS):
        jobs.append((seed, "hard", "FULL", "P1", True, None))

    all_results = run_batch(jobs)

    # Group results
    grouped = defaultdict(list)
    for r in all_results:
        if r.get("error") and r.get("predictor") == "P1" and r.get("factor_mode") == "FULL":
            # no_tutor fallback
            key = ("no_tutor", "—")
        else:
            key = (r["factor_mode"], r["predictor"])
        grouped[key].append(r)

    # Report table
    print(f"\n  {'Predictor':8s} {'Mode':12s} {'Survival':>8s} {'Δ vs P1-FULL':>12s}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*12}")

    ref_surv = None
    for pred in predictors:
        for fm in factor_modes:
            key = (fm, pred)
            recs = grouped.get(key, [])
            if not recs:
                continue
            surv = np.mean([r["survived"] for r in recs])
            if pred == "P1" and fm == "FULL":
                ref_surv = surv
            delta = f"{surv - ref_surv:+.3f}" if ref_surv is not None else ""
            print(f"  {pred:8s} {fm:12s} {surv:8.3f} {delta:>12s}")
        print()

    # No-tutor baseline
    nt_recs = [r for r in all_results
               if r.get("predictor") == "P1" and r.get("factor_mode") == "FULL"
               and r.get("survived") is not None]
    nt_no_tutor = grouped.get(("no_tutor", "—"), [])
    if nt_no_tutor and ref_surv is not None:
        surv = np.mean([r["survived"] for r in nt_no_tutor])
        print(f"  {'—':8s} {'no_tutor':12s} {surv:8.3f} {surv - ref_surv:+12.3f}")

    # Key analysis
    print("\n--- Key Analysis ---")
    p1_full = np.mean([r["survived"] for r in grouped[("FULL", "P1")]])
    p3_full = np.mean([r["survived"] for r in grouped[("FULL", "P3")]])
    p4_full = np.mean([r["survived"] for r in grouped[("FULL", "P4")]])
    p1_gt = np.mean([r["survived"] for r in grouped[("G_THETA", "P1")]])
    p3_gt = np.mean([r["survived"] for r in grouped[("G_THETA", "P3")]])

    print(f"  P1 FULL = {p1_full:.3f}, P3 FULL = {p3_full:.3f}")
    print(f"  P1 G_THETA = {p1_gt:.3f}")

    if p3_full >= p1_gt - 0.03:
        print("  → P3 recovers FULL to G_THETA level → problem was E[z] summary, NOT z itself")
    elif p3_full < p1_gt - 0.03:
        print("  → P3 still < G_THETA → z is genuinely harmful even with mixture predictor")

    if abs(p3_full - p4_full) < 0.03:
        print(f"  → P3 ≈ P4 ({p3_full:.3f} vs {p4_full:.3f}) → z adds no info to route prediction")
    elif p3_full > p4_full + 0.03:
        print(f"  → P3 > P4 ({p3_full:.3f} vs {p4_full:.3f}) → z IS useful under mixture predictor")


# ═══════════════════════════════════════════════════════════════
# Exp 2: z calibration audit
# ═══════════════════════════════════════════════════════════════

def exp2_z_calibration():
    print("\n" + "=" * 70)
    print(f"Exp 2: z Calibration Audit (hard, {SEEDS} seeds)")
    print("=" * 70)

    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.teachers.joint_goal_pref_posterior import DEFAULT_TEMPT_GRID

    runner = LatticeV2Runner()

    records = []
    for seed in range(SEEDS):
        try:
            s = runner.reset(
                seed=seed, difficulty="hard", scenario_family=FAMILY,
                robot_belief_mode=True, intervention_family_mode=True,
                item_drop_enabled=True, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                factor_mode="FULL")

            # Record pre-warning posterior
            jgpp = s.gtet_posterior
            if jgpp is not None:
                q = jgpp._weights()
                marg_z = q.sum(axis=(0, 1))
                ez = sum(marg_z[i] * DEFAULT_TEMPT_GRID[i]
                         for i in range(len(DEFAULT_TEMPT_GRID)))
                map_zi = int(np.argmax(marg_z))
                z_map = DEFAULT_TEMPT_GRID[map_zi]
                ent = -sum(p * np.log(p + 1e-12) for p in marg_z)

                # Joint MAP
                flat = np.argmax(q)
                gi, pi, zi = np.unravel_index(flat, q.shape)
                z_joint_map = DEFAULT_TEMPT_GRID[zi]
            else:
                ez = z_map = z_joint_map = ent = 0.0

            # Run episode to get outcome
            while not s.done:
                s = runner.step(s)
            m = runner.get_metrics(s)

            records.append({
                "seed": seed, "E[z]": ez, "MAP_z": z_map,
                "joint_MAP_z": z_joint_map, "entropy_z": ent,
                "survived": bool(m["survived"]),
                "marg_z": [float(x) for x in marg_z],
            })
        except:
            pass

    if not records:
        print("  No valid records!")
        return

    # Summary stats
    survs = [r for r in records if r["survived"]]
    fails = [r for r in records if not r["survived"]]

    print(f"\n  Total: {len(records)}, survived: {len(survs)}, failed: {len(fails)}")
    print(f"\n  {'':15s} {'E[z]':>8s} {'MAP_z':>8s} {'JMAP_z':>8s} {'H(z)':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, recs in [("all", records), ("survived", survs), ("failed", fails)]:
        if not recs:
            continue
        ez = np.mean([r["E[z]"] for r in recs])
        mz = np.mean([r["MAP_z"] for r in recs])
        jmz = np.mean([r["joint_MAP_z"] for r in recs])
        ent = np.mean([r["entropy_z"] for r in recs])
        print(f"  {label:15s} {ez:8.3f} {mz:8.3f} {jmz:8.3f} {ent:8.3f}")

    # Marginal z distribution
    all_marg = np.mean([r["marg_z"] for r in records], axis=0)
    print(f"\n  Mean q(z) distribution:")
    for i, zv in enumerate(DEFAULT_TEMPT_GRID):
        bar = "█" * int(all_marg[i] * 50)
        print(f"    z={zv:.1f}: {all_marg[i]:.3f} {bar}")

    print(f"\n  Prior: {dict(zip(DEFAULT_TEMPT_GRID, (0.4, 0.3, 0.2, 0.1)))}")
    print(f"  → Notice: prior is heavily skewed toward z=0.0 (40%)")

    # Diagnosis
    if np.mean([r["E[z]"] for r in records]) < 0.25:
        print("\n  ⚠ E[z] systematically < 0.25 — confirms prior-driven low bias")
        print("    The prior (0.4, 0.3, 0.2, 0.1) pushes E[z] ≈ 0.2")
        print("    This causes P1 to ALWAYS predict lower route")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Workers: {N_WORKERS}, CPU: {os.cpu_count()}")

    exp1_predictor()
    exp2_z_calibration()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
