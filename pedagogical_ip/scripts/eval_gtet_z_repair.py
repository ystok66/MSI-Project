"""GTET-L Step 4d — z-update repair + full re-audit (12 workers).

Exp A: z-update sanity (q(z) no longer frozen?)
Exp B: Predictor re-audit P1/P2/P3/P4 after z repair
Exp C: Factor ablation re-run after z repair
Exp D: Stronger temptation cue audit
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
                latent_mode=True, patch_radius=2, prefix_horizon=5)
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
# Exp A: z-update sanity
# ═══════════════════════════════════════════════════════════════

def exp_a():
    print("=" * 70)
    print(f"Exp A: z-Update Sanity (hard, {SEEDS} seeds)")
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
                factor_mode="FULL", predictor_mode="P1")

            while not s.done:
                s = runner.step(s)

            jgpp = s.gtet_posterior
            m = runner.get_metrics(s)
            if jgpp is not None:
                q = jgpp._weights()
                marg_z = q.sum(axis=(0, 1))
                ez = sum(marg_z[i] * DEFAULT_TEMPT_GRID[i]
                         for i in range(len(DEFAULT_TEMPT_GRID)))
                map_zi = int(np.argmax(marg_z))
                z_map = DEFAULT_TEMPT_GRID[map_zi]
                ent = -sum(p * np.log(p + 1e-12) for p in marg_z)

                flat = np.argmax(q)
                _, _, jzi = np.unravel_index(flat, q.shape)
                z_jmap = DEFAULT_TEMPT_GRID[jzi]

                # KL(q(z) || prior)
                prior = np.array([0.4, 0.3, 0.2, 0.1])
                kl = sum(marg_z[i] * np.log((marg_z[i] + 1e-12) / (prior[i] + 1e-12))
                         for i in range(4))
            else:
                ez = z_map = z_jmap = ent = kl = 0.0
                marg_z = [0.25]*4

            records.append({
                "seed": seed, "E[z]": ez, "MAP_z": z_map,
                "joint_MAP_z": z_jmap, "entropy_z": ent,
                "KL_from_prior": kl,
                "survived": bool(m["survived"]),
                "marg_z": [float(x) for x in marg_z],
            })
        except:
            pass

    if not records:
        print("  No valid records!")
        return

    survs = [r for r in records if r["survived"]]
    fails = [r for r in records if not r["survived"]]

    print(f"\n  Total: {len(records)}, survived: {len(survs)}, failed: {len(fails)}")
    print(f"\n  {'':15s} {'E[z]':>8s} {'MAP_z':>8s} {'JMAP_z':>8s} {'H(z)':>8s} {'KL':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, recs in [("all", records), ("survived", survs), ("failed", fails)]:
        if not recs:
            continue
        ez = np.mean([r["E[z]"] for r in recs])
        mz = np.mean([r["MAP_z"] for r in recs])
        jmz = np.mean([r["joint_MAP_z"] for r in recs])
        ent = np.mean([r["entropy_z"] for r in recs])
        kl = np.mean([r["KL_from_prior"] for r in recs])
        print(f"  {label:15s} {ez:8.3f} {mz:8.3f} {jmz:8.3f} {ent:8.3f} {kl:8.4f}")

    all_marg = np.mean([r["marg_z"] for r in records], axis=0)
    prior = [0.4, 0.3, 0.2, 0.1]
    print(f"\n  Mean q(z) distribution (after z-update repair):")
    for i, zv in enumerate(DEFAULT_TEMPT_GRID):
        bar = "█" * int(all_marg[i] * 50)
        print(f"    z={zv:.1f}: {all_marg[i]:.3f} (prior={prior[i]:.1f}) {bar}")

    mean_kl = np.mean([r["KL_from_prior"] for r in records])
    if mean_kl > 0.01:
        print(f"\n  ✓ q(z) HAS MOVED from prior (mean KL = {mean_kl:.4f})")
    else:
        print(f"\n  ✗ q(z) still frozen at prior (mean KL = {mean_kl:.4f})")

    # Check if survived/failed have different E[z]
    if survs and fails:
        ez_s = np.mean([r["E[z]"] for r in survs])
        ez_f = np.mean([r["E[z]"] for r in fails])
        print(f"\n  E[z] survived={ez_s:.3f} vs failed={ez_f:.3f} gap={ez_s - ez_f:+.3f}")


# ═══════════════════════════════════════════════════════════════
# Exp B: Predictor re-audit after z repair
# ═══════════════════════════════════════════════════════════════

def exp_b():
    print("\n" + "=" * 70)
    print(f"Exp B: Predictor P1/P2/P3/P4 after z-repair (hard, {SEEDS}s, {N_WORKERS}w)")
    print("=" * 70)

    predictors = ["P1", "P2", "P3", "P4"]
    factor_modes = ["FULL", "G_THETA", "G_Z"]

    jobs = []
    for pred in predictors:
        for fm in factor_modes:
            for seed in range(SEEDS):
                jobs.append((seed, "hard", fm, pred, False, None))
    # no_tutor
    for seed in range(SEEDS):
        jobs.append((seed, "hard", "FULL", "P1", True, None))

    all_results = run_batch(jobs)

    grouped = defaultdict(list)
    no_tutor_recs = []
    for r in all_results:
        if "error" not in r:
            grouped[(r["factor_mode"], r["predictor"])].append(r)
        else:
            no_tutor_recs.append(r)

    print(f"\n  {'Pred':6s} {'Mode':12s} {'Surv':>6s} {'Δ vs P1-FULL':>13s}")
    print(f"  {'-'*6} {'-'*12} {'-'*6} {'-'*13}")

    ref_surv = None
    for pred in predictors:
        for fm in factor_modes:
            recs = grouped.get((fm, pred), [])
            if not recs:
                continue
            surv = np.mean([r["survived"] for r in recs])
            if pred == "P1" and fm == "FULL":
                ref_surv = surv
            delta = f"{surv - ref_surv:+.3f}" if ref_surv is not None else ""
            print(f"  {pred:6s} {fm:12s} {surv:6.3f} {delta:>13s}")
        print()

    # Key analysis
    p1f = np.mean([r["survived"] for r in grouped[("FULL", "P1")]])
    p3f = np.mean([r["survived"] for r in grouped[("FULL", "P3")]])
    p4f = np.mean([r["survived"] for r in grouped[("FULL", "P4")]])
    p1gt = np.mean([r["survived"] for r in grouped[("G_THETA", "P1")]])
    p3gt = np.mean([r["survived"] for r in grouped[("G_THETA", "P3")]])

    print("--- Key Analysis ---")
    print(f"  P1 FULL={p1f:.3f}  P3 FULL={p3f:.3f}  P4 FULL={p4f:.3f}")
    print(f"  P1 G_THETA={p1gt:.3f}")

    if p3f >= p1gt - 0.03:
        print("  → P3 FULL ≥ P1 G_THETA: z IS useful under mixture predictor!")
        print("  → Problem was E[z] summary, not z itself")
    else:
        print("  → P3 FULL < P1 G_THETA: z still harmful even with mixture predictor")

    if p3f > p4f + 0.03:
        print(f"  → P3 > P4: z adds info under mixture ({p3f:.3f} vs {p4f:.3f})")
    elif abs(p3f - p4f) <= 0.03:
        print(f"  → P3 ≈ P4: z neutral under mixture ({p3f:.3f} vs {p4f:.3f})")
    else:
        print(f"  → P3 < P4: z harmful even under mixture ({p3f:.3f} vs {p4f:.3f})")


# ═══════════════════════════════════════════════════════════════
# Exp C: Factor ablation re-run (using best predictor)
# ═══════════════════════════════════════════════════════════════

def exp_c():
    print("\n" + "=" * 70)
    print(f"Exp C: Factor Ablation (best predictor per mode, hard, {SEEDS}s)")
    print("=" * 70)

    # Test with P3 (route mixture) — the candidate canonical
    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z",
                    "G_ONLY", "THETA_ONLY", "Z_ONLY"]

    for pred in ["P3", "P4"]:
        print(f"\n--- Predictor = {pred} ---")
        jobs = []
        for fm in factor_modes:
            for seed in range(SEEDS):
                jobs.append((seed, "hard", fm, pred, False, None))

        # Baselines
        for seed in range(SEEDS):
            jobs.append((seed, "hard", "FULL", pred, True, None))

        all_results = run_batch(jobs)
        grouped = defaultdict(list)
        for r in all_results:
            grouped[r["factor_mode"]].append(r)

        surv_full = None
        for fm in factor_modes:
            recs = [r for r in grouped[fm] if "error" not in r]
            s = report(fm, recs, ref_surv=surv_full)
            if fm == "FULL":
                surv_full = s

        nt_recs = [r for r in all_results if r.get("error")]
        if nt_recs:
            report("no_tutor", nt_recs, ref_surv=surv_full)


# ═══════════════════════════════════════════════════════════════
# Exp D: Stronger temptation cue audit
# ═══════════════════════════════════════════════════════════════

def exp_d():
    print("\n" + "=" * 70)
    print(f"Exp D: Temptation Cue Strength Audit (hard, {SEEDS}s)")
    print("=" * 70)

    # Test: does stronger tempt cue improve z discrimination?
    # We'll test by running with different user_cfg lure_strength
    configs = [
        ("baseline",       None),
        ("high_tempt",     {"lure_strength": 0.95, "tempt_offset_z": 0.90}),
    ]

    for label, ucfg in configs:
        print(f"\n--- {label} ---")
        jobs = []
        for fm in ["FULL", "G_THETA"]:
            for pred in ["P3", "P4"]:
                for seed in range(SEEDS):
                    jobs.append((seed, "hard", fm, pred, False, ucfg))

        all_results = run_batch(jobs)
        grouped = defaultdict(list)
        for r in all_results:
            grouped[(r["factor_mode"], r["predictor"])].append(r)

        for pred in ["P3", "P4"]:
            for fm in ["FULL", "G_THETA"]:
                recs = grouped.get((fm, pred), [])
                if recs:
                    surv = np.mean([r["survived"] for r in recs])
                    print(f"  {pred} {fm:12s}: surv={surv:.3f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print(f"Workers: {N_WORKERS}, CPU: {os.cpu_count()}")

    exp_a()
    exp_b()
    exp_c()
    exp_d()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL STEP 4d EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
