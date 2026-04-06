"""GTET-L Step 4b — Fair Dispatch Re-Audit (PARALLEL).

Multi-process parallel execution for ~4-8x speedup.
Each (seed, factor_mode) pair is an independent job.
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
from copy import copy
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

FAMILY = "goal_preference_temptation_entanglement_lattice"
SEEDS = 30
N_WORKERS = min(12, os.cpu_count() or 4)  # 12 parallel workers


# ═══════════════════════════════════════════════════════════════
# Worker function — runs in separate process
# ═══════════════════════════════════════════════════════════════

def _run_single_episode(args):
    """Run a single episode. Must be top-level for pickling."""
    seed, diff, factor_mode, no_tutor, allowed_list, ucfg = args

    # Import inside worker to avoid pickling issues
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()

    try:
        allowed = frozenset(allowed_list) if allowed_list else None

        if no_tutor:
            s = runner.reset(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=False, intervention_family_mode=False,
                item_drop_enabled=False, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                user_cfg=ucfg)
        else:
            kw = dict(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=True, intervention_family_mode=True,
                item_drop_enabled=True, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                factor_mode=factor_mode, user_cfg=ucfg)
            if allowed:
                kw["allowed_interventions"] = allowed
            s = runner.reset(**kw)

        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        return {
            "seed": seed, "factor_mode": factor_mode,
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "warn": int(m.get("warnings", 0)),
        }
    except Exception as e:
        return {
            "seed": seed, "factor_mode": factor_mode,
            "survived": False, "reached_goal": False, "warn": 0,
            "error": str(e),
        }


def run_batch(diff, factor_modes, seeds=SEEDS, no_tutor=False,
              allowed_list=None, ucfg=None):
    """Run a batch of episodes in parallel. Returns {mode: [results]}."""
    jobs = []
    for fm in factor_modes:
        for seed in range(seeds):
            jobs.append((seed, diff, fm, no_tutor, allowed_list, ucfg))

    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_episode, j): j for j in jobs}
        for f in as_completed(futures):
            r = f.result()
            results[r["factor_mode"]].append(r)

    return dict(results)


def report(label, recs, ref_surv=None):
    n = len(recs)
    surv = np.mean([r["survived"] for r in recs])
    goal = np.mean([r["reached_goal"] for r in recs])
    warn = np.mean([r["warn"] for r in recs])
    delta = f"Δ={surv - ref_surv:+.3f}" if ref_surv is not None else ""
    print(f"  {label:25s}: surv={surv:.3f} goal={goal:.3f} "
          f"warn={warn:.1f} n={n} {delta}")
    return surv


# ═══════════════════════════════════════════════════════════════
# Exp B: Factor ablation on fair dispatch
# ═══════════════════════════════════════════════════════════════

def exp_b():
    print("=" * 70)
    print(f"Exp B: Factor Ablation — Fair Dispatch (hard, {SEEDS} seeds, {N_WORKERS} workers)")
    print("=" * 70)

    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z",
                    "G_ONLY", "THETA_ONLY", "Z_ONLY"]

    # Run all factor modes in parallel
    all_results = run_batch("hard", factor_modes)

    surv_full = None
    all_survs = {}
    for fm in factor_modes:
        s = report(fm, all_results[fm], ref_surv=surv_full)
        all_survs[fm] = s
        if fm == "FULL":
            surv_full = s

    # no_tutor baseline (parallel)
    nt_results = run_batch("hard", ["no_tutor"], no_tutor=True)
    s = report("no_tutor", nt_results.get("no_tutor", []), ref_surv=surv_full)
    all_survs["no_tutor"] = s

    # no_warn baseline
    nw_results = run_batch("hard", ["no_warn"],
                           allowed_list=["WAIT", "UNLOCK", "ITEM_DROP"])
    s = report("no_warn", nw_results.get("no_warn", []), ref_surv=surv_full)
    all_survs["no_warn"] = s

    # Summary
    best_abl = max(all_survs[m] for m in factor_modes if m != "FULL")
    delta_joint = all_survs["FULL"] - best_abl
    print(f"\n  Δ_joint (strict max) = {delta_joint:+.3f}")
    for fm in factor_modes:
        if fm == "FULL":
            continue
        d = all_survs["FULL"] - all_survs[fm]
        label = "NECESSARY" if d > 0.03 else ("HARMFUL" if d < -0.03 else "neutral")
        print(f"  Δ_factor({fm:12s}) = {d:+.3f}  [{label}]")


# ═══════════════════════════════════════════════════════════════
# Exp C: Lift U vs P (KL only — fast, no episode needed)
# ═══════════════════════════════════════════════════════════════

def exp_c():
    print("\n" + "=" * 70)
    print(f"Exp C: Lift U vs P — KL comparison ({SEEDS} seeds)")
    print("=" * 70)

    from src.teachers.gtet_factor_adapter import (
        build_factor_restricted_view, compute_posterior_epistemic_modifier,
    )
    from src.teachers.joint_goal_pref_posterior import (
        JointGoalPrefPosterior, THETA_2, DEFAULT_TEMPT_GRID,
        DEFAULT_TEMPT_PRIOR,
    )
    from src.agents.stochastic_agent_policy import BranchAttributes

    kl_u = defaultdict(list)
    kl_p = defaultdict(list)

    for seed in range(SEEDS):
        jgpp = JointGoalPrefPosterior(
            pref_types=THETA_2,
            tempt_grid=DEFAULT_TEMPT_GRID,
            tempt_prior=DEFAULT_TEMPT_PRIOR,
        )
        branches = [
            BranchAttributes(safety_score=0.9, temptation_score=0.1,
                             texture_novelty=0.2, shortcut_bonus=0.0,
                             risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.8,
                             texture_novelty=0.7, shortcut_bonus=0.2,
                             risk_penalty=0.5),
        ]
        for t in range(10):
            jgpp.update(None, branches, 0 if t < 5 else 1)
        q_full = jgpp._weights()

        for mode in ["G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
            q_u = build_factor_restricted_view(jgpp, mode, lift_mode="uniform")
            q_p = build_factor_restricted_view(jgpp, mode, lift_mode="prior")
            kl_u[mode].append(compute_posterior_epistemic_modifier(q_full, q_u))
            kl_p[mode].append(compute_posterior_epistemic_modifier(q_full, q_p))

    print(f"\n  {'Mode':12s} | {'KL_U':>8s} | {'KL_P':>8s} | {'diff':>8s}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for mode in ["G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
        u_m, p_m = np.mean(kl_u[mode]), np.mean(kl_p[mode])
        print(f"  {mode:12s} | {u_m:8.4f} | {p_m:8.4f} | {u_m-p_m:+8.4f}")
    order_u = sorted(kl_u.keys(), key=lambda m: np.mean(kl_u[m]))
    order_p = sorted(kl_p.keys(), key=lambda m: np.mean(kl_p[m]))
    print(f"\n  Ordering U: {' < '.join(order_u)}")
    print(f"  Ordering P: {' < '.join(order_p)}")
    print(f"  Same: {'YES' if order_u == order_p else 'NO'}")


# ═══════════════════════════════════════════════════════════════
# Exp D: Modifier F0/F1/F2/F3 (parallel)
# ═══════════════════════════════════════════════════════════════

def exp_d():
    print("\n" + "=" * 70)
    print(f"Exp D: Modifier F0/F1/F2/F3 — Fair Dispatch (hard, {SEEDS} seeds)")
    print("=" * 70)
    # All modifiers produce identical results (already confirmed twice).
    # Quick serial check with just FULL + G_Z to confirm.
    import src.envs.lattice_v2_runner as runner_mod
    from src.teachers.joint_goal_pref_posterior import DEFAULT_TEMPT_GRID
    original_fn = runner_mod._apply_gtet_factor_modifier

    def make_variant_fn(variant):
        def _v(s, icfg):
            from src.teachers.gtet_factor_adapter import (
                build_factor_restricted_view, compute_posterior_epistemic_modifier,
                compute_posterior_risk_modifier,
            )
            jgpp = s.gtet_posterior
            if jgpp is None: return icfg
            runner_mod._simulate_gtet_posterior_update(s)
            q_full = jgpp._weights()
            q_restricted = build_factor_restricted_view(jgpp, s.factor_mode)
            kl = compute_posterior_epistemic_modifier(q_full, q_restricted)
            rb = compute_posterior_risk_modifier(q_restricted, DEFAULT_TEMPT_GRID)
            icfg_new = copy(icfg)
            if variant == "F0": pass
            elif variant in ("F1","F2"):
                if kl > 0.01:
                    icfg_new.warn_effect_weight *= max(0.3, 1.0 - kl*0.5)
                    icfg_new.learning_gain_weight *= max(0.3, 1.0 - kl*0.3)
            if variant in ("F1","F3"):
                if rb > 0.02:
                    icfg_new.catastrophe_weight *= (1.0 + rb*2.0)
            s.gtet_action_log.append({"t": s.t, "variant": variant})
            return icfg_new
        return _v

    # Run serially but only 3 modes × 4 variants = manageable
    for variant in ["F0", "F1", "F2", "F3"]:
        runner_mod._apply_gtet_factor_modifier = make_variant_fn(variant)
        print(f"\n--- {variant} ---")
        for fm in ["FULL", "G_Z"]:
            results = []
            for seed in range(SEEDS):
                results.append(_run_single_episode(
                    (seed, "hard", fm, False, None, None)))
            report(fm, results)
    runner_mod._apply_gtet_factor_modifier = original_fn


# ═══════════════════════════════════════════════════════════════
# Exp E: z-sensitivity shifted configs (parallel)
# ═══════════════════════════════════════════════════════════════

def exp_e():
    print("\n" + "=" * 70)
    print(f"Exp E: z-sensitivity shifted configs — Fair Dispatch ({SEEDS} seeds)")
    print("=" * 70)

    configs = [
        ("baseline",       None),
        ("high_tempt",     {"lure_strength": 0.95, "tempt_offset_z": 0.90}),
        ("late_disambig",  {"goal_cue_leadlag": -4, "deadline_slack_final": 1.02}),
        ("tempt+tight",    {"lure_strength": 0.95, "tempt_offset_z": 0.90,
                            "deadline_slack_final": 1.02}),
    ]
    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY"]

    for cfg_label, ucfg in configs:
        print(f"\n--- {cfg_label} ---")

        # Parallel batch
        all_r = run_batch("hard", factor_modes, ucfg=ucfg)
        surv_full = None
        for fm in factor_modes:
            s = report(fm, all_r.get(fm, []), ref_surv=surv_full)
            if fm == "FULL": surv_full = s

        # no_tutor
        nt = run_batch("hard", ["no_tutor"], no_tutor=True, ucfg=ucfg)
        report("no_tutor", nt.get("no_tutor", []), ref_surv=surv_full)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.time()

    print(f"Using {N_WORKERS} parallel workers (CPU count: {os.cpu_count()})")
    print()

    exp_b()
    exp_c()
    exp_d()
    exp_e()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL FAIR DISPATCH EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
