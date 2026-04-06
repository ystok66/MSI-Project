"""GTET-L Redundancy Audit — Lift, Modifier, and z-sensitivity.

Exp 1: Lift U vs Lift P
Exp 2: Modifier F1 (both), F2 (warn-only), F3 (catastrophe-only)
Exp 3: z-sensitivity in shifted configs
Exp 4: Full factor ranking table
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from copy import copy
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.gtet_factor_adapter import (
    build_factor_restricted_view,
    compute_posterior_epistemic_modifier,
    compute_posterior_risk_modifier,
)

FAMILY = "goal_preference_temptation_entanglement_lattice"
SEEDS = 30
runner = LatticeV2Runner()


def run_ep(seed, diff, factor_mode="FULL", no_tutor=False,
           allowed=None, ucfg=None):
    try:
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
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "warn": int(m.get("warnings", 0)),
            "steps": int(m["steps"]),
        }
    except:
        return {"survived": False, "reached_goal": False, "warn": 0, "steps": 0}


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
# Exp 1: Lift U vs Lift P (KL and survival comparison)
# ═══════════════════════════════════════════════════════════════

def exp1_lift_audit():
    print("=" * 70)
    print("Exp 1: Lift U vs Lift P — KL comparison (hard, 30 seeds)")
    print("=" * 70)

    # Run episodes with FULL to get posteriors, then compare lifts offline
    from src.teachers.joint_goal_pref_posterior import (
        JointGoalPrefPosterior, THETA_2, DEFAULT_TEMPT_GRID,
        DEFAULT_TEMPT_PRIOR,
    )
    from src.agents.stochastic_agent_policy import BranchAttributes

    kl_u = defaultdict(list)
    kl_p = defaultdict(list)

    for seed in range(SEEDS):
        # Create posterior and simulate updates
        jgpp = JointGoalPrefPosterior(
            pref_types=THETA_2,
            tempt_grid=DEFAULT_TEMPT_GRID,
            tempt_prior=DEFAULT_TEMPT_PRIOR,
        )

        # 10 updates with contrasting branches
        branches = [
            BranchAttributes(safety_score=0.9, temptation_score=0.1,
                             texture_novelty=0.2, shortcut_bonus=0.0,
                             risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.8,
                             texture_novelty=0.7, shortcut_bonus=0.2,
                             risk_penalty=0.5),
        ]
        for t in range(10):
            obs = 0 if t < 5 else 1
            jgpp.update(None, branches, obs)

        q_full = jgpp._weights()

        for mode in ["G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
            q_u = build_factor_restricted_view(jgpp, mode, lift_mode="uniform")
            q_p = build_factor_restricted_view(jgpp, mode, lift_mode="prior")

            kl_u[mode].append(compute_posterior_epistemic_modifier(q_full, q_u))
            kl_p[mode].append(compute_posterior_epistemic_modifier(q_full, q_p))

    print(f"\n  {'Mode':12s} | {'KL_U mean':>10s} {'KL_U std':>10s} | "
          f"{'KL_P mean':>10s} {'KL_P std':>10s} | {'diff':>8s}")
    print(f"  {'-'*12}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*8}")
    for mode in ["G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
        u_m = np.mean(kl_u[mode])
        u_s = np.std(kl_u[mode])
        p_m = np.mean(kl_p[mode])
        p_s = np.std(kl_p[mode])
        diff = u_m - p_m
        print(f"  {mode:12s} | {u_m:10.4f} {u_s:10.4f} | "
              f"{p_m:10.4f} {p_s:10.4f} | {diff:+8.4f}")

    # Check if ordering is the same
    order_u = sorted(kl_u.keys(), key=lambda m: np.mean(kl_u[m]))
    order_p = sorted(kl_p.keys(), key=lambda m: np.mean(kl_p[m]))
    print(f"\n  KL ordering U: {' < '.join(order_u)}")
    print(f"  KL ordering P: {' < '.join(order_p)}")
    print(f"  Same ordering: {'YES' if order_u == order_p else 'NO'}")

    # Verdict
    max_diff = max(abs(np.mean(kl_u[m]) - np.mean(kl_p[m]))
                   for m in kl_u.keys())
    print(f"\n  Max |KL_U - KL_P|: {max_diff:.4f}")
    if max_diff < 0.1:
        print("  → Lifts are equivalent. Keep simpler Lift U.")
    else:
        print("  → Lifts differ. Check survival impact before deciding.")


# ═══════════════════════════════════════════════════════════════
# Exp 2: Modifier audit F1/F2/F3
# ═══════════════════════════════════════════════════════════════

def exp2_modifier_audit():
    print("\n" + "=" * 70)
    print("Exp 2: Modifier Audit F1/F2/F3 — survival on hard, 30 seeds")
    print("=" * 70)

    # We need to test with modifier variants. Since the modifier is in
    # _apply_gtet_factor_modifier, we'll monkey-patch it for each variant.
    import src.envs.lattice_v2_runner as runner_mod
    from src.teachers.joint_goal_pref_posterior import DEFAULT_TEMPT_GRID

    original_fn = runner_mod._apply_gtet_factor_modifier

    def make_variant_fn(variant):
        """Create modifier function for variant F1/F2/F3."""
        def _variant_modifier(s, icfg):
            from src.teachers.gtet_factor_adapter import (
                build_factor_restricted_view,
                compute_posterior_epistemic_modifier,
                compute_posterior_risk_modifier,
            )
            jgpp = s.gtet_posterior
            if jgpp is None:
                return icfg

            runner_mod._simulate_gtet_posterior_update(s)

            q_full = jgpp._weights()
            q_restricted = build_factor_restricted_view(jgpp, s.factor_mode)
            kl_div = compute_posterior_epistemic_modifier(q_full, q_restricted)
            risk_bias = compute_posterior_risk_modifier(
                q_restricted, DEFAULT_TEMPT_GRID)

            icfg_new = copy(icfg)

            if variant in ("F1", "F2") and kl_div > 0.01:
                icfg_new.warn_effect_weight *= max(0.3, 1.0 - kl_div * 0.5)
                icfg_new.learning_gain_weight *= max(0.3, 1.0 - kl_div * 0.3)

            if variant in ("F1", "F3") and risk_bias > 0.02:
                icfg_new.catastrophe_weight *= (1.0 + risk_bias * 2.0)

            s.gtet_action_log.append({
                "t": s.t, "factor_mode": s.factor_mode,
                "kl_div": float(kl_div), "risk_bias": float(risk_bias),
                "variant": variant,
            })
            return icfg_new
        return _variant_modifier

    for variant in ["F1", "F2", "F3"]:
        print(f"\n--- Modifier {variant} ---")
        runner_mod._apply_gtet_factor_modifier = make_variant_fn(variant)

        for factor_label, factor_cfg in [
            ("FULL", {}),
            ("G_Z", {"factor_mode": "G_Z"}),
            ("THETA_Z", {"factor_mode": "THETA_Z"}),
        ]:
            recs = []
            for seed in range(SEEDS):
                r = run_ep(seed, "hard", **factor_cfg)
                recs.append(r)
            report(factor_label, recs)

    # Restore
    runner_mod._apply_gtet_factor_modifier = original_fn


# ═══════════════════════════════════════════════════════════════
# Exp 3: z-sensitivity in harder configs
# ═══════════════════════════════════════════════════════════════

def exp3_z_sensitivity():
    print("\n" + "=" * 70)
    print("Exp 3: z-sensitivity in shifted configs (30 seeds)")
    print("=" * 70)

    configs = [
        ("baseline",       {}),
        ("high_tempt",     {"lure_strength": 0.95, "tempt_offset_z": 0.90}),
        ("late_disambig",  {"goal_cue_leadlag": -4, "deadline_slack_final": 1.02}),
        ("tempt+tight",    {"lure_strength": 0.95, "tempt_offset_z": 0.90,
                            "deadline_slack_final": 1.02}),
    ]

    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY"]

    for cfg_label, ucfg in configs:
        print(f"\n--- {cfg_label} ---")
        surv_full = None
        for fm in factor_modes:
            recs = []
            for seed in range(SEEDS):
                r = run_ep(seed, "hard", factor_mode=fm,
                           ucfg=ucfg if ucfg else None)
                recs.append(r)
            s = report(fm, recs, ref_surv=surv_full)
            if fm == "FULL":
                surv_full = s

        # Also run no_tutor for reference
        recs = [run_ep(seed, "hard", no_tutor=True,
                        ucfg=ucfg if ucfg else None)
                for seed in range(SEEDS)]
        report("no_tutor", recs, ref_surv=surv_full)


# ═══════════════════════════════════════════════════════════════
# Exp 4: Full factor ranking table
# ═══════════════════════════════════════════════════════════════

def exp4_ranking():
    print("\n" + "=" * 70)
    print("Exp 4: Full Factor Ranking (hard, 30 seeds)")
    print("=" * 70)

    factor_modes = ["FULL", "G_THETA", "G_Z", "THETA_Z",
                    "G_ONLY", "THETA_ONLY", "Z_ONLY"]

    surv_full = None
    all_survs = {}
    for fm in factor_modes:
        recs = [run_ep(seed, "hard", factor_mode=fm)
                for seed in range(SEEDS)]
        s = report(fm, recs, ref_surv=surv_full)
        all_survs[fm] = s
        if fm == "FULL":
            surv_full = s

    # Add baselines
    recs = [run_ep(seed, "hard", no_tutor=True)
            for seed in range(SEEDS)]
    s = report("no_tutor", recs, ref_surv=surv_full)
    all_survs["no_tutor"] = s

    recs = [run_ep(seed, "hard",
                   allowed=frozenset({"WAIT", "UNLOCK", "ITEM_DROP"}))
            for seed in range(SEEDS)]
    s = report("no_warn", recs, ref_surv=surv_full)
    all_survs["no_warn"] = s

    # Summary
    print("\n--- Factor Necessity Summary ---")
    best_ablated = max(all_survs[m] for m in factor_modes if m != "FULL")
    delta_joint = all_survs["FULL"] - best_ablated
    print(f"  Δ_joint (strict max) = {delta_joint:+.3f}")

    for fm in factor_modes:
        if fm == "FULL":
            continue
        d = all_survs["FULL"] - all_survs[fm]
        label = "NECESSARY" if d > 0.03 else "redundant?"
        print(f"  Δ_factor({fm:12s}) = {d:+.3f}  [{label}]")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exp1_lift_audit()
    exp2_modifier_audit()
    exp3_z_sensitivity()
    exp4_ranking()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
