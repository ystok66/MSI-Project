"""Step 5: Contract Closure — Held-out Promotion Audit + Sensitivity + Parity.

Combines three audits in one script:

A. HELD-OUT FAMILY AUDIT
   Arms: v2.1 vs v2 baseline, across ALL available families
   Tests generalization beyond TIC-v4 core subtypes

B. PARAMETER SENSITIVITY / PLATEAU AUDIT
   Sweeps β_T, β_L, β_D, β_C, δ, τ_N, λ_cc around defaults
   Reports plateau width where SelGap+TBSR stay within tolerance

C. TERNARY vs BINARY CLOSURE (final confirmation)
   v2.1 three-outcome vs binary rollback

D. FALLBACK RATE AUDIT
   Tracks p_self_posterior_shadow fallback across families

E. MIRROR / SIDE-SWAP PARITY
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import PROBE_NAMES, LESSON_V2_BY_NAME
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.micro_bayes_shadow_v2_1 import MicroBayesShadowV2_1
from src.teachers.credit_correction import CreditCorrection
from src.teachers.p_self_posterior_shadow import compute_p_self_posterior, PSelfMode
from src.teachers.internalization_observer import A1MtObserverFrozen
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob
from src.agents.behavior_probes import BEHAVIOR_ZONES

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

ALL_LESSONS = []
for name in sorted(LESSON_V2_BY_NAME.keys()):
    ALL_LESSONS.append((name, LESSON_V2_BY_NAME[name]))

N_SESSIONS = 2
N_STEPS = 20
N_SEEDS = 6


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_single(m, scorer, theta, lesson, step_idx, seed, rng, force_binary=False):
    """Run one step. Returns result dict with fallback tracking."""
    ub = {p: 0.4 + 0.1 * (step_idx / N_STEPS) for p in PROBE_NAMES}
    et = generate_episode_from_lesson_v2(
        lesson, step_idx + seed * 10000, theta, ub, rng)
    ep, spec, gm, cfg_e, meta, sc = et
    fb, ww = apply_fix(meta, sc)
    fv = np.full_like(fb, 0.3)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    for _ in range(3):
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL: continue
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

    lib = BranchConceptLibrary()
    scr = BranchScorerProbe(lr=0.05, l2=0.01)
    ss = summarize_branch(sc.safe_cells, fb, fv, lp)
    sr = summarize_branch(sc.risky_cells, fb, fv, lp)
    lib.update("safe", ss); lib.update("risky", sr)
    scr.update(build_scorer_input(ss, lib), 1.0)
    scr.update(build_scorer_input(sr, lib), 0.0)

    dc = getattr(sc, 'commit_depth', 3)
    dr = getattr(sc, 'reveal_depth', 2)
    risk = getattr(sc, 'risk_level', 0.3)
    tempt = getattr(sc, 'temptation_strength', 0.0)
    subtype = getattr(ep, 'subtype', '')
    p_self_base = estimate_self_discovery_prob(dc, dr)
    has_self_ev = (2 >= dc - 1) or p_self_base > 0.5

    # Three-outcome posterior
    ps_r = compute_p_self_posterior(
        PSelfMode.POSTERIOR_C, dc, dr,
        tau_hat=m.tau, nu_hat=m.nu, gamma_gen_hat=m.gamma_gen)
    p_s = ps_r["p_self"]
    p_f = ps_r["p_fail"]
    p_u = ps_r.get("p_undecided", 0.0)
    is_fallback = ps_r.get("fallback", False)

    if force_binary:
        p_f = 1.0 - p_s
        p_u = 0.0

    from src.envs.observation_mask import make_observation_mask
    fork = sc.fork_cell
    mask_a = make_observation_mask(sc.branch_a_cells, fork, 2)
    mask_b = make_observation_mask(sc.branch_b_cells, fork, 2)
    vis_a = [c for c, mm in zip(sc.branch_a_cells, mask_a) if mm > 0.5]
    vis_b = [c for c, mm in zip(sc.branch_b_cells, mask_b) if mm > 0.5]
    sa = summarize_branch(vis_a, fb, fv, lp)
    sb = summarize_branch(vis_b, fb, fv, lp)
    sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
    sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
    delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
    dvoi = max(float(1.0/(1.0+np.exp(-abs(sa2[0]-sb2[0])))) -
               float(1.0/(1.0+np.exp(-abs(sa[0]-sb[0])))), 0)
    zones = BEHAVIOR_ZONES.get("safe", {})
    novelty = 0.3 if subtype == "beneficial_novelty" else 0.0
    self_ev = 0.7 if has_self_ev else 0.3

    act, dose, info = scorer.score(
        m, delta_s, dvoi, tempt, risk,
        p_s, p_f, p_u, subtype, has_self_ev, zones, novelty, self_ev)

    bas = BranchAttributes(
        safety_score=float(ss[0]), risk_penalty=0.1,
        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                          else sc.tempt_score_b)
    bar = BranchAttributes(
        safety_score=float(sr[0]), risk_penalty=risk,
        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                          else sc.tempt_score_a)
    ac = sample_factored_choice([bas, bar], theta, m, AP, rng,
                                [0.0, 0.0], [False, False])
    correct = (ac == sc.oracle_safe_branch_id)
    warned = dose > 0
    self_disc = correct and not warned and p_self_base > 0.5
    need_warn = (risk > 0.3 and p_self_base < 0.4)

    if warned:
        m.update_trust(warn_helpful=(risk > 0.25 and correct))
        if p_self_base < 0.5: m.update_dependence(blind_obey=True)
        m.update_gamma_gen(sustained_pressure=True)
    else:
        if self_disc:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
    if not correct and tempt > 0.5:
        m.update_gamma_spec(tempt_error=True)
    m.update_risk(risk if not correct else 0.05, 0.15)
    m.snapshot()

    leakage = info.get("leakage", 0.0)
    return {
        "correct": correct, "warned": warned, "self_disc": self_disc,
        "subtype": subtype, "need_warn": need_warn,
        "leakage": leakage, "fallback": is_fallback,
        "p_undecided": p_u,
    }


def metrics_from(steps):
    if not steps: return {}
    tbsr = np.mean([r["correct"] for r in steps])
    wr = np.mean([r["warned"] for r in steps])
    nec = [r for r in steps if r["need_warn"]]
    unnec = [r for r in steps if not r["need_warn"]]
    wr_n = np.mean([r["warned"] for r in nec]) if nec else 0
    wr_u = np.mean([r["warned"] for r in unnec]) if unnec else 0
    sg = wr_n - wr_u
    sd = np.mean([r["self_disc"] for r in steps])
    fb_rate = np.mean([r["fallback"] for r in steps])
    leak = np.mean([r["leakage"] for r in steps if r["warned"]]) if any(r["warned"] for r in steps) else 0
    return {"tbsr": round(tbsr,4), "wr": round(wr,4), "sel_gap": round(sg,4),
            "sd": round(sd,4), "fallback_rate": round(fb_rate,4),
            "leakage": round(leak,4)}


def run_family_audit(L):
    """Part A: Held-out family audit."""
    L.append("## A. Held-Out Family Audit\n\n")
    L.append("| Lesson | n | TBSR | WR | SelGap | SD | Fallback |\n")
    L.append("|:-------|:-:|:----:|:--:|:------:|:--:|:--------:|\n")

    for name, lesson in ALL_LESSONS:
        all_steps = []
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                rng = np.random.default_rng(sid * 10000)
                m = FactoredInternalizationState(); m.snapshot()
                scorer = MicroBayesShadowV2_1(agent_params=AP)
                for step_i in range(N_STEPS):
                    r = run_single(m, scorer, th, lesson, step_i + sid*100,
                                    sid, rng)
                    all_steps.append(r)
        met = metrics_from(all_steps)
        L.append(f"| {name} | {len(all_steps)} | {met['tbsr']:.3f} | "
                 f"{met['wr']:.3f} | {met['sel_gap']:.3f} | "
                 f"{met['sd']:.3f} | {met['fallback_rate']:.3f} |\n")
        print(f"  family: {name} done", file=sys.stderr)

    L.append("\n")


def run_sensitivity_audit(L):
    """Part B: Parameter sensitivity / plateau audit."""
    L.append("## B. Parameter Sensitivity (Plateau Audit)\n\n")

    default_params = {
        "beta_task": 1.5, "beta_learn": 2.5, "beta_dep": 2.0, "beta_cost": 1.5,
        "delta_threshold": 0.5, "tau_necessity": 0.2,
    }
    sweep_ranges = {
        "beta_task":       [0.5, 1.0, 1.5, 2.0, 2.5],
        "beta_learn":      [1.0, 1.5, 2.5, 3.5, 4.0],
        "beta_dep":        [0.5, 1.0, 2.0, 3.0, 4.0],
        "beta_cost":       [0.5, 1.0, 1.5, 2.0, 3.0],
        "delta_threshold": [0.0, 0.25, 0.5, 0.75, 1.0],
        "tau_necessity":   [0.0, 0.1, 0.2, 0.4, 0.6],
    }

    target_lessons = []
    for nm in ["tic_rescue_heavy", "tic_self_discovery", "false_suppression",
               "beneficial_novelty", "blind_activation_corridor", "ppmrb_standard"]:
        if nm in LESSON_V2_BY_NAME:
            target_lessons.append(LESSON_V2_BY_NAME[nm])

    L.append("| Param | Value | TBSR | WR | SelGap | SD |\n")
    L.append("|:------|:-----:|:----:|:--:|:------:|:--:|\n")

    for param, values in sweep_ranges.items():
        for val in values:
            kwargs = dict(default_params)
            kwargs[param] = val
            # credit lambda_cc sweep
            cc = CreditCorrection()
            scorer = MicroBayesShadowV2_1(agent_params=AP, credit=cc, **kwargs)

            all_steps = []
            for th in ["safe", "shiny"]:
                for sid in range(3):
                    rng = np.random.default_rng(sid * 10000)
                    m = FactoredInternalizationState(); m.snapshot()
                    for step_i in range(15):
                        les = target_lessons[step_i % len(target_lessons)]
                        r = run_single(m, scorer, th, les, step_i+sid*100, sid, rng)
                        all_steps.append(r)
            met = metrics_from(all_steps)
            marker = " ✦" if val == default_params.get(param) else ""
            L.append(f"| {param} | {val}{marker} | {met['tbsr']:.3f} | "
                     f"{met['wr']:.3f} | {met['sel_gap']:.3f} | {met['sd']:.3f} |\n")
        L.append(f"| — | — | — | — | — | — |\n")
        print(f"  sensitivity: {param} done", file=sys.stderr)

    # λ_cc sweep
    L.append("| λ_cc | Value | TBSR | WR | SelGap | SD |\n")
    L.append("|:-----|:-----:|:----:|:--:|:------:|:--:|\n")
    for lcc in [0.0, 0.3, 0.6, 0.9, 1.0]:
        cc = CreditCorrection(lambda_credit=lcc)
        scorer = MicroBayesShadowV2_1(agent_params=AP, credit=cc)
        all_steps = []
        for th in ["safe", "shiny"]:
            for sid in range(3):
                rng = np.random.default_rng(sid * 10000)
                m = FactoredInternalizationState(); m.snapshot()
                for step_i in range(15):
                    les = target_lessons[step_i % len(target_lessons)]
                    r = run_single(m, scorer, th, les, step_i+sid*100, sid, rng)
                    all_steps.append(r)
        met = metrics_from(all_steps)
        marker = " ✦" if lcc == 0.6 else ""
        L.append(f"| λ_cc | {lcc}{marker} | {met['tbsr']:.3f} | "
                 f"{met['wr']:.3f} | {met['sel_gap']:.3f} | {met['sd']:.3f} |\n")
    print(f"  sensitivity: lambda_cc done", file=sys.stderr)
    L.append("\n")


def run_ternary_closure(L):
    """Part C: Ternary vs binary final confirmation."""
    L.append("## C. Ternary vs Binary Closure\n\n")

    target_lessons = []
    for nm in ["tic_rescue_heavy", "tic_self_discovery", "false_suppression",
               "beneficial_novelty", "ppmrb_standard", "soft_boundary_tradeoff",
               "blind_activation_corridor", "verified_warn"]:
        if nm in LESSON_V2_BY_NAME:
            target_lessons.append(LESSON_V2_BY_NAME[nm])

    for mode_name, force_bin in [("ternary (default)", False), ("binary rollback", True)]:
        all_steps = []
        scorer = MicroBayesShadowV2_1(agent_params=AP)
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                rng = np.random.default_rng(sid * 10000)
                m = FactoredInternalizationState(); m.snapshot()
                for step_i in range(N_STEPS):
                    les = target_lessons[step_i % len(target_lessons)]
                    r = run_single(m, scorer, th, les, step_i+sid*100, sid, rng,
                                    force_binary=force_bin)
                    all_steps.append(r)

        met = metrics_from(all_steps)
        L.append(f"### {mode_name}\n\n")
        L.append(f"TBSR={met['tbsr']:.4f}, WR={met['wr']:.4f}, SelGap={met['sel_gap']:.4f}, "
                 f"SD={met['sd']:.4f}\n\n")

        # Per-subtype
        L.append("| Subtype | n | WR | SD |\n")
        L.append("|:--------|:-:|:--:|:--:|\n")
        for st in ["self_discovery_teach", "self_discovery_needed", "boundary_obs",
                    "warn_rescue", "false_suppression_cost", "beneficial_novelty",
                    "blind_corridor", "soft_gradual", "verified_warn"]:
            ss = [r for r in all_steps if r["subtype"] == st]
            if ss:
                L.append(f"| {st} | {len(ss)} | {np.mean([r['warned'] for r in ss]):.3f} | "
                         f"{np.mean([r['self_disc'] for r in ss]):.3f} |\n")
        L.append("\n")
        print(f"  ternary closure: {mode_name} done", file=sys.stderr)


def run_parity_audit(L):
    """Part D+E: Fallback rate + mirror parity."""
    L.append("## D. Fallback Rate by Variant\n\n")

    target_lessons = []
    for nm in ["tic_rescue_heavy", "tic_self_discovery", "false_suppression",
               "beneficial_novelty", "ppmrb_standard"]:
        if nm in LESSON_V2_BY_NAME:
            target_lessons.append(LESSON_V2_BY_NAME[nm])

    for mode in [PSelfMode.POSTERIOR_C, PSelfMode.POSTERIOR_A, PSelfMode.POSTERIOR_B]:
        fb_count = 0
        total = 0
        for th in ["safe", "shiny"]:
            for sid in range(4):
                rng = np.random.default_rng(sid * 10000)
                m = FactoredInternalizationState(); m.snapshot()
                for step_i in range(15):
                    les = target_lessons[step_i % len(target_lessons)]
                    ub = {p: 0.5 for p in PROBE_NAMES}
                    et = generate_episode_from_lesson_v2(les, step_i+sid*100, th, ub, rng)
                    ep, spec, gm, cfg_e, meta, sc = et
                    dc = getattr(sc, 'commit_depth', 3)
                    dr = getattr(sc, 'reveal_depth', 2)
                    ps_r = compute_p_self_posterior(
                        mode, dc, dr,
                        tau_hat=m.tau, nu_hat=m.nu, gamma_gen_hat=m.gamma_gen)
                    if ps_r.get("fallback", False):
                        fb_count += 1
                    total += 1
        rate = fb_count / max(total, 1)
        L.append(f"- **{mode.value}**: fallback rate = {rate:.4f} ({fb_count}/{total})\n")
    L.append("\n")

    # Mirror parity
    L.append("## E. Mirror / Side-Swap Parity\n\n")
    L.append("| Theta | WR | SelGap | SD |\n")
    L.append("|:------|:--:|:------:|:--:|\n")
    for th in ["safe", "shiny"]:
        all_steps = []
        scorer = MicroBayesShadowV2_1(agent_params=AP)
        for sid in range(N_SEEDS):
            rng = np.random.default_rng(sid * 10000)
            m = FactoredInternalizationState(); m.snapshot()
            for step_i in range(N_STEPS):
                les = target_lessons[step_i % len(target_lessons)]
                r = run_single(m, scorer, th, les, step_i+sid*100, sid, rng)
                all_steps.append(r)
        met = metrics_from(all_steps)
        L.append(f"| {th} | {met['wr']:.4f} | {met['sel_gap']:.4f} | {met['sd']:.4f} |\n")
    print(f"  parity done", file=sys.stderr)
    L.append("\n")


def main():
    print("═══ Step 5: Contract Closure Audit ═══\n", file=sys.stderr)
    L = ["# Step 5: Contract Closure + Held-out Promotion Audit\n\n"]

    run_family_audit(L)
    run_sensitivity_audit(L)
    run_ternary_closure(L)
    run_parity_audit(L)

    rpt = out / "step5_contract_closure_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
