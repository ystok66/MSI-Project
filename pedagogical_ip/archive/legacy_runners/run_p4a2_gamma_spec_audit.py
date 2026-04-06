"""P4-A.2: γ_spec Semantic Audit — trait vs state.

Audit 1 (State): Does γ̂_spec correlate with conditional resist rate?
Audit 2 (Trait): Under fixed tempt/risk, does γ̂_spec track γ_spec(true)?
Verdict: Which semantic fits better?
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from scipy import stats as sp_stats

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session_tracking(theta, seed, hidden_tempt=0.0, n_teach=20,
                         gamma_spec_init=None):
    """Full session returning per-step γ_spec + resist/follow events."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    if gamma_spec_init is not None:
        m.gamma_spec = gamma_spec_init
    tutor = BCICTv4(agent_params=AP, use_dose=False)  # 2-act canonical
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    tempt_events = 0; resist_events = 0

    for step in range(n_teach):
        les = ALL_LESSONS[step % len(ALL_LESSONS)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed*100, theta, ub, rng)
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
        lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe", ss); lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)
        action, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        eff_lure = tempt + hidden_tempt
        risky_branch = 1 - sc.oracle_safe_branch_id
        tempt_scores = [0.0, 0.0]
        tempt_scores[risky_branch] = hidden_tempt
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b) + tempt_scores[0])
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a) + tempt_scores[1])
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0
        self_disc = correct and not warned and p_self > 0.5

        # Track temptation events
        is_tempt = eff_lure >= 0.3
        if is_tempt:
            tempt_events += 1
            if correct: resist_events += 1

        # Update true m
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if p_self < 0.5: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            lure=eff_lure, agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen,
                    "gamma_spec": m.gamma_spec},
        )
        observer.update(ev)

        # Conditional resist rate so far
        r_resist = resist_events / max(tempt_events, 1)

        records.append({
            "step": step, "theta": theta, "hidden_tempt": hidden_tempt,
            "correct": correct, "lure": eff_lure,
            "gamma_spec_hat": observer.gamma_spec_hat,
            "gamma_spec_true": m.gamma_spec,
            "r_resist": r_resist,
            "nu_hat": observer.nu_hat,
            "nu_true": m.nu,
            "tempt_events": tempt_events,
            "resist_events": resist_events,
        })
    return records


def main():
    print("═══ P4-A.2: γ_spec Semantic Audit ═══\n", file=sys.stderr)
    L = ["# P4-A.2: γ_spec Semantic Audit\n\n"]

    # ─── Audit 1: State — γ̂_spec vs conditional resist rate ─
    L.append("## Audit 1: State Semantic — γ̂_spec vs Resist Rate\n\n")
    print("Audit 1: State...", file=sys.stderr)
    L.append("| θ | tempt | γ̂_spec(final) | r_resist | "
             "Corr(γ̂_spec, r_resist) | ν̂(final) | Δν̂ |\n")
    L.append("|:-:|:-----:|:-------------:|:--------:|"
             ":---------------------:|:--------:|:---:|\n")

    corr_by_theta = {}
    for th in ["safe", "shiny"]:
        gs_all = []; rr_all = []; nu_base = None
        for ht_label, ht in [("0.0", 0.0), ("0.3", 0.3), ("0.6", 0.6), ("1.0", 1.0)]:
            gs_finals = []; rr_finals = []; nu_finals = []
            for sid in range(NS):
                recs = run_session_tracking(th, sid, hidden_tempt=ht)
                gs_finals.append(recs[-1]["gamma_spec_hat"])
                rr_finals.append(recs[-1]["r_resist"])
                nu_finals.append(recs[-1]["nu_hat"])
                gs_all.append(recs[-1]["gamma_spec_hat"])
                rr_all.append(recs[-1]["r_resist"])
            nu_mean = np.mean(nu_finals)
            if nu_base is None: nu_base = nu_mean
            delta_nu = nu_mean - nu_base
            # Per-tempt correlation
            if len(set(gs_finals)) > 1 and len(set(rr_finals)) > 1:
                c, _ = sp_stats.pearsonr(gs_finals, rr_finals)
            else:
                c = float('nan')
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                th, ht_label, np.mean(gs_finals), np.mean(rr_finals),
                c, nu_mean, delta_nu))
        # Overall correlation across tempt levels
        if len(gs_all) > 3:
            c_all, p_all = sp_stats.pearsonr(gs_all, rr_all)
            corr_by_theta[th] = (c_all, p_all)
            L.append(f"\n**θ={th} overall: Corr(γ̂_spec, r_resist) = {c_all:.4f} "
                     f"(p={p_all:.4f})**\n\n")

    # ─── Audit 2: Trait — fixed tempt, sweep γ_spec(true) ──
    L.append("## Audit 2: Trait Semantic — Fixed Tempt, Sweep γ_spec_init\n\n")
    print("\nAudit 2: Trait...", file=sys.stderr)
    L.append("| θ | tempt | γ_spec_init | γ̂_spec(final) | γ_spec(true,final) | "
             "r_resist |\n")
    L.append("|:-:|:-----:|:-----------:|:-------------:|:------------------:|"
             ":--------:|\n")

    trait_corrs = {}
    for th in ["safe"]:
        for ht in [0.3, 0.6]:
            gs_hats = []; gs_trues = []; gs_inits = []
            for gs_init in [0.1, 0.3, 0.5, 0.7]:
                finals_hat = []; finals_true = []; finals_rr = []
                for sid in range(NS):
                    recs = run_session_tracking(th, sid, hidden_tempt=ht,
                                                gamma_spec_init=gs_init)
                    finals_hat.append(recs[-1]["gamma_spec_hat"])
                    finals_true.append(recs[-1]["gamma_spec_true"])
                    finals_rr.append(recs[-1]["r_resist"])
                L.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                    th, ht, gs_init,
                    np.mean(finals_hat), np.mean(finals_true), np.mean(finals_rr)))
                gs_hats.extend(finals_hat)
                gs_trues.extend(finals_true)
                gs_inits.extend([gs_init] * NS)
            if len(gs_hats) > 3:
                c_trait, p_trait = sp_stats.pearsonr(gs_hats, gs_trues)
                trait_corrs[(th, ht)] = (c_trait, p_trait)
                L.append(f"\n**θ={th}, tempt={ht}: "
                         f"Corr(γ̂_spec, γ_spec_true) = {c_trait:.4f} (p={p_trait:.4f})**\n\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")

    state_ok = all(c > 0.3 for c, _ in corr_by_theta.values() if not np.isnan(c))
    trait_ok = all(c > 0.3 for c, _ in trait_corrs.values() if not np.isnan(c))

    if state_ok and not trait_ok:
        L.append("> **γ̂_spec is best interpreted as a BEHAVIORAL STATE** "
                 "(correlated with conditional resist rate) rather than a "
                 "trait-like latent (weakly correlated with true γ_spec).\n")
        L.append("\n**Semantic label: `gamma_spec_state`**\n")
    elif trait_ok and not state_ok:
        L.append("> **γ̂_spec is best interpreted as a TRAIT LATENT** "
                 "(tracks true γ_spec under fixed conditions).\n")
        L.append("\n**Semantic label: `gamma_spec_trait`**\n")
    elif state_ok and trait_ok:
        L.append("> **γ̂_spec has DUAL semantics**: correlates with both "
                 "resist rate (state) and true γ_spec (trait). "
                 "Recommend defaulting to STATE interpretation.\n")
        L.append("\n**Semantic label: `gamma_spec_state` (primary)**\n")
    else:
        state_vals = [f"{c:.3f}" for c, _ in corr_by_theta.values()]
        trait_vals = [f"{c:.3f}" for c, _ in trait_corrs.values()]
        L.append(f"> **γ̂_spec semantics UNCLEAR.** "
                 f"State corrs: {state_vals}, Trait corrs: {trait_vals}. "
                 "Needs further investigation.\n")

    rpt = out / "p4a2_gamma_spec_semantic_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
