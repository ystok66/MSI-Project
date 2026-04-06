"""P4-C.3: 3D vs 4D Macro Utility + P5-prep: κ Observability Audit.

Part 1: Does γ_spec_state add macro value? (STOP/TEACH on temptation suite)
Part 2: κ signal feasibility (risk error accumulation, collinearity)
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
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(theta, seed, hidden_tempt=0.0, n_teach=20):
    """Full session returning true m, 3D hat, 4D hat, risk errors."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP, use_dose=False)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    risk_errors = []

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

        # Risk error for κ probe
        risk_hat = lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4))
        risk_err = abs(risk_hat - risk)
        risk_errors.append(risk_err)

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
        est = observer.get_estimate()

        records.append({
            "step": step, "theta": theta,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat_3d": {"tau": est["tau"], "nu": est["nu"],
                          "gamma_gen": est["gamma_gen"]},
            "m_hat_4d": est,
            "gamma_spec_hat": est["gamma_spec"],
            "risk_err": risk_err,
            "risk_true": risk,
            "correct": correct,
        })
    return records, risk_errors


def macro_scores(m_dict, cat):
    """Compute lesson scores for ranking."""
    tau = m_dict.get("tau", 0.3)
    nu = m_dict.get("nu", 0.1)
    gg = m_dict.get("gamma_gen", 0.0)
    gs = m_dict.get("gamma_spec", 0.0)
    scores = []
    for l in cat:
        g = np.mean(l.gain)
        s = g * (1 - nu) * (1 - gg) * tau
        # 4D bonus: if gamma_spec available, boost temptation-related lessons
        if "gamma_spec" in m_dict and hasattr(l, 'dose_profile'):
            dp = l.dose_profile
            if dp in ("high", "aggressive"):
                s *= (1 + 0.1 * (1 - gs))  # more tempt-vulnerable → prioritize
        scores.append(s)
    return scores


def stop_score(m_dict):
    nu = m_dict.get("nu", 0.1)
    gg = m_dict.get("gamma_gen", 0.0)
    return EPS_0 + A_S * nu + B_S * gg


def main():
    print("═══ P4-C.3: Macro Utility + κ Audit ═══\n", file=sys.stderr)
    L = ["# P4-C.3: 3D vs 4D Macro Utility + κ Observability\n\n"]

    # ═══ Part 1: 3D vs 4D Macro ══════════════════════════
    L.append("## Part 1: 3D vs 4D Macro Utility\n\n")
    print("Part 1: Macro 3D vs 4D...", file=sys.stderr)

    L.append("### STOP Agreement\n\n")
    L.append("| θ | tempt | STOP(3D) | STOP(4D) | STOP(oracle) | "
             "Agree(3D) | Agree(4D) |\n")
    L.append("|:-:|:-----:|:--------:|:--------:|:------------:|"
             ":---------:|:---------:|\n")

    for th in ["safe", "shiny"]:
        for ht_label, ht in [("none", 0.0), ("al=0.6", 0.6), ("cf=1.0", 1.0)]:
            agree_3d = 0; agree_4d = 0; n = 0
            for sid in range(NS):
                recs, _ = run_session(th, sid, hidden_tempt=ht)
                rl = recs[-1]
                s_o = stop_score(rl["m_true"])
                s_3d = stop_score(rl["m_hat_3d"])
                s_4d = stop_score(rl["m_hat_4d"])
                n += 1
                if (s_o > STOP_THRESH) == (s_3d > STOP_THRESH): agree_3d += 1
                if (s_o > STOP_THRESH) == (s_4d > STOP_THRESH): agree_4d += 1
            L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f}% | {:.1f}% |\n".format(
                th, ht_label,
                stop_score(recs[-1]["m_hat_3d"]),
                stop_score(recs[-1]["m_hat_4d"]),
                stop_score(recs[-1]["m_true"]),
                100*agree_3d/n, 100*agree_4d/n))

    L.append("\n### TEACH Ranking (Kendall τ + Top-1)\n\n")
    L.append("| θ | tempt | Kendall(3D) | Kendall(4D) | Top1(3D) | Top1(4D) |\n")
    L.append("|:-:|:-----:|:-----------:|:-----------:|:--------:|:--------:|\n")
    CAT = list(LESSON_CATALOG_V2)
    for th in ["safe", "shiny"]:
        for ht_label, ht in [("none", 0.0), ("al=0.6", 0.6), ("cf=1.0", 1.0)]:
            kt3_all = []; kt4_all = []; t1_3 = []; t1_4 = []
            for sid in range(NS):
                recs, _ = run_session(th, sid, hidden_tempt=ht)
                rl = recs[-1]
                so = macro_scores(rl["m_true"], CAT)
                s3 = macro_scores(rl["m_hat_3d"], CAT)
                s4 = macro_scores(rl["m_hat_4d"], CAT)
                kt3, _ = sp_stats.kendalltau(so, s3)
                kt4, _ = sp_stats.kendalltau(so, s4)
                kt3_all.append(kt3); kt4_all.append(kt4)
                t1_3.append(1.0 if np.argsort(so)[-1] == np.argsort(s3)[-1] else 0.0)
                t1_4.append(1.0 if np.argsort(so)[-1] == np.argsort(s4)[-1] else 0.0)
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.0f}% | {:.0f}% |\n".format(
                th, ht_label,
                np.mean(kt3_all), np.mean(kt4_all),
                100*np.mean(t1_3), 100*np.mean(t1_4)))

    # ═══ Part 2: κ Observability ═════════════════════════
    L.append("\n## Part 2: κ Observability Audit\n\n")
    print("\nPart 2: κ observability...", file=sys.stderr)

    L.append("### Risk Error Signal\n\n")
    L.append("| θ | Mean |e_risk| | Std | Range | Non-zero% |\n")
    L.append("|:-:|:-------------:|:---:|:-----:|:---------:|\n")
    all_re = {}
    for th in ["safe", "shiny"]:
        errs = []
        for sid in range(NS):
            _, risk_errs = run_session(th, sid)
            errs.extend(risk_errs)
        all_re[th] = errs
        L.append("| {} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] | {:.1f}% |\n".format(
            th, np.mean(errs), np.std(errs), np.min(errs), np.max(errs),
            100*sum(1 for e in errs if e > 0.01)/len(errs)))

    # Collinearity with existing dims
    L.append("\n### Collinearity: κ-proxy vs Existing Dims\n\n")
    L.append("| θ | Corr(e_risk, τ̂) | Corr(e_risk, ν̂) | "
             "Corr(e_risk, γ̂_gen) | Corr(e_risk, γ̂_spec) |\n")
    L.append("|:-:|:---------------:|:---------------:|"
             ":-------------------:|:--------------------:|\n")
    for th in ["safe", "shiny"]:
        taus = []; nus = []; ggs = []; gss = []; errs = []
        for sid in range(NS):
            recs, risk_errs = run_session(th, sid)
            for r, e in zip(recs, risk_errs):
                taus.append(r["m_hat_4d"]["tau"])
                nus.append(r["m_hat_4d"]["nu"])
                ggs.append(r["m_hat_4d"]["gamma_gen"])
                gss.append(r["m_hat_4d"]["gamma_spec"])
                errs.append(e)
        ct = sp_stats.pearsonr(errs, taus)[0] if len(set(taus)) > 1 else 0
        cn = sp_stats.pearsonr(errs, nus)[0] if len(set(nus)) > 1 else 0
        cg = sp_stats.pearsonr(errs, ggs)[0] if len(set(ggs)) > 1 else 0
        cs = sp_stats.pearsonr(errs, gss)[0] if len(set(gss)) > 1 else 0
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, ct, cn, cg, cs))

    # κ accumulation simulation
    L.append("\n### Simulated κ Trajectory (EMA of 1−e_risk)\n\n")
    L.append("| θ | κ(final) | κ stability (last 5 std) |\n")
    L.append("|:-:|:--------:|:-----------------------:|\n")
    for th in ["safe", "shiny"]:
        kappas_final = []; kappas_stab = []
        for sid in range(NS):
            _, risk_errs = run_session(th, sid)
            kappa = 0.3  # init
            kappa_trace = []
            for e in risk_errs:
                kappa = np.clip(0.95 * kappa + 0.03 * (1 - e) - 0.02 * e, 0, 1)
                kappa_trace.append(kappa)
            kappas_final.append(kappa_trace[-1])
            kappas_stab.append(np.std(kappa_trace[-5:]))
        L.append("| {} | {:.4f} | {:.4f} |\n".format(
            th, np.mean(kappas_final), np.mean(kappas_stab)))

    # ═══ Verdict ═════════════════════════════════════════
    L.append("\n## Verdict\n\n")
    L.append("### Macro Utility\n\n")
    L.append("> See STOP agreement and Kendall τ above for 3D vs 4D comparison.\n\n")
    L.append("### κ Observability\n\n")
    mean_re = np.mean(all_re["safe"] + all_re["shiny"])
    if mean_re > 0.05:
        L.append(f"> **κ signal present.** Mean |e_risk| = {mean_re:.4f}. "
                 "Risk error is detectable and could support a 5th dimension.\n")
    else:
        L.append(f"> **κ signal weak.** Mean |e_risk| = {mean_re:.4f}. "
                 "May need specialized families to generate sufficient signal.\n")

    rpt = out / "p4c3_macro_utility_kappa_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
