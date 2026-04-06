"""P5-B: 5D No-Score Integration Eval.

Confirms 5D state-estimator (τ̂,ν̂,γ̂_gen,γ̂_spec_state,κ̂) does not
break 3D micro view or macro stability.

Exp 1: 5D micro no-score (3D view unchanged)
Exp 2: 5D macro no-score  
Exp 3: κ observability replication (signal, orthogonality, stability)
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
from src.curriculum.lesson_library_v2 import (
    LESSON_CATALOG_V2, BALANCED_ACTIVE_LESSONS, PROBE_NAMES,
)
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


def run_5d_session(lessons, theta, seed, n_teach=20, hidden_tempt=0.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    for step in range(n_teach):
        les = lessons[step % len(lessons)]
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

        # Oracle
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, _ = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer-only: micro view = 3D (κ NOT used in Q)
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]
        m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, _ = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

        action, dose = act_o, dose_o
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        eff_lure = tempt + hidden_tempt
        risky_branch = 1 - sc.oracle_safe_branch_id
        tempt_scores = [0.0, 0.0]; tempt_scores[risky_branch] = hidden_tempt
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

        # Risk prediction for κ (agent's cost/risk model)
        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat,
            lure=eff_lure, agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)
        est5 = observer.get_estimate()

        diverge = (act_o != act_i)
        active = (act_o != "WAIT") or (act_i != "WAIT")

        records.append({
            "step": step, "theta": theta,
            "act_oracle": act_o, "act_infer": act_i,
            "diverge": diverge, "active": active,
            "correct": correct,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "est5": est5,
            "risk_err": abs(risk - risk_hat),
        })
    return records


def main():
    print("═══ P5-B: 5D No-Score Integration ═══\n", file=sys.stderr)
    L = ["# P5-B: 5D No-Score Integration Evaluation\n\n"]
    L.append("**Observer: 5D (τ̂,ν̂,γ̂_gen,γ̂_spec_state,κ̂) | Micro: 3D view | "
             "Tutor: 2-act**\n\n")

    # Exp 1: Micro stability
    L.append("## Exp 1: 5D Micro No-Score Stability\n\n")
    print("Exp 1: Micro...", file=sys.stderr)
    L.append("| Suite | θ | DivAll | Div@Act | n_act | Success |\n")
    L.append("|-------|:-:|:------:|:-------:|:-----:|:-------:|\n")
    for sname, lessons in [("Canonical", ALL_LESSONS),
                            ("Active", BALANCED_ACTIVE_LESSONS)]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_5d_session(lessons, th, sid))
            n = len(recs)
            da = sum(r["diverge"] for r in recs) / n
            na = sum(1 for r in recs if r["active"])
            dact = sum(1 for r in recs if r["diverge"] and r["active"]) / max(na, 1)
            succ = sum(r["correct"] for r in recs) / n
            L.append("| {} | {} | {:.4f} | {:.4f} | {} | {:.3f} |\n".format(
                sname, th, da, dact, na, succ))

    # Exp 2: Macro stability
    L.append("\n## Exp 2: 5D Macro No-Score Stability\n\n")
    print("\nExp 2: Macro...", file=sys.stderr)
    L.append("| θ | STOP Agree | Top-1 | Kendall τ | κ̂(final) |\n")
    L.append("|:-:|:----------:|:-----:|:---------:|:---------:|\n")
    CAT = list(LESSON_CATALOG_V2)
    for th in ["safe", "shiny"]:
        stop_ok = 0; n = 0
        kappas = []
        for sid in range(NS):
            recs = run_5d_session(ALL_LESSONS, th, sid)
            rl = recs[-1]; mt = rl["m_true"]; mh = rl["est5"]
            so = EPS_0 + A_S * mt["nu"] + B_S * mt["gamma_gen"]
            si = EPS_0 + A_S * mh["nu"] + B_S * mh["gamma_gen"]
            n += 1
            if (so > STOP_THRESH) == (si > STOP_THRESH): stop_ok += 1
            kappas.append(mh["kappa"])
        # Kendall
        recs_last = run_5d_session(ALL_LESSONS, th, 0)[-1]
        mt = recs_last["m_true"]; mh = recs_last["est5"]
        so = [np.mean(l.gain)*(1-mt["nu"])*(1-mt["gamma_gen"])*mt["tau"] for l in CAT]
        sh = [np.mean(l.gain)*(1-mh["nu"])*(1-mh["gamma_gen"])*mh["tau"] for l in CAT]
        top1 = 1.0 if np.argsort(so)[-1] == np.argsort(sh)[-1] else 0.0
        kt, _ = sp_stats.kendalltau(so, sh)
        L.append("| {} | {:.1f}% | {:.0f} | {:.4f} | {:.4f} |\n".format(
            th, 100*stop_ok/n, top1, kt, np.mean(kappas)))

    # Exp 3: κ replication
    L.append("\n## Exp 3: κ Observability Replication\n\n")
    print("\nExp 3: κ replication...", file=sys.stderr)
    L.append("### Signal & Orthogonality\n\n")
    L.append("| θ | κ̂(final) | σ(κ̂) | Corr(κ̂,τ̂) | Corr(κ̂,ν̂) | "
             "Corr(κ̂,γ̂_gen) | Corr(κ̂,γ̂_spec) |\n")
    L.append("|:-:|:--------:|:----:|:----------:|:----------:|"
             ":--------------:|:---------------:|\n")
    for th in ["safe", "shiny"]:
        kaps = []; taus = []; nus = []; ggs = []; gss = []
        for sid in range(NS):
            recs = run_5d_session(ALL_LESSONS, th, sid)
            for r in recs:
                e = r["est5"]
                kaps.append(e["kappa"]); taus.append(e["tau"])
                nus.append(e["nu"]); ggs.append(e["gamma_gen"])
                gss.append(e["gamma_spec"])
        ck_t = sp_stats.pearsonr(kaps, taus)[0] if len(set(taus)) > 1 else 0
        ck_n = sp_stats.pearsonr(kaps, nus)[0] if len(set(nus)) > 1 else 0
        ck_g = sp_stats.pearsonr(kaps, ggs)[0] if len(set(ggs)) > 1 else 0
        ck_s = sp_stats.pearsonr(kaps, gss)[0] if len(set(gss)) > 1 else 0
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, np.mean(kaps), np.std(kaps), ck_t, ck_n, ck_g, ck_s))

    # Verdict
    L.append("\n## Verdict\n\n")
    L.append("> 5D no-score integration complete. κ̂ is in Layer 1; "
             "micro uses 3D view; macro reports full 5D.\n")

    rpt = out / "p5b_5d_no_score_integration.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
