"""P4-B: 4D Observer Formal Evaluation.

Exp 1: 4D micro infer-only (canonical, active, temptation)
Exp 2: 4D macro hybrid (α=0.5, α=1.0)  
Exp 3: ν contamination (4D version)
Exp 4: State semantics replication
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


def run_4d_session(lessons, theta, seed, n_teach=20, hidden_tempt=0.0):
    """Full session with 4D observer + 2-act canonical tutor."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP, use_dose=False)  # 2-act canonical
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    tempt_events = 0; resist_events = 0

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

        # Oracle decision (using true m)
        tutor_oracle = BCICTv4(agent_params=AP, use_dose=False)
        action_oracle, dose_oracle, _ = tutor_oracle.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer-only decision (using m̂ from 4D observer)
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.gamma_spec = est["gamma_spec"]
        m_hat.snapshot()
        tutor_infer = BCICTv4(agent_params=AP, use_dose=False)
        action_infer, dose_infer, _ = tutor_infer.decide(sc, fb, lp, lib, scr, 2, m_hat)

        # Actual decision uses oracle
        action, dose = action_oracle, dose_oracle
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

        is_tempt = eff_lure >= 0.3
        if is_tempt:
            tempt_events += 1
            if correct: resist_events += 1

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
        r_resist = resist_events / max(tempt_events, 1)

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "correct": correct, "active": dose > 0,
            "action_oracle": action_oracle, "action_infer": action_infer,
            "diverge": action_oracle != action_infer,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            "gamma_spec_hat": observer.gamma_spec_hat,
            "r_resist": r_resist,
            "hidden_tempt": hidden_tempt,
        })
    return records


def main():
    print("═══ P4-B: 4D Observer Formal Evaluation ═══\n", file=sys.stderr)
    L = ["# P4-B: 4D Observer Formal Evaluation\n\n"]
    L.append("**Observer: 4D (τ̂, ν̂, γ̂_gen, γ̂_spec_state) | Tutor: 2-act canonical**\n\n")

    # ─── Exp 1: Micro Infer-Only ─────────────────────────
    L.append("## Exp 1: 4D Micro Infer-Only\n\n")
    print("Exp 1: Micro infer-only...", file=sys.stderr)
    suites = [
        ("Canonical", list(LESSON_CATALOG_V2), [("none", 0.0)]),
        ("Balanced Active", BALANCED_ACTIVE_LESSONS, [("none", 0.0)]),
        ("Temptation", list(LESSON_CATALOG_V2),
         [("none", 0.0), ("aligned=0.6", 0.6), ("conflict=1.0", 1.0)]),
    ]
    L.append("| Suite | θ | tempt | Div All | Div@Active | n_active | Success |\n")
    L.append("|-------|:-:|:-----:|:-------:|:----------:|:--------:|:-------:|\n")
    for suite_name, lessons, tempts in suites:
        for th in ["safe", "shiny"]:
            for ht_label, ht in tempts:
                recs = []
                for sid in range(NS):
                    recs.extend(run_4d_session(lessons, th, sid, hidden_tempt=ht))
                n = len(recs)
                div_all = sum(r["diverge"] for r in recs) / n
                n_act = sum(1 for r in recs if r["active"])
                div_act = (sum(1 for r in recs if r["diverge"] and r["active"])
                           / max(n_act, 1))
                succ = sum(r["correct"] for r in recs) / n
                L.append("| {} | {} | {} | {:.4f} | {:.4f} | {} | {:.3f} |\n".format(
                    suite_name, th, ht_label, div_all, div_act, n_act, succ))

    # ─── Exp 2: Macro Hybrid ─────────────────────────────
    L.append("\n## Exp 2: 4D Macro Hybrid\n\n")
    print("\nExp 2: Macro hybrid...", file=sys.stderr)
    L.append("| θ | α | STOP Agree | Top-1 | Kendall τ |\n")
    L.append("|:-:|:-:|:----------:|:-----:|:---------:|\n")
    for th in ["safe", "shiny"]:
        for alpha in [0.5, 1.0]:
            recs = []
            for sid in range(NS):
                recs.extend(run_4d_session(ALL_LESSONS, th, sid))
            stop_agree = 0; n = len(recs)
            for r in recs:
                mt = r["m_true"]; mh = r["m_hat"]
                eo = EPS_0 + A_S * mt["nu"] + B_S * mt["gamma_gen"]
                # Hybrid: blend oracle and observer
                nu_h = (1-alpha)*mt["nu"] + alpha*mh["nu"]
                gg_h = (1-alpha)*mt["gamma_gen"] + alpha*mh["gamma_gen"]
                ei = EPS_0 + A_S * nu_h + B_S * gg_h
                if (eo > STOP_THRESH) == (ei > STOP_THRESH): stop_agree += 1
            # Top-1 + Kendall on last step
            from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2 as CAT
            rl = recs[-1]; mt = rl["m_true"]; mh = rl["m_hat"]
            so = [np.mean(l.gain)*(1-mt["nu"])*(1-mt["gamma_gen"])*mt["tau"] for l in CAT]
            nu_h = (1-alpha)*mt["nu"]+alpha*mh["nu"]
            gg_h = (1-alpha)*mt["gamma_gen"]+alpha*mh["gamma_gen"]
            tau_h = (1-alpha)*mt["tau"]+alpha*mh["tau"]
            sh = [np.mean(l.gain)*(1-nu_h)*(1-gg_h)*tau_h for l in CAT]
            top1 = 1.0 if np.argsort(so)[-1] == np.argsort(sh)[-1] else 0.0
            kt, _ = sp_stats.kendalltau(so, sh)
            L.append("| {} | {} | {:.3f} | {:.0f} | {:.4f} |\n".format(
                th, alpha, stop_agree/n, top1, kt))

    # ─── Exp 3: ν Contamination ──────────────────────────
    L.append("\n## Exp 3: ν Contamination (4D)\n\n")
    print("\nExp 3: ν contamination...", file=sys.stderr)
    L.append("| θ | ν̂(t=0) | ν̂(t=0.6) | ν̂(t=1.0) | Δν̂(0→1) |\n")
    L.append("|:-:|:------:|:--------:|:--------:|:--------:|\n")
    for th in ["safe", "shiny"]:
        nu_by_t = {}
        for ht in [0.0, 0.6, 1.0]:
            nus = []
            for sid in range(NS):
                recs = run_4d_session(ALL_LESSONS, th, sid, hidden_tempt=ht)
                nus.append(recs[-1]["m_hat"]["nu"])
            nu_by_t[ht] = np.mean(nus)
        delta = nu_by_t[1.0] - nu_by_t[0.0]
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, nu_by_t[0.0], nu_by_t[0.6], nu_by_t[1.0], delta))

    # ─── Exp 4: State Semantics Replication ──────────────
    L.append("\n## Exp 4: State Semantics (γ̂_spec vs Resist Rate)\n\n")
    print("\nExp 4: State semantics...", file=sys.stderr)
    L.append("| θ | tempt | γ̂_spec(final) | r_resist | per-tempt Corr |\n")
    L.append("|:-:|:-----:|:-------------:|:--------:|:--------------:|\n")
    for th in ["safe", "shiny"]:
        gs_all = []; rr_all = []
        for ht_label, ht in [("0.0", 0.0), ("0.3", 0.3), ("0.6", 0.6), ("1.0", 1.0)]:
            gs_f = []; rr_f = []
            for sid in range(NS):
                recs = run_4d_session(ALL_LESSONS, th, sid, hidden_tempt=ht)
                gs_f.append(recs[-1]["gamma_spec_hat"])
                rr_f.append(recs[-1]["r_resist"])
                gs_all.append(recs[-1]["gamma_spec_hat"])
                rr_all.append(recs[-1]["r_resist"])
            c = sp_stats.pearsonr(gs_f, rr_f)[0] if len(set(gs_f)) > 1 else float('nan')
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                th, ht_label, np.mean(gs_f), np.mean(rr_f), c))
        if len(gs_all) > 3:
            c_all, p_all = sp_stats.pearsonr(gs_all, rr_all)
            L.append(f"\n**θ={th} overall: Corr = {c_all:.4f} (p={p_all:.4f})**\n\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")
    L.append("> 4D observer formal evaluation complete. "
             "See metrics above for pass/fail on each experiment.\n")

    rpt = out / "p4b_4d_observer_eval.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
