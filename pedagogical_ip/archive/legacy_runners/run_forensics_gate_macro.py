"""Active divergence forensics + tie-aware gate + macro ranking replay.

Exp 1: Forensics — dissect every active divergence, log per-dose Q values
Exp 2: Tie-aware gate (raw vs WAIT-gate vs SOFT-gate)
Exp 3: Macro lesson ranking replay (Kendall τ, top-1 agreement)
Exp 4: Hidden temptation aligned vs conflicting
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
from src.agents.behavior_probes import all_probes
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserver, ObsEvent
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


def decide_all_doses(tutor, sc, fb, lp, lib, scr, obs, m):
    """Call tutor.decide for each dose to get per-dose Q breakdown."""
    # Get the main decision
    best_action, best_dose, info = tutor.decide(sc, fb, lp, lib, scr, obs, m)
    Q_best = info.get("Q", 0)

    # Create a temp tutor without dose to get WAIT vs WARN only
    tutor_nodose = BCICTv4(agent_params=AP, use_dose=False)
    tutor_nodose.calibrator = tutor.calibrator
    tutor_nodose._zones = tutor._zones
    a_nd, d_nd, i_nd = tutor_nodose.decide(sc, fb, lp, lib, scr, obs, m)

    return best_action, best_dose, Q_best, {
        "best": best_action, "best_dose": best_dose, "Q": Q_best,
    }


def run_forensic_session(lessons, theta, seed, n_teach=16, hidden_tempt=0.0,
                         force_active_dose=None, gate_mode="raw"):
    """Run session with full forensic logging.

    gate_mode: "raw" | "wait_gate" | "soft_gate"
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserver(); observer.reset()
    records = []

    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / max(n_teach, 1) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
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

        # Oracle decision
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        Q_oracle = info_oracle.get("Q", 0)

        # Infer decision using m̂
        m_hat_state = m.copy()
        m_hat_state.tau = observer.tau_hat
        m_hat_state.nu = observer.nu_hat
        m_hat_state.gamma_gen = observer.gamma_gen_hat
        a_infer, dose_infer, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat_state)
        Q_infer = info_infer.get("Q", 0)

        # Gate logic
        margin_infer = abs(Q_infer)  # proxy for margin
        conf = observer.get_confidence()
        mean_conf = np.mean([conf.get("tau", 0.5), conf.get("nu", 0.5), conf.get("gamma_gen", 0.5)])

        if gate_mode == "wait_gate":
            if margin_infer < 0.5 or mean_conf < 0.3:
                a_gated = "WAIT"; dose_gated = 0.0
            else:
                a_gated = a_infer; dose_gated = dose_infer
        elif gate_mode == "soft_gate":
            if margin_infer < 0.5 or mean_conf < 0.3:
                if a_infer == "WARN":
                    a_gated = "SOFT"; dose_gated = 0.5
                else:
                    a_gated = "WAIT"; dose_gated = 0.0
            else:
                a_gated = a_infer; dose_gated = dose_infer
        else:
            a_gated = a_infer; dose_gated = dose_infer

        # Actual dose (oracle-driven for fair comparison)
        dose_actual = dose_oracle if force_active_dose is None else force_active_dose

        # Agent acts
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
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
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose_actual > 0; follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev
        bn = ep.subtype == "beneficial_novelty" and correct

        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if not has_self_ev: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()

        probes = all_probes(m, AP, theta) if step % 2 == 0 else {}
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose_actual, warned=warned, follow_warn=follow_warn,
            warn_correct=(warned and risk > 0.25), warn_wrong=(warned and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc, beneficial_novelty=bn,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        oracle_nonwait = dose_oracle > 0
        infer_nonwait = dose_infer > 0
        is_active = oracle_nonwait or infer_nonwait
        diverge = (a_oracle != a_infer)
        diverge_gated = (a_oracle != a_gated)

        records.append({
            "step": step, "theta": theta, "family": les.name, "subtype": ep.subtype,
            "correct": correct, "dc_minus_dr": dc - dr,
            "a_oracle": a_oracle, "dose_oracle": dose_oracle, "Q_oracle": Q_oracle,
            "a_infer": a_infer, "dose_infer": dose_infer, "Q_infer": Q_infer,
            "a_gated": a_gated, "dose_gated": dose_gated,
            "diverge": diverge, "diverge_gated": diverge_gated,
            "is_active": is_active, "margin_infer": margin_infer,
            "mean_conf": mean_conf,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            "hidden_tempt": hidden_tempt, "risk": risk,
        })
    return records


def main():
    print("═══ Forensics + Gate + Macro ═══\n", file=sys.stderr)
    L = ["# Active Divergence Forensics + Tie-Aware Gate\n\n"]

    # ─── Exp 1: Forensics on active divergence cases ─────
    L.append("## Exp 1: Active Divergence Forensics\n\n")
    print("Exp 1: Forensics...", file=sys.stderr)
    # Run with forced active to generate divergences
    all_divs = []
    for fd in [None, 0.5, 1.0]:
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs = run_forensic_session(ALL_LESSONS, th, sid,
                                            force_active_dose=fd)
                for r in recs:
                    if r["diverge"]:
                        all_divs.append(r)

    if all_divs:
        L.append(f"**Total divergent steps found: {len(all_divs)}**\n\n")
        # Classify divergence types
        types = {}
        for d in all_divs:
            key = f"{d['a_oracle']}→{d['a_infer']}"
            types[key] = types.get(key, 0) + 1
        L.append("### Divergence Type Breakdown\n\n")
        L.append("| Oracle → Infer | Count | % |\n|:---:|:---:|:---:|\n")
        for k, v in sorted(types.items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {v} | {100*v/len(all_divs):.1f}% |\n")

        # Margin analysis
        margins = [d["margin_infer"] for d in all_divs]
        L.append(f"\n### Margin on Divergent Steps\n\n")
        L.append(f"- Mean margin: {np.mean(margins):.4f}\n")
        L.append(f"- Median margin: {np.median(margins):.4f}\n")
        L.append(f"- Min margin: {np.min(margins):.4f}\n")
        L.append(f"- Max margin: {np.max(margins):.4f}\n")

        # Q difference
        dq = [abs(d["Q_oracle"] - d["Q_infer"]) for d in all_divs]
        L.append(f"\n### Q Difference on Divergent Steps\n\n")
        L.append(f"- Mean |ΔQ|: {np.mean(dq):.4f}\n")
        L.append(f"- Max |ΔQ|: {np.max(dq):.4f}\n")

        # m difference on divergent steps
        m_diffs = []
        for d in all_divs:
            mt = d["m_true"]; mh = d["m_hat"]
            m_diffs.append({
                "Δτ": abs(mt["tau"] - mh["tau"]),
                "Δν": abs(mt["nu"] - mh["nu"]),
                "Δγ": abs(mt["gamma_gen"] - mh["gamma_gen"]),
            })
        L.append(f"\n### m̂ Error on Divergent Steps\n\n")
        L.append(f"- Mean Δτ: {np.mean([d['Δτ'] for d in m_diffs]):.6f}\n")
        L.append(f"- Mean Δν: {np.mean([d['Δν'] for d in m_diffs]):.6f}\n")
        L.append(f"- Mean Δγ: {np.mean([d['Δγ'] for d in m_diffs]):.6f}\n")

        # Family distribution
        fam_counts = {}
        for d in all_divs:
            fam_counts[d["family"]] = fam_counts.get(d["family"], 0) + 1
        L.append(f"\n### Family Distribution of Divergences\n\n")
        L.append("| Family | Count |\n|--------|:-----:|\n")
        for k, v in sorted(fam_counts.items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {v} |\n")

        print(f"  Found {len(all_divs)} divergences; types={types}", file=sys.stderr)
    else:
        L.append("**No divergences found even with forced dose.**\n\n")
        print("  No divergences found!", file=sys.stderr)

    # ─── Exp 2: Tie-Aware Gate ───────────────────────────
    L.append("\n## Exp 2: Tie-Aware Gate Comparison\n\n")
    L.append("| Gate | Regime | θ | Div All | Div@Active | Div@Hard | n_div |\n")
    L.append("|------|--------|:-:|:-------:|:----------:|:--------:|:-----:|\n")
    print("\nExp 2: Gate comparison...", file=sys.stderr)
    for gate in ["raw", "wait_gate", "soft_gate"]:
        for regime, fd in [("natural", None), ("active_0.5", 0.5)]:
            for th in ["safe", "shiny"]:
                recs = []
                for sid in range(NS):
                    recs.extend(run_forensic_session(ALL_LESSONS, th, sid,
                                                     force_active_dose=fd, gate_mode=gate))
                n = len(recs)
                div_all = sum(r["diverge_gated"] for r in recs) / n if n else 0
                active = [r for r in recs if r["is_active"]]
                div_active = (sum(r["diverge_gated"] for r in active) / len(active)) if active else 0
                hard = [r for r in recs if r["margin_infer"] < 3.0]
                div_hard = (sum(r["diverge_gated"] for r in hard) / len(hard)) if hard else 0
                n_div = sum(r["diverge_gated"] for r in recs)
                L.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {} |\n".format(
                    gate, regime, th, div_all, div_active, div_hard, n_div))

    # ─── Exp 3: Macro Lesson Ranking Replay ──────────────
    L.append("\n## Exp 3: Macro Lesson Ranking Replay\n\n")
    L.append("| α | θ | Top-1 Agree | Kendall τ | Spearman ρ |\n")
    L.append("|:-:|:-:|:-----------:|:---------:|:----------:|\n")
    print("\nExp 3: Macro ranking...", file=sys.stderr)
    try:
        cc = CurriculumControllerV13()
        for alpha in [0.0, 0.5, 1.0]:
            for th in ["safe", "shiny"]:
                top1_agree = 0; kendall_taus = []; spearman_rhos = []
                total = 0
                for sid in range(NS):
                    rng = np.random.default_rng(sid * 1000)
                    m = FactoredInternalizationState(); m.snapshot()
                    observer = A1MtObserver(); observer.reset()
                    # Run a short session to build up some m state
                    for step in range(8):
                        les = ALL_LESSONS[step % len(ALL_LESSONS)]
                        ub = {p: 0.5 for p in PROBE_NAMES}
                        et = generate_episode_from_lesson_v2(les, step + sid*100, th, ub, rng)
                        ep, spec, gm, cfg_e, meta, sc = et
                        fb, ww = apply_fix(meta, sc)
                        dc = getattr(sc, 'commit_depth', 3)
                        dr = getattr(sc, 'reveal_depth', 2)
                        p_self = estimate_self_discovery_prob(dc, dr)
                        risk = getattr(sc, 'risk_level', 0.3)
                        ac = rng.integers(0, 2)
                        correct = (ac == sc.oracle_safe_branch_id)
                        warned = rng.random() < 0.1
                        if warned:
                            m.update_trust(warn_helpful=True)
                        if correct and p_self > 0.5:
                            m.update_dependence(self_discovery=True)
                        m.update_risk(risk, 0.15); m.snapshot()
                        ev = ObsEvent(
                            episode_id=sid, step_id=step, theta_post=th,
                            dose=1.0 if warned else 0.0, warned=warned,
                            follow_warn=warned and correct,
                            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
                            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
                            self_discovery=correct and not warned and p_self > 0.5,
                            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
                        )
                        observer.update(ev)

                    # Score lessons with oracle m vs hybrid m
                    scores_oracle = []
                    scores_infer = []
                    for les in ALL_LESSONS:
                        # Simplified scoring using lesson gain and state
                        gain = np.mean(les.gain) if hasattr(les, 'gain') else 0.5
                        s_oracle = gain * (1.0 - m.nu) * (1.0 - m.gamma_gen) * m.tau
                        hyb_nu = (1-alpha)*m.nu + alpha*observer.nu_hat
                        hyb_gg = (1-alpha)*m.gamma_gen + alpha*observer.gamma_gen_hat
                        hyb_tau = (1-alpha)*m.tau + alpha*observer.tau_hat
                        s_infer = gain * (1.0 - hyb_nu) * (1.0 - hyb_gg) * hyb_tau
                        scores_oracle.append(s_oracle)
                        scores_infer.append(s_infer)

                    rank_o = np.argsort(scores_oracle)[::-1]
                    rank_i = np.argsort(scores_infer)[::-1]
                    if rank_o[0] == rank_i[0]:
                        top1_agree += 1
                    if len(scores_oracle) > 1:
                        kt, _ = sp_stats.kendalltau(scores_oracle, scores_infer)
                        sr, _ = sp_stats.spearmanr(scores_oracle, scores_infer)
                        kendall_taus.append(kt)
                        spearman_rhos.append(sr)
                    total += 1

                kt_mean = np.mean(kendall_taus) if kendall_taus else 0
                sr_mean = np.mean(spearman_rhos) if spearman_rhos else 0
                agree = top1_agree / max(total, 1)
                L.append("| {} | {} | {:.3f} | {:.4f} | {:.4f} |\n".format(
                    alpha, th, agree, kt_mean, sr_mean))
                print(f"  α={alpha} θ={th}: top1={agree:.3f} kendall={kt_mean:.4f}",
                      file=sys.stderr)
    except Exception as e:
        L.append(f"Error: {e}\n")
        print(f"  Macro ranking error: {e}", file=sys.stderr)

    # ─── Exp 4: Aligned vs Conflicting Temptation ────────
    L.append("\n## Exp 4: Aligned vs Conflicting Temptation\n\n")
    L.append("| Variant | θ | Tempt | Corr_ν | MAE_ν | Div All |\n")
    L.append("|---------|:-:|:-----:|:------:|:-----:|:-------:|\n")
    print("\nExp 4: Temptation variants...", file=sys.stderr)
    for variant, desc in [("aligned", "Aligned"), ("conflicting", "Conflicting")]:
        for th in ["safe", "shiny"]:
            # Aligned: tempt toward learner preference direction
            # Conflicting: tempt against learner preference
            if variant == "aligned":
                ht = 0.8 if th == "shiny" else 0.0  # shiny attracted to risky
            else:
                ht = 0.8 if th == "safe" else 0.0    # safe attracted to risky

            recs = []
            for sid in range(NS):
                recs.extend(run_forensic_session(ALL_LESSONS, th, sid, hidden_tempt=ht))
            # Compute ν corr
            nt = [r["m_true"]["nu"] for r in recs]
            nh = [r["m_hat"]["nu"] for r in recs]
            corr_nu = float(np.corrcoef(nt, nh)[0,1]) if np.std(nt) > 1e-6 else 0
            mae_nu = float(np.mean(np.abs(np.array(nt) - np.array(nh))))
            div = sum(r["diverge"] for r in recs) / len(recs)
            L.append("| {} | {} | {:.1f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                desc, th, ht, corr_nu, mae_nu, div))

    rpt = out / "forensics_gate_macro_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
