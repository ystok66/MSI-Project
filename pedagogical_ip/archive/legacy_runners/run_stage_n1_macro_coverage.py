"""Stage N+1 P1: Online Macro-Hybrid Pilot + P2: Coverage Benchmark.

Exp 1: Online macro-hybrid (tutor uses m_hybrid for STOP/EVAL/TEACH)
Exp 2: Online macro infer-only (α=1.0)
Exp 3: Policy coverage benchmark (event mix by family/θ/timing)
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
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)

# STOP params (from canonical controller)
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_macro_session(lessons, theta, seed, n_teach=20, alpha=0.0,
                      hidden_tempt=0.0):
    """Online macro session: STOP/EVAL/TEACH decisions use m_hybrid.

    The full teaching loop with macro-level decisions driven by hybrid state.
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    stopped_oracle = None; stopped_hybrid = None

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

        # Oracle micro decision
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        # Hybrid micro decision
        m_hyb = m.copy()
        m_hyb.tau = (1 - alpha) * m.tau + alpha * observer.tau_hat
        m_hyb.nu = (1 - alpha) * m.nu + alpha * observer.nu_hat
        m_hyb.gamma_gen = (1 - alpha) * m.gamma_gen + alpha * observer.gamma_gen_hat
        a_hybrid, dose_hybrid, info_hybrid = tutor.decide(sc, fb, lp, lib, scr, 2, m_hyb)

        # ── Macro: STOP decision ──
        eps_oracle = EPS_0 + A_S * m.nu + B_S * m.gamma_gen
        eps_hybrid = EPS_0 + A_S * m_hyb.nu + B_S * m_hyb.gamma_gen
        stop_oracle = eps_oracle > STOP_THRESH
        stop_hybrid = eps_hybrid > STOP_THRESH
        if stopped_oracle is None and stop_oracle:
            stopped_oracle = step
        if stopped_hybrid is None and stop_hybrid:
            stopped_hybrid = step

        # ── Macro: Lesson ranking ──
        scores_oracle = []
        scores_hybrid = []
        for l in lessons:
            gain = np.mean(l.gain) if hasattr(l, 'gain') else 0.5
            s_o = gain * (1.0 - m.nu) * (1.0 - m.gamma_gen) * m.tau
            s_h = gain * (1.0 - m_hyb.nu) * (1.0 - m_hyb.gamma_gen) * m_hyb.tau
            scores_oracle.append(s_o)
            scores_hybrid.append(s_h)
        rank_o = np.argsort(scores_oracle)[::-1]
        rank_h = np.argsort(scores_hybrid)[::-1]
        top1_agree = (rank_o[0] == rank_h[0])

        # Agent acts (using oracle-driven dose for fair comparison)
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
        warned = dose_oracle > 0; follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev

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
            dose=dose_oracle, warned=warned, follow_warn=follow_warn,
            warn_correct=(warned and risk > 0.25), warn_wrong=(warned and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "correct": correct, "dose_oracle": dose_oracle, "dose_hybrid": dose_hybrid,
            "a_oracle": a_oracle, "a_hybrid": a_hybrid,
            "micro_diverge": (a_oracle != a_hybrid),
            "eps_oracle": eps_oracle, "eps_hybrid": eps_hybrid,
            "stop_agree": (stop_oracle == stop_hybrid),
            "top1_agree": top1_agree,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            "dc_minus_dr": dc - dr, "p_self": p_self,
            "warned": warned, "self_disc": self_disc,
        })

    # Compute ranking correlation on final state
    if len(scores_oracle) > 1:
        kt, _ = sp_stats.kendalltau(scores_oracle, scores_hybrid)
        sr_corr, _ = sp_stats.spearmanr(scores_oracle, scores_hybrid)
    else:
        kt = sr_corr = 1.0

    return records, {
        "stopped_oracle": stopped_oracle, "stopped_hybrid": stopped_hybrid,
        "kendall": round(kt, 4), "spearman": round(sr_corr, 4),
    }


def main():
    print("═══ Stage N+1: Macro Pilot + Coverage ═══\n", file=sys.stderr)
    L = ["# Stage N+1: Online Macro-Hybrid Pilot + Coverage\n\n"]

    # ─── Exp 1: Macro-Hybrid (α sweep) ───────────────────
    L.append("## Exp 1: Online Macro-Hybrid\n\n")
    L.append("| α | θ | STOP Agree | Micro Div | Top-1 Agree | Kendall τ | "
             "Spearman ρ | Δε_stop |\n")
    L.append("|:-:|:-:|:----------:|:---------:|:-----------:|:---------:|"
             ":----------:|:-------:|\n")
    print("Exp 1: Macro-hybrid...", file=sys.stderr)
    for alpha in [0.0, 0.5, 1.0]:
        for th in ["safe", "shiny"]:
            all_recs = []; all_meta = []
            for sid in range(NS):
                recs, meta = run_macro_session(ALL_LESSONS, th, sid, alpha=alpha)
                all_recs.extend(recs); all_meta.append(meta)
            n = len(all_recs)
            stop_agree = sum(r["stop_agree"] for r in all_recs) / n
            micro_div = sum(r["micro_diverge"] for r in all_recs) / n
            top1 = sum(r["top1_agree"] for r in all_recs) / n
            d_eps = np.mean([abs(r["eps_oracle"] - r["eps_hybrid"]) for r in all_recs])
            kt = np.mean([m["kendall"] for m in all_meta])
            sr = np.mean([m["spearman"] for m in all_meta])
            L.append("| {} | {} | {:.3f} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                alpha, th, stop_agree, micro_div, top1, kt, sr, d_eps))
            print(f"  α={alpha} θ={th}: STOP={stop_agree:.3f} top1={top1:.3f} "
                  f"kendall={kt:.4f}", file=sys.stderr)

    # ─── Exp 2: Macro Infer-Only (α=1.0) + Temptation ───
    L.append("\n## Exp 2: Macro Infer-Only + Temptation\n\n")
    L.append("| Config | θ | STOP Agree | Micro Div | Top-1 | Kendall τ |\n")
    L.append("|--------|:-:|:----------:|:---------:|:-----:|:---------:|\n")
    print("\nExp 2: Infer-only + tempt...", file=sys.stderr)
    for ht_label, ht in [("canonical", 0.0), ("tempt=0.6", 0.6), ("tempt=1.0", 1.0)]:
        for th in ["safe", "shiny"]:
            recs_all = []; meta_all = []
            for sid in range(NS):
                recs, meta = run_macro_session(ALL_LESSONS, th, sid, alpha=1.0,
                                               hidden_tempt=ht)
                recs_all.extend(recs); meta_all.append(meta)
            n = len(recs_all)
            stop_agree = sum(r["stop_agree"] for r in recs_all) / n
            micro_div = sum(r["micro_diverge"] for r in recs_all) / n
            top1 = sum(r["top1_agree"] for r in recs_all) / n
            kt = np.mean([m["kendall"] for m in meta_all])
            L.append("| {} | {} | {:.3f} | {:.4f} | {:.3f} | {:.4f} |\n".format(
                ht_label, th, stop_agree, micro_div, top1, kt))

    # ─── Exp 3: Policy Coverage Benchmark ────────────────
    L.append("\n## Exp 3: Policy Coverage Benchmark\n\n")
    L.append("### By θ\n\n")
    L.append("| θ | Steps | warn | dose>0 | selfdisc | blind | trust | active |\n")
    L.append("|:-:|:-----:|:----:|:------:|:--------:|:-----:|:-----:|:------:|\n")
    print("\nExp 3: Coverage...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        recs_all = []
        for sid in range(NS):
            recs, _ = run_macro_session(ALL_LESSONS, th, sid, alpha=0.0)
            recs_all.extend(recs)
        n = len(recs_all)
        warn = sum(1 for r in recs_all if r["dose_oracle"] >= 1.0) / n
        dose = sum(1 for r in recs_all if r["dose_oracle"] > 0) / n
        sd = sum(1 for r in recs_all if r.get("self_disc", False)) / n
        blind = sum(1 for r in recs_all if r.get("warned", False) and
                    not r.get("self_disc", False)) / n
        trust = warn  # trust events = warn events
        active = sum(1 for r in recs_all if r["dose_oracle"] > 0) / n
        L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, n, warn, dose, sd, blind, trust, active))

    L.append("\n### By dc−dr Bin\n\n")
    L.append("| dc−dr | n | selfdisc | blind | warned |\n")
    L.append("|:-----:|:-:|:--------:|:-----:|:------:|\n")
    all_coverage = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            recs, _ = run_macro_session(ALL_LESSONS, th, sid, alpha=0.0)
            all_coverage.extend(recs)
    for gap_label, lo, hi in [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3+", 3, 99)]:
        sub = [r for r in all_coverage if lo <= r.get("dc_minus_dr", 1) <= hi]
        if not sub: continue
        n = len(sub)
        sd = sum(1 for r in sub if r.get("self_disc", False)) / n
        bl = sum(1 for r in sub if r.get("warned", False)) / n
        wn = sum(1 for r in sub if r["dose_oracle"] >= 1.0) / n
        L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            gap_label, n, sd, bl, wn))

    L.append("\n### By Family\n\n")
    L.append("| Family | n | selfdisc | warned | active |\n")
    L.append("|--------|:-:|:--------:|:------:|:------:|\n")
    fam_groups = {}
    for r in all_coverage:
        fam_groups.setdefault(r["family"], []).append(r)
    for fam, recs in sorted(fam_groups.items()):
        n = len(recs)
        sd = sum(1 for r in recs if r.get("self_disc", False)) / n
        wn = sum(1 for r in recs if r["dose_oracle"] >= 1.0) / n
        act = sum(1 for r in recs if r["dose_oracle"] > 0) / n
        L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(fam, n, sd, wn, act))

    rpt = out / "stage_n1_macro_coverage_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
