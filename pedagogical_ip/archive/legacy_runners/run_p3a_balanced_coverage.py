"""P3-A: Balanced Active Coverage + Infer-Only on Balanced Suite.

Exp 1: Coverage verification (new families produce active events?)
Exp 2: Infer-only on balanced suite (Diverge@Active, R_active)
Exp 3: Per-family forensics (where do divergences concentrate?)
Exp 4: Macro pilot on balanced suite
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
ALL_LESSONS = list(LESSON_CATALOG_V2)  # Now includes 3 new ACTIVE families
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_balanced_session(lessons, theta, seed, n_teach=20, hidden_tempt=0.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []

    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
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

        # Oracle
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        Q_oracle = info_oracle.get("Q", 0)

        # Infer-only
        m_hat_state = m.copy()
        m_hat_state.tau = observer.tau_hat
        m_hat_state.nu = observer.nu_hat
        m_hat_state.gamma_gen = observer.gamma_gen_hat
        a_infer, dose_infer, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat_state)
        Q_infer = info_infer.get("Q", 0)

        # Active regret: Q_oracle(a_oracle) - Q_oracle(a_infer)
        # Since we pick the best Q from the oracle's perspective, regret ≥ 0
        active_regret = max(Q_oracle - Q_infer, 0.0) if a_oracle != a_infer else 0.0

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

        oracle_nonwait = dose_oracle > 0
        infer_nonwait = dose_infer > 0
        is_active = oracle_nonwait or infer_nonwait

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "correct": correct, "dose_oracle": dose_oracle, "dose_infer": dose_infer,
            "a_oracle": a_oracle, "a_infer": a_infer,
            "Q_oracle": Q_oracle, "Q_infer": Q_infer,
            "diverge": (a_oracle != a_infer),
            "is_active": is_active, "active_regret": active_regret,
            "warned": warned, "self_disc": self_disc,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
        })
    return records


def main():
    print("═══ P3-A: Balanced Active Coverage ═══\n", file=sys.stderr)
    L = ["# P3-A: Balanced Active Coverage Suite\n\n"]
    L.append(f"**Total lessons: {len(ALL_LESSONS)}** "
             f"(10 original + 3 new ACTIVE families)\n\n")

    # ─── Exp 1: Coverage Verification ────────────────────
    L.append("## Exp 1: Coverage Verification\n\n")
    L.append("| Family | n | selfdisc | **warned** | **active** | **dose>0** |\n")
    L.append("|--------|:-:|:--------:|:----------:|:----------:|:----------:|\n")
    print("Exp 1: Coverage...", file=sys.stderr)
    all_cov = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_cov.extend(run_balanced_session(ALL_LESSONS, th, sid))
    fam_groups = {}
    for r in all_cov:
        fam_groups.setdefault(r["family"], []).append(r)
    n_active_families = 0
    for fam in sorted(fam_groups.keys()):
        recs = fam_groups[fam]
        n = len(recs)
        sd = sum(1 for r in recs if r.get("self_disc", False)) / n
        wn = sum(1 for r in recs if r["dose_oracle"] >= 1.0) / n
        act = sum(1 for r in recs if r["dose_oracle"] > 0) / n
        dose = act
        if act > 0: n_active_families += 1
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |\n".format(
            fam, n, sd, wn, act, dose))
    L.append(f"\n**Families with active>0: {n_active_families}**\n")
    print(f"  Active families: {n_active_families}", file=sys.stderr)

    # ─── Exp 2: Infer-Only on Full Catalog ───────────────
    L.append("\n## Exp 2: Infer-Only on Full Catalog (incl. Active Families)\n\n")
    L.append("| θ | Div All | Div@Active | n_active | R_active | "
             "Div@Hard | Success |\n")
    L.append("|:-:|:-------:|:----------:|:--------:|:--------:|"
             ":--------:|:-------:|\n")
    print("\nExp 2: Infer-only...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        recs = []
        for sid in range(NS):
            recs.extend(run_balanced_session(ALL_LESSONS, th, sid))
        n = len(recs)
        div_all = sum(r["diverge"] for r in recs) / n
        active = [r for r in recs if r["is_active"]]
        div_active = (sum(r["diverge"] for r in active) / len(active)) if active else 0
        r_active = (np.mean([r["active_regret"] for r in active])) if active else 0
        hard = [r for r in recs if abs(r["Q_oracle"]) < 3.5]
        div_hard = (sum(r["diverge"] for r in hard) / len(hard)) if hard else 0
        success = sum(r["correct"] for r in recs) / n
        L.append("| {} | {:.4f} | {:.4f} | {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, div_all, div_active, len(active), r_active, div_hard, success))
        print(f"  θ={th}: div_all={div_all:.4f} div@active={div_active:.4f} "
              f"n_active={len(active)} R_active={r_active:.4f}", file=sys.stderr)

    # ─── Exp 3: Per-Family Divergence Forensics ──────────
    L.append("\n## Exp 3: Per-Family Divergence Forensics\n\n")
    L.append("| Family | n | Div All | n_active | Div@Active | R_active |\n")
    L.append("|--------|:-:|:-------:|:--------:|:----------:|:--------:|\n")
    print("\nExp 3: Per-family forensics...", file=sys.stderr)
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_balanced_session(ALL_LESSONS, th, sid))
    fam_groups2 = {}
    for r in all_recs:
        fam_groups2.setdefault(r["family"], []).append(r)
    for fam in sorted(fam_groups2.keys()):
        recs = fam_groups2[fam]
        n = len(recs)
        div = sum(r["diverge"] for r in recs) / n
        active = [r for r in recs if r["is_active"]]
        div_act = (sum(r["diverge"] for r in active) / len(active)) if active else 0
        r_act = (np.mean([r["active_regret"] for r in active])) if active else 0
        L.append("| {} | {} | {:.4f} | {} | {:.4f} | {:.4f} |\n".format(
            fam, n, div, len(active), div_act, r_act))

    # Divergence type breakdown
    divs = [r for r in all_recs if r["diverge"]]
    if divs:
        L.append(f"\n**Total divergences: {len(divs)} / {len(all_recs)}**\n\n")
        L.append("### Divergence Type Breakdown\n\n")
        L.append("| Oracle→Infer | Count |\n|:---:|:---:|\n")
        types = {}
        for d in divs:
            key = f"{d['a_oracle']}→{d['a_infer']}"
            types[key] = types.get(key, 0) + 1
        for k, v in sorted(types.items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {v} |\n")
        L.append(f"\n- Mean active regret on divergent: "
                 f"{np.mean([d['active_regret'] for d in divs]):.4f}\n")
        L.append(f"- Mean |ΔQ|: {np.mean([abs(d['Q_oracle']-d['Q_infer']) for d in divs]):.4f}\n")
    else:
        L.append("\n**No divergences found.**\n")

    # ─── Exp 4: Macro on Balanced Suite ──────────────────
    L.append("\n## Exp 4: Macro Pilot on Balanced Suite (α=1.0)\n\n")
    L.append("| θ | STOP Agree | Top-1 | Kendall τ | Δε_stop |\n")
    L.append("|:-:|:----------:|:-----:|:---------:|:-------:|\n")
    print("\nExp 4: Macro balanced...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        recs = []
        for sid in range(NS):
            recs.extend(run_balanced_session(ALL_LESSONS, th, sid))
        # STOP replay
        eps_o = []; eps_i = []; agree = 0
        scores_o = []; scores_i = []
        for r in recs:
            mt = r["m_true"]; mh = r["m_hat"]
            eo = EPS_0 + A_S * mt["nu"] + B_S * mt["gamma_gen"]
            ei = EPS_0 + A_S * mh["nu"] + B_S * mh["gamma_gen"]
            eps_o.append(eo); eps_i.append(ei)
            if (eo > STOP_THRESH) == (ei > STOP_THRESH): agree += 1
        # Ranking
        from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2 as CAT
        for l in CAT:
            gain = np.mean(l.gain) if hasattr(l, 'gain') else 0.5
            r0 = recs[-1]
            mt = r0["m_true"]; mh = r0["m_hat"]
            scores_o.append(gain * (1-mt["nu"]) * (1-mt["gamma_gen"]) * mt["tau"])
            scores_i.append(gain * (1-mh["nu"]) * (1-mh["gamma_gen"]) * mh["tau"])
        rank_o = np.argsort(scores_o)[::-1]
        rank_i = np.argsort(scores_i)[::-1]
        top1 = 1.0 if rank_o[0] == rank_i[0] else 0.0
        kt, _ = sp_stats.kendalltau(scores_o, scores_i)
        delta = np.mean(np.abs(np.array(eps_i) - np.array(eps_o)))
        stop_agree = agree / len(recs)
        L.append("| {} | {:.3f} | {:.3f} | {:.4f} | {:.4f} |\n".format(
            th, stop_agree, top1, kt, delta))

    rpt = out / "p3a_balanced_coverage_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
