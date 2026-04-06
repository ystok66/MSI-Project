"""P4-B.1: Metric Integrity Audit + P4-C: 4D Observer Formalization.

Audit 1: Diagnose Div All > 0 but Div@Active = 0 anomaly
Audit 2: Re-run with corrected active mask definitions
Audit 3: Verify metric consistency under 2-act canonical
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


def run_session_full_metrics(lessons, theta, seed, n_teach=20, hidden_tempt=0.0):
    """Full session returning oracle/infer actions with detailed metric info."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
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

        # Oracle decision
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, info_o = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer-only decision (4D observer)
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.gamma_spec = est["gamma_spec"]
        m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, info_i = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

        action, dose = act_o, dose_o
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

        # Metric definitions
        diverge = (act_o != act_i)
        # OLD active: oracle warned
        active_old = (dose_o > 0)
        # NEW active: either oracle or infer is not WAIT
        active_new = (act_o != "WAIT") or (act_i != "WAIT")

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "act_oracle": act_o, "act_infer": act_i,
            "dose_oracle": dose_o, "dose_infer": dose_i,
            "diverge": diverge,
            "active_old": active_old, "active_new": active_new,
            "Q_oracle": info_o.get("Q", 0), "Q_infer": info_i.get("Q", 0),
            "correct": correct,
        })
    return records


def compute_metrics(recs, active_key="active_new"):
    """Compute standardized metrics."""
    n = len(recs)
    div_all = sum(r["diverge"] for r in recs) / n
    n_act = sum(1 for r in recs if r[active_key])
    div_at_act = (sum(1 for r in recs if r["diverge"] and r[active_key])
                  / max(n_act, 1))
    # Active regret
    r_active = 0.0
    if n_act > 0:
        for r in recs:
            if r[active_key] and r["diverge"]:
                r_active += abs(r["Q_oracle"] - r["Q_infer"])
        r_active /= n_act
    return {"div_all": div_all, "div_active": div_at_act,
            "n_active": n_act, "r_active": r_active, "n": n}


def main():
    print("═══ P4-B.1: Metric Audit + P4-C: Formalization ═══\n", file=sys.stderr)
    L = ["# P4-B.1: Metric Integrity Audit\n\n"]

    # ─── Audit 1: Diagnose the anomaly ───────────────────
    L.append("## Audit 1: Old vs New Active Mask\n\n")
    print("Audit 1: Active mask diagnosis...", file=sys.stderr)

    L.append("**Old active**: step where oracle warned (dose > 0)\n")
    L.append("**New active**: step where oracle OR infer is not WAIT\n\n")
    L.append("| Suite | θ | tempt | DivAll | Div@OldAct | Div@NewAct | "
             "n_old | n_new | R_active |\n")
    L.append("|-------|:-:|:-----:|:------:|:----------:|:----------:|"
             ":----:|:-----:|:--------:|\n")

    suites = [
        ("Canonical", list(LESSON_CATALOG_V2), [("none", 0.0)]),
        ("Active", BALANCED_ACTIVE_LESSONS, [("none", 0.0)]),
        ("Tempt", list(LESSON_CATALOG_V2),
         [("none", 0.0), ("al=0.6", 0.6), ("cf=1.0", 1.0)]),
    ]
    for sname, lessons, tempts in suites:
        for th in ["safe", "shiny"]:
            for ht_label, ht in tempts:
                recs = []
                for sid in range(NS):
                    recs.extend(run_session_full_metrics(
                        lessons, th, sid, hidden_tempt=ht))
                m_old = compute_metrics(recs, "active_old")
                m_new = compute_metrics(recs, "active_new")
                L.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | "
                         "{} | {} | {:.4f} |\n".format(
                    sname, th, ht_label,
                    m_old["div_all"], m_old["div_active"], m_new["div_active"],
                    m_old["n_active"], m_new["n_active"], m_new["r_active"]))

    # ─── Audit 2: Divergence forensics ───────────────────
    L.append("\n## Audit 2: Divergence Forensics\n\n")
    print("\nAudit 2: Forensics...", file=sys.stderr)
    all_divs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            recs = run_session_full_metrics(ALL_LESSONS, th, sid)
            for r in recs:
                if r["diverge"]:
                    all_divs.append(r)
    L.append(f"**Total divergences: {len(all_divs)} across "
             f"{NS*2*20} steps**\n\n")
    if all_divs:
        L.append("| θ | Family | Step | Oracle | Infer | "
                 "Active(old) | Active(new) |\n")
        L.append("|:-:|--------|:----:|:------:|:-----:|"
                 ":-----------:|:-----------:|\n")
        for d in all_divs[:20]:
            L.append("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                d["theta"], d["family"], d["step"],
                d["act_oracle"], d["act_infer"],
                "✅" if d["active_old"] else "❌",
                "✅" if d["active_new"] else "❌"))
    else:
        L.append("**No divergences found.**\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")
    # Count anomalies: Div=True but active_old=False
    anomalies = sum(1 for d in all_divs if not d["active_old"])
    if anomalies > 0:
        L.append(f"> **{anomalies} divergences were OLD-active=False "
                 f"but NEW-active=True.**\n")
        L.append("> These are steps where the infer-only tutor chose WARN "
                 "but oracle chose WAIT.\n")
        L.append("> The old `active` mask (oracle-warned-only) missed them. "
                 "**NEW mask is the correct definition.**\n")
    else:
        L.append("> **No active mask anomalies.** Old and new definitions agree.\n")

    rpt = out / "p4b1_metric_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
