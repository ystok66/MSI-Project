"""P6-A: κ̂ Macro Bonus OOD Robustness.

Tests κ bonus (β=0.02) under three OOD conditions:
  1. Temptation-rich (hidden_tempt=0.6)
  2. Balanced-active suite
  3. Shifted risk priors (risk × 1.5, risk × 0.5)
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
NS = 12
ALL_LESSONS = list(LESSON_CATALOG_V2)
CAT = list(LESSON_CATALOG_V2)
RISK_FAMILIES = {"tic_rescue_heavy", "blind_activation_corridor",
                 "warn_symmetric_rescue"}
BETA_K = 0.02
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_ood_session(lessons, theta, seed, n_teach=20,
                    hidden_tempt=0.0, risk_scale=1.0):
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

        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, _ = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, _ = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

        action, dose = act_o, dose_o
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = min(getattr(sc, 'risk_level', 0.3) * risk_scale, 1.0)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        eff_lure = tempt + hidden_tempt
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b) + hidden_tempt * 0.5)
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a) + hidden_tempt * 0.5)
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
        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat, lure=eff_lure,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)
        est5 = observer.get_estimate()
        diverge = (act_o != act_i)
        active = (act_o != "WAIT") or (act_i != "WAIT")
        owr = (act_o == "WAIT" and act_i == "WARN")
        records.append({
            "step": step, "diverge": diverge, "active": active,
            "correct": correct, "owr": owr, "est": est5,
        })
    return records


def eval_condition(cond_name, lessons, thetas, hidden_tempt, risk_scale):
    """Run full Protocol A+B for one OOD condition."""
    rows = {}
    for th in thetas:
        all_recs = []
        for sid in range(NS):
            all_recs.extend(run_ood_session(
                lessons, th, sid, hidden_tempt=hidden_tempt,
                risk_scale=risk_scale))
        n = len(all_recs)
        na = sum(1 for r in all_recs if r["active"])
        da = sum(r["diverge"] for r in all_recs) / n
        dact = sum(1 for r in all_recs if r["diverge"] and r["active"]) / max(na, 1)
        owr = sum(r["owr"] for r in all_recs) / n
        succ = sum(r["correct"] for r in all_recs) / n

        # Macro: STOP + Kendall (last session)
        recs_last = run_ood_session(lessons, th, 0, hidden_tempt=hidden_tempt,
                                     risk_scale=risk_scale)
        est = recs_last[-1]["est"]
        so_base = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
                   for l in CAT]
        so_bonus = []
        for i, l in enumerate(CAT):
            s = so_base[i]
            if l.name in RISK_FAMILIES:
                s += BETA_K * abs(est["kappa"] - 0.3)
            so_bonus.append(s)
        t1_base = 1; t1_bonus = 1  # Always agree with self
        kt_base = 1.0
        kt_bonus, _ = sp_stats.kendalltau(so_base, so_bonus)
        rank_base = list(np.argsort(so_base)[::-1])
        rank_bonus = list(np.argsort(so_bonus)[::-1])
        shifts = []
        for i, l in enumerate(CAT):
            if l.name in RISK_FAMILIES:
                shifts.append(rank_base.index(i) - rank_bonus.index(i))
        avg_shift = np.mean(shifts) if shifts else 0

        # Top-1 same?
        t1_same = (rank_base[0] == rank_bonus[0])

        # STOP
        s_stop = EPS_0 + A_S * est["nu"] + B_S * est["gamma_gen"]

        rows[th] = {
            "DivAll": da, "Div@Act": dact, "OWR": owr, "Success": succ,
            "Shift": avg_shift, "Top1Same": t1_same,
            "Kendall": kt_bonus, "STOP": s_stop, "kappa": est["kappa"],
        }
    return rows


def main():
    print("═══ P6-A: OOD Robustness ═══\n", file=sys.stderr)
    L = ["# P6-A: κ̂ Macro Bonus OOD Robustness (β=0.02)\n\n"]

    conditions = [
        ("Canonical", ALL_LESSONS, ["safe", "shiny"], 0.0, 1.0),
        ("Temptation-rich", ALL_LESSONS, ["safe", "shiny"], 0.6, 1.0),
        ("Balanced-Active", BALANCED_ACTIVE_LESSONS, ["safe", "shiny"], 0.0, 1.0),
        ("High-risk (×1.5)", ALL_LESSONS, ["safe", "shiny"], 0.0, 1.5),
        ("Low-risk (×0.5)", ALL_LESSONS, ["safe", "shiny"], 0.0, 0.5),
    ]

    L.append("## Micro Protocol A\n\n")
    L.append("| Condition | θ | DivAll | Div@Act | OWR | Success |\n")
    L.append("|-----------|:-:|:------:|:-------:|:---:|:-------:|\n")

    all_rows = {}
    for cname, lessons, thetas, ht, rs in conditions:
        print(f"  {cname}...", file=sys.stderr)
        rows = eval_condition(cname, lessons, thetas, ht, rs)
        all_rows[cname] = rows
        for th in thetas:
            r = rows[th]
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} |\n".format(
                cname, th, r["DivAll"], r["Div@Act"], r["OWR"], r["Success"]))

    L.append("\n## Macro Protocol B (κ-Bonus)\n\n")
    L.append("| Condition | θ | Risk Shift | Top-1 Same | Kendall | "
             "κ̂(final) |\n")
    L.append("|-----------|:-:|:----------:|:----------:|:-------:|"
             ":---------:|\n")
    for cname, _, thetas, _, _ in conditions:
        rows = all_rows[cname]
        for th in thetas:
            r = rows[th]
            L.append("| {} | {} | {:+.1f} | {} | {:.4f} | {:.4f} |\n".format(
                cname, th, r["Shift"], "✅" if r["Top1Same"] else "❌",
                r["Kendall"], r["kappa"]))

    # Verdict
    L.append("\n## Verdict\n\n")
    all_t1 = all(rows[th]["Top1Same"]
                 for rows in all_rows.values()
                 for th in rows)
    all_shifts_pos = all(rows[th]["Shift"] >= 0
                         for rows in all_rows.values()
                         for th in rows)
    L.append(f"> **Top-1 stable across all OOD**: {'✅' if all_t1 else '❌'}\n")
    L.append(f"> **Risk shifts non-negative**: {'✅' if all_shifts_pos else '❌'}\n")
    if all_t1 and all_shifts_pos:
        L.append("> **κ̂ macro bonus (β=0.02) passes OOD robustness.**\n")
    else:
        L.append("> **Some OOD conditions need investigation.**\n")

    rpt = out / "p6a_ood_robustness.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
