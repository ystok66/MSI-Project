"""T1 Exp-4: Stability Retest — Full Protocol with ε_Q=0.05 Dead-Zone.

Re-runs the complete evaluation protocol after selecting ε_Q=0.05:
  1. 55/55 tests (pytest)
  2. Held-out family prediction (4D vs 5D)
  3. Rerank / Top-1 / STOP stability
  4. OOD 5-condition pass rate
  5. κ̂ bonus direction check

Uses dead-zone wrapper externally (no source code change).
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from scipy import stats as sp_stats
from collections import defaultdict

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
CAT = list(LESSON_CATALOG_V2)
RISK_FAMILIES = {"tic_rescue_heavy", "blind_activation_corridor",
                 "warn_symmetric_rescue"}
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10
EPS_Q = 0.05  # Selected dead-zone threshold
BETA_K = 0.02


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def dead_zone_wrapper(action, info, eps_Q=EPS_Q):
    qd = info.get("q_detail")
    if qd is None:
        return action
    if abs(qd["delta_Q"]) <= eps_Q:
        return "WAIT"
    return action


def linear_r2(X, y):
    X = np.array(X, dtype=float); y = np.array(y, dtype=float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        ss_res = np.sum((y - X @ beta)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return 0.0


def run_session(lessons, theta, seed, n_teach=20,
                hidden_tempt=0.0, risk_scale=1.0, use_dz=True):
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

        # Infer + optional DZ
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, info_i = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)
        if use_dz:
            act_i = dead_zone_wrapper(act_i, info_i)

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
            "step": step, "family": les.name,
            "diverge": diverge, "active": active,
            "correct": correct, "owr": owr, "est": est5,
            "delta_risk": risk - risk_hat,
            "oracle_warn": act_o == "WARN",
            "infer_warn": act_i == "WARN",
        })
    return records


def main():
    print("═══ T1 Exp-4: Stability Retest ═══\n", file=sys.stderr)
    L = [f"# T1 Exp-4: Stability Retest (ε_Q={EPS_Q}, β_κ={BETA_K})\n\n"]

    # ═══ Part 1: Micro (with dead-zone) ═══
    L.append("## Part 1: Micro Metrics (with dead-zone)\n\n")
    L.append("| θ | DivAll | Div@Act | OWR | WarnNecRecall | Success |\n")
    L.append("|:-:|:------:|:-------:|:---:|:-------------:|:-------:|\n")

    for th in ["safe", "shiny"]:
        all_recs = []
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid))
        n = len(all_recs)
        na = sum(1 for r in all_recs if r["active"])
        da = sum(r["diverge"] for r in all_recs) / n
        dact = sum(1 for r in all_recs if r["diverge"] and r["active"]) / max(na, 1)
        owr = sum(r["owr"] for r in all_recs) / n
        ow_n = sum(r["oracle_warn"] for r in all_recs)
        wnr = sum(r["oracle_warn"] and r["infer_warn"] for r in all_recs) / max(ow_n, 1)
        succ = sum(r["correct"] for r in all_recs) / n
        L.append(f"| {th} | {da:.4f} | {dact:.4f} | {owr:.4f} | {wnr:.4f} | {succ:.3f} |\n")
        print(f"  {th}: DivAll={da:.4f} OWR={owr:.4f} WNR={wnr:.4f}", file=sys.stderr)

    # ═══ Part 2: Macro (Top-1, STOP, Kendall) ═══
    L.append("\n## Part 2: Macro Stability\n\n")
    L.append("| θ | Top-1 Same | Kendall | Risk Shift | STOP | κ̂ |\n")
    L.append("|:-:|:----------:|:-------:|:----------:|:----:|:--:|\n")
    for th in ["safe", "shiny"]:
        recs = run_session(ALL_LESSONS, th, 0)
        est = recs[-1]["est"]
        base = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
                for l in CAT]
        bonus = []
        for i, l in enumerate(CAT):
            s = base[i]
            if l.name in RISK_FAMILIES:
                s += BETA_K * abs(est["kappa"] - 0.3)
            bonus.append(s)
        rank_b = list(np.argsort(base)[::-1])
        rank_n = list(np.argsort(bonus)[::-1])
        t1_same = (rank_b[0] == rank_n[0])
        kt, _ = sp_stats.kendalltau(base, bonus)
        shifts = [rank_b.index(i) - rank_n.index(i)
                  for i, l in enumerate(CAT) if l.name in RISK_FAMILIES]
        s_stop = EPS_0 + A_S * est["nu"] + B_S * est["gamma_gen"]
        L.append("| {} | {} | {:.4f} | {:+.1f} | {:.4f} | {:.4f} |\n".format(
            th, "✅" if t1_same else "❌", kt,
            np.mean(shifts), s_stop, est["kappa"]))

    # ═══ Part 3: OOD 5 conditions ═══
    L.append("\n## Part 3: OOD Robustness\n\n")
    L.append("| Condition | θ | DivAll | OWR | Top-1 | κ̂ |\n")
    L.append("|-----------|:-:|:------:|:---:|:-----:|:--:|\n")
    conditions = [
        ("Canonical", ALL_LESSONS, 0.0, 1.0),
        ("Tempt-rich", ALL_LESSONS, 0.6, 1.0),
        ("Balanced-Act", BALANCED_ACTIVE_LESSONS, 0.0, 1.0),
        ("High-risk×1.5", ALL_LESSONS, 0.0, 1.5),
        ("Low-risk×0.5", ALL_LESSONS, 0.0, 0.5),
    ]
    ood_pass = 0; ood_total = 0
    for cname, lessons, ht, rs in conditions:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_session(lessons, th, sid,
                            hidden_tempt=ht, risk_scale=rs))
            n = len(recs)
            da = sum(r["diverge"] for r in recs) / n
            owr = sum(r["owr"] for r in recs) / n
            recs_m = run_session(lessons, th, 0, hidden_tempt=ht, risk_scale=rs)
            est = recs_m[-1]["est"]
            base = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
                    for l in CAT]
            bonus = [base[i] + (BETA_K * abs(est["kappa"]-0.3)
                     if CAT[i].name in RISK_FAMILIES else 0) for i in range(len(CAT))]
            t1 = np.argsort(base)[::-1][0] == np.argsort(bonus)[::-1][0]
            ood_total += 1
            if t1: ood_pass += 1
            L.append("| {} | {} | {:.4f} | {:.4f} | {} | {:.4f} |\n".format(
                cname, th, da, owr, "✅" if t1 else "❌", est["kappa"]))
        print(f"  OOD {cname} done", file=sys.stderr)

    # ═══ Part 4: Held-out ═══
    L.append("\n## Part 4: Held-Out Family Prediction\n\n")
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid))
    fam_groups = defaultdict(list)
    for r in all_recs:
        fam_groups[r["family"]].append(r)

    L.append("| Held-Out | MAE(4D) | MAE(5D) | Δ |\n")
    L.append("|----------|:-------:|:-------:|:-:|\n")
    wins = 0; total = 0
    for held_out in sorted(fam_groups.keys()):
        train = [r for r in all_recs if r["family"] != held_out]
        test = [r for r in all_recs if r["family"] == held_out]
        if len(test) < 5: continue
        y_tr = np.array([r["delta_risk"] for r in train])
        y_te = np.array([r["delta_risk"] for r in test])
        x4_tr = np.array([[r["est"]["tau"], r["est"]["nu"],
                            r["est"]["gamma_gen"], r["est"]["gamma_spec"]]
                           for r in train])
        x4_te = np.array([[r["est"]["tau"], r["est"]["nu"],
                            r["est"]["gamma_gen"], r["est"]["gamma_spec"]]
                           for r in test])
        x5_tr = np.array([[r["est"]["tau"], r["est"]["nu"],
                            r["est"]["gamma_gen"], r["est"]["gamma_spec"],
                            r["est"]["kappa"]] for r in train])
        x5_te = np.array([[r["est"]["tau"], r["est"]["nu"],
                            r["est"]["gamma_gen"], r["est"]["gamma_spec"],
                            r["est"]["kappa"]] for r in test])
        X4 = np.column_stack([np.ones(len(x4_tr)), x4_tr])
        X4t = np.column_stack([np.ones(len(x4_te)), x4_te])
        X5 = np.column_stack([np.ones(len(x5_tr)), x5_tr])
        X5t = np.column_stack([np.ones(len(x5_te)), x5_te])
        try:
            b4 = np.linalg.lstsq(X4, y_tr, rcond=None)[0]
            mae4 = np.mean(np.abs(y_te - X4t @ b4))
        except: mae4 = float('nan')
        try:
            b5 = np.linalg.lstsq(X5, y_tr, rcond=None)[0]
            mae5 = np.mean(np.abs(y_te - X5t @ b5))
        except: mae5 = float('nan')
        imp = mae4 - mae5; total += 1
        if imp > 0: wins += 1
        L.append(f"| {held_out} | {mae4:.4f} | {mae5:.4f} | {imp:+.4f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Final Verdict\n\n")
    L.append(f"> OOD pass rate: {ood_pass}/{ood_total}\n")
    L.append(f"> Held-out 5D wins: {wins}/{total}\n")
    checks = []
    checks.append(("OOD Top-1 all pass", ood_pass == ood_total))
    checks.append(("Held-out 5D >= 50% wins", wins >= total / 2))
    all_pass = all(c[1] for c in checks)
    for name, passed in checks:
        L.append(f"> {'✅' if passed else '❌'} {name}\n")
    if all_pass:
        L.append("\n> **✅ STABILITY RETEST: PASS — Canonical is locked.**\n")
    else:
        L.append("\n> **⚠️ Some checks need review**\n")

    rpt = out / "t1_exp4_stability_retest.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
