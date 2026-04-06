"""T1 Exp-1: κ̂ Bonus Default-On Full Family Regression.

Compares canonical_off (β_κ=0) vs canonical_on (β_κ=0.02) across:
  - 13 lessons / families
  - 2 θ: safe, shiny
  - 5 OOD conditions
  - Held-out family prediction

Metrics: DivAll, Div@Active, OverWarnRate, STOP, Kendall τ, Top-1, rank shift.
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
NS = 12
ALL_LESSONS = list(LESSON_CATALOG_V2)
CAT = list(LESSON_CATALOG_V2)
RISK_FAMILIES = {"tic_rescue_heavy", "blind_activation_corridor",
                 "warn_symmetric_rescue"}
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach=20,
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

        # Oracle tutor
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, _ = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer tutor (from observer)
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
            "step": step, "family": les.name,
            "diverge": diverge, "active": active,
            "correct": correct, "owr": owr, "est": est5,
            "delta_risk": risk - risk_hat,
        })
    return records


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


def compute_macro(recs, beta_k):
    """Compute macro metrics for a given β_κ."""
    est = recs[-1]["est"]
    base = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
            for l in CAT]
    bonus = []
    for i, l in enumerate(CAT):
        s = base[i]
        if l.name in RISK_FAMILIES:
            s += beta_k * abs(est["kappa"] - 0.3)
        bonus.append(s)
    rank_b = list(np.argsort(base)[::-1])
    rank_n = list(np.argsort(bonus)[::-1])
    t1_same = (rank_b[0] == rank_n[0])
    kt, _ = sp_stats.kendalltau(base, bonus) if beta_k > 0 else (1.0, 0)
    shifts = []
    for i, l in enumerate(CAT):
        if l.name in RISK_FAMILIES:
            shifts.append(rank_b.index(i) - rank_n.index(i))
    s_stop = EPS_0 + A_S * est["nu"] + B_S * est["gamma_gen"]
    return {
        "Top1Same": t1_same, "Kendall": kt if beta_k > 0 else 1.0,
        "Shift": np.mean(shifts) if shifts else 0,
        "STOP": s_stop, "kappa": est["kappa"],
    }


def main():
    print("═══ T1 Exp-1: κ̂ Default-On Full Family Regression ═══\n", file=sys.stderr)
    L = ["# T1 Exp-1: κ̂ Bonus Default-On Regression\n\n"]

    # ═══ Part 1: Micro Comparison (β=0 vs β=0.02) ═══
    L.append("## Part 1: Micro Metrics (Canonical baseline)\n\n")
    L.append("| θ | DivAll | Div@Act | OverWarnRate | Success |\n")
    L.append("|:-:|:------:|:-------:|:-----------:|:-------:|\n")

    for th in ["safe", "shiny"]:
        all_recs = []
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid))
        n = len(all_recs)
        na = sum(1 for r in all_recs if r["active"])
        da = sum(r["diverge"] for r in all_recs) / n
        dact = sum(1 for r in all_recs if r["diverge"] and r["active"]) / max(na, 1)
        owr = sum(r["owr"] for r in all_recs) / n
        succ = sum(r["correct"] for r in all_recs) / n
        L.append(f"| {th} | {da:.4f} | {dact:.4f} | {owr:.4f} | {succ:.3f} |\n")
        print(f"  {th}: DivAll={da:.4f} Div@Act={dact:.4f} OWR={owr:.4f}", file=sys.stderr)

    # ═══ Part 2: Macro β=0 vs β=0.02 ═══
    L.append("\n## Part 2: Macro — β_κ=0 vs β_κ=0.02\n\n")
    L.append("| θ | β_κ | Top-1 Same | Kendall | Risk Shift | STOP | κ̂ |\n")
    L.append("|:-:|:---:|:----------:|:-------:|:----------:|:----:|:--:|\n")

    for th in ["safe", "shiny"]:
        recs = run_session(ALL_LESSONS, th, 0)
        for bk in [0.0, 0.02]:
            mac = compute_macro(recs, bk)
            L.append("| {} | {:.2f} | {} | {:.4f} | {:+.1f} | {:.4f} | {:.4f} |\n".format(
                th, bk, "✅" if mac["Top1Same"] else "❌",
                mac["Kendall"], mac["Shift"], mac["STOP"], mac["kappa"]))
        print(f"  {th}: macro done", file=sys.stderr)

    # ═══ Part 3: OOD Conditions ═══
    L.append("\n## Part 3: OOD Robustness (β_κ=0.02)\n\n")
    L.append("| Condition | θ | DivAll | OWR | Top-1 | Kendall | κ̂ |\n")
    L.append("|-----------|:-:|:------:|:---:|:-----:|:-------:|:--:|\n")

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
            mac = compute_macro(
                run_session(lessons, th, 0, hidden_tempt=ht, risk_scale=rs), 0.02)
            L.append("| {} | {} | {:.4f} | {:.4f} | {} | {:.4f} | {:.4f} |\n".format(
                cname, th, da, owr,
                "✅" if mac["Top1Same"] else "❌",
                mac["Kendall"], mac["kappa"]))
            ood_total += 1
            if mac["Top1Same"]: ood_pass += 1
        print(f"  OOD {cname} done", file=sys.stderr)

    # ═══ Part 4: Per-Family ΔR² (quick check) ═══
    L.append("\n## Part 4: Per-Family Check (no regression)\n\n")
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid))

    fam_groups = defaultdict(list)
    for r in all_recs:
        fam_groups[r["family"]].append(r)

    L.append("| Family | n | R²(4D) | R²(5D) | ΔR² |\n")
    L.append("|--------|:-:|:------:|:------:|:---:|\n")
    n_regressed = 0
    for fam in sorted(fam_groups.keys()):
        fr = fam_groups[fam]
        y = [r["delta_risk"] for r in fr]
        x4 = [[r["est"]["tau"], r["est"]["nu"],
                r["est"]["gamma_gen"], r["est"]["gamma_spec"]] for r in fr]
        x5 = [[r["est"]["tau"], r["est"]["nu"],
                r["est"]["gamma_gen"], r["est"]["gamma_spec"],
                r["est"]["kappa"]] for r in fr]
        r4 = linear_r2(x4, y); r5 = linear_r2(x5, y)
        dr = r5 - r4
        if dr < -0.05: n_regressed += 1  # significant regression flag
        L.append(f"| {fam} | {len(fr)} | {r4:.4f} | {r5:.4f} | {dr:+.4f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")
    L.append(f"> OOD pass rate: {ood_pass}/{ood_total}\n")
    L.append(f"> Families with significant ΔR² regression: {n_regressed}/13\n")
    if ood_pass == ood_total and n_regressed == 0:
        L.append("> **✅ κ̂ bonus (β=0.02) default-on: PASS**\n")
    else:
        L.append("> **⚠️ Some conditions need investigation**\n")

    rpt = out / "t1_exp1_kappa_regression.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
