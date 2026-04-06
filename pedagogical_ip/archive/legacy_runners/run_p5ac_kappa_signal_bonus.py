"""P5-A: κ Family Signal Audit + P5-C: Macro κ Bonus.

Part 1: Per-family κ̂ trajectories, sign accuracy, δ_risk correlation
Part 2: Controlled macro TEACH bonus when κ̂ is low → risk lessons prioritized
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

# Classify risk-relevant lessons
RISK_FAMILIES = {"tic_rescue_heavy", "blind_activation_corridor",
                 "warn_symmetric_rescue"}


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_family_session(theta, seed, n_teach=20):
    """Session returning per-step family, κ̂, δ_risk, all dims."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    records = []

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

        action, dose, info = BCICTv4(agent_params=AP, use_dose=False).decide(
            sc, fb, lp, lib, scr, 2, m)
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b)
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a)
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
        delta_risk = risk - risk_hat

        kappa_before = observer.kappa_hat
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)
        kappa_after = observer.kappa_hat
        delta_kappa = kappa_after - kappa_before
        est = observer.get_estimate()

        # Sign accuracy: did κ̂ move in the same direction as δ_risk?
        if abs(delta_risk) > 0.01 and abs(delta_kappa) > 1e-6:
            sign_match = (delta_kappa > 0) == (delta_risk > 0)
        else:
            sign_match = None  # no meaningful signal

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "risk": risk, "risk_hat": risk_hat,
            "delta_risk": delta_risk, "delta_kappa": delta_kappa,
            "kappa": kappa_after, "sign_match": sign_match,
            "est": est, "correct": correct,
            "is_risk_family": les.name in RISK_FAMILIES,
        })
    return records


def main():
    print("═══ P5-A/C: κ Signal + Macro Bonus ═══\n", file=sys.stderr)
    L = ["# P5-A: κ Family Signal Audit + P5-C: Macro Bonus\n\n"]

    # ═══ Part 1: Family Signal ════════════════════════════
    L.append("## Part 1: Per-Family κ̂ Signal\n\n")
    print("Part 1: Family signal...", file=sys.stderr)

    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_family_session(th, sid))

    # Per-family summary
    fams = {}
    for r in all_recs:
        fams.setdefault(r["family"], []).append(r)

    L.append("| Family | n | Mean δ_risk | Mean Δκ̂ | Sign Acc | "
             "Corr(κ̂,δ) | κ̂ range |\n")
    L.append("|--------|:-:|:----------:|:-------:|:--------:|"
             ":----------:|:--------:|\n")
    for fam in sorted(fams.keys()):
        fr = fams[fam]; n = len(fr)
        dr = [r["delta_risk"] for r in fr]
        dk = [r["delta_kappa"] for r in fr]
        sigs = [r["sign_match"] for r in fr if r["sign_match"] is not None]
        sig_acc = np.mean(sigs) if sigs else float('nan')
        kaps = [r["kappa"] for r in fr]
        if len(set(dr)) > 1 and len(set(dk)) > 1:
            corr = sp_stats.pearsonr(dr, dk)[0]
        else:
            corr = float('nan')
        L.append("| {} | {} | {:.4f} | {:.6f} | {:.1f}% | {:.4f} | "
                 "[{:.4f},{:.4f}] |\n".format(
            fam, n, np.mean(dr), np.mean(dk), 100*sig_acc, corr,
            np.min(kaps), np.max(kaps)))

    # Overall
    all_dr = [r["delta_risk"] for r in all_recs]
    all_dk = [r["delta_kappa"] for r in all_recs]
    all_sig = [r["sign_match"] for r in all_recs if r["sign_match"] is not None]
    all_corr = sp_stats.pearsonr(all_dr, all_dk)[0]
    L.append(f"\n**Overall: Sign Acc = {100*np.mean(all_sig):.1f}%, "
             f"Corr(κ̂,δ_risk) = {all_corr:.4f}**\n\n")

    # Collinearity recheck
    L.append("### Collinearity (5D)\n\n")
    L.append("| Dim | Corr(κ̂, dim) |\n")
    L.append("|-----|:------------:|\n")
    kaps = [r["est"]["kappa"] for r in all_recs]
    for dim in ["tau", "nu", "gamma_gen", "gamma_spec"]:
        vals = [r["est"][dim] for r in all_recs]
        c = sp_stats.pearsonr(kaps, vals)[0] if len(set(vals)) > 1 else 0
        L.append(f"| {dim} | {c:.4f} |\n")

    # ═══ Part 2: Macro κ Bonus ═══════════════════════════
    L.append("\n## Part 2: Macro κ Bonus — Risk Lesson Priority\n\n")
    print("\nPart 2: Macro bonus...", file=sys.stderr)

    CAT = list(LESSON_CATALOG_V2)
    EPS_0 = 0.30; A_S = 0.15; B_S = 0.10
    beta_kappa = 0.05

    L.append("### Lesson Ranking: Baseline vs κ-Bonus\n\n")
    L.append("| θ | Baseline Top-3 | κ-Bonus Top-3 | Risk-lesson Rank Shift |\n")
    L.append("|:-:|:--------------:|:-------------:|:----------------------:|\n")

    for th in ["safe", "shiny"]:
        # Get final state from one representative session
        recs = run_family_session(th, 0)
        rl = recs[-1]; est = rl["est"]

        # Baseline scores (3D)
        scores_base = []
        for l in CAT:
            g = np.mean(l.gain)
            s = g * (1 - est["nu"]) * (1 - est["gamma_gen"]) * est["tau"]
            scores_base.append(s)

        # κ-bonus scores
        scores_bonus = []
        for i, l in enumerate(CAT):
            s = scores_base[i]
            if l.name in RISK_FAMILIES:
                s += beta_kappa * (1 - est["kappa"])
            scores_bonus.append(s)

        rank_base = np.argsort(scores_base)[::-1]
        rank_bonus = np.argsort(scores_bonus)[::-1]

        top3_base = [CAT[i].name for i in rank_base[:3]]
        top3_bonus = [CAT[i].name for i in rank_bonus[:3]]

        # Rank shift for risk families
        shifts = []
        for i, l in enumerate(CAT):
            if l.name in RISK_FAMILIES:
                r_base = list(rank_base).index(i)
                r_bonus = list(rank_bonus).index(i)
                shifts.append(r_base - r_bonus)  # positive = moved up

        L.append("| {} | {} | {} | {} |\n".format(
            th, ", ".join(top3_base[:3]), ", ".join(top3_bonus[:3]),
            f"avg +{np.mean(shifts):.1f}" if shifts else "N/A"))

    # Multi-seed TEACH top-1 stability
    L.append("\n### TEACH Top-1 Stability Under κ-Bonus\n\n")
    L.append("| θ | Baseline Top-1 Agree | κ-Bonus Top-1 Agree | Kendall(base) | "
             "Kendall(bonus) |\n")
    L.append("|:-:|:--------------------:|:-------------------:|:-------------:|"
             ":--------------:|\n")
    for th in ["safe", "shiny"]:
        t1_base_matches = 0; t1_bonus_matches = 0; n = 0
        kt_bases = []; kt_bonuses = []
        for sid in range(NS):
            recs = run_family_session(th, sid)
            rl = recs[-1]; est = rl["est"]; mt = rl["est"]
            # Oracle scores (from true m)
            so = [np.mean(l.gain)*(1-rl["est"]["nu"])*(1-rl["est"]["gamma_gen"]
                  )*rl["est"]["tau"] for l in CAT]
            # Base
            sb = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
                  for l in CAT]
            # Bonus
            sk = []
            for i, l in enumerate(CAT):
                s = sb[i]
                if l.name in RISK_FAMILIES:
                    s += beta_kappa * (1 - est["kappa"])
                sk.append(s)
            n += 1
            if np.argsort(so)[-1] == np.argsort(sb)[-1]: t1_base_matches += 1
            if np.argsort(so)[-1] == np.argsort(sk)[-1]: t1_bonus_matches += 1
            kt_b, _ = sp_stats.kendalltau(so, sb)
            kt_k, _ = sp_stats.kendalltau(so, sk)
            kt_bases.append(kt_b); kt_bonuses.append(kt_k)
        L.append("| {} | {:.0f}% | {:.0f}% | {:.4f} | {:.4f} |\n".format(
            th, 100*t1_base_matches/n, 100*t1_bonus_matches/n,
            np.mean(kt_bases), np.mean(kt_bonuses)))

    # ═══ Verdict ═════════════════════════════════════════
    L.append("\n## Verdict\n\n")
    L.append(f"> **κ̂ signal audit**: Overall sign accuracy = "
             f"{100*np.mean(all_sig):.1f}%, Corr(κ̂,δ_risk) = {all_corr:.4f}\n\n")
    if all_corr > 0.1:
        L.append("> **κ̂ is a responsive risk-calibration state.** "
                 "It moves in the correct direction relative to risk prediction error.\n")
    elif all_corr > 0.0:
        L.append("> **κ̂ shows weak positive response.** "
                 "May need stronger risk families.\n")
    else:
        L.append("> **κ̂ does not clearly track δ_risk.** Investigate.\n")

    rpt = out / "p5ac_kappa_signal_bonus.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
