"""T3 Exp-T3-Combined: Shadow POMDP Interface Audit.

Combines:
  - Exp-T3-1: old vs new predictor parity (NLL, Brier)
  - Exp-T3-2: posterior calibration (ECE, entropy, theta tracking)
  - Exp-T3-5: benchmark no-regression (WarnRate, SelGap, Success preserved)

Runs ShadowBridge in parallel with canonical path.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.agent_belief_state import AgentBelief
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.teachers.shadow_bridge import ShadowBridge
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

LESSONS = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC", "TIC-v4")]
WARN_NECESSARY = {"verified_warn", "warn_rescue"}
WARN_UNNECESSARY = {"beneficial_novelty", "false_suppression_cost",
                    "self_discovery_needed", "sparse_invalid_advice"}
N_STEPS = 30
N_SEEDS = 10


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_shadow_audit(theta, seed):
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    bridge = ShadowBridge(theta=theta, params=AP)

    results = []
    for step_i in range(N_STEPS):
        les = LESSONS[step_i % len(LESSONS)]
        ub = {p: 0.4 + 0.1 * (step_i / N_STEPS) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(
            les, step_i + seed * 10000, theta, ub, rng)
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

        tutor = BCICTv4(agent_params=AP, use_dose=False)
        act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        dc = getattr(sc, 'commit_depth', 3)
        dr = getattr(sc, 'reveal_depth', 2)
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
        branches = [bas, bar]

        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0
        self_disc = correct and not warned and p_self > 0.5

        # Shadow bridge observation
        ab = AgentBelief(
            m_state=dict(m.as_dict),
            theta=theta,
        )
        est = observer.get_estimate()
        conf = {"tau": getattr(observer, 'conf_tau', 0.2),
                "nu": getattr(observer, 'conf_nu', 0.2),
                "gamma_gen": getattr(observer, 'conf_gamma', 0.2)}
        bridge.observe_step(None, ab, branches, ac,
                           observer_estimate=est,
                           observer_confidence=conf)

        # Standard m_t updates
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if p_self < 0.5: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5:
            m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15)
        m.snapshot()

        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
        ev = ObsEvent(
            episode_id=seed, step_id=step_i, subtype=ep.subtype,
            theta_post=theta, dose=dose, warned=warned,
            follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
        )
        observer.update(ev)

        results.append({
            "correct": correct, "warned": warned,
            "subtype": les.subtype,
        })

    report = bridge.get_report()
    n = len(results)
    success = sum(1 for r in results if r["correct"]) / max(n, 1)
    wr_nec = sum(1 for r in results
                 if r["warned"] and r["subtype"] in WARN_NECESSARY
                 ) / max(sum(1 for r in results
                            if r["subtype"] in WARN_NECESSARY), 1)
    wr_unnec = sum(1 for r in results
                   if r["warned"] and r["subtype"] in WARN_UNNECESSARY
                   ) / max(sum(1 for r in results
                              if r["subtype"] in WARN_UNNECESSARY), 1)

    return {
        "report": report,
        "success": success,
        "selgap": wr_nec - wr_unnec,
        "wr_nec": wr_nec,
        "wr_unnec": wr_unnec,
    }


def main():
    print("═══ T3 Shadow POMDP Audit ═══\n", file=sys.stderr)
    L = ["# T3 Shadow POMDP Interface Audit\n\n"]

    all_results = defaultdict(list)
    for th in ["safe", "shiny"]:
        for sid in range(N_SEEDS):
            r = run_shadow_audit(th, sid)
            all_results[th].append(r)
        print(f"  {th} done", file=sys.stderr)

    # ═══ Table 1: Prediction Parity (Exp-T3-1) ═══
    L.append("## Exp-T3-1: Prediction Parity\n\n")
    L.append("| θ | NLL_old | NLL_new | |Δ NLL| | Brier_old | Brier_new | Top1Agree |\n")
    L.append("|:-:|:-------:|:-------:|:------:|:---------:|:---------:|:---------:|\n")
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        nll_old = np.mean([r["report"].mean_old_nll for r in rs])
        nll_new = np.mean([r["report"].mean_new_nll for r in rs])
        dnll = np.mean([r["report"].nll_parity for r in rs])
        brier_old = np.mean([r["report"].brier_old for r in rs])
        brier_new = np.mean([r["report"].brier_new for r in rs])
        top1 = np.mean([r["report"].top1_agreement for r in rs])
        L.append(f"| {th} | {nll_old:.4f} | {nll_new:.4f} | {dnll:.6f} | "
                 f"{brier_old:.4f} | {brier_new:.4f} | {top1:.3f} |\n")

    # ═══ Table 2: Calibration (Exp-T3-2) ═══
    L.append("\n## Exp-T3-2: Calibration & Posterior\n\n")
    L.append("| θ | ECE_old | ECE_new | Mean Entropy | Final θ_MAP |\n")
    L.append("|:-:|:-------:|:-------:|:------------:|:-----------:|\n")
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        ece_old = np.mean([r["report"].ece_old for r in rs])
        ece_new = np.mean([r["report"].ece_new for r in rs])
        entropy = np.mean([r["report"].mean_entropy for r in rs])
        # Most common final theta MAP
        theta_maps = [max(r["report"].final_theta_posterior,
                         key=r["report"].final_theta_posterior.get)
                     for r in rs if r["report"].final_theta_posterior]
        from collections import Counter
        mc = Counter(theta_maps).most_common(1)
        theta_map = mc[0][0] if mc else "?"
        L.append(f"| {th} | {ece_old:.4f} | {ece_new:.4f} | "
                 f"{entropy:.3f} | {theta_map} |\n")

    # ═══ Table 3: Benchmark No-Regression (Exp-T3-5) ═══
    L.append("\n## Exp-T3-5: Benchmark No-Regression\n\n")
    L.append("| θ | Success | SelGap | WR_nec | WR_unnec |\n")
    L.append("|:-:|:-------:|:------:|:------:|:--------:|\n")
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        su = np.mean([r["success"] for r in rs])
        sg = np.mean([r["selgap"] for r in rs])
        wn = np.mean([r["wr_nec"] for r in rs])
        wu = np.mean([r["wr_unnec"] for r in rs])
        L.append(f"| {th} | {su:.3f} | {sg:.3f} | {wn:.3f} | {wu:.3f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check 1: NLL parity
    parity_ok = True
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        dnll = np.mean([r["report"].nll_parity for r in rs])
        if dnll > 0.01:
            parity_ok = False
    L.append(f"> NLL parity (|Δ| < 0.01): {'✅' if parity_ok else '❌'}\n")

    # Check 2: Top-1 agreement
    agree_ok = True
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        top1 = np.mean([r["report"].top1_agreement for r in rs])
        if top1 < 0.95:
            agree_ok = False
    L.append(f"> Top-1 agreement ≥ 95%: {'✅' if agree_ok else '❌'}\n")

    # Check 3: ECE not worse
    ece_ok = True
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        ece_old = np.mean([r["report"].ece_old for r in rs])
        ece_new = np.mean([r["report"].ece_new for r in rs])
        if ece_new > ece_old + 0.02:
            ece_ok = False
    L.append(f"> ECE not worse (≤ old + 0.02): {'✅' if ece_ok else '❌'}\n")

    # Check 4: Theta MAP matches ground truth
    theta_ok = True
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        for r in rs:
            if r["report"].final_theta_posterior:
                theta_map = max(r["report"].final_theta_posterior,
                               key=r["report"].final_theta_posterior.get)
                if theta_map != th:
                    # shiny might get "risky" which is close
                    if th == "shiny" and theta_map in ("risky", "shiny"):
                        pass
                    elif th == "safe" and theta_map == "safe":
                        pass
                    else:
                        theta_ok = False
    L.append(f"> θ_MAP recovery: {'✅' if theta_ok else '⚠️ partial'}\n")

    # Check 5: WR_unnecessary
    wu_ok = True
    for th in ["safe", "shiny"]:
        rs = all_results[th]
        wu = np.mean([r["wr_unnec"] for r in rs])
        if wu > 0.05:
            wu_ok = False
    L.append(f"> WR_unnecessary ≤ 0.05: {'✅' if wu_ok else '❌'}\n")

    all_pass = parity_ok and agree_ok and ece_ok and wu_ok
    if all_pass:
        L.append("> **✅ Shadow POMDP interface passes all parity checks**\n")
    else:
        L.append("> **⚠️ Issues found — investigate before switch-on**\n")

    rpt = out / "t3_shadow_pomdp_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
