"""Track A — Final Enhancement Classification.

Part 1: B1/B3 Redundancy Analysis
  - Corr(ΔB1, ΔB3) across lessons and states
  - Whether B1 switches on uncertainty-heavy lessons distinct from B3

Part 2: B2 Micro-Family Validation
  - U-Safe / U-Danger paired families
  - SDR_micro, first-visit, second-visit metrics
"""
import sys, os
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice, compute_factored_utility,
)
from src.agents.behavior_probes import all_probes
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13, ControllerV13Config
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.family_prior import FamilyPrior
from src.curriculum.mastery_model import MasteryModel

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 20


def make_controller(theta, eig=False, zpd=False):
    cn = [l.name for l in LESSON_CATALOG_V2]
    cfg = ControllerV13Config(total_budget=8.0, risk_budget_mode="theta")
    fp = FamilyPrior(enabled=True, use_saturation=True, use_rep_penalty=False)
    c = CurriculumControllerV13(
        cfg=cfg, theta=theta, family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn, theta=theta))
    c.use_eig_uncertainty = eig
    c.use_zpd_feature = zpd
    return c


# ─── Part 1: B1/B3 Redundancy ─────────────────────────────

def redundancy_analysis():
    """Score all lessons under base / B1 / B3 at multiple mastery states."""
    L = ["# B1 / B3 Redundancy Analysis\n\n"]
    L.append("> Compare Δ^B1(x,ℓ) vs Δ^B3(x,ℓ) across mastery states and lessons\n\n")

    mastery_states = [
        {"label": "low", "vals": {p: 0.2 for p in PROBE_NAMES}},
        {"label": "mid", "vals": {p: 0.5 for p in PROBE_NAMES}},
        {"label": "high", "vals": {p: 0.8 for p in PROBE_NAMES}},
        {"label": "mixed", "vals": {"RC": 0.7, "TR": 0.3, "EP": 0.5, "VA": 0.6, "IA": 0.2}},
    ]

    all_delta_b1 = []; all_delta_b3 = []
    switch_b1_only = 0; switch_b3_only = 0; switch_both = 0; switch_same = 0
    total_comparisons = 0

    L.append("## Score Deltas by Mastery State\n\n")

    for ms in mastery_states:
        L.append(f"### Mastery = {ms['label']}\n\n")
        L.append("| Lesson | J_base | J_B1 | J_B3 | Δ^B1 | Δ^B3 | argmax_changed? |\n")
        L.append("|--------|:------:|:----:|:----:|:----:|:----:|:---:|\n")

        for th in ["safe", "shiny"]:
            c_base = make_controller(th)
            c_b1 = make_controller(th, eig=True)
            c_b3 = make_controller(th, zpd=True)

            m = FactoredInternalizationState(); m.snapshot()
            # Score each lesson
            scores_base = {}; scores_b1 = {}; scores_b3 = {}
            for les in LESSON_CATALOG_V2:
                j_base, _ = c_base._score_lesson(les, ms["vals"], m)
                j_b1, _ = c_b1._score_lesson(les, ms["vals"], m)
                j_b3, _ = c_b3._score_lesson(les, ms["vals"], m)
                scores_base[les.name] = j_base
                scores_b1[les.name] = j_b1
                scores_b3[les.name] = j_b3
                d_b1 = j_b1 - j_base; d_b3 = j_b3 - j_base
                all_delta_b1.append(d_b1); all_delta_b3.append(d_b3)
                if th == "safe":  # log only safe to keep table readable
                    switched = "—"
                    if abs(d_b1) > 1e-4 or abs(d_b3) > 1e-4:
                        switched = "✓"
                    L.append(f"| {les.name} | {j_base:.4f} | {j_b1:.4f} | {j_b3:.4f} | {d_b1:+.4f} | {d_b3:+.4f} | {switched} |\n")

            # Check argmax changes
            base_best = max(scores_base, key=scores_base.get)
            b1_best = max(scores_b1, key=scores_b1.get)
            b3_best = max(scores_b3, key=scores_b3.get)
            total_comparisons += 1
            if b1_best != base_best and b3_best != base_best:
                if b1_best == b3_best:
                    switch_same += 1
                else:
                    switch_both += 1
            elif b1_best != base_best:
                switch_b1_only += 1
            elif b3_best != base_best:
                switch_b3_only += 1

    # Correlation
    if len(all_delta_b1) > 2:
        corr = np.corrcoef(all_delta_b1, all_delta_b3)[0, 1]
    else:
        corr = 0.0

    L.append("\n## Summary\n\n")
    L.append(f"- **Corr(Δ^B1, Δ^B3)** = {corr:.4f}\n")
    L.append(f"- B1 only switches argmax: {switch_b1_only}/{total_comparisons}\n")
    L.append(f"- B3 only switches argmax: {switch_b3_only}/{total_comparisons}\n")
    L.append(f"- Both switch to same lesson: {switch_same}/{total_comparisons}\n")
    L.append(f"- Both switch to different lessons: {switch_both}/{total_comparisons}\n\n")

    if corr > 0.9:
        L.append("> **Verdict**: B1 and B3 are highly correlated → likely acting on same ranking axis → **keep B3 only**\n\n")
    elif corr > 0.5:
        L.append("> **Verdict**: B1 and B3 have moderate correlation → partially overlapping → **keep B3 primary, B1 optional**\n\n")
    else:
        L.append("> **Verdict**: B1 and B3 have low correlation → complementary → **keep both as distinct optional modules**\n\n")

    return L


# ─── Part 2: B2 Micro-Family Validation ──────────────────

def b2_micro_family():
    """Test B2 on paired unknown-risk families."""
    L = ["# B2 Micro-Family Validation\n\n"]
    L.append("> U-Safe: unknown path actually safe | U-Danger: unknown path actually dangerous\n\n")

    rng = np.random.default_rng(42)
    NS_micro = 50

    # Build matched branch pairs
    # U-Safe: unknown branch has risk_unc=0.8 but risk_penalty=0.05
    # U-Danger: unknown branch has risk_unc=0.8 but risk_penalty=0.6

    scenarios = {
        "U-Safe": {"known_rp": 0.15, "unknown_rp": 0.05, "unknown_unc": 0.8},
        "U-Danger": {"known_rp": 0.15, "unknown_rp": 0.60, "unknown_unc": 0.8},
    }

    L.append("| Scenario | Metric | Base | B2 | Δ |\n")
    L.append("|----------|--------|:----:|:--:|:-:|\n")

    sdr_micro_total = 0; sdr_micro_n = 0

    for sc_name, sc_params in scenarios.items():
        choices_base = []; choices_b2 = []
        utils_base = []; utils_b2 = []

        for trial in range(NS_micro):
            m = FactoredInternalizationState(); m.snapshot()
            # Vary kappa slightly across trials
            m.kappa = 0.5 + rng.random() * 1.5

            b_known = BranchAttributes(
                safety_score=0.6, risk_penalty=sc_params["known_rp"],
                temptation_score=0.2)
            b_unknown = BranchAttributes(
                safety_score=0.4, risk_penalty=sc_params["unknown_rp"],
                temptation_score=0.4 + rng.random() * 0.3)

            # Base (no B2)
            u_base_known = compute_factored_utility(b_known, "safe", m, AP)
            u_base_unknown = compute_factored_utility(b_unknown, "safe", m, AP)
            choice_base = 0 if u_base_known > u_base_unknown else 1
            choices_base.append(choice_base)
            utils_base.append((u_base_known, u_base_unknown))

            # B2 (with risk uncertainty)
            u_b2_known = compute_factored_utility(
                b_known, "safe", m, AP,
                risk_unc=0.1, use_epistemic_risk=True)  # known path: low unc
            u_b2_unknown = compute_factored_utility(
                b_unknown, "safe", m, AP,
                risk_unc=sc_params["unknown_unc"], use_epistemic_risk=True)
            choice_b2 = 0 if u_b2_known > u_b2_unknown else 1
            choices_b2.append(choice_b2)
            utils_b2.append((u_b2_known, u_b2_unknown))

            if choice_base != choice_b2:
                sdr_micro_total += 1
            sdr_micro_n += 1

        # Compute metrics
        # P_enter_unknown
        p_enter_base = sum(1 for c in choices_base if c == 1) / NS_micro
        p_enter_b2 = sum(1 for c in choices_b2 if c == 1) / NS_micro
        mean_du_unknown = np.mean([u[1] - b[1] for u, b in zip(utils_b2, utils_base)])

        L.append(f"| {sc_name} | P(enter unknown) | {p_enter_base:.2f} | {p_enter_b2:.2f} | {p_enter_b2 - p_enter_base:+.2f} |\n")
        L.append(f"| {sc_name} | mean ΔU_unknown | — | — | {mean_du_unknown:+.4f} |\n")

    sdr_micro = sdr_micro_total / max(sdr_micro_n, 1)
    L.append(f"\n- **SDR_micro** = {sdr_micro:.3f} ({sdr_micro_total}/{sdr_micro_n})\n\n")

    # Second-visit correction test
    L.append("## Second-Visit Correction (U-Danger)\n\n")
    L.append("| Visit | P(avoid danger) Base | P(avoid danger) B2 |\n")
    L.append("|-------|:-:|:-:|\n")

    for visit, unc_after in [("1st", 0.8), ("2nd", 0.15)]:
        avoid_base = 0; avoid_b2 = 0
        for trial in range(NS_micro):
            m = FactoredInternalizationState(); m.snapshot()
            m.kappa = 0.5 + rng.random() * 1.5
            b_known = BranchAttributes(safety_score=0.6, risk_penalty=0.15, temptation_score=0.2)
            b_danger = BranchAttributes(safety_score=0.4, risk_penalty=0.60, temptation_score=0.5)

            u_base_k = compute_factored_utility(b_known, "safe", m, AP)
            u_base_d = compute_factored_utility(b_danger, "safe", m, AP)
            if u_base_k > u_base_d: avoid_base += 1

            u_b2_k = compute_factored_utility(b_known, "safe", m, AP,
                risk_unc=0.1, use_epistemic_risk=True)
            u_b2_d = compute_factored_utility(b_danger, "safe", m, AP,
                risk_unc=unc_after, use_epistemic_risk=True)
            if u_b2_k > u_b2_d: avoid_b2 += 1

        L.append(f"| {visit} (unc={unc_after}) | {avoid_base/NS_micro:.2f} | {avoid_b2/NS_micro:.2f} |\n")

    # Verdict
    L.append("\n## Verdict\n\n")

    return L


def main():
    print("═══ Track A: Final Enhancement Classification ═══\n", file=sys.stderr)

    # Part 1: Redundancy
    print("Part 1: B1/B3 Redundancy...", file=sys.stderr)
    r1 = redundancy_analysis()

    # Part 2: B2 Micro
    print("Part 2: B2 Micro Family...", file=sys.stderr)
    r2 = b2_micro_family()

    # Combine
    lines = ["# Enhancement Final Classification Report\n\n"]
    lines.append("> Phase 7 Track A: close B1/B2/B3 story\n\n")
    lines.extend(r1)
    lines.append("\n---\n\n")
    lines.extend(r2)

    rpt = out / "enhancement_final_classification.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
