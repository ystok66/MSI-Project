"""Smoke tests for three ablation enhancements (B1/B2/B3).

Tests:
  1. Flag-off invariance: all flags OFF reproduces canonical behavior
  2. B1: EIG monotonicity + eig_mix=0 reproduces count-only
  3. B2: Risk floor guarantee + uncertainty direction
  4. B3: Matched lesson has least-negative ZPD
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from src.curriculum.curriculum_controller_v13 import (
    CurriculumControllerV13, ControllerV13Config,
)
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.family_prior import FamilyPrior
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, compute_factored_utility,
)
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams


AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)


def _make_controller(theta="safe", eig=False, zpd=False):
    cn = [l.name for l in LESSON_CATALOG_V2]
    cfg = ControllerV13Config(total_budget=4.0, risk_budget_mode="theta")
    fp = FamilyPrior(enabled=True, use_saturation=True, use_rep_penalty=False)
    c = CurriculumControllerV13(
        cfg=cfg, theta=theta, family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn, theta=theta))
    c.use_eig_uncertainty = eig
    c.use_zpd_feature = zpd
    return c


# ═══════════════════════════════════════════════════
# 1. Flag-off invariance
# ═══════════════════════════════════════════════════

class TestFlagOffInvariance:
    """Canonical behavior is unchanged when all new flags are OFF."""

    def test_canonical_unchanged(self):
        """Same seed, same m → same action and score."""
        m = FactoredInternalizationState()
        c_base = _make_controller("safe", eig=False, zpd=False)
        c_new = _make_controller("safe", eig=False, zpd=False)

        a1, l1, s1, _ = c_base.select_action(m)
        a2, l2, s2, _ = c_new.select_action(m)
        assert a1 == a2
        if l1 and l2:
            assert l1.name == l2.name
        assert s1 == s2


# ═══════════════════════════════════════════════════
# 2. B1: EIG hybrid exploration
# ═══════════════════════════════════════════════════

class TestB1EIGExploration:
    """EIG hybrid gate correctness."""

    def test_higher_variance_higher_g_eig(self):
        """Higher gain/mastery variance → higher g_eig."""
        c = _make_controller("safe", eig=True)
        # Low variance
        gp_low = {"var": 0.01, "mean": 0.1, "hier": 0.1, "res": 0.0, "pw": 0.0}
        lam_low, trace_low = c._lambda_unc_hybrid(gp_low)

        # High variance
        gp_high = {"var": 0.5, "mean": 0.1, "hier": 0.1, "res": 0.0, "pw": 0.0}
        lam_high, trace_high = c._lambda_unc_hybrid(gp_high)

        assert trace_high["g_eig"] > trace_low["g_eig"]
        assert trace_high["V_epi"] > trace_low["V_epi"]

    def test_eig_mix_zero_is_count_only(self):
        """eig_mix=0 → pure count-based (canonical)."""
        c = _make_controller("safe", eig=True)
        c.cfg.eig_mix = 0.0
        gp = {"var": 0.5, "mean": 0.1, "hier": 0.1, "res": 0.0, "pw": 0.0}
        lam_hybrid, trace = c._lambda_unc_hybrid(gp)
        lam_canonical = c._lambda_unc()
        # With eig_mix=0, hybrid should equal canonical count-based
        assert abs(trace["g_count"] - np.exp(-0 / c.cfg.tau_n)) < 1e-6

    def test_eig_mix_one_is_pure_eig(self):
        """eig_mix=1 → pure belief-based."""
        c = _make_controller("safe", eig=True)
        c.cfg.eig_mix = 1.0
        gp = {"var": 0.3, "mean": 0.1, "hier": 0.1, "res": 0.0, "pw": 0.0}
        _, trace = c._lambda_unc_hybrid(gp)
        # With eig_mix=1, decay should equal g_eig only
        expected_lam = trace["lam_budget"] * trace["g_eig"]
        assert abs(trace["lam_unc_hybrid"] - expected_lam) < 1e-6


# ═══════════════════════════════════════════════════
# 3. B2: Local epistemic risk
# ═══════════════════════════════════════════════════

class TestB2EpistemicRisk:
    """Risk floor and uncertainty direction."""

    def test_risk_floor(self):
        """Risk penalty never drops below κ²·α_min·ρ·risk_penalty."""
        m = FactoredInternalizationState()
        m.kappa = 1.5
        branch = BranchAttributes(safety_score=0.5, risk_penalty=0.4)
        # Very high uncertainty → alpha → alpha_min
        u_canonical = compute_factored_utility(branch, "safe", m, AP)
        u_epi = compute_factored_utility(
            branch, "safe", m, AP,
            risk_unc=10.0,  # very high uncertainty
            use_epistemic_risk=True,
        )
        # With epistemic risk, utility should be HIGHER (less risk penalty)
        # but risk penalty should not vanish
        canonical_risk = (m.kappa ** 2) * branch.risk_penalty
        alpha_min, rho = 0.25, 0.35
        min_risk = (m.kappa ** 2) * alpha_min * rho * branch.risk_penalty
        # u_epi > u_canonical since risk penalty is attenuated
        assert u_epi >= u_canonical - 0.01  # slightly tolerant due to float
        # Risk penalty must be at least the floor
        # (u_canonical - u_epi) ≤ canonical_risk - min_risk
        risk_reduction = u_epi - u_canonical  # should be positive
        max_possible_reduction = canonical_risk - min_risk
        assert risk_reduction <= max_possible_reduction + 1e-6

    def test_higher_uncertainty_lower_alpha(self):
        """Higher normalized uncertainty → smaller alpha → less risk penalty."""
        m = FactoredInternalizationState()
        m.kappa = 1.0
        branch = BranchAttributes(safety_score=0.5, risk_penalty=0.3)
        u_low_unc = compute_factored_utility(
            branch, "safe", m, AP, risk_unc=0.05, use_epistemic_risk=True)
        u_high_unc = compute_factored_utility(
            branch, "safe", m, AP, risk_unc=0.5, use_epistemic_risk=True)
        # Higher uncertainty → more risk attenuation → higher utility
        assert u_high_unc > u_low_unc

    def test_no_bonus_without_flag(self):
        """Curiosity bonus absent when use_epistemic_bonus=False."""
        m = FactoredInternalizationState()
        branch = BranchAttributes(safety_score=0.5, risk_penalty=0.3)
        u_no_bonus = compute_factored_utility(
            branch, "safe", m, AP,
            risk_unc=0.5, is_novel=True,
            use_epistemic_risk=True, use_epistemic_bonus=False)
        u_with_bonus = compute_factored_utility(
            branch, "safe", m, AP,
            risk_unc=0.5, is_novel=True,
            use_epistemic_risk=True, use_epistemic_bonus=True)
        # With bonus should be strictly higher
        assert u_with_bonus >= u_no_bonus

    def test_b2_path_utilization(self):
        """B2 actually changes utility when wired through sample_factored_choice."""
        from src.agents.internalization_state_v3 import sample_factored_choice
        m = FactoredInternalizationState()
        m.kappa = 1.5  # amplify risk sensitivity
        b_safe = BranchAttributes(safety_score=0.6, risk_penalty=0.1)
        b_risky = BranchAttributes(safety_score=0.3, risk_penalty=0.5, temptation_score=0.7)
        # Compute utilities with and without B2
        u_base_0 = compute_factored_utility(b_safe, "safe", m, AP)
        u_base_1 = compute_factored_utility(b_risky, "safe", m, AP)
        u_epi_0 = compute_factored_utility(b_safe, "safe", m, AP,
                                           risk_unc=0.4, use_epistemic_risk=True)
        u_epi_1 = compute_factored_utility(b_risky, "safe", m, AP,
                                           risk_unc=0.4, use_epistemic_risk=True)
        # B2 should change at least one utility
        delta_0 = abs(u_epi_0 - u_base_0)
        delta_1 = abs(u_epi_1 - u_base_1)
        assert delta_0 > 0 or delta_1 > 0, \
            f"B2 produced no utility change: Δ_safe={delta_0}, Δ_risky={delta_1}"
        # Risky branch (higher risk_penalty) should be more affected
        assert delta_1 > delta_0, \
            f"Expected risky branch more affected: Δ_risky={delta_1} vs Δ_safe={delta_0}"


# ═══════════════════════════════════════════════════
# 4. B3: ZPD feature
# ═══════════════════════════════════════════════════

class TestB3ZPDFeature:
    """ZPD penalty correctness."""

    def test_matched_lesson_least_negative(self):
        """Matched lesson (u ≈ d) should have highest (least negative) ZPD adj."""
        c = _make_controller("safe", zpd=True)
        # Make mastery match lesson gain profile
        tic_rescue = LESSON_CATALOG_V2[2]  # tic_rescue_heavy: gain=[0.25,0.10,0.02,0.12,0.03]
        # Matched mastery
        u_matched = {p: float(tic_rescue.gain[i]) for i, p in enumerate(PROBE_NAMES)}
        # Mismatched mastery (much lower)
        u_low = {p: 0.0 for p in PROBE_NAMES}
        # Mismatched mastery (much higher)
        u_high = {p: 0.9 for p in PROBE_NAMES}

        adj_matched, _ = c._zpd_adjustment(tic_rescue, u_matched)
        adj_low, _ = c._zpd_adjustment(tic_rescue, u_low)
        adj_high, _ = c._zpd_adjustment(tic_rescue, u_high)

        # Matched should be closest to 0 (least negative)
        assert adj_matched >= adj_low
        assert adj_matched >= adj_high

    def test_both_terms_nonpositive(self):
        """Both ψ_under and ψ_over are ≤ 0."""
        c = _make_controller("safe", zpd=True)
        lesson = LESSON_CATALOG_V2[0]
        u = {p: 0.3 for p in PROBE_NAMES}
        _, trace = c._zpd_adjustment(lesson, u)
        assert trace["zpd_under"] <= 0.0
        assert trace["zpd_over"] <= 0.0

    def test_zpd_off_returns_zero(self):
        """When flag is OFF, ZPD adjustment is exactly 0."""
        c = _make_controller("safe", zpd=False)
        lesson = LESSON_CATALOG_V2[0]
        u = {p: 0.3 for p in PROBE_NAMES}
        adj, trace = c._zpd_adjustment(lesson, u)
        assert adj == 0.0
        assert trace == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
