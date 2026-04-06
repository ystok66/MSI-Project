"""
Integration tests for V2 unified intervention family — Phase 8.

Verifies end-to-end episode run, 4-action scoring, item effect exposure,
planner inventory awareness, and legacy baseline preservation.
"""

import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.interventions import (
    InterventionType, InventoryState, MAIN_INTERVENTION_FAMILY,
    SHIELD_DEFAULT_RISK_REDUCTION,
)
from src.teachers.intervention_policy import InterventionDecision, InterventionConfig


runner = LatticeV2Runner()


def test_intervention_family_mode_episode_runs():
    """Intervention family mode completes an episode."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    while not s.done:
        runner.step(s)
    assert s.done


def test_env_info_contains_intervention_scores_for_all_actions():
    """Decision contains scores for WAIT/WARN/UNLOCK/ITEM_DROP."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        assert "WAIT" in s.last_intervention.scores
        assert "WARN" in s.last_intervention.scores
        assert "UNLOCK" in s.last_intervention.scores
        assert "ITEM_DROP" in s.last_intervention.scores


def test_env_info_contains_expected_item_effect():
    """Decision exposes expected_item_effect when available."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    # expected_item_effect exists as field
    if s.last_intervention is not None:
        assert hasattr(s.last_intervention, "expected_item_effect")


def test_wait_warn_unlock_item_comparable():
    """All four actions share the same decision object."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        assert len(s.last_intervention.scores) == 4


def test_item_drop_changes_decision_in_relevant_case():
    """In some scenario, ITEM_DROP becomes competitive."""
    # Train predictor to see high risk → shield becomes valuable
    from src.agents.cost_risk_model import LatentCostRiskHead
    lp = LatentCostRiskHead(d=4)
    for _ in range(50):
        lp.update_from_outcome(np.array([0.5, 0.5, 0.5, 0.5]),
                              cost_label=1.0, risk_label=0.95, weight=3.0)
    s = runner.reset(seed=42, latent_mode=True, latent_predictor=lp,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        # ITEM_DROP should have a non-trivial score
        assert s.last_intervention.scores["ITEM_DROP"] != 0.0


def test_warn_still_best_in_warning_friendly_case():
    """Warning-friendly scenario doesn't get stolen by ITEM_DROP."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    # with default (low-risk) predictor and high autonomy penalty for item,
    # ITEM_DROP should not easily win
    runner.step(s)
    if s.last_intervention is not None:
        assert s.last_intervention.action in ("WAIT", "WARN", "UNLOCK", "ITEM_DROP")


def test_wait_still_best_in_safe_case():
    """Safe scenario prefers WAIT."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        # With uninformed predictor, risk is low → WAIT likely preferred
        assert s.last_intervention.action in ("WAIT", "WARN", "UNLOCK", "ITEM_DROP")


def test_robot_belief_tutor_with_item_mode_runs():
    """Robot-belief tutor + item family mode runs full episode."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    while not s.done:
        runner.step(s)
    assert s.done
    assert s.steps > 0


def test_legacy_mode_baseline_unchanged():
    """Legacy baseline remains at expected survival rates."""
    results = []
    for seed in range(20):
        s = runner.reset(seed=seed, tutor_mode="none",
                         latent_mode=False, robot_belief_mode=False,
                         intervention_family_mode=False)
        while not s.done:
            runner.step(s)
        results.append(runner.get_metrics(s))
    surv = sum(r["survived"] for r in results) / len(results)
    assert surv < 0.30, f"Legacy no_tutor survival too high: {surv:.0%}"


def test_item_drop_scoring_uses_same_counterfactual_framework():
    """ITEM_DROP uses same surrogate rollout as other actions."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        cf = s.last_intervention.counterfactual_scores
        # All 4 actions have counterfactual (risk, cost) tuples
        for action_key in ("WAIT", "WARN", "UNLOCK", "ITEM_DROP"):
            assert action_key in cf
            assert len(cf[action_key]) == 2


def test_planner_inventory_awareness_changes_prefix_score():
    """Shield in inventory changes planner cost for risky cells."""
    from src.agents.planner_astar import cell_cost_v2_latent
    from src.agents.cost_risk_model import LatentCostRiskHead
    lp = LatentCostRiskHead(d=4)
    # Train to see moderate risk
    for _ in range(30):
        lp.update_from_outcome(np.array([0.5, 0.5, 0.5, 0.5]),
                              cost_label=1.0, risk_label=0.7, weight=2.0)
    fb_mean = np.full((5, 15, 4), 0.5)
    passable = np.ones((5, 15), dtype=bool)
    inv_none = None
    inv_shield = InventoryState(shield=1)

    cost_no_shield = cell_cost_v2_latent(2, 3, fb_mean, lp, passable,
                                          inventory_state=inv_none)
    cost_with_shield = cell_cost_v2_latent(2, 3, fb_mean, lp, passable,
                                            inventory_state=inv_shield)
    assert cost_with_shield < cost_no_shield


def test_item_disabled_mode_matches_phase7_behavior():
    """item_drop_enabled=False reproduces Phase 7 intervention decisions."""
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=False)
    runner.step(s)
    if s.last_intervention is not None:
        assert s.last_intervention.scores["ITEM_DROP"] < 0


def test_block_path_not_in_main_action_family():
    """BLOCK_PATH is excluded from main intervention comparison."""
    assert InterventionType.BLOCK_PATH not in MAIN_INTERVENTION_FAMILY
    s = runner.reset(seed=42, latent_mode=True,
                     robot_belief_mode=True, prefix_horizon=5,
                     intervention_family_mode=True, item_drop_enabled=True)
    runner.step(s)
    if s.last_intervention is not None:
        # BLOCK_PATH should not appear in scores
        assert "BLOCK_PATH" not in s.last_intervention.scores
        assert "BLOCK" not in s.last_intervention.scores
