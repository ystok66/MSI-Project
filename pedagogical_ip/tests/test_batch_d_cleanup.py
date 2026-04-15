"""
Batch D: P2 Maintenance Cleanup — Tests.

Covers:
1. PlannerWeights dataclass defaults match canonical values
2. PlannerWeights is frozen (immutable)
3. RobotBelief deprecated property compat
4. Runner and RobotBelief share same PlannerWeights
5. V0 observation dead code removed
6. Deprecated markers on legacy planner functions
7. cell_cost_v2_latent lambda_r → lambda_risk rename
8. PlannerWeights threaded through canonical runner path
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from dataclasses import FrozenInstanceError


# ── 1. PlannerWeights defaults ──────────────────────────────────────


def test_plannerweights_defaults_match_canonical():
    """PlannerWeights defaults must match the de-facto canonical values
    that were previously scattered across plan_from_belief and RobotBelief."""
    from src.agents.planner_weights import PlannerWeights
    pw = PlannerWeights()
    assert pw.lambda_cost == 1.0
    assert pw.lambda_risk == 3.0
    assert pw.lambda_uc == 0.1
    assert pw.lambda_ur == 0.1


def test_plannerweights_frozen():
    """PlannerWeights should be immutable to prevent accidental mutation."""
    from src.agents.planner_weights import PlannerWeights
    pw = PlannerWeights()
    with pytest.raises(FrozenInstanceError):
        pw.lambda_risk = 5.0


def test_plannerweights_custom_values():
    """Custom PlannerWeights should preserve values exactly."""
    from src.agents.planner_weights import PlannerWeights
    pw = PlannerWeights(lambda_cost=2.0, lambda_risk=5.0, lambda_uc=0.2, lambda_ur=0.3)
    assert pw.lambda_cost == 2.0
    assert pw.lambda_risk == 5.0
    assert pw.lambda_uc == 0.2
    assert pw.lambda_ur == 0.3


# ── 2. RobotBelief PlannerWeights integration ──────────────────────


def test_robotbelief_plannerweights_init():
    """RobotBelief should hold agent_planner_weights with correct defaults."""
    from src.teachers.robot_belief import RobotBelief, init_robot_belief
    from src.agents.planner_weights import PlannerWeights

    mean = np.zeros((5, 10, 4))
    var = np.ones((5, 10, 4)) * 0.5

    rb = init_robot_belief(mean, var)
    assert isinstance(rb.agent_planner_weights, PlannerWeights)
    assert rb.agent_planner_weights.lambda_risk == 3.0
    assert rb.agent_planner_weights.lambda_cost == 1.0
    assert rb.agent_planner_weights.lambda_uc == 0.1
    assert rb.agent_planner_weights.lambda_ur == 0.1


def test_robotbelief_plannerweights_custom():
    """init_robot_belief with explicit PlannerWeights should pass them through."""
    from src.teachers.robot_belief import init_robot_belief
    from src.agents.planner_weights import PlannerWeights

    mean = np.zeros((5, 10, 4))
    var = np.ones((5, 10, 4)) * 0.5
    pw = PlannerWeights(lambda_risk=7.0, lambda_uc=0.3)

    rb = init_robot_belief(mean, var, planner_weights=pw)
    assert rb.agent_planner_weights.lambda_risk == 7.0
    assert rb.agent_planner_weights.lambda_uc == 0.3
    assert rb.agent_planner_weights.lambda_cost == 1.0  # default preserved


def test_robotbelief_deprecated_properties():
    """Deprecated properties should delegate to agent_planner_weights."""
    from src.teachers.robot_belief import init_robot_belief
    from src.agents.planner_weights import PlannerWeights

    mean = np.zeros((5, 10, 4))
    var = np.ones((5, 10, 4)) * 0.5
    pw = PlannerWeights(lambda_risk=4.0, lambda_cost=2.0, lambda_uc=0.15, lambda_ur=0.25)

    rb = init_robot_belief(mean, var, planner_weights=pw)

    # Deprecated properties must match canonical source
    assert rb.agent_risk_weight == pw.lambda_risk
    assert rb.agent_lambda_c == pw.lambda_cost
    assert rb.agent_lambda_uc == pw.lambda_uc
    assert rb.agent_lambda_ur == pw.lambda_ur
    assert rb.agent_uncertainty_weight == pw.lambda_uc


def test_robotbelief_legacy_init_compat():
    """init_robot_belief with legacy individual params should still work."""
    from src.teachers.robot_belief import init_robot_belief

    mean = np.zeros((5, 10, 4))
    var = np.ones((5, 10, 4)) * 0.5

    # Legacy caller: passes agent_risk_weight directly (not PlannerWeights)
    rb = init_robot_belief(mean, var, agent_risk_weight=5.0, agent_uncertainty_weight=0.2)
    assert rb.agent_planner_weights.lambda_risk == 5.0
    assert rb.agent_planner_weights.lambda_uc == 0.2


# ── 3. Runner PlannerWeights integration ────────────────────────────


def test_runner_state_has_planner_weights():
    """V2EpisodeState should have a planner_weights field."""
    from src.envs.lattice_v2_runner import V2EpisodeState
    import dataclasses
    field_names = [f.name for f in dataclasses.fields(V2EpisodeState)]
    assert "planner_weights" in field_names


def test_runner_and_robotbelief_share_weights():
    """Runner and RobotBelief should receive the same PlannerWeights."""
    from src.agents.planner_weights import PlannerWeights

    pw = PlannerWeights()
    # Simulate what the runner does:
    from src.teachers.robot_belief import init_robot_belief
    mean = np.zeros((5, 10, 4))
    var = np.ones((5, 10, 4)) * 0.5
    rb = init_robot_belief(mean, var, planner_weights=pw)

    assert rb.agent_planner_weights.lambda_risk == pw.lambda_risk
    assert rb.agent_planner_weights.lambda_cost == pw.lambda_cost
    assert rb.agent_planner_weights.lambda_uc == pw.lambda_uc
    assert rb.agent_planner_weights.lambda_ur == pw.lambda_ur


# ── 4. Observation model V0 removal ────────────────────────────────


def test_observation_model_no_v0_observation_class():
    """Observation dataclass should no longer be importable from observation_model."""
    from src.agents import observation_model
    assert not hasattr(observation_model, "Observation"), (
        "V0 Observation dataclass should have been removed"
    )


def test_observation_model_no_v0_generate():
    """generate_observations should no longer be importable."""
    from src.agents import observation_model
    assert not hasattr(observation_model, "generate_observations"), (
        "V0 generate_observations should have been removed"
    )


def test_observation_model_canonical_still_works():
    """observe_features and observe_features_patch should still be importable."""
    from src.agents.observation_model import observe_features, observe_features_patch
    assert callable(observe_features)
    assert callable(observe_features_patch)


# ── 5. Deprecated markers ──────────────────────────────────────────


def test_plan_next_action_deprecated_marker():
    """plan_next_action V0 should have DEPRECATED in its source."""
    import inspect
    from src.agents.planner_astar import plan_next_action
    source = inspect.getsource(plan_next_action)
    assert "DEPRECATED" in source


def test_plan_next_action_v2_deprecated_marker():
    """plan_next_action_v2 should have DEPRECATED in its source."""
    import inspect
    from src.agents.planner_astar import plan_next_action_v2
    source = inspect.getsource(plan_next_action_v2)
    assert "DEPRECATED" in source


def test_cell_cost_v2_deprecated_marker():
    """cell_cost_v2 should have DEPRECATED in its source."""
    import inspect
    from src.agents.planner_astar import cell_cost_v2
    source = inspect.getsource(cell_cost_v2)
    assert "DEPRECATED" in source


# ── 6. cell_cost_v2_latent lambda_risk rename ──────────────────────


def test_cell_cost_v2_latent_accepts_lambda_risk():
    """cell_cost_v2_latent should accept lambda_risk (not lambda_r)."""
    import inspect
    from src.agents.planner_astar import cell_cost_v2_latent
    sig = inspect.signature(cell_cost_v2_latent)
    param_names = list(sig.parameters.keys())
    assert "lambda_risk" in param_names, f"Expected lambda_risk in params, got: {param_names}"
    assert "lambda_r" not in param_names, "lambda_r should have been renamed to lambda_risk"


# ── 7. Import stability ────────────────────────────────────────────


def test_bounded_agent_still_importable():
    """BoundedRationalAgent should still be importable (deprecated, not deleted)."""
    from src.agents.bounded_agent import BoundedRationalAgent
    assert BoundedRationalAgent is not None


def test_bounded_astar_still_importable():
    """bounded_astar should still be importable (active, used by cause_scoring)."""
    from src.agents.planner_astar import bounded_astar
    assert callable(bounded_astar)


def test_plan_with_alternatives_v2_not_deprecated():
    """plan_with_alternatives_v2 is canonical — should NOT be deprecated."""
    import inspect
    from src.agents.planner_astar import plan_with_alternatives_v2
    source = inspect.getsource(plan_with_alternatives_v2)
    # First line of function body should NOT contain DEPRECATED
    first_lines = source.split('\n')[:5]
    first_lines_str = '\n'.join(first_lines)
    assert "DEPRECATED" not in first_lines_str
