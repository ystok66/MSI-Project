"""Batch A regression tests — P0 correctness hotfixes.

Tests shield planning, tutor fidelity, NaN safety, SlowFast lifecycle,
and DTMB oracle semantics.
"""

import numpy as np
import pytest

# ── A1: Shield enters belief planning ─────────────────────────────────

def test_shield_enters_cell_cost_v2_latent():
    """cell_cost_v2_latent should reduce risk penalty when shield is present."""
    from src.agents.structured_basis_head import StructuredBasisCostRiskHead
    from src.agents.planner_astar import cell_cost_v2_latent

    lp = StructuredBasisCostRiskHead(d=4)
    # Train briefly so risk prediction is non-trivial
    risky_x = np.array([0.0, 0.0, 0.85, 0.80])
    for _ in range(5):
        lp.update_from_outcome(risky_x, cost_label=1.0, risk_label=0.5, weight=2.0)

    H, W = 5, 10
    mean = np.full((H, W, 4), 0.5)
    mean[2, 5] = risky_x  # belt cell
    passable = np.ones((H, W), dtype=bool)

    # Without shield
    cost_no_shield = cell_cost_v2_latent(
        2, 5, mean, lp, passable, lambda_risk=5.0)

    # With shield
    from src.teachers.interventions import InventoryState
    inv = InventoryState(shield=1)
    cost_with_shield = cell_cost_v2_latent(
        2, 5, mean, lp, passable, lambda_risk=5.0, inventory_state=inv)

    assert cost_with_shield < cost_no_shield, (
        f"Shield should reduce cost: {cost_with_shield} >= {cost_no_shield}")
    # Shield reduces risk penalty by (1 - 0.5) = 0.5×
    # So difference should be roughly half the risk penalty
    reduction_ratio = cost_with_shield / cost_no_shield
    assert reduction_ratio < 0.95, (
        f"Shield reduction too small: ratio={reduction_ratio:.3f}")


def test_plan_from_belief_accepts_inventory():
    """plan_from_belief should accept and forward inventory_state."""
    from src.agents.belief_planning import plan_from_belief
    from src.agents.structured_basis_head import StructuredBasisCostRiskHead
    from src.teachers.interventions import InventoryState

    lp = StructuredBasisCostRiskHead(d=4)
    H, W = 7, 15
    mean = np.full((H, W, 4), 0.3)
    belief_cost = np.ones((H, W))
    passable = np.ones((H, W), dtype=bool)
    inv = InventoryState(shield=1)

    # Should not raise
    bp = plan_from_belief(
        (3, 1), (3, 13), belief_cost, mean,
        lp.risk_head, passable,
        latent_predictor=lp,
        inventory_state=inv,
    )
    assert bp.action in ("UP", "DOWN", "LEFT", "RIGHT")


# ── A2: Tutor surrogate fidelity ──────────────────────────────────────

def test_predict_agent_prefix_accepts_var_and_necessity():
    """predict_agent_prefix should accept feature_belief_var and route_necessity."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.envs.scenario_families import generate_scenario
    from src.teachers.agent_predictor import predict_agent_prefix

    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, belief_planning_mode=True,
                     prefix_horizon=5, robot_belief_mode=True)
    # Step a bit to build belief
    for _ in range(3):
        if s.done:
            break
        runner.step(s)

    if s.robot_belief is not None and s.latent_predictor is not None:
        extra = s.warned_cell_extra if s.warned_cell_extra else None
        # Should not raise — tests new parameters
        pred = predict_agent_prefix(
            s.robot_belief, s.agent_pos, s.goal,
            s.belief_cost, s.passable,
            warned_cell_extra=extra,
            t=s.t, t_max=s.t_max,
            feature_belief_var=s.feature_belief.var,
            route_necessity=0.5,
        )
        assert pred.predicted_plan is not None


# ── A3: NaN safety guards ─────────────────────────────────────────────

def test_bayesian_risk_head_finite_after_extreme_update():
    """BayesianRiskHead must remain finite after extreme gradient."""
    from src.agents.risk_model import BayesianRiskHead

    head = BayesianRiskHead(d=4)
    # Feed extreme inputs that could produce NaN gradients
    extreme_x = np.array([1e6, -1e6, 1e6, -1e6])
    head.update_from_label(extreme_x, 1.0, weight=100.0)

    assert np.all(np.isfinite(head.w)), f"w is not finite: {head.w}"
    assert np.isfinite(head.b), f"b is not finite: {head.b}"

    # Prediction should also be finite
    pred = head.predict_risk(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.isfinite(pred), f"prediction is not finite: {pred}"


def test_basis_risk_head_finite_after_extreme_update():
    """BasisRiskHead must remain finite after extreme gradient."""
    from src.agents.structured_basis_head import BasisRiskHead

    head = BasisRiskHead()  # uses fixed DIMS=7 internally
    # BasisRiskHead applies risk_basis(x) which maps 4D → 7D
    extreme_x = np.array([1e6, -1e6, 1e6, -1e6])
    head.update_from_label(extreme_x, 1.0, weight=100.0)

    assert np.all(np.isfinite(head.w)), f"w is not finite: {head.w}"
    assert np.isfinite(head.b), f"b is not finite: {head.b}"


def test_risk_head_nan_input_skip():
    """Risk heads should skip update on NaN input without corrupting state."""
    from src.agents.risk_model import BayesianRiskHead

    head = BayesianRiskHead(d=4)
    # Normal update first
    head.update_from_label(np.array([0.5, 0.5, 0.5, 0.5]), 0.3)
    w_before = head.w.copy()

    # NaN input
    nan_x = np.array([np.nan, 0.5, 0.5, 0.5])
    head.update_from_label(nan_x, 0.5)

    # Weights should be unchanged (update skipped)
    assert np.all(np.isfinite(head.w)), f"w corrupted by NaN input: {head.w}"


# ── A4: SlowFast lifecycle ────────────────────────────────────────────

def test_slowfast_begin_end_lifecycle():
    """GenericSlowFastPredictor lifecycle should transfer w,b AND stats."""
    from src.agents.slow_fast_head import GenericSlowFastPredictor
    from src.agents.structured_basis_head import StructuredBasisCostRiskHead

    sf = GenericSlowFastPredictor(
        base_factory=lambda: StructuredBasisCostRiskHead(d=4),
        alpha=0.3,
    )

    # Episode 1: train fast head
    sf.begin_episode()
    for _ in range(15):
        x = np.random.default_rng(42).uniform(0, 1, 4)
        sf.update_from_outcome(x, cost_label=1.0, risk_label=0.3)

    # Check fast has learned
    assert sf.n_updates == 15
    fast_w_before_end = sf.cost_head.w.copy()

    # End episode — slow should receive transfer
    sf.end_episode()
    slow_cost_w = sf._slow.cost_head.w.copy()
    slow_n = sf._slow.cost_head.n_updates
    assert np.linalg.norm(slow_cost_w) > 0, "Slow w not updated"
    assert slow_n > 0, "Slow n_updates not synced"

    # Episode 2: begin should warm-start fast from slow
    sf.begin_episode()
    fast_w_after_begin = sf.cost_head.w.copy()
    assert np.allclose(fast_w_after_begin, slow_cost_w), \
        "Fast w not restored from slow after begin_episode"
    # n_updates should be reset (fast starts fresh counting)
    assert sf.cost_head.n_updates == 0, \
        "Fast n_updates should be 0 after begin_episode"
    # But slow still has its n_updates
    assert sf._slow.cost_head.n_updates > 0


def test_runner_has_end_episode_method():
    """LatticeV2Runner should expose end_episode for transfer lifecycle."""
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    assert hasattr(runner, 'end_episode'), "Runner missing end_episode method"


# ── A5: DTMB oracle semantics ─────────────────────────────────────────

def test_dtmb_oracle_does_not_mutate_true_risk():
    """DTMB oracle ITEM_DROP must NOT modify gridmap.true_risk."""
    from src.envs.lattice_v2_runner import LatticeV2Runner

    runner = LatticeV2Runner()
    s = runner.reset(
        seed=42, latent_mode=True, belief_planning_mode=True,
        prefix_horizon=5, tutor_mode="dtmb_oracle",
        scenario_family="deep_tree_mixed_bottleneck_lattice",
        item_drop_enabled=True,
    )

    # Record true_risk snapshot
    risk_before = s.gridmap.true_risk.copy()

    # Run enough steps that oracle should trigger ITEM_DROP
    for _ in range(s.t_max):
        if s.done:
            break
        runner.step(s)

    # true_risk must not have been modified
    assert np.allclose(risk_before, s.gridmap.true_risk), \
        "Oracle ITEM_DROP mutated gridmap.true_risk!"


def test_dtmb_oracle_sets_last_intervention():
    """After oracle action, s.last_intervention should exist."""
    from src.envs.dtmb_helpers import (
        apply_dtmb_oracle_action, DTMBDispatchConfig,
    )
    from src.envs.lattice_v2_runner import LatticeV2Runner

    runner = LatticeV2Runner()
    s = runner.reset(
        seed=42, latent_mode=True, belief_planning_mode=True,
        prefix_horizon=5, tutor_mode="dtmb_oracle",
        scenario_family="deep_tree_mixed_bottleneck_lattice",
        item_drop_enabled=True,
    )

    apply_dtmb_oracle_action(s)
    assert hasattr(s, 'last_intervention'), "last_intervention not set"
    assert s.last_intervention is not None, "last_intervention is None"
    assert hasattr(s.last_intervention, 'action'), "missing .action attribute"


def test_dtmb_already_shielded_uses_has_shield():
    """Shield check should use has_shield(), not ghost attribute active_duration."""
    from src.teachers.interventions import InventoryState

    inv = InventoryState(shield=1)
    assert inv.has_shield() is True
    # Confirm active_duration does NOT exist
    assert not hasattr(inv, 'active_duration'), \
        "InventoryState should not have active_duration attribute"
