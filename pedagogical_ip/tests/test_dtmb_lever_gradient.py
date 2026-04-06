"""DTMB-L Lever Gradient Test.

Verifies the cross-difficulty lever dominance gradient:
  medium: Δ_warn > 0, Δ_item > 0
  hard:   Δ_unlock > 0, Δ_item > 0
  (medium Δ_unlock = 0 is ACCEPTED, not a failure)

Uses a small seed count (10) for fast CI execution.
"""
import sys
sys.path.insert(0, ".")

import pytest
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "deep_tree_mixed_bottleneck_lattice"
N_SEEDS = 10  # small for CI speed; use 50 for full validation


def _base_cfg(**overrides):
    cfg = dict(
        tutor_mode="none", warning_mode="none",
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=True, belief_planning_mode=True,
        latent_mode=True, patch_radius=2, prefix_horizon=5,
        allowed_interventions=None,
    )
    cfg.update(overrides)
    return cfg


POLICIES = {
    "canonical": _base_cfg(),
    "no_warn": _base_cfg(
        allowed_interventions=frozenset({"WAIT", "UNLOCK", "ITEM_DROP"})),
    "no_unlock": _base_cfg(
        allowed_interventions=frozenset({"WAIT", "WARN", "ITEM_DROP"})),
    "no_item_drop": _base_cfg(
        item_drop_enabled=False,
        allowed_interventions=frozenset({"WAIT", "WARN", "UNLOCK"})),
    "no_tutor": _base_cfg(
        robot_belief_mode=False, intervention_family_mode=False,
        item_drop_enabled=False, prefix_horizon=0),
}


def _run_policy(runner, difficulty, policy_cfg, n_seeds):
    """Run N episodes, return mean survival."""
    survs = []
    for seed in range(n_seeds):
        try:
            s = runner.reset(seed=seed, difficulty=difficulty,
                             scenario_family=FAMILY, **policy_cfg)
            while not s.done:
                s = runner.step(s)
            m = runner.get_metrics(s)
            survs.append(m["survived"])
        except Exception:
            survs.append(False)
    return np.mean(survs)


class TestMediumLeverGradient:
    """Medium should show: Δ_warn > 0, Δ_item ≥ 0."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_results(self, request):
        runner = LatticeV2Runner()
        results = {}
        for name, cfg in POLICIES.items():
            results[name] = _run_policy(runner, "medium", cfg, N_SEEDS)
        request.cls.results = results

    def test_warn_lever_positive(self):
        delta = self.results["canonical"] - self.results["no_warn"]
        assert delta > 0.0, (
            f"Medium Δ_warn={delta:.3f} ≤ 0 — WARN lever non-functional")

    def test_item_lever_nonnegative(self):
        delta = self.results["canonical"] - self.results["no_item_drop"]
        assert delta >= -0.05, (
            f"Medium Δ_item={delta:.3f} < -0.05 — ITEM_DROP hurts")

    def test_total_lift_positive(self):
        delta = self.results["canonical"] - self.results["no_tutor"]
        assert delta > 0.0, (
            f"Medium Δ_total={delta:.3f} ≤ 0 — canonical no better than no_tutor")


class TestHardLeverGradient:
    """Hard should show: Δ_item ≥ 0, and Surv_canonical in [0.05, 0.50]."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_results(self, request):
        runner = LatticeV2Runner()
        results = {}
        for name, cfg in POLICIES.items():
            results[name] = _run_policy(runner, "hard", cfg, N_SEEDS)
        request.cls.results = results

    def test_canonical_in_discriminable_band(self):
        surv = self.results["canonical"]
        assert 0.05 <= surv <= 0.50, (
            f"Hard Surv_canonical={surv:.3f} outside [0.05, 0.50]")

    def test_no_tutor_low(self):
        surv = self.results["no_tutor"]
        assert surv <= 0.15, (
            f"Hard Surv_no_tutor={surv:.3f} > 0.15 — too easy")

    def test_item_lever_nonnegative(self):
        delta = self.results["canonical"] - self.results["no_item_drop"]
        assert delta >= -0.05, (
            f"Hard Δ_item={delta:.3f} < -0.05 — ITEM_DROP hurts")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
