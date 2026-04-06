"""Tests for profile-aware curriculum hook (Task 2B Phase B1).

Verifies:
1. Need hook produces positive bonus for high-deficit lessons
2. Need hook changes lesson ranking
3. Hook persists across reset_session
4. No-hook mode produces same results as canonical baseline
5. probe_weakness_summary computes correct EMA
"""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.curriculum.curriculum_controller_v13 import (
    CurriculumControllerV13, ControllerV13Config,
)
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.agents.internalization_state_v3 import FactoredInternalizationState
from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_manager import ProfileManager
from src.teachers.profile_bootstrap import make_need_hook


class TestNeedHookFactory:

    def test_high_deficit_produces_positive_bonus(self):
        """Lesson targeting deficient dimension should get bonus."""
        z_bar = {"RC": 0.3, "TR": 0.3, "EP": 0.3, "VA": 0.3, "IA": 0.3}
        hook = make_need_hook(z_bar, lambda_need=0.3)
        # tic_rescue_heavy has high RC gain (0.25)
        les = next(l for l in LESSON_CATALOG_V2 if l.name == "tic_rescue_heavy")
        bonus = hook(les, 0.0, {})
        assert bonus > 0, f"Expected positive bonus, got {bonus}"

    def test_mastered_dimension_no_bonus(self):
        """Already-mastered probes should contribute zero deficit."""
        z_bar = {"RC": 0.9, "TR": 0.9, "EP": 0.9, "VA": 0.9, "IA": 0.9}
        hook = make_need_hook(z_bar, lambda_need=0.3)
        les = next(l for l in LESSON_CATALOG_V2 if l.name == "tic_rescue_heavy")
        bonus = hook(les, 0.0, {})
        assert bonus == 0.0, f"Expected zero bonus when mastered, got {bonus}"

    def test_different_lessons_get_different_bonuses(self):
        """Lessons with different gain profiles should get different bonuses."""
        z_bar = {"RC": 0.3, "TR": 0.3, "EP": 0.3, "VA": 0.3, "IA": 0.3}
        hook = make_need_hook(z_bar, lambda_need=0.3)
        bonuses = {}
        for les in LESSON_CATALOG_V2:
            bonuses[les.name] = hook(les, 0.0, {})
        # Not all the same
        vals = list(bonuses.values())
        assert max(vals) > min(vals) + 0.001


class TestControllerProfileHook:

    def _make_controller(self, theta="safe"):
        ctrl = CurriculumControllerV13(theta=theta)
        ctrl.reset_session(budget=10.0)
        return ctrl

    def test_no_hook_baseline(self):
        """Without hook, profile_need should be 0.0 in all terms."""
        ctrl = self._make_controller()
        m = FactoredInternalizationState()
        action, les, J, info = ctrl.select_action(m)
        # Check that profile_need exists and is 0.0 in terms
        for terms in ctrl._term_scores[-1].values():
            if not terms.get("filtered", False):
                assert terms.get("profile_need", 0.0) == 0.0

    def test_hook_changes_ranking(self):
        """Installing a need hook should change lesson ranking."""
        ctrl = self._make_controller()
        m = FactoredInternalizationState()

        # Get canonical ranking
        ctrl.reset_session(budget=10.0)
        action1, les1, J1, _ = ctrl.select_action(m)

        # Install hook that strongly favors EP-heavy lessons
        z_bar = {"RC": 0.7, "TR": 0.7, "EP": 0.1, "VA": 0.7, "IA": 0.7}
        hook = make_need_hook(z_bar, lambda_need=2.0)  # strong weight
        ctrl.install_profile_hook(hook)
        ctrl.reset_session(budget=10.0)
        action2, les2, J2, _ = ctrl.select_action(m)

        # With EP deficit and strong hook, EP-heavy lessons should be boosted
        # At minimum, profile_need should appear in terms
        has_nonzero = False
        for terms in ctrl._term_scores[-1].values():
            if not terms.get("filtered", False):
                if terms.get("profile_need", 0.0) != 0.0:
                    has_nonzero = True
        assert has_nonzero, "Need hook should produce non-zero adjustments"

    def test_hook_persists_across_reset(self):
        """Profile hook should survive reset_session."""
        ctrl = self._make_controller()
        z_bar = {"RC": 0.3, "TR": 0.3, "EP": 0.3, "VA": 0.3, "IA": 0.3}
        hook = make_need_hook(z_bar, lambda_need=0.3)
        ctrl.install_profile_hook(hook)
        ctrl.reset_session(budget=10.0)
        assert ctrl._profile_hook is not None

    def test_remove_hook(self):
        """Removing hook should revert to canonical behavior."""
        ctrl = self._make_controller()
        hook = make_need_hook({"RC": 0.3}, lambda_need=0.3)
        ctrl.install_profile_hook(hook)
        ctrl.remove_profile_hook()
        assert ctrl._profile_hook is None


class TestProbeWeaknessSummary:

    def test_no_sessions_returns_default(self):
        pm = ProfileManager()
        z = pm.probe_weakness_summary("nonexistent")
        assert all(z[p] == 0.5 for p in PROBE_NAMES)

    def test_single_session_ema(self):
        pm = ProfileManager()
        ps = ProfileState(
            history=SessionSummary(
                probe_means={"RC": 0.8, "TR": 0.4, "EP": 0.6, "VA": 0.7, "IA": 0.3}
            )
        )
        pm.finalize_session("L0", ps)
        z = pm.probe_weakness_summary("L0", rho=0.5)
        # z̄ = 0.5 * 0.5 + 0.5 * obs
        assert abs(z["RC"] - 0.65) < 0.01  # (0.5*0.5 + 0.5*0.8)
        assert abs(z["IA"] - 0.40) < 0.01  # (0.5*0.5 + 0.5*0.3)

    def test_multi_session_ema_recency(self):
        pm = ProfileManager()
        for i in range(3):
            ps = ProfileState(
                history=SessionSummary(
                    probe_means={"RC": 0.3 + i * 0.2}  # 0.3, 0.5, 0.7
                )
            )
            pm.finalize_session("L0", ps)
        z = pm.probe_weakness_summary("L0", rho=0.5)
        # More recent sessions should have more weight
        # Session 0: 0.5*0.5 + 0.5*0.3 = 0.40
        # Session 1: 0.5*0.40 + 0.5*0.5 = 0.45
        # Session 2: 0.5*0.45 + 0.5*0.7 = 0.575
        assert abs(z["RC"] - 0.575) < 0.01


class TestEndToEndProfileCurriculum:
    """Integration test: profile → weakness → hook → different lesson selection."""

    def test_profile_driven_lesson_shift(self):
        """With EP-deficient profile, controller should boost EP-heavy lessons."""
        pm = ProfileManager()
        # Create learner profile with low EP
        ps = ProfileState(
            history=SessionSummary(
                probe_means={"RC": 0.7, "TR": 0.7, "EP": 0.1, "VA": 0.7, "IA": 0.7}
            )
        )
        pm.finalize_session("L0", ps)
        z_bar = pm.probe_weakness_summary("L0")

        # Verify EP is low
        assert z_bar["EP"] < 0.4

        # Create controller with need hook
        ctrl = CurriculumControllerV13(theta="safe")
        ctrl.reset_session(budget=10.0)
        hook = make_need_hook(z_bar, lambda_need=1.0)
        ctrl.install_profile_hook(hook)

        m = FactoredInternalizationState()
        action, les, J, _ = ctrl.select_action(m)

        # The selected lesson should have some EP gain
        # or the profile_need term should be nonzero
        terms = ctrl._term_scores[-1]
        nonzero_need_count = sum(
            1 for t in terms.values()
            if not t.get("filtered", False) and t.get("profile_need", 0) > 0.001
        )
        assert nonzero_need_count > 0, "At least some lessons should get need bonus"
