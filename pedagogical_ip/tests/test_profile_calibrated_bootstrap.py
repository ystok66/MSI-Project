"""Tests for calibrated bootstrap (Task 3A).

Verifies:
1. Shrinkage math: ρ_c = λ_c · (1 - c̄_T)
2. Low confidence → more shrinkage toward prior
3. High confidence → near-full carry-over
4. Backward compatibility: use_calibration=False gives same result as before
5. Custom calib_fn works after shrinkage
"""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np

from src.teachers.profile_state import ProfileState
from src.teachers.profile_bootstrap import bootstrap_observer
from src.teachers.internalization_observer import A1MtObserver, A1MtObserverFrozen


PRIOR = {"tau": 0.3, "nu": 0.1, "gamma_gen": 0.0,
         "gamma_spec": 0.0, "kappa": 0.3}


class TestCalibrationMath:

    def test_no_calibration_same_as_before(self):
        """use_calibration=False should give identical result to raw carry-over."""
        obs1 = A1MtObserver(); obs1.reset()
        obs2 = A1MtObserver(); obs2.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.7, "nu": 0.3, "gamma_gen": 0.15,
                            "gamma_spec": 0.08, "kappa": 0.45},
            confidence={"tau": 0.5, "nu": 0.4, "gamma_gen": 0.3},
        )
        bootstrap_observer(obs1, profile, use_calibration=False)
        bootstrap_observer(obs2, profile, use_calibration=False)
        assert abs(obs1.tau_hat - obs2.tau_hat) < 1e-12
        assert abs(obs1.tau_hat - 0.7) < 1e-9  # full carry-over

    def test_low_confidence_shrinks_toward_prior(self):
        """Low confidence should shrink toward prior (at session_idx > 0)."""
        obs = A1MtObserver(); obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.9, "nu": 0.5, "gamma_gen": 0.4,
                            "gamma_spec": 0.2, "kappa": 0.6},
            confidence={"tau": 0.1, "nu": 0.1, "gamma_gen": 0.1},  # low conf
            session_idx=3,  # session_scale = log(4)/log(4) = 1.0
        )
        bootstrap_observer(obs, profile, use_calibration=True, lambda_c=0.5)
        # c_tau=0.1, session_scale=1.0, ρ_tau = 0.5*(1-0.1)*1.0 = 0.45
        # w_carry = (1-0.45)*1.0 = 0.55
        # tau_hat = 0.55*0.9 + 0.45*0.3 = 0.495+0.135 = 0.63
        assert abs(obs.tau_hat - 0.63) < 0.02
        # Should be substantially closer to prior than full carry-over
        assert abs(obs.tau_hat - PRIOR["tau"]) < abs(0.9 - PRIOR["tau"])

    def test_high_confidence_near_full_carryover(self):
        """High confidence should give near-full carry-over."""
        obs = A1MtObserver(); obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.9, "nu": 0.5, "gamma_gen": 0.4,
                            "gamma_spec": 0.2, "kappa": 0.6},
            confidence={"tau": 0.9, "nu": 0.9, "gamma_gen": 0.9},  # high conf
            session_idx=3,
        )
        bootstrap_observer(obs, profile, use_calibration=True, lambda_c=0.3)
        # c_tau=0.9, session_scale=1.0, ρ_tau = 0.3*(1-0.9)*1.0 = 0.03
        # w_carry = (1-0.03)*1.0 = 0.97 → very close to raw
        assert abs(obs.tau_hat - 0.9) < 0.05

    def test_lambda_c_zero_means_no_shrinkage(self):
        """λ_c = 0 should give full carry-over regardless of confidence."""
        obs = A1MtObserver(); obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.8, "nu": 0.4, "gamma_gen": 0.2,
                            "gamma_spec": 0.1, "kappa": 0.5},
            confidence={"tau": 0.1, "nu": 0.1, "gamma_gen": 0.1},
        )
        bootstrap_observer(obs, profile, use_calibration=True, lambda_c=0.0)
        assert abs(obs.tau_hat - 0.8) < 1e-9  # full carry-over

    def test_shrinkage_monotonic_in_confidence(self):
        """Higher confidence → tau_hat closer to carry-over value."""
        results = []
        for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
            obs = A1MtObserver(); obs.reset()
            profile = ProfileState(
                m_hat_terminal={"tau": 0.9},
                confidence={"tau": c, "nu": c, "gamma_gen": c},
                session_idx=3,
            )
            bootstrap_observer(obs, profile, use_calibration=True, lambda_c=0.5)
            results.append(obs.tau_hat)
        # tau_hat should be monotonically increasing with confidence
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1] + 1e-9, \
                f"Not monotonic at conf={[0.1,0.3,0.5,0.7,0.9][i]}: {results}"


class TestCalibrationOnFrozen:

    def test_works_on_frozen_observer(self):
        """Calibrated bootstrap should work on A1MtObserverFrozen."""
        obs = A1MtObserverFrozen()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.7, "nu": 0.3, "gamma_gen": 0.15,
                            "gamma_spec": 0.08, "kappa": 0.45},
            confidence={"tau": 0.5, "nu": 0.4, "gamma_gen": 0.3},
        )
        # Should NOT raise
        bootstrap_observer(obs, profile, use_calibration=True, lambda_c=0.3)
        assert obs.tau_hat != PRIOR["tau"]  # not just prior


class TestCustomCalibFn:

    def test_custom_fn_called_after_shrinkage(self):
        """Custom calib_fn should be called and can further modify state."""
        called = []
        def my_calib(observer, profile):
            called.append(True)
            observer.kappa_hat = 0.99  # override

        obs = A1MtObserver(); obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.6, "kappa": 0.4},
            confidence={"tau": 0.5, "nu": 0.5, "gamma_gen": 0.5},
        )
        bootstrap_observer(obs, profile, use_calibration=True,
                          calib_fn=my_calib, lambda_c=0.3)
        assert len(called) == 1
        assert obs.kappa_hat == 0.99
