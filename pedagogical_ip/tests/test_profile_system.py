"""Tests for Persistent Learner Profile system (Task 2 Phase 1).

5 test groups:
1. ProfileState roundtrip (serialize/deserialize)
2. Observer bootstrap (starts from previous terminal, not default)
3. Persistent vs reset (different trajectories)
4. ProfileManager (aggregate, drift, I/O)
5. Frozen observer compatibility (bootstrap works on A1MtObserverFrozen)
"""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_manager import ProfileManager
from src.teachers.profile_bootstrap import (
    bootstrap_observer, bootstrap_agent_state, finalize_session,
)
from src.teachers.internalization_observer import (
    A1MtObserver, A1MtObserverFrozen, ObsEvent,
)
from src.agents.internalization_state_v3 import FactoredInternalizationState


# ═══════════════════════════════════════════════════════
# Group 1: ProfileState roundtrip
# ═══════════════════════════════════════════════════════

class TestProfileStateRoundtrip:

    def test_basic_roundtrip(self):
        ps = ProfileState(
            learner_id="test_learner", session_idx=3, theta="shiny",
            m_terminal={"kappa": 0.8, "tau": 0.6, "nu": 0.2,
                        "gamma_spec": 0.1, "gamma_gen": 0.05},
            m_hat_terminal={"tau": 0.55, "nu": 0.18, "gamma_gen": 0.04,
                            "gamma_spec": 0.09, "kappa": 0.35},
            confidence={"tau": 0.7, "nu": 0.5, "gamma_gen": 0.4},
        )
        d = ps.to_dict()
        ps2 = ProfileState.from_dict(d)
        assert ps2.learner_id == "test_learner"
        assert ps2.session_idx == 3
        assert ps2.theta == "shiny"
        assert abs(ps2.m_hat_terminal["tau"] - 0.55) < 1e-9
        assert abs(ps2.m_terminal["kappa"] - 0.8) < 1e-9

    def test_session_summary_roundtrip(self):
        ss = SessionSummary(
            n_warn=5, n_wait=15, n_steps=20,
            subtype_counts={"wait_clean": 8, "warn_trap": 4},
            probe_means={"RC": 0.6, "TR": 0.5},
            warn_rate_by_subtype={"wait_clean": 0.1, "warn_trap": 0.8},
        )
        d = ss.to_dict()
        ss2 = SessionSummary.from_dict(d)
        assert ss2.n_warn == 5
        assert ss2.warn_rate == 5 / 20
        assert ss2.subtype_counts["warn_trap"] == 4

    def test_copy_independence(self):
        ps = ProfileState(m_hat_terminal={"tau": 0.5})
        ps2 = ps.copy()
        ps2.m_hat_terminal["tau"] = 0.99
        assert ps.m_hat_terminal["tau"] == 0.5  # original unchanged


# ═══════════════════════════════════════════════════════
# Group 2: Observer bootstrap
# ═══════════════════════════════════════════════════════

class TestObserverBootstrap:

    def test_bootstrap_sets_state(self):
        obs = A1MtObserver()
        obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.7, "nu": 0.3, "gamma_gen": 0.15,
                            "gamma_spec": 0.08, "kappa": 0.45},
            confidence={"tau": 0.8, "nu": 0.6, "gamma_gen": 0.5},
        )
        bootstrap_observer(obs, profile)
        assert abs(obs.tau_hat - 0.7) < 1e-9
        assert abs(obs.nu_hat - 0.3) < 1e-9
        assert abs(obs.gamma_gen_hat - 0.15) < 1e-9
        assert abs(obs.gamma_spec_hat - 0.08) < 1e-9
        assert abs(obs.kappa_hat - 0.45) < 1e-9
        assert abs(obs.conf_tau - 0.8) < 1e-9

    def test_bootstrap_different_from_default(self):
        obs1 = A1MtObserver(); obs1.reset()
        obs2 = A1MtObserver(); obs2.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.7, "nu": 0.3, "gamma_gen": 0.15,
                            "gamma_spec": 0.08, "kappa": 0.45},
        )
        bootstrap_observer(obs2, profile)
        # obs1 is default, obs2 is bootstrapped — should differ
        assert abs(obs1.tau_hat - obs2.tau_hat) > 0.1
        assert abs(obs1.nu_hat - obs2.nu_hat) > 0.1

    def test_bootstrap_with_partial_eta(self):
        obs = A1MtObserver(); obs.reset()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.9, "nu": 0.5, "gamma_gen": 0.3,
                            "gamma_spec": 0.2, "kappa": 0.6},
        )
        bootstrap_observer(obs, profile, eta=0.5)
        # tau should be 0.5*0.9 + 0.5*0.3 = 0.6
        assert abs(obs.tau_hat - 0.6) < 1e-9
        # nu should be 0.5*0.5 + 0.5*0.1 = 0.3
        assert abs(obs.nu_hat - 0.3) < 1e-9

    def test_bootstrap_resets_counters(self):
        obs = A1MtObserver(); obs.reset()
        obs._step_counter = 42
        obs._recent_events_tau = 3
        profile = ProfileState(
            m_hat_terminal={"tau": 0.5, "nu": 0.2, "gamma_gen": 0.1,
                            "gamma_spec": 0.0, "kappa": 0.3},
        )
        bootstrap_observer(obs, profile)
        assert obs._step_counter == 0
        assert obs._recent_events_tau == 0

    def test_agent_state_bootstrap(self):
        m = FactoredInternalizationState()
        profile = ProfileState(
            m_terminal={"kappa": 0.8, "tau": 0.7, "nu": 0.25,
                        "gamma_spec": 0.15, "gamma_gen": 0.1},
        )
        bootstrap_agent_state(m, profile)
        assert abs(m.kappa - 0.8) < 1e-9
        assert abs(m.tau - 0.7) < 1e-9
        assert len(m.kappa_history) == 1  # snapshot() called

    def test_finalize_session_creates_profile(self):
        obs = A1MtObserver(); obs.reset()
        # Simulate some updates
        for _ in range(5):
            obs.update(ObsEvent(dose=0.0, warned=False))
        m_true = FactoredInternalizationState()
        m_true.tau = 0.5; m_true.nu = 0.15
        ps = finalize_session(obs, m_true, session_idx=0, theta="safe")
        assert ps.session_idx == 0
        assert ps.theta == "safe"
        assert "tau" in ps.m_hat_terminal
        assert "tau" in ps.history.calibration_error
        assert ps.history.n_steps == 5

    def test_finalize_calibration_error(self):
        obs = A1MtObserver(); obs.reset()
        obs.tau_hat = 0.5
        m_true = FactoredInternalizationState()
        m_true.tau = 0.7
        ps = finalize_session(obs, m_true, session_idx=0, theta="safe")
        assert abs(ps.history.calibration_error["tau"] - 0.2) < 1e-6


# ═══════════════════════════════════════════════════════
# Group 3: Persistent vs Reset trajectories
# ═══════════════════════════════════════════════════════

class TestPersistentVsReset:

    def _run_session(self, obs, n_steps=10):
        """Run observer through n_steps of dummy events."""
        for i in range(n_steps):
            ev = ObsEvent(
                dose=0.5 if i % 3 == 0 else 0.0,
                warned=(i % 3 == 0),
                follow_warn=(i % 3 == 0),
                p_self=0.3, risk=0.3,
                agent_choice=0, oracle_safe=0,
                self_discovery=(i % 5 == 0),
                lure=0.4 if i % 4 == 0 else 0.0,
            )
            obs.update(ev)
        return obs.get_estimate()

    def test_persistent_differs_from_reset(self):
        """Two sessions: persistent should carry over, reset should not."""
        # Session 1
        obs1 = A1MtObserver(); obs1.reset()
        est1 = self._run_session(obs1)

        # Session 2 (reset)
        obs_reset = A1MtObserver(); obs_reset.reset()
        est_reset = self._run_session(obs_reset)

        # Session 2 (persistent)
        profile = finalize_session(
            obs1, FactoredInternalizationState(), 0, "safe")
        obs_persist = A1MtObserver(); obs_persist.reset()
        bootstrap_observer(obs_persist, profile)
        est_persist_before = obs_persist.get_estimate()

        # Persistent should start from session 1 terminal, not default
        assert abs(est_persist_before["tau"] - est1["tau"]) < 1e-6
        assert abs(est_persist_before["nu"] - est1["nu"]) < 1e-6

        # After running session 2, persistent and reset should differ
        est_persist_after = self._run_session(obs_persist)
        # Not identical (different starting points)
        diff = sum(abs(est_persist_after[k] - est_reset[k])
                   for k in ["tau", "nu", "gamma_gen"])
        assert diff > 0.01, f"Persistent and reset produced same trajectory: diff={diff}"


# ═══════════════════════════════════════════════════════
# Group 4: ProfileManager
# ═══════════════════════════════════════════════════════

class TestProfileManager:

    def test_basic_flow(self):
        pm = ProfileManager()
        assert pm.latest("learner_0") is None
        ps = ProfileState(session_idx=0,
                          m_hat_terminal={"tau": 0.5, "nu": 0.2,
                                         "gamma_gen": 0.1, "gamma_spec": 0.0,
                                         "kappa": 0.3})
        pm.finalize_session("learner_0", ps)
        assert pm.session_count("learner_0") == 1
        latest = pm.latest("learner_0")
        assert latest.m_hat_terminal["tau"] == 0.5

    def test_aggregate_history(self):
        pm = ProfileManager()
        for i in range(3):
            ps = ProfileState(
                session_idx=i,
                history=SessionSummary(
                    n_warn=2 + i, n_wait=10,
                    probe_means={"RC": 0.5 + i * 0.1}),
            )
            pm.finalize_session("L0", ps)
        agg = pm.aggregate_history("L0")
        assert agg["n_sessions"] == 3
        assert agg["total_steps"] == 0  # steps not set in this test
        assert abs(agg["probe_means"]["RC"] - 0.6) < 1e-6  # mean of 0.5, 0.6, 0.7

    def test_drift_metric(self):
        pm = ProfileManager()
        for i in range(4):
            ps = ProfileState(
                session_idx=i,
                m_hat_terminal={"tau": 0.3 + i * 0.1, "nu": 0.1,
                                "gamma_gen": 0.0, "gamma_spec": 0.0,
                                "kappa": 0.3},
            )
            pm.finalize_session("L1", ps)
        drift = pm.drift_metric("L1", window=3)
        assert drift["per_dim"]["tau"] > 0.05  # tau is drifting
        assert drift["per_dim"]["nu"] == 0.0   # nu is stable

    def test_json_roundtrip(self):
        pm = ProfileManager()
        ps = ProfileState(
            session_idx=0, theta="shiny",
            m_hat_terminal={"tau": 0.6, "nu": 0.2,
                            "gamma_gen": 0.1, "gamma_spec": 0.0,
                            "kappa": 0.35},
            history=SessionSummary(n_warn=3, n_wait=7),
        )
        pm.finalize_session("L0", ps)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        pm.save_profiles(path)
        pm2 = ProfileManager()
        pm2.load_profiles(path)
        assert pm2.session_count("L0") == 1
        loaded = pm2.latest("L0")
        assert abs(loaded.m_hat_terminal["tau"] - 0.6) < 1e-9
        path.unlink()

    def test_reset_learner(self):
        pm = ProfileManager()
        pm.finalize_session("L0", ProfileState())
        assert pm.session_count("L0") == 1
        pm.reset_learner("L0")
        assert pm.session_count("L0") == 0


# ═══════════════════════════════════════════════════════
# Group 5: A1MtObserverFrozen compatibility
# ═══════════════════════════════════════════════════════

class TestFrozenObserverBootstrap:

    def test_bootstrap_works_on_frozen(self):
        """Frozen observer allows setting hat-state (not in _FROZEN_PARAMS)."""
        obs = A1MtObserverFrozen()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.7, "nu": 0.3, "gamma_gen": 0.15,
                            "gamma_spec": 0.08, "kappa": 0.45},
            confidence={"tau": 0.8, "nu": 0.6, "gamma_gen": 0.5},
        )
        # This should NOT raise
        bootstrap_observer(obs, profile)
        assert abs(obs.tau_hat - 0.7) < 1e-9
        assert abs(obs.kappa_hat - 0.45) < 1e-9

    def test_frozen_still_rejects_frozen_params(self):
        """Frozen observer still guards frozen params after bootstrap."""
        obs = A1MtObserverFrozen()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.5, "nu": 0.2, "gamma_gen": 0.1,
                            "gamma_spec": 0.0, "kappa": 0.3},
        )
        bootstrap_observer(obs, profile)
        with pytest.raises(AttributeError):
            obs.beta_tau_probe = 0.5  # Should still be frozen

    def test_finalize_on_frozen(self):
        obs = A1MtObserverFrozen()
        profile = ProfileState(
            m_hat_terminal={"tau": 0.6, "nu": 0.2, "gamma_gen": 0.1,
                            "gamma_spec": 0.0, "kappa": 0.35},
        )
        bootstrap_observer(obs, profile)
        for _ in range(3):
            obs.update(ObsEvent(dose=0.0, warned=False, risk=0.3))
        result = obs.finalize_to_profile()
        assert "m_hat_terminal" in result
        assert "confidence" in result
        assert result["n_steps"] == 3
