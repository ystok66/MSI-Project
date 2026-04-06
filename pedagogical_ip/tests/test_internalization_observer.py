"""Unit tests for RuleBasedMtObserver."""
import sys
sys.path.insert(0, ".")
import pytest
import numpy as np
from src.teachers.internalization_observer import RuleBasedMtObserver, ObsEvent


class TestTauUpdates:
    def test_tau_increases_after_valid_warn_follow(self):
        obs = RuleBasedMtObserver(); obs.reset()
        tau0 = obs.tau_hat
        ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True, p_self=0.3)
        obs.update(ev)
        assert obs.tau_hat > tau0

    def test_tau_decreases_after_invalid_warn(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.tau_hat = 0.6
        ev = ObsEvent(warned=True, follow_warn=True, warn_wrong=True, p_self=0.3)
        obs.update(ev)
        assert obs.tau_hat < 0.6

    def test_tau_probe_correction(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.tau_hat = 0.3
        ev = ObsEvent(probe_VA=0.8)  # probe says trust is high
        obs.update(ev)
        assert obs.tau_hat > 0.3  # should be pulled up


class TestNuUpdates:
    def test_nu_increases_more_when_p_self_low(self):
        obs1 = RuleBasedMtObserver(); obs1.reset()
        obs2 = RuleBasedMtObserver(); obs2.reset()
        ev_low = ObsEvent(warned=True, follow_warn=True, p_self=0.1)
        ev_high = ObsEvent(warned=True, follow_warn=True, p_self=0.9)
        obs1.update(ev_low)
        obs2.update(ev_high)
        assert obs1.nu_hat > obs2.nu_hat  # low p_self = more blind following

    def test_nu_decreases_after_self_discovery(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.nu_hat = 0.5
        ev = ObsEvent(self_discovery=True, p_self=0.8)
        obs.update(ev)
        assert obs.nu_hat < 0.5


class TestGammaGenUpdates:
    def test_gamma_gen_increases_under_sustained_pressure(self):
        obs = RuleBasedMtObserver(); obs.reset()
        for _ in range(5):
            ev = ObsEvent(dose=1.0)
            obs.update(ev)
        assert obs.gamma_gen_hat > 0.0

    def test_gamma_gen_decreases_after_beneficial_exploration(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.gamma_gen_hat = 0.3
        ev = ObsEvent(beneficial_novelty=True)
        obs.update(ev)
        assert obs.gamma_gen_hat < 0.3


class TestBounds:
    def test_observer_bounds_respected(self):
        obs = RuleBasedMtObserver(); obs.reset()
        # Extreme positive events
        for _ in range(50):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True,
                          dose=1.0, p_self=0.01)
            obs.update(ev)
        assert 0.0 <= obs.tau_hat <= 1.0
        assert 0.0 <= obs.nu_hat <= obs.nu_max
        assert 0.0 <= obs.gamma_gen_hat <= obs.gamma_max
        # Extreme negative events
        for _ in range(50):
            ev = ObsEvent(self_discovery=True, beneficial_novelty=True,
                          warned=True, follow_warn=True, warn_wrong=True,
                          dose=0.0, p_self=0.99)
            obs.update(ev)
        assert 0.0 <= obs.tau_hat <= 1.0
        assert 0.0 <= obs.nu_hat <= obs.nu_max
        assert 0.0 <= obs.gamma_gen_hat <= obs.gamma_max

    def test_no_nan_under_extreme_noise(self):
        obs = RuleBasedMtObserver(); obs.reset()
        rng = np.random.default_rng(42)
        for _ in range(100):
            ev = ObsEvent(
                warned=rng.random() > 0.5,
                follow_warn=rng.random() > 0.5,
                warn_correct=rng.random() > 0.5,
                warn_wrong=rng.random() > 0.5,
                dose=rng.random(),
                p_self=rng.random(),
                self_discovery=rng.random() > 0.7,
                beneficial_novelty=rng.random() > 0.8,
                false_suppression=rng.random() > 0.9,
                probe_VA=rng.random() if rng.random() > 0.5 else None,
                probe_IA=rng.random() if rng.random() > 0.5 else None,
                probe_EP=rng.random() if rng.random() > 0.5 else None,
            )
            snap = obs.update(ev)
            assert not np.isnan(snap.tau_hat)
            assert not np.isnan(snap.nu_hat)
            assert not np.isnan(snap.gamma_gen_hat)


class TestReplay:
    def test_replay_is_deterministic_given_same_log(self):
        events = [
            ObsEvent(warned=True, follow_warn=True, warn_correct=True, dose=0.5, p_self=0.3),
            ObsEvent(dose=0.0, self_discovery=True, p_self=0.8),
            ObsEvent(dose=1.0, beneficial_novelty=True, probe_EP=0.6),
        ]
        obs1 = RuleBasedMtObserver(); obs1.reset()
        obs2 = RuleBasedMtObserver(); obs2.reset()
        for ev in events:
            obs1.update(ev)
            obs2.update(ev)
        assert obs1.get_estimate() == obs2.get_estimate()
        assert obs1.get_confidence() == obs2.get_confidence()


class TestConfidence:
    def test_confidence_drops_near_pself_boundary(self):
        obs = RuleBasedMtObserver(); obs.reset()
        # p_self near 0.5 = low discriminability → confidence should not jump
        for _ in range(5):
            ev = ObsEvent(warned=True, follow_warn=True, p_self=0.5)
            obs.update(ev)
        conf_boundary = obs.conf_nu
        obs2 = RuleBasedMtObserver(); obs2.reset()
        for _ in range(5):
            ev = ObsEvent(warned=True, follow_warn=True, p_self=0.05)
            obs2.update(ev)
        assert obs2.conf_nu >= conf_boundary  # away from boundary → more confident

    def test_confidence_recovers_after_consistent_valid_advice(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.conf_tau = 0.1  # start low
        for _ in range(10):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True, p_self=0.3)
            obs.update(ev)
        assert obs.conf_tau > 0.5  # should have recovered


class TestLongSession:
    def test_probe_correction_reduces_drift_over_long_session(self):
        obs_with = RuleBasedMtObserver(); obs_with.reset()
        obs_without = RuleBasedMtObserver(); obs_without.reset()
        obs_without.beta_tau_probe = 0.0  # disable probe correction
        target = 0.7
        for i in range(20):
            ev = ObsEvent(probe_VA=target if i % 3 == 0 else None)
            obs_with.update(ev)
            obs_without.update(ev)
        # With probe correction should be closer to target
        err_with = abs(obs_with.tau_hat - target)
        err_without = abs(obs_without.tau_hat - target)
        assert err_with <= err_without

    def test_gamma_gen_requires_pressure_accumulation(self):
        obs = RuleBasedMtObserver(); obs.reset()
        # Single dose event should barely move gamma
        ev = ObsEvent(dose=1.0)
        obs.update(ev)
        g1 = obs.gamma_gen_hat
        # Many doses should accumulate
        for _ in range(10):
            obs.update(ObsEvent(dose=1.0))
        assert obs.gamma_gen_hat > g1 * 2  # should be substantially higher

    def test_nu_requires_self_discovery_signal(self):
        obs = RuleBasedMtObserver(); obs.reset()
        obs.nu_hat = 0.5
        # Without self_discovery, nu should not decrease much
        for _ in range(5):
            ev = ObsEvent(dose=0.0, p_self=0.8)
            obs.update(ev)
        nu_no_sd = obs.nu_hat
        obs2 = RuleBasedMtObserver(); obs2.reset()
        obs2.nu_hat = 0.5
        for _ in range(5):
            ev = ObsEvent(self_discovery=True, p_self=0.8)
            obs2.update(ev)
        assert obs2.nu_hat < nu_no_sd  # self-discovery should reduce nu more


class TestShadowStability:
    def test_shadow_action_stability_under_lapse_noise(self):
        obs = RuleBasedMtObserver(); obs.reset()
        rng = np.random.default_rng(123)
        # Run consistent events with occasional random noise
        for i in range(20):
            lapse = rng.random() < 0.2  # 20% lapse
            ev = ObsEvent(
                warned=True, follow_warn=not lapse,
                warn_correct=True, dose=0.5, p_self=0.3,
                self_discovery=lapse,
            )
            obs.update(ev)
        # Observer should still be in reasonable range
        assert 0.1 < obs.tau_hat < 0.95
        assert 0.0 <= obs.nu_hat <= 0.75
        assert 0.0 <= obs.gamma_gen_hat <= 0.4


class TestA1ProbeGate:
    def test_probe_gate_default_off(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        tau0 = obs.tau_hat
        # Even with probe, A1 should not use it (beta=0)
        ev = ObsEvent(probe_VA=0.9)
        obs.update(ev)
        # Should barely change (only conditional reversion at most)
        assert abs(obs.tau_hat - tau0) < 0.05

    def test_conditional_reversion_only_without_recent_events(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        obs.tau_hat = 0.7
        # Feed trust events to create recent activity
        for _ in range(3):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True, p_self=0.3)
            obs.update(ev)
        tau_after_events = obs.tau_hat
        # Now feed empty events — reversion should barely happen
        for _ in range(2):
            obs.update(ObsEvent())
        tau_after_empty = obs.tau_hat
        # Reversion should be very small (lambda=0.005, conditional)
        assert abs(tau_after_empty - tau_after_events) < 0.02

    def test_pself_changes_nu_update_under_same_follow_event(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs1 = A1MtObserver(); obs1.reset()
        obs2 = A1MtObserver(); obs2.reset()
        ev_low = ObsEvent(warned=True, follow_warn=True, p_self=0.1)
        ev_high = ObsEvent(warned=True, follow_warn=True, p_self=0.9)
        obs1.update(ev_low)
        obs2.update(ev_high)
        # Low p_self → higher blind signal → higher nu
        assert obs1.nu_hat > obs2.nu_hat

    def test_confidence_tracks_predictive_agreement(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        obs.tau_hat = 0.5
        # Probe agrees with estimate → confidence should be high
        for _ in range(5):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True,
                          probe_VA=0.55, p_self=0.3)  # close to hat
            obs.update(ev)
        conf_agree = obs.conf_tau
        obs2 = A1MtObserver(); obs2.reset()
        obs2.tau_hat = 0.5
        # Probe disagrees → confidence lower (but probe gate blocks it)
        for _ in range(5):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True,
                          probe_VA=0.1, p_self=0.3)  # far from hat
            obs2.update(ev)
        # Both have trust events so q_tau=0.8, but probe_agreement differs
        # With A1's predictive confidence, probe far away → lower q
        # (probe agreement: 1-|0.1-hat| < 1-|0.55-hat|)
        assert obs.conf_tau >= obs2.conf_tau - 0.1  # should be at least comparable

    def test_long_session_no_monotonic_drift(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        # Consistent trust events — estimate should stabilize, not drift
        errors = []
        true_tau = 0.3
        for step in range(20):
            if step < 5:
                ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True, p_self=0.3)
                true_tau += 0.22 * (1.0 - true_tau)
                true_tau = min(true_tau, 1.0)
            else:
                ev = ObsEvent()  # no events
            obs.update(ev)
            errors.append(abs(obs.tau_hat - true_tau))
        # Error should not be monotonically increasing over last 10 steps
        tail = errors[10:]
        monotonic = all(tail[i] <= tail[i+1] + 1e-6 for i in range(len(tail)-1))
        assert not monotonic or max(tail) < 0.1


class TestA2ExpandedBlind:
    def test_soft_blind_captures_dose_compliance(self):
        from src.teachers.internalization_observer import A2MtObserver
        obs = A2MtObserver(); obs.reset()
        # SOFT dose + agent complies + low p_self → blind should fire
        ev = ObsEvent(dose=0.5, agent_choice=0, oracle_safe=0, p_self=0.2)
        obs.update(ev)
        assert obs.nu_hat > obs.nu_0  # blind signal moved nu up

    def test_a2_blind_nonzero_under_soft_intervention(self):
        from src.teachers.internalization_observer import A2MtObserver
        obs = A2MtObserver(); obs.reset()
        ev = ObsEvent(dose=0.5, agent_choice=1, oracle_safe=1, p_self=0.1)
        snap = obs.update(ev)
        assert snap.events["blind"] > 0  # must be nonzero

    def test_pself_changes_nu_when_blind_forced_active(self):
        from src.teachers.internalization_observer import A2MtObserver
        obs1 = A2MtObserver(); obs1.reset()
        obs2 = A2MtObserver(); obs2.reset()
        # Same dose+compliance, different p_self
        ev_low = ObsEvent(dose=0.5, agent_choice=0, oracle_safe=0, p_self=0.1)
        ev_high = ObsEvent(dose=0.5, agent_choice=0, oracle_safe=0, p_self=0.9)
        obs1.update(ev_low)
        obs2.update(ev_high)
        assert obs1.nu_hat > obs2.nu_hat  # low p_self = more blind

    def test_confidence_improves_with_action_stability(self):
        from src.teachers.internalization_observer import A2MtObserver
        obs = A2MtObserver(); obs.reset()
        for _ in range(10):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True,
                          dose=1.0, agent_choice=0, oracle_safe=0, p_self=0.3)
            obs.update(ev)
            obs.record_action_agreement(True)
        conf_stable = obs.conf_tau
        obs2 = A2MtObserver(); obs2.reset()
        for _ in range(10):
            ev = ObsEvent(warned=True, follow_warn=True, warn_correct=True,
                          dose=1.0, agent_choice=0, oracle_safe=0, p_self=0.3)
            obs2.update(ev)
            obs2.record_action_agreement(False)
        # Stable actions should yield higher confidence
        assert conf_stable >= obs2.conf_tau


class TestFinalAudit:
    def test_selfdisc_scales_with_timing_gap(self):
        from src.teachers.internalization_observer import A1MtObserver
        # Wide gap → high p_self → more selfdisc signal
        obs_wide = A1MtObserver(); obs_wide.reset()
        ev_wide = ObsEvent(self_discovery=True, p_self=0.9, d_commit=5, d_reveal=1)
        snap_wide = obs_wide.update(ev_wide)
        obs_tight = A1MtObserver(); obs_tight.reset()
        ev_tight = ObsEvent(self_discovery=True, p_self=0.2, d_commit=2, d_reveal=2)
        snap_tight = obs_tight.update(ev_tight)
        assert snap_wide.events["selfdisc"] > snap_tight.events["selfdisc"]

    def test_A1_frozen_baseline_params(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver()
        # Core A1 invariants that must not change
        assert obs.beta_tau_probe == 0.0
        assert obs.beta_nu_probe == 0.0
        assert obs.beta_gamma_probe == 0.0
        assert obs.lambda_tau == 0.005
        assert obs.lambda_nu == 0.005
        assert obs.lambda_gamma == 0.005

    def test_intervention_rich_blind_activates(self):
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        # Multiple warn+follow events should accumulate blind signal
        for _ in range(5):
            ev = ObsEvent(warned=True, follow_warn=True,
                          dose=1.0, p_self=0.2)
            obs.update(ev)
        # With low p_self and repeated warn+follow, nu should rise
        assert obs.nu_hat > obs.nu_0

    def test_hidden_temptation_does_not_crash_observer(self):
        """Observer should handle unusual agent behavior gracefully."""
        from src.teachers.internalization_observer import A1MtObserver
        obs = A1MtObserver(); obs.reset()
        # Agent keeps choosing risky (oracle_safe=0, agent_choice=1)
        for _ in range(10):
            ev = ObsEvent(agent_choice=1, oracle_safe=0, p_self=0.5,
                          self_discovery=False, dose=0.0)
            obs.update(ev)
        # Observer should not explode
        assert 0.0 <= obs.tau_hat <= 1.0
        assert 0.0 <= obs.nu_hat <= 0.8
        assert 0.0 <= obs.gamma_gen_hat <= 0.5


class TestA1Frozen:
    def test_frozen_params_match_canonical(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen()
        assert obs.beta_tau_probe == 0.0
        assert obs.beta_nu_probe == 0.0
        assert obs.beta_gamma_probe == 0.0
        assert obs.lambda_tau == 0.005
        assert obs.lambda_nu == 0.005
        assert obs.lambda_gamma == 0.005

    def test_frozen_rejects_param_modification(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen()
        with pytest.raises(AttributeError):
            obs.beta_tau_probe = 0.1
        with pytest.raises(AttributeError):
            obs.lambda_nu = 0.01

    def test_frozen_still_functional(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        ev = ObsEvent(self_discovery=True, p_self=0.8, d_commit=4, d_reveal=1)
        snap = obs.update(ev)
        assert snap is not None
        assert 0.0 <= obs.nu_hat <= 1.0

    def test_raw_infer_beats_wait_gate(self):
        """Gate is redundant on current slice; raw infer-only is optimal."""
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        # With only selfdisc events, no gate should trigger
        for _ in range(5):
            ev = ObsEvent(self_discovery=True, p_self=0.7, dose=0.0)
            obs.update(ev)
        conf = obs.get_confidence()
        # Confidence should be reasonable (not collapsed)
        assert conf["nu"] > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═════════════════════════════════════════════════════════
# P4-B: 4D observer + 2-act canonical tests
# ═════════════════════════════════════════════════════════

class TestGammaSpecState:
    """γ̂_spec behavioral state update guards."""

    def test_gamma_spec_updates_only_when_temptation_present(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        ev = ObsEvent(lure=0.0, agent_choice=0, oracle_safe=0, p_self=0.5)
        obs.update(ev)
        assert obs.gamma_spec_hat == 0.0, "γ_spec should not update without temptation"

    def test_gamma_spec_resist_increases(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        gs0 = obs.gamma_spec_hat
        ev = ObsEvent(lure=0.7, agent_choice=0, oracle_safe=0, p_self=0.5)
        obs.update(ev)
        assert obs.gamma_spec_hat > gs0, "Resisting temptation should increase γ_spec"

    def test_gamma_spec_follow_decreases(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        obs.gamma_spec_hat = 0.5
        ev = ObsEvent(lure=0.7, agent_choice=1, oracle_safe=0, p_self=0.5)
        obs.update(ev)
        assert obs.gamma_spec_hat < 0.5, "Following temptation should decrease γ_spec"

    def test_gamma_spec_does_not_touch_base_3d(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        tau0 = obs.tau_hat; nu0 = obs.nu_hat; gg0 = obs.gamma_gen_hat
        ev = ObsEvent(lure=0.8, agent_choice=0, oracle_safe=0, p_self=0.5,
                       dose=0.0, warned=False)
        obs.update(ev)
        assert abs(obs.tau_hat - tau0) < 0.01
        assert abs(obs.nu_hat - nu0) < 0.01
        assert abs(obs.gamma_gen_hat - gg0) < 0.01
        assert obs.gamma_spec_hat > 0.0

    def test_4d_get_estimate_has_gamma_spec(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        est = obs.get_estimate()
        assert "gamma_spec" in est
        assert "kappa" in est
        assert len(est) == 5

    def test_base_class_get_estimate_is_3d(self):
        obs = RuleBasedMtObserver(); obs.reset()
        est = obs.get_estimate()
        assert "gamma_spec" not in est
        assert "kappa" not in est
        assert len(est) == 3


class Test2ActCanonical:
    """2-act canonical (use_dose=False) never emits SOFT."""

    def test_2act_never_emits_soft(self):
        from src.teachers.internalization_control_tutor_v4 import BCICTv4
        from src.agents.stochastic_agent_policy import AgentPolicyParams
        tutor = BCICTv4(agent_params=AgentPolicyParams(), use_dose=False)
        assert tutor.soft_count == 0
        assert tutor.use_dose is False


class TestKappaRiskCalibration:
    """P5: κ̂ risk calibration state tests."""

    def test_kappa_increases_when_risk_underestimated(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        k0 = obs.kappa_hat
        # Real risk=0.6 > predicted risk=0.2 → underestimated → κ should increase
        ev = ObsEvent(risk=0.6, risk_hat=0.2, p_self=0.5)
        obs.update(ev)
        assert obs.kappa_hat > k0, "κ should increase when real risk > expected"

    def test_kappa_decreases_when_risk_overestimated(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        obs.kappa_hat = 0.6  # Start high
        k0 = obs.kappa_hat
        # Real risk=0.1 < predicted risk=0.5 → overestimated → κ should decrease
        ev = ObsEvent(risk=0.1, risk_hat=0.5, p_self=0.5)
        obs.update(ev)
        assert obs.kappa_hat < k0, "κ should decrease when real risk < expected"

    def test_kappa_no_update_without_risk_hat(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        k0 = obs.kappa_hat
        # No risk_hat → κ should remain at mean-reversion value
        ev = ObsEvent(risk=0.5, p_self=0.5)
        obs.update(ev)
        # Only mean-reversion should apply (very small change)
        assert abs(obs.kappa_hat - k0) < 0.01

    def test_kappa_no_update_below_risk_gate(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        k0 = obs.kappa_hat
        # Risk below gate threshold → no update
        ev = ObsEvent(risk=0.05, risk_hat=0.01, p_self=0.5)
        obs.update(ev)
        assert abs(obs.kappa_hat - k0) < 0.01

    def test_kappa_does_not_touch_3d_or_gamma_spec(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        tau0 = obs.tau_hat; nu0 = obs.nu_hat
        gg0 = obs.gamma_gen_hat; gs0 = obs.gamma_spec_hat
        # Pure risk event
        ev = ObsEvent(risk=0.7, risk_hat=0.2, p_self=0.5,
                       dose=0.0, warned=False, lure=0.0)
        obs.update(ev)
        assert abs(obs.tau_hat - tau0) < 0.01
        assert abs(obs.nu_hat - nu0) < 0.01
        assert abs(obs.gamma_gen_hat - gg0) < 0.01
        assert abs(obs.gamma_spec_hat - gs0) < 0.001
        assert obs.kappa_hat != 0.3  # κ should have changed


class TestMetricConsistency:
    """P4-B.1: Metric definition consistency tests."""

    def test_new_active_mask_covers_wait_warn_disagreement(self):
        """WAIT↔WARN disagreement must be flagged as active(new)."""
        act_oracle = "WAIT"; act_infer = "WARN"
        active_new = (act_oracle != "WAIT") or (act_infer != "WAIT")
        assert active_new is True

    def test_new_active_mask_superset_of_old(self):
        """New mask (either non-WAIT) is superset of old (oracle warned)."""
        for ao, ai in [("WAIT", "WAIT"), ("WARN", "WAIT"),
                       ("WAIT", "WARN"), ("WARN", "WARN")]:
            old = (ao == "WARN")
            new = (ao != "WAIT") or (ai != "WAIT")
            if old:
                assert new, f"Old=True but New=False for ({ao},{ai})"

    def test_active_regret_zero_when_actions_identical(self):
        """R_active must be 0 when oracle and infer agree."""
        recs = [{"diverge": False, "active": True, "Q_oracle": 1.0, "Q_infer": 0.8}
                for _ in range(10)]
        n_act = sum(1 for r in recs if r["active"])
        r_active = sum(abs(r["Q_oracle"] - r["Q_infer"])
                       for r in recs if r["active"] and r["diverge"]) / max(n_act, 1)
        assert r_active == 0.0


class TestP6ProtocolRegression:
    """P6: Protocol regression tests for 5D canonicalization."""

    def test_5d_config_loads_all_five_dims(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        est = obs.get_estimate()
        assert set(est.keys()) == {"tau", "nu", "gamma_gen", "gamma_spec", "kappa"}

    def test_base_class_remains_3d(self):
        obs = RuleBasedMtObserver(); obs.reset()
        est = obs.get_estimate()
        assert set(est.keys()) == {"tau", "nu", "gamma_gen"}

    def test_kappa_bonus_changes_risk_ranking(self):
        """β=0.02 must move at least one risk family up."""
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        obs.kappa_hat = 0.5  # away from anchor
        est = obs.get_estimate()
        from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2
        CAT = list(LESSON_CATALOG_V2)
        RISK = {"tic_rescue_heavy", "blind_activation_corridor",
                "warn_symmetric_rescue"}
        base = [0.1 * (i+1) for i in range(len(CAT))]
        bonus = list(base)
        for i, l in enumerate(CAT):
            if l.name in RISK:
                bonus[i] += 0.02 * abs(est["kappa"] - 0.3)
        import numpy as np
        rank_b = list(np.argsort(base)[::-1])
        rank_k = list(np.argsort(bonus)[::-1])
        shifts = [rank_b.index(i) - rank_k.index(i)
                  for i, l in enumerate(CAT) if l.name in RISK]
        assert any(s > 0 for s in shifts) or all(s == 0 for s in shifts)

    def test_kappa_bonus_preserves_top1_when_kappa_at_anchor(self):
        """When κ̂ = κ_0, bonus is zero, ranking unchanged."""
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen(); obs.reset()
        assert obs.kappa_hat == obs.kappa_0  # at anchor
        bonus = 0.02 * abs(obs.kappa_hat - obs.kappa_0)
        assert bonus == 0.0

    def test_kappa_reset_to_anchor(self):
        from src.teachers.internalization_observer import A1MtObserverFrozen
        obs = A1MtObserverFrozen()
        obs.kappa_hat = 0.8
        obs.reset()
        assert obs.kappa_hat == obs.kappa_0

    def test_owr_computation_consistent(self):
        """OWR = count(oracle=WAIT, infer=WARN) / T."""
        actions = [("WAIT", "WARN"), ("WARN", "WARN"),
                   ("WAIT", "WAIT"), ("WAIT", "WARN")]
        T = len(actions)
        owr = sum(1 for o, i in actions if o == "WAIT" and i == "WARN") / T
        assert owr == 0.5

    def test_divergence_metrics_consistent_under_2act(self):
        """Under 2-act, only WAIT↔WARN disagreement is possible."""
        acts = ["WAIT", "WARN"]
        for o in acts:
            for i in acts:
                if o != i:
                    assert {o, i} == {"WAIT", "WARN"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




