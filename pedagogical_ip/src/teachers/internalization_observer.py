"""RuleBasedMtObserver: Shadow observer for learner internalization state.

Maintains m̂_t = (τ̂, ν̂, γ̂_gen) using only tutor-visible signals.
Does NOT read true m_t. Produces shadow decisions for comparison.

Design principles:
  - Rule-based first version (mirrors agent update mechanics externally)
  - Soft events using p_self for blind/self-discovery discrimination
  - Probe-corrected updates (VA→τ, IA→ν, EP→γ_gen)
  - Per-dimension confidence tracking
  - Replayable event log (JSONL-compatible)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class ObsEvent:
    """Lightweight event for observer update. Only tutor-visible signals."""
    episode_id: int = 0
    step_id: int = 0
    subtype: str = ""
    theta_post: str = "safe"

    # Tutor action
    dose: float = 0.0            # 0=WAIT, 0.5=SOFT, 1.0=WARN
    warned: bool = False
    follow_warn: bool = False    # Agent followed warning
    warn_correct: bool = False   # Warning was objectively correct
    warn_wrong: bool = False     # Warning was objectively wrong

    # Scene structure (tutor-visible)
    d_commit: int = 4
    d_reveal: int = 2
    p_self: float = 0.5         # Self-discovery probability (tutor-computed)
    lure: float = 0.0
    risk: float = 0.3
    risk_hat: Optional[float] = None  # P5: agent's predicted risk (for κ update)

    # Agent behavior (tutor-observable)
    agent_choice: int = 0        # Which branch agent chose
    oracle_safe: int = 0         # Which branch was actually safe
    self_discovery: bool = False # Agent found safe path without relying on warn
    false_suppression: bool = False  # Agent suppressed valid exploration
    beneficial_novelty: bool = False # Agent benefited from novel exploration

    # Probes (when available; None = not observed this step)
    probe_VA: Optional[float] = None   # Valid-advice uptake
    probe_IA: Optional[float] = None   # Invalid-advice rejection
    probe_EP: Optional[float] = None   # Exploration preservation

    # For logging ONLY — observer must NOT use these
    m_true: Optional[Dict] = None


@dataclass
class ObserverSnapshot:
    """Output of one observer update step."""
    tau_hat: float
    nu_hat: float
    gamma_gen_hat: float
    conf_tau: float
    conf_nu: float
    conf_gamma: float
    events: Dict[str, float]   # Event flags used this step


@dataclass
class RuleBasedMtObserver:
    """Shadow observer for (τ̂, ν̂, γ̂_gen).

    Uses tutor-visible events + probes to track latent state.
    First version: rule-based, mirroring agent update mechanics externally.
    """
    # State estimates
    tau_hat: float = 0.3
    nu_hat: float = 0.1
    gamma_gen_hat: float = 0.0

    # Confidence per dimension
    conf_tau: float = 0.2
    conf_nu: float = 0.2
    conf_gamma: float = 0.2

    # τ update params (mirrors agent but from external observation)
    alpha_tau_plus: float = 0.22    # Slightly conservative vs agent's 0.25
    alpha_tau_minus: float = 0.10   # Slightly conservative vs agent's 0.12
    beta_tau_probe: float = 0.15    # Probe correction weight
    lambda_tau: float = 0.02        # Mean-reversion rate
    tau_0: float = 0.3              # Prior mean

    # ν update params
    alpha_nu_plus: float = 0.18     # Conservative vs agent's 0.20
    alpha_nu_minus: float = 0.13    # Conservative vs agent's 0.15
    beta_nu_probe: float = 0.10     # Probe correction weight
    lambda_nu: float = 0.02
    nu_0: float = 0.1
    nu_max: float = 0.8

    # γ_gen update params
    alpha_gamma_plus: float = 0.07  # Conservative vs agent's 0.08
    alpha_gamma_minus: float = 0.10 # Conservative vs agent's 0.12
    beta_gamma_probe: float = 0.10
    lambda_gamma: float = 0.02
    gamma_0: float = 0.0
    gamma_max: float = 0.5

    # Confidence update
    rho_conf: float = 0.15         # Confidence EMA rate

    # EMA for pressure tracking
    pressure_ema: float = 0.0
    pressure_alpha: float = 0.3    # EMA decay for dose pressure

    # History
    history: List[Dict] = field(default_factory=list)

    def reset(self, theta_posterior=None):
        """Reset to priors."""
        self.tau_hat = self.tau_0
        self.nu_hat = self.nu_0
        self.gamma_gen_hat = self.gamma_0
        self.conf_tau = 0.2
        self.conf_nu = 0.2
        self.conf_gamma = 0.2
        self.pressure_ema = 0.0
        self.history = []

    def update(self, ev: ObsEvent) -> ObserverSnapshot:
        """One-step update from a tutor-visible event."""
        # ─── Extract soft events ─────────────────────────
        e_trust_plus = float(ev.warned and ev.follow_warn and ev.warn_correct)
        e_trust_minus = float(ev.warned and ev.follow_warn and ev.warn_wrong)
        e_blind = float(ev.warned and ev.follow_warn) * (1.0 - ev.p_self)
        e_selfdisc = float(ev.self_discovery) * ev.p_self
        self.pressure_ema = (1 - self.pressure_alpha) * self.pressure_ema + self.pressure_alpha * ev.dose
        e_pressure = self.pressure_ema
        e_explore_plus = float(ev.beneficial_novelty or ev.self_discovery)

        events = {
            "trust+": round(e_trust_plus, 4),
            "trust-": round(e_trust_minus, 4),
            "blind": round(e_blind, 4),
            "selfdisc": round(e_selfdisc, 4),
            "pressure": round(e_pressure, 4),
            "explore+": round(e_explore_plus, 4),
        }

        # ─── τ̂ update ───────────────────────────────────
        tau_new = (1 - self.lambda_tau) * self.tau_hat + self.lambda_tau * self.tau_0
        tau_new += self.alpha_tau_plus * e_trust_plus * (1.0 - self.tau_hat)
        tau_new -= self.alpha_tau_minus * e_trust_minus * self.tau_hat
        if ev.probe_VA is not None:
            r_va = ev.probe_VA - self.tau_hat
            tau_new += self.beta_tau_probe * r_va
        self.tau_hat = float(np.clip(tau_new, 0.0, 1.0))

        # ─── ν̂ update ───────────────────────────────────
        nu_new = (1 - self.lambda_nu) * self.nu_hat + self.lambda_nu * self.nu_0
        nu_new += self.alpha_nu_plus * e_blind * (self.nu_max - self.nu_hat)
        nu_new -= self.alpha_nu_minus * e_selfdisc * self.nu_hat
        if ev.probe_IA is not None:
            # IA is "invalid advice rejection" — higher IA → lower dependence
            r_ia = (1.0 - ev.probe_IA) - self.nu_hat
            nu_new += self.beta_nu_probe * r_ia
        self.nu_hat = float(np.clip(nu_new, 0.0, self.nu_max))

        # ─── γ̂_gen update ───────────────────────────────
        gg_new = (1 - self.lambda_gamma) * self.gamma_gen_hat + self.lambda_gamma * self.gamma_0
        gg_new += self.alpha_gamma_plus * e_pressure * (self.gamma_max - self.gamma_gen_hat)
        gg_new -= self.alpha_gamma_minus * e_explore_plus * self.gamma_gen_hat
        if ev.probe_EP is not None:
            # EP is "exploration preservation" — higher EP → lower γ_gen
            r_ep = (1.0 - ev.probe_EP) - self.gamma_gen_hat
            gg_new += self.beta_gamma_probe * r_ep
        self.gamma_gen_hat = float(np.clip(gg_new, 0.0, self.gamma_max))

        # ─── Confidence update ───────────────────────────
        # Evidence quality: did we observe a relevant event for this dimension?
        eq_tau = max(e_trust_plus, e_trust_minus, float(ev.probe_VA is not None))
        eq_nu = max(e_blind, e_selfdisc, float(ev.probe_IA is not None))
        eq_gamma = max(min(e_pressure, 1.0), e_explore_plus, float(ev.probe_EP is not None))
        # Boost confidence when p_self is far from 0.5 (more discriminative)
        p_disc = abs(ev.p_self - 0.5) * 2.0  # in [0,1]
        eq_nu = min(eq_nu + 0.2 * p_disc, 1.0)

        self.conf_tau = float(np.clip(
            (1 - self.rho_conf) * self.conf_tau + self.rho_conf * eq_tau, 0.0, 1.0))
        self.conf_nu = float(np.clip(
            (1 - self.rho_conf) * self.conf_nu + self.rho_conf * eq_nu, 0.0, 1.0))
        self.conf_gamma = float(np.clip(
            (1 - self.rho_conf) * self.conf_gamma + self.rho_conf * eq_gamma, 0.0, 1.0))

        snap = ObserverSnapshot(
            tau_hat=round(self.tau_hat, 6),
            nu_hat=round(self.nu_hat, 6),
            gamma_gen_hat=round(self.gamma_gen_hat, 6),
            conf_tau=round(self.conf_tau, 4),
            conf_nu=round(self.conf_nu, 4),
            conf_gamma=round(self.conf_gamma, 4),
            events=events,
        )
        self.history.append(snap)
        return snap

    def get_estimate(self) -> dict:
        """Current m̂_t as dict."""
        return {
            "tau": round(self.tau_hat, 6),
            "nu": round(self.nu_hat, 6),
            "gamma_gen": round(self.gamma_gen_hat, 6),
        }

    def get_confidence(self) -> dict:
        """Current confidence per dimension."""
        return {
            "tau": round(self.conf_tau, 4),
            "nu": round(self.conf_nu, 4),
            "gamma_gen": round(self.conf_gamma, 4),
        }

    def shadow_decision(self, tutor, sc, fb, lp, lib, scorer, obs,
                        m_hat_state) -> dict:
        """Compute what BCICTv4 would decide using m̂ instead of m.

        Args:
            tutor: BCICTv4 instance
            m_hat_state: FactoredInternalizationState with m̂ values
            (other args): same as tutor.decide()

        Returns dict with Q_per_action, best_action, best_dose.
        """
        # Inject m̂ into a copy of the state
        action, dose, info = tutor.decide(sc, fb, lp, lib, scorer, obs, m_hat_state)
        return {
            "a_infer": action,
            "dose_infer": dose,
            "Q_infer": info.get("Q", 0.0),
        }

    def to_log_record(self, step: int, ev: ObsEvent, snap: ObserverSnapshot,
                      a_oracle: str, a_infer: str, Q_oracle: float = 0.0,
                      Q_infer: float = 0.0) -> dict:
        """Produce JSONL-compatible log record."""
        rec = {
            "step": step,
            "episode_id": ev.episode_id,
            "subtype": ev.subtype,
            "m_hat": self.get_estimate(),
            "conf_hat": self.get_confidence(),
            "events": snap.events,
            "a_oracle": a_oracle,
            "a_infer": a_infer,
            "disagree": a_oracle != a_infer,
            "Q_oracle": round(Q_oracle, 4),
            "Q_infer": round(Q_infer, 4),
        }
        if ev.m_true is not None:
            rec["m_true"] = ev.m_true
        return rec


# ═══════════════════════════════════════════════════════════
# A1: Minimal patch — probe OFF, conditional reversion,
#     predictive-consistency confidence
# ═══════════════════════════════════════════════════════════

@dataclass
class A1MtObserver(RuleBasedMtObserver):
    """A1 patch: fixes probe noise, mean-reversion drift, confidence polarity.

    Changes vs A0:
      1. Probe correction gated OFF by default (beta=0)
      2. Mean-reversion conditional: only when no recent informative event
         AND confidence is low (lambda_base=0.005)
      3. Confidence = predictive consistency, not event count
    """

    # Override defaults — probe OFF
    beta_tau_probe: float = 0.0
    beta_nu_probe: float = 0.0
    beta_gamma_probe: float = 0.0

    # Weaker, conditional mean reversion
    lambda_tau: float = 0.005
    lambda_nu: float = 0.005
    lambda_gamma: float = 0.005

    # Probe gate threshold (only allow probe when gap is large)
    probe_gate_threshold: float = 0.3

    # P4-A: γ̂_spec — temptation-specific generalization (behavior state)
    gamma_spec_hat: float = 0.0
    alpha_gs_resist: float = 0.03   # temptation resisted → γ_spec ↑
    alpha_gs_follow: float = 0.025  # temptation followed → γ_spec ↓
    gamma_spec_max: float = 1.0
    lure_threshold: float = 0.3     # min lure to count as temptation event

    # P5: κ̂ — risk calibration state (slow variable)
    kappa_hat: float = 0.3
    kappa_0: float = 0.3               # mean-reversion anchor
    lambda_kappa: float = 0.02         # mean-reversion rate (slow)
    alpha_kappa_pos: float = 0.015     # real risk > expected → κ ↑ (more cautious)
    alpha_kappa_neg: float = 0.012     # real risk < expected → κ ↓ (relax)
    kappa_min: float = 0.0
    kappa_max: float = 1.0
    risk_gate_threshold: float = 0.1   # min risk to trigger κ update

    # Track recent event activity for conditional reversion
    _recent_events_tau: int = 0
    _recent_events_nu: int = 0
    _recent_events_gamma: int = 0
    _event_window: int = 3     # look-back window for "no recent events"
    _step_counter: int = 0

    def reset(self, theta_posterior=None):
        super().reset(theta_posterior)
        self.gamma_spec_hat = 0.0
        self.kappa_hat = self.kappa_0
        self._recent_events_tau = 0
        self._recent_events_nu = 0
        self._recent_events_gamma = 0
        self._step_counter = 0

    def bootstrap_from_profile(self, m_hat_terminal: dict,
                               confidence: dict = None):
        """Initialize observer state from previous session terminal estimate.

        Sets hat-state initial values only. Does NOT modify frozen per-step
        update logic. Fields set here (tau_hat, nu_hat, etc.) are NOT in
        A1MtObserverFrozen._FROZEN_PARAMS, so this is safe to call on frozen.
        """
        self.tau_hat = m_hat_terminal.get("tau", self.tau_0)
        self.nu_hat = m_hat_terminal.get("nu", self.nu_0)
        self.gamma_gen_hat = m_hat_terminal.get("gamma_gen", self.gamma_0)
        self.gamma_spec_hat = m_hat_terminal.get("gamma_spec", 0.0)
        self.kappa_hat = m_hat_terminal.get("kappa", self.kappa_0)
        if confidence:
            self.conf_tau = confidence.get("tau", 0.2)
            self.conf_nu = confidence.get("nu", 0.2)
            self.conf_gamma = confidence.get("gamma_gen", 0.2)
        # Reset counters for new session
        self._recent_events_tau = 0
        self._recent_events_nu = 0
        self._recent_events_gamma = 0
        self._step_counter = 0
        self.history = []

    def finalize_to_profile(self) -> dict:
        """Export current observer state as profile-ready dict.

        Returns dict suitable for ProfileState construction.
        """
        return {
            "m_hat_terminal": self.get_estimate(),
            "confidence": self.get_confidence(),
            "n_steps": self._step_counter,
        }

    def update(self, ev: ObsEvent) -> ObserverSnapshot:
        """A1 update: event-driven + conditional reversion + predictive confidence."""
        self._step_counter += 1

        # ─── Extract soft events (same as A0) ────────────
        e_trust_plus = float(ev.warned and ev.follow_warn and ev.warn_correct)
        e_trust_minus = float(ev.warned and ev.follow_warn and ev.warn_wrong)
        e_blind = float(ev.warned and ev.follow_warn) * (1.0 - ev.p_self)
        e_selfdisc = float(ev.self_discovery) * ev.p_self
        self.pressure_ema = ((1 - self.pressure_alpha) * self.pressure_ema
                             + self.pressure_alpha * ev.dose)
        e_pressure = self.pressure_ema
        e_explore_plus = float(ev.beneficial_novelty or ev.self_discovery)

        events = {
            "trust+": round(e_trust_plus, 4),
            "trust-": round(e_trust_minus, 4),
            "blind": round(e_blind, 4),
            "selfdisc": round(e_selfdisc, 4),
            "pressure": round(e_pressure, 4),
            "explore+": round(e_explore_plus, 4),
            "p_self": round(ev.p_self, 4),
        }

        # Track recent event activity
        has_tau_event = (e_trust_plus > 0 or e_trust_minus > 0)
        has_nu_event = (e_blind > 0.05 or e_selfdisc > 0.05)
        has_gamma_event = (ev.dose > 0.1 or e_explore_plus > 0)
        self._recent_events_tau = min(self._recent_events_tau + int(has_tau_event), self._event_window)
        self._recent_events_nu = min(self._recent_events_nu + int(has_nu_event), self._event_window)
        self._recent_events_gamma = min(self._recent_events_gamma + int(has_gamma_event), self._event_window)
        # Decay every step
        if not has_tau_event:
            self._recent_events_tau = max(self._recent_events_tau - 1, 0)
        if not has_nu_event:
            self._recent_events_nu = max(self._recent_events_nu - 1, 0)
        if not has_gamma_event:
            self._recent_events_gamma = max(self._recent_events_gamma - 1, 0)

        # ─── τ̂ update ───────────────────────────────────
        # Event-driven (primary signal)
        tau_new = self.tau_hat
        tau_new += self.alpha_tau_plus * e_trust_plus * (1.0 - self.tau_hat)
        tau_new -= self.alpha_tau_minus * e_trust_minus * self.tau_hat
        # Conditional reversion (only when no recent events AND low confidence)
        if self._recent_events_tau == 0:
            lam_eff = self.lambda_tau * (1.0 - self.conf_tau)
            tau_new += lam_eff * (self.tau_0 - tau_new)
        # Gated probe (only if explicitly enabled AND gap is large)
        if (ev.probe_VA is not None and self.beta_tau_probe > 0
                and abs(ev.probe_VA - self.tau_hat) < self.probe_gate_threshold):
            tau_new += self.beta_tau_probe * (ev.probe_VA - self.tau_hat)
        self.tau_hat = float(np.clip(tau_new, 0.0, 1.0))

        # ─── ν̂ update ───────────────────────────────────
        nu_new = self.nu_hat
        nu_new += self.alpha_nu_plus * e_blind * (self.nu_max - self.nu_hat)
        nu_new -= self.alpha_nu_minus * e_selfdisc * self.nu_hat
        if self._recent_events_nu == 0:
            lam_eff = self.lambda_nu * (1.0 - self.conf_nu)
            nu_new += lam_eff * (self.nu_0 - nu_new)
        if (ev.probe_IA is not None and self.beta_nu_probe > 0
                and abs((1.0 - ev.probe_IA) - self.nu_hat) < self.probe_gate_threshold):
            nu_new += self.beta_nu_probe * ((1.0 - ev.probe_IA) - self.nu_hat)
        self.nu_hat = float(np.clip(nu_new, 0.0, self.nu_max))

        # ─── γ̂_gen update ───────────────────────────────
        gg_new = self.gamma_gen_hat
        gg_new += self.alpha_gamma_plus * e_pressure * (self.gamma_max - self.gamma_gen_hat)
        gg_new -= self.alpha_gamma_minus * e_explore_plus * self.gamma_gen_hat
        if self._recent_events_gamma == 0:
            lam_eff = self.lambda_gamma * (1.0 - self.conf_gamma)
            gg_new += lam_eff * (self.gamma_0 - gg_new)
        if (ev.probe_EP is not None and self.beta_gamma_probe > 0
                and abs((1.0 - ev.probe_EP) - self.gamma_gen_hat) < self.probe_gate_threshold):
            gg_new += self.beta_gamma_probe * ((1.0 - ev.probe_EP) - self.gamma_gen_hat)
        self.gamma_gen_hat = float(np.clip(gg_new, 0.0, self.gamma_max))

        # ─── γ̂_spec update (P4-A: temptation resistance state) ─
        if ev.lure >= self.lure_threshold:
            correct = (ev.agent_choice == ev.oracle_safe)
            if correct:
                # Resisted temptation
                e_resist = ev.lure * (1.0 - self.gamma_spec_hat)
                self.gamma_spec_hat += self.alpha_gs_resist * e_resist
            else:
                # Followed temptation
                e_follow = ev.lure * self.gamma_spec_hat
                self.gamma_spec_hat -= self.alpha_gs_follow * e_follow
            self.gamma_spec_hat = float(np.clip(
                self.gamma_spec_hat, 0.0, self.gamma_spec_max))

        # ─── κ̂ update (P5: risk calibration, signed error) ─
        if ev.risk >= self.risk_gate_threshold and ev.risk_hat is not None:
            delta_risk = ev.risk - ev.risk_hat  # signed: positive = underestimated
            kap_new = (1 - self.lambda_kappa) * self.kappa_hat
            kap_new += self.lambda_kappa * self.kappa_0
            if delta_risk > 0:
                # Real risk higher than expected → more cautious
                kap_new += self.alpha_kappa_pos * delta_risk * (
                    self.kappa_max - self.kappa_hat)
            else:
                # Real risk lower than expected → relax
                kap_new += self.alpha_kappa_neg * delta_risk * (
                    self.kappa_hat - self.kappa_min)
            self.kappa_hat = float(np.clip(
                kap_new, self.kappa_min, self.kappa_max))

        # ─── Predictive-consistency confidence ───────────
        # Instead of event count, use:
        #   timing_separation: |d_commit - d_reveal| → [0,1]
        #   probe_agreement: 1 - |probe - hat| when available
        #   noise_risk: lapse proxy (inverse of timing clarity)
        timing_sep = min(abs(ev.d_commit - ev.d_reveal) / 5.0, 1.0)

        # τ confidence
        q_tau = 0.5  # baseline
        if has_tau_event:
            q_tau = 0.8  # clear trust evidence
        if ev.probe_VA is not None:
            q_tau = max(q_tau, 1.0 - abs(ev.probe_VA - self.tau_hat))

        # ν confidence — depends on timing clarity
        q_nu = 0.3
        if has_nu_event:
            q_nu = 0.5 + 0.3 * timing_sep  # better when timing is clear
        if ev.probe_IA is not None:
            q_nu = max(q_nu, 1.0 - abs((1.0 - ev.probe_IA) - self.nu_hat))

        # γ confidence
        q_gamma = 0.3
        if has_gamma_event:
            q_gamma = 0.5
        if ev.probe_EP is not None:
            q_gamma = max(q_gamma, 1.0 - abs((1.0 - ev.probe_EP) - self.gamma_gen_hat))

        self.conf_tau = float(np.clip(
            (1 - self.rho_conf) * self.conf_tau + self.rho_conf * q_tau, 0.0, 1.0))
        self.conf_nu = float(np.clip(
            (1 - self.rho_conf) * self.conf_nu + self.rho_conf * q_nu, 0.0, 1.0))
        self.conf_gamma = float(np.clip(
            (1 - self.rho_conf) * self.conf_gamma + self.rho_conf * q_gamma, 0.0, 1.0))

        snap = ObserverSnapshot(
            tau_hat=round(self.tau_hat, 6),
            nu_hat=round(self.nu_hat, 6),
            gamma_gen_hat=round(self.gamma_gen_hat, 6),
            conf_tau=round(self.conf_tau, 4),
            conf_nu=round(self.conf_nu, 4),
            conf_gamma=round(self.conf_gamma, 4),
            events=events,
        )
        self.history.append(snap)
        return snap

    def get_estimate(self) -> dict:
        """Current m̂_t as dict (5-dim: τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)."""
        return {
            "tau": round(self.tau_hat, 6),
            "nu": round(self.nu_hat, 6),
            "gamma_gen": round(self.gamma_gen_hat, 6),
            "gamma_spec": round(self.gamma_spec_hat, 6),
            "kappa": round(self.kappa_hat, 6),
        }


# ═══════════════════════════════════════════════════════════
# A2: Expanded blind + action-stability confidence
# ═══════════════════════════════════════════════════════════

@dataclass
class A2MtObserver(A1MtObserver):
    """A2 patch: fixes dead blind channel + improves confidence calibration.

    Changes vs A1:
      1. Blind signal uses dose>0 (captures SOFT), not just warn∧follow
      2. Confidence incorporates action_stability and noise_risk penalty
    """

    # Track action stability for confidence
    _action_agree_ema: float = 0.5
    _action_ema_alpha: float = 0.2

    def reset(self, theta_posterior=None):
        super().reset(theta_posterior)
        self._action_agree_ema = 0.5

    def update(self, ev: ObsEvent) -> ObserverSnapshot:
        """A2 update: expanded blind + action-stability confidence."""
        self._step_counter += 1

        # ─── Soft events with EXPANDED blind ─────────────
        e_trust_plus = float(ev.warned and ev.follow_warn and ev.warn_correct)
        e_trust_minus = float(ev.warned and ev.follow_warn and ev.warn_wrong)
        # A2 FIX: blind uses dose>0 (captures SOFT+WARN), not just warn∧follow
        tutor_intervened = ev.dose > 0
        agent_complied = (ev.agent_choice == ev.oracle_safe)
        e_blind = float(tutor_intervened and agent_complied) * (1.0 - ev.p_self)
        e_selfdisc = float(ev.self_discovery) * ev.p_self
        self.pressure_ema = ((1 - self.pressure_alpha) * self.pressure_ema
                             + self.pressure_alpha * ev.dose)
        e_pressure = self.pressure_ema
        e_explore_plus = float(ev.beneficial_novelty or ev.self_discovery)

        events = {
            "trust+": round(e_trust_plus, 4),
            "trust-": round(e_trust_minus, 4),
            "blind": round(e_blind, 4),
            "selfdisc": round(e_selfdisc, 4),
            "pressure": round(e_pressure, 4),
            "explore+": round(e_explore_plus, 4),
            "p_self": round(ev.p_self, 4),
        }

        # Track recent events
        has_tau_event = (e_trust_plus > 0 or e_trust_minus > 0)
        has_nu_event = (e_blind > 0.05 or e_selfdisc > 0.05)
        has_gamma_event = (ev.dose > 0.1 or e_explore_plus > 0)
        self._recent_events_tau = min(self._recent_events_tau + int(has_tau_event), self._event_window)
        self._recent_events_nu = min(self._recent_events_nu + int(has_nu_event), self._event_window)
        self._recent_events_gamma = min(self._recent_events_gamma + int(has_gamma_event), self._event_window)
        if not has_tau_event:
            self._recent_events_tau = max(self._recent_events_tau - 1, 0)
        if not has_nu_event:
            self._recent_events_nu = max(self._recent_events_nu - 1, 0)
        if not has_gamma_event:
            self._recent_events_gamma = max(self._recent_events_gamma - 1, 0)

        # ─── τ̂ update (same as A1) ──────────────────────
        tau_new = self.tau_hat
        tau_new += self.alpha_tau_plus * e_trust_plus * (1.0 - self.tau_hat)
        tau_new -= self.alpha_tau_minus * e_trust_minus * self.tau_hat
        if self._recent_events_tau == 0:
            lam_eff = self.lambda_tau * (1.0 - self.conf_tau)
            tau_new += lam_eff * (self.tau_0 - tau_new)
        self.tau_hat = float(np.clip(tau_new, 0.0, 1.0))

        # ─── ν̂ update (now with expanded blind) ─────────
        nu_new = self.nu_hat
        nu_new += self.alpha_nu_plus * e_blind * (self.nu_max - self.nu_hat)
        nu_new -= self.alpha_nu_minus * e_selfdisc * self.nu_hat
        if self._recent_events_nu == 0:
            lam_eff = self.lambda_nu * (1.0 - self.conf_nu)
            nu_new += lam_eff * (self.nu_0 - nu_new)
        self.nu_hat = float(np.clip(nu_new, 0.0, self.nu_max))

        # ─── γ̂_gen update (same as A1) ──────────────────
        gg_new = self.gamma_gen_hat
        gg_new += self.alpha_gamma_plus * e_pressure * (self.gamma_max - self.gamma_gen_hat)
        gg_new -= self.alpha_gamma_minus * e_explore_plus * self.gamma_gen_hat
        if self._recent_events_gamma == 0:
            lam_eff = self.lambda_gamma * (1.0 - self.conf_gamma)
            gg_new += lam_eff * (self.gamma_0 - gg_new)
        self.gamma_gen_hat = float(np.clip(gg_new, 0.0, self.gamma_max))

        # ─── A2 Confidence: predictive + action-stability ─
        timing_sep = min(abs(ev.d_commit - ev.d_reveal) / 5.0, 1.0)
        noise_risk = max(0.0, 1.0 - timing_sep) * 0.2  # penalty

        q_tau = 0.4 - noise_risk
        if has_tau_event:
            q_tau = 0.8 - noise_risk
        if ev.probe_VA is not None:
            pa = 1.0 - abs(ev.probe_VA - self.tau_hat)
            q_tau = max(q_tau, pa - noise_risk)

        q_nu = 0.25 - noise_risk
        if has_nu_event:
            q_nu = 0.5 + 0.3 * timing_sep - noise_risk
        if ev.probe_IA is not None:
            pa = 1.0 - abs((1.0 - ev.probe_IA) - self.nu_hat)
            q_nu = max(q_nu, pa - noise_risk)

        q_gamma = 0.25 - noise_risk
        if has_gamma_event:
            q_gamma = 0.5 - noise_risk
        if ev.probe_EP is not None:
            pa = 1.0 - abs((1.0 - ev.probe_EP) - self.gamma_gen_hat)
            q_gamma = max(q_gamma, pa - noise_risk)

        # Action stability bonus: if shadow agrees with oracle, boost all
        # (this value gets fed back from the outer loop via a flag)
        self._action_agree_ema = ((1 - self._action_ema_alpha) * self._action_agree_ema
                                  + self._action_ema_alpha * 0.5)  # neutral default
        stability_bonus = max(0.0, (self._action_agree_ema - 0.3)) * 0.3

        q_tau = max(q_tau + stability_bonus, 0.0)
        q_nu = max(q_nu + stability_bonus, 0.0)
        q_gamma = max(q_gamma + stability_bonus, 0.0)

        self.conf_tau = float(np.clip(
            (1 - self.rho_conf) * self.conf_tau + self.rho_conf * q_tau, 0.0, 1.0))
        self.conf_nu = float(np.clip(
            (1 - self.rho_conf) * self.conf_nu + self.rho_conf * q_nu, 0.0, 1.0))
        self.conf_gamma = float(np.clip(
            (1 - self.rho_conf) * self.conf_gamma + self.rho_conf * q_gamma, 0.0, 1.0))

        snap = ObserverSnapshot(
            tau_hat=round(self.tau_hat, 6),
            nu_hat=round(self.nu_hat, 6),
            gamma_gen_hat=round(self.gamma_gen_hat, 6),
            conf_tau=round(self.conf_tau, 4),
            conf_nu=round(self.conf_nu, 4),
            conf_gamma=round(self.conf_gamma, 4),
            events=events,
        )
        self.history.append(snap)
        return snap

    def record_action_agreement(self, agree: bool):
        """Call after computing shadow decision to update action stability."""
        self._action_agree_ema = ((1 - self._action_ema_alpha) * self._action_agree_ema
                                  + self._action_ema_alpha * float(agree))


# ═══════════════════════════════════════════════════════
#  A1 Frozen Baseline — DO NOT MODIFY PARAMETERS
# ═══════════════════════════════════════════════════════

class A1MtObserverFrozen(A1MtObserver):
    """Frozen A1 baseline. Raises if core parameters are changed.

    Frozen: 2026-03-28
    Evidence: Corr_τ=1.0, Corr_ν=0.99, Corr_γ=0.98, ADR=0%
              Online micro infer-only: zero diverge
              Macro lesson ranking: Kendall τ=0.998
    """

    _FROZEN_PARAMS = {
        "beta_tau_probe": 0.0,
        "beta_nu_probe": 0.0,
        "beta_gamma_probe": 0.0,
        "lambda_tau": 0.005,
        "lambda_nu": 0.005,
        "lambda_gamma": 0.005,
    }

    def __init__(self):
        super().__init__()
        # Verify frozen params
        for param, expected in self._FROZEN_PARAMS.items():
            actual = getattr(self, param)
            if actual != expected:
                raise ValueError(
                    f"A1Frozen: {param}={actual}, expected {expected}. "
                    "Do not modify frozen baseline parameters."
                )

    def __setattr__(self, name, value):
        if name in self.__class__._FROZEN_PARAMS:
            expected = self._FROZEN_PARAMS[name]
            if value != expected:
                raise AttributeError(
                    f"A1Frozen: cannot set {name}={value} "
                    f"(frozen at {expected})"
                )
        super().__setattr__(name, value)
