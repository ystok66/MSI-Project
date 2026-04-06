"""J3 — Preference-Aware Tutor v2: observation value + posterior-conditioned.

Key upgrades from pref_v1:
  1. Uses stochastic agent policy model for accurate likelihood
  2. Observation value from posterior predictive variance
  3. Posterior-sensitive: as q(θ) becomes certain, WAIT value decreases

v2.1 calibration (confidence-gated autonomy):
  c_t  = 1 - H(q_t(θ)) / log|Θ|   — posterior confidence
  A_t  = λ_A · c_t · p_self · G_self — autonomy bonus in Q_wait
  M'_t = λ_M · (1-κ_M·c_t) · (1-p_self)  — gated missed-window
  T'_t = λ_T · tempt · max(1-p_self,ε) · (1-c_t+ρ) — gated temptation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, compute_choice_probs,
)
from ..agents.preference_posterior_v2 import PreferencePosteriorV2
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class PrefV2Config:
    """Tutor v2 configuration."""
    # Inherited from v4
    lambda_s: float = 1.0
    lambda_i: float = 2.0
    lambda_m: float = 1.2    # missed-window (reduced from 1.5)
    lambda_c: float = 0.05
    lambda_r: float = 0.3
    lambda_v: float = 2.0
    lambda_f: float = 1.5
    tau_v: float = 1.0
    tau_m: float = 1.0
    confidence_threshold: float = 0.7
    # Preference-specific v2
    lambda_p: float = 1.5     # preference info gain from warning
    lambda_t: float = 1.5     # temptation risk premium (reduced from 2.0)
    lambda_o: float = 2.0     # observation value (waiting to infer θ)
    lambda_a: float = 2.5     # autonomy value (increased from 0.3)
    # v2.1 confidence gating
    kappa_m: float = 0.6      # missed-window gate strength
    eps_nec: float = 0.15     # necessity floor for tempt gate
    rho_unc: float = 0.05     # uncertainty floor for tempt gate


class PreferenceAwarePolicyV2:
    """Tutor that uses actual agent policy model for preference inference."""

    def __init__(
        self,
        config: Optional[PrefV2Config] = None,
        agent_params: Optional[AgentPolicyParams] = None,
    ):
        self.cfg = config or PrefV2Config()
        self.agent_params = agent_params or AgentPolicyParams()
        self.pref_posterior = PreferencePosteriorV2()
        self.warn_count = 0
        self.wait_count = 0

    def _build_branch_attrs(self, sc, summary, is_risky: bool) -> BranchAttributes:
        """Build BranchAttributes from branch summary + scenario config."""
        tempt = getattr(sc, 'tempt_score_b' if is_risky else 'tempt_score_a', 0.0)
        return BranchAttributes(
            safety_score=float(summary[0]),
            temptation_score=tempt,
            texture_novelty=float(abs(summary[0] - 0.5)) if len(summary) > 0 else 0.0,
            shortcut_bonus=0.0,
            risk_penalty=float(1.0 - summary[0]) if is_risky else 0.0,
        )

    def decide(
        self,
        sc,
        fb_mod: np.ndarray,
        lp,
        lib: BranchConceptLibrary,
        scorer: BranchScorerProbe,
        obs_radius: int = 2,
    ) -> tuple[str, dict]:
        """Decide WAIT or WARN with preference + observation value."""
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)
        p_self = estimate_self_discovery_prob(d_commit, d_reveal, tau_v=cfg.tau_v)
        p_fail_wait = estimate_failure_if_wait(d_commit, d_reveal, tau_f=cfg.tau_v)

        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs_radius)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs_radius)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        s_a_pre = summarize_branch(vis_a, fb_mod, fv, lp)
        s_b_pre = summarize_branch(vis_b, fb_mod, fv, lp)
        margin_pre = abs(s_a_pre[0] - s_b_pre[0])

        s_a_post = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
        margin_post = abs(s_a_post[0] - s_b_post[0])

        u_pre = _sigmoid(cfg.tau_m * margin_pre)
        u_post = _sigmoid(cfg.tau_m * margin_post)
        dvoi = max(u_post - u_pre, 0)
        delta_s = max(margin_post - margin_pre, 0)

        # Redundancy
        inp_a = build_scorer_input(s_a_pre, lib)
        inp_b = build_scorer_input(s_b_pre, lib)
        sc_a = scorer.score(inp_a)
        sc_b = scorer.score(inp_b)
        confidence = abs(sc_a - sc_b)
        redundancy = max(confidence - cfg.confidence_threshold, 0)

        g_self = max(margin_post - margin_pre, 0)

        # ── Preference-aware terms (v2) ──
        # Build branch attributes for preference model
        is_a_risky = (sc.oracle_safe_branch_id != 0)
        ba_safe = self._build_branch_attrs(sc, s_a_pre, is_risky=is_a_risky)
        ba_risky = self._build_branch_attrs(sc, s_b_pre, is_risky=not is_a_risky)
        branches = [ba_safe, ba_risky]

        # Observation value: how informative would observing agent's choice be?
        obs_value = self.pref_posterior.posterior_predictive_variance(
            branches, self.agent_params)

        # ── v2.1: posterior confidence ──
        pref_entropy = self.pref_posterior.entropy
        max_ent = max(self.pref_posterior.max_entropy, 1e-6)
        pref_uncertainty = pref_entropy / max_ent
        c_t = 1.0 - pref_uncertainty  # posterior confidence ∈ [0,1]

        # ΔPrefInfo: warning helps disambiguate preference
        delta_pref = self.pref_posterior.expected_info_gain_from_observation(
            branches, self.agent_params) if pref_uncertainty > 0.3 else 0.0

        # ── v2.1: gated temptation ──
        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        g_nec = max(1.0 - p_self, cfg.eps_nec)       # necessity gate
        g_unc = (1.0 - c_t) + cfg.rho_unc             # uncertainty gate
        tempt_risk_gated = cfg.lambda_t * tempt_str * g_nec * g_unc
        tempt_risk_raw = cfg.lambda_t * tempt_str * pref_uncertainty  # for diagnostics

        # ── v2.1: gated missed-window ──
        missed_raw = cfg.lambda_m * (1.0 - p_self)
        missed_gated = cfg.lambda_m * (1.0 - cfg.kappa_m * c_t) * (1.0 - p_self)

        # ── v2.1: autonomy bonus ──
        autonomy_bonus = cfg.lambda_a * c_t * p_self * g_self

        # ── Q values ──
        Q_warn = (cfg.lambda_s * delta_s
                  + cfg.lambda_i * dvoi
                  + missed_gated
                  + cfg.lambda_p * delta_pref
                  + tempt_risk_gated
                  - cfg.lambda_c * 1.0
                  - cfg.lambda_r * redundancy)

        Q_wait = (cfg.lambda_v * p_self * g_self
                  + cfg.lambda_o * obs_value
                  + autonomy_bonus
                  - cfg.lambda_f * p_fail_wait)

        action = "WARN" if Q_warn > Q_wait else "WAIT"

        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "p_self": round(p_self, 4), "dvoi": round(dvoi, 4),
            "delta_s": round(delta_s, 4),
            "margin_pre": round(margin_pre, 4), "margin_post": round(margin_post, 4),
            "d_commit": d_commit, "d_reveal": d_reveal,
            "obs_value": round(obs_value, 4),
            "delta_pref": round(delta_pref, 4),
            "pref_entropy": round(pref_entropy, 4),
            "pref_uncertainty": round(pref_uncertainty, 4),
            "confidence_c_t": round(c_t, 4),
            "autonomy_bonus": round(autonomy_bonus, 4),
            "tempt_risk_raw": round(tempt_risk_raw, 4),
            "tempt_risk_gated": round(tempt_risk_gated, 4),
            "missed_window_raw": round(missed_raw, 4),
            "missed_window_gated": round(missed_gated, 4),
            "pref_predicted": self.pref_posterior.predicted_type,
            "pref_confidence": round(self.pref_posterior.predicted_prob, 4),
        }
        return action, diag

    def observe_agent_choice(
        self, chosen_idx: int, branches: list[BranchAttributes],
    ):
        """Update posterior from observed stochastic agent choice."""
        self.pref_posterior.update_from_choice(
            chosen_idx, branches, self.agent_params)

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)
