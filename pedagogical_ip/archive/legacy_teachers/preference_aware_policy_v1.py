"""I2 — Preference-Aware Tutor Policy v1.

Extends v4 with:
  - ΔPrefInfo: expected entropy reduction of q(θ) from warning
  - Temptation risk: if agent is drawn to temptation branch, urgency increases
  - Autonomy tradeoff: waiting reveals preference type via behavior observation

Q_warn = (v4 terms) + λ_P·ΔPrefInfo - λ_A·autonomy_cost
Q_wait = (v4 terms) + λ_O·observation_value
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.preference_posterior import PreferencePosterior
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class PrefAwareTutorConfig:
    """Weights for preference-aware tutor."""
    # v4 inherited
    lambda_s: float = 1.0
    lambda_i: float = 2.0
    lambda_m: float = 1.5
    lambda_c: float = 0.05
    lambda_r: float = 0.3
    lambda_v: float = 2.0
    lambda_f: float = 1.5
    tau_v: float = 1.0
    tau_m: float = 1.0
    confidence_threshold: float = 0.7
    # Preference-specific
    lambda_p: float = 1.0     # preference info gain
    lambda_a: float = 0.5     # autonomy cost (waiting lets robot observe)
    lambda_o: float = 0.8     # observation value for preference inference
    lambda_t: float = 1.5     # temptation risk premium


class PreferenceAwarePolicyV1:
    """Tutor that reasons about agent's hidden preferences."""

    def __init__(self, config: Optional[PrefAwareTutorConfig] = None):
        self.cfg = config or PrefAwareTutorConfig()
        self.pref_posterior = PreferencePosterior()
        self.warn_count = 0
        self.wait_count = 0

    def decide(
        self,
        sc,
        fb_mod: np.ndarray,
        lp,
        lib: BranchConceptLibrary,
        scorer: BranchScorerProbe,
        obs_radius: int = 2,
        temptation_strength: float = 0.0,
    ) -> tuple[str, dict]:
        """Decide WAIT or WARN considering preference uncertainty."""
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)

        p_self = estimate_self_discovery_prob(d_commit, d_reveal, tau_v=cfg.tau_v)
        p_fail_wait = estimate_failure_if_wait(d_commit, d_reveal, tau_f=cfg.tau_v)

        # Branch summaries
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

        # ── Preference-specific terms ──
        # Branch attributes for preference inference
        branch_attrs_a = np.array([
            s_a_pre[0],   # safety proxy
            1 - s_a_pre[0],  # risk proxy
            getattr(sc, 'tempt_score_a', 0.0),
            0.0,  # shortcut not applicable
        ])
        branch_attrs_b = np.array([
            s_b_pre[0],
            1 - s_b_pre[0],
            getattr(sc, 'tempt_score_b', 0.0),
            0.0,
        ])

        # ΔPrefInfo: how much would warning help us disambiguate preference?
        delta_pref_info = self.pref_posterior.entropy_reduction_from_warn(
            branch_attrs_a, branch_attrs_b)

        # Preference uncertainty
        pref_entropy = self.pref_posterior.entropy
        pref_uncertainty = pref_entropy / max(self.pref_posterior.max_entropy, 1e-6)

        # Temptation risk: if temptation is high and we're uncertain about preference
        tempt_risk = temptation_strength * pref_uncertainty

        # Observation value: waiting lets robot observe agent's choice → infer θ
        obs_value = pref_uncertainty * 0.5  # high uncertainty → waiting is informative

        # ── Q values ──
        Q_warn = (cfg.lambda_s * delta_s
                  + cfg.lambda_i * dvoi
                  + cfg.lambda_m * (1.0 - p_self)
                  + cfg.lambda_p * delta_pref_info
                  + cfg.lambda_t * tempt_risk
                  - cfg.lambda_c * 1.0
                  - cfg.lambda_r * redundancy)

        Q_wait = (cfg.lambda_v * p_self * g_self
                  + cfg.lambda_o * obs_value
                  - cfg.lambda_f * p_fail_wait
                  - cfg.lambda_a * (1.0 - pref_uncertainty))

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
            "delta_pref_info": round(delta_pref_info, 4),
            "pref_entropy": round(pref_entropy, 4),
            "pref_uncertainty": round(pref_uncertainty, 4),
            "tempt_risk": round(tempt_risk, 4),
            "obs_value": round(obs_value, 4),
            "pref_predicted": self.pref_posterior.predicted_type,
            "pref_predicted_prob": round(self.pref_posterior.predicted_prob, 4),
        }
        return action, diag

    def observe_agent_choice(self, branch_attrs: np.ndarray, chose: bool):
        """Update preference posterior based on observed agent behavior."""
        self.pref_posterior.update(branch_attrs, chose)

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
