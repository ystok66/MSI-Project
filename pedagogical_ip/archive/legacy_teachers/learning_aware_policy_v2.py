"""C2 — Learning-Aware Tutor Policy v2: Decision + Urgency.

Replaces entropy-based IG with decision-aware information gain (DIG),
and adds urgency/missed-window terms.

Q_teach(WARN) = λ_S·ΔS + λ_D·DIG + λ_A·ΔM + λ_U·U - λ_C·C - λ_R·R
Q_teach(WAIT) = -λ_M·P_miss

Where:
  ΔS  = predicted success improvement (margin proxy)
  DIG = decision Bayes risk reduction
  ΔM  = margin gain (autonomy proxy)
  U   = urgency: σ((d_reveal - d_commit) / τ_u)
  P_miss = missed-window penalty
  C   = intervention cost
  R   = redundancy penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..metrics.decision_info import (
    decision_bayes_risk,
    decision_info_gain,
    compute_branch_posteriors,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


@dataclass
class TutorV2Config:
    """Tunable weights for Q_teach."""
    lambda_s: float = 1.0    # success gain
    lambda_d: float = 2.0    # decision info gain (replaces entropy IG)
    lambda_a: float = 0.5    # autonomy proxy (margin gain)
    lambda_u: float = 1.5    # urgency
    lambda_m: float = 1.0    # missed-window penalty for WAIT
    lambda_c: float = 0.05   # intervention cost (low — we saw 0.1 was too high)
    lambda_r: float = 0.3    # redundancy penalty
    tau_u: float = 2.0       # urgency temperature
    confidence_threshold: float = 0.7  # above this, warning is redundant


class LearningAwarePolicyV2:
    """Decision-aware, urgency-sensitive WAIT vs WARN.

    Uses decision Bayes risk instead of Shannon entropy,
    and adds urgency/missed-window terms.
    """

    def __init__(self, config: Optional[TutorV2Config] = None):
        self.cfg = config or TutorV2Config()
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
    ) -> tuple[str, dict]:
        """Decide WAIT or WARN.

        Returns (action, diagnostics_dict).
        """
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        # ── Pre-warning state: agent's visible-only view ──
        from ..envs.observation_mask import make_observation_mask
        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs_radius)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs_radius)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        s_a_pre = summarize_branch(vis_a, fb_mod, fv, lp)
        s_b_pre = summarize_branch(vis_b, fb_mod, fv, lp)
        p_safe_a_pre, p_safe_b_pre = compute_branch_posteriors(
            s_a_pre, s_b_pre, scorer, build_scorer_input, lib)
        br_pre = decision_bayes_risk(p_safe_a_pre, p_safe_b_pre)

        # ── Post-warning state: full-branch view ──
        s_a_post = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
        p_safe_a_post, p_safe_b_post = compute_branch_posteriors(
            s_a_post, s_b_post, scorer, build_scorer_input, lib)
        br_post = decision_bayes_risk(p_safe_a_post, p_safe_b_post)

        # ── Components ──
        dig = max(br_pre - br_post, 0)
        margin_pre = abs(s_a_pre[0] - s_b_pre[0])
        margin_post = abs(s_a_post[0] - s_b_post[0])
        delta_m = max(margin_post - margin_pre, 0)
        delta_s = delta_m  # success proxy ≈ margin improvement

        # Urgency: how close is commitment vs reveal
        d_commit = obs_radius + 1  # steps to pass visible zone
        d_reveal = sc.reveal_depth  # depth before strong cues appear
        urgency = _sigmoid((d_reveal - d_commit) / cfg.tau_u)

        # Missed window: if WAIT, will strong cues be visible before commit?
        p_miss = 1.0 if d_commit < d_reveal else 0.0

        # Redundancy: if already confident, warning is redundant
        confidence = max(p_safe_a_pre, p_safe_b_pre)
        redundancy = max(confidence - cfg.confidence_threshold, 0.0)

        # ── Q values ──
        Q_warn = (cfg.lambda_s * delta_s
                  + cfg.lambda_d * dig
                  + cfg.lambda_a * delta_m
                  + cfg.lambda_u * urgency
                  - cfg.lambda_c * 1.0
                  - cfg.lambda_r * redundancy)

        Q_wait = -cfg.lambda_m * p_miss

        action = "WARN" if Q_warn > Q_wait else "WAIT"

        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4),
            "Q_wait": round(Q_wait, 4),
            "BR_pre": round(br_pre, 4),
            "BR_post": round(br_post, 4),
            "DIG": round(dig, 4),
            "margin_pre": round(margin_pre, 4),
            "margin_post": round(margin_post, 4),
            "delta_m": round(delta_m, 4),
            "urgency": round(urgency, 4),
            "p_miss": round(p_miss, 4),
            "confidence": round(confidence, 4),
            "redundancy": round(redundancy, 4),
        }

        return action, diag

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
