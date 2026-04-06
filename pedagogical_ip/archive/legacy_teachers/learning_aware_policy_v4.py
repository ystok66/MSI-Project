"""G3 — Tutor v4: Smooth Cross-Family Selective WAIT vs WARN.

Key upgrades from v3:
  1. Smooth p_self (sigmoid) instead of binary V_self
  2. DVOI (soft margin-based decision value) instead of entropy
  3. Explicit Q_warn vs Q_wait comparison with smooth transition

Q_warn = λ_S·ΔS + λ_I·DVOI + λ_M·(1-p_self) - λ_C·cost - λ_R·redundancy
Q_wait = λ_V·p_self·Ĝ_self - λ_F·P(fail|wait)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import (
    estimate_self_discovery_prob,
    estimate_failure_if_wait,
)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class TutorV4Config:
    """Tunable weights for v4 Q_teach."""
    # WARN components
    lambda_s: float = 1.0     # success gain
    lambda_i: float = 2.0     # DVOI (decision value of information)
    lambda_m: float = 1.5     # missed-window ~ (1 - p_self)
    lambda_c: float = 0.05    # intervention cost
    lambda_r: float = 0.3     # redundancy penalty
    # WAIT components
    lambda_v: float = 2.0     # self-discovery value
    lambda_f: float = 1.5     # failure-if-wait penalty
    # Smoothing
    tau_v: float = 1.0        # p_self temperature
    tau_m: float = 1.0        # DVOI margin sensitivity
    confidence_threshold: float = 0.7


class LearningAwarePolicyV4:
    """Cross-family selective WAIT vs WARN with smooth p_self + DVOI."""

    def __init__(self, config: Optional[TutorV4Config] = None):
        self.cfg = config or TutorV4Config()
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
        """Decide WAIT or WARN with smooth selectivity."""
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        # ── Timing parameters ──
        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)

        # Smooth self-discovery probability
        p_self = estimate_self_discovery_prob(
            d_commit, d_reveal, margin=0.0, tau_v=cfg.tau_v)

        # Failure-if-wait probability
        p_fail_wait = estimate_failure_if_wait(
            d_commit, d_reveal, tau_f=cfg.tau_v)

        # ── Pre-warning: visible-only summaries ──
        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs_radius)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs_radius)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        s_a_pre = summarize_branch(vis_a, fb_mod, fv, lp)
        s_b_pre = summarize_branch(vis_b, fb_mod, fv, lp)
        margin_pre = abs(s_a_pre[0] - s_b_pre[0])

        # ── Post-warning: full-branch summaries ──
        s_a_post = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
        margin_post = abs(s_a_post[0] - s_b_post[0])

        # ── DVOI: soft margin-based decision value ──
        u_pre = _sigmoid(cfg.tau_m * margin_pre)
        u_post = _sigmoid(cfg.tau_m * margin_post)
        dvoi = max(u_post - u_pre, 0)

        delta_s = max(margin_post - margin_pre, 0)

        # ── Redundancy ──
        inp_a = build_scorer_input(s_a_pre, lib)
        inp_b = build_scorer_input(s_b_pre, lib)
        sc_a = scorer.score(inp_a)
        sc_b = scorer.score(inp_b)
        confidence = abs(sc_a - sc_b)
        redundancy = max(confidence - cfg.confidence_threshold, 0)

        # ── Self-discovery gain estimate ──
        # If agent self-discovers, expected margin gain ≈ margin_post - margin_pre
        g_self = max(margin_post - margin_pre, 0)

        # ── Q values ──
        Q_warn = (cfg.lambda_s * delta_s
                  + cfg.lambda_i * dvoi
                  + cfg.lambda_m * (1.0 - p_self)
                  - cfg.lambda_c * 1.0
                  - cfg.lambda_r * redundancy)

        Q_wait = (cfg.lambda_v * p_self * g_self
                  - cfg.lambda_f * p_fail_wait)

        action = "WARN" if Q_warn > Q_wait else "WAIT"

        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "p_self": round(p_self, 4), "p_fail_wait": round(p_fail_wait, 4),
            "dvoi": round(dvoi, 4), "delta_s": round(delta_s, 4),
            "margin_pre": round(margin_pre, 4), "margin_post": round(margin_post, 4),
            "g_self": round(g_self, 4), "confidence": round(confidence, 4),
            "d_commit": d_commit, "d_reveal": d_reveal,
            "delta": d_commit - d_reveal,
        }
        return action, diag

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
