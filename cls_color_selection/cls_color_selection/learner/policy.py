"""
policy.py — Learner color-selection policy (Phase 1 simplified).

Three components:
  1. Select set scoring: fill utility - risk penalty - waste penalty
  2. Confirm rule: rule-based (fill ratio threshold)
  3. Greedy selection: add one ball at a time maximizing marginal utility
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..config import LearnerConfig
from ..interfaces import CandidateBall
from .risk_belief import DangerTypeBelief
from ..environment.state import QueryState


class ColorSelectionPolicy:
    """Learner policy for selecting balls and deciding when to confirm.

    Phase 1: greedy set selection + rule-based confirm.
    """

    def __init__(self, cfg: LearnerConfig):
        self.cfg = cfg

    def select_set(
        self,
        state: QueryState,
        risk_belief: DangerTypeBelief,
        rng: np.random.Generator,
    ) -> List[int]:
        """Select a subset of balls from the candidate pool.

        Uses greedy marginal utility: add one ball at a time.

        Args:
            state: current query state
            risk_belief: danger type belief model
            rng: random generator

        Returns:
            List of indices into state.candidate_pool
        """
        pool = state.candidate_pool
        needed = state.needed_colors()
        gaps = state.color_gaps()

        if not needed or not pool:
            return []

        # Compute per-ball marginal utilities
        n = len(pool)

        # Pre-compute danger posteriors for all balls
        if n > 0:
            X = np.stack([b.observed_vec for b in pool])
            posteriors = risk_belief.batch_posterior(X)
            p_danger = 1.0 - posteriors[:, 0]  # P(z_i ≠ 0 | x_i)
        else:
            p_danger = np.array([])

        # ── Risk gate: skip needed-but-too-risky balls if safer alternative exists ──
        gated_indices = set()
        if self.cfg.risk_gate_tau > 0 and n > 0:
            # For each needed color, find min p_danger among needed balls of that color
            color_min_risk = {}  # color -> min p_danger
            for i, ball in enumerate(pool):
                if ball.color in needed:
                    prev = color_min_risk.get(ball.color, 1.0)
                    color_min_risk[ball.color] = min(prev, p_danger[i])

            # Check if ANY needed color has a safe-enough ball
            has_safe_needed = any(v < self.cfg.risk_gate_tau for v in color_min_risk.values())

            if has_safe_needed:
                for i, ball in enumerate(pool):
                    if (ball.color in needed
                            and p_danger[i] > self.cfg.risk_gate_tau
                            and color_min_risk.get(ball.color, 0) < p_danger[i]):
                        gated_indices.add(i)

        # Greedy selection: pick balls that maximize marginal utility
        selected_indices = []
        selected_colors: Dict[str, int] = {}  # track how many of each color selected
        remaining_gaps = dict(gaps)

        for _ in range(min(n, sum(gaps.values()))):
            best_idx = -1
            best_util = -np.inf

            for i in range(n):
                if i in selected_indices or i in gated_indices:
                    continue

                ball = pool[i]
                c = ball.color

                # Fill utility: 1 if this color is still needed
                color_already = selected_colors.get(c, 0)
                gap_left = remaining_gaps.get(c, 0) - color_already
                g_fill = 1.0 if gap_left > 0 else 0.0

                # Waste penalty: picking unneeded color
                g_waste = 0.0 if c in needed else 1.0

                # Risk: P(danger for this ball)
                p_d = p_danger[i]

                # Marginal utility
                util = (self.cfg.alpha_fill * g_fill
                        - self.cfg.alpha_risk * p_d
                        - self.cfg.alpha_waste * g_waste)

                if util > best_util:
                    best_util = util
                    best_idx = i

            if best_idx < 0 or best_util < -2.0:
                # ── Step 2: raise stop bar after hint ──
                break

            selected_indices.append(best_idx)
            c = pool[best_idx].color
            selected_colors[c] = selected_colors.get(c, 0) + 1

        # Epsilon-exploration: with small prob, pick a random ball instead
        eps = self.cfg.epsilon_policy
        # ── Step 2: reduce exploration after hint ──
        hinted = getattr(state, 'hinted_this_query', False)
        if hinted and self.cfg.enable_hint_autonomy_shift:
            eps = eps * (1.0 - self.cfg.hint_exploration_drop)

        if rng.random() < eps and n > 0:
            random_idx = rng.integers(0, n)
            if random_idx not in selected_indices:
                selected_indices = [random_idx]

        return selected_indices

    def should_confirm(self, state: QueryState) -> bool:
        """Decide whether to confirm the current completion.

        Phase 1 rule-based:
          - Confirm if all positions filled
          - Confirm if fill ratio ≥ threshold
          - Step 2: lower threshold after hint (learner commits earlier)
        """
        if state.is_complete:
            return True

        threshold = self.cfg.confirm_fill_threshold
        # ── Step 2: lower confirm threshold after hint ──
        hinted = getattr(state, 'hinted_this_query', False)
        if hinted and self.cfg.enable_hint_autonomy_shift:
            threshold = max(0.0, threshold - self.cfg.hint_confirm_bonus)

        if state.fill_ratio >= threshold:
            return True
        return False

    def should_courage_trigger(self, state: QueryState) -> bool:
        """Check if courage should be triggered.

        Courage triggers when:
          - enable_courage is True
          - consecutive retries ≥ n_retry_courage
        """
        if not self.cfg.enable_courage:
            return False
        return state.consecutive_retries >= self.cfg.n_retry_courage
