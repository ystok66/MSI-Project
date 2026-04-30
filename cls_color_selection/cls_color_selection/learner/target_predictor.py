"""
target_predictor.py — Target prediction manager with caching.

Manages Y* predictions per query, handles re-prediction after
feedback updates.

Supports hint-conditioned inference (Step 1):
  - hard mode: filter beam to traces matching hint positions
  - soft mode: reweight beam by hint compatibility
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cls_wrapper import CLSSequencePredictor


class TargetPredictor:
    """Manages CLS target predictions and beam posteriors.

    Caches predictions per query to avoid redundant beam searches.
    Invalidates cache when feedback update modifies the cortex.
    """

    def __init__(self, cls_predictor: CLSSequencePredictor):
        self.cls = cls_predictor
        self._target_cache: Dict[tuple, List[str]] = {}
        self._beam_cache: Dict[tuple, list] = {}

    def predict_target(self, words: List[str]) -> List[str]:
        """Get or compute Y* for the given words."""
        key = tuple(words)
        if key not in self._target_cache:
            self._target_cache[key] = self.cls.predict_target(words)
        return self._target_cache[key]

    def beam_posterior(self, words: List[str]) -> list:
        """Get or compute beam posterior for the given words."""
        key = tuple(words)
        if key not in self._beam_cache:
            self._beam_cache[key] = self.cls.beam_posterior(words)
        return self._beam_cache[key]

    # ── Hint-conditioned inference (Step 1) ──

    def beam_posterior_with_hint(
        self,
        words: List[str],
        hint_positions: List[Tuple[int, str]],
        *,
        beta_hint: float = 2.0,
        mode: str = "hard",
    ) -> list:
        """Compute beam posterior conditioned on hint payload.

        Hard mode:
          q_hint(π | x, h) ∝ exp(score(π)) · 1[Y(π) matches all hinted positions]

        Soft mode:
          Compat(π; h) = Σ_{(i,c_i)∈h} 1[Y_i(π)=c_i]
          q_hint(π | x, h) ∝ exp(score(π) + β_hint · Compat(π; h))

        Args:
            words: query input words
            hint_positions: list of (position, color) from tutor hint
            beta_hint: soft-mode reweighting strength
            mode: "hard" or "soft"

        Returns:
            Reweighted beam: list of (new_score, trace, Y_k)
        """
        beam = self.beam_posterior(words)
        if not beam or not hint_positions:
            return beam

        reweighted = []
        for score, trace, Y_k in beam:
            compat = _compute_hint_compatibility(Y_k, hint_positions)

            if mode == "hard":
                # Keep only traces that match ALL hint positions
                if compat == len(hint_positions):
                    reweighted.append((score, trace, Y_k))
            else:
                # Soft reweight
                new_score = score + beta_hint * compat
                reweighted.append((new_score, trace, Y_k))

        if not reweighted:
            # Fallback: if hard mode filters out everything, return original
            return beam

        # Renormalize (for soft mode, scores need renormalization)
        if mode == "soft":
            scores = np.array([s for s, _, _ in reweighted])
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            reweighted = [(float(weights[i]), t, y)
                          for i, (_, t, y) in enumerate(reweighted)]

        return reweighted

    def predict_target_with_hint(
        self,
        words: List[str],
        hint_positions: List[Tuple[int, str]],
        *,
        beta_hint: float = 2.0,
        mode: str = "hard",
    ) -> List[str]:
        """Predict target conditioned on hint payload.

        Returns Y* from the best trace in the hint-conditioned beam.

        Args:
            words: query input words
            hint_positions: hint payload
            beta_hint: soft-mode strength
            mode: "hard" or "soft"

        Returns:
            Predicted target output conditioned on hint evidence
        """
        beam = self.beam_posterior_with_hint(
            words, hint_positions,
            beta_hint=beta_hint, mode=mode)
        if not beam:
            return self.predict_target(words)

        # Return Y from highest-scoring trace
        best = max(beam, key=lambda x: x[0])
        return list(best[2])

    def invalidate_cache(self, words: Optional[List[str]] = None):
        """Invalidate cached predictions after feedback update.

        Args:
            words: if provided, invalidate only this query.
                   If None, invalidate all.
        """
        if words is not None:
            key = tuple(words)
            self._target_cache.pop(key, None)
            self._beam_cache.pop(key, None)
        else:
            self._target_cache.clear()
            self._beam_cache.clear()

    def invalidate_all(self):
        """Clear all caches."""
        self._target_cache.clear()
        self._beam_cache.clear()


def _compute_hint_compatibility(
    Y_k: List[str],
    hint_positions: List[Tuple[int, str]],
) -> int:
    """Count how many hint positions match trace output Y_k."""
    compat = 0
    for pos, color in hint_positions:
        if pos < len(Y_k) and Y_k[pos] == color:
            compat += 1
    return compat


def merge_completion_after_target_flip(
    completion: List[Optional[str]],
    assist_mask: List[bool],
    new_target: List[str],
) -> List[Optional[str]]:
    """Merge completion after target prediction changes.

    Rules:
      - If new target is longer/shorter, resize completion
      - Preserve tutor-hinted positions if still aligned
      - Clear learner-filled positions (they were targeting old target)

    Args:
        completion: current completion array
        assist_mask: True at tutor-hinted positions
        new_target: updated target from hint-conditioned prediction

    Returns:
        Updated completion array
    """
    new_len = len(new_target)
    new_completion = [None] * new_len

    for i in range(min(len(completion), new_len)):
        if completion[i] is not None:
            if i < len(assist_mask) and assist_mask[i]:
                # Tutor-hinted: keep if still consistent
                if completion[i] == new_target[i]:
                    new_completion[i] = completion[i]
                # else: hinted position now inconsistent — drop it
            else:
                # Learner-filled: re-check against new target
                # Keep only if it still matches
                if completion[i] == new_target[i]:
                    new_completion[i] = completion[i]

    return new_completion
