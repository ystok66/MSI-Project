"""
target_predictor.py — Target prediction manager with caching.

Manages Y* predictions per query, handles re-prediction after
feedback updates.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

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
