"""
bg.py — Basal Ganglia Selector: candidate gating + resource allocation.

Responsible for:
  1. Beam width modulation based on CA1 novelty signal
  2. RSA utility soft rerank of candidates (adjustment D: no hard filter)
  3. Exploration mode control
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from cls_learner.interfaces import MemBias
from cls_learner.config import CLSConfig


class BGSelector:
    """
    Basal Ganglia candidate selector.

    Modulates beam search resources and applies RSA utility reranking.

    Per adjustment D: RSA utility is applied as a SOFT rerank term,
    never as a hard filter, to avoid pruning correct solutions.
    """

    def __init__(self, cfg: CLSConfig):
        self.cfg = cfg

    def adjust_beam(self, base_beam: int,
                    mem_bias: Optional[MemBias] = None) -> int:
        """
        Adjust beam width based on HPC novelty signal.

        In explore mode (novel input): expand beam for broader search.
        In retrieve mode (familiar): keep default beam for efficiency.
        """
        if mem_bias is None:
            return base_beam

        if mem_bias.mode == 'explore':
            expanded = int(base_beam * (1.0 + self.cfg.bg_explore_factor))
            return min(expanded, self.cfg.bg_max_beam_expand)
        elif mem_bias.mode == 'mixed':
            # Modest expansion
            expanded = int(base_beam * (1.0 + self.cfg.bg_explore_factor * 0.5))
            return min(expanded, self.cfg.bg_max_beam_expand)
        else:
            return base_beam

    def rerank_candidates(self, candidates: List[Tuple[float, object]],
                          mem_bias: Optional[MemBias] = None,
                          trace_lengths: Optional[List[int]] = None
                          ) -> List[Tuple[float, object]]:
        """
        Soft rerank candidates with RSA utility bonus.

        score_total = score_model + rsa_bonus - cost

        rsa_bonus is applied as a rerank term (not a hard filter).
        Currently conservative: only applies when explicitly enabled.
        """
        alpha = self.cfg.bg_rsa_rerank_alpha
        if alpha <= 0 or not candidates:
            return candidates

        cost_per_op = self.cfg.bg_rsa_cost_per_op

        reranked = []
        for i, (score, obj) in enumerate(candidates):
            bonus = 0.0

            # Cost penalty: longer traces are penalized
            if trace_lengths and i < len(trace_lengths):
                bonus -= cost_per_op * trace_lengths[i]

            reranked.append((score + alpha * bonus, obj))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    def compute_utility(self, score_model: float,
                        trace_length: int,
                        mem_bias: Optional[MemBias] = None) -> float:
        """
        Compute RSA-inspired utility for a single candidate.

        U = score_model - α * cost_per_op * trace_length + explore_bonus

        This is the log-domain version per adjustment D/5.
        """
        alpha = self.cfg.bg_rsa_rerank_alpha
        cost = self.cfg.bg_rsa_cost_per_op * trace_length

        utility = score_model - alpha * cost

        # Exploration bonus from HPC novelty
        if mem_bias and mem_bias.mode == 'explore':
            # In explore mode, add a small diversity bonus
            utility += 0.1 * mem_bias.lam_mem

        return utility
