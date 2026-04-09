"""
semantic_scorer.py — Deterministic mismatch scoring for option evaluation.

Implements §7.1 deterministic fallback (fixed_semantics_baseline):
    M_t(j) = Σ_ℓ w_t,ℓ · 1[ŷ_j,ℓ ≠ y*_ℓ]
    S_sem,t(j) = -M_t(j) / τ_sem

Uses the grammar renderer to predict each option's output,
then compares cell-by-cell against the target.

This is the ORACLE scorer — grammar is given, render is perfect.
Kept as fixed_semantics_baseline for all comparisons.
"""
from __future__ import annotations
from typing import List, Optional, Dict, TYPE_CHECKING
import numpy as np

from ..interfaces import Option, SemanticScorerProtocol
from ..grammar.task_adapter import Grammar, TaskAdapter
from .semantic_protocol import SemanticPosteriorProtocol

if TYPE_CHECKING:
    from ..interfaces import Example


class DeterministicSemanticScorer(SemanticPosteriorProtocol, SemanticScorerProtocol):
    """Deterministic mismatch-based semantic scorer (fixed_semantics_baseline).

    For each option, renders F_G(ν) and counts cell mismatches
    against the target output Y*. Grammar is given — this is the oracle.

    Implements SemanticPosteriorProtocol for drop-in compatibility
    with the CLS learner pathway.
    """

    def __init__(self, grammar: Grammar, tau_sem: float = 1.0):
        self.grammar = grammar
        self.tau_sem = tau_sem
        self._render_cache: Dict[tuple, Optional[List[str]]] = {}

    def _render(self, text: List[str]) -> Optional[List[str]]:
        """Cached render."""
        key = tuple(text)
        if key not in self._render_cache:
            self._render_cache[key] = TaskAdapter.render(text, self.grammar)
        return self._render_cache[key]

    def score_option(self, target_output: List[str],
                     option_text: List[str],
                     memory_payload: object = None,
                     attention_weights: Optional[np.ndarray] = None,
                     ) -> float:
        """S_sem = -M(j) / τ_sem where M = weighted mismatch.

        When attention_weights provided, M = Σ w_ℓ · 1[ŷ_ℓ ≠ y*_ℓ]
        Otherwise M = count(mismatches) (uniform weights).
        """
        predicted = self._render(option_text)
        if predicted is None:
            return -len(target_output) / self.tau_sem

        L = len(target_output)
        L_pred = len(predicted)

        if attention_weights is not None and len(attention_weights) == L:
            # Weighted mismatch (HIGHLIGHT pathway)
            w = attention_weights
            if L_pred != L:
                min_len = min(L, L_pred)
                mismatch = sum(w[i] for i in range(min_len)
                               if predicted[i] != target_output[i])
                mismatch += sum(w[i] for i in range(min_len, L))
            else:
                mismatch = sum(w[i] for i in range(L)
                               if predicted[i] != target_output[i])
        else:
            # Uniform mismatch (no HIGHLIGHT)
            if L_pred != L:
                min_len = min(L, L_pred)
                matches = sum(1 for i in range(min_len)
                              if predicted[i] == target_output[i])
                mismatch = max(L, L_pred) - matches
            else:
                mismatch = sum(1 for i in range(L)
                               if predicted[i] != target_output[i])

        return -mismatch / self.tau_sem

    def predict_output(self, option_text: List[str],
                       memory_payload: object = None) -> List[str]:
        """Predict rendered output for option_text."""
        result = self._render(option_text)
        return result if result is not None else []

    def uncertainty(self, target_output: List[str],
                    option_text: List[str],
                    memory_payload: object = None) -> float:
        """Semantic uncertainty: 1 if can't render, 0 if perfect match."""
        predicted = self._render(option_text)
        if predicted is None:
            return 1.0
        L = max(len(target_output), 1)
        if len(predicted) != len(target_output):
            return 1.0
        mismatches = sum(1 for i in range(len(target_output))
                         if predicted[i] != target_output[i])
        return mismatches / L

    def score_menu(self, target_output: List[str],
                   menu: List[Option],
                   weights: Optional[np.ndarray] = None,
                   ) -> np.ndarray:
        """Score all options in a menu.

        Args:
            target_output: Y* target
            menu: list of Options
            weights: (L,) attention weights over target cells (optional)

        Returns:
            (K,) array of semantic scores
        """
        L = len(target_output)
        if weights is None:
            weights = np.ones(L) / L

        scores = np.zeros(len(menu))
        for j, opt in enumerate(menu):
            predicted = self._render(opt.text)
            if predicted is None:
                scores[j] = -L / self.tau_sem
                continue

            if len(predicted) != L:
                min_len = min(L, len(predicted))
                mismatch = sum(weights[i] for i in range(min_len)
                               if predicted[i] != target_output[i])
                # Extra or missing cells penalized
                mismatch += sum(weights[i] for i in range(min_len, L))
            else:
                mismatch = sum(weights[i] for i in range(L)
                               if predicted[i] != target_output[i])

            scores[j] = -mismatch / self.tau_sem

        return scores

    def posterior_probs(self, target_output: List[str],
                        menu: List[Option],
                        attention_weights: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
        """P_L(j | Y*, G) via softmax over mismatch scores."""
        scores = self.score_menu(target_output, menu, attention_weights)
        if len(scores) == 0:
            return np.array([])
        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        return probs / (probs.sum() + 1e-10)

    def semantic_entropy(self, target_output: List[str],
                         menu: List[Option]) -> float:
        """H_t^sem: entropy of softmax over semantic scores."""
        probs = self.posterior_probs(target_output, menu)
        p_pos = probs[probs > 0]
        if len(p_pos) == 0:
            return 0.0
        return -float(np.sum(p_pos * np.log(p_pos)))

    def study(self, support: List['Example'],
              n_em: int = 2, use_hpc: bool = True) -> None:
        """No-op for deterministic scorer (grammar is the oracle)."""
        pass

    def clear_cache(self):
        """Clear render cache (e.g., after refresh)."""
        self._render_cache.clear()
