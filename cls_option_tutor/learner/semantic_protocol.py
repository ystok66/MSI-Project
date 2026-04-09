"""
semantic_protocol.py — Protocol for semantic posterior scoring.

Defines the interface that both DeterministicSemanticPosterior (fixed baseline)
and CLSSemanticPosterior (CLS learner) must implement.

Richer than the old SemanticScorerProtocol: adds study(), posterior_probs(),
and semantic_entropy() for proper probabilistic reasoning.
"""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..interfaces import Example, Option


class SemanticPosteriorProtocol:
    """Protocol for semantic posterior scoring over option menus.

    Lifecycle:
        scorer = create_scorer(grammar, support, ...)
        scorer.study(support)                        # learn from support
        probs = scorer.posterior_probs(target, menu)  # P_L(j | Y*, support)
        score = scorer.score_option(target, text)     # log P_L(correct | j)
        pred  = scorer.predict_output(text)           # Y_hat_j
        u     = scorer.uncertainty(target, text)      # [0, 1]

    Two implementations:
        - DeterministicSemanticPosterior: oracle render (baseline)
        - CLSSemanticPosterior: CLS three-layer system (canonical)
    """

    def study(self, support: List['Example'],
              n_em: int = 2, use_hpc: bool = True) -> None:
        """Learn from support examples.

        For deterministic scorer: no-op (grammar is the oracle).
        For CLS scorer: reset_episode() + study(support) with EM.
        """
        pass  # default no-op for backward compatibility

    def score_option(self, target_output: List[str],
                     option_text: List[str],
                     memory_payload: object = None,
                     attention_weights: Optional[np.ndarray] = None,
                     ) -> float:
        """Score how well option_text explains target_output.

        Args:
            target_output: Y*
            option_text: option tokens
            memory_payload: legacy unused
            attention_weights: (L,) weights over target cells.
                When provided, mismatch is weighted: M = Σ w_ℓ·1[ŷ_ℓ ≠ y*_ℓ]
                When None, uniform weights (1/L).

        Returns: S_sem(j) = -M(j)/τ (higher = better, 0 = perfect).
        """
        raise NotImplementedError

    def predict_output(self, option_text: List[str],
                       memory_payload: object = None) -> List[str]:
        """Predict rendered output for option_text."""
        raise NotImplementedError

    def uncertainty(self, target_output: List[str],
                    option_text: List[str],
                    memory_payload: object = None) -> float:
        """Semantic uncertainty (0 = certain, 1 = max uncertainty)."""
        raise NotImplementedError

    def posterior_probs(self, target_output: List[str],
                        menu: List['Option'],
                        attention_weights: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
        """Normalized posterior P_L(j | Y*, support) over menu options.

        Args:
            target_output: Y* target
            menu: list of Options
            attention_weights: (L,) weights over target cells (optional)

        Returns:
            (K,) array summing to 1.0
        """
        # Default: softmax over score_option values (with attention weights)
        K = len(menu)
        if K == 0:
            return np.array([])
        scores = np.array([
            self.score_option(target_output, opt.text,
                              attention_weights=attention_weights)
            for opt in menu
        ])
        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        return probs / (probs.sum() + 1e-10)

    def semantic_entropy(self, target_output: List[str],
                          menu: List['Option']) -> float:
        """H(P_L) = -Σ P_L(j) log P_L(j). Higher = more uncertain."""
        probs = self.posterior_probs(target_output, menu)
        p_pos = probs[probs > 0]
        if len(p_pos) == 0:
            return 0.0
        return -float(np.sum(p_pos * np.log(p_pos)))

    def clear_cache(self) -> None:
        """Clear any render/prediction caches (e.g., after refresh)."""
        pass
