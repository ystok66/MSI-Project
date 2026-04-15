"""
cls_adapter.py — CLS semantic posterior adapter for the option tutor.

Implements SemanticPosteriorProtocol using CLSAgent's three-layer system.

S1 approach (canonical):
    predict(ν_j) → Ŷ_j → mismatch(Ŷ_j, Y*) → softmax → P_L(j | Y*, D_sup)

CLS is optional — falls back gracefully to DeterministicSemanticScorer
if BASIC/cls_learner is unavailable.
"""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from ..interfaces import SemanticScorerProtocol, Example, Option
from ..grammar.task_adapter import Grammar, TaskAdapter
from .semantic_protocol import SemanticPosteriorProtocol


class NegativeMemory:
    """Store wrong-pick programs; apply exact-match penalty to scoring.

    Used in 'negative_memory' reveal mode to avoid cortex pollution.
    Instead of feeding reveal data back into CLS EM, we store it in
    a separate memory and apply a scoring penalty.

    S_sem_new(j) = S_sem(j) - α_neg · NegMatch(j)
    where NegMatch(j) = 1 if program j matches a revealed wrong program.
    """

    def __init__(self, alpha_neg: float = 2.0):
        self._bad_programs: set = set()
        self.alpha_neg = alpha_neg

    def add(self, program: list) -> None:
        """Record a wrong-pick program."""
        self._bad_programs.add(tuple(program))

    def penalty(self, program: list) -> float:
        """Return negative penalty if program is known-wrong."""
        return -self.alpha_neg if tuple(program) in self._bad_programs else 0.0

    @property
    def size(self) -> int:
        return len(self._bad_programs)


class CLSSemanticPosterior(SemanticPosteriorProtocol, SemanticScorerProtocol):
    """CLS-backed semantic posterior scorer.

    Uses CLSAgent to learn grammar rules from support examples via EM,
    then scores options by predicting their output and comparing to target.

    This is the canonical CLS learner pathway (S1 approach):
        1. study(support) → CLSAgent.reset_episode() + CLSAgent.study()
        2. predict(ν_j) → Ŷ_j
        3. mismatch(Ŷ_j, Y*) → S_sem(j)
        4. softmax → P_L(j | Y*, D_sup)
    """

    def __init__(self, grammar: Grammar, tau_sem: float = 1.0,
                 lambda_neg: float = 0.0):
        self.grammar = grammar
        self.tau_sem = tau_sem
        self.lambda_neg = lambda_neg       # penalty scale for negative evidence
        self._agent = None
        self._studied = False
        self._predict_cache = {}
        # (program_tuple, target_output_tuple) → accumulated penalty weight
        # Keys are conditioned on target_output so the same wrong program on a
        # DIFFERENT target does not receive an unjustified penalty.
        self._neg_evidence: "dict[tuple, float]" = {}

    def study(self, support: List[Example],
              n_em: int = 2, use_hpc: bool = True) -> None:
        """Learn from support examples via CLS three-layer system.

        Args:
            support: examples to learn from (already subsetted to n_sup)
            n_em: EM iterations for cortex learning
            use_hpc: whether to use hippocampal memory
        """
        self._predict_cache.clear()
        self._support_history = list(support)  # track for incremental
        self._n_em = n_em
        self._use_hpc = use_hpc
        try:
            # Ensure cls_learner is importable (lives in BASIC/)
            import os, sys
            _basic_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'BASIC')
            _basic_dir = os.path.abspath(_basic_dir)
            if _basic_dir not in sys.path:
                sys.path.insert(0, _basic_dir)

            from cls_learner.agent import CLSAgent
            from cls_learner.config import CLSConfig
            from cls_learner.interfaces import Example as CLSExample

            cfg = CLSConfig(mode='ast', use_hpc=use_hpc, n_em=n_em)
            self._agent = CLSAgent(cfg)
            self._agent.reset_episode()

            cls_support = [
                CLSExample(words=ex.words, output=ex.output)
                for ex in support
            ]
            self._agent.study(cls_support, verbose=False)
            self._studied = True
        except (ImportError, Exception) as e:
            # CLS not available — will fall back to grammar render
            self._agent = None
            self._studied = False

    def incremental_study(self, new_examples: List[Example],
                          n_em_override: int = None) -> None:
        """Add new examples and re-run CLS study (teaching phase learning).

        Called during Phase 3 when learner observes new (program, output) pairs
        from wrong-pick reveals or correct-pick positive reinforcement.
        Accumulates into _support_history and re-trains.

        Args:
            new_examples: new supervision examples to absorb
            n_em_override: if set, overrides self._n_em for this update.
                Use n_em_override=1 for lightweight correct-pick updates to
                reduce overfitting risk vs full wrong-reveal restudy.
        """
        if not new_examples:
            return

        self._support_history.extend(new_examples)
        self._predict_cache.clear()

        # Re-study from scratch with expanded support
        _n_em = n_em_override if n_em_override is not None else self._n_em
        self.study(self._support_history,
                   n_em=_n_em, use_hpc=self._use_hpc)

    def freeze(self) -> None:
        """Freeze CLS predictions for evaluation phase.

        Pre-caches all pending predictions and prevents further updates.
        Negative evidence is also cleared so eval scores reflect only the
        CLS model state, not session-accumulated penalties.
        """
        # Clear cache so next predictions use latest CLS state
        self._predict_cache.clear()
        # Clear negative evidence: eval phase uses frozen CLS model only
        self._neg_evidence.clear()

    def predict_output(self, option_text: List[str],
                       memory_payload: object = None) -> List[str]:
        """Predict output using CLS agent if available, else grammar render."""
        key = tuple(option_text)
        if key in self._predict_cache:
            return self._predict_cache[key]

        result = None
        if self._agent is not None and self._studied:
            try:
                result = self._agent.predict(option_text, verbose=False)
            except Exception:
                pass

        if result is None:
            # Fallback to grammar render (oracle)
            rendered = TaskAdapter.render(option_text, self.grammar)
            result = rendered if rendered is not None else []

        self._predict_cache[key] = result
        return result

    def score_option(self, target_output: List[str],
                     option_text: List[str],
                     memory_payload: object = None,
                     attention_weights: Optional[np.ndarray] = None,
                     ) -> float:
        """S_sem = -weighted_mismatch / τ_sem (S1 approach).

        When attention_weights provided, M = Σ w_ℓ · 1[ŷ_ℓ ≠ y*_ℓ]
        Otherwise M = count(mismatches) (uniform weights).
        """
        predicted = self.predict_output(option_text, memory_payload)
        if not predicted:
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
            # Uniform mismatch
            if L_pred != L:
                min_len = min(L, L_pred)
                matches = sum(1 for i in range(min_len)
                              if predicted[i] == target_output[i])
                mismatch = max(L, L_pred) - matches
            else:
                mismatch = sum(1 for i in range(L)
                               if predicted[i] != target_output[i])

        base_score = -mismatch / self.tau_sem

        # Apply negative evidence penalty if any has been accumulated
        # for this (program, target_output) pair (nonreveal mode).
        if self.lambda_neg > 0.0:
            neg_pen = self.get_negative_penalty(option_text, target_output)
            if neg_pen > 0.0:
                base_score -= self.lambda_neg * neg_pen

        return base_score

    # ── Negative evidence (nonreveal mode) ─────────────────────────────────

    def add_negative_evidence(
        self,
        words: list,
        target_output: list,
        weight: float = 1.0,
    ) -> None:
        """Record that 'words' does NOT produce 'target_output'.

        Called in nonreveal mode when learner picks wrong: we know
        (words, target_output) is an incorrect pairing, but we do NOT
        know the actual rendered output of 'words'.

        Key design: conditioned on target_output so the same wrong program
        on a DIFFERENT target query does not receive an unfair penalty.

        Args:
            words: program text of the wrong option (list of str)
            target_output: Y* of the current query (list of str)
            weight: accumulation weight (default 1.0, use eta_negative from cfg)
        """
        key = (tuple(words), tuple(target_output))
        self._neg_evidence[key] = self._neg_evidence.get(key, 0.0) + weight

    def get_negative_penalty(
        self,
        words: list,
        target_output: list,
    ) -> float:
        """Return accumulated negative penalty for (words, target_output).

        Returns 0.0 if no evidence has been recorded for this pair.
        """
        key = (tuple(words), tuple(target_output))
        return self._neg_evidence.get(key, 0.0)

    def clear_negative_evidence(self) -> None:
        """Reset all negative evidence (e.g. at block start)."""
        self._neg_evidence.clear()

    def uncertainty(self, target_output: List[str],
                    option_text: List[str],
                    memory_payload: object = None) -> float:
        """Semantic uncertainty: mismatch fraction [0, 1]."""
        predicted = self.predict_output(option_text, memory_payload)
        if not predicted:
            return 1.0
        L = max(len(target_output), 1)
        if len(predicted) != len(target_output):
            return 1.0
        mismatches = sum(1 for i in range(len(target_output))
                         if predicted[i] != target_output[i])
        return mismatches / L

    def posterior_probs(self, target_output: List[str],
                        menu: List[Option],
                        attention_weights: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
        """P_L(j | Y*, D_sup) = softmax(-mismatch/τ) over menu options."""
        K = len(menu)
        if K == 0:
            return np.array([])

        scores = np.zeros(K)
        L = len(target_output)
        weights = attention_weights if attention_weights is not None else np.ones(L) / L

        for j, opt in enumerate(menu):
            predicted = self.predict_output(opt.text)
            if not predicted:
                scores[j] = -L / self.tau_sem
                continue

            if len(predicted) != L:
                min_len = min(L, len(predicted))
                mismatch = sum(weights[i] for i in range(min_len)
                               if predicted[i] != target_output[i])
                mismatch += sum(weights[i] for i in range(min_len, L))
            else:
                mismatch = sum(weights[i] for i in range(L)
                               if predicted[i] != target_output[i])

            scores[j] = -mismatch / self.tau_sem

        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        return probs / (probs.sum() + 1e-10)

    def clear_cache(self) -> None:
        """Clear prediction cache (e.g., after refresh)."""
        self._predict_cache.clear()

    @property
    def is_cls_active(self) -> bool:
        """Whether CLS agent is loaded and studied."""
        return self._agent is not None and self._studied


# ── Factory ────────────────────────────────────────────────────

def create_scorer(
    grammar: Grammar,
    support: Optional[List[Example]] = None,
    use_cls: bool = False,
    n_sup: int = 5,
    n_em: int = 2,
    use_hpc: bool = True,
    tau_sem: float = 1.0,
    lambda_neg: float = 0.0,
) -> SemanticPosteriorProtocol:
    """Factory: create the best available scorer.

    Args:
        grammar: parsed grammar
        support: support examples (needed for CLS)
        use_cls: True to attempt CLS adapter
        n_sup: number of support examples to use (subsetted)
        n_em: EM iterations for CLS learning
        use_hpc: whether CLS uses HPC memory
        tau_sem: semantic mismatch temperature
        lambda_neg: negative evidence penalty scale (0 = off, default)

    Returns:
        SemanticPosteriorProtocol implementation
    """
    if use_cls and support is not None:
        scorer = CLSSemanticPosterior(grammar, tau_sem, lambda_neg=lambda_neg)
        # Subsample support to n_sup examples
        sub_support = support[:min(n_sup, len(support))]
        if sub_support:
            scorer.study(sub_support, n_em=n_em, use_hpc=use_hpc)
        else:
            # L0 learner: initialize CLS with empty support (raw prior)
            scorer.study([], n_em=0, use_hpc=False)

        if scorer.is_cls_active:
            return scorer
        # else: CLS unavailable, fall through to deterministic

    # Default: deterministic scorer (oracle)
    from .semantic_scorer import DeterministicSemanticScorer
    return DeterministicSemanticScorer(grammar, tau_sem)
