"""
model.py — HBPI probabilistic model: per-word Dirichlet posteriors.

Tracks three families of distributions per word:
  1. P(type | w)   ∈ Δ{PRIM, UNARY, BINARY}    — Dirichlet(γ)
  2. P(color | w)  ∈ Δ{BLUE, RED, ...}           — Dirichlet(α)  [if PRIM]
  3. P(repeat | w) ∈ Δ{2, 3, 4}                  — Dirichlet(δ)  [if UNARY]

MAP estimates used for execution; full posteriors used for scoring.

Scoring (per parse p):
  log P(p | Θ) = -λ_len * |p|
                 + Σ_nodes log P(type(node) | w)
                 + Σ_unary_nodes log P(n | w)

Soft likelihood:
  log P(y | p, Θ) = -α_edit * edit_distance(ŷ, y)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────

COLORS = ['BLUE', 'RED', 'GREEN', 'YELLOW', 'PURPLE', 'PINK']
COLOR_SET = set(COLORS)
N_COLORS = len(COLORS)

TYPES = ['PRIM', 'UNARY', 'BINARY']
TYPE_PRIM, TYPE_UNARY, TYPE_BINARY = 0, 1, 2

REPEAT_SET = [2, 3, 4]       # MVP repeat factors
N_REPEATS = len(REPEAT_SET)

BINARY_MODES = ['swap', 'concat']  # swap=R+L, concat=L+R
N_BINARY_MODES = len(BINARY_MODES)


# ── Hyperparameters ────────────────────────────────────────────

class HBPIHyperparams:
    """All tunable knobs in one place."""
    def __init__(self,
                 gamma0: float = 1.0,      # Dirichlet prior for type
                 alpha0: float = 1.0,      # Dirichlet prior for color
                 delta0: float = 1.0,      # Dirichlet prior for repeat
                 lambda_len: float = 0.2,  # MDL length penalty
                 alpha_edit: float = 1.0,  # soft-likelihood sharpness
                 sub_weight: float = 0.3,  # alignment credit for substitution
                 K_span: int = 20,         # top-K parses per span
                 K_full: int = 50,         # top-K full parses
                 em_iters: int = 5):       # EM iterations
        self.gamma0 = gamma0
        self.alpha0 = alpha0
        self.delta0 = delta0
        self.lambda_len = lambda_len
        self.alpha_edit = alpha_edit
        self.sub_weight = sub_weight
        self.K_span = K_span
        self.K_full = K_full
        self.em_iters = em_iters


# ── Per-word distributions ─────────────────────────────────────

class WordModel:
    """Dirichlet posterior counts for one word."""

    def __init__(self, hp: HBPIHyperparams):
        self.type_counts = np.full(3, hp.gamma0)          # [PRIM, UNARY, BINARY]
        self.color_counts = np.full(N_COLORS, hp.alpha0)  # [BLUE, RED, ...]
        self.repeat_counts = np.full(N_REPEATS, hp.delta0)  # [n=2, n=3, n=4]
        self.binary_mode_counts = np.ones(N_BINARY_MODES)  # [swap, concat]

    @property
    def type_probs(self) -> np.ndarray:
        """Posterior mean P(type | w)."""
        return self.type_counts / self.type_counts.sum()

    @property
    def color_probs(self) -> np.ndarray:
        """Posterior mean P(color | w)."""
        return self.color_counts / self.color_counts.sum()

    @property
    def repeat_probs(self) -> np.ndarray:
        """Posterior mean P(repeat_n | w)."""
        return self.repeat_counts / self.repeat_counts.sum()

    @property
    def binary_mode_probs(self) -> np.ndarray:
        """Posterior mean P(binary_mode | w)."""
        return self.binary_mode_counts / self.binary_mode_counts.sum()

    def map_color(self) -> str:
        """Return the MAP color for this word."""
        return COLORS[int(np.argmax(self.color_counts))]

    def map_type(self) -> str:
        """Return the MAP type for this word."""
        return TYPES[int(np.argmax(self.type_counts))]

    def log_type_prob(self, type_idx: int) -> float:
        """log P(type | w)."""
        p = self.type_probs
        return float(np.log(p[type_idx] + 1e-30))

    def log_color_prob(self, color: str) -> float:
        """log P(color | w)."""
        if color not in COLOR_SET:
            return -20.0  # unknown color
        p = self.color_probs
        return float(np.log(p[COLORS.index(color)] + 1e-30))

    def log_repeat_prob(self, n: int) -> float:
        """log P(repeat_n | w)."""
        if n not in REPEAT_SET:
            return -20.0
        p = self.repeat_probs
        return float(np.log(p[REPEAT_SET.index(n)] + 1e-30))

    def log_binary_mode_prob(self, mode: str) -> float:
        """log P(binary_mode | w)."""
        if mode not in BINARY_MODES:
            return -20.0
        p = self.binary_mode_probs
        return float(np.log(p[BINARY_MODES.index(mode)] + 1e-30))


# ── Full model ─────────────────────────────────────────────────

class HBPIModel:
    """Complete HBPI model: collection of per-word distributions + scoring."""

    def __init__(self, hp: Optional[HBPIHyperparams] = None):
        self.hp = hp or HBPIHyperparams()
        self.words: Dict[str, WordModel] = {}

    def ensure(self, word: str) -> WordModel:
        """Lazily create a WordModel for a new word."""
        if word not in self.words:
            self.words[word] = WordModel(self.hp)
        return self.words[word]

    def map_color(self, word: str) -> str:
        """MAP color for a word (used during AST execution)."""
        return self.ensure(word).map_color()

    def word_type_probs(self, word: str) -> np.ndarray:
        """P(type | word)."""
        return self.ensure(word).type_probs

    # ── Scoring ────────────────────────────────────────────────

    def log_prior(self, ast) -> float:
        """
        MDL + type/repeat prior:
          log P(p | Θ) = -λ_len * |p| + Σ_nodes log P(type(node) | w) + Σ_unary log P(n | w)

        MDL penalty applied ONCE at top level; _node_log_probs accumulates
        per-node type/repeat contributions without MDL.
        """
        return -self.hp.lambda_len * ast.node_count + self._node_log_probs(ast)

    def _node_log_probs(self, ast) -> float:
        """Accumulate per-node type + repeat log-probs (no MDL penalty)."""
        from .grammar import Prim, Concat, Unary, Binary

        if isinstance(ast, Prim):
            return self.ensure(ast.word).log_type_prob(TYPE_PRIM)

        elif isinstance(ast, Concat):
            return (self._node_log_probs(ast.left) +
                    self._node_log_probs(ast.right))

        elif isinstance(ast, Unary):
            return (self.ensure(ast.op_word).log_type_prob(TYPE_UNARY) +
                    self.ensure(ast.op_word).log_repeat_prob(ast.repeat_n) +
                    self._node_log_probs(ast.arg))

        elif isinstance(ast, Binary):
            return (self.ensure(ast.op_word).log_type_prob(TYPE_BINARY) +
                    self.ensure(ast.op_word).log_binary_mode_prob(ast.binary_mode) +
                    self._node_log_probs(ast.left) +
                    self._node_log_probs(ast.right))

        return 0.0

    def log_likelihood(self, pred: List[str], gold: List[str]) -> float:
        """
        Soft likelihood:
          log P(y | p, Θ) = -α_edit * edit_distance(ŷ, y)
        """
        from .align import edit_distance
        d = edit_distance(pred, gold)
        return -self.hp.alpha_edit * d

    def score_parse(self, ast, gold: List[str]) -> float:
        """
        Total log score = log P(p | Θ) + log P(y | p, Θ).
        """
        from .executor import execute
        pred, prov = execute(ast, self)
        return self.log_prior(ast) + self.log_likelihood(pred, gold)

    # ── EM M-step: reset and update ────────────────────────────

    def reset_counts(self):
        """Reset all counts to priors (called at start of each M-step)."""
        for wm in self.words.values():
            wm.type_counts = np.full(3, self.hp.gamma0)
            wm.color_counts = np.full(N_COLORS, self.hp.alpha0)
            wm.repeat_counts = np.full(N_REPEATS, self.hp.delta0)
            wm.binary_mode_counts = np.ones(N_BINARY_MODES)

    def accumulate_counts(self, ast, gold: List[str], weight: float):
        """
        Accumulate expected counts from one parse with posterior weight.
        Uses alignment for color credit assignment.
        """
        from .grammar import Prim, Concat, Unary, Binary
        from .executor import execute
        from .align import edit_align, AlignOp

        # Execute to get pred + provenance
        pred, prov = execute(ast, self)

        # Accumulate type counts from AST structure
        self._accumulate_type_counts(ast, weight)

        # Accumulate color counts from alignment
        alignment = edit_align(pred, gold)
        for op, pi, gi in alignment:
            if pi < 0:
                continue  # insertion in gold — no pred token to credit
            leaf_word = prov[pi]
            wm = self.ensure(leaf_word)
            if op == AlignOp.MATCH:
                color_idx = COLORS.index(gold[gi])
                wm.color_counts[color_idx] += weight * 1.0
            elif op == AlignOp.SUB:
                # Substitution: partial credit to gold color
                color_idx = COLORS.index(gold[gi])
                wm.color_counts[color_idx] += weight * self.hp.sub_weight
            # DEL: pred token has no gold match → no color credit

    def _accumulate_type_counts(self, ast, weight: float):
        """Recursively accumulate type and repeat counts from AST."""
        from .grammar import Prim, Concat, Unary, Binary

        if isinstance(ast, Prim):
            self.ensure(ast.word).type_counts[TYPE_PRIM] += weight

        elif isinstance(ast, Concat):
            self._accumulate_type_counts(ast.left, weight)
            self._accumulate_type_counts(ast.right, weight)

        elif isinstance(ast, Unary):
            wm = self.ensure(ast.op_word)
            wm.type_counts[TYPE_UNARY] += weight
            # Repeat count
            if ast.repeat_n in REPEAT_SET:
                wm.repeat_counts[REPEAT_SET.index(ast.repeat_n)] += weight
            self._accumulate_type_counts(ast.arg, weight)

        elif isinstance(ast, Binary):
            wm = self.ensure(ast.op_word)
            wm.type_counts[TYPE_BINARY] += weight
            if ast.binary_mode in BINARY_MODES:
                wm.binary_mode_counts[BINARY_MODES.index(ast.binary_mode)] += weight
            self._accumulate_type_counts(ast.left, weight)
            self._accumulate_type_counts(ast.right, weight)

    def snapshot(self) -> Dict:
        """Return a human-readable snapshot of learned distributions."""
        result = {}
        for w, wm in sorted(self.words.items()):
            tp = wm.type_probs
            result[w] = {
                'map_type': wm.map_type(),
                'type_probs': {TYPES[i]: round(tp[i], 3) for i in range(3)},
                'map_color': wm.map_color() if tp[TYPE_PRIM] > 0.3 else '—',
            }
            if tp[TYPE_UNARY] > 0.1:
                rp = wm.repeat_probs
                result[w]['repeat_probs'] = {
                    str(n): round(rp[i], 3) for i, n in enumerate(REPEAT_SET)
                }
        return result
