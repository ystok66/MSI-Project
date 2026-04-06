"""
parser.py — Chart-based parse enumeration with top-K pruning.

Enumerates candidate ASTs for an input token sequence using memoized DP
over spans (i, j). Each span produces a list of ASTs ranked by structural
prior score (MDL + type probs). At the full span, re-rank with likelihood.

Two modes:
  - Training (gold available): explore all type assignments, re-rank with likelihood
  - Query (no gold): use MAP type constraints from learned model to restrict search

Rules applied per span (i, j):
  1. Prim:         if i == j AND token's MAP type is PRIM, produce Prim(tokens[i])
  2. Postfix Unary: if token at j has MAP type UNARY, for arg ∈ Parses(i, j-1)
  3. Infix Binary:  for split m, if token at m has MAP type BINARY
  4. Concat:        for each split k (always allowed)

Deduplication by canonical serialization. Top-K pruning per span.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import heapq

from .grammar import AST, Prim, Concat, Unary, Binary, BINARY_MODES
from .model import HBPIModel, REPEAT_SET, TYPE_PRIM, TYPE_UNARY, TYPE_BINARY


def enumerate_parses(tokens: List[str],
                     model: HBPIModel,
                     gold: Optional[List[str]] = None,
                     constrained: bool = False
                     ) -> List[Tuple[float, AST]]:
    """
    Enumerate candidate parses for a token sequence.

    Args:
        tokens: input words [w0, w1, ..., wn-1]
        model: current HBPI model (for scoring)
        gold: expected output (for final re-ranking with likelihood); None for query-time
        constrained: if True, only allow parses consistent with MAP types

    Returns:
        List of (log_score, ast) sorted descending by score.
        Length ≤ K_full.
    """
    n = len(tokens)
    if n == 0:
        return []

    K_span = model.hp.K_span
    K_full = model.hp.K_full

    # Pre-compute MAP types for constrained mode
    # A word is "OP-like" if its MAP type is UNARY or BINARY
    # In unconstrained mode, all type assignments are explored (threshold 0.01)
    word_can_be = {}  # word -> set of allowed types
    for w in set(tokens):
        wm = model.ensure(w)
        tp = wm.type_probs
        if constrained:
            # Hard constraint: only allow MAP type
            # But also allow PRIM if it's close to MAP (within 0.1)
            map_t = wm.map_type()
            allowed = {map_t}
            # Always allow a word to be PRIM in a single-token span
            # (needed for structural completeness)
            allowed.add('PRIM')
            word_can_be[w] = allowed
        else:
            allowed = set()
            if tp[TYPE_PRIM] > 0.01:
                allowed.add('PRIM')
            if tp[TYPE_UNARY] > 0.01:
                allowed.add('UNARY')
            if tp[TYPE_BINARY] > 0.01:
                allowed.add('BINARY')
            word_can_be[w] = allowed

    # Memoized chart
    chart: Dict[Tuple[int, int], List[Tuple[float, AST]]] = {}

    def get_span(i: int, j: int) -> List[Tuple[float, AST]]:
        if (i, j) in chart:
            return chart[(i, j)]

        candidates: Dict[str, Tuple[float, AST]] = {}

        def add(score: float, ast: AST):
            key = ast.canonical()
            if key not in candidates or score > candidates[key][0]:
                candidates[key] = (score, ast)

        span_len = j - i + 1

        # Rule 1: Prim (single token)
        if span_len == 1:
            w = tokens[i]
            if 'PRIM' in word_can_be.get(w, {'PRIM'}):
                ast = Prim(w)
                score = model.log_prior(ast)
                add(score, ast)

        # Rule 2: Postfix Unary — tokens[j] is operator, arg = (i, j-1)
        if span_len >= 2:
            op_w = tokens[j]
            if 'UNARY' in word_can_be.get(op_w, set()):
                args = get_span(i, j - 1)
                for r_n in REPEAT_SET:
                    for arg_score, arg_ast in args:
                        ast = Unary(op_w, r_n, arg_ast)
                        score = model.log_prior(ast)
                        add(score, ast)

        # Rule 3: Infix Binary — tokens[m] is operator between L and R
        if span_len >= 3:
            for m in range(i + 1, j):
                op_w = tokens[m]
                if 'BINARY' in word_can_be.get(op_w, set()):
                    lefts = get_span(i, m - 1)
                    rights = get_span(m + 1, j)
                    for bmode in BINARY_MODES:
                        for ls, la in lefts:
                            for rs, ra in rights:
                                ast = Binary(op_w, bmode, la, ra)
                                score = model.log_prior(ast)
                                add(score, ast)

        # Rule 4: Concat — any split point
        if span_len >= 2:
            for k in range(i, j):
                lefts = get_span(i, k)
                rights = get_span(k + 1, j)
                for ls, la in lefts:
                    for rs, ra in rights:
                        ast = Concat(la, ra)
                        score = model.log_prior(ast)
                        add(score, ast)

        # Top-K pruning by prior score
        all_cands = list(candidates.values())
        if len(all_cands) > K_span:
            all_cands = heapq.nlargest(K_span, all_cands, key=lambda x: x[0])

        chart[(i, j)] = all_cands
        return all_cands

    # Build chart bottom-up
    for span_len in range(1, n + 1):
        for i in range(n - span_len + 1):
            j = i + span_len - 1
            get_span(i, j)

    # Get full-span candidates
    full_candidates = get_span(0, n - 1)

    if gold is not None:
        # Re-rank with full score = prior + likelihood
        scored = []
        for prior_score, ast in full_candidates:
            total_score = model.score_parse(ast, gold)
            scored.append((total_score, ast))
    else:
        scored = full_candidates

    # Sort descending and take top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:K_full]
