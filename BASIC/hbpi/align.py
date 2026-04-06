"""
align.py — Levenshtein distance + alignment backtrace over color-token sequences.

Used for:
  1. Soft likelihood: P(y | p, Θ) = exp(-α * edit_distance(ŷ, y))
  2. EM M-step: alignment path maps predicted tokens to gold tokens,
     enabling soft credit assignment to leaf words via provenance.
"""

from __future__ import annotations
from typing import List, Tuple
from enum import Enum


class AlignOp(Enum):
    MATCH = 'match'       # pred[i] == gold[j]
    SUB   = 'substitute'  # pred[i] != gold[j]
    INS   = 'insert'      # gold token with no pred counterpart
    DEL   = 'delete'      # pred token with no gold counterpart


def edit_distance(pred: List[str], gold: List[str]) -> int:
    """Token-level Levenshtein distance."""
    m, n = len(pred), len(gold)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cost = 0 if pred[i-1] == gold[j-1] else 1
            tmp = dp[j]
            dp[j] = min(dp[j] + 1,      # delete pred[i]
                        dp[j-1] + 1,     # insert gold[j]
                        prev + cost)      # match/sub
            prev = tmp
    return dp[n]


def edit_align(pred: List[str], gold: List[str]) -> List[Tuple[AlignOp, int, int]]:
    """
    Full alignment backtrace.

    Returns a list of (op, pred_idx, gold_idx) tuples.
    For INS: pred_idx = -1.  For DEL: gold_idx = -1.
    """
    m, n = len(pred), len(gold)

    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i-1] == gold[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)

    # Backtrace
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if pred[i-1] == gold[j-1] else 1
            if dp[i][j] == dp[i-1][j-1] + cost:
                if cost == 0:
                    ops.append((AlignOp.MATCH, i-1, j-1))
                else:
                    ops.append((AlignOp.SUB, i-1, j-1))
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append((AlignOp.DEL, i-1, -1))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append((AlignOp.INS, -1, j-1))
            j -= 1
        else:
            break  # should not happen

    ops.reverse()
    return ops
