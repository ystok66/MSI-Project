"""
diagnostics_45.py — Phase 4.5 decision diagnostics.

Metrics:
  1. DecisionDiff: fraction of queries where two conditions differ
  2. HintSetJaccard: Jaccard similarity of hinted position sets
  3. HintPrecision: fraction of hinted positions that were actually wrong
  4. UncertaintySensitivity: does hint rate vary with beam entropy?
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_decision_diff(
    diag_a: List[Dict],
    diag_b: List[Dict],
) -> Dict[str, float]:
    """Compare hint decisions between two conditions.

    Args:
        diag_a, diag_b: hint diagnostic logs from two tutor runs
                        (same queries, same order)

    Returns:
        {'decision_diff': float, 'hint_set_jaccard': float}
    """
    n = min(len(diag_a), len(diag_b))
    if n == 0:
        return {'decision_diff': 0.0, 'hint_set_jaccard': 1.0}

    n_diff = 0
    jaccards = []

    for i in range(n):
        da = diag_a[i].get('decision', 'WAIT')
        db = diag_b[i].get('decision', 'WAIT')

        if da != db:
            n_diff += 1

        # Jaccard on hinted positions
        sa = set(diag_a[i].get('hinted_positions', []))
        sb = set(diag_b[i].get('hinted_positions', []))
        if sa or sb:
            jaccard = len(sa & sb) / max(len(sa | sb), 1)
            jaccards.append(jaccard)

    return {
        'decision_diff': n_diff / n,
        'hint_set_jaccard': float(np.mean(jaccards)) if jaccards else 1.0,
        'n_compared': n,
    }


def compute_hint_precision(
    hint_diag_log: List[Dict],
    wrong_masks: List[List[bool]],
) -> Dict[str, float]:
    """Fraction of hinted positions that were actually wrong.

    HintPrec = # hinted positions that were wrong / # all hinted positions

    A perfect HintPrec=1.0 means tutor only hints wrong positions.

    Args:
        hint_diag_log: from InverseTutor._hint_diag_log
        wrong_masks: corresponding wrong_mask per query

    Returns:
        {'hint_precision': float, 'n_hints_given': int, 'n_positions_hinted': int}
    """
    n_correct_hints = 0
    n_total_hints = 0
    n_hint_events = 0

    for i, diag in enumerate(hint_diag_log):
        if diag.get('decision') != 'HINT':
            continue

        n_hint_events += 1
        positions = diag.get('hinted_positions', [])
        mask = wrong_masks[i] if i < len(wrong_masks) else []

        for pos in positions:
            n_total_hints += 1
            if pos < len(mask) and not mask[pos]:
                n_correct_hints += 1  # Position was actually wrong → good hint

    precision = n_correct_hints / max(n_total_hints, 1)
    return {
        'hint_precision': precision,
        'n_hints_given': n_hint_events,
        'n_positions_hinted': n_total_hints,
    }


def compute_uncertainty_sensitivity(
    hint_diag_log: List[Dict],
) -> Dict[str, float]:
    """Does hint rate vary with beam entropy?

    Split queries into low/high entropy (median split), compare hint rates.

    Returns:
        {'hint_rate_low_H': float, 'hint_rate_high_H': float,
         'sensitivity': float}
    """
    entropies = []
    hints = []

    for diag in hint_diag_log:
        H = diag.get('H_beam_norm', 0.0)
        is_hint = 1.0 if diag.get('decision') == 'HINT' else 0.0
        entropies.append(H)
        hints.append(is_hint)

    if len(entropies) < 4:
        return {'hint_rate_low_H': 0.0, 'hint_rate_high_H': 0.0,
                'sensitivity': 0.0}

    median_H = float(np.median(entropies))
    low_hints = [h for H, h in zip(entropies, hints) if H <= median_H]
    high_hints = [h for H, h in zip(entropies, hints) if H > median_H]

    rate_low = float(np.mean(low_hints)) if low_hints else 0.0
    rate_high = float(np.mean(high_hints)) if high_hints else 0.0

    return {
        'hint_rate_low_H': rate_low,
        'hint_rate_high_H': rate_high,
        'sensitivity': rate_high - rate_low,
        'median_H': median_H,
    }
