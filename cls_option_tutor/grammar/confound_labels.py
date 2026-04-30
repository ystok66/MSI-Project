"""
confound_labels.py — Lightweight diagnostic confound labeling for Phase 6.4.

Labels current distractors from the existing program pool.
Does NOT expand grammar or add new operator families.

Usage:
    labels = label_options(qs.menu, qs.target_output)
    for opt, label in zip(qs.menu, labels):
        print(f"{opt.index}: {label}")
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import List, Optional, Tuple

from ..interfaces import Option


class ConfoundType(str, Enum):
    CORRECT = "correct"
    NEAR_OUTPUT = "near_output"
    ORDER_LIKE = "order_like"
    CARDINALITY_LIKE = "cardinality_like"
    SCOPE_LIKE = "scope_like"
    FAR_DISTRACTOR = "far_distractor"


class DiagnosticRiskLabel(str, Enum):
    SAFE_DIAGNOSTIC_WRONG = "safe_diagnostic_wrong"
    BOUNDED_DIAGNOSTIC_WRONG = "bounded_diagnostic_wrong"
    HIGH_RISK_LURE = "high_risk_lure"
    SAFE_FAR = "safe_far"
    RISKY_FAR = "risky_far"
    SAFE_RANDOM_WRONG = "safe_random_wrong"  # Phase 6E: matched-risk non-diagnostic control
    CORRECT = "correct"


# Diagnostic set: confound types that encode informative confusion
_DIAGNOSTIC_SET = {
    ConfoundType.NEAR_OUTPUT,
    ConfoundType.ORDER_LIKE,
    ConfoundType.CARDINALITY_LIKE,
    ConfoundType.SCOPE_LIKE,
}


def _hamming_frac(a: list, b: list) -> float:
    """Normalized Hamming distance between two sequences."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    return diff / max_len


def _bigrams(seq: list) -> set:
    """Return set of bigrams from a sequence."""
    return set(zip(seq, seq[1:])) if len(seq) >= 2 else set()


def _shared_bigram_ratio(a: list, b: list) -> float:
    """Fraction of bigrams shared between two sequences."""
    a_bi = _bigrams(a)
    b_bi = _bigrams(b)
    union = a_bi | b_bi
    if not union:
        return 0.0
    return len(a_bi & b_bi) / len(union)


def label_confound(
    opt_output: list,
    target_output: list,
    is_correct: bool = False,
) -> ConfoundType:
    """Label a single option with its confound type.

    Uses only rendered output comparison — does NOT depend on
    hidden correctness except to tag the correct option.

    Args:
        opt_output: The rendered output of this option
        target_output: The correct target output
        is_correct: Whether this IS the correct option
    """
    if is_correct:
        return ConfoundType.CORRECT

    h = _hamming_frac(opt_output, target_output)

    # Near output: h <= 0.25
    if h <= 0.25:
        return ConfoundType.NEAR_OUTPUT

    # Order-like: same bag of tokens, different order
    if Counter(opt_output) == Counter(target_output) and opt_output != target_output:
        return ConfoundType.ORDER_LIKE

    # Cardinality-like: length differs by exactly 1
    if abs(len(opt_output) - len(target_output)) == 1:
        return ConfoundType.CARDINALITY_LIKE

    # Scope-like: shared bigram ratio >= 0.5
    if _shared_bigram_ratio(opt_output, target_output) >= 0.5:
        return ConfoundType.SCOPE_LIKE

    return ConfoundType.FAR_DISTRACTOR


def label_diagnostic_risk(
    confound: ConfoundType,
    risk_class: int,
) -> DiagnosticRiskLabel:
    """Combine confound type with risk class into diagnostic risk label."""
    if confound == ConfoundType.CORRECT:
        return DiagnosticRiskLabel.CORRECT

    if confound in _DIAGNOSTIC_SET:
        if risk_class == 0:
            return DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG
        elif risk_class in (1, 2):
            return DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG
        else:
            return DiagnosticRiskLabel.HIGH_RISK_LURE
    else:
        if risk_class == 0:
            return DiagnosticRiskLabel.SAFE_FAR
        else:
            return DiagnosticRiskLabel.RISKY_FAR


def label_options(
    menu: List[Option],
    target_output: list,
) -> Tuple[List[ConfoundType], List[DiagnosticRiskLabel]]:
    """Label all options in a menu.

    Returns:
        confound_types: List of ConfoundType labels
        diag_risk: List of DiagnosticRiskLabel labels
    """
    confounds = []
    diag_risks = []

    for opt in menu:
        opt_out = list(opt.rendered_output) if hasattr(opt, 'rendered_output') and opt.rendered_output else []
        ct = label_confound(opt_out, target_output, is_correct=opt.is_correct)
        dr = label_diagnostic_risk(ct, opt.risk_class)
        confounds.append(ct)
        diag_risks.append(dr)

    return confounds, diag_risks
