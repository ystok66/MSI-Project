"""
goal_b_state.py — Protocol-local state for Goal B experiments.

v3: compute_history_factor uses trace-level output generation
to measure how "far" this trace's structural choices are from
the wrong submissions. The key insight is that GT-compatible
traces differ in their GENERATIVE FLEXIBILITY — some traces
could, with small parameter changes, also produce the wrong
output, while others are structurally locked to only GT.

Traces that are "close" to producing wrong outputs should
get LOWER weight (their structural choices are suspect).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class WrongObservation:
    """One wrong feedback observation (query-local, no grammar write)."""
    submitted: List[str]
    wrong_mask: List[bool]
    top1_before: Optional[List[str]] = None
    step_index: int = 0


@dataclass
class GoalBProtocolState:
    """Per-query protocol state accumulating wrong history."""
    words: List[str]
    ground_truth: List[str]
    wrong_history: List[WrongObservation] = field(default_factory=list)

    def add_wrong(
        self,
        submitted: List[str],
        mask: List[bool],
        top1_before: Optional[List[str]] = None,
    ):
        obs = WrongObservation(
            submitted=submitted,
            wrong_mask=mask,
            top1_before=top1_before,
            step_index=len(self.wrong_history),
        )
        self.wrong_history.append(obs)

    @property
    def n_wrongs(self) -> int:
        return len(self.wrong_history)


def compute_history_factor(
    trace_item: tuple,
    ground_truth: List[str],
    wrong_history: List[WrongObservation],
    lambda_hist: float = 1.0,
) -> float:
    """Compute C(π; D⁻) — history conditioning factor for one GT-compatible trace.

    v3: Uses structural features of the trace that DIFFERENTIATE it
    from wrong outputs:

    1. Length distance: how far is the trace's output length structure
       from each wrong output's length?  (REPEAT k values matter here)

    2. Emit pattern distance: how different is the trace's color
       sequence structure from the wrong outputs?

    A trace whose structural parameters (repeat-k, role assignments)
    are maximally incompatible with generating the wrong outputs gets
    HIGHER weight — it's the most "confirmed" by the wrong evidence.

    Args:
        trace_item: (score, [ASTNode], [TraceStep]) from constrained beam
        ground_truth: correct GT output
        wrong_history: list of WrongObservation
        lambda_hist: conditioning strength

    Returns:
        C(π; D⁻) factor (>= 0)
    """
    if not wrong_history:
        return 1.0

    trace_steps = trace_item[-1]
    if not trace_steps:
        return 1.0

    # Extract trace structural features
    features = _extract_trace_features(trace_steps)

    total_score = 0.0

    for obs in wrong_history:
        submitted = obs.submitted

        # Feature 1: Output length distance
        # If trace uses REPEAT(k=3) to produce GT of length 3,
        # and wrong output is length 2, then this trace is "far"
        # from wrong → high score.
        # If trace uses CONCAT(EMIT, EMIT, EMIT) to produce length 3,
        # a length-2 wrong could come from dropping one EMIT → close.
        length_distance = _length_structural_distance(
            features, len(ground_truth), len(submitted))

        # Feature 2: Color pattern compatibility
        # How many positions could this trace's role structure
        # "explain" in the wrong output?
        pattern_distance = _pattern_structural_distance(
            features, ground_truth, submitted)

        # Combined distance: higher = trace is more incompatible w/ wrong
        total_score += length_distance + pattern_distance

    if wrong_history:
        avg_score = total_score / len(wrong_history)
    else:
        avg_score = 0.0

    return np.exp(lambda_hist * avg_score)


def _extract_trace_features(trace_steps) -> dict:
    """Extract structural features from trace steps."""
    features = {
        'roles': [],       # [(word, role), ...]
        'repeat_ks': [],   # [k1, k2, ...] for REPEAT steps
        'n_emits': 0,      # direct EMIT count
        'n_repeats': 0,    # REPEAT count
        'total_output_len': 0,  # total output length this trace generates
        'word_role_map': {},    # {word: role}
    }

    output_len = 0
    for step in trace_steps:
        features['roles'].append((step.word, step.role))
        features['word_role_map'][step.word] = step.role

        if step.role == 'EMIT':
            features['n_emits'] += 1
            output_len += 1
        elif step.role == 'REPEAT':
            k = step.repeat_k or 1
            features['repeat_ks'].append(k)
            features['n_repeats'] += 1
            output_len += k

    features['total_output_len'] = output_len
    return features


def _length_structural_distance(
    features: dict,
    gt_len: int,
    wrong_len: int,
) -> float:
    """How structurally "locked" is this trace's length to GT?

    Key insight: a trace using REPEAT(k=3) can ONLY produce length 3
    from that word. To produce length 2, it would need k=2 — a
    discrete structural change. This is "far".

    A trace using 3 × EMIT can produce length 2 by dropping one EMIT.
    This is "closer" structurally.

    Returns:
      Higher = trace is more length-locked to GT (good for history)
    """
    if gt_len == wrong_len:
        return 0.0  # same length → no length-based discrimination

    length_diff = abs(wrong_len - gt_len)

    # If trace uses REPEAT, it's structurally locked to GT length
    if features['repeat_ks']:
        # Each repeat-k is a discrete constraint
        # The more of the output comes from repeats,
        # the harder it is to produce a different length
        repeat_fraction = sum(features['repeat_ks']) / max(features['total_output_len'], 1)
        return length_diff * repeat_fraction
    else:
        # Pure EMIT/CONCAT trace — more flexible
        return length_diff * 0.2  # small distance


def _pattern_structural_distance(
    features: dict,
    ground_truth: List[str],
    wrong_output: List[str],
) -> float:
    """How different is this trace's color pattern from wrong output?

    Uses the fact that different traces may use different words
    to produce colors at the same position. If the wrong output
    has a color at position j that's different from GT, and this
    trace uses a REPEAT to fill that position, then the trace
    CANNOT produce that wrong color without changing its repeat
    source — making it "far" from wrong.

    Returns:
      Higher = trace pattern is more incompatible with wrong
    """
    score = 0.0
    min_len = min(len(ground_truth), len(wrong_output))

    n_diff = 0
    for j in range(min_len):
        if ground_truth[j] != wrong_output[j]:
            n_diff += 1

    if n_diff == 0:
        # Outputs agree on shared positions — only length differs
        return 0.0

    # Fraction of positions with different colors
    diff_fraction = n_diff / max(min_len, 1)

    # Traces with REPEAT are more constrained at diff positions
    if features['repeat_ks']:
        # If repeat produces a block, and that block disagrees
        # with wrong, the whole block is "locked"
        repeat_coverage = sum(features['repeat_ks']) / max(features['total_output_len'], 1)
        score = diff_fraction * (1.0 + repeat_coverage)
    else:
        score = diff_fraction * 0.5

    return score
