"""
goal_b_support.py — Support set builder for Goal B Route A-support.

Core idea: instead of only using Π_GT for the final correct commit,
augment the support with "rescue traces" derived from wrong submissions.

Rescue traces are found by running constrained beam search against
each wrong output, then filtering for structural proximity to GT.

This gives the final M-step access to trace structures that are
relevant to the wrong history but not contained in pure Π_GT.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


def build_goal_b_support(
    predictor,
    words: List[str],
    ground_truth: List[str],
    history_state,
    *,
    max_gt_traces: int = 16,
    max_rescue_traces: int = 8,
) -> dict:
    """Build augmented support set for Route A-support.

    1. Get GT-compatible traces (standard constrained beam)
    2. For each wrong submission in history, get wrong-compatible traces
    3. Filter wrong traces by structural proximity to GT
    4. Merge and deduplicate

    Args:
        predictor: CLSSequencePredictor
        words: query input
        ground_truth: correct output
        history_state: GoalBProtocolState with wrong history
        max_gt_traces: max GT-compatible traces to keep
        max_rescue_traces: max rescue traces total

    Returns:
        dict with gt_traces, rescue_traces, merged_traces, diag
    """
    agent = predictor.get_agent()
    if agent is None:
        return _empty_result()

    library = agent.cortex.library
    priors = agent.priors

    # Ensure vocabulary
    for w in words:
        agent.cortex._ensure_concept(w)

    # Step 1: GT-compatible traces
    gt_vecs = _color_to_vecs(ground_truth)
    if not gt_vecs:
        return _empty_result()

    try:
        from ns_learner.ns_ast import infer_top_k_ast
        gt_traces = infer_top_k_ast(
            words, gt_vecs, library, priors, k=max_gt_traces)
    except Exception:
        gt_traces = []

    if not gt_traces:
        return _empty_result()

    # Step 2: Rescue traces from wrong submissions
    rescue_traces = []
    wrong_trace_sources = []

    if history_state is not None:
        for obs in history_state.wrong_history:
            wrong_vecs = _color_to_vecs(obs.submitted)
            if not wrong_vecs:
                continue

            try:
                wrong_traces = infer_top_k_ast(
                    words, wrong_vecs, library, priors,
                    k=max_rescue_traces)
            except Exception:
                continue

            # Filter: keep wrong traces structurally close to GT
            for wt in wrong_traces:
                proximity = _structural_proximity(wt, ground_truth, obs.submitted)
                if proximity > 0:
                    rescue_traces.append((wt, proximity, obs.step_index))
                    wrong_trace_sources.append(obs.step_index)

    # Sort rescue by proximity, keep top
    rescue_traces.sort(key=lambda x: -x[1])
    rescue_traces = rescue_traces[:max_rescue_traces]

    # Step 3: Merge and deduplicate
    seen_keys = set()
    merged = []

    for t in gt_traces:
        key = _trace_key(t)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(t)

    n_new = 0
    for t, prox, src in rescue_traces:
        key = _trace_key(t)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(t)
            n_new += 1

    return {
        'gt_traces': gt_traces,
        'rescue_traces': [(t, p) for t, p, s in rescue_traces],
        'merged_traces': merged,
        'diag': {
            'n_gt': len(gt_traces),
            'n_rescue_candidates': len(rescue_traces),
            'n_rescue_new': n_new,
            'support_size': len(merged),
            'support_new_fraction': n_new / max(len(merged), 1),
        },
    }


def _structural_proximity(
    wrong_trace: tuple,
    ground_truth: List[str],
    wrong_output: List[str],
) -> float:
    """Score how structurally relevant a wrong-compatible trace is to GT.

    R(π; GT, ŷ) = sim_len(Y(π), GT) + pos_match(Y(π), GT)

    A wrong trace that's structurally close to a GT interpretation
    (e.g. same roles, just different emit color) is a useful rescue
    trace — it provides structural disambiguation information.

    Returns:
        proximity score >= 0 (higher = more useful as rescue)
    """
    trace_steps = wrong_trace[-1]
    if not trace_steps:
        return 0.0

    # Compute the trace's output length
    trace_len = 0
    for step in trace_steps:
        if step.role == 'EMIT':
            trace_len += 1
        elif step.role == 'REPEAT':
            trace_len += (step.repeat_k or 1)

    gt_len = len(ground_truth)
    wrong_len = len(wrong_output)

    # Length similarity to GT (0 to 1)
    len_diff = abs(trace_len - gt_len)
    sim_len = max(0.0, 1.0 - len_diff / max(gt_len, 1))

    # Position match: how many positions in wrong output agree with GT?
    min_len = min(len(wrong_output), len(ground_truth))
    n_match = sum(1 for i in range(min_len)
                  if wrong_output[i] == ground_truth[i])
    pos_match = n_match / max(min_len, 1)

    # Role overlap with GT-typical structure
    # (traces that share more role assignments are more structurally similar)
    role_bonus = 0.0
    n_emit = sum(1 for s in trace_steps if s.role == 'EMIT')
    n_repeat = sum(1 for s in trace_steps if s.role == 'REPEAT')

    # Prefer traces with similar role distribution to GT length
    if n_repeat > 0 and trace_len > 0:
        # Traces using REPEAT are structurally more informative
        role_bonus = 0.3

    proximity = sim_len + pos_match + role_bonus

    # Threshold: must have at least some structural relevance
    if sim_len < 0.3:
        return 0.0

    return proximity


def _trace_key(trace_item: tuple) -> tuple:
    """Structural deduplication key for a trace."""
    trace_steps = trace_item[-1]
    parts = []
    for step in trace_steps:
        rk = getattr(step, 'repeat_k', None)
        parts.append((step.word, step.role, rk))
    return tuple(parts)


def _color_to_vecs(colors: List[str]) -> list:
    try:
        from ns_learner.ns_concept import color_to_vec
        return [color_to_vec(c) for c in colors]
    except Exception:
        return []


def _empty_result():
    return {
        'gt_traces': [],
        'rescue_traces': [],
        'merged_traces': [],
        'diag': {
            'n_gt': 0, 'n_rescue_candidates': 0, 'n_rescue_new': 0,
            'support_size': 0, 'support_new_fraction': 0.0,
        },
    }
