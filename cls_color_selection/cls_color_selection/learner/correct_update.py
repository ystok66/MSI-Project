"""
correct_update.py — Clean learner-side correct-answer grammar update.

Goal B module: apply GT-compatible constrained E-step + M-step
to the real learner's grammar, WITHOUT going through hint/assist.

This is the learner-side counterpart of TutorLearnerModel.update_from_output(),
adapted to work with the real CLSSequencePredictor.

Design decision:
  We reuse the SAME constrained beam mechanism as CLS study(),
  but applied as a MID-EPISODE update (not a full support re-study).
  This ensures correct_update is semantically comparable to wrong_update:
  both operate on the current grammar via differential-style updates.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


def apply_correct_answer(
    predictor,
    words: List[str],
    ground_truth: List[str],
    *,
    eta_correct: float = 1.0,
    update_depth: str = 'full_trace',
    mode: str = 'direct',
    history_state=None,
    lambda_hist: float = 1.0,
) -> dict:
    """Apply correct-answer update to learner grammar.

    Modes:
      'direct': Pure GT constrained E-step + M-step.
                No wrong history used.

      'route_a_support': Build augmented support set from wrong history.
                         Π_goalB = Π_GT ∪ Π_rescue(D⁻)
                         Then standard weighted M-step over merged set.

      'route_b': Standard GT constrained beam on CURRENT grammar state
                 (which may have been weakly modified by prior wrongs).
                 Same as direct but on a different θ.

    Args:
        predictor: CLSSequencePredictor (real learner)
        words: query input words
        ground_truth: correct output color sequence
        eta_correct: learning rate for M-step
        update_depth: 'role_only' | 'role_emit' | 'full_trace'
        mode: 'direct' | 'route_a_support' | 'route_b'
        history_state: GoalBProtocolState (required for route_a_support)
        lambda_hist: unused in current implementation, reserved

    Returns:
        dict with diagnostics
    """
    diag = {
        'n_traces': 0,
        'top_trace_score': 0.0,
        'update_applied': False,
        'mode': mode,
        'support_size': 0,
        'support_new_fraction': 0.0,
    }

    if not ground_truth:
        return diag

    agent = predictor.get_agent()
    if agent is None:
        return diag

    library = agent.cortex.library
    priors = agent.priors

    for w in words:
        agent.cortex._ensure_concept(w)

    if mode == 'route_a_support' and history_state is not None and history_state.n_wrongs > 0:
        # Route A-support: augmented support set
        from .goal_b_support import build_goal_b_support

        support_result = build_goal_b_support(
            predictor, words, ground_truth, history_state)

        traces = support_result['merged_traces']
        support_diag = support_result['diag']

        diag['support_size'] = support_diag['support_size']
        diag['support_new_fraction'] = support_diag['support_new_fraction']
        diag['n_rescue_new'] = support_diag['n_rescue_new']

        if not traces:
            return diag

        diag['n_traces'] = len(traces)
        diag['top_trace_score'] = float(traces[0][0])

        # Standard weighted M-step over merged support
        _depth_controlled_m_step(
            library, traces, words, eta_correct, update_depth)

    else:
        # Direct / Route B: standard GT constrained beam
        target_vecs = _color_to_vecs(ground_truth)
        if not target_vecs:
            return diag

        try:
            from ns_learner.ns_ast import infer_top_k_ast
            traces = infer_top_k_ast(words, target_vecs, library, priors)
        except Exception:
            return diag

        if not traces:
            return diag

        diag['n_traces'] = len(traces)
        diag['top_trace_score'] = float(traces[0][0])
        diag['support_size'] = len(traces)

        _depth_controlled_m_step(
            library, traces, words, eta_correct, update_depth)

    diag['update_applied'] = True
    return diag


def _color_to_vecs(colors: List[str]) -> list:
    """Convert color names to target vectors for constrained beam."""
    try:
        from ns_learner.ns_concept import color_to_vec
        return [color_to_vec(c) for c in colors]
    except Exception:
        return []


def _depth_controlled_m_step(
    library: dict,
    traces: list,
    words: List[str],
    eta: float,
    depth: str,
):
    """Apply weighted M-step from constrained traces.

    Mirrors TutorLearnerModel._depth_controlled_m_step().
    
    traces format from infer_top_k_ast:
      [(score, [ASTNode], [TraceStep]), ...]
    TraceSteps have .word, .role, .emit_vec, .repeat_k
    """
    if not traces:
        return

    # Compute trace posteriors (softmax of scores)
    scores = np.array([t[0] for t in traces])
    log_q = scores - logsumexp(scores)
    q = np.exp(log_q)

    for k, trace_item in enumerate(traces):
        weight = eta * q[k]
        if weight < 1e-15:
            continue

        # TraceSteps are the last element in the tuple
        trace_steps = trace_item[-1]

        for step in trace_steps:
            word = step.word
            if word not in library:
                continue
            concept = library[word]

            # Level 1: role_counts (always)
            role = step.role
            concept.role_counts[role] = concept.role_counts.get(role, 0.0) + weight

            if depth in ('role_emit', 'full_trace'):
                # Level 2: emission stats
                if role == 'EMIT' and hasattr(step, 'emit_vec') and step.emit_vec is not None:
                    vec = step.emit_vec
                    concept.emit_stats['sum_w'] += weight
                    concept.emit_stats['sum_wx'] += weight * vec
                    concept.emit_stats['sum_wx2'] += weight * (vec ** 2)

                    from ns_learner.ns_concept import vec_to_color
                    c = vec_to_color(vec)
                    concept.color_counts[c] = concept.color_counts.get(c, 0.0) + weight

            if depth == 'full_trace':
                # Level 3: repeat counts
                if role == 'REPEAT' and hasattr(step, 'repeat_k') and step.repeat_k is not None:
                    k_rep = step.repeat_k
                    if k_rep in concept.repeat_counts:
                        concept.repeat_counts[k_rep] += weight


def _depth_controlled_m_step_weighted(
    library: dict,
    traces: list,
    q_weights: np.ndarray,
    eta: float,
    depth: str,
):
    """M-step with pre-computed (history-conditioned) weights.

    Same as _depth_controlled_m_step but uses externally provided
    posterior weights instead of computing from trace scores.
    """
    if not traces:
        return

    for k, trace_item in enumerate(traces):
        weight = eta * q_weights[k]
        if weight < 1e-15:
            continue

        trace_steps = trace_item[-1]

        for step in trace_steps:
            word = step.word
            if word not in library:
                continue
            concept = library[word]

            role = step.role
            concept.role_counts[role] = concept.role_counts.get(role, 0.0) + weight

            if depth in ('role_emit', 'full_trace'):
                if role == 'EMIT' and hasattr(step, 'emit_vec') and step.emit_vec is not None:
                    vec = step.emit_vec
                    concept.emit_stats['sum_w'] += weight
                    concept.emit_stats['sum_wx'] += weight * vec
                    concept.emit_stats['sum_wx2'] += weight * (vec ** 2)

                    from ns_learner.ns_concept import vec_to_color
                    c = vec_to_color(vec)
                    concept.color_counts[c] = concept.color_counts.get(c, 0.0) + weight

            if depth == 'full_trace':
                if role == 'REPEAT' and hasattr(step, 'repeat_k') and step.repeat_k is not None:
                    k_rep = step.repeat_k
                    if k_rep in concept.repeat_counts:
                        concept.repeat_counts[k_rep] += weight
