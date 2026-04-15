"""
trace_analysis.py — Extract trace-level structural uncertainty signals.

Phase 5 core: compute per-word structural uncertainty (H_role, H_rep,
H_emit) and project to output positions via alignment matrix A_{iw}.

This is the key module that lets full_trace potentially outperform
role_only in teaching decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


@dataclass
class WordStructureAnalysis:
    """Per-word structural uncertainty from beam traces."""
    word: str
    H_role: float       # role entropy
    H_rep: float        # repeat_k entropy
    H_emit: float       # emit color entropy
    role_dist: Dict[str, float]   # {role: prob}
    rep_dist: Dict[int, float]    # {k: prob}
    emit_dist: Dict[str, float]   # {color: prob}


@dataclass
class TraceSalienceResult:
    """Full trace-level analysis for a query."""
    word_analyses: Dict[str, WordStructureAnalysis]
    alignment: np.ndarray    # shape (n_positions, n_words): A_{iw}
    T_i: np.ndarray          # shape (n_positions,): trace salience per position
    words: List[str]         # word list (column order of alignment)


def _simulate_position_alignment(
    trace: list,
    n_output: int,
) -> Dict[int, str]:
    """Simulate trace execution to find which word controls each output position.

    Returns: {output_position: word} mapping
    """
    pos_to_word: Dict[int, str] = {}
    pos = 0

    for step in trace:
        word = getattr(step, 'word', None)
        role = getattr(step, 'role', None)
        if word is None or role is None:
            continue

        if role == 'EMIT':
            if pos < n_output:
                pos_to_word[pos] = word
            pos += 1

        elif role == 'REPEAT':
            k = getattr(step, 'repeat_k', 2) or 2
            arity = getattr(step, 'arity', 1) or 1
            # REPEAT duplicates the top arity items k times
            # The repeated sections are controlled by this word
            n_repeated = arity * k
            for j in range(n_repeated):
                if pos < n_output:
                    pos_to_word[pos] = word
                pos += 1
            # Actually pos was already advanced by EMIT steps before
            # This is approximate — we attribute repeat expansion to the repeat word
            pass

        elif role in ('SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX'):
            # Infix operations consume next word and rearrange
            # The b_word controls one position
            b_word = getattr(step, 'b_word', None)
            if b_word and pos < n_output:
                pos_to_word[pos] = b_word
                pos += 1

    return pos_to_word


def _build_alignment_from_trace(
    trace: list,
    output_len: int,
    words: List[str],
) -> np.ndarray:
    """Build position-word alignment vector from a single trace.

    Returns: array of shape (output_len,) with word indices.
             -1 if position is unaligned.
    """
    word_to_idx = {w: i for i, w in enumerate(words)}
    alignment = np.full(output_len, -1, dtype=int)

    # Simple approach: walk through trace steps and track output positions
    out_pos = 0
    for step in trace:
        word = getattr(step, 'word', None)
        role = getattr(step, 'role', None)
        if word is None or role is None:
            continue

        w_idx = word_to_idx.get(word, -1)

        if role == 'EMIT':
            if out_pos < output_len:
                alignment[out_pos] = w_idx
            out_pos += 1

        elif role == 'REPEAT':
            k = getattr(step, 'repeat_k', 2) or 2
            # REPEAT of previous: attribute to this word
            # This is approximate — we just mark future positions
            for _ in range(k - 1):  # k-1 extra copies
                if out_pos < output_len:
                    alignment[out_pos] = w_idx
                out_pos += 1

        elif role in ('SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX'):
            b_word = getattr(step, 'b_word', None)
            if b_word:
                b_idx = word_to_idx.get(b_word, w_idx)
                if out_pos < output_len:
                    alignment[out_pos] = b_idx
                out_pos += 1
            # The main word's effect depends on role type
            if role == 'OVER_INFIX':
                # OVER: A, B, A — extra position for A
                if out_pos < output_len:
                    alignment[out_pos] = w_idx
                out_pos += 1

    return alignment


def analyze_trace_salience(
    beam: List[Tuple[float, list, List[str]]],
    words: List[str],
    gamma_r: float = 1.0,
    gamma_k: float = 0.5,
    gamma_e: float = 0.5,
) -> TraceSalienceResult:
    """Full trace-level analysis from beam posterior.

    Args:
        beam: [(score, trace, Y_k), ...] from CLSSequencePredictor
        words: input words for the query
        gamma_r: weight for role entropy in T_i
        gamma_k: weight for repeat entropy in T_i
        gamma_e: weight for emit entropy in T_i

    Returns:
        TraceSalienceResult with per-word analysis, alignment matrix, and T_i
    """
    if not beam:
        return TraceSalienceResult(
            word_analyses={}, alignment=np.zeros((0, 0)),
            T_i=np.array([]), words=words)

    K = len(beam)
    scores = np.array([b[0] for b in beam])
    log_q = scores - logsumexp(scores)
    q_k = np.exp(log_q)

    # Max output length across beam
    max_len = max((len(b[2]) for b in beam), default=0)
    n_words = len(words)

    if max_len == 0 or n_words == 0:
        return TraceSalienceResult(
            word_analyses={}, alignment=np.zeros((0, 0)),
            T_i=np.array([]), words=words)

    # ── Per-word structural distributions ──

    # Accumulate weighted counts
    role_counts: Dict[str, Dict[str, float]] = {w: {} for w in words}
    rep_counts: Dict[str, Dict[int, float]] = {w: {} for w in words}
    emit_counts: Dict[str, Dict[str, float]] = {w: {} for w in words}

    for k_idx in range(K):
        weight = q_k[k_idx]
        trace = beam[k_idx][1]
        y_k = beam[k_idx][2]

        for step in trace:
            w = getattr(step, 'word', None)
            role = getattr(step, 'role', None)
            if w is None or role is None or w not in role_counts:
                continue

            # Role distribution
            role_counts[w][role] = role_counts[w].get(role, 0) + weight

            # Repeat distribution
            if role == 'REPEAT':
                rk = getattr(step, 'repeat_k', 2) or 2
                rep_counts[w][rk] = rep_counts[w].get(rk, 0) + weight

            # Emit distribution (color)
            if role == 'EMIT' and hasattr(step, 'emit_vec') and step.emit_vec is not None:
                # Map emit_vec to nearest color
                from ns_learner.ns_concept import vec_to_color
                try:
                    color = vec_to_color(step.emit_vec)
                except Exception:
                    color = 'UNK'
                emit_counts[w][color] = emit_counts[w].get(color, 0) + weight

    # Compute per-word entropies
    word_analyses: Dict[str, WordStructureAnalysis] = {}
    for w in words:
        # Role entropy
        r_dist = role_counts[w]
        H_role = _entropy(r_dist)

        # Repeat entropy
        rp_dist = rep_counts[w]
        H_rep = _entropy(rp_dist)

        # Emit entropy
        e_dist = emit_counts[w]
        H_emit = _entropy(e_dist)

        word_analyses[w] = WordStructureAnalysis(
            word=w, H_role=H_role, H_rep=H_rep, H_emit=H_emit,
            role_dist=r_dist, rep_dist=rp_dist, emit_dist=e_dist)

    # ── Position-word alignment matrix A_{iw} ──
    # A[i, j] = P(position i controlled by word j | beam)
    A = np.zeros((max_len, n_words))

    for k_idx in range(K):
        weight = q_k[k_idx]
        trace = beam[k_idx][1]
        y_k = beam[k_idx][2]
        out_len = len(y_k)

        align = _build_alignment_from_trace(trace, out_len, words)
        for i in range(min(out_len, max_len)):
            w_idx = align[i]
            if 0 <= w_idx < n_words:
                A[i, w_idx] += weight

    # Normalize rows (each position sums to ~1)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    A = A / row_sums

    # ── Position-level trace salience T_i ──
    T_i = np.zeros(max_len)
    for i in range(max_len):
        for j, w in enumerate(words):
            wa = word_analyses.get(w)
            if wa is None:
                continue
            T_i[i] += A[i, j] * (
                gamma_r * wa.H_role +
                gamma_k * wa.H_rep +
                gamma_e * wa.H_emit
            )

    return TraceSalienceResult(
        word_analyses=word_analyses,
        alignment=A,
        T_i=T_i,
        words=words)


def _entropy(dist: Dict, eps: float = 1e-30) -> float:
    """Shannon entropy from a dict of {label: probability_mass}."""
    total = sum(dist.values())
    if total < eps:
        return 0.0
    probs = np.array([v / total for v in dist.values()])
    probs = probs[probs > eps]
    return float(-np.sum(probs * np.log(probs)))
