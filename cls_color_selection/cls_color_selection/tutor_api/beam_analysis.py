"""
beam_analysis.py — Extract distributional signals from beam posterior.

Phase 4.5 core: convert raw beam [(score, trace, Y_k)] into
per-query and per-position uncertainty signals.

Used by hint_policy.py to make fine-grained teaching decisions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


@dataclass
class BeamQueryAnalysis:
    """Query-level signals from beam posterior."""
    # Posterior weights (softmax of scores)
    q_k: np.ndarray          # shape (K,)
    # Candidate outputs
    Y_k: List[List[str]]     # K candidate color sequences

    # Core signals
    p_exact: float            # P(top-1 == gt)
    H_beam: float             # beam entropy (nats)
    H_beam_norm: float        # normalized: H / log(K)
    margin: float             # q_1 - q_2 (top-1/top-2 margin)
    E_wrong: float            # expected wrong ratio vs gt

    # Per-position analysis
    positions: List[PositionAnalysis] = field(default_factory=list)


@dataclass
class PositionAnalysis:
    """Per-position signals."""
    idx: int
    p_wrong: float            # P(position i is wrong)
    H_i: float                # color entropy at position i
    color_dist: Dict[str, float]  # {color: prob}
    gt_color: Optional[str] = None  # ground truth at this position


def analyze_beam(
    beam: List[Tuple[float, list, List[str]]],
    gt: Optional[List[str]] = None,
) -> BeamQueryAnalysis:
    """Full beam analysis: query-level + per-position.

    Args:
        beam: [(score, trace, Y_k), ...] from CLSSequencePredictor
        gt: ground truth output (if known)

    Returns:
        BeamQueryAnalysis with all distributional signals
    """
    if not beam:
        return BeamQueryAnalysis(
            q_k=np.array([1.0]), Y_k=[gt or []],
            p_exact=1.0, H_beam=0.0, H_beam_norm=0.0,
            margin=1.0, E_wrong=0.0)

    K = len(beam)
    scores = np.array([b[0] for b in beam])
    Y_k = [b[2] for b in beam]

    # Softmax posterior
    log_q = scores - logsumexp(scores)
    q_k = np.exp(log_q)

    # p_exact: probability that beam output matches gt exactly
    p_exact = 0.0
    if gt is not None:
        for k in range(K):
            if Y_k[k] == list(gt):
                p_exact += q_k[k]

    # Beam entropy
    H_beam = -np.sum(q_k * np.log(np.maximum(q_k, 1e-30)))
    H_beam_norm = H_beam / max(np.log(max(K, 2)), 1e-10)

    # Top-1 / Top-2 margin
    sorted_q = np.sort(q_k)[::-1]
    margin = sorted_q[0] - (sorted_q[1] if K > 1 else 0.0)

    # Expected wrong ratio
    E_wrong = 0.0
    if gt is not None:
        gt_list = list(gt)
        L = len(gt_list)
        if L > 0:
            for k in range(K):
                n_wrong = sum(1 for i in range(min(len(Y_k[k]), L))
                              if Y_k[k][i] != gt_list[i])
                # Also count length mismatches as wrong
                n_wrong += abs(len(Y_k[k]) - L)
                E_wrong += q_k[k] * (n_wrong / L)

    # Per-position analysis
    positions = []
    max_len = max((len(y) for y in Y_k), default=0)
    if gt is not None:
        max_len = max(max_len, len(gt))

    for i in range(max_len):
        # Color distribution at position i
        color_dist: Dict[str, float] = {}
        for k in range(K):
            if i < len(Y_k[k]):
                c = Y_k[k][i]
                color_dist[c] = color_dist.get(c, 0.0) + q_k[k]
            else:
                color_dist['<EMPTY>'] = color_dist.get('<EMPTY>', 0.0) + q_k[k]

        # p_wrong at position i
        p_wrong = 0.0
        gt_color = None
        if gt is not None and i < len(gt):
            gt_color = gt[i]
            p_wrong = 1.0 - color_dist.get(gt_color, 0.0)

        # Position entropy
        probs = np.array(list(color_dist.values()))
        probs = probs[probs > 0]
        H_i = -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0

        positions.append(PositionAnalysis(
            idx=i, p_wrong=p_wrong, H_i=H_i,
            color_dist=color_dist, gt_color=gt_color))

    return BeamQueryAnalysis(
        q_k=q_k, Y_k=Y_k,
        p_exact=p_exact, H_beam=H_beam, H_beam_norm=H_beam_norm,
        margin=margin, E_wrong=E_wrong,
        positions=positions)


def compute_p_succ_wait(p_exact: float, c_left: int) -> float:
    """P(success before timeout | WAIT).

    Assumes each confirm attempt has independent p_exact of being right.
    P_succ = 1 - (1 - p_exact)^c_left
    """
    if c_left <= 0:
        return 0.0
    return 1.0 - (1.0 - min(p_exact, 1.0)) ** c_left


def compute_p_succ_with_hints(
    analysis: BeamQueryAnalysis,
    hint_positions: List[int],
    c_left: int,
) -> float:
    """P(success | hint these positions).

    If we fix positions in hint_positions to their correct color,
    the learner only needs to get the remaining positions right.

    Approximate: for unhinted wrong positions, use their p_wrong.
    """
    if not analysis.positions:
        return compute_p_succ_wait(analysis.p_exact, c_left)

    # After hinting, remaining error probability
    p_all_remaining_correct = 1.0
    for pos in analysis.positions:
        if pos.idx in hint_positions:
            continue  # This position is fixed → correct
        if pos.gt_color is not None:
            p_correct_i = 1.0 - pos.p_wrong
            p_all_remaining_correct *= max(p_correct_i, 0.01)

    # P(success per confirm) = p_all_remaining_correct
    return 1.0 - (1.0 - p_all_remaining_correct) ** max(c_left, 1)
