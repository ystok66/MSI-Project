"""
goal_b_metrics.py — Grammar learning metrics for Goal B protocol.

Tracks: gt_rank, gt_mass, beam_entropy, margin, top1, probe_acc.
All metrics are snapshot-based: call before/after each update step.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


@dataclass
class GrammarSnapshot:
    """One snapshot of grammar state for a query."""
    step_label: str           # e.g. 'initial', 'after_wrong_1', 'after_correct'
    step_index: int
    # Beam-level
    H_beam: float             # beam entropy (nats)
    gt_rank: int              # rank of first GT-matching candidate (1-indexed, inf if absent)
    gt_mass: float            # total posterior mass on GT-matching candidates
    top1: Optional[List[str]] # top-1 beam output
    top1_score: float         # top-1 score
    margin: float             # q_1 - q_2
    n_beam: int               # beam size
    # Probe-level (optional, filled by runner)
    probe_acc: Optional[float] = None
    probe_ll: Optional[float] = None


def take_grammar_snapshot(
    predictor,
    words: List[str],
    ground_truth: List[str],
    step_label: str,
    step_index: int,
) -> GrammarSnapshot:
    """Snapshot current grammar state for one query.

    Args:
        predictor: CLSSequencePredictor
        words: query input
        ground_truth: correct output
        step_label: descriptive label
        step_index: 0-indexed step counter

    Returns:
        GrammarSnapshot
    """
    beam = predictor.beam_posterior(words)
    if not beam:
        return GrammarSnapshot(
            step_label=step_label, step_index=step_index,
            H_beam=0.0, gt_rank=999, gt_mass=0.0,
            top1=None, top1_score=0.0, margin=0.0, n_beam=0,
        )

    K = len(beam)
    scores = np.array([b[0] for b in beam])
    log_q = scores - logsumexp(scores)
    q = np.exp(log_q)

    # Beam entropy
    H = -np.sum(q * np.log(np.clip(q, 1e-30, 1.0)))

    # GT rank and mass
    gt_rank = 999
    gt_mass = 0.0
    for k in range(K):
        Y_k = beam[k][2]
        if Y_k == ground_truth:
            gt_mass += q[k]
            if gt_rank == 999:
                gt_rank = k + 1  # 1-indexed

    # Top-1 and margin
    top1 = beam[0][2]
    top1_score = float(q[0])
    margin = float(q[0] - q[1]) if K > 1 else 1.0

    return GrammarSnapshot(
        step_label=step_label,
        step_index=step_index,
        H_beam=float(H),
        gt_rank=gt_rank,
        gt_mass=float(gt_mass),
        top1=top1,
        top1_score=top1_score,
        margin=margin,
        n_beam=K,
    )


def compute_probe_metrics(
    predictor,
    probe_queries: List[Tuple[List[str], List[str]]],
) -> Dict[str, float]:
    """Compute probe accuracy and log-likelihood on held-out queries.

    Args:
        predictor: CLSSequencePredictor
        probe_queries: [(words, gt_output), ...]

    Returns:
        dict with probe_acc, probe_ll, probe_n
    """
    if not probe_queries:
        return {'probe_acc': 0.0, 'probe_ll': 0.0, 'probe_n': 0}

    n_correct = 0
    total_ll = 0.0

    for words, gt in probe_queries:
        beam = predictor.beam_posterior(words)
        if not beam:
            continue

        # Top-1 accuracy
        top1 = beam[0][2]
        if top1 == gt:
            n_correct += 1

        # GT log-likelihood
        scores = np.array([b[0] for b in beam])
        log_q = scores - logsumexp(scores)
        q = np.exp(log_q)

        gt_mass = sum(q[k] for k in range(len(beam)) if beam[k][2] == gt)
        total_ll += np.log(max(gt_mass, 1e-30))

    n = len(probe_queries)
    return {
        'probe_acc': n_correct / n if n > 0 else 0.0,
        'probe_ll': total_ll / n if n > 0 else 0.0,
        'probe_n': n,
    }


def snapshot_to_dict(snap: GrammarSnapshot) -> dict:
    """Convert snapshot to serializable dict."""
    return {
        'step_label': snap.step_label,
        'step_index': snap.step_index,
        'H_beam': snap.H_beam,
        'gt_rank': snap.gt_rank,
        'gt_mass': snap.gt_mass,
        'top1': snap.top1,
        'top1_score': snap.top1_score,
        'margin': snap.margin,
        'n_beam': snap.n_beam,
        'probe_acc': snap.probe_acc,
        'probe_ll': snap.probe_ll,
    }
