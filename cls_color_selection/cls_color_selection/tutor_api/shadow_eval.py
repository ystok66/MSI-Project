"""
shadow_eval.py — Estimate future eval quality from shadow state.

Two methods:
  1. Probe eval: predict on held-out queries using shadow grammar
  2. Risk calibration: shadow risk accuracy vs true danger labels
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .shadow_snapshot import ShadowLearnerSnapshot
from .shadow_update import shadow_predict_target, shadow_beam_entropy
from .shadow_clone import write_shadow_to_real_risk
from ..interfaces import Example


def probe_eval_accuracy(
    snapshot: ShadowLearnerSnapshot,
    probe_queries: List[Example],
) -> float:
    """Estimate grammar accuracy by predicting on held-out queries.

    Returns fraction of queries where shadow prediction == gold.
    """
    if not probe_queries:
        return 0.0

    n_correct = 0
    n_total = 0

    for query in probe_queries:
        pred = shadow_predict_target(snapshot, query.words)
        if pred is not None:
            n_total += 1
            if pred == query.output:
                n_correct += 1

    return n_correct / max(n_total, 1)


def probe_eval_beam_entropy(
    snapshot: ShadowLearnerSnapshot,
    probe_queries: List[Example],
) -> float:
    """Estimate grammar uncertainty via mean beam entropy on probes."""
    if not probe_queries:
        return 0.0

    entropies = []
    for query in probe_queries:
        h = shadow_beam_entropy(snapshot, query.words)
        entropies.append(h)

    return float(np.mean(entropies)) if entropies else 0.0


def risk_calibration_score(
    snapshot: ShadowLearnerSnapshot,
    test_balls: List[dict],  # [{'observed_vec': x, 'is_danger': bool}, ...]
) -> float:
    """Estimate risk model accuracy on test balls.

    Returns fraction correctly classified (P(safe)>0.5 for safe, P(safe)<0.5 for danger).
    """
    if not test_balls or snapshot.risk is None:
        return 0.5

    risk = write_shadow_to_real_risk(snapshot)
    n_correct = 0

    for ball in test_balls:
        x = np.array(ball['observed_vec'])
        post = risk.single_ball_posterior(x)
        pred_safe = post[0] > 0.5
        actual_safe = not ball['is_danger']
        if pred_safe == actual_safe:
            n_correct += 1

    return n_correct / len(test_balls)


def estimate_shadow_eval_gain(
    snapshot_before: ShadowLearnerSnapshot,
    snapshot_after: ShadowLearnerSnapshot,
    probe_queries: List[Example],
) -> float:
    """Estimate G_eval = Â_eval(after) - Â_eval(before).

    This is the core Phase 3 utility: the counterfactual eval improvement
    predicted by the shadow learner.
    """
    acc_before = probe_eval_accuracy(snapshot_before, probe_queries)
    acc_after = probe_eval_accuracy(snapshot_after, probe_queries)
    return acc_after - acc_before
