"""
PRS Metrics — Family 3 session-level metrics.

Computes TBSR, APD, TransferGap, and DependenceProxy from block results.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_tbsr(results: list[dict]) -> float:
    """Time-Bounded Success Rate = P(goal reached before deadline).

    A success requires both survival AND reaching the goal within t_max.
    """
    if not results:
        return 0.0
    return float(np.mean([r.get("success", False) for r in results]))


def compute_survival(results: list[dict]) -> float:
    """Survival rate = P(survived)."""
    if not results:
        return 0.0
    return float(np.mean([r.get("survived", False) for r in results]))


def compute_apd(trained_results: list[dict],
                baseline_results: list[dict]) -> float:
    """Agent Performance Delta = Perf(trained) - Perf(baseline).

    Positive APD means tutor-on training helped in tutor-off conditions.
    """
    perf_trained = compute_tbsr(trained_results)
    perf_baseline = compute_tbsr(baseline_results)
    return perf_trained - perf_baseline


def compute_transfer_gap(block_b: list[dict], block_x: list[dict]) -> float:
    """TransferGap = Perf_B - Perf_X.

    Measures how much performance drops when distribution shifts.
    Small gap = more generalizable learning.
    """
    return compute_tbsr(block_b) - compute_tbsr(block_x)


def compute_dependence_proxy(block_a: list[dict], block_b: list[dict],
                              k: int = 5) -> float:
    """Tutor dependence proxy: performance drop from last-k Block A to first-k Block B.

    DependenceProxy = Perf(A, last k) - Perf(B, first k)
    High value = agent depends on tutor, performance drops immediately.
    """
    last_a = block_a[-k:] if len(block_a) >= k else block_a
    first_b = block_b[:k] if len(block_b) >= k else block_b

    if not last_a or not first_b:
        return 0.0

    perf_a = compute_tbsr(last_a)
    perf_b = compute_tbsr(first_b)
    return perf_a - perf_b


def compute_boredom_proxy(results: list[dict]) -> float:
    """Boredom proxy: fraction of episodes with low information gain + high cost.

    Proxy: episodes that are very long (steps/t_max > 0.9) but still succeed.
    These suggest the agent is wandering without learning.
    """
    if not results:
        return 0.0
    bored = 0
    for r in results:
        t_max = r.get("t_max", 1)
        steps = r.get("steps", 0)
        if t_max > 0 and steps / t_max > 0.9 and r.get("success", False):
            bored += 1
    return bored / len(results)


def compute_frustration_proxy(results: list[dict]) -> float:
    """Frustration proxy: fraction of episodes with repeated failure.

    Proxy: episodes that die (not survived) with few steps (steps/t_max < 0.3).
    These suggest early, repeated, discouraging failures.
    """
    if not results:
        return 0.0
    frustrated = 0
    for r in results:
        t_max = r.get("t_max", 1)
        steps = r.get("steps", 0)
        if not r.get("survived", True) and t_max > 0 and steps / t_max < 0.3:
            frustrated += 1
    return frustrated / len(results)


def compute_session_metrics(block_results: dict) -> dict:
    """Compute all session-level metrics from block results.

    Args:
        block_results: {"A": [...], "B": [...], "C": [...], "D": [...]}

    Returns:
        Dict with per-block and cross-block metrics.
    """
    a = block_results.get("A", [])
    b = block_results.get("B", [])
    c = block_results.get("C", [])
    d = block_results.get("D", [])

    metrics = {
        # Per-block TBSR
        "tbsr_A": compute_tbsr(a),
        "tbsr_B": compute_tbsr(b),
        "tbsr_C": compute_tbsr(c),
        "tbsr_D": compute_tbsr(d),

        # Per-block survival
        "surv_A": compute_survival(a),
        "surv_B": compute_survival(b),
        "surv_C": compute_survival(c),
        "surv_D": compute_survival(d),

        # Transfer gaps
        "transfer_gap_C": compute_transfer_gap(b, c),
        "transfer_gap_D": compute_transfer_gap(b, d),

        # Dependence proxy (k=5)
        "dependence_proxy": compute_dependence_proxy(a, b, k=5),

        # Boredom / frustration (lightweight)
        "boredom_A": compute_boredom_proxy(a),
        "frustration_A": compute_frustration_proxy(a),
        "frustration_B": compute_frustration_proxy(b),
        "frustration_C": compute_frustration_proxy(c),
        "frustration_D": compute_frustration_proxy(d),

        # Block sizes
        "n_A": len(a), "n_B": len(b), "n_C": len(c), "n_D": len(d),
    }

    # Per-family breakdown within each block
    for block_id, recs in block_results.items():
        for fam_tag in ["dtmb", "gtet"]:
            full_name = ("deep_tree_mixed_bottleneck_lattice" if fam_tag == "dtmb"
                         else "goal_preference_temptation_entanglement_lattice")
            fam_recs = [r for r in recs if r.get("family") == full_name]
            if fam_recs:
                metrics[f"tbsr_{block_id}_{fam_tag}"] = compute_tbsr(fam_recs)
                metrics[f"surv_{block_id}_{fam_tag}"] = compute_survival(fam_recs)

    # Learning curve: predictor updates over time
    all_eps = a + b + c + d
    if all_eps and all_eps[-1].get("predictor_n_updates", 0) > 0:
        metrics["predictor_updates_end_A"] = (
            a[-1].get("predictor_n_updates", 0) if a else 0)
        metrics["predictor_updates_end_session"] = (
            all_eps[-1].get("predictor_n_updates", 0))

    return metrics
