"""
metrics.py — Compute teach/eval metrics from query results.

All metrics from Phase 1, Section 9 of the implementation guide.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

from ..interfaces import QueryResult
from ..constants import Outcome


def compute_metrics(results: List[QueryResult], prefix: str = '') -> Dict[str, float]:
    """Compute all standard metrics from a list of query results.

    Args:
        results: list of QueryResult
        prefix: 'Teach' or 'Eval' prefix for metric names

    Returns:
        Dict of metric_name → value
    """
    if not results:
        return {}

    n = len(results)
    n_success = sum(1 for r in results if r.outcome == Outcome.SUCCESS)
    n_death = sum(1 for r in results if r.outcome == Outcome.DEATH)
    n_timeout = sum(1 for r in results if r.outcome == Outcome.TIMEOUT)

    # Confirm counts for successful queries only
    success_confirms = [r.confirm_count for r in results if r.outcome == Outcome.SUCCESS]
    mean_confirm_success = float(np.mean(success_confirms)) if success_confirms else 0.0

    # Retry counts
    mean_retry = float(np.mean([r.retry_count for r in results]))

    # Danger select count (useful for immortal baselines)
    total_danger_select = sum(r.danger_select_count for r in results)

    # Stuck retry
    n_stuck = sum(1 for r in results if r.stuck_retry_events > 0)

    metrics = {
        f'{prefix}SuccessRate': n_success / n,
        f'{prefix}DeathRate': n_death / n,
        f'{prefix}TimeoutRate': n_timeout / n,
        f'{prefix}ConfirmMean@Success': mean_confirm_success,
        f'{prefix}RetryMean': mean_retry,
        f'{prefix}DangerSelectCount': total_danger_select / n,
        f'{prefix}StuckRetryRate': n_stuck / n,
        f'{prefix}N': n,
    }
    return metrics


def compute_diagnostic_metrics(
    results: List[QueryResult],
    risk_calibration: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute intermediate diagnostic metrics.

    These are for understanding learner internals, not for final reporting.
    """
    diag = {}

    if risk_calibration:
        diag['RiskPosteriorCalibration'] = risk_calibration.get('calibration', 0.0)
        diag['DangerTypeTop1Acc'] = risk_calibration.get('top1_acc', 0.0)

    return diag


# Convenience type alias
from typing import Optional
