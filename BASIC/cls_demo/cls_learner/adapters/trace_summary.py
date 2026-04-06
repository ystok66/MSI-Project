"""
trace_summary.py — Adapter for extracting TraceSummary from beam traces.

Provides utility functions that convert raw beam search output
into TraceSummary objects for HPC storage and analysis.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from scipy.special import logsumexp

from cls_learner.interfaces import TraceSummary
from ns_learner.ns_concept import ROLES


def traces_to_summary(traces: Optional[list],
                      library: dict = None,
                      priors=None) -> TraceSummary:
    """
    Convert beam search traces to a TraceSummary.

    Args:
        traces: list of (score, trace) from infer_top_k
        library: concept library (for color extraction)
        priors: GlobalPriors (for color scoring params)

    Returns:
        TraceSummary with weighted role distributions.
    """
    per_word_role: Dict[str, str] = {}
    per_word_color: Dict[str, str] = {}
    trace_roles: Dict[str, Dict[str, float]] = {}

    if not traces:
        return TraceSummary(
            per_word_role=per_word_role,
            per_word_color=per_word_color,
            trace_roles=trace_roles,
        )

    scores = np.array([t[0] for t in traces])
    if len(scores) > 1:
        log_w = scores - logsumexp(scores)
        weights = np.exp(log_w)
    else:
        weights = np.array([1.0])

    for (score, trace), w in zip(traces, weights):
        for step in trace:
            word = step.word
            role = step.role
            if word not in trace_roles:
                trace_roles[word] = {r: 0.0 for r in ROLES}
            trace_roles[word][role] += w

    # MAP role per word
    for word, role_dist in trace_roles.items():
        best_role = max(role_dist, key=role_dist.get)
        per_word_role[word] = best_role

        if best_role == 'EMIT' and library and word in library and priors:
            per_word_color[word] = library[word].map_color(
                priors.nig, priors.eps_obj, priors.tau_inc,
                delta=priors.delta)

    return TraceSummary(
        per_word_role=per_word_role,
        per_word_color=per_word_color,
        trace_roles=trace_roles,
        score=float(scores[0]) if len(scores) > 0 else 0.0,
        raw_traces=traces,
    )
