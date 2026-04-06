"""Actionability Metrics: PCR (Policy Change Rate) and AM (Actionability Margin).

PCR = fraction of states where new policy differs from base policy
AM  = variance of ΔQ across candidate actions (how much new mechanism matters)
"""

from __future__ import annotations
import numpy as np


def policy_change_rate(decisions_new: list, decisions_base: list) -> float:
    """PCR: fraction of states where policies disagree."""
    n = min(len(decisions_new), len(decisions_base))
    if n == 0:
        return 0.0
    changes = sum(1 for a, b in zip(decisions_new[:n], decisions_base[:n]) if a != b)
    return round(changes / n, 4)


def actionability_margin(delta_Q_per_action: list) -> float:
    """AM: variance of ΔQ across candidate actions.
    Input: list of [ΔQ_wait, ΔQ_soft, ΔQ_hard] per timestep.
    """
    if not delta_Q_per_action:
        return 0.0
    vars_per_step = [float(np.var(dq)) for dq in delta_Q_per_action]
    return round(float(np.mean(vars_per_step)), 6)


def curriculum_change_rate(seq_new: list, seq_base: list) -> float:
    """Fraction of lesson slots where curriculum choices differ."""
    n = min(len(seq_new), len(seq_base))
    if n == 0:
        return 0.0
    changes = sum(1 for a, b in zip(seq_new[:n], seq_base[:n]) if a != b)
    return round(changes / n, 4)
