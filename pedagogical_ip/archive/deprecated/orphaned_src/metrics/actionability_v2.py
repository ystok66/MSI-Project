"""Closed-Loop Actionability Metrics: LF, ERCR, micro_PCR."""

from __future__ import annotations
import numpy as np


def lesson_fidelity(ep_params, lesson) -> float:
    """LF: how closely the generated episode matches intent."""
    return ep_params.fidelity_to(lesson)


def episode_realization_change_rate(params_a: list, params_b: list) -> float:
    """ERCR: fraction of episodes where realized subtypes differ."""
    n = min(len(params_a), len(params_b))
    if n == 0:
        return 0.0
    changes = sum(1 for a, b in zip(params_a[:n], params_b[:n])
                  if a.subtype != b.subtype)
    return round(changes / n, 4)


def micro_policy_change_rate(decisions_a: list, decisions_b: list) -> float:
    """micro_PCR: fraction of steps where micro-tutor decides differently."""
    n = min(len(decisions_a), len(decisions_b))
    if n == 0:
        return 0.0
    changes = sum(1 for a, b in zip(decisions_a[:n], decisions_b[:n]) if a != b)
    return round(changes / n, 4)
