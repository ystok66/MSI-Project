"""
learning_increment_metrics.py — DeltaProbeSR and related metrics.

Computes the learning increment from pre/post autonomous probe results
plus teach-phase metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

from .autonomous_probe import ProbeResult


@dataclass
class LearningIncrementResult:
    """Full learning-increment benchmark result for one condition run."""
    # Probe
    pre_probe_sr: float
    post_probe_sr: float
    delta_probe_sr: float
    pre_first_ok: float
    post_first_ok: float
    delta_first_ok: float
    # Teach
    teach_sr: float
    # Safety
    damage_sum: float
    death_rate: float
    timeout_rate: float
    # Efficiency
    learning_per_damage: float
    learning_per_intervention: float
    assist_gap: float
    # Counts
    n_interventions: int
    wrong_reveal_count: int
    correct_pick_count: int


def compute_learning_increment(
    pre_probe: ProbeResult,
    post_probe: ProbeResult,
    teach_sr: float,
    damage_sum: float,
    death_rate: float,
    timeout_rate: float,
    n_interventions: int,
    wrong_reveal_count: int,
    correct_pick_count: int,
) -> LearningIncrementResult:
    """Compute full learning-increment metrics from pre/post probes."""
    delta_sr = post_probe.sr - pre_probe.sr
    delta_fok = post_probe.first_ok - pre_probe.first_ok
    assist_gap = teach_sr - post_probe.sr

    lpd = delta_sr / (1.0 + damage_sum)
    lpi = delta_sr / (1.0 + n_interventions)

    return LearningIncrementResult(
        pre_probe_sr=pre_probe.sr,
        post_probe_sr=post_probe.sr,
        delta_probe_sr=delta_sr,
        pre_first_ok=pre_probe.first_ok,
        post_first_ok=post_probe.first_ok,
        delta_first_ok=delta_fok,
        teach_sr=teach_sr,
        damage_sum=damage_sum,
        death_rate=death_rate,
        timeout_rate=timeout_rate,
        learning_per_damage=lpd,
        learning_per_intervention=lpi,
        assist_gap=assist_gap,
        n_interventions=n_interventions,
        wrong_reveal_count=wrong_reveal_count,
        correct_pick_count=correct_pick_count,
    )
