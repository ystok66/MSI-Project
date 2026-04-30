from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


@dataclass(frozen=True)
class LearnerProfile:
    name: str
    risk_weight: float
    unknown_penalty: float
    revisit_penalty: float
    info_bonus: float
    softmax_beta: float


def default_profiles() -> list[LearnerProfile]:
    return [
        LearnerProfile("risk_averse", risk_weight=8.0, unknown_penalty=0.9, revisit_penalty=0.25, info_bonus=0.05, softmax_beta=3.0),
        LearnerProfile("balanced", risk_weight=4.0, unknown_penalty=0.45, revisit_penalty=0.20, info_bonus=0.15, softmax_beta=2.2),
        LearnerProfile("curious_explorer", risk_weight=2.0, unknown_penalty=0.05, revisit_penalty=0.12, info_bonus=0.55, softmax_beta=1.7),
        LearnerProfile("goal_greedy", risk_weight=1.2, unknown_penalty=0.15, revisit_penalty=0.08, info_bonus=0.00, softmax_beta=2.8),
        LearnerProfile("confused_looping", risk_weight=2.5, unknown_penalty=0.35, revisit_penalty=-0.08, info_bonus=0.02, softmax_beta=0.7),
    ]


def uniform_profile_belief(profiles: Iterable[LearnerProfile]) -> dict[str, float]:
    names = [p.name for p in profiles]
    if not names:
        return {}
    v = 1.0 / len(names)
    return {name: v for name in names}


def normalize_belief(raw: Mapping[str, float], floor: float = 1e-9) -> dict[str, float]:
    clipped = {k: max(floor, float(v)) for k, v in raw.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        n = max(1, len(clipped))
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def softmax(values: Mapping[str, float], beta: float = 1.0) -> dict[str, float]:
    if not values:
        return {}
    m = max(values.values())
    exps = {k: math.exp(beta * (v - m)) for k, v in values.items()}
    z = sum(exps.values())
    if z <= 0:
        return {k: 1.0 / len(values) for k in values}
    return {k: v / z for k, v in exps.items()}


def likelihood_from_action_prob(prob: float, epsilon: float = 1e-4) -> float:
    return max(epsilon, min(1.0, float(prob)))
