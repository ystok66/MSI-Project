"""
query_families.py — Family A/B/C/D query generation.

Implements §14 of the spec. Each family has distinct properties
that exercise different tutor capabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..interfaces import Option
from ..env.danger_model import DangerModel, generate_danger_vector


@dataclass
class FamilySpec:
    """Specification for a query family."""
    name: str
    description: str
    # Generation parameters
    n_near_miss: int = 0            # near-miss distractors (semantic)
    danger_scale: float = 1.0       # multiplier on typical danger
    has_lure: bool = False          # inject one high-damage plausible lure
    semantic_ambiguity: str = "low"  # "low", "medium", "high"


FAMILY_A = FamilySpec(
    name="highlight_critical",
    description="Many semantically plausible near-miss distractors, mild danger. "
                "Tutor should help by highlighting diagnostic cells.",
    n_near_miss=5,
    danger_scale=0.4,
    has_lure=False,
    semantic_ambiguity="high",
)

FAMILY_B = FamilySpec(
    name="ban_critical",
    description="1-2 highly plausible but high-damage lure options. "
                "Tutor should help by banning a dangerous distractor.",
    n_near_miss=1,
    danger_scale=1.5,
    has_lure=True,
    semantic_ambiguity="medium",
)

FAMILY_C = FamilySpec(
    name="skip_clean",
    description="Learner likely already knows this pattern, or learning value "
                "is low. Tutor should mostly WAIT or SKIP.",
    n_near_miss=0,
    danger_scale=0.3,
    has_lure=False,
    semantic_ambiguity="low",
)

FAMILY_D = FamilySpec(
    name="mixed_hard",
    description="Semantic ambiguity + danger lure + time pressure. "
                "Main stress-test family.",
    n_near_miss=3,
    danger_scale=1.2,
    has_lure=True,
    semantic_ambiguity="high",
)

FAMILIES = {
    "A": FAMILY_A,
    "B": FAMILY_B,
    "C": FAMILY_C,
    "D": FAMILY_D,
}


def adjust_menu_for_family(
    menu: List[Option],
    family: FamilySpec,
    danger_model: DangerModel,
    rng: np.random.Generator,
) -> List[Option]:
    """Post-process a menu to match family characteristics.

    Modifies danger vectors to match family-specific danger profiles.
    Does NOT change semantic content (text/rendered output).
    """
    adjusted = []
    for opt in menu:
        new_opt = Option(
            index=opt.index,
            text=opt.text,
            danger_vec=opt.danger_vec.copy(),
            is_correct=opt.is_correct,
            rendered_output=opt.rendered_output,
        )

        if opt.is_correct:
            # Correct option: danger independent of correctness
            pass
        elif family.has_lure and not opt.is_correct and len(adjusted) == 1:
            # First distractor becomes a high-danger lure
            new_opt.danger_vec = _make_high_danger_vec(
                danger_model, rng, target_damage=4.0)
        else:
            # Scale danger
            new_opt.danger_vec = opt.danger_vec * family.danger_scale

        adjusted.append(new_opt)

    return adjusted


def _make_high_danger_vec(
    danger_model: DangerModel,
    rng: np.random.Generator,
    target_damage: float = 4.0,
    max_attempts: int = 100,
) -> np.ndarray:
    """Generate a danger vector that produces high expected damage."""
    m = (danger_model.w_d.shape[0] - 1) // 2
    best_v = None
    best_damage = -1.0

    for _ in range(max_attempts):
        v = generate_danger_vector(m, rng) * 1.5  # wider distribution
        d = danger_model.expected_damage(v)
        if d > best_damage:
            best_damage = d
            best_v = v.copy()
        if d >= target_damage:
            return v

    return best_v if best_v is not None else generate_danger_vector(m, rng)


def get_family_spec(family_name: str) -> FamilySpec:
    """Look up a family spec by name."""
    if family_name in FAMILIES:
        return FAMILIES[family_name]
    raise ValueError(f"Unknown family: {family_name}. "
                     f"Available: {list(FAMILIES.keys())}")
