"""AEG-v2: Mastery-conditioned adaptive episode generator.

lesson + learner state → episode params → scenario.
Adjusts severity/novelty/advice based on current mastery.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .lesson_library_v2 import LessonV2, LESSON_V2_BY_NAME, PROBE_NAMES
from .adaptive_episode_generator import (
    EpisodeParams, LESSON_SUBTYPE_MAP, LESSON_ADVICE_MAP,
    LESSON_MECHANISM_MAP,
)
from ..envs.teaching_internalization_corridor import (
    TICEpisodeSpec, generate_tic_scenario,
)
from ..envs.teaching_internalization_corridor_v4 import SUBTYPE_V4_PARAMS


def generate_episode_from_lesson_v2(
    lesson: LessonV2, episode_idx: int, theta_true: str,
    mastery: dict = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Mastery-conditioned: lesson + mastery → episode.

    Adjustments based on mastery:
    - Low EP mastery → reduce novelty intensity
    - Low VA mastery → more low-self-evidence advice
    - High IA mastery → can use stronger conflicting advice
    """
    if rng is None:
        rng = np.random.default_rng(episode_idx)
    if mastery is None:
        mastery = {p: 0.5 for p in PROBE_NAMES}

    subtypes = LESSON_SUBTYPE_MAP.get(lesson.name, ["boundary_obs"])
    subtype = rng.choice(subtypes)
    sp = SUBTYPE_V4_PARAMS.get(subtype, SUBTYPE_V4_PARAMS.get("boundary_obs"))

    # ─── Mastery-conditioned adjustments ───
    ep_mastery = mastery.get("EP", 0.5)
    va_mastery = mastery.get("VA", 0.5)
    ia_mastery = mastery.get("IA", 0.5)

    severity = lesson.severity
    # If EP mastery is low, start with gentler beneficial_novelty
    if subtype == "beneficial_novelty" and ep_mastery < 0.4:
        severity = max(severity - 0.15, 0.2)
    # If VA mastery is low, increase self-evidence for advice lessons
    if subtype in ("sparse_valid_advice", "verified_warn") and va_mastery < 0.4:
        severity = max(severity - 0.1, 0.2)
    # If IA mastery is high, can use stronger conflicting
    if subtype == "sparse_invalid_advice" and ia_mastery > 0.6:
        severity = min(severity + 0.1, 0.9)

    dc = int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1))
    dr = int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1))

    # Mastery-conditioned reveal: low mastery → more reveal
    if ep_mastery < 0.35:
        dr = max(dr - 1, sp["d_r"][0])

    lure = round(float(rng.uniform(
        sp["lure"][0] + 0.1 * severity,
        min(sp["lure"][1] + 0.1 * severity, 1.0))), 3)
    risk = round(float(rng.uniform(
        sp["risk"][0] + 0.05 * severity,
        min(sp["risk"][1] + 0.05 * severity, 1.0))), 3)
    novelty = 0.3 if "novelty" in subtype else 0.0
    advice = LESSON_ADVICE_MAP.get(lesson.name, "none")
    mechanism = LESSON_MECHANISM_MAP.get(lesson.name, "repair_RC")

    # Dose budget: mastery-conditioned
    dose_budget = lesson.dose_profile
    # If ν-related mastery is healthy, allow more dose; if at risk, reduce
    if ia_mastery < 0.4:  # IA still high (bad) → reduce dose
        dose_budget = min(dose_budget, 0.5)

    ep_params = EpisodeParams(
        family=lesson.family, subtype=subtype, severity=severity,
        d_commit=dc, d_reveal=dr, lure_strength=lure, risk_level=risk,
        novelty=novelty, advice_mode=advice,
        dose_budget=dose_budget, hard_limit=lesson.hint_budget,
        target_mechanism=mechanism,
    )

    spec = TICEpisodeSpec(
        episode_idx=episode_idx, theta_true=theta_true, subtype=subtype,
        phase="A", mirror=int(rng.integers(0, 2)),
        d_commit=dc, d_reveal=dr, lure_strength=lure, risk_level=risk,
        cue_layout_seed=int(rng.integers(0, 100000)),
    )

    gm, cfg, meta, sc = generate_tic_scenario(spec)
    sc.is_verified_warn = (subtype == "verified_warn")
    sc.is_self_discovery_needed = (subtype == "self_discovery_needed")
    sc.is_false_suppression = (subtype == "false_suppression_cost")
    sc.is_sparse_valid = (subtype == "sparse_valid_advice")
    sc.is_sparse_invalid = (subtype == "sparse_invalid_advice")
    sc.is_beneficial_novelty = (subtype == "beneficial_novelty")
    sc.risky_branch_actually_good = (subtype in ("false_suppression_cost", "beneficial_novelty"))
    sc.advice_valid = (subtype == "sparse_valid_advice")
    sc.advice_invalid = (subtype == "sparse_invalid_advice")

    return ep_params, spec, gm, cfg, meta, sc
