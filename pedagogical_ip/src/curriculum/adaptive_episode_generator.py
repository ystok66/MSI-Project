"""AEG-v1: Adaptive Episode Generator.

lesson → ψ (episode params) → actual TIC scenario.

The actuator that was missing in CCT-v1:
macro selects lesson → generator produces matching episode → micro acts within it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .lesson_library import Lesson, LESSON_BY_NAME
from ..envs.teaching_internalization_corridor import (
    TICEpisodeSpec, generate_tic_scenario,
)
from ..envs.teaching_internalization_corridor_v4 import SUBTYPE_V4_PARAMS


@dataclass
class EpisodeParams:
    """Realized episode parameters from a lesson."""
    family: str
    subtype: str
    severity: float
    d_commit: int
    d_reveal: int
    lure_strength: float
    risk_level: float
    novelty: float
    advice_mode: str  # none / sparse_valid / sparse_invalid
    dose_budget: float
    hard_limit: int
    target_mechanism: str

    def fidelity_to(self, lesson: Lesson) -> float:
        """LF: how closely this episode matches the intended lesson."""
        d = 0.0
        d += 0.3 * (0 if self.subtype == lesson.subtype else 1)
        d += 0.2 * abs(self.severity - lesson.severity)
        d += 0.2 * abs(self.dose_budget - lesson.dose_profile)
        d += 0.15 * abs(self.novelty - (0.3 if 'novelty' in lesson.subtype else 0.0))
        d += 0.15 * (0 if self.family == lesson.family else 1)
        return round(max(1.0 - d, 0.0), 4)


# ─── Lesson → Subtype mappings ───

LESSON_SUBTYPE_MAP = {
    "ppmrb_standard":       ["boundary_obs", "temptation_repeat"],
    "ppmrb_self_discovery":  ["self_discovery_teach", "self_discovery_needed"],
    "tic_rescue_heavy":      ["warn_rescue"],
    "tic_temptation":        ["temptation_repeat"],
    "tic_self_discovery":    ["self_discovery_needed"],
    "sparse_valid_advice":   ["sparse_valid_advice"],
    "sparse_invalid_advice": ["sparse_invalid_advice"],
    "beneficial_novelty":    ["beneficial_novelty"],
    "verified_warn":         ["verified_warn"],
    "false_suppression":     ["false_suppression_cost"],
    # P3-A: Balanced active coverage
    "warn_symmetric_rescue":    ["warn_rescue"],
    "soft_boundary_tradeoff":   ["soft_gradual"],
    "blind_activation_corridor": ["blind_corridor"],
}

LESSON_ADVICE_MAP = {
    "sparse_valid_advice":   "sparse_valid",
    "sparse_invalid_advice": "sparse_invalid",
}

LESSON_MECHANISM_MAP = {
    "ppmrb_standard":       "repair_RC",
    "ppmrb_self_discovery":  "repair_EP",
    "tic_rescue_heavy":      "repair_RC",
    "tic_temptation":        "repair_TR",
    "tic_self_discovery":    "repair_EP",
    "sparse_valid_advice":   "repair_VA",
    "sparse_invalid_advice": "repair_IA",
    "beneficial_novelty":    "repair_EP",
    "verified_warn":         "repair_VA",
    "false_suppression":     "repair_EP",
    # P3-A: Balanced active coverage
    "warn_symmetric_rescue":    "repair_RC",
    "soft_boundary_tradeoff":   "repair_TR",
    "blind_activation_corridor": "repair_RC",
}


def generate_episode_from_lesson(
    lesson: Lesson, episode_idx: int, theta_true: str,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """lesson → EpisodeParams → TICEpisodeSpec → (gm, cfg, meta, sc).

    Returns: (episode_params, tic_spec, gm, cfg, meta, sc)
    """
    if rng is None:
        rng = np.random.default_rng(episode_idx)

    subtypes = LESSON_SUBTYPE_MAP.get(lesson.name, ["boundary_obs"])
    subtype = rng.choice(subtypes)
    sp = SUBTYPE_V4_PARAMS.get(subtype, SUBTYPE_V4_PARAMS.get("boundary_obs"))

    severity = lesson.severity
    dc = int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1))
    dr = int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1))
    lure = round(float(rng.uniform(
        sp["lure"][0] + 0.1 * severity,
        min(sp["lure"][1] + 0.1 * severity, 1.0))), 3)
    risk = round(float(rng.uniform(
        sp["risk"][0] + 0.05 * severity,
        min(sp["risk"][1] + 0.05 * severity, 1.0))), 3)
    novelty = 0.3 if "novelty" in subtype else 0.0
    advice = LESSON_ADVICE_MAP.get(lesson.name, "none")
    mechanism = LESSON_MECHANISM_MAP.get(lesson.name, "repair_RC")

    ep_params = EpisodeParams(
        family=lesson.family, subtype=subtype, severity=severity,
        d_commit=dc, d_reveal=dr, lure_strength=lure, risk_level=risk,
        novelty=novelty, advice_mode=advice,
        dose_budget=lesson.dose_profile, hard_limit=lesson.hint_budget,
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


def generate_transfer_episode(phase: str, episode_idx: int, theta_true: str,
                              rng=None):
    """Generate transfer phase episodes (B/C/D/E) — not controlled by macro."""
    if rng is None:
        rng = np.random.default_rng(episode_idx)

    if phase == "B":
        subtype = rng.choice(["temptation_repeat", "self_discovery_needed",
                              "false_suppression_cost", "beneficial_novelty"])
    elif phase == "C":
        subtype = "sparse_valid_advice"
    elif phase == "D":
        subtype = "sparse_invalid_advice"
    elif phase == "E":
        subtype = "beneficial_novelty"
    else:
        subtype = "boundary_obs"

    sp = SUBTYPE_V4_PARAMS.get(subtype, SUBTYPE_V4_PARAMS["boundary_obs"])
    spec = TICEpisodeSpec(
        episode_idx=episode_idx, theta_true=theta_true, subtype=subtype,
        phase=phase, mirror=int(rng.integers(0, 2)),
        d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
        d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
        lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
        risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
        cue_layout_seed=int(rng.integers(0, 100000)),
    )
    gm, cfg, meta, sc = generate_tic_scenario(spec)
    sc.is_beneficial_novelty = (subtype == "beneficial_novelty")
    sc.is_sparse_valid = (subtype == "sparse_valid_advice")
    sc.is_sparse_invalid = (subtype == "sparse_invalid_advice")
    sc.risky_branch_actually_good = (subtype in ("false_suppression_cost", "beneficial_novelty"))
    sc.advice_valid = (subtype == "sparse_valid_advice")
    sc.advice_invalid = (subtype == "sparse_invalid_advice")

    advice = "sparse_valid" if phase == "C" else ("sparse_invalid" if phase == "D" else "none")
    ep_params = EpisodeParams(
        family="transfer", subtype=subtype, severity=0.5,
        d_commit=spec.d_commit, d_reveal=spec.d_reveal,
        lure_strength=spec.lure_strength, risk_level=spec.risk_level,
        novelty=0.3 if subtype == "beneficial_novelty" else 0.0,
        advice_mode=advice, dose_budget=0.0, hard_limit=0,
        target_mechanism="transfer",
    )
    return ep_params, spec, gm, cfg, meta, sc
