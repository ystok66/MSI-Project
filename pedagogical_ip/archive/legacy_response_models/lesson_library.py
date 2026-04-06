"""Lesson Library: catalog of lesson types with mechanism signatures.

Each lesson has:
  - name, family, subtype
  - severity, dose_profile
  - empirical mechanism signature: E[Δκ, Δτ, Δν, Δγs, Δγg]
  - which behavior probe it primarily targets
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Lesson:
    """One lesson type with its mechanism fingerprint."""
    name: str
    family: str
    subtype: str
    severity: float = 0.5       # 0=gentle, 1=intense
    dose_profile: float = 0.5   # typical warning dose
    hint_budget: int = 3        # max interventions per episode

    # Empirical mechanism signature: expected Δ per state dimension
    # [Δκ, Δτ, Δν, Δγs, Δγg]
    delta_state: np.ndarray = None

    # Which probe this lesson primarily targets
    primary_probe: str = ""
    # Expected probe deltas: [ΔRC, ΔTR, ΔEP, ΔVA, ΔIA]
    delta_probe: np.ndarray = None

    # Cost: teacher effort / episode length
    cost: float = 1.0

    def __post_init__(self):
        if self.delta_state is None:
            self.delta_state = np.zeros(5)
        if self.delta_probe is None:
            self.delta_probe = np.zeros(5)


# ─── Pre-calibrated lesson catalog ───
# Signatures estimated from ICT-v2, BC-v4, and FICA rollouts.
# State order: [Δκ, Δτ, Δν, Δγs, Δγg]
# Probe order: [ΔRC, ΔTR, ΔEP, ΔVA, ΔIA]

LESSON_CATALOG: List[Lesson] = [
    Lesson("ppmrb_standard", "PP-MRB", "mixed",
           severity=0.4, dose_profile=0.3, hint_budget=2,
           delta_state=np.array([0.05, 0.03, -0.02, 0.01, -0.01]),
           delta_probe=np.array([0.02, 0.01, 0.02, 0.02, -0.01]),
           primary_probe="EP", cost=0.8),

    Lesson("ppmrb_self_discovery", "PP-MRB", "self_discovery_teach",
           severity=0.3, dose_profile=0.0, hint_budget=0,
           delta_state=np.array([0.03, 0.01, -0.04, 0.00, -0.02]),
           delta_probe=np.array([0.01, 0.00, 0.04, 0.01, -0.02]),
           primary_probe="EP", cost=0.6),

    Lesson("tic_rescue_heavy", "TIC", "warn_rescue",
           severity=0.8, dose_profile=1.0, hint_budget=5,
           delta_state=np.array([0.08, 0.05, 0.06, 0.02, 0.05]),
           delta_probe=np.array([0.04, 0.02, -0.06, 0.03, 0.04]),
           primary_probe="RC", cost=1.5),

    Lesson("tic_temptation", "TIC", "temptation_repeat",
           severity=0.6, dose_profile=0.5, hint_budget=3,
           delta_state=np.array([0.04, 0.02, 0.03, 0.04, 0.02]),
           delta_probe=np.array([0.02, 0.05, -0.02, 0.01, 0.02]),
           primary_probe="TR", cost=1.2),

    Lesson("tic_self_discovery", "TIC", "self_discovery_needed",
           severity=0.5, dose_profile=0.0, hint_budget=0,
           delta_state=np.array([0.04, 0.02, -0.03, 0.01, -0.03]),
           delta_probe=np.array([0.02, 0.01, 0.05, 0.02, -0.03]),
           primary_probe="EP", cost=1.0),

    Lesson("sparse_valid_advice", "TIC-v4", "sparse_valid_advice",
           severity=0.4, dose_profile=0.3, hint_budget=2,
           delta_state=np.array([0.02, 0.05, 0.01, 0.00, 0.00]),
           delta_probe=np.array([0.01, 0.00, 0.00, 0.06, 0.00]),
           primary_probe="VA", cost=0.9),

    Lesson("sparse_invalid_advice", "TIC-v4", "sparse_invalid_advice",
           severity=0.5, dose_profile=0.0, hint_budget=0,
           delta_state=np.array([0.01, -0.01, -0.03, 0.00, 0.00]),
           delta_probe=np.array([0.01, 0.00, 0.00, -0.01, -0.04]),
           primary_probe="IA", cost=1.0),

    Lesson("beneficial_novelty", "TIC-v4", "beneficial_novelty",
           severity=0.4, dose_profile=0.0, hint_budget=0,
           delta_state=np.array([0.02, 0.01, -0.02, -0.01, -0.04]),
           delta_probe=np.array([0.01, 0.00, 0.06, 0.01, -0.02]),
           primary_probe="EP", cost=0.8),

    Lesson("verified_warn", "TIC-v4", "verified_warn",
           severity=0.5, dose_profile=0.5, hint_budget=2,
           delta_state=np.array([0.03, 0.04, 0.01, 0.01, 0.01]),
           delta_probe=np.array([0.02, 0.01, -0.01, 0.04, 0.01]),
           primary_probe="VA", cost=1.0),

    Lesson("false_suppression", "TIC-v4", "false_suppression_cost",
           severity=0.5, dose_profile=0.0, hint_budget=0,
           delta_state=np.array([0.02, 0.01, -0.02, -0.02, -0.03]),
           delta_probe=np.array([0.01, -0.01, 0.05, 0.01, -0.02]),
           primary_probe="EP", cost=0.9),
]

LESSON_BY_NAME = {l.name: l for l in LESSON_CATALOG}


def get_lesson_names() -> List[str]:
    return [l.name for l in LESSON_CATALOG]
