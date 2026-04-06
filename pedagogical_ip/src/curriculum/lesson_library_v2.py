"""Lesson Library v2: lessons with prerequisites and mastery gain signatures."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

PROBE_NAMES = ["RC", "TR", "EP", "VA", "IA"]


@dataclass
class LessonV2:
    """Lesson with prerequisite and gain vectors."""
    name: str
    family: str
    subtype: str
    severity: float = 0.5
    dose_profile: float = 0.5
    hint_budget: int = 3
    cost: float = 1.0

    # State delta signature: [Δκ, Δτ, Δν, Δγs, Δγg]
    delta_state: np.ndarray = None

    # Mastery gain signature: how much this lesson improves each dimension
    gain: np.ndarray = None  # [g_RC, g_TR, g_EP, g_VA, g_IA]

    # Prerequisite: minimum mastery needed for this lesson to be effective
    prereq: np.ndarray = None  # [p_RC, p_TR, p_EP, p_VA, p_IA]

    # ZPD target: ideal mastery level for this lesson to be maximally effective
    # Semantically: "what mastery should the student have for this lesson"
    # NOT the same as gain (which is what the lesson produces).
    zpd_target: np.ndarray = None  # [d_RC, d_TR, d_EP, d_VA, d_IA]

    # ZPD mask: which dimensions this lesson actually affects (0 or 1)
    # Prevents penalizing mismatch on irrelevant dimensions
    zpd_mask: np.ndarray = None  # [m_RC, m_TR, m_EP, m_VA, m_IA]

    # Overteach risk: how much ν and γg typically increase
    nu_push: float = 0.0
    gg_push: float = 0.0

    def __post_init__(self):
        if self.delta_state is None:
            self.delta_state = np.zeros(5)
        if self.gain is None:
            self.gain = np.zeros(5)
        if self.prereq is None:
            self.prereq = np.zeros(5)
        # Auto-derive ZPD target and mask if not explicitly set
        if self.zpd_target is None:
            # ZPD target = prereq + half of gain range
            # Represents "student should be past prereq but not yet mastered"
            self.zpd_target = np.clip(self.prereq + 0.5 * self.gain, 0.0, 1.0)
        if self.zpd_mask is None:
            # Mask = 1 where gain > threshold (lesson targets this dimension)
            self.zpd_mask = (self.gain > 0.05).astype(float)

    def feasibility(self, mastery: dict, beta=8.0) -> float:
        """σ(β·(u·p - τ)): how feasible is this lesson given current mastery."""
        u = np.array([mastery.get(p, 0.5) for p in PROBE_NAMES])
        prereq_score = float(np.dot(u, self.prereq))
        threshold = float(np.sum(self.prereq)) * 0.6
        x = beta * (prereq_score - threshold)
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    def effective_gain(self, mastery: dict) -> np.ndarray:
        """feas(ℓ|u) · g · (1 - u): diminishing returns on mastered skills."""
        feas = self.feasibility(mastery)
        u = np.array([mastery.get(p, 0.5) for p in PROBE_NAMES])
        return feas * self.gain * (1.0 - u)


LESSON_CATALOG_V2: List[LessonV2] = [
    LessonV2("ppmrb_standard", "PP-MRB", "mixed",
             severity=0.4, dose_profile=0.3, hint_budget=2, cost=0.8,
             delta_state=np.array([0.05, 0.03, -0.02, 0.01, -0.01]),
             gain=np.array([0.15, 0.08, 0.12, 0.10, 0.05]),
             prereq=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
             nu_push=-0.02, gg_push=-0.01),

    LessonV2("ppmrb_self_discovery", "PP-MRB", "self_discovery_teach",
             severity=0.3, dose_profile=0.0, hint_budget=0, cost=0.6,
             delta_state=np.array([0.03, 0.01, -0.04, 0.00, -0.02]),
             gain=np.array([0.05, 0.03, 0.25, 0.08, 0.10]),
             prereq=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),
             nu_push=-0.04, gg_push=-0.02),

    LessonV2("tic_rescue_heavy", "TIC", "warn_rescue",
             severity=0.8, dose_profile=1.0, hint_budget=5, cost=1.5,
             delta_state=np.array([0.08, 0.05, 0.06, 0.02, 0.05]),
             gain=np.array([0.25, 0.10, 0.02, 0.12, 0.03]),
             prereq=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.06, gg_push=0.05),

    LessonV2("tic_temptation", "TIC", "temptation_repeat",
             severity=0.6, dose_profile=0.5, hint_budget=3, cost=1.2,
             delta_state=np.array([0.04, 0.02, 0.03, 0.04, 0.02]),
             gain=np.array([0.08, 0.22, 0.05, 0.05, 0.05]),
             prereq=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.03, gg_push=0.02),

    LessonV2("tic_self_discovery", "TIC", "self_discovery_needed",
             severity=0.5, dose_profile=0.0, hint_budget=0, cost=1.0,
             delta_state=np.array([0.04, 0.02, -0.03, 0.01, -0.03]),
             gain=np.array([0.08, 0.05, 0.20, 0.10, 0.08]),
             prereq=np.array([0.25, 0.0, 0.0, 0.0, 0.0]),
             nu_push=-0.03, gg_push=-0.03),

    LessonV2("sparse_valid_advice", "TIC-v4", "sparse_valid_advice",
             severity=0.4, dose_profile=0.3, hint_budget=2, cost=0.9,
             delta_state=np.array([0.02, 0.05, 0.01, 0.00, 0.00]),
             gain=np.array([0.05, 0.03, 0.03, 0.25, 0.05]),
             prereq=np.array([0.2, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.01, gg_push=0.00),

    LessonV2("sparse_invalid_advice", "TIC-v4", "sparse_invalid_advice",
             severity=0.5, dose_profile=0.0, hint_budget=0, cost=1.0,
             delta_state=np.array([0.01, -0.01, -0.03, 0.00, 0.00]),
             gain=np.array([0.05, 0.03, 0.05, 0.03, 0.25]),
             prereq=np.array([0.2, 0.0, 0.0, 0.3, 0.0]),
             nu_push=-0.03, gg_push=0.00),

    LessonV2("beneficial_novelty", "TIC-v4", "beneficial_novelty",
             severity=0.4, dose_profile=0.0, hint_budget=0, cost=0.8,
             delta_state=np.array([0.02, 0.01, -0.02, -0.01, -0.04]),
             gain=np.array([0.05, 0.03, 0.28, 0.05, 0.08]),
             prereq=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),
             nu_push=-0.02, gg_push=-0.04),

    LessonV2("verified_warn", "TIC-v4", "verified_warn",
             severity=0.5, dose_profile=0.5, hint_budget=2, cost=1.0,
             delta_state=np.array([0.03, 0.04, 0.01, 0.01, 0.01]),
             gain=np.array([0.08, 0.05, 0.03, 0.20, 0.05]),
             prereq=np.array([0.2, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.01, gg_push=0.01),

    LessonV2("false_suppression", "TIC-v4", "false_suppression_cost",
             severity=0.5, dose_profile=0.0, hint_budget=0, cost=0.9,
             delta_state=np.array([0.02, 0.01, -0.02, -0.02, -0.03]),
             gain=np.array([0.05, 0.02, 0.22, 0.05, 0.08]),
             prereq=np.array([0.3, 0.0, 0.15, 0.0, 0.0]),
             nu_push=-0.02, gg_push=-0.03),

    # ─── P3-A: Balanced Active Coverage Suite ────────────
    # These families are designed to produce nontrivial WARN/SOFT/blind
    # events under natural tutor policy, breaking tic_rescue_heavy monopoly.

    LessonV2("warn_symmetric_rescue", "ACTIVE", "warn_rescue",
             severity=0.85, dose_profile=0.9, hint_budget=4, cost=1.4,
             delta_state=np.array([0.07, 0.04, 0.05, 0.01, 0.04]),
             gain=np.array([0.20, 0.15, 0.03, 0.08, 0.05]),
             prereq=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.05, gg_push=0.04),

    LessonV2("soft_boundary_tradeoff", "ACTIVE", "soft_gradual",
             severity=0.6, dose_profile=0.6, hint_budget=3, cost=1.1,
             delta_state=np.array([0.04, 0.03, 0.02, 0.02, 0.02]),
             gain=np.array([0.12, 0.10, 0.08, 0.10, 0.08]),
             prereq=np.array([0.1, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.02, gg_push=0.02),

    LessonV2("blind_activation_corridor", "ACTIVE", "blind_corridor",
             severity=0.75, dose_profile=0.8, hint_budget=4, cost=1.3,
             delta_state=np.array([0.06, 0.04, 0.04, 0.01, 0.03]),
             gain=np.array([0.15, 0.12, 0.05, 0.10, 0.05]),
             prereq=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
             nu_push=0.04, gg_push=0.03),
]

# Balanced suite: original + new active families
BALANCED_ACTIVE_LESSONS = [l for l in LESSON_CATALOG_V2
                           if l.family == "ACTIVE" or l.name == "tic_rescue_heavy"]

LESSON_V2_BY_NAME = {l.name: l for l in LESSON_CATALOG_V2}

