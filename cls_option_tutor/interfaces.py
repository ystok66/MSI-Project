"""
interfaces.py — Shared dataclasses for the CLS Option Tutor.

All inter-module communication uses these types.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ── Core task data ─────────────────────────────────────────────

@dataclass
class Example:
    """One input→output pair for support or query."""
    words: List[str]
    output: List[str]
    meta: dict = field(default_factory=dict)


@dataclass
class Option:
    """One candidate option in a menu.

    Attributes:
        index: position in current menu (0..K-1)
        text: symbolic utterance / text description
        danger_vec: hidden danger vector v ∈ R^m
        risk_class: discrete risk level (0=safe, 1-4=risky) [V2]
        is_correct: ground-truth flag (hidden from learner)
        rendered_output: F_G(text) — cached rendered result (hidden until reveal)
    """
    index: int
    text: List[str]                     # utterance tokens
    danger_vec: np.ndarray              # (m,) danger vector
    is_correct: bool                    # hidden from learner
    risk_class: int = 0                 # 0=safe, 1-4=risky [V2]
    rendered_output: Optional[List[str]] = None  # F_G(ν), computed lazily


@dataclass
class RevealEvent:
    """Record of a wrong-choice reveal within a query."""
    round_t: int
    option_index: int
    option_text: List[str]
    revealed_output: List[str]          # F_G(ν_chosen)
    damage: int                         # realized damage (clipped integer)
    expected_damage: float              # μ_d(v) — hidden from learner
    danger_vec: np.ndarray              # v — the danger vector
    risk_class: int = 0                 # [V2] discrete risk class


@dataclass
class RiskHintEvent:
    """Record of a RISK_HINT from tutor. [V2]"""
    round_t: int
    option_index: int
    eta: float = 0.8                    # weak label confidence


@dataclass
class LearnerStep:
    """One learner action within a round."""
    round_t: int
    query_id: int
    action: str                         # "pick" or "refresh"
    pick_index: Optional[int] = None    # which option was picked
    correct: Optional[bool] = None
    damage: Optional[int] = None
    hp_before: int = 0
    hp_after: int = 0
    menu_size: int = 0                  # effective menu after bans
    semantic_scores: Optional[List[float]] = None
    danger_preds: Optional[List[float]] = None


@dataclass
class TutorStep:
    """One tutor action within a round."""
    round_t: int
    query_id: int
    action: str                         # "WAIT", "RISK_HINT", "HIGHLIGHT", "SKIP"
    ban_index: Optional[int] = None     # which option was banned (legacy)
    hint_index: Optional[int] = None    # which option got risk hint [V2]
    highlight_cells: Optional[Tuple[int, ...]] = None
    q_scores: Optional[dict] = None


# ── Semantic scorer protocol ───────────────────────────────────

class SemanticScorerProtocol:
    """Predictor-agnostic interface for semantic scoring.

    Any scorer (CLS cortex, CLS cortex+HPC, stub) must implement this.
    """
    def score_option(self, target_output: List[str],
                     option_text: List[str],
                     memory_payload: object = None) -> float:
        """Score how well option_text explains target_output. Higher = better match."""
        raise NotImplementedError

    def predict_output(self, option_text: List[str],
                       memory_payload: object = None) -> List[str]:
        """Predict rendered output for option_text."""
        raise NotImplementedError

    def uncertainty(self, target_output: List[str],
                    option_text: List[str],
                    memory_payload: object = None) -> float:
        """Semantic uncertainty (0 = certain, 1 = max uncertainty)."""
        raise NotImplementedError


# ── Policy state snapshot for inverse planning ─────────────────

@dataclass
class PolicyStateSnapshot:
    """Immutable per-step snapshot of learner policy inputs.

    Records the minimal sufficient state to evaluate
    π_L(action | state, profile) for RSA-style inverse planning.
    Created before each learner action; used by profile inference.
    """
    query_id: int
    step_idx: int                       # step within the query
    target_output: List[str]            # Y*

    # Menu pre-state (text + danger only, no is_correct)
    option_texts: List[List[str]]       # text for each active option
    option_danger_vecs: List[np.ndarray]  # danger vectors
    option_indices: List[int]           # original menu indices

    # Intervention state
    active_bans: List[int]              # banned indices at this step
    active_highlights: Tuple[int, ...]  # highlighted cells

    # Learner internal state
    hp_before: int
    attempt_idx: int                    # = rounds_used
    refresh_count: int
    max_refreshes: int

    # Fields with defaults must come after fields without defaults
    active_risk_hints: List[int] = field(default_factory=list)  # [V2]

    # Hazard/severity head posterior snapshot [V2]
    hazard_posterior_mean: Optional[np.ndarray] = None
    severity_posterior_mean: Optional[np.ndarray] = None
    # Legacy danger head (kept for backward compat)
    danger_posterior_mean: Optional[np.ndarray] = None
    danger_posterior_cov: Optional[np.ndarray] = None

    # Risk classes for active options [V2]
    option_risk_classes: Optional[List[int]] = None

    # Attention weights at time of decision
    attention_weights: Optional[np.ndarray] = None

    # Semantic scores computed at decision time
    semantic_scores: Optional[np.ndarray] = None

    # What the learner actually did
    learner_action: Optional[str] = None       # "pick" or "refresh"
    learner_pick_index: Optional[int] = None   # if pick

