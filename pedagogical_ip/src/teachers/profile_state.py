"""Persistent Learner Profile — State Definitions.

Data structures for cross-session persistence of learner state.
Design principle: m_T (true) is oracle-only; m̂_T (estimated) is tutor-consumable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy


@dataclass
class SessionSummary:
    """Lightweight session-level summary (no step-by-step transcript).

    Stored per session for profile-aware curriculum decisions.
    """
    n_warn: int = 0
    n_wait: int = 0
    n_steps: int = 0
    subtype_counts: Dict[str, int] = field(default_factory=dict)
    probe_means: Dict[str, float] = field(default_factory=dict)
    warn_rate_by_subtype: Dict[str, float] = field(default_factory=dict)
    transfer_success: float = 0.0
    calibration_error: Dict[str, float] = field(default_factory=dict)

    @property
    def warn_rate(self) -> float:
        total = self.n_warn + self.n_wait
        return self.n_warn / max(total, 1)

    def to_dict(self) -> dict:
        return {
            "n_warn": self.n_warn, "n_wait": self.n_wait,
            "n_steps": self.n_steps,
            "subtype_counts": dict(self.subtype_counts),
            "probe_means": dict(self.probe_means),
            "warn_rate_by_subtype": dict(self.warn_rate_by_subtype),
            "transfer_success": self.transfer_success,
            "calibration_error": dict(self.calibration_error),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionSummary":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ProfileState:
    """Cross-session persistent learner profile.

    Invariants:
      - m_terminal is oracle ground truth (analysis only, never consumed by tutor)
      - m_hat_terminal is observer estimate (tutor-consumable for bootstrap)
      - confidence is per-dimension observer confidence at session end
    """
    learner_id: str = "default"
    session_idx: int = 0
    theta: str = "safe"

    # 5D true terminal state — ORACLE ONLY, not for tutor consumption
    m_terminal: Dict[str, float] = field(default_factory=lambda: {
        "kappa": 1.0, "tau": 0.3, "nu": 0.1,
        "gamma_spec": 0.0, "gamma_gen": 0.0,
    })

    # 5D observer estimate terminal state — tutor-consumable
    m_hat_terminal: Dict[str, float] = field(default_factory=lambda: {
        "tau": 0.3, "nu": 0.1, "gamma_gen": 0.0,
        "gamma_spec": 0.0, "kappa": 0.3,
    })

    # Per-dimension confidence at session end
    confidence: Dict[str, float] = field(default_factory=lambda: {
        "tau": 0.2, "nu": 0.2, "gamma_gen": 0.2,
    })

    # Session summary
    history: SessionSummary = field(default_factory=SessionSummary)

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "session_idx": self.session_idx,
            "theta": self.theta,
            "m_terminal": dict(self.m_terminal),
            "m_hat_terminal": dict(self.m_hat_terminal),
            "confidence": dict(self.confidence),
            "history": self.history.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileState":
        hist = d.pop("history", {})
        ps = cls(**{k: v for k, v in d.items()
                    if k in cls.__dataclass_fields__ and k != "history"})
        ps.history = SessionSummary.from_dict(hist) if isinstance(hist, dict) else hist
        return ps

    def copy(self) -> "ProfileState":
        return copy.deepcopy(self)
