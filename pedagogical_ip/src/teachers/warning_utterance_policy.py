"""Warning Utterance Policy — Shadow-mode subtype selection.

When option = WARN, selects utterance subtype:
  π_utt(x_t | WARN) → u_t ∈ {hint, alert, explain, directive}

Shadow-only: logs subtype selection, does NOT affect canonical decisions.

Selection rules:
  hint:      p_self > 0.5, p_blind < 0.3 → learner can still discover
  alert:     p_blind > 0.5, time pressure moderate → interrupt wrong path
  explain:   p_blind < 0.5, time ample, transfer goal → teach reasoning
  directive: p_timeout > 0.7 OR catastrophe → direct instruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


WARN_SUBTYPES = ("hint", "alert", "explain", "directive")


@dataclass
class UtteranceDecision:
    """Result of warning subtype selection."""
    subtype: str               # from WARN_SUBTYPES
    scores: Dict[str, float]   # subtype → score
    reason: str = ""


class WarningUtterancePolicy:
    """Shadow-mode warning utterance subtype selector.

    Selects from {hint, alert, explain, directive} based on:
    - p_self: self-discovery probability
    - p_blind: blind-commit risk
    - p_timeout: timeout risk
    - time_remaining: fraction of budget remaining
    - m_hat: learner state estimate
    """

    def __init__(self, hint_bias: float = 0.3,
                 alert_threshold: float = 0.5,
                 directive_threshold: float = 0.7):
        self.hint_bias = hint_bias
        self.alert_threshold = alert_threshold
        self.directive_threshold = directive_threshold
        self._history: List[UtteranceDecision] = []

    def select_subtype(self,
                       p_self: float = 0.0,
                       p_blind: float = 0.0,
                       p_timeout: float = 0.0,
                       time_remaining: float = 1.0,
                       m_hat: Optional[dict] = None,
                       scenario_family: str = "",
                       ) -> UtteranceDecision:
        """Select warning utterance subtype.

        Args:
            p_self: probability learner discovers on own
            p_blind: blind-commit risk
            p_timeout: timeout risk
            time_remaining: fraction of episode budget remaining (0-1)
            m_hat: learner state estimate
            scenario_family: for logging
        """
        nu_hat = (m_hat or {}).get("nu", 0.1)

        scores = {}

        # hint: learner can still discover
        scores["hint"] = (
            self.hint_bias
            + 1.0 * p_self
            - 0.5 * p_blind
            - 0.3 * p_timeout
            + 0.3 * time_remaining
            - 0.2 * nu_hat  # high dependence → less hint, more direct
        )

        # alert: urgent interruption needed
        scores["alert"] = (
            0.5 * p_blind
            + 0.3 * p_timeout
            - 0.2 * p_self
            + 0.1 * (1.0 - time_remaining)
        )

        # explain: teach reasoning when time allows
        scores["explain"] = (
            0.3 * time_remaining
            + 0.2 * (1.0 - p_blind)
            + 0.2 * (1.0 - p_timeout)
            - 0.1 * nu_hat
        )

        # directive: emergency direct instruction
        scores["directive"] = (
            0.8 * max(p_timeout - self.directive_threshold, 0.0)
            + 0.5 * max(p_blind - self.alert_threshold, 0.0)
            - 0.3 * time_remaining
            - 0.2 * p_self
        )

        chosen = max(scores, key=scores.get)

        # Determine reason
        if chosen == "hint":
            reason = "learner_can_discover"
        elif chosen == "alert":
            reason = "blind_commit_risk"
        elif chosen == "explain":
            reason = "time_for_teaching"
        else:
            reason = "emergency_directive"

        decision = UtteranceDecision(
            subtype=chosen,
            scores=scores,
            reason=reason,
        )
        self._history.append(decision)
        return decision

    def get_report(self) -> Dict:
        """Aggregate statistics over warning subtype selections."""
        if not self._history:
            return {"n_calls": 0}

        n = len(self._history)
        counts = {s: 0 for s in WARN_SUBTYPES}
        for d in self._history:
            counts[d.subtype] += 1
        freqs = {s: c / n for s, c in counts.items()}

        return {
            "n_calls": n,
            "counts": counts,
            "frequencies": freqs,
            "dominant": max(counts, key=counts.get),
        }

    def reset(self):
        self._history = []
