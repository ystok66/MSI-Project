"""
hint_reliance.py — Persistent hint trust / reliance state (Step 3).

Tracks cross-query learner reliance on tutor hints.
When trust is high, hint bias and autonomy shift are amplified.
When trust is low (from unhelpful hints), effects are dampened.

Usage:
    reliance = HintRelianceState()

    # After each hint:
    reliance.record_hint(helpful=True/False)

    # Get effective parameters:
    beta_eff = reliance.effective_beta(beta_base)
    confirm_bonus_eff = reliance.effective_confirm_bonus(bonus_base)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HintRelianceState:
    """Cross-query learner reliance on tutor hints.

    Trust evolves based on whether hints are helpful:
      - helpful hint (leads to success) → trust increases
      - unhelpful hint (leads to timeout/fail) → trust decreases
      - no hint → trust decays toward baseline

    Trust modulates:
      - β_hint_eff = β_hint_base × trust
      - hint_confirm_bonus_eff = hint_confirm_bonus × trust
      - hint_exploration_drop_eff = hint_exploration_drop × trust

    Trust range: [0.0, 1.0], default 0.5 (uninformative prior).
    """
    trust: float = 0.5
    n_hints_seen: int = 0
    n_hints_helpful: int = 0
    n_hints_harmful: int = 0
    n_queries_with_hint: int = 0
    n_queries_without_hint: int = 0

    # Tuning parameters
    lr_up: float = 0.15       # trust increase per helpful hint
    lr_down: float = 0.10     # trust decrease per unhelpful hint
    decay_rate: float = 0.02  # trust decay per no-hint query toward baseline
    baseline: float = 0.5     # resting trust level

    def record_hint(self, helpful: bool):
        """Record a hint event and update trust.

        Args:
            helpful: True if the query succeeded after hint,
                     False if the query timed out / failed after hint
        """
        self.n_hints_seen += 1
        if helpful:
            self.n_hints_helpful += 1
            self.trust = min(1.0, self.trust + self.lr_up * (1.0 - self.trust))
        else:
            self.n_hints_harmful += 1
            self.trust = max(0.0, self.trust - self.lr_down * self.trust)

    def record_query_with_hint(self, success: bool):
        """Record a query that received at least one hint.

        Args:
            success: whether the query ended in success
        """
        self.n_queries_with_hint += 1
        self.record_hint(helpful=success)

    def record_query_no_hint(self):
        """Record a query that received no hints.

        Trust decays toward baseline when not exercised.
        """
        self.n_queries_without_hint += 1
        # Decay toward baseline
        self.trust += self.decay_rate * (self.baseline - self.trust)

    def effective_beta(self, beta_base: float) -> float:
        """Effective hint bias strength modulated by trust."""
        return beta_base * self.trust

    def effective_confirm_bonus(self, bonus_base: float) -> float:
        """Effective confirm threshold reduction modulated by trust."""
        return bonus_base * self.trust

    def effective_exploration_drop(self, drop_base: float) -> float:
        """Effective exploration reduction modulated by trust."""
        return drop_base * self.trust

    def summary(self) -> dict:
        """Return diagnostics dict."""
        return {
            'trust': self.trust,
            'n_hints_seen': self.n_hints_seen,
            'n_hints_helpful': self.n_hints_helpful,
            'n_hints_harmful': self.n_hints_harmful,
            'n_queries_with_hint': self.n_queries_with_hint,
            'n_queries_without_hint': self.n_queries_without_hint,
            'helpful_rate': (
                self.n_hints_helpful / max(self.n_hints_seen, 1)),
        }
