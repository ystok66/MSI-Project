"""Dose budget tracker — extracted from v2 controller for canonical use."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DoseBudgetTracker:
    """Track and enforce dose budget per episode."""
    budget: float = 1.0
    hard_limit: int = 3
    warns_used: int = 0
    dose_spent: float = 0.0

    def reset(self, ep_params):
        self.budget = ep_params.dose_budget
        self.hard_limit = ep_params.hard_limit
        self.warns_used = 0
        self.dose_spent = 0.0

    def feasible_doses(self) -> list:
        doses = [0.0]
        if self.budget - self.dose_spent >= 0.5:
            doses.append(0.5)
        if self.budget - self.dose_spent >= 1.0 and self.warns_used < self.hard_limit:
            doses.append(1.0)
        return doses

    def consume(self, dose: float):
        self.dose_spent += dose
        if dose >= 1.0:
            self.warns_used += 1
