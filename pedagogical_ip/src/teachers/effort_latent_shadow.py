"""Effort Latent Shadow: lightweight latent effort state.

Replaces heuristic DepCost = λ_blind · p_blind with a causal
effort-based model:

  e_t ∈ [0, 1]  — learner's current autonomous effort level
  ν_t ≈ 1 - e_t — dependence is inverse of effort

Effort dynamics:
  e_{t+1} = clip(
    e_t
    + η_sd   · 1[self-discovery success]
    - η_dir  · 1[directed WARN]
    - η_bo   · 1[blind obey]
  , 0, 1)

EffortLoss(a) = E[(e_t - e_{t+1}^(a))_+]
  = expected drop in effort caused by action a.

Shadow-only. Does NOT modify any frozen module or state.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EffortLatentShadow:
    """Lightweight latent effort state for EffortLoss computation.

    Tracks an internal e_t that represents learner's autonomous effort.
    Used ONLY for shadow scoring — does NOT write to FactoredInternalizationState.
    """
    # Current effort level
    effort: float = 0.5

    # Effort dynamics rates
    eta_sd: float = 0.08      # self-discovery boosts effort
    eta_dir: float = 0.05     # directed WARN reduces effort
    eta_bo: float = 0.10      # blind obedience (no evidence) strongly reduces effort
    eta_decay: float = 0.01   # natural decay toward baseline

    # Baseline effort (equilibrium without interventions)
    effort_baseline: float = 0.4

    def predict_effort(
        self,
        dose: float,
        p_self: float,
        has_self_ev: bool,
        is_self_discovery: bool = False,
    ) -> tuple[float, dict]:
        """Predict next effort e_{t+1} under action a.

        Returns: (e_next, info_dict)
        """
        e = self.effort

        if dose == 0:
            # WAIT: effort can increase if self-discovery happens,
            # otherwise slight decay toward baseline
            if is_self_discovery or p_self > 0.6:
                # Self-discovery path: high probability of effort gain
                e_next = e + self.eta_sd * p_self
            else:
                # No clear self-discovery: slight regression to baseline
                e_next = e + self.eta_decay * (self.effort_baseline - e)
        else:
            # WARN: effort drops
            if has_self_ev:
                # Learner had own evidence — WARN is less effort-damaging
                # (learner was already engaging)
                e_next = e - self.eta_dir * 0.5
            else:
                # No self-evidence: blind obedience risk
                e_next = e - self.eta_bo

        e_next = float(np.clip(e_next, 0.0, 1.0))

        return e_next, {
            "effort_now": round(self.effort, 4),
            "effort_next": round(e_next, 4),
            "effort_delta": round(e_next - self.effort, 4),
            "dose": dose,
        }

    def compute_effort_loss(
        self,
        dose: float,
        p_self: float,
        has_self_ev: bool,
    ) -> tuple[float, dict]:
        """Compute EffortLoss(a) = max(e_t - e_{t+1}^(a), 0).

        Only counts effort DROPS, not gains.
        """
        e_next, info = self.predict_effort(dose, p_self, has_self_ev)
        loss = max(self.effort - e_next, 0.0)

        info["effort_loss"] = round(float(loss), 4)
        return loss, info

    def update(
        self,
        dose: float,
        p_self: float,
        has_self_ev: bool,
        self_discovery: bool = False,
    ):
        """Update effort state after observing outcome.

        Call this AFTER the episode step to evolve e_t → e_{t+1}.
        """
        e = self.effort

        if dose == 0:
            if self_discovery:
                e += self.eta_sd
            else:
                e += self.eta_decay * (self.effort_baseline - e)
        else:
            if has_self_ev:
                e -= self.eta_dir * 0.5
            else:
                e -= self.eta_bo

        self.effort = float(np.clip(e, 0.0, 1.0))

    def reset(self):
        self.effort = 0.5
