"""Credit correction: do()-corrected learning credit assignment.

Separates "learner learned" from "tutor did it for the learner".

Core idea:
  ρ_self(a) ∈ [0, 1] = fraction of behavior improvement attributable
  to the learner's own choice, NOT the tutor's direct intervention.

  ρ_self(WAIT)  = 1.0  — any improvement is 100% learner credit
  ρ_self(WARN)  = 1 - λ_credit · p_directed

  p_directed: probability that this WARN directly changed the branch choice
  (i.e., without the WARN, the learner would have chosen differently).

LearnGain_do(a) = max(L_now - L_next, 0) · ρ_self(a)

Shadow-only. Does NOT modify any frozen module.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CreditCorrection:
    """do()-corrected credit assignment for learning gain.

    Implements the causal separation:
      - WAIT: all improvement is learner's own
      - WARN: improvement is discounted by how much the tutor
              directly substituted for learner's own decision
    """
    # Credit discount weight
    lambda_credit: float = 0.6

    # Directed-intervention probability estimation parameters
    # p_directed = f(p_self, has_self_ev, risk, tempt)
    # High when: p_self is low (learner wouldn't have gotten it alone)
    #            AND has_self_ev is False (learner has no own evidence)
    #            AND risk is high (tutor's WARN critically changes outcome)

    def compute_rho_self(
        self,
        dose: float,
        p_self: float,
        p_fail: float,
        p_undecided: float,
        has_self_ev: bool,
        risk: float,
        tempt: float,
    ) -> dict:
        """Compute ρ_self and p_directed for an action.

        Returns: {rho_self, p_directed, credit_discount}
        """
        if dose == 0:
            # WAIT: 100% learner credit
            return {
                "rho_self": 1.0,
                "p_directed": 0.0,
                "credit_discount": 0.0,
            }

        # p_directed: how much did WARN substitute for learner's own choice?
        # Components:
        #   1. Learner couldn't have done it alone: (1 - p_self)
        #   2. Learner had no own evidence: stronger directed effect
        #   3. High risk: tutor's information is more critical
        #   4. Temptation: tutor's warning counteracts lure

        # Base: probability learner would have gone wrong without WARN
        p_wrong_without = 1.0 - p_self

        # Evidence discount: if learner has own evidence, WARN is less "directed"
        # (learner may have arrived at the same conclusion independently)
        evidence_factor = 0.3 if has_self_ev else 0.85

        # Risk amplification: higher risk → WARN is more critical → more directed
        risk_factor = 0.5 + 0.5 * risk

        # Temptation: if tempted, WARN is actively fighting a preference
        tempt_factor = 1.0 + 0.3 * tempt

        p_directed = float(np.clip(
            p_wrong_without * evidence_factor * risk_factor * tempt_factor,
            0.0, 1.0
        ))

        credit_discount = self.lambda_credit * p_directed
        rho_self = max(1.0 - credit_discount, 0.0)

        return {
            "rho_self": round(float(rho_self), 4),
            "p_directed": round(float(p_directed), 4),
            "credit_discount": round(float(credit_discount), 4),
        }

    def corrected_learn_gain(
        self,
        learn_gain_raw: float,
        dose: float,
        p_self: float,
        p_fail: float,
        p_undecided: float,
        has_self_ev: bool,
        risk: float,
        tempt: float,
    ) -> tuple[float, dict]:
        """Apply do()-correction to raw learning gain.

        Returns: (corrected_gain, info_dict)
        """
        rho = self.compute_rho_self(
            dose, p_self, p_fail, p_undecided, has_self_ev, risk, tempt)

        corrected = learn_gain_raw * rho["rho_self"]
        leakage = learn_gain_raw - corrected

        info = {
            **rho,
            "learn_gain_raw": round(float(learn_gain_raw), 4),
            "learn_gain_do": round(float(corrected), 4),
            "leakage": round(float(leakage), 4),
        }

        return corrected, info
