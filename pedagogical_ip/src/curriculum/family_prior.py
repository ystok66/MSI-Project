"""Family Prior — θ-conditional additive lesson bonus with saturation.

Addresses family coverage imbalance with history-conditioned diminishing returns:

  b_eff(ℓ, q, h) = b_raw(ℓ, q) · exp(-n_fam(ℓ,h) / τ_fam) - λ_rep · log(1 + n_fam(ℓ,h))

Usage:
    fp = FamilyPrior()
    bonus = fp.bonus(lesson, theta, family_counts=cct.family_counts())
    J_adjusted = J + bonus
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


# Canonical defaults from Stage 6.6 sweep:
DEFAULT_FAMILY_PRIORS = {
    "safe": {
        "PP-MRB":  0.00,
        "TIC":     0.00,
        "TIC-v4":  0.10,
    },
    "shiny": {
        "PP-MRB":  0.20,
        "TIC":    -0.30,
        "TIC-v4":  0.25,
    },
}

# Per-θ saturation time constants
DEFAULT_TAU_FAM = {
    "safe": 3.0,    # safe: slower decay (less prone to concentration)
    "shiny": 2.0,   # shiny: faster decay (more prone to PP-MRB concentration)
}

# Per-θ repetition penalty
DEFAULT_LAMBDA_REP = {
    "safe": 0.00,    # safe: no extra penalty needed
    "shiny": 0.03,   # shiny: light penalty to encourage diversification
}


@dataclass
class FamilyPrior:
    """Additive θ-conditional family bonus with history-conditioned saturation.

    b_eff(ℓ, q, h) = b_raw(ℓ, q) · exp(-n_fam / τ_fam) - λ_rep · log(1 + n_fam)
    """
    priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_FAMILY_PRIORS))
    tau_fam: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TAU_FAM))
    lambda_rep: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LAMBDA_REP))
    enabled: bool = True
    use_saturation: bool = True
    use_rep_penalty: bool = False  # canonical: decay only (rep hurts shiny -4pp)

    def bonus(self, lesson, theta: str, posterior_q: Optional[dict] = None,
              family_counts: Optional[dict] = None) -> float:
        """Compute additive family bonus with optional saturation.

        Args:
            lesson: LessonV2 with .family attribute
            theta: learner type string
            posterior_q: optional dict {theta: probability}
            family_counts: dict {family_name: count} from curriculum history
        """
        if not self.enabled:
            return 0.0

        fam = getattr(lesson, "family", None)
        if fam is None:
            return 0.0

        # Raw prior
        if posterior_q is not None:
            b_raw = sum(prob * self.priors.get(th, {}).get(fam, 0.0)
                        for th, prob in posterior_q.items())
        else:
            b_raw = self.priors.get(theta, {}).get(fam, 0.0)

        n_fam = (family_counts or {}).get(fam, 0)

        # Saturation: exp decay on positive bonuses only
        if self.use_saturation and n_fam > 0 and b_raw > 0:
            tau = self.tau_fam.get(theta, 3.0)
            b_raw = b_raw * np.exp(-n_fam / tau)

        # Repetition penalty (always applies, independent of prior sign)
        rep = 0.0
        if self.use_rep_penalty and n_fam > 0:
            lam = self.lambda_rep.get(theta, 0.0)
            rep = lam * np.log(1 + n_fam)

        return float(b_raw - rep)

    def set_prior(self, theta: str, family: str, value: float):
        if theta not in self.priors:
            self.priors[theta] = {}
        self.priors[theta][family] = value

    def summary(self) -> dict:
        return {"enabled": self.enabled, "priors": dict(self.priors),
                "use_saturation": self.use_saturation, "use_rep_penalty": self.use_rep_penalty,
                "tau_fam": dict(self.tau_fam), "lambda_rep": dict(self.lambda_rep)}

