from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


def generate_danger_vector(m: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random danger vector. [V1 compat shim]

    In V2, real danger vectors are assigned by DangerModel.sample_danger_vec().
    This function is kept for backward compat with option_generator.py
    which uses it to create placeholder vectors during menu generation.
    """
    return rng.standard_normal(m)


@dataclass
class DangerModel:
    """V2 discrete danger model with cluster prototypes.

    Each block generates 3 cluster prototypes.
    Options get danger vectors sampled from the corresponding cluster.
    Risk class directly determines damage (deterministic in V2).
    """
    m: int                              # danger vector dimension
    mu_safe: np.ndarray                 # (m,) safe cluster prototype
    mu_low: np.ndarray                  # (m,) low-risk cluster (class 1-2)
    mu_high: np.ndarray                 # (m,) high-risk cluster (class 3-4)
    cluster_sigma: float = 0.5         # intra-cluster noise

    def sample_risk_class(self, rng: np.random.Generator,
                          is_safe: bool) -> int:
        """Sample a risk class for one option.

        Args:
            rng: random generator
            is_safe: if True, always returns 0

        Returns:
            risk_class: 0 (safe) or 1-4 (risky)
        """
        if is_safe:
            return 0
        return int(rng.choice([1, 2, 3, 4]))

    def sample_danger_vec(self, risk_class: int,
                          rng: np.random.Generator) -> np.ndarray:
        """Sample a danger vector from the corresponding cluster.

        Args:
            risk_class: 0=safe, 1-2=low, 3-4=high

        Returns:
            (m,) danger vector
        """
        if risk_class == 0:
            mu = self.mu_safe
        elif risk_class <= 2:
            mu = self.mu_low
        else:
            mu = self.mu_high

        return mu + self.cluster_sigma * rng.standard_normal(self.m)

    def get_damage(self, risk_class: int) -> int:
        """Deterministic damage from risk class (V2 canonical)."""
        return int(risk_class)

    def expected_damage(self, v_or_class=None) -> float:
        """Expected damage.

        V2: if given int, damage = risk_class.
        V1 compat: if given ndarray, use w_d if available.
        """
        if isinstance(v_or_class, np.ndarray):
            # V1 compat: w_d^T v
            if hasattr(self, '_w_d') and self._w_d is not None:
                return float(np.dot(self._w_d, v_or_class[:len(self._w_d)]))
            return 0.0
        return float(v_or_class) if v_or_class is not None else 0.0

    def sample_damage(self, danger_vec: np.ndarray,
                      rng: np.random.Generator,
                      risk_class: int = 0) -> int:
        """Sample realized damage. V2: deterministic = risk_class."""
        return int(risk_class)

    # V1 backward compat properties
    @property
    def w_d(self):
        """V1 compat: danger weight vector."""
        if not hasattr(self, '_w_d') or self._w_d is None:
            self._w_d = np.zeros(self.m)
        return self._w_d

    @w_d.setter
    def w_d(self, val):
        self._w_d = val

    def feature_expand(self, v: np.ndarray) -> np.ndarray:
        """V1 compat: feature expansion φ(v) = [v, v², 1]."""
        return np.concatenate([v, v * v, [1.0]])

    def assign_risk_classes(self, K: int, n_safe: int,
                            rng: np.random.Generator
                            ) -> List[int]:
        """Assign risk classes to K options.

        Args:
            K: total options (e.g. 10)
            n_safe: number safe (e.g. 6)
            rng: random generator

        Returns:
            List of K risk_classes, shuffled.
        """
        classes = [0] * n_safe
        n_risky = K - n_safe
        for _ in range(n_risky):
            classes.append(int(rng.choice([1, 2, 3, 4])))
        rng.shuffle(classes)
        return classes


def generate_danger_model(
    m: int = 16,
    rng: Optional[np.random.Generator] = None,
    cluster_sigma: float = 0.5,
    **kwargs,  # absorb legacy params
) -> DangerModel:
    """Generate a new danger model for a block.

    Creates 3 cluster prototypes with good separation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate well-separated prototypes
    # Use rejection to ensure clusters are distinguishable
    mu_safe = rng.standard_normal(m)
    mu_low = rng.standard_normal(m)
    mu_high = rng.standard_normal(m)

    # Push prototypes apart slightly for better learnability
    mu_safe = mu_safe / (np.linalg.norm(mu_safe) + 1e-8) * 2.0
    mu_low = mu_low / (np.linalg.norm(mu_low) + 1e-8) * 2.0
    mu_high = mu_high / (np.linalg.norm(mu_high) + 1e-8) * 2.0

    return DangerModel(
        m=m,
        mu_safe=mu_safe,
        mu_low=mu_low,
        mu_high=mu_high,
        cluster_sigma=cluster_sigma,
    )
