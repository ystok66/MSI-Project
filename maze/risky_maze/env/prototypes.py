from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MazeScenarioConfig


@dataclass
class PrototypeBank:
    means: np.ndarray
    cluster_std: float
    obs_noise: float

    @property
    def n_classes(self) -> int:
        return int(self.means.shape[0])

    def sample_feature(self, class_idx: int, rng: np.random.Generator) -> np.ndarray:
        return self.means[class_idx] + rng.normal(
            0.0,
            self.cluster_std,
            size=self.means.shape[1],
        )

    def observe(self, feature: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return feature + rng.normal(0.0, self.obs_noise, size=feature.shape[0])

    @classmethod
    def random(
        cls,
        cfg: MazeScenarioConfig,
        rng: np.random.Generator,
    ) -> "PrototypeBank":
        means = rng.normal(
            0.0,
            1.5,
            size=(cfg.n_trap_types + 1, cfg.risk_dim),
        )
        means[0] *= 0.6
        return cls(means=means, cluster_std=cfg.cluster_std, obs_noise=cfg.obs_noise)
