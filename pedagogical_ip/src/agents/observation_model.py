"""
Feature observation model for lattice_v2.

Canonical observation functions:
  observe_features()       — radius-1 feature observation (self + neighbors)
  observe_features_patch() — configurable radius with 3-tier noise model

These functions observe 4D feature vectors from cell_features, not
cost/risk scalars. The runner dispatches to observe_features_patch()
when patch_radius > 1, otherwise observe_features().

Note: The V0 observation interface (generate_observations / Observation
dataclass) was removed in Batch D — zero callers, fully superseded.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ── Feature observation for lattice_v2 ──────────────────────────

@dataclass
class FeatureObservation:
    """Noisy feature observations for a set of cells."""
    positions: list[tuple[int, int]]
    feature_obs: list[np.ndarray]   # noisy d-dim feature per cell
    feature_var: list[float]        # observation noise variance


def observe_features(
    agent_pos: tuple[int, int],
    true_features: np.ndarray,    # (H, W, d)
    cell_types: np.ndarray,       # (H, W)
    self_noise_var: float = 0.01,
    neighbor_noise_var: float = 0.08,
    neighbor_radius: int = 1,
    rng: np.random.Generator | None = None,
) -> FeatureObservation:
    """
    Generate noisy feature observations for lattice_v2.

    Agent observes feature vectors (NOT risk scalars):
      - Self:     σ² = 0.01 (near-exact)
      - Neighbor: σ² = 0.08 (informative but blurry)

    Walls return None features (not included in output).
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W, d = true_features.shape
    r0, c0 = agent_pos

    from ..envs.map_generator import CellType

    positions = []
    feature_obs = []
    feature_vars = []

    for dr in range(-neighbor_radius, neighbor_radius + 1):
        for dc in range(-neighbor_radius, neighbor_radius + 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                continue
            if cell_types[r, c] == CellType.WALL:
                continue

            is_self = (dr == 0 and dc == 0)
            noise_var = self_noise_var if is_self else neighbor_noise_var

            true_f = true_features[r, c]
            noise = rng.normal(0, np.sqrt(noise_var), size=d)
            obs_f = np.clip(true_f + noise, 0.0, 1.0)

            positions.append((r, c))
            feature_obs.append(obs_f)
            feature_vars.append(noise_var)

    return FeatureObservation(
        positions=positions,
        feature_obs=feature_obs,
        feature_var=feature_vars,
    )


def observe_features_patch(
    agent_pos: tuple[int, int],
    true_features: np.ndarray,    # (H, W, d)
    cell_types: np.ndarray,       # (H, W)
    patch_radius: int = 1,
    self_noise_var: float = 0.01,
    neighbor_noise_var: float = 0.08,
    far_noise_var: float = 0.20,
    rng: np.random.Generator | None = None,
) -> FeatureObservation:
    """Generate noisy feature observations over a configurable local patch.

    Phase 5: patch-based multi-cell observation.

    Noise model (discrete 3-tier, no continuous decay):
      - distance 0 (self):      σ² = self_noise_var
      - distance 1 (neighbor):  σ² = neighbor_noise_var
      - distance 2+ (far):      σ² = far_noise_var

    IMPORTANT: when patch_radius=1, this delegates to observe_features()
    to guarantee strictly legacy-compatible RNG call order.
    """
    # Legacy-compatible fast path
    if patch_radius <= 1:
        return observe_features(
            agent_pos, true_features, cell_types,
            self_noise_var=self_noise_var,
            neighbor_noise_var=neighbor_noise_var,
            neighbor_radius=max(patch_radius, 1),
            rng=rng,
        )

    if rng is None:
        rng = np.random.default_rng()

    from ..envs.map_generator import CellType

    H, W, d = true_features.shape
    r0, c0 = agent_pos

    positions = []
    feature_obs = []
    feature_vars = []

    for dr in range(-patch_radius, patch_radius + 1):
        for dc in range(-patch_radius, patch_radius + 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                continue
            if cell_types[r, c] == CellType.WALL:
                continue

            dist = abs(dr) + abs(dc)
            if dist == 0:
                noise_var = self_noise_var
            elif dist == 1:
                noise_var = neighbor_noise_var
            else:
                noise_var = far_noise_var

            true_f = true_features[r, c]
            noise = rng.normal(0, np.sqrt(noise_var), size=d)
            obs_f = np.clip(true_f + noise, 0.0, 1.0)

            positions.append((r, c))
            feature_obs.append(obs_f)
            feature_vars.append(noise_var)

    return FeatureObservation(
        positions=positions,
        feature_obs=feature_obs,
        feature_var=feature_vars,
    )
