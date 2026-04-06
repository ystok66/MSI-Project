"""
DEPRECATED — V0 observation model. Not used by the V2 canonical runner path.

Observation model: generate noisy cues from the true map.
Agent sees current cell near-exactly and neighbors with Gaussian noise.

The V2 runner uses observe_features() / observe_features_patch() which
observe 4D feature vectors from cell_features, not cost/risk scalars.
This file is kept for backward compatibility and reference only.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    """Noisy observations for a set of cells."""
    positions: list[tuple[int, int]]
    cost_obs: list[float]
    risk_obs: list[float]
    cost_var: list[float]    # observation noise variance
    risk_var: list[float]


def generate_observations(
    agent_pos: tuple[int, int],
    true_cost: np.ndarray,
    true_risk: np.ndarray,
    self_noise_var: float = 0.001,
    neighbor_noise_var: float = 1.0,
    neighbor_radius: int = 1,
    rng: np.random.Generator | None = None,
) -> Observation:
    """
    Generate noisy observations for the agent's current position
    and its neighborhood.

    Current cell: near-exact (very low noise).
    Neighbors: noisy Gaussian around true values.
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = true_cost.shape
    r0, c0 = agent_pos

    positions = []
    cost_obs = []
    risk_obs = []
    cost_vars = []
    risk_vars = []

    for dr in range(-neighbor_radius, neighbor_radius + 1):
        for dc in range(-neighbor_radius, neighbor_radius + 1):
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < H and 0 <= c < W):
                continue

            is_self = (dr == 0 and dc == 0)
            noise_var = self_noise_var if is_self else neighbor_noise_var

            tc = true_cost[r, c]
            tr = true_risk[r, c]

            # For walls (inf cost), observe very high cost
            if np.isinf(tc):
                obs_c = 100.0  # sentinel for "impassable"
            else:
                obs_c = max(0.0, tc + rng.normal(0, np.sqrt(noise_var)))
            obs_r = float(np.clip(tr + rng.normal(0, np.sqrt(noise_var)), 0.0, 1.0))

            positions.append((r, c))
            cost_obs.append(obs_c)
            risk_obs.append(obs_r)
            cost_vars.append(noise_var)
            risk_vars.append(noise_var)

    return Observation(
        positions=positions,
        cost_obs=cost_obs,
        risk_obs=risk_obs,
        cost_var=cost_vars,
        risk_var=risk_vars,
    )


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
