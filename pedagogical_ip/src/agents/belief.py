"""
Belief map representation and Bayesian update logic.

The agent maintains per-cell Gaussian beliefs over cost and risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy


@dataclass
class BeliefMap:
    """Per-cell Gaussian beliefs for cost and risk."""
    height: int
    width: int
    cost_mean: np.ndarray      # (H, W)
    cost_var: np.ndarray       # (H, W)
    risk_mean: np.ndarray      # (H, W)
    risk_var: np.ndarray       # (H, W)
    visited_mask: np.ndarray   # (H, W) bool

    # Default prior values (used by reset)
    _prior_cost_mean: float = 1.5
    _prior_cost_var: float = 4.0
    _prior_risk_mean: float = 0.1
    _prior_risk_var: float = 0.25

    @property
    def H(self) -> int:
        """Height alias for CellBelief protocol."""
        return self.height

    @property
    def W(self) -> int:
        """Width alias for CellBelief protocol."""
        return self.width

    @classmethod
    def from_prior(
        cls,
        height: int,
        width: int,
        prior_cost_mean: float = 1.5,
        prior_cost_var: float = 4.0,
        prior_risk_mean: float = 0.1,
        prior_risk_var: float = 0.25,
    ) -> BeliefMap:
        """Create a belief map with uniform priors."""
        return cls(
            height=height,
            width=width,
            cost_mean=np.full((height, width), prior_cost_mean, dtype=np.float64),
            cost_var=np.full((height, width), prior_cost_var, dtype=np.float64),
            risk_mean=np.full((height, width), prior_risk_mean, dtype=np.float64),
            risk_var=np.full((height, width), prior_risk_var, dtype=np.float64),
            visited_mask=np.zeros((height, width), dtype=bool),
            _prior_cost_mean=prior_cost_mean,
            _prior_cost_var=prior_cost_var,
            _prior_risk_mean=prior_risk_mean,
            _prior_risk_var=prior_risk_var,
        )

    def get_belief(self, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) for a cell — CellBelief protocol.

        Returns:
            mean: array([cost_mean, risk_mean])
            var:  array([cost_var, risk_var])
        """
        mean = np.array([self.cost_mean[row, col], self.risk_mean[row, col]])
        var = np.array([self.cost_var[row, col], self.risk_var[row, col]])
        return mean, var

    def copy(self) -> BeliefMap:
        """Deep copy."""
        return BeliefMap(
            height=self.height,
            width=self.width,
            cost_mean=self.cost_mean.copy(),
            cost_var=self.cost_var.copy(),
            risk_mean=self.risk_mean.copy(),
            risk_var=self.risk_var.copy(),
            visited_mask=self.visited_mask.copy(),
            _prior_cost_mean=self._prior_cost_mean,
            _prior_cost_var=self._prior_cost_var,
            _prior_risk_mean=self._prior_risk_mean,
            _prior_risk_var=self._prior_risk_var,
        )

    def reset(self, **kwargs) -> None:
        """Reset all beliefs to prior — CellBelief protocol."""
        pcm = kwargs.get("prior_cost_mean", self._prior_cost_mean)
        pcv = kwargs.get("prior_cost_var", self._prior_cost_var)
        prm = kwargs.get("prior_risk_mean", self._prior_risk_mean)
        prv = kwargs.get("prior_risk_var", self._prior_risk_var)
        self.cost_mean[:] = pcm
        self.cost_var[:] = pcv
        self.risk_mean[:] = prm
        self.risk_var[:] = prv
        self.visited_mask[:] = False

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a dict of arrays for NPZ saving."""
        return {
            "belief_cost_mean": self.cost_mean.copy(),
            "belief_cost_var": self.cost_var.copy(),
            "belief_risk_mean": self.risk_mean.copy(),
            "belief_risk_var": self.risk_var.copy(),
        }

    def total_variance(self) -> float:
        """Sum of all cost + risk variances (scalar summary)."""
        return float(self.cost_var.sum() + self.risk_var.sum())


def bayesian_update(
    prior_mean: float,
    prior_var: float,
    obs: float,
    obs_var: float,
) -> tuple[float, float]:
    """
    Scalar Kalman-style Gaussian update.

    posterior_var  = 1 / (1/prior_var + 1/obs_var)
    posterior_mean = posterior_var * (prior_mean/prior_var + obs/obs_var)
    """
    if obs_var <= 0:
        # Perfect observation
        return obs, 1e-10
    inv_post_var = 1.0 / prior_var + 1.0 / obs_var
    post_var = 1.0 / inv_post_var
    post_mean = post_var * (prior_mean / prior_var + obs / obs_var)
    return post_mean, post_var


def update_belief_cell(
    belief: BeliefMap,
    row: int,
    col: int,
    obs_cost: float,
    obs_risk: float,
    obs_cost_var: float,
    obs_risk_var: float,
) -> None:
    """Update belief for a single cell in-place."""
    if not (0 <= row < belief.height and 0 <= col < belief.width):
        return

    new_cost_mean, new_cost_var = bayesian_update(
        belief.cost_mean[row, col],
        belief.cost_var[row, col],
        obs_cost,
        obs_cost_var,
    )
    new_risk_mean, new_risk_var = bayesian_update(
        belief.risk_mean[row, col],
        belief.risk_var[row, col],
        obs_risk,
        obs_risk_var,
    )
    belief.cost_mean[row, col] = new_cost_mean
    belief.cost_var[row, col] = new_cost_var
    belief.risk_mean[row, col] = max(0.0, min(1.0, new_risk_mean))
    belief.risk_var[row, col] = new_risk_var


def apply_warning_to_belief(
    belief: BeliefMap,
    warning_type: str,
    strength: float = 0.4,
) -> None:
    """
    Apply a symbolic warning to the agent's belief map.

    Warnings shift risk_mean up/down for the relevant region.
    """
    H, W = belief.height, belief.width
    mid_col = W // 2

    if warning_type == "LEFT_AREA_RISKY":
        cols = range(0, mid_col)
        for r in range(H):
            for c in cols:
                belief.risk_mean[r, c] = min(1.0, belief.risk_mean[r, c] + strength)
                belief.risk_var[r, c] *= 0.5   # reduce uncertainty
    elif warning_type == "RIGHT_AREA_RISKY":
        cols = range(mid_col, W)
        for r in range(H):
            for c in cols:
                belief.risk_mean[r, c] = min(1.0, belief.risk_mean[r, c] + strength)
                belief.risk_var[r, c] *= 0.5
    elif warning_type == "DOOR_PATH_SAFE":
        # Reduce risk belief near doors — we mark a 3×3 patch around each door
        # (caller should pass door positions; here we affect center region)
        mid_r = H // 2
        for r in range(max(0, mid_r - 1), min(H, mid_r + 2)):
            for c in range(max(0, mid_col - 1), min(W, mid_col + 2)):
                belief.risk_mean[r, c] = max(0.0, belief.risk_mean[r, c] - strength)
                belief.cost_mean[r, c] = max(0.5, belief.cost_mean[r, c] - 1.0)
                belief.risk_var[r, c] *= 0.5
    elif warning_type == "CURRENT_PLAN_RISKY":
        # Generic: raise overall risk belief slightly
        belief.risk_mean[:] = np.minimum(1.0, belief.risk_mean + strength * 0.3)
        belief.risk_var[:] *= 0.8


# ── v1a: RSA-based warning update ────────────────────────────────────

# Region masks for the v1a utterance vocabulary
_REGION_MASKS_CACHE: dict[tuple[int, int], dict[str, np.ndarray]] = {}


def _get_region_masks(H: int, W: int) -> dict[str, np.ndarray]:
    """Get or build region masks (cached)."""
    key = (H, W)
    if key not in _REGION_MASKS_CACHE:
        mid_r, mid_c = H // 2, W // 2
        masks: dict[str, np.ndarray] = {}
        masks["LEFT_RISKY"] = np.zeros((H, W), dtype=bool)
        masks["LEFT_RISKY"][:, :mid_c] = True
        masks["RIGHT_RISKY"] = np.zeros((H, W), dtype=bool)
        masks["RIGHT_RISKY"][:, mid_c:] = True
        masks["UPPER_RISKY"] = np.zeros((H, W), dtype=bool)
        masks["UPPER_RISKY"][:mid_r, :] = True
        masks["LOWER_RISKY"] = np.zeros((H, W), dtype=bool)
        masks["LOWER_RISKY"][mid_r:, :] = True
        masks["DOOR_PATH_SAFE"] = np.zeros((H, W), dtype=bool)
        r_lo, r_hi = max(0, mid_r - 1), min(H, mid_r + 2)
        c_lo, c_hi = max(0, mid_c - 1), min(W, mid_c + 2)
        masks["DOOR_PATH_SAFE"][r_lo:r_hi, c_lo:c_hi] = True
        masks["CURRENT_PATH_RISKY"] = np.ones((H, W), dtype=bool)  # fallback: all
        _REGION_MASKS_CACHE[key] = masks
    return _REGION_MASKS_CACHE[key]


def apply_rsa_warning(
    belief: BeliefMap,
    utterance: str,
    warn_sensitivity: float = 0.5,
    pseudo_obs_var: float = 0.5,
) -> None:
    """
    v1a: Apply a symbolic RSA warning via precision-weighted fusion.

    Risky utterance → pseudo-observation y=1.0 on region cells.
    Safe utterance  → pseudo-observation y=0.0.
    Update strength depends on warn_sensitivity (particle-specific trait).
    Vectorized over all cells in the region mask.
    """
    H, W = belief.height, belief.width
    masks = _get_region_masks(H, W)

    # Determine region mask
    mask = masks.get(utterance)
    if mask is None:
        return

    # Risky → y=1.0;  Safe → y=0.0
    is_risky = utterance != "DOOR_PATH_SAFE"
    y_pseudo = 1.0 if is_risky else 0.0

    # Effective observation variance (lower = stronger update)
    # Scaled by warn_sensitivity (higher sensitivity → lower obs_var → bigger update)
    eff_obs_var = pseudo_obs_var / max(warn_sensitivity, 0.01)

    # Vectorized precision fusion over masked cells
    old_prec = 1.0 / np.maximum(belief.risk_var[mask], 1e-10)
    obs_prec = warn_sensitivity / max(eff_obs_var, 1e-10)
    new_prec = old_prec + obs_prec
    new_var = 1.0 / new_prec

    new_mean = new_var * (
        belief.risk_mean[mask] * old_prec + y_pseudo * obs_prec
    )

    belief.risk_var[mask] = new_var
    belief.risk_mean[mask] = np.clip(new_mean, 0.0, 1.0)


def log_det_risk_var(belief: BeliefMap) -> float:
    """
    Compute log-determinant of risk variance: Σ_i log σ²_i.

    Used for information gain: IG = log_det_before - log_det_after.
    """
    return float(np.sum(np.log(np.maximum(belief.risk_var, 1e-10))))
