"""
Symbolic RSA Warning Module — v1a.

Fixed utterance inventory with diagonal-Gaussian RSA scoring.
Regions are defined as cell masks; scoring uses KL-based inclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Utterance inventory ──────────────────────────────────────────────
UTTERANCE_VOCAB = [
    "LEFT_RISKY",
    "RIGHT_RISKY",
    "UPPER_RISKY",
    "LOWER_RISKY",
    "DOOR_PATH_SAFE",
    "CURRENT_PATH_RISKY",
]

# Whether an utterance signals danger (True) or safety (False)
UTTERANCE_IS_RISKY = {
    "LEFT_RISKY": True,
    "RIGHT_RISKY": True,
    "UPPER_RISKY": True,
    "LOWER_RISKY": True,
    "DOOR_PATH_SAFE": False,
    "CURRENT_PATH_RISKY": True,
}


# ── Region definitions ───────────────────────────────────────────────
def _build_region_masks(H: int, W: int) -> dict[str, np.ndarray]:
    """Build boolean masks for each candidate region."""
    mid_r, mid_c = H // 2, W // 2
    masks: dict[str, np.ndarray] = {}

    m = np.zeros((H, W), dtype=bool)
    m[:, :mid_c] = True
    masks["LEFT_RISKY"] = m.copy()

    m[:] = False
    m[:, mid_c:] = True
    masks["RIGHT_RISKY"] = m.copy()

    m[:] = False
    m[:mid_r, :] = True
    masks["UPPER_RISKY"] = m.copy()

    m[:] = False
    m[mid_r:, :] = True
    masks["LOWER_RISKY"] = m.copy()

    # Door path: center band (rows mid_r-1..mid_r+1, cols mid_c-1..mid_c+1)
    m[:] = False
    r_lo, r_hi = max(0, mid_r - 1), min(H, mid_r + 2)
    c_lo, c_hi = max(0, mid_c - 1), min(W, mid_c + 2)
    m[r_lo:r_hi, c_lo:c_hi] = True
    masks["DOOR_PATH_SAFE"] = m.copy()

    # Current path: placeholder (will be overridden at scoring time)
    masks["CURRENT_PATH_RISKY"] = np.zeros((H, W), dtype=bool)

    return masks


def _region_risk_summary(
    true_risk: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Compute (mean_risk, var_risk) for a region — the region's Gaussian concept B_r."""
    cells = true_risk[mask]
    if len(cells) == 0:
        return 0.0, 1.0
    return float(cells.mean()), float(cells.var() + 0.01)


# ── RSA scoring (vectorized) ────────────────────────────────────────
def _kl_diag_gaussian(
    mu_a: np.ndarray, var_a: np.ndarray,
    mu_b: np.ndarray, var_b: np.ndarray,
) -> float:
    """
    KL(A || B) for diagonal Gaussians, summed over dimensions.

    KL = 0.5 * Σ [ log(var_b/var_a) + var_a/var_b + (mu_a-mu_b)²/var_b - 1 ]
    """
    var_a = np.maximum(var_a, 1e-8)
    var_b = np.maximum(var_b, 1e-8)
    kl = 0.5 * np.sum(
        np.log(var_b / var_a) + var_a / var_b
        + (mu_a - mu_b) ** 2 / var_b - 1.0
    )
    return float(kl)


def score_utterances(
    learner_belief_risk_mean: np.ndarray,
    learner_belief_risk_var: np.ndarray,
    true_risk: np.ndarray,
    agent_pos: tuple[int, int],
    current_plan: list[tuple[int, int]] | None = None,
    alpha: float = 5.0,
    beta: float = 0.1,
    tau: float = 1.0,
) -> dict[str, float]:
    """
    Score each utterance using diagonal-Gaussian RSA.

    Returns dict mapping utterance -> S1 utility score.
    Higher = more informative warning.
    """
    H, W = true_risk.shape
    masks = _build_region_masks(H, W)

    # Build current-path mask from agent plan
    if current_plan and len(current_plan) > 1:
        cp_mask = np.zeros((H, W), dtype=bool)
        for r, c in current_plan[:8]:  # first 8 steps
            if 0 <= r < H and 0 <= c < W:
                cp_mask[r, c] = True
                # Also include 1-ring neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        cp_mask[nr, nc] = True
        masks["CURRENT_PATH_RISKY"] = cp_mask
    else:
        # Fallback: 3x3 around agent
        cp_mask = np.zeros((H, W), dtype=bool)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
                if 0 <= nr < H and 0 <= nc < W:
                    cp_mask[nr, nc] = True
        masks["CURRENT_PATH_RISKY"] = cp_mask

    scores: dict[str, float] = {}

    for utt in UTTERANCE_VOCAB:
        mask = masks[utt]
        n_cells = mask.sum()
        if n_cells == 0:
            scores[utt] = -np.inf
            continue

        # Region concept B_r: true risk in the region
        mu_b, var_b = _region_risk_summary(true_risk, mask)

        # Learner's current belief A_t: what learner thinks about this region
        mu_a = learner_belief_risk_mean[mask].mean()
        var_a = learner_belief_risk_var[mask].mean() + 0.01

        # ── v1f: Directional danger-underestimation scoring ──
        # Only reward utterances about regions where:
        #   (a) true risk is genuinely HIGH, AND
        #   (b) learner UNDERESTIMATES the danger
        #
        # OLD (broken): symmetric KL(B||A) rewarded ANY miscalibration,
        #   including "learner thinks 0.1 but truth is 0.0" (safe region).
        #   This caused RSA to pick LEFT_RISKY (risk=0.0) over RIGHT_RISKY.
        #
        # NEW: danger_underestimation × information_need
        is_risky_utt = UTTERANCE_IS_RISKY.get(utt, True)

        if is_risky_utt:
            # For "X_RISKY" utterances: reward if true risk > learner belief
            # max(0, true - belief) -- only underestimation of danger matters
            danger_gap = max(0.0, mu_b - mu_a)

            # Also consider max true risk in region (not just mean)
            max_risk_in_region = float(true_risk[mask].max())
            max_underest = max(0.0, max_risk_in_region - mu_a)

            # Combined: mean underestimation + peak danger signal
            underest_score = danger_gap + 0.5 * max_underest

            # Information structure: how uncertain is learner about this region?
            info_need = float(np.mean(learner_belief_risk_var[mask]))

            utility = underest_score * (1.0 + info_need) / tau
        else:
            # For "X_SAFE" utterances (e.g. DOOR_PATH_SAFE):
            # reward if true risk is LOW and learner overestimates risk
            safety_gap = max(0.0, mu_a - mu_b)
            utility = safety_gap / tau

        # Log-det of region variance (volume penalty for vague concepts)
        log_det = np.log(var_b + 1e-8)

        # S1 score: alpha * (utility - volume_penalty)
        scores[utt] = alpha * (utility - beta * log_det)

    return scores


def select_best_warning(
    learner_belief_risk_mean: np.ndarray,
    learner_belief_risk_var: np.ndarray,
    true_risk: np.ndarray,
    agent_pos: tuple[int, int],
    current_plan: list[tuple[int, int]] | None = None,
    alpha: float = 5.0,
    beta: float = 0.1,
    tau: float = 1.0,
) -> tuple[str, float]:
    """
    Select the best warning utterance via symbolic RSA.

    Returns (utterance, score).
    """
    scores = score_utterances(
        learner_belief_risk_mean, learner_belief_risk_var,
        true_risk, agent_pos, current_plan,
        alpha, beta, tau,
    )
    best_utt = max(scores, key=scores.get)  # type: ignore
    return best_utt, scores[best_utt]


# ── PragmaticWarner adapter ─────────────────────────────────────────

class RSAWarner:
    """Adapter implementing PragmaticWarner for the v0-v1d RSA warning system.

    select_utterance: picks the most informative region-based utterance.
    listener_update: applies precision-fusion warning update to BeliefMap.

    All logic delegates to existing functions — no new behavior.
    """

    def __init__(self, alpha: float = 5.0, beta: float = 0.1, tau: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

    def select_utterance(self, state_info: dict) -> str | None:
        """S1: select best warning using RSA scoring.

        Required state_info keys:
          - learner_belief_risk_mean: (H, W) ndarray
          - learner_belief_risk_var: (H, W) ndarray
          - true_risk: (H, W) ndarray
          - agent_pos: (row, col)
          - current_plan: list of (row, col) or None
        """
        utt, score = select_best_warning(
            learner_belief_risk_mean=state_info["learner_belief_risk_mean"],
            learner_belief_risk_var=state_info["learner_belief_risk_var"],
            true_risk=state_info["true_risk"],
            agent_pos=state_info["agent_pos"],
            current_plan=state_info.get("current_plan"),
            alpha=self.alpha,
            beta=self.beta,
            tau=self.tau,
        )
        if score <= 0:
            return None
        return utt

    def listener_update(self, utterance: str, belief, **kwargs):
        """L0: apply warning to BeliefMap using precision fusion.

        Delegates to belief.apply_warning_to_belief().
        Returns the warning message string for logging.
        """
        from ..agents.belief import apply_warning_to_belief

        warn_sensitivity = kwargs.get("warn_sensitivity", 1.0)
        apply_warning_to_belief(belief, utterance, warn_sensitivity)
        return {"utterance": utterance, "applied": True}

