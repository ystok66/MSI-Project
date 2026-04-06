"""
Warning Update V2 — planner-relevant warnings.

KEY CHANGE from V1: Warning is no longer optimized for "prediction change"
but for "action change" (lane choice shift).

Two mechanisms:
1. Pseudo-label injection into shared risk_head (unchanged)
2. Lane-level warning bias: temporary lane penalty that the planner sees

Utterance selection: action-gap objective
  u* = argmax_u [ J_risky(after u) - J_safe(after u) ]
  = select the utterance that maximally favors safe lane in the planner
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from dataclasses import dataclass

import numpy as np

from .risk_model import BayesianRiskHead
from .feature_belief import FeatureBeliefMap


class Utterance(Enum):
    RISKY_TEXTURE_AHEAD = "risky_texture_ahead"
    UPPER_LANE_RISKY = "upper_lane_risky"
    SAFE_DETOUR_OPEN = "safe_detour_open"


# Feature vector: [lane_id, gate_flag, texture_cue_1, texture_cue_2]
PROTOTYPES = {
    Utterance.RISKY_TEXTURE_AHEAD: np.array([0.5, 0.0, 0.85, 0.80]),
    Utterance.UPPER_LANE_RISKY:    np.array([0.0, 0.0, 0.70, 0.60]),
    Utterance.SAFE_DETOUR_OPEN:    np.array([1.0, 0.0, 0.05, 0.05]),
}

PSEUDO_LABELS = {
    Utterance.RISKY_TEXTURE_AHEAD: 0.8,
    Utterance.UPPER_LANE_RISKY:    0.7,
    Utterance.SAFE_DETOUR_OPEN:    0.0,
}


@dataclass
class WarningEffect:
    """Record of warning application."""
    utterance: Utterance
    target_cells: list[tuple[int, int]]
    matching_weights: list[float]
    risk_before: list[float]
    risk_after: list[float]
    lane_bias_applied: float  # aggregate lane penalty


def compute_lane_bias(
    utterance: Utterance,
    lane_cells: list[tuple[int, int]],
    feature_belief: FeatureBeliefMap,
    tau: float = 0.3,
) -> float:
    """
    Compute aggregate lane-level warning bias.

    b_warn(lane) = sum_j alpha_j(u) * y_u

    This is added as a lump penalty to the warned lane's total cost.
    """
    proto = PROTOTYPES[utterance]
    y_label = PSEUDO_LABELS[utterance]
    bias = 0.0
    for r, c in lane_cells:
        x_hat = feature_belief.get_mean(r, c)
        dist_sq = float(np.sum((x_hat - proto) ** 2))
        alpha = float(np.exp(-dist_sq / tau))
        bias += alpha * y_label
    return bias


def apply_warning(
    utterance: Utterance,
    upcoming_cells: list[tuple[int, int]],
    feature_belief: FeatureBeliefMap,
    risk_head: BayesianRiskHead,
    warned_lane_bias: dict,
    segment_index: int,
    weight: float = 5.0,
    tau: float = 0.3,
    lambda_lane_warn: float = 5.0,
) -> WarningEffect:
    """
    Apply warning with BOTH mechanisms:
    1. Pseudo-label injection into risk_head (for cross-episode learning)
    2. Lane-level bias into warned_lane_bias dict (for within-episode planner)
    """
    proto = PROTOTYPES[utterance]
    y_label = PSEUDO_LABELS[utterance]

    target_cells, matching_weights = [], []
    risk_before, risk_after = [], []

    for r, c in upcoming_cells:
        x_hat = feature_belief.get_mean(r, c)
        dist_sq = float(np.sum((x_hat - proto) ** 2))
        alpha = float(np.exp(-dist_sq / tau))
        effective_weight = weight * alpha

        p_before = risk_head.predict_risk(x_hat)
        if effective_weight > 0.01:
            risk_head.update_from_label(x_hat, y_label, weight=effective_weight)
        p_after = risk_head.predict_risk(x_hat)

        target_cells.append((r, c))
        matching_weights.append(alpha)
        risk_before.append(p_before)
        risk_after.append(p_after)

    # Lane-level bias: aggregate penalty on warned risky cells
    lane_bias = compute_lane_bias(utterance, upcoming_cells, feature_belief, tau)
    scaled_bias = lambda_lane_warn * lane_bias

    # Store bias keyed by segment — planner will add this to risky-lane cells
    warned_lane_bias[segment_index] = scaled_bias

    return WarningEffect(
        utterance=utterance,
        target_cells=target_cells,
        matching_weights=matching_weights,
        risk_before=risk_before,
        risk_after=risk_after,
        lane_bias_applied=scaled_bias,
    )


def select_best_warning_action_gap(
    candidate_cells: list[tuple[int, int]],
    feature_belief: FeatureBeliefMap,
    risk_head: BayesianRiskHead,
    tau: float = 0.3,
    lambda_lane_warn: float = 5.0,
) -> Utterance | None:
    """
    Action-gap utterance selection.

    For each candidate utterance, estimate how much lane-level bias
    it would create. Select the one with the largest bias (= most
    likely to shift planner from risky to safe lane).

    This is a lightweight surrogate for the full planner-replan approach:
    instead of re-running A*, we estimate the cost gap increase.
    """
    best_u = None
    best_gap = 0.5  # minimum threshold: bias must be at least 0.5

    for utt in Utterance:
        bias = compute_lane_bias(utt, candidate_cells, feature_belief, tau)
        gap = lambda_lane_warn * bias
        if gap > best_gap:
            best_gap = gap
            best_u = utt

    return best_u


# Backward compat
def select_best_warning(candidate_cells, feature_belief, risk_head, tau=0.3):
    return select_best_warning_action_gap(
        candidate_cells, feature_belief, risk_head, tau)


# ── PragmaticWarner adapter ─────────────────────────────────────────

class LaneWarner:
    """Adapter implementing PragmaticWarner for the V2 lane-based warning system.

    select_utterance: action-gap selection over 3 prototype utterances.
    listener_update: pseudo-label injection + lane-level bias.

    All logic delegates to existing functions — no new behavior.
    """

    def __init__(self, tau: float = 0.3, lambda_lane_warn: float = 5.0,
                 weight: float = 5.0):
        self.tau = tau
        self.lambda_lane_warn = lambda_lane_warn
        self.weight = weight

    def select_utterance(self, state_info: dict) -> str | None:
        """S1: pick the utterance with the largest action-gap bias.

        Required state_info keys:
          - candidate_cells: list of (row, col) — risky lane cells
          - feature_belief: FeatureBeliefMap instance
          - risk_head: BayesianRiskHead instance
        """
        utt = select_best_warning_action_gap(
            candidate_cells=state_info["candidate_cells"],
            feature_belief=state_info["feature_belief"],
            risk_head=state_info["risk_head"],
            tau=self.tau,
            lambda_lane_warn=self.lambda_lane_warn,
        )
        if utt is None:
            return None
        return utt.value  # Utterance enum → string

    def listener_update(self, utterance: str, belief, **kwargs):
        """L0: apply pseudo-label + lane-level bias.

        Required kwargs:
          - upcoming_cells: list of (row, col)
          - risk_head: BayesianRiskHead instance
          - warned_lane_bias: dict to store bias (mutated in-place)
          - segment_index: int
        """
        # Map string back to Utterance enum
        utt_enum = None
        for u in Utterance:
            if u.value == utterance:
                utt_enum = u
                break
        if utt_enum is None:
            return None

        effect = apply_warning(
            utterance=utt_enum,
            upcoming_cells=kwargs["upcoming_cells"],
            feature_belief=belief,
            risk_head=kwargs["risk_head"],
            warned_lane_bias=kwargs["warned_lane_bias"],
            segment_index=kwargs["segment_index"],
            weight=self.weight,
            tau=self.tau,
            lambda_lane_warn=self.lambda_lane_warn,
        )
        return effect


# ── Step 2: Variant Router ─────────────────────────────────────────

# Warning variants:
#   legacy_bias      — original dual-mechanism (pseudo-label + lane bias)
#   rsa_obs_l0       — RSA literal listener belief update only
#   rsa_obs_s1       — RSA pragmatic speaker (S1) belief update only
#   rsa_obs_s1_trust — RSA S1 + trust gate (ablation)
#   rsa_plus_phase10 — RSA S1 + Phase 10 apply_warn_update (hybrid ablation)

VALID_WARNING_VARIANTS = frozenset([
    "legacy_bias", "rsa_obs_l0", "rsa_obs_s1",
    "rsa_obs_s1_trust", "rsa_plus_phase10",
])


def apply_warning_dispatch(
    warning_variant: str,
    utterance: Utterance,
    upcoming_cells: list[tuple[int, int]],
    feature_belief: FeatureBeliefMap,
    risk_head: BayesianRiskHead,
    warned_lane_bias: dict,
    segment_index: int,
    weight: float = 5.0,
    tau: float = 0.3,
    lambda_lane_warn: float = 5.0,
    # RSA-specific
    rsa_channel=None,
    rsa_belief_state=None,
    rsa_context: dict = None,
    rsa_utterance=None,
    tau_hat: float = 0.3,
) -> dict:
    """Dispatch warning to legacy or RSA based on variant.

    For legacy: calls apply_warning() as before.
    For RSA: calls rsa_channel.update_belief() — NO pseudo-label, NO lane bias.

    Returns:
        dict with variant-specific diagnostics
    """
    if warning_variant not in VALID_WARNING_VARIANTS:
        raise ValueError(f"Unknown warning_variant: {warning_variant}. "
                         f"Valid: {VALID_WARNING_VARIANTS}")

    if warning_variant == "legacy_bias":
        effect = apply_warning(
            utterance, upcoming_cells, feature_belief, risk_head,
            warned_lane_bias, segment_index,
            weight=weight, tau=tau, lambda_lane_warn=lambda_lane_warn)
        return {
            "variant": "legacy_bias",
            "lane_bias": effect.lane_bias_applied,
            "n_cells": len(effect.target_cells),
        }

    # RSA variants — pure belief update, no pseudo-label, no lane bias
    if rsa_channel is None or rsa_belief_state is None:
        raise ValueError(f"RSA variant '{warning_variant}' requires "
                         f"rsa_channel and rsa_belief_state")

    if rsa_context is None:
        rsa_context = {"has_left_branch": True, "has_right_branch": True}

    if rsa_utterance is None:
        from .rsa_warning_channel import RSAUtterance
        _legacy_to_rsa = {
            Utterance.UPPER_LANE_RISKY: RSAUtterance.WARN_LEFT,
            Utterance.RISKY_TEXTURE_AHEAD: RSAUtterance.WARN_AHEAD,
            Utterance.SAFE_DETOUR_OPEN: RSAUtterance.GENERIC_WARN,
        }
        rsa_utterance = _legacy_to_rsa.get(utterance, RSAUtterance.WARN_AHEAD)

    rsa_mode_map = {
        "rsa_obs_l0": "l0",
        "rsa_obs_s1": "s1",
        "rsa_obs_s1_trust": "s1_trust",
        "rsa_plus_phase10": "s1",
    }
    rsa_mode = rsa_mode_map[warning_variant]

    info = rsa_channel.update_belief(
        rsa_belief_state, rsa_utterance, rsa_context,
        variant=rsa_mode, tau_hat=tau_hat)

    # Hybrid: ALSO do legacy (both paths active)
    if warning_variant == "rsa_plus_phase10":
        legacy_effect = apply_warning(
            utterance, upcoming_cells, feature_belief, risk_head,
            warned_lane_bias, segment_index,
            weight=weight, tau=tau, lambda_lane_warn=lambda_lane_warn)
        info["legacy_lane_bias"] = legacy_effect.lane_bias_applied
        info["hybrid"] = True

    return info


# ── Phase 1A: Adapter Functions ────────────────────────────────────
# These consume WarningBeliefDelta — they do NOT define warning semantics.

def apply_planner_adapter(delta, warned_cell_extra: dict) -> None:
    """Write planner penalties from WarningBeliefDelta to warned_cell_extra.

    This replaces the legacy lane_bias → _build_warned_cell_extra chain
    for RSA variants. For legacy_bias variant, this is NOT called (legacy
    writes its own lane_bias directly).

    Args:
        delta: WarningBeliefDelta instance
        warned_cell_extra: mutable dict, planner reads this
    """
    warned_cell_extra.update(delta.planner_cell_penalties)


def apply_pseudolabel_adapter(delta, risk_head) -> int:
    """Inject pseudo-labels from WarningBeliefDelta into risk_head.

    Only called when legacy compatibility or hybrid mode is needed.

    Args:
        delta: WarningBeliefDelta instance
        risk_head: BayesianRiskHead instance (mutated)

    Returns:
        Number of pseudo-label updates applied
    """
    n = 0
    for entry in delta.pseudo_label_pkg:
        risk_head.update_from_label(entry.z_proto, entry.y_label, weight=entry.weight)
        n += 1
    return n


# ── Step 2: Segment → RSA Context Mapping ──────────────────────────


def map_segment_to_rsa_context(seg_meta, gridmap=None) -> dict:
    """Build RSA context dict from SegmentMeta topology.

    Maps segment structure to RSA hypothesis space WITHOUT hardcoding
    row numbers. Uses risky_row / safe_row semantics:
      - risky_row → the "risky side" (mapped to left_risky or right_risky)
      - safe_row → the "safe side"

    The naming convention: row 1 → "upper" → "left" in RSA hypothesis,
    row 3 → "lower" → "right". But the mapping only matters for matching
    utterances to hypotheses — test_mirror_symmetry verifies invariance.

    Returns dict suitable for rsa_warning_channel functions.
    """
    has_left = True
    has_right = True

    # Determine which RSA side the risky row maps to
    if seg_meta.risky_row <= 2:
        # Upper row → "left" in RSA convention
        risky_side = "left"
    else:
        risky_side = "right"

    return {
        "has_left_branch": has_left,
        "has_right_branch": has_right,
        "risky_side": risky_side,
        "segment_index": seg_meta.index,
    }


def map_legacy_to_rsa_utterance(legacy_utt, risky_side: str = "left"):
    """Map legacy Utterance enum → RSAUtterance using segment semantics.

    The risky_side tells us which direction the warning should point:
      - UPPER_LANE_RISKY → WARN_LEFT if risky_side=="left" else WARN_RIGHT
      - RISKY_TEXTURE_AHEAD → WARN_AHEAD
      - SAFE_DETOUR_OPEN → GENERIC_WARN

    This avoids hardcoding row→side everywhere.
    """
    from .rsa_warning_channel import RSAUtterance

    if legacy_utt == Utterance.UPPER_LANE_RISKY:
        if risky_side == "left":
            return RSAUtterance.WARN_LEFT
        else:
            return RSAUtterance.WARN_RIGHT
    elif legacy_utt == Utterance.RISKY_TEXTURE_AHEAD:
        return RSAUtterance.WARN_AHEAD
    elif legacy_utt == Utterance.SAFE_DETOUR_OPEN:
        return RSAUtterance.GENERIC_WARN
    else:
        return RSAUtterance.WARN_AHEAD
