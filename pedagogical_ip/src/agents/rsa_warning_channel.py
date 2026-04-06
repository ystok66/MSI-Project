"""RSA Warning Channel — observation-based warning semantics.

Replaces the legacy dual-mechanism warning (pseudo-label + lane bias) with
an RSA-based belief update channel:

    u_t → S1(u_t|r,c_t) → b_t+(r) → planner adapter

Warning only modifies AgentBelief. It does NOT directly modify the planner,
risk head, or lane bias. The planner reads updated belief through the adapter.

Hypothesis space (fixed 4-way, segment-level):
    R = {left_risky, right_risky, both_safe, hazard_ahead}

Three hyperparameters only:
    λ_sem:   semantic matching sharpness
    α_RSA:   speaker rationality
    λ_C:     utterance cost weight

Shadow-only. Does NOT modify any frozen module.
Does NOT change canonical micro action space {WAIT, WARN}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════
# Risk Hypothesis Space
# ═══════════════════════════════════════════════════════════

class RiskHypothesis(Enum):
    """Segment-level risk hypotheses (fixed 4-way)."""
    LEFT_RISKY = "left_risky"
    RIGHT_RISKY = "right_risky"
    BOTH_SAFE = "both_safe"
    HAZARD_AHEAD = "hazard_ahead"


ALL_HYPOTHESES = list(RiskHypothesis)
N_HYPOTHESES = len(ALL_HYPOTHESES)


# ═══════════════════════════════════════════════════════════
# Utterance Inventory
# ═══════════════════════════════════════════════════════════

class RSAUtterance(Enum):
    """Warning utterance types for RSA channel."""
    WARN_LEFT = "warn_left"           # specific: left/upper lane risky
    WARN_RIGHT = "warn_right"         # specific: right/lower lane risky
    WARN_AHEAD = "warn_ahead"         # semi-specific: hazard ahead
    GENERIC_WARN = "generic_warn"     # generic caution


ALL_UTTERANCES = list(RSAUtterance)
N_UTTERANCES = len(ALL_UTTERANCES)

# Utterance cost: specific < generic (more effort to be specific)
UTTERANCE_COST = {
    RSAUtterance.WARN_LEFT: 0.0,       # free: directional
    RSAUtterance.WARN_RIGHT: 0.0,      # free: directional
    RSAUtterance.WARN_AHEAD: 0.2,      # slightly costly
    RSAUtterance.GENERIC_WARN: 0.5,    # vague = cheap for speaker
}


# ═══════════════════════════════════════════════════════════
# Semantic Match Function
# ═══════════════════════════════════════════════════════════

def _semantic_match(utterance: RSAUtterance, hypothesis: RiskHypothesis,
                    context: dict) -> float:
    """Compute match(u, r, c).

    Three factors:
    1. Side alignment: does u point to the same side that r says is risky?
    2. Specificity: specific u matches specific r better than generic matches anything
    3. Branch availability: does the context have the branches u refers to?

    Returns:
        match score ∈ [0, 1]
    """
    has_left = context.get("has_left_branch", True)
    has_right = context.get("has_right_branch", True)

    if utterance == RSAUtterance.WARN_LEFT:
        if not has_left:
            return 0.0  # utterance refers to nonexistent branch
        if hypothesis == RiskHypothesis.LEFT_RISKY:
            return 1.0  # perfect alignment
        elif hypothesis == RiskHypothesis.HAZARD_AHEAD:
            return 0.5  # partial: left is ahead
        elif hypothesis == RiskHypothesis.BOTH_SAFE:
            return 0.05  # contradicts
        else:
            return 0.1  # misaligned (right is risky, not left)

    elif utterance == RSAUtterance.WARN_RIGHT:
        if not has_right:
            return 0.0
        if hypothesis == RiskHypothesis.RIGHT_RISKY:
            return 1.0
        elif hypothesis == RiskHypothesis.HAZARD_AHEAD:
            return 0.5
        elif hypothesis == RiskHypothesis.BOTH_SAFE:
            return 0.05
        else:
            return 0.1

    elif utterance == RSAUtterance.WARN_AHEAD:
        if hypothesis == RiskHypothesis.HAZARD_AHEAD:
            return 1.0
        elif hypothesis in (RiskHypothesis.LEFT_RISKY, RiskHypothesis.RIGHT_RISKY):
            return 0.6  # hazard is ahead, just not specified which side
        else:
            return 0.1

    elif utterance == RSAUtterance.GENERIC_WARN:
        # Generic: slightly elevates all non-safe hypotheses
        if hypothesis == RiskHypothesis.BOTH_SAFE:
            return 0.1
        else:
            return 0.4  # diffuse warning

    return 0.1


# ═══════════════════════════════════════════════════════════
# L0: Literal Listener
# ═══════════════════════════════════════════════════════════

def literal_listener(
    utterance: RSAUtterance,
    context: dict,
    prior: np.ndarray,
    lambda_sem: float = 3.0,
) -> np.ndarray:
    """L0(r|u,c) ∝ exp(λ_sem · match(u,r,c)) · P(r|c).

    Args:
        utterance: the warning utterance
        context: dict with has_left_branch, has_right_branch, etc.
        prior: P(r|c), shape (N_HYPOTHESES,)
        lambda_sem: semantic matching sharpness

    Returns:
        L0 posterior, shape (N_HYPOTHESES,), normalized
    """
    log_lik = np.array([
        lambda_sem * _semantic_match(utterance, h, context)
        for h in ALL_HYPOTHESES
    ])
    log_post = log_lik + np.log(prior + 1e-10)
    log_post -= np.max(log_post)  # numerical stability
    post = np.exp(log_post)
    return post / (post.sum() + 1e-10)


# ═══════════════════════════════════════════════════════════
# S1: Pragmatic Speaker
# ═══════════════════════════════════════════════════════════

def pragmatic_speaker(
    hypothesis: RiskHypothesis,
    context: dict,
    prior: np.ndarray,
    alpha_rsa: float = 2.0,
    lambda_sem: float = 3.0,
    lambda_c: float = 1.0,
) -> np.ndarray:
    """S1(u|r,c) ∝ exp(α · [log L0(r|u,c) − λ_C · C(u)]).

    Args:
        hypothesis: the true risk state r
        context: branch availability etc.
        prior: prior over hypotheses for L0
        alpha_rsa: speaker rationality
        lambda_sem: passed to L0
        lambda_c: utterance cost weight

    Returns:
        S1 distribution over utterances, shape (N_UTTERANCES,)
    """
    r_idx = ALL_HYPOTHESES.index(hypothesis)
    scores = np.zeros(N_UTTERANCES)

    for u_idx, utt in enumerate(ALL_UTTERANCES):
        l0 = literal_listener(utt, context, prior, lambda_sem)
        # How well does this utterance communicate the true hypothesis?
        informativeness = np.log(l0[r_idx] + 1e-10)
        cost = UTTERANCE_COST[utt]
        scores[u_idx] = alpha_rsa * (informativeness - lambda_c * cost)

    scores -= np.max(scores)
    probs = np.exp(scores)
    return probs / (probs.sum() + 1e-10)


# ═══════════════════════════════════════════════════════════
# Belief Update
# ═══════════════════════════════════════════════════════════

@dataclass
class RSABeliefState:
    """Agent's belief over local risk hypotheses at a segment."""
    belief: np.ndarray = field(
        default_factory=lambda: np.ones(N_HYPOTHESES) / N_HYPOTHESES)
    n_updates: int = 0

    def entropy(self) -> float:
        b = self.belief + 1e-10
        return -float(np.sum(b * np.log(b)))


def listener_update_l0(
    belief_state: RSABeliefState,
    utterance: RSAUtterance,
    context: dict,
    lambda_sem: float = 3.0,
) -> dict:
    """L0 belief update: b+(r) ∝ L0(r|u,c) · b⁻(r).

    Actually this uses L0 as the likelihood directly (no speaker model).
    """
    H_before = belief_state.entropy()
    prior = belief_state.belief.copy()

    # L0 as likelihood
    posterior = literal_listener(utterance, context, prior, lambda_sem)
    belief_state.belief = posterior
    belief_state.n_updates += 1

    H_after = belief_state.entropy()
    return {
        "delta_H": round(H_before - H_after, 4),
        "H_before": round(H_before, 4),
        "H_after": round(H_after, 4),
        "prior": prior.tolist(),
        "posterior": posterior.tolist(),
        "variant": "l0",
    }


def listener_update_s1(
    belief_state: RSABeliefState,
    utterance: RSAUtterance,
    context: dict,
    lambda_sem: float = 3.0,
    alpha_rsa: float = 2.0,
    lambda_c: float = 1.0,
) -> dict:
    """S1 belief update: b+(r) ∝ S1(u|r,c) · b⁻(r).

    Uses the pragmatic speaker model as the likelihood.
    """
    H_before = belief_state.entropy()
    prior = belief_state.belief.copy()

    # For each hypothesis, compute S1(u|r,c)
    u_idx = ALL_UTTERANCES.index(utterance)
    log_lik = np.zeros(N_HYPOTHESES)
    for r_idx, hyp in enumerate(ALL_HYPOTHESES):
        s1_dist = pragmatic_speaker(
            hyp, context, prior, alpha_rsa, lambda_sem, lambda_c)
        log_lik[r_idx] = np.log(s1_dist[u_idx] + 1e-10)

    # Bayesian update
    log_post = log_lik + np.log(prior + 1e-10)
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= (posterior.sum() + 1e-10)

    belief_state.belief = posterior
    belief_state.n_updates += 1

    H_after = belief_state.entropy()
    return {
        "delta_H": round(H_before - H_after, 4),
        "H_before": round(H_before, 4),
        "H_after": round(H_after, 4),
        "prior": prior.tolist(),
        "posterior": posterior.tolist(),
        "variant": "s1",
    }


def listener_update_s1_trust(
    belief_state: RSABeliefState,
    utterance: RSAUtterance,
    context: dict,
    tau_hat: float = 0.3,
    lambda_sem: float = 3.0,
    alpha_rsa: float = 2.0,
    lambda_c: float = 1.0,
    eta_min: float = 0.3,
    eta_max: float = 2.0,
) -> dict:
    """S1 belief update with trust-gated evidence strength.

    b+(r) ∝ [S1(u|r,c)]^η_τ · b⁻(r)
    where η_τ = clip(τ̂, η_min, η_max)

    Higher trust → stronger evidence from warning.
    Lower trust → warning is discounted.
    """
    H_before = belief_state.entropy()
    prior = belief_state.belief.copy()
    eta_tau = float(np.clip(tau_hat, eta_min, eta_max))

    u_idx = ALL_UTTERANCES.index(utterance)
    log_lik = np.zeros(N_HYPOTHESES)
    for r_idx, hyp in enumerate(ALL_HYPOTHESES):
        s1_dist = pragmatic_speaker(
            hyp, context, prior, alpha_rsa, lambda_sem, lambda_c)
        log_lik[r_idx] = eta_tau * np.log(s1_dist[u_idx] + 1e-10)

    log_post = log_lik + np.log(prior + 1e-10)
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= (posterior.sum() + 1e-10)

    belief_state.belief = posterior
    belief_state.n_updates += 1

    H_after = belief_state.entropy()
    return {
        "delta_H": round(H_before - H_after, 4),
        "H_before": round(H_before, 4),
        "H_after": round(H_after, 4),
        "eta_tau": round(eta_tau, 4),
        "prior": prior.tolist(),
        "posterior": posterior.tolist(),
        "variant": "s1_trust",
    }


# ═══════════════════════════════════════════════════════════
# Planner Adapter
# ═══════════════════════════════════════════════════════════

def belief_to_risk_update(
    belief_state: RSABeliefState,
    segment_side: str = "left",
) -> float:
    """Convert hypothesis belief to a risk penalty delta for the planner.

    Maps b_t+(r) to a scalar risk adjustment for the warned segment:
        Δρ = E_r ~ b+[ρ(r)] − E_r ~ uniform[ρ(r)]

    The planner adds this to existing risk predictions.

    Args:
        belief_state: current posterior over risk hypotheses
        segment_side: "left" or "right" — which side this segment is on

    Returns:
        risk delta ∈ [-0.5, 0.5] — positive means more dangerous
    """
    b = belief_state.belief

    # Risk contribution of each hypothesis for this segment side
    if segment_side == "left":
        # left_risky → high risk, right_risky → low risk
        risk_map = np.array([0.8, 0.1, 0.1, 0.5])  # [left, right, both_safe, hazard]
    else:
        risk_map = np.array([0.1, 0.8, 0.1, 0.5])

    expected_risk = float(b @ risk_map)
    prior_risk = float(np.ones(N_HYPOTHESES) / N_HYPOTHESES @ risk_map)

    return round(expected_risk - prior_risk, 4)


def belief_to_log_odds_update(
    belief_state: RSABeliefState,
    segment_side: str = "left",
) -> float:
    """Log-odds risk update for per-cell planner adapter.

    logit(ρ̃_j) = logit(ρ̂_j) + λ_u · log[P(u|r_j=1,c) / P(u|r_j=0,c)]

    Returns the additive log-odds term.
    """
    b = belief_state.belief

    if segment_side == "left":
        p_risky = b[0] + 0.5 * b[3]  # left_risky + half of hazard_ahead
        p_safe = b[1] + b[2] + 0.5 * b[3]
    else:
        p_risky = b[1] + 0.5 * b[3]
        p_safe = b[0] + b[2] + 0.5 * b[3]

    p_risky = max(p_risky, 1e-6)
    p_safe = max(p_safe, 1e-6)

    return round(float(np.log(p_risky / p_safe)), 4)


# ═══════════════════════════════════════════════════════════
# RSA Warning Channel (unified interface)
# ═══════════════════════════════════════════════════════════

@dataclass
class RSAWarningChannel:
    """Unified RSA warning channel.

    Params:
        lambda_sem: semantic matching sharpness (default 3.0)
        alpha_rsa:  speaker rationality (default 2.0)
        lambda_c:   utterance cost weight (default 1.0)
    """
    lambda_sem: float = 3.0
    alpha_rsa: float = 2.0
    lambda_c: float = 1.0

    def select_utterance(
        self,
        true_risk_side: str,
        context: dict,
        prior: np.ndarray = None,
    ) -> RSAUtterance:
        """S1 speaker: select best utterance given true risk hypothesis.

        Args:
            true_risk_side: "left", "right", "both_safe", "ahead"
            context: branch availability
            prior: hypothesis prior (uniform if None)
        """
        hyp_map = {
            "left": RiskHypothesis.LEFT_RISKY,
            "right": RiskHypothesis.RIGHT_RISKY,
            "both_safe": RiskHypothesis.BOTH_SAFE,
            "ahead": RiskHypothesis.HAZARD_AHEAD,
        }
        hyp = hyp_map.get(true_risk_side, RiskHypothesis.HAZARD_AHEAD)

        if prior is None:
            prior = np.ones(N_HYPOTHESES) / N_HYPOTHESES

        s1_dist = pragmatic_speaker(
            hyp, context, prior,
            self.alpha_rsa, self.lambda_sem, self.lambda_c)

        return ALL_UTTERANCES[int(np.argmax(s1_dist))]

    def update_belief(
        self,
        belief_state: RSABeliefState,
        utterance: RSAUtterance,
        context: dict,
        variant: str = "s1",
        tau_hat: float = 0.3,
    ) -> dict:
        """Apply belief update from warning.

        Args:
            belief_state: mutable belief state
            utterance: the warning
            context: branch availability
            variant: "l0" | "s1" | "s1_trust"
            tau_hat: trust estimate (only used for s1_trust)

        Returns:
            diagnostics dict
        """
        if variant == "l0":
            return listener_update_l0(
                belief_state, utterance, context, self.lambda_sem)
        elif variant == "s1":
            return listener_update_s1(
                belief_state, utterance, context,
                self.lambda_sem, self.alpha_rsa, self.lambda_c)
        elif variant == "s1_trust":
            return listener_update_s1_trust(
                belief_state, utterance, context,
                tau_hat, self.lambda_sem, self.alpha_rsa, self.lambda_c)
        else:
            raise ValueError(f"Unknown RSA variant: {variant}")

    def get_risk_adapter(
        self,
        belief_state: RSABeliefState,
        segment_side: str,
    ) -> dict:
        """Get planner risk adapter output.

        Returns dict with risk_delta and log_odds for the given segment side.
        """
        return {
            "risk_delta": belief_to_risk_update(belief_state, segment_side),
            "log_odds": belief_to_log_odds_update(belief_state, segment_side),
            "belief": belief_state.belief.tolist(),
            "entropy": belief_state.entropy(),
        }


# ═══════════════════════════════════════════════════════════
# Phase 1A: Unified Warning Entry Point
# ═══════════════════════════════════════════════════════════

def _risk_map_for_side(segment_side: str) -> np.ndarray:
    """Risk contribution of each hypothesis for a given segment side."""
    if segment_side == "left":
        return np.array([0.8, 0.1, 0.1, 0.5])  # [left, right, both_safe, hazard]
    else:
        return np.array([0.1, 0.8, 0.1, 0.5])


def _expected_risk(belief: np.ndarray, segment_side: str) -> float:
    """E[ρ|b] for the given segment side."""
    return float(belief @ _risk_map_for_side(segment_side))


def compute_warning_belief_delta(
    rsa_channel: RSAWarningChannel,
    belief_state: RSABeliefState,
    utterance: RSAUtterance,
    context: dict,
    segment_risky_cells: list,
    segment_side: str,
    lambda_lane_warn: float = 5.0,
    variant: str = "s1",
    tau_hat: float = 0.3,
    enable_pseudolabel: bool = False,
    feature_belief=None,
    pseudo_tau: float = 0.3,
    pseudo_weight: float = 5.0,
):
    """Unified entry point: compute WarningBeliefDelta from RSA semantics.

    ALL warning variants route through this. The variant determines the
    RSA update mode (l0, s1, s1_trust). Downstream adapters are controlled
    by flags (enable_pseudolabel) and consumed separately.

    This function MUTATES belief_state (applies the RSA update).

    Args:
        rsa_channel: RSAWarningChannel instance
        belief_state: mutable RSA belief state (will be updated)
        utterance: RSA utterance to apply
        context: branch availability dict
        segment_risky_cells: list of (row, col) for risky cells
        segment_side: "left" or "right"
        lambda_lane_warn: scaling for planner cell penalties
        variant: "l0" | "s1" | "s1_trust"
        tau_hat: trust estimate for s1_trust
        enable_pseudolabel: if True, build pseudo_label_pkg
        feature_belief: needed for pseudo-label weight computation
        pseudo_tau: matching temperature for pseudo-labels
        pseudo_weight: base weight for pseudo-label injection

    Returns:
        WarningBeliefDelta
    """
    from .warning_belief_delta import WarningBeliefDelta, PseudoLabelEntry
    from .warning_update import PROTOTYPES, PSEUDO_LABELS, Utterance

    # 1. Snapshot prior
    prior = belief_state.belief.copy()
    rho_prior = _expected_risk(prior, segment_side)

    # 2. Apply RSA belief update
    rsa_diag = rsa_channel.update_belief(
        belief_state, utterance, context,
        variant=variant, tau_hat=tau_hat,
    )

    # 3. Compute risk deltas
    posterior = belief_state.belief.copy()
    rho_post = _expected_risk(posterior, segment_side)
    rho_uniform = _expected_risk(
        np.ones(N_HYPOTHESES) / N_HYPOTHESES, segment_side)

    delta_rho_inc = round(rho_post - rho_prior, 6)
    delta_rho_uniform = round(rho_post - rho_uniform, 6)

    # 4. Build planner cell penalties
    #    Scale risk_delta by lambda_lane_warn for each risky cell
    risk_delta = belief_to_risk_update(belief_state, segment_side)
    cell_penalties = {}
    for rc in segment_risky_cells:
        cell_penalties[rc] = risk_delta * lambda_lane_warn

    # 5. Optional pseudo-label package
    pseudo_pkg = []
    if enable_pseudolabel and feature_belief is not None:
        for utt_enum in Utterance:
            proto = PROTOTYPES[utt_enum]
            y_label = PSEUDO_LABELS[utt_enum]
            # Match each risky cell to this prototype
            for r, c in segment_risky_cells:
                x_hat = feature_belief.get_mean(r, c)
                dist_sq = float(np.sum((x_hat - proto) ** 2))
                alpha = float(np.exp(-dist_sq / pseudo_tau))
                eff_w = pseudo_weight * alpha
                if eff_w > 0.01:
                    pseudo_pkg.append(PseudoLabelEntry(
                        z_proto=x_hat.copy(),
                        y_label=y_label,
                        weight=eff_w,
                    ))

    # 6. Pack diagnostics
    diagnostics = {
        "rsa_variant": variant,
        "rho_prior": round(rho_prior, 4),
        "rho_post": round(rho_post, 4),
        "risk_delta": risk_delta,
        "entropy_before": rsa_diag.get("H_before", 0),
        "entropy_after": rsa_diag.get("H_after", 0),
        "delta_H": rsa_diag.get("delta_H", 0),
    }

    return WarningBeliefDelta(
        utterance=utterance.value,
        variant=variant,
        context=context,
        prior_belief=prior,
        posterior_belief=posterior,
        delta_rho_inc=delta_rho_inc,
        delta_rho_uniform=delta_rho_uniform,
        planner_cell_penalties=cell_penalties,
        pseudo_label_pkg=pseudo_pkg,
        diagnostics=diagnostics,
    )

