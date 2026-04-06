"""Posterior p_self Shadow — Bayesian self-discovery probability estimation.

Replaces the geometric p_self with posterior / posterior-predictive versions.

═══ FORMAL EVENT DEFINITIONS (Step 5 Contract) ═══

All variants output a TERNARY tuple: (p_self, p_fail, p_undecided)
with p_self + p_fail + p_undecided = 1.

  p_self:      Pr(learner discovers correct branch under do(WAIT),
               before commit/failure window closes)
  p_fail:      Pr(learner commits to wrong branch or times out
               under do(WAIT), irreversible error)
  p_undecided: Pr(learner remains uncommitted at window end,
               neither discovered nor failed — still observing)
               = 1 − p_self − p_fail

Variants:
  A: Posterior Fusion (minimal)
  B: Posterior Predictive (single-point, falls back to A if no branches)
  C: Three-Outcome Model (recommended default for v2.1)

The recommended default for micro_bayes_shadow_v2.1 is POSTERIOR_C.
All other variants are retained for ablation/diagnostics.

Shadow-only. Does NOT modify any frozen module.
3D-consumption only: (τ̂, ν̂, γ̂_gen). No κ̂ / γ̂_spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Dict
import numpy as np

from ..metrics.self_discovery import estimate_self_discovery_prob, _sigmoid


class PSelfMode(Enum):
    """p_self computation mode."""
    BASELINE = "baseline"            # geometric sigmoid (current canonical)
    OLD_BLEND = "old_blend"          # Phase 7 belief-horizon blend
    POSTERIOR_A = "posterior_A"       # Variant A: posterior fusion
    POSTERIOR_B = "posterior_B"       # Variant B: posterior predictive (default)
    POSTERIOR_C = "posterior_C"       # Variant C: three-outcome model


# ═══════════════════════════════════════════════════════════
# Variant A: Posterior Fusion
# ═══════════════════════════════════════════════════════════

def _compute_cognitive_prior(tau_hat: float, nu_hat: float,
                              gamma_gen_hat: float,
                              info_window: float) -> float:
    """Compute p_prior from observer estimates — agent's cognitive capacity.

    Higher τ̂ (trust useful warnings → also trusts own evidence) → higher prior.
    Higher ν̂ (dependent on tutor) → lower prior.
    Higher γ̂_gen (exploration suppressed) → lower prior.
    Wider info window → more time to discover → higher prior.

    3D-consumption only: does NOT use κ̂ or γ̂_spec.
    """
    # Autonomy = capacity for independent learning
    autonomy = max(1.0 - nu_hat, 0.05) * max(1.0 - gamma_gen_hat, 0.1)
    # Evidence sensitivity = can agent recognize the signal?
    evidence_sensitivity = 0.3 + 0.7 * tau_hat
    # Temporal = does the geometry give enough time?
    temporal = min(info_window, 1.0)

    p_prior = autonomy * evidence_sensitivity * temporal
    return float(np.clip(p_prior, 0.01, 0.99))


def compute_p_self_posterior_A(
    d_commit: int,
    d_reveal: int,
    tau_hat: float,
    nu_hat: float,
    gamma_gen_hat: float,
    margin: float = 0.0,
    tau_v: float = 1.0,
) -> Dict[str, float]:
    """Variant A: Posterior fusion of geometric likelihood and cognitive prior.

    p_self^post = (p_geom · p_prior) / (p_geom · p_prior + (1-p_geom)(1-p_prior))

    This is the minimal Bayesian upgrade: odds multiplication.

    Returns dict with p_self, p_fail, p_geom, p_prior (for diagnostics).
    """
    p_geom = estimate_self_discovery_prob(d_commit, d_reveal, margin, tau_v)
    info_window = max(d_commit - d_reveal, 0) / max(d_commit, 1)
    p_prior = _compute_cognitive_prior(tau_hat, nu_hat, gamma_gen_hat, info_window)

    # Posterior fusion (product of odds)
    numer = p_geom * p_prior
    denom = numer + (1.0 - p_geom) * (1.0 - p_prior)
    p_self = numer / max(denom, 1e-10)

    # p_fail: independent estimate (not complement)
    # Higher when d_commit < d_reveal AND agent is dependent
    fail_geometric = 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=1.5)
    fail_cognitive = nu_hat * (1.0 - tau_hat)
    p_fail = min(fail_geometric + 0.3 * fail_cognitive, 1.0 - p_self)

    # p_undecided: remainder (ternary API closure)
    p_undecided = max(1.0 - p_self - p_fail, 0.0)

    return {
        "p_self": round(float(p_self), 4),
        "p_fail": round(float(p_fail), 4),
        "p_undecided": round(float(p_undecided), 4),
        "p_geom": round(float(p_geom), 4),
        "p_prior": round(float(p_prior), 4),
        "variant": "A",
        "fallback": False,
    }


# ═══════════════════════════════════════════════════════════
# Variant B: Posterior Predictive (single-point)
# ═══════════════════════════════════════════════════════════

def compute_p_self_posterior_B(
    d_commit: int,
    d_reveal: int,
    tau_hat: float,
    nu_hat: float,
    gamma_gen_hat: float,
    branches=None,
    agent_params=None,
    margin: float = 0.0,
    tau_v: float = 1.0,
    obs_depth: int = 0,
) -> Dict[str, float]:
    """Variant B: Posterior predictive self-discovery probability.

    p_self = P(discover before commit | b_t, do(WAIT))

    Single-point approximation: evaluates the agent's probability of
    choosing the safe branch AT the decision point, given that it has had
    time to observe cues (under do(WAIT), i.e., no tutor intervention).

    Uses bounded-rational action model: softmax(β·U) + lapse, conditioned
    on current belief state and available cue visibility.

    If branches/agent_params are not provided, falls back to Variant A.
    """
    p_geom = estimate_self_discovery_prob(d_commit, d_reveal, margin, tau_v)
    info_window = max(d_commit - d_reveal, 0) / max(d_commit, 1)
    p_prior = _compute_cognitive_prior(tau_hat, nu_hat, gamma_gen_hat, info_window)

    # If no branches available, fall back to variant A
    if branches is None or agent_params is None or len(branches) < 2:
        result = compute_p_self_posterior_A(
            d_commit, d_reveal, tau_hat, nu_hat, gamma_gen_hat, margin, tau_v)
        result["variant"] = "B_fallback"
        result["fallback"] = True
        return result

    # === Single-point posterior predictive ===
    # Under do(WAIT): agent observes cues for (d_commit - d_reveal) steps.
    # At decision point, how likely is agent to choose safe branch?

    from ..agents.stochastic_agent_policy import compute_choice_probs

    # Compute agent's choice probability given current branches
    # Use "safe" theta as the truth model (agent is trying to be safe)
    probs_safe = compute_choice_probs(branches, "safe", agent_params)
    probs_shiny = compute_choice_probs(branches, "shiny", agent_params)

    # Posterior-weighted: how aligned is agent choice with safety?
    # With no tutor: agent relies on own cues.
    # Cue visibility modulates how much branch info is available
    cue_visible = float(obs_depth >= d_reveal) if d_reveal > 0 else 1.0
    cue_partial = min(max(obs_depth - d_reveal + 1, 0) / max(d_commit - d_reveal + 1, 1), 1.0)

    # p_safe_given_cues: agent chooses correctly IF cues are visible
    p_safe_if_cues = float(probs_safe[0])  # branch 0 = safe typically

    # p_safe_if_blind: depends on prior bias
    # Without cues, agent falls back to preference priors
    p_safe_if_blind = float(probs_shiny[0])  # more uncertain

    # Weighted average by cue visibility
    p_correct_choice = cue_partial * p_safe_if_cues + (1.0 - cue_partial) * p_safe_if_blind

    # Autonomy modulation: high ν̂ → even with correct information, agent
    # may still follow old tutor patterns instead of own judgment
    autonomy_factor = max(1.0 - nu_hat * 0.8, 0.2)

    # Exploration suppression: high γ̂_gen → agent avoids novel paths
    exploration_factor = max(1.0 - gamma_gen_hat * 0.5, 0.5)

    p_self = p_correct_choice * autonomy_factor * exploration_factor

    # Independent p_fail estimate
    # Failure = commits to wrong branch before cues become available
    p_commit_blind = 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=1.5)
    p_wrong_if_blind = 1.0 - p_safe_if_blind
    p_fail = p_commit_blind * p_wrong_if_blind * (1.0 + nu_hat * 0.3)
    p_fail = min(float(p_fail), 1.0 - p_self)
    p_undecided = max(1.0 - float(np.clip(p_self, 0, 1)) - float(np.clip(p_fail, 0, 1)), 0.0)

    return {
        "p_self": round(float(np.clip(p_self, 0.0, 1.0)), 4),
        "p_fail": round(float(np.clip(p_fail, 0.0, 1.0)), 4),
        "p_undecided": round(float(p_undecided), 4),
        "p_geom": round(float(p_geom), 4),
        "p_prior": round(float(p_prior), 4),
        "p_correct_choice": round(float(p_correct_choice), 4),
        "cue_partial": round(float(cue_partial), 4),
        "autonomy_factor": round(float(autonomy_factor), 4),
        "variant": "B",
        "fallback": False,
    }


# ═══════════════════════════════════════════════════════════
# Variant C: Three-Outcome Model
# ═══════════════════════════════════════════════════════════

def compute_p_self_posterior_C(
    d_commit: int,
    d_reveal: int,
    tau_hat: float,
    nu_hat: float,
    gamma_gen_hat: float,
    margin: float = 0.0,
    tau_v: float = 1.0,
) -> Dict[str, float]:
    """Variant C: Three-outcome model.

    p_self + p_fail + p_undecided = 1

    Does NOT assume p_fail = 1 - p_self. Instead:
    - p_self: agent discovers cue AND makes correct choice
    - p_fail: agent commits to wrong branch (irreversible error)
    - p_undecided: agent enters low-information continuation (neither
      discovered nor failed — still walking, hasn't committed)

    p_undecided serves as a calibration diagnostic: if baseline's
    p_fail = 1 - p_self is correct, p_undecided should be ≈ 0.
    """
    p_geom = estimate_self_discovery_prob(d_commit, d_reveal, margin, tau_v)
    info_window = max(d_commit - d_reveal, 0) / max(d_commit, 1)
    p_prior = _compute_cognitive_prior(tau_hat, nu_hat, gamma_gen_hat, info_window)

    # Posterior fusion p_self (same as Variant A)
    numer = p_geom * p_prior
    denom = numer + (1.0 - p_geom) * (1.0 - p_prior)
    p_self = numer / max(denom, 1e-10)

    # For three-outcome: we need to partition 1 - p_self into fail vs undecided.
    # p_fail: probability of irreversible BAD commitment
    #   High when: d_commit ≪ d_reveal (blind commitment zone is short)
    #              AND agent is poor at self-correction (low τ̂)
    commit_urgency = 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=2.0)
    error_prone = (1.0 - tau_hat) * 0.6 + nu_hat * 0.4
    p_fail_raw = commit_urgency * error_prone

    # p_undecided: agent continues without committing (explores slowly)
    #   High when: d_commit is large relative to d_reveal
    #              → agent is still in the uncommitted zone
    exploration_inertia = gamma_gen_hat * 0.4 + (1.0 - tau_hat) * 0.3
    p_undecided_raw = (1.0 - commit_urgency) * exploration_inertia

    # Normalize: p_fail + p_undecided = 1 - p_self
    remainder = max(1.0 - p_self, 0.0)
    raw_total = p_fail_raw + p_undecided_raw + 1e-10
    p_fail = remainder * (p_fail_raw / raw_total)
    p_undecided = remainder * (p_undecided_raw / raw_total)

    # Ensure remaining mass goes to the larger of the two
    leftover = remainder - p_fail - p_undecided
    if leftover > 0.01:
        p_fail += leftover * (p_fail_raw / raw_total)
        p_undecided += leftover * (p_undecided_raw / raw_total)

    return {
        "p_self": round(float(np.clip(p_self, 0.0, 1.0)), 4),
        "p_fail": round(float(np.clip(p_fail, 0.0, 1.0)), 4),
        "p_undecided": round(float(np.clip(p_undecided, 0.0, 1.0)), 4),
        "p_geom": round(float(p_geom), 4),
        "p_prior": round(float(p_prior), 4),
        "variant": "C",
        "fallback": False,
    }


# ═══════════════════════════════════════════════════════════
# Unified dispatcher
# ═══════════════════════════════════════════════════════════

def compute_p_self_posterior(
    mode: PSelfMode,
    d_commit: int,
    d_reveal: int,
    tau_hat: float = 0.3,
    nu_hat: float = 0.1,
    gamma_gen_hat: float = 0.0,
    margin: float = 0.0,
    tau_v: float = 1.0,
    branches=None,
    agent_params=None,
    obs_depth: int = 0,
) -> Dict[str, float]:
    """Unified dispatcher for all p_self variants.

    Contract (Step 5):
      ALWAYS returns dict with at minimum:
        {p_self, p_fail, p_undecided, variant, fallback}
      where p_self + p_fail + p_undecided = 1.

    Recommended default: POSTERIOR_C (three-outcome model).
    """
    if mode == PSelfMode.BASELINE:
        p = estimate_self_discovery_prob(d_commit, d_reveal, margin, tau_v)
        p_f = 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=1.5)
        p_f = min(p_f, 1.0 - p)
        p_u = max(1.0 - p - p_f, 0.0)
        return {
            "p_self": round(float(p), 4),
            "p_fail": round(float(p_f), 4),
            "p_undecided": round(float(p_u), 4),
            "variant": "baseline",
            "fallback": False,
        }

    elif mode == PSelfMode.OLD_BLEND:
        p_geom = estimate_self_discovery_prob(d_commit, d_reveal, margin, tau_v)
        risk_aware = min(2.0, 1.0)
        update_gain = max(1.0 - nu_hat, 0.1)
        info_window = max(d_commit - d_reveal, 0) / max(d_commit, 1)
        p_belief = risk_aware * update_gain * info_window
        eta = 0.5
        p = (1.0 - eta) * p_geom + eta * min(p_belief, 1.0)
        p_f = 1.0 - estimate_self_discovery_prob(d_commit, d_reveal, tau_v=1.5)
        p_f = min(p_f, 1.0 - p)
        p_u = max(1.0 - p - p_f, 0.0)
        return {
            "p_self": round(float(p), 4),
            "p_fail": round(float(p_f), 4),
            "p_undecided": round(float(p_u), 4),
            "p_geom": round(float(p_geom), 4),
            "variant": "old_blend",
            "fallback": False,
        }

    elif mode == PSelfMode.POSTERIOR_A:
        return compute_p_self_posterior_A(
            d_commit, d_reveal, tau_hat, nu_hat, gamma_gen_hat, margin, tau_v)

    elif mode == PSelfMode.POSTERIOR_B:
        return compute_p_self_posterior_B(
            d_commit, d_reveal, tau_hat, nu_hat, gamma_gen_hat,
            branches, agent_params, margin, tau_v, obs_depth)

    elif mode == PSelfMode.POSTERIOR_C:
        return compute_p_self_posterior_C(
            d_commit, d_reveal, tau_hat, nu_hat, gamma_gen_hat, margin, tau_v)

    else:
        raise ValueError(f"Unknown PSelfMode: {mode}")
