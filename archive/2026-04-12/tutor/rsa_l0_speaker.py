"""
rsa_l0_speaker.py — L0 Speaker utility computation for tutor action selection.

Computes U_S0(a) = λ_task * G_task(a) + λ_teach * G_teach(a)

Used exclusively by TutorPolicy.select_action_l0() in single-thread L0 mode.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def compute_l0_utility(
    action: str,
    qs,                             # QueryState
    snap,                           # LearnerStateSnapshot
    active: list,                   # list of Option
    rsa_cfg,                        # RSAConfig
    scorer,                         # DeterministicSemanticScorer
    action_arg: Optional[int] = None,      # j for BAN (active-menu index)
    action_cells: Optional[Tuple] = None,  # H for HIGHLIGHT
) -> float:
    """Compute L0 speaker utility U_S0 for a given action.

    Args:
        action: "HIGHLIGHT" | "BAN" | "WAIT" | "PASS"
        qs: current QueryState (read-only)
        snap: LearnerStateSnapshot from current learner state
        active: active menu options
        rsa_cfg: RSAConfig
        scorer: DeterministicSemanticScorer for mismatch computation
        action_arg: for BAN, the active-menu index j (0-indexed)
        action_cells: for HIGHLIGHT, tuple of cell indices H

    Returns:
        U_S0 ≥ 0 (WAIT has U=0 baseline)
    """
    if action == "WAIT":
        return 0.0

    if action == "HIGHLIGHT":
        return _u_highlight(action_cells, qs, snap, active, rsa_cfg, scorer)

    if action == "BAN":
        return _u_ban(action_arg, qs, snap, active, rsa_cfg)

    if action == "PASS":
        return 0.0  # PASS driven by task/safety, not teach gain; return 0 here

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HIGHLIGHT utility
# ─────────────────────────────────────────────────────────────────────────────

def _u_highlight(cells, qs, snap, active, rsa_cfg, scorer) -> float:
    """U_S0(HIGHLIGHT(H)).

    G_task^HL = ΔP_corr(HL) — improvement in P(correct pick | highlight)
        = Σ_j 1[j is correct] * (q_post(j) - q_pre(j))

    G_teach^HL = log q_post(j*) - log q_pre(j*)
        where j* = correct option index in active menu

    U_S0 = λ_task * G_task + λ_teach * G_teach
    """
    if cells is None or len(cells) == 0:
        return 0.0

    K = len(active)
    L = len(qs.target_output)
    cell_set = set(c for c in cells if 0 <= c < L)
    non_cell_set = set(range(L)) - cell_set

    # Compute RSA semantic log-bias for this HIGHLIGHT
    log_liks = np.zeros(K)
    for j, opt in enumerate(active):
        rendered = scorer.predict_output(opt.text) if scorer else None
        if rendered is None or len(rendered) != L:
            log_liks[j] = -rsa_cfg.omega_hl  # max mismatch
            continue

        mismatch_H = (sum(1 for c in cell_set if rendered[c] != qs.target_output[c])
                      / max(len(cell_set), 1))
        mismatch_barH = (sum(1 for c in non_cell_set if rendered[c] != qs.target_output[c])
                         / max(len(non_cell_set), 1)) if non_cell_set else 0.0

        s_hl = -mismatch_H + rsa_cfg.lambda_ctx * mismatch_barH
        log_liks[j] = rsa_cfg.omega_hl * s_hl

    # Log-normalize: log P_S0(HL | j) - log Z
    log_Z = np.log(np.sum(np.exp(log_liks - np.max(log_liks))) + 1e-10) + np.max(log_liks)
    log_bias = log_liks - log_Z

    # Pre-highlight semantic scores
    sem_pre = np.array(snap.semantic_scores)
    sem_post = sem_pre + log_bias

    # Convert to probabilities via softmax (temperature=1 for ratio comparison)
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)

    q_pre = _softmax(sem_pre)
    q_post = _softmax(sem_post)

    # Find correct option in active menu
    correct_idx = None
    for j, opt in enumerate(active):
        if opt.is_correct:
            correct_idx = j
            break

    # G_task: ΔP_corr
    g_task = 0.0
    if correct_idx is not None:
        g_task = float(q_post[correct_idx] - q_pre[correct_idx])

    # G_teach: log q_post(j*) - log q_pre(j*)
    g_teach = 0.0
    if correct_idx is not None:
        log_q_post = float(np.log(q_post[correct_idx] + 1e-10))
        log_q_pre  = float(np.log(q_pre[correct_idx] + 1e-10))
        g_teach = log_q_post - log_q_pre

    u = rsa_cfg.lambda_task * g_task + rsa_cfg.lambda_teach * g_teach
    return max(u, 0.0)  # WAIT baseline = 0


# ─────────────────────────────────────────────────────────────────────────────
# BAN utility
# ─────────────────────────────────────────────────────────────────────────────

def _u_ban(ban_active_idx: int, qs, snap, active, rsa_cfg) -> float:
    """U_S0(BAN(j_b)).

    G_task^BAN = ΔP_corr — increase in P(learner picks correct option)
        After BAN(j_b): j_b is removed from active menu → learner redistributes

    G_teach^BAN = Δlogit P(r_{j_b}=1) * P_L(j_b)
        = ω_ban * pick_prob(j_b)
        Weighted by learner's pick probability (BAN that which learner would pick)

    U_S0 = λ_task * G_task + λ_teach * G_teach - c_ban
    """
    if ban_active_idx is None or ban_active_idx < 0 or ban_active_idx >= len(active):
        return 0.0

    K = len(active)
    pick_probs = np.array(snap.pick_probs)  # (K,)
    p_ban = float(pick_probs[ban_active_idx])

    # G_teach: ω_ban * P_L(j_b)
    # More valuable to BAN an option learner actually considers picking
    g_teach = float(rsa_cfg.omega_ban * p_ban)

    # G_task: ΔP_corr
    # If j_b is risky but not correct, removing it redistributes probability to correct option
    # Simple estimate: ΔP_corr ≈ P_L(j_b) * P_L(j*) / (1 - P_L(j_b))
    correct_idx = next((j for j, o in enumerate(active) if o.is_correct), None)
    g_task = 0.0
    if correct_idx is not None and ban_active_idx != correct_idx:
        p_corr = float(pick_probs[correct_idx])
        remaining_mass = max(1 - p_ban, 1e-6)
        g_task = p_ban * (p_corr / remaining_mass)

    u = rsa_cfg.lambda_task * g_task + rsa_cfg.lambda_teach * g_teach
    # ban_parametric_penalty: constant that discourages BAN-heavy teaching.
    # BAN produces high in-context utility (removes bad options) but low
    # parametric learning transfer (eval phase doesn't benefit). Penalty
    # shifts BAN utility down so HIGHLIGHT competes more fairly.
    # Default: 0.0 (no penalty). Exp G3c uses -0.3. Set -inf to fully disable BAN.
    u = u + rsa_cfg.ban_parametric_penalty
    return max(u, 0.0)
