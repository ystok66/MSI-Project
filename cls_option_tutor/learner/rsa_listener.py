# rsa_listener.py - L1 RSA Listener (LEGACY / UNSUPPORTED)
#
# STATUS: ARCHIVED PATH - not used by the J-based option-level tutor.
#   The J-tutor mainline runs with use_rsa=False (default).
#   Retained to avoid ImportError when use_rsa=True is set.
#   See archive/2026-04-12/ for the full history.
#   DO NOT extend this module. See option_level_tutor.py + g_learn.py.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class RSABeliefUpdate:
    """Output of one RSA L1 inference step.

    All arrays are shape [K] where K = len(active_menu).

    semantic_log_bias[j] = log P_S0(action | j) - log_Z
        Represents the log-likelihood ratio update for semantic posterior.
        Added to S_CLS(j) in pick utility: U_RSA(j) += α_sem * b_sem(j)

    risk_logit_shift[j] = Δlogit P(r_j=1 | action)
        Represents the logit-space update for risk posterior.
        For BAN(j_b): shift[j_b] = +ω_ban, all others = 0.

    pass_abort: True if tutor issued PASS (query should abort).
    """
    semantic_log_bias: np.ndarray   # (K,) float
    risk_logit_shift: np.ndarray    # (K,) float
    pass_abort: bool = False


class RSAListener:
    """L1 RSA listener — converts tutor actions to learner posterior updates.

    Design principles:
      1. Stateless per call (no internal mutable state)
      2. HIGHLIGHT carries only semantic information
      3. BAN carries only risk information
      4. PASS = abort (no update)
      5. WAIT / unknown = identity (no update)

    The listener uses DeterministicSemanticScorer for per-cell mismatch
    computation (needed for HIGHLIGHT semantic likelihood).
    """

    def __init__(
        self,
        omega_hl: float = 2.0,
        lambda_ctx: float = 0.5,
        omega_ban: float = 3.0,
    ):
        """
        Args:
            omega_hl: HIGHLIGHT log-likelihood strength ω_hl
            lambda_ctx: contrastive weight for non-highlighted cells λ_ctx
            omega_ban: BAN logit shift strength ω_ban
        """
        self.omega_hl = omega_hl
        self.lambda_ctx = lambda_ctx
        self.omega_ban = omega_ban

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def compute_sem_gate(
        self,
        q_prior: np.ndarray,
        gate_type: str = "entropy",
        gate_lo: float = 0.25,
        gate_hi: float = 0.90,
    ) -> float:
        """Compute semantic gate g ∈ [0,1] from pre-RSA base posterior q_t^(0).

        q_prior must be the BASE decision distribution — i.e. computed from
        CLS + risk + uncertainty channels BEFORE any RSA bias is applied.
        Do NOT pass raw sem_scores; those ignore risk/unc channels.

        Canonical "entropy" mode: U-shape gate
            h = H(q_prior) / log(K)     [normalized entropy, ∈ [0,1]]
            g = 4 * h * (1 - h)         [U-parabola: 0 at extremes, 1 at h=0.5]

        Effect:
            h ≈ 0  → learner already confident → g ≈ 0 (don't disturb)
            h ≈ 1  → learner totally lost → g ≈ 0 (RSA signal unreliable)
            h ≈ 0.5 → learner genuinely uncertain → g ≈ 1 (best time to teach)

        Research fallback "threshold" mode:
            g = 1 if gate_lo <= h <= gate_hi else 0

        Args:
            q_prior: (K,) pre-RSA base probability distribution (normalized)
            gate_type: "entropy" (canonical) | "threshold" | "none"
            gate_lo: lower entropy threshold (threshold mode only)
            gate_hi: upper entropy threshold (threshold mode only)

        Returns:
            g ∈ [0.0, 1.0]
        """
        if gate_type == "none":
            return 1.0

        K = len(q_prior)
        if K <= 1:
            return 1.0

        # Normalize safely
        q = np.array(q_prior, dtype=float)
        q = np.clip(q, 1e-10, None)
        q = q / q.sum()

        # Normalized entropy h ∈ [0, 1]
        log_K = np.log(K)
        h = -float(np.sum(q * np.log(q))) / log_K
        h = float(np.clip(h, 0.0, 1.0))

        if gate_type == "entropy":
            # U-shape: g = 4*h*(1-h)
            return float(4.0 * h * (1.0 - h))
        elif gate_type == "threshold":
            return 1.0 if gate_lo <= h <= gate_hi else 0.0
        else:
            return 1.0

    def observe_tutor_action(
        self,
        action: str,
        target_output: List[str],
        active_texts: List[List[str]],    # (K,) option programs
        rendered_outputs: List[Optional[List[str]]],  # (K,) F_G(text) or None
        action_arg: Optional[int] = None,   # j_b for BAN
        action_cells: Optional[Tuple[int, ...]] = None,  # H for HIGHLIGHT
        sem_gate: float = 1.0,   # pre-computed gate value from compute_sem_gate()
    ) -> RSABeliefUpdate:
        """Compute L1 RSA posterior update from a tutor action.

        Args:
            action: "WAIT" | "BAN" | "HIGHLIGHT" | "PASS" | "RISK_HINT" | "SKIP"
            target_output: Y* — target output cells (L,)
            active_texts: list of K option programs (text tokens)
            rendered_outputs: list of K rendered F_G(text_j) or None
            action_arg: for BAN, the active-menu index j_b (0-indexed in active menu)
            action_cells: for HIGHLIGHT, tuple of cell indices H
            sem_gate: g ∈ [0,1] from compute_sem_gate(q_prior).
                      Scales the semantic log-bias: b_sem_gated = g * b_sem.
                      Caller is responsible for computing q_prior from base utility
                      (CLS + risk + unc) BEFORE RSA bias is applied.

        Returns:
            RSABeliefUpdate with semantic_log_bias, risk_logit_shift, pass_abort
        """
        K = len(active_texts)
        identity = RSABeliefUpdate(
            semantic_log_bias=np.zeros(K),
            risk_logit_shift=np.zeros(K),
            pass_abort=False,
        )

        if action == "HIGHLIGHT" and action_cells is not None and len(action_cells) > 0:
            return self._highlight_update(
                target_output, active_texts, rendered_outputs, action_cells,
                sem_gate=sem_gate)

        elif action == "BAN" and action_arg is not None:
            return self._ban_update(K, action_arg)

        elif action == "PASS":
            return RSABeliefUpdate(
                semantic_log_bias=np.zeros(K),
                risk_logit_shift=np.zeros(K),
                pass_abort=True,
            )

        else:
            # WAIT, RISK_HINT (legacy), SKIP, unknown → identity
            return identity

    # ─────────────────────────────────────────────────────────────────────
    # HIGHLIGHT: semantic RSA update
    # ─────────────────────────────────────────────────────────────────────

    def _highlight_update(
        self,
        target_output: List[str],
        active_texts: List[List[str]],
        rendered_outputs: List[Optional[List[str]]],
        cells: Tuple[int, ...],
        sem_gate: float = 1.0,
    ) -> RSABeliefUpdate:
        """Compute semantic log-bias from HIGHLIGHT(H).

        For each option j:
            M_H(j)    = fraction of highlighted cells that mismatch
            M_barH(j) = fraction of non-highlighted cells that mismatch
            s_HL(j)   = -M_H(j) + λ_ctx * M_barH(j)
            log P_S0(HL | j) = ω_hl * s_HL(j)

        Z-normalize so bias is centered (sum of biases ≈ 0 across K).
        Then apply semantic gate: b_sem_gated = sem_gate * b_sem

        sem_gate = g = 4*h*(1-h) where h = H(q_t^0)/log(K).
        When g=0: full suppression (learner too confident or too lost).
        When g=1: full RSA signal.
        """
        K = len(active_texts)
        L = len(target_output)
        cell_set = set(c for c in cells if 0 <= c < L)
        non_cell_set = set(range(L)) - cell_set

        s_hl = np.zeros(K)

        for j, (text, rendered) in enumerate(zip(active_texts, rendered_outputs)):
            if rendered is None or len(rendered) != L:
                s_hl[j] = -1.0 + self.lambda_ctx * 0.0
                continue

            # M_H(j): mismatch on highlighted cells
            if len(cell_set) > 0:
                mismatch_H = sum(
                    1 for c in cell_set
                    if rendered[c] != target_output[c]
                ) / len(cell_set)
            else:
                mismatch_H = 0.0

            # M_barH(j): mismatch on non-highlighted cells
            if len(non_cell_set) > 0:
                mismatch_barH = sum(
                    1 for c in non_cell_set
                    if rendered[c] != target_output[c]
                ) / len(non_cell_set)
            else:
                mismatch_barH = 0.0

            # s_HL(j) = -M_H(j) + λ_ctx * M_barH(j)
            s_hl[j] = -mismatch_H + self.lambda_ctx * mismatch_barH

        # Log-likelihood: log P_S0(HL | j) = ω_hl * s_HL(j)
        log_liks = self.omega_hl * s_hl

        # Normalize: log_bias[j] = log P_S0 - log Z (log-softmax style)
        log_Z = np.log(np.sum(np.exp(log_liks - np.max(log_liks))) + 1e-10) + np.max(log_liks)
        semantic_log_bias = log_liks - log_Z

        # Apply U-shape entropy gate: b_sem_gated = g * b_sem
        # sem_gate = 0 → suppress; sem_gate = 1 → full signal
        semantic_log_bias = sem_gate * semantic_log_bias

        return RSABeliefUpdate(
            semantic_log_bias=semantic_log_bias,
            risk_logit_shift=np.zeros(K),
            pass_abort=False,
        )

    # ─────────────────────────────────────────────────────────────────────
    # BAN: risk RSA update
    # ─────────────────────────────────────────────────────────────────────

    def _ban_update(self, K: int, ban_active_idx: int) -> RSABeliefUpdate:
        """Compute risk logit shift from BAN(j_b).

        For the banned option j_b:
            logit P(r_{j_b}=1 | BAN) = logit P(r_{j_b}=1) + ω_ban

        For all other options: no update.

        The shift is applied by the caller (LearnerAgent) to the risk
        posterior and then reflected in pick utility as:
            μ_d_tilde(j_b) = p_risk_post(j_b) * mu_s(j_b)
        """
        risk_logit_shift = np.zeros(K)
        if 0 <= ban_active_idx < K:
            risk_logit_shift[ban_active_idx] = self.omega_ban

        return RSABeliefUpdate(
            semantic_log_bias=np.zeros(K),
            risk_logit_shift=risk_logit_shift,
            pass_abort=False,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Utility: apply logit shift to probability
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def apply_logit_shift(p: float, delta_logit: float) -> float:
        """Shift a probability in logit space.

        p_new = σ(logit(p) + delta_logit)

        Args:
            p: prior probability ∈ (0, 1)
            delta_logit: logit-space shift (positive = more risky)

        Returns:
            posterior probability ∈ (0, 1)
        """
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        logit_p = np.log(p / (1 - p))
        logit_post = logit_p + delta_logit
        return float(1.0 / (1.0 + np.exp(-logit_post)))
