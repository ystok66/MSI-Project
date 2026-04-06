"""V5.4 — RSA Warning v2.

Strict Literal Listener → Speaker → Pragmatic Listener pipeline.
Warning as semantic evidence factor, not path bias.

Utterance set: {warn_left, warn_right, warn_front, silence}
World states:  {left_risky, right_risky, front_risky, ambiguous}

L0:   P(z|u)  ∝ exp(sem(u,z))         — pure semantics, no cost
S1:   P(u|z)  ∝ exp(α·log P_L0(z|u) - λ·|u| - c(u))
L1:   P(z|u)  ∝ P(z) · P_S1(u|z)     — Bayesian pragmatic listener
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── World states and utterances ──────────────────────────────────
WORLD_STATES = ["left_risky", "right_risky", "front_risky", "ambiguous"]
UTTERANCES = ["warn_left", "warn_right", "warn_front", "silence"]

N_STATES = len(WORLD_STATES)
N_UTTERANCES = len(UTTERANCES)

# State indices
Z_LEFT = 0
Z_RIGHT = 1
Z_FRONT = 2
Z_AMBIG = 3

# Utterance indices
U_LEFT = 0
U_RIGHT = 1
U_FRONT = 2
U_SILENCE = 3


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class RSAConfig:
    """Configuration for RSA warning system."""
    alpha: float = 2.0       # speaker rationality
    lambda_len: float = 0.1  # utterance length cost
    eta: float = 3.0         # evidence factor strength
    prior_ambig: float = 0.3 # prior probability of ambiguous state


class RSAWarningV2:
    """RSA-based warning system with strict L0→S1→L1 layering.

    The warning is treated as semantic evidence, not a path bias:
    it updates the agent's belief about world state via L1 posterior.
    """

    def __init__(self, config: Optional[RSAConfig] = None):
        self.config = config or RSAConfig()
        self._build_sem_matrix()

    def _build_sem_matrix(self):
        """Build semantic compatibility matrix sem[u, z].

        sem(warn_left, left_risky)  = +3  (truthful, natural)
        sem(warn_left, right_risky) = -2  (contradictory)
        sem(warn_left, front_risky) = -1  (irrelevant)
        sem(warn_left, ambiguous)   = +0.5 (partially valid)
        sem(silence, *)             = 0   (uninformative)
        """
        S = np.zeros((N_UTTERANCES, N_STATES))

        # warn_left: strongly associated with left_risky
        S[U_LEFT, Z_LEFT]   = +3.0
        S[U_LEFT, Z_RIGHT]  = -2.0
        S[U_LEFT, Z_FRONT]  = -1.0
        S[U_LEFT, Z_AMBIG]  = +0.5

        # warn_right: strongly associated with right_risky
        S[U_RIGHT, Z_RIGHT] = +3.0
        S[U_RIGHT, Z_LEFT]  = -2.0
        S[U_RIGHT, Z_FRONT] = -1.0
        S[U_RIGHT, Z_AMBIG] = +0.5

        # warn_front: strongly associated with front_risky
        S[U_FRONT, Z_FRONT] = +3.0
        S[U_FRONT, Z_LEFT]  = -1.0
        S[U_FRONT, Z_RIGHT] = -1.0
        S[U_FRONT, Z_AMBIG] = +0.5

        # silence: uninformative (uniform)
        S[U_SILENCE, :] = 0.0

        self.sem = S

    def literal_listener(self) -> np.ndarray:
        """P_L0(z|u): pure semantic interpretation, no utterance cost.

        Returns (N_UTTERANCES, N_STATES) matrix.
        Each row is a probability distribution over states.
        """
        return _softmax(self.sem, axis=1)

    def speaker(self) -> np.ndarray:
        """P_S1(u|z): rational speaker.

        U(u,z) = α·log P_L0(z|u) - λ·|u| - c(u)

        Returns (N_STATES, N_UTTERANCES) matrix.
        Each row is a probability distribution over utterances.
        """
        L0 = self.literal_listener()  # (U, Z)
        alpha = self.config.alpha
        lam = self.config.lambda_len

        # Utterance costs: silence is free, others have length cost
        costs = np.array([lam, lam, lam, 0.0])

        # Utility: U[z, u] = α·log P_L0(z|u) - c(u)
        U = np.zeros((N_STATES, N_UTTERANCES))
        for z in range(N_STATES):
            for u in range(N_UTTERANCES):
                log_l0 = np.log(max(L0[u, z], 1e-10))
                U[z, u] = alpha * log_l0 - costs[u]

        return _softmax(U, axis=1)

    def pragmatic_listener(self, prior: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        """P_L1(z|u): pragmatic listener.

        P_L1(z|u) ∝ P(z) · P_S1(u|z)

        Returns (N_UTTERANCES, N_STATES) matrix.
        """
        if prior is None:
            # Default prior: slight ambiguity
            p_ambig = self.config.prior_ambig
            p_each = (1 - p_ambig) / 3
            prior = np.array([p_each, p_each, p_each, p_ambig])

        S1 = self.speaker()  # (Z, U)

        L1 = np.zeros((N_UTTERANCES, N_STATES))
        for u in range(N_UTTERANCES):
            for z in range(N_STATES):
                L1[u, z] = prior[z] * S1[z, u]
            row_sum = L1[u].sum()
            if row_sum > 0:
                L1[u] /= row_sum
            else:
                L1[u] = prior

        return L1

    def choose_utterance(self, true_state_idx: int) -> tuple[int, str]:
        """Teacher chooses optimal utterance given true world state.

        Returns (utterance_idx, utterance_name).
        """
        S1 = self.speaker()
        u_idx = int(np.argmax(S1[true_state_idx]))
        return u_idx, UTTERANCES[u_idx]

    def classify_risk_state(self, branch_a_risk: float, branch_b_risk: float,
                            branch_a_side: str = "left",
                            front_risk: float = 0.0,
                            gap_threshold: float = 0.05) -> int:
        """Map continuous risk predictions to discrete world state.

        Parameters
        ----------
        branch_a_risk, branch_b_risk : predicted mean risk for each branch
        branch_a_side : 'left' or 'right' (which physical side is branch A)
        front_risk : risk of going straight
        gap_threshold : minimum risk gap to classify as non-ambiguous

        Returns
        -------
        state_idx : index into WORLD_STATES
        """
        gap = abs(branch_a_risk - branch_b_risk)
        if gap < gap_threshold:
            return Z_AMBIG

        if branch_a_side == "left":
            if branch_a_risk > branch_b_risk:
                return Z_LEFT
            else:
                return Z_RIGHT
        else:
            if branch_a_risk > branch_b_risk:
                return Z_RIGHT
            else:
                return Z_LEFT

    def warning_evidence_factor(
        self, utterance_idx: int, cell_risk_logit: float
    ) -> float:
        """Warning as evidence factor for belief update.

        ψ_warn(u | z_i) ∝ exp(η · s_risk(z_i))  if u indicates danger
                        = 1.0                     if silence

        Returns multiplicative factor for belief update.
        """
        if utterance_idx == U_SILENCE:
            return 1.0

        eta = self.config.eta
        return float(np.exp(eta * min(cell_risk_logit, 3.0)))

    def update_belief_with_warning(
        self,
        prior_state: np.ndarray,
        utterance_idx: int,
    ) -> np.ndarray:
        """Update state belief using L1 posterior.

        q⁺(z) ∝ q⁻(z) · P_S1(u|z)

        Parameters
        ----------
        prior_state : (N_STATES,) prior over world states
        utterance_idx : which utterance was received

        Returns
        -------
        posterior : (N_STATES,) updated belief
        """
        S1 = self.speaker()  # (Z, U)
        likelihood = S1[:, utterance_idx]
        posterior = prior_state * likelihood
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = prior_state
        return posterior
