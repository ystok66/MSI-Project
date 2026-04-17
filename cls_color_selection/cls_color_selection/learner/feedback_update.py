"""
feedback_update.py — Confirm feedback grammar update via Bayesian reweighting.

Two modes:
  A. wrong_only: learner knows "it's wrong" but not where
  B. wrong_positions: learner knows which positions are wrong

Both use differential M-step on the existing CLS cortex:
  Δn_{w,r} = η_fb · Σ_k (q̃_k - q_k) · C_{w,r}(π_k)

This directly modifies NeuroConcept sufficient statistics without
full retraining.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp

from ..config import LearnerConfig


class FeedbackUpdater:
    """Confirm feedback updater for CLS grammar learner.

    Reweights beam posterior based on feedback likelihood,
    then applies differential M-step to cortex concept library.
    """

    def __init__(self, cfg: LearnerConfig):
        self.cfg = cfg

    def compute_feedback_likelihood_wrong_only(
        self,
        Y_hat: List[str],
        Y_k: List[str],
    ) -> float:
        """P(F_wrong | Ŷ, Y_k) for wrong_only mode.

        If Y_k == Ŷ: return ε_wrong (near zero — suppress matching trace)
        If Y_k ≠ Ŷ: return 1 - ε_wrong (keep non-matching traces)
        """
        if Y_hat == Y_k:
            return self.cfg.eps_wrong
        else:
            return 1.0 - self.cfg.eps_wrong

    def compute_feedback_likelihood_wrong_positions(
        self,
        Y_hat: List[str],
        Y_k: List[str],
        mask: List[bool],
        assist_mask: Optional[List[bool]] = None,
    ) -> float:
        """P(F_mask | Ŷ, Y_k, m, m^assist) for wrong_positions mode.

        For each position ℓ:
          s_{k,ℓ} = (1 - ε_eq) if y_{k,ℓ} == ŷ_ℓ else ε_eq
          w_ℓ = ρ_assist if m^assist_ℓ else 1.0

        log P(F) = Σ_{ℓ: m=True} w_ℓ · log s_{k,ℓ}
                 + Σ_{ℓ: m=False} w_ℓ · log(1 - s_{k,ℓ})

        When assist_mask is None or rho_assist=1.0, equivalent to original.
        """
        eps = self.cfg.eps_eq
        rho = getattr(self.cfg, 'rho_assist', 1.0)
        L = min(len(Y_hat), len(Y_k), len(mask))
        log_lik = 0.0

        for ell in range(L):
            # Evidence weight for this position
            w = 1.0
            if assist_mask and ell < len(assist_mask) and assist_mask[ell]:
                w = rho  # Discount evidence at tutor-assisted positions

            # Match probability
            if ell < len(Y_k) and ell < len(Y_hat):
                if Y_k[ell] == Y_hat[ell]:
                    s = 1.0 - eps
                else:
                    s = eps
            else:
                s = eps  # length mismatch counts as mismatch

            if mask[ell]:
                # Position marked correct — want s high
                log_lik += w * np.log(max(s, 1e-30))
            else:
                # Position marked wrong — want (1-s) high
                log_lik += w * np.log(max(1.0 - s, 1e-30))

        return np.exp(log_lik)

    def reweight_beam_posterior(
        self,
        beam: List[Tuple[float, list, List[str]]],
        Y_hat: List[str],
        feedback: dict,
        assist_mask: Optional[List[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reweight beam posterior using feedback likelihood.

        Args:
            beam: [(score_k, trace_k, Y_k), ...] from CLSSequencePredictor
            Y_hat: the output the learner confirmed (submitted)
            feedback: {'mode': 'wrong_only'|'wrong_positions', 'mask': [...]}
            assist_mask: which positions were tutor-assisted (Phase 6)

        Returns:
            (q_old, q_new): original and reweighted posterior arrays
        """
        if not beam:
            return np.array([]), np.array([])

        K = len(beam)
        mode = feedback.get('mode', self.cfg.feedback_mode)
        mask = feedback.get('mask', [])

        # Compute base posterior q_k (softmax over scores)
        scores = np.array([b[0] for b in beam])
        log_q = scores - logsumexp(scores)
        q_old = np.exp(log_q)

        # Compute feedback likelihoods
        fb_liks = np.zeros(K)
        for k in range(K):
            Y_k = beam[k][2]
            if mode == 'wrong_only':
                fb_liks[k] = self.compute_feedback_likelihood_wrong_only(Y_hat, Y_k)
            elif mode == 'wrong_positions':
                fb_liks[k] = self.compute_feedback_likelihood_wrong_positions(
                    Y_hat, Y_k, mask, assist_mask=assist_mask)
            else:
                fb_liks[k] = 1.0  # no update

        # Reweight: q̃_k ∝ q_k · P(F | Ŷ, Y_k)
        unnorm = q_old * fb_liks
        total = unnorm.sum()
        if total < 1e-30:
            q_new = q_old.copy()  # fallback: no change
        else:
            q_new = unnorm / total

        return q_old, q_new

    def differential_m_step(
        self,
        library: dict,
        beam: List[Tuple[float, list, List[str]]],
        q_old: np.ndarray,
        q_new: np.ndarray,
    ):
        """Apply differential M-step to the CLS concept library.

        For each word w and role r in the beam traces:
          Δn_{w,r} = η_fb · Σ_k (q̃_k - q_k) · C_{w,r}(π_k)

        For emission stats:
          ΔS_w = η_fb · Σ_k (q̃_k - q_k) · T_w(π_k)

        Directly modifies the NeuroConcept objects in the library.

        Args:
            library: Dict[str, NeuroConcept] from CLSAgent.cortex
            beam: [(score, trace, Y_k), ...] with trace as list of steps
            q_old: original beam posterior
            q_new: reweighted beam posterior
        """
        eta = self.cfg.eta_fb
        K = len(beam)

        for k in range(K):
            delta_q = q_new[k] - q_old[k]
            if abs(delta_q) < 1e-15:
                continue

            trace = beam[k][1]
            weight = eta * delta_q

            for step in trace:
                word = step.word
                role = step.role
                if word not in library:
                    continue
                concept = library[word]

                # Update role counts
                concept.role_counts[role] = concept.role_counts.get(role, 0.0) + weight

                # Update emission stats if EMIT
                if role == 'EMIT' and hasattr(step, 'emit_vec') and step.emit_vec is not None:
                    vec = step.emit_vec
                    concept.emit_stats['sum_w'] += weight
                    concept.emit_stats['sum_wx'] += weight * vec
                    concept.emit_stats['sum_wx2'] += weight * (vec ** 2)

                    # Also update discrete color counts
                    if hasattr(step, 'emit_vec'):
                        from ns_learner.ns_concept import vec_to_color
                        c = vec_to_color(vec)
                        concept.color_counts[c] = concept.color_counts.get(c, 0.0) + weight

                # Update repeat counts if REPEAT
                if role == 'REPEAT' and hasattr(step, 'repeat_k') and step.repeat_k is not None:
                    k_rep = step.repeat_k
                    if k_rep in concept.repeat_counts:
                        concept.repeat_counts[k_rep] += weight

    def apply_feedback(
        self,
        predictor,
        words: List[str],
        Y_hat: List[str],
        feedback: dict,
        assist_mask: Optional[List[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full feedback pipeline: beam → reweight → differential M-step.

        Args:
            predictor: CLSSequencePredictor
            words: query words
            Y_hat: submitted (wrong) output
            feedback: feedback dict with mode and mask
            assist_mask: which positions were tutor-assisted (Phase 6)

        Returns:
            (q_old, q_new) for diagnostics
        """
        beam = predictor.beam_posterior(words)
        if not beam:
            return np.array([]), np.array([])

        q_old, q_new = self.reweight_beam_posterior(
            beam, Y_hat, feedback, assist_mask=assist_mask)

        # Apply differential M-step
        library = predictor.get_library()
        self.differential_m_step(library, beam, q_old, q_new)

        return q_old, q_new
