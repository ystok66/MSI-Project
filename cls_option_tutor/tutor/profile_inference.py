"""
profile_inference.py — Policy-grounded learner profile estimation (R2).

RSA-style inverse planning: infer compact profile ψ by MAP:

    ψ̂ = argmax_ψ [ log p₀(ψ) + Σ_t log π_L(aₜ | sₜ, ψ) ]

Uses the learner's own utility function to evaluate P(action | state, profile),
not ad-hoc behavioral rules.

V1 uses discrete grid search over a small parameter lattice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from itertools import product

from ..config import TutorConfig, LearnerConfig
from ..interfaces import LearnerStep, PolicyStateSnapshot
from ..env.state import ProfileState


@dataclass
class ProfilePosterior:
    """Posterior distribution over learner profiles."""
    grid_points: List[ProfileState]
    log_weights: np.ndarray  # unnormalized log-posterior

    @property
    def map_profile(self) -> ProfileState:
        """Maximum a posteriori profile."""
        idx = int(np.argmax(self.log_weights))
        return self.grid_points[idx]

    @property
    def posterior_probs(self) -> np.ndarray:
        """Normalized posterior probabilities."""
        shifted = self.log_weights - np.max(self.log_weights)
        probs = np.exp(shifted)
        return probs / (probs.sum() + 1e-10)

    def expected_profile(self) -> ProfileState:
        """Posterior expectation over profiles."""
        probs = self.posterior_probs
        fields = ["lambda_risk", "lambda_refresh", "s_order",
                  "s_scope", "g_highlight", "semantic_competence"]
        expected = {}
        for f in fields:
            expected[f] = sum(
                probs[i] * getattr(self.grid_points[i], f)
                for i in range(len(self.grid_points))
            )
        return ProfileState(**expected)


class ProfileInference:
    """Policy-grounded profile inference via inverse planning.

    Uses PolicyStateSnapshots recorded by the learner to evaluate
    the learner's own utility function under candidate profiles.
    This is the correct lightweight RSA analogue.
    """

    def __init__(self, cfg: TutorConfig):
        self.cfg = cfg
        self._grid: Optional[List[ProfileState]] = None
        self._posterior: Optional[ProfilePosterior] = None

    def build_grid(self, n: int = 5) -> List[ProfileState]:
        """Build grid of candidate profiles.

        Compact grid:
        - lambda_risk in {0.5, 1.0, 1.5, 2.0}
        - lambda_refresh in {0.0, 0.5, 1.0}
        - g_highlight in {0.0, 0.5, 1.0}
        - s_order, s_scope fixed at {0.0, 0.5} x {0.0, 0.5}
        - semantic_competence in {0.0, 0.5, 1.0}  (novice/intermediate/advanced)

        Total: 4 * 3 * 3 * 2 * 2 * 3 = 432 grid points
        """
        risk_vals = [0.5, 1.0, 1.5, 2.0]
        refresh_vals = [0.0, 0.5, 1.0]
        hl_vals = [0.0, 0.5, 1.0]
        order_vals = [0.0, 0.5]
        scope_vals = [0.0, 0.5]
        sem_comp_vals = [0.0, 0.5, 1.0]  # novice / intermediate / advanced

        grid = []
        for lr, lref, gh, so, ss, sc in product(
                risk_vals, refresh_vals, hl_vals, order_vals, scope_vals, sem_comp_vals):
            grid.append(ProfileState(
                lambda_risk=float(lr),
                lambda_refresh=float(lref),
                s_order=float(so),
                s_scope=float(ss),
                g_highlight=float(gh),
                semantic_competence=float(sc),
            ))

        self._grid = grid
        return grid

    def infer(self, obs_trace: List[LearnerStep],
              menu_danger_means: Optional[List[float]] = None,
              snapshots: Optional[List[PolicyStateSnapshot]] = None,
              ) -> ProfilePosterior:
        """Infer learner profile from observation-phase trace.

        Args:
            obs_trace: learner actions during observation queries
            menu_danger_means: (legacy, unused with snapshots)
            snapshots: PolicyStateSnapshots from observation phase

        Returns:
            ProfilePosterior with MAP estimate and full posterior
        """
        if self._grid is None:
            self.build_grid(self.cfg.profile_grid_size)

        n_grid = len(self._grid)
        log_w = np.zeros(n_grid)

        for i, profile in enumerate(self._grid):
            if snapshots:
                log_w[i] = self._log_likelihood_policy(profile, snapshots)
            else:
                # Fallback to behavioral heuristic if no snapshots
                log_w[i] = self._log_likelihood_heuristic(profile, obs_trace)

        self._posterior = ProfilePosterior(
            grid_points=self._grid,
            log_weights=log_w,
        )
        return self._posterior

    def _log_likelihood_policy(
        self,
        profile: ProfileState,
        snapshots: List[PolicyStateSnapshot],
    ) -> float:
        """V2 policy-grounded log-likelihood (RSA inverse planning).

        For each snapshot, compute:
            log P(action | state, profile) = log softmax(beta * U(a; s, profile))

        V2: includes p_ko and time pressure in the utility model.
        """
        ll = 0.0
        beta_L = 4.0  # learner softmax temperature

        for snap in snapshots:
            K = len(snap.option_texts)
            if K == 0:
                continue

            sem_scores = snap.semantic_scores
            if sem_scores is None or len(sem_scores) == 0:
                continue

            # Compute danger predictions using snapshot's posterior
            danger_preds = np.zeros(K)
            ko_probs = np.zeros(K)
            if snap.danger_posterior_mean is not None:
                for j in range(K):
                    v = snap.option_danger_vecs[j]
                    m = min(len(snap.danger_posterior_mean), len(v))
                    danger_preds[j] = float(
                        snap.danger_posterior_mean[:m] @ v[:m])
                    # Rough KO proxy: danger / hp
                    if snap.hp_before > 0:
                        ko_probs[j] = max(0.0, min(1.0,
                            danger_preds[j] / snap.hp_before))

            # V2: budget state
            r_t = (5 - snap.attempt_idx) / max(5, 1)  # normalized remaining time

            # Utility under candidate profile
            sc = profile.semantic_competence
            U_pick = (sc * profile.lambda_risk * sem_scores
                      - profile.lambda_risk * danger_preds
                      - 1.0 * ko_probs)  # alpha_ko = 1.0

            # Refresh utility (V2: time pressure)
            can_refresh = snap.refresh_count < snap.max_refreshes
            if can_refresh and K > 0:
                H_sem = _entropy(sem_scores)
                mean_d = float(np.mean(danger_preds))
                mean_ko = float(np.mean(ko_probs))
                U_refresh = (profile.lambda_refresh * H_sem
                             + profile.lambda_risk * mean_d
                             + 1.0 * mean_ko
                             - 0.3
                             - 0.3 * (1.0 - r_t))  # time penalty
            else:
                U_refresh = -100.0

            # Full utility vector
            all_U = np.concatenate([U_pick, [U_refresh]])

            # Softmax probabilities
            shifted = all_U - np.max(all_U)
            exp_u = np.exp(beta_L * shifted)
            probs = exp_u / (exp_u.sum() + 1e-10)

            # Which action was taken?
            if snap.learner_action == "refresh":
                action_idx = K
            elif snap.learner_action == "pick" and snap.learner_pick_index is not None:
                try:
                    action_idx = snap.option_indices.index(snap.learner_pick_index)
                except ValueError:
                    continue
            else:
                continue

            p = probs[action_idx] if action_idx < len(probs) else 1e-10
            ll += np.log(max(p, 1e-10))

        return ll

    def _log_likelihood_heuristic(
        self,
        profile: ProfileState,
        trace: List[LearnerStep],
    ) -> float:
        """Legacy heuristic log-likelihood (fallback when no snapshots).

        Simple behavioral model, kept for backward compatibility.
        """
        ll = 0.0
        for step in trace:
            if step.action == "refresh":
                ll += profile.lambda_refresh - 0.3
            elif step.action == "pick":
                if step.correct:
                    ll += 0.5
                else:
                    d = step.damage if step.damage else 0
                    ll -= profile.lambda_risk * d * 0.2
        return ll

    @property
    def current_profile(self) -> ProfileState:
        """Get current MAP profile (or default if not yet inferred)."""
        if self._posterior is not None:
            return self._posterior.map_profile
        return ProfileState()


def _entropy(scores: np.ndarray) -> float:
    """Softmax entropy from raw scores."""
    if len(scores) == 0:
        return 0.0
    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs = probs / (probs.sum() + 1e-10)
    p_pos = probs[probs > 0]
    return -float(np.sum(p_pos * np.log(p_pos)))
