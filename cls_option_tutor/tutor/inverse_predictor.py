"""
inverse_predictor.py — Clean inverse shadow predictor.

Uses ONLY public observation history (ObservedStep) to predict learner behavior.
MUST NOT hold any reference to LearnerAgent or its components.

Internal state:
  - shadow_model:     ShadowLearnerModel (scorer + danger_head + attention)
  - log_profile_weights: (M,) posterior over profile hypotheses
  - profile_grid:     list of ProfileHypothesis

Observe order (critical):
  1. Compute action likelihood for each profile BEFORE updating shadow
  2. Update profile posterior with observed action likelihood
  3. THEN update shadow semantic/risk from reveal/damage

This ensures: shadow state at step t predicts action at step t,
not action at step t+1.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np

from .predictor import LearnerPredictor, FullActionDist
from .learner_model import (
    ProfileHypothesis,
    ShadowLearnerModel,
    PROFILE_GRID,
)
from .profile_inference import (
    update_profile_posterior,
    init_uniform_log_weights,
    posterior_entropy,
    posterior_probs,
)
from .observation_adapter import ObservedStep


class InverseShadowPredictor:
    """Clean inverse shadow predictor using only public observation history.

    MUST NOT hold any reference to LearnerAgent or its components.
    All predictions come from the shadow model + profile mixture.

    Args:
        shadow_model:   independently initialized ShadowLearnerModel
        profile_grid:   list of ProfileHypothesis (default: PROFILE_GRID)
        eta_prof:       likelihood temperature for profile posterior update
        rollout_mode:   "proxy" (geometric approximation) or "shadow_mc"
        n_rollout:      number of rollouts for shadow_mc mode
        update_semantic: if False, shadow scorer is not updated from reveals
        update_risk:     if False, shadow danger_head is not updated
    """

    def __init__(
        self,
        shadow_model: ShadowLearnerModel,
        profile_grid: Optional[List[ProfileHypothesis]] = None,
        eta_prof: float = 1.0,
        rollout_mode: str = "proxy",
        n_rollout: int = 8,
        update_semantic: bool = True,
        update_risk: bool = True,
    ):
        self._shadow = shadow_model
        self._profiles = profile_grid or list(PROFILE_GRID)
        self._eta_prof = eta_prof
        self._rollout_mode = rollout_mode
        self._n_rollout = n_rollout
        self._update_semantic = update_semantic
        self._update_risk = update_risk

        M = len(self._profiles)
        self._log_weights = init_uniform_log_weights(M)

        # Diagnostics
        self._observe_count = 0
        self._last_likelihoods: Optional[np.ndarray] = None
        # NLL / accuracy trackers
        self._refresh_total = 0
        self._refresh_correct = 0  # predicted p_refresh >= 0.5 and action was refresh
        self._full_action_nll_sum = 0.0
        self._pick_nll_sum = 0.0
        self._pick_count = 0

    # ── Protocol: observe ─────────────────────────────────────────────────────

    def observe(self, obs_step: ObservedStep) -> None:
        """Process one public observation step.

        ORDER IS CRITICAL:
        1. Compute action likelihood per profile (BEFORE shadow update)
        2. Update profile posterior
        3. Update shadow state from public feedback
        """
        # ── 1. Compute action likelihoods per profile ──
        likelihoods = np.zeros(len(self._profiles))

        for pi, profile in enumerate(self._profiles):
            li = self._action_likelihood(obs_step, profile)
            likelihoods[pi] = li

        self._last_likelihoods = likelihoods.copy()

        # ── 2. Update profile posterior ──
        self._log_weights = update_profile_posterior(
            self._log_weights,
            likelihoods,
            eta_prof=self._eta_prof,
        )

        # ── 3. Update shadow state from public feedback ──
        # Reset attention for new query FIRST (before highlight)
        if obs_step.round_t == 0:
            self._shadow.reset_attention(len(obs_step.target_output))

        # Apply highlight attention boost (after reset, so round-0 highlight persists)
        if obs_step.active_highlights:
            self._shadow.update_attention_highlight(obs_step.active_highlights)

        # Update from reveal (wrong pick only)
        # update_from_reveal handles BOTH semantic and risk updates internally.
        # Do NOT update risk separately — that would double-count.
        if (obs_step.learner_action == "pick"
                and obs_step.pick_correct is False):
            # Find the picked option text
            picked_text = None
            if obs_step.learner_pick_index is not None:
                # Map pick_index (menu index) to position in option_texts
                for i, idx in enumerate(obs_step.option_indices):
                    if idx == obs_step.learner_pick_index:
                        picked_text = list(obs_step.option_texts[i])
                        break
            if picked_text is not None:
                revealed = (list(obs_step.revealed_output)
                            if obs_step.revealed_output else None)
                self._shadow.update_from_reveal(
                    wrong_text=picked_text,
                    revealed_output=revealed,
                    danger_vec=obs_step.revealed_danger_vec,
                    damage=obs_step.pick_damage,
                    update_semantic=self._update_semantic,
                    update_risk=self._update_risk,
                )

        self._observe_count += 1

        # ── 4. Accumulate NLL diagnostics ──
        w = posterior_probs(self._log_weights)
        # Full action NLL: -log Σ_ψ w(ψ) * π_ψ(a_t | x_t)
        mixture_lik = float(np.clip(w @ likelihoods, 1e-15, 1.0))
        self._full_action_nll_sum += -np.log(mixture_lik)

        if obs_step.learner_action == "refresh":
            self._refresh_total += 1
            # Compute mixture p_refresh for accuracy
            dvecs = list(obs_step.option_danger_vecs)
            p_ref_mix = 0.0
            for pi, profile in enumerate(self._profiles):
                pr = self._shadow.predict_refresh_prob(
                    option_danger_vecs=dvecs,
                    hp=obs_step.hp_before,
                    profile=profile,
                    refreshes_used=obs_step.rounds_before,
                )
                p_ref_mix += w[pi] * pr
            if p_ref_mix >= 0.5:
                self._refresh_correct += 1
        elif obs_step.learner_action == "pick":
            # Pick NLL (mixture pick likelihood, no refresh component)
            self._pick_count += 1
            pick_lik = float(np.clip(mixture_lik, 1e-15, 1.0))
            self._pick_nll_sum += -np.log(pick_lik)

    # ── Protocol: pick_dist ───────────────────────────────────────────────────

    def pick_dist(self, qs, active: list, spec: dict) -> np.ndarray:
        """Mixture pick distribution for existing Q formulas.

        Returns (K_active,) array.
        """
        probs = posterior_probs(self._log_weights)
        K = len(active)
        if K == 0:
            return np.array([])

        mixture = np.zeros(K)
        target = list(qs.target_output)
        texts = [list(o.text) for o in active]
        dvecs = [o.danger_vec for o in active]
        banned = qs.banned_indices

        for pi, profile in enumerate(self._profiles):
            p_pick = self._shadow.predict_pick_probs(
                target_output=target,
                option_texts=texts,
                option_danger_vecs=dvecs,
                profile=profile,
                spec=spec,
                banned_indices=banned,
                highlighted_cells=qs.highlighted_cells,
            )
            mixture += probs[pi] * p_pick

        # Normalize
        s = mixture.sum()
        if s > 0:
            mixture /= s
        return mixture

    # ── Protocol: full_action_dist ────────────────────────────────────────────

    def full_action_dist(self, qs, active: list, spec: dict) -> FullActionDist:
        """Full action distribution including refresh probability."""
        pick_probs = self.pick_dist(qs, active, spec)
        probs = posterior_probs(self._log_weights)

        # Mixture refresh probability
        dvecs = [o.danger_vec for o in active]
        p_refresh = 0.0
        for pi, profile in enumerate(self._profiles):
            pr = self._shadow.predict_refresh_prob(
                option_danger_vecs=dvecs,
                hp=qs.hp,
                profile=profile,
                refreshes_used=qs.refreshes_used,
                max_refreshes=qs.max_refreshes,
            )
            p_refresh += probs[pi] * pr

        # Scale pick_probs by (1 - p_refresh)
        pick_scaled = pick_probs * (1.0 - p_refresh)

        active_indices = [o.index for o in active]
        return FullActionDist(
            active_indices=active_indices,
            pick_probs=pick_scaled,
            p_refresh=float(p_refresh),
        )

    # ── Protocol: rollout ─────────────────────────────────────────────────────

    def rollout(
        self,
        qs,
        active: list,
        spec: dict,
        n: int,
    ) -> Tuple[float, float, float]:
        """Rollout using shadow model.

        rollout_mode="proxy": geometric approximation (no deepcopy).
        rollout_mode="shadow_mc": Monte Carlo with shadow deepcopy.
        """
        if self._rollout_mode == "proxy":
            return self._rollout_proxy(qs, active, spec)
        else:
            return self._rollout_shadow_mc(qs, active, spec, n)

    # ── Protocol: clone ───────────────────────────────────────────────────────

    def clone(self) -> 'InverseShadowPredictor':
        """Deep-copy for rollout branching."""
        cloned = InverseShadowPredictor(
            shadow_model=self._shadow.deep_copy(),
            profile_grid=list(self._profiles),
            eta_prof=self._eta_prof,
            rollout_mode=self._rollout_mode,
            n_rollout=self._n_rollout,
            update_semantic=self._update_semantic,
            update_risk=self._update_risk,
        )
        cloned._log_weights = self._log_weights.copy()
        cloned._observe_count = self._observe_count
        return cloned

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Return current predictor diagnostics including NLL tracking."""
        probs = posterior_probs(self._log_weights)
        n = max(self._observe_count, 1)
        return {
            "profile_posterior": {
                p.name: float(probs[i])
                for i, p in enumerate(self._profiles)
            },
            "profile_entropy": posterior_entropy(self._log_weights),
            "observe_count": self._observe_count,
            "last_likelihoods": (
                self._last_likelihoods.tolist()
                if self._last_likelihoods is not None else None
            ),
            # NLL metrics
            "full_action_nll": self._full_action_nll_sum / n,
            "pick_nll": (
                self._pick_nll_sum / max(self._pick_count, 1)
                if self._pick_count > 0 else None
            ),
            # Refresh metrics
            "refresh_n": self._refresh_total,
            "refresh_acc_at_0.5": (
                self._refresh_correct / max(self._refresh_total, 1)
                if self._refresh_total > 0 else None
            ),
        }

    # ── Internal: action likelihood ───────────────────────────────────────────

    def _action_likelihood(
        self,
        step: ObservedStep,
        profile: ProfileHypothesis,
    ) -> float:
        """Compute π_ψ(a_t | x_t) for one profile.

        Handles both pick and refresh actions.
        """
        target = list(step.target_output)
        texts = [list(t) for t in step.option_texts]
        dvecs = list(step.option_danger_vecs)
        banned = set(step.active_bans)

        # Spec for current intervention state
        spec = {"action": "WAIT"}
        if step.active_highlights:
            spec["highlight_cells"] = step.active_highlights
        if step.active_bans:
            spec["ban_index"] = step.active_bans[0] if len(step.active_bans) == 1 else None

        if step.learner_action == "refresh":
            # Refresh likelihood
            p_ref = self._shadow.predict_refresh_prob(
                option_danger_vecs=dvecs,
                hp=step.hp_before,
                profile=profile,
                refreshes_used=step.rounds_before,
            )
            return float(max(p_ref, 1e-8))

        elif step.learner_action == "pick" and step.learner_pick_index is not None:
            # Pick likelihood
            p_ref = self._shadow.predict_refresh_prob(
                option_danger_vecs=dvecs,
                hp=step.hp_before,
                profile=profile,
                refreshes_used=step.rounds_before,
            )
            pick_probs = self._shadow.predict_pick_probs(
                target_output=target,
                option_texts=texts,
                option_danger_vecs=dvecs,
                profile=profile,
                spec=spec,
                banned_indices=banned,
                highlighted_cells=step.active_highlights or None,
                option_indices=list(step.option_indices),
            )
            # Map menu index to position
            picked_pos = None
            for i, idx in enumerate(step.option_indices):
                if idx == step.learner_pick_index:
                    picked_pos = i
                    break

            if picked_pos is not None and picked_pos < len(pick_probs):
                p_pick = float(pick_probs[picked_pos])
                # Joint: (1 - p_refresh) * p_pick_given_pick
                total = float(max((1.0 - p_ref) * p_pick, 1e-8))
                return total

        return 1e-8  # fallback for unknown actions

    # ── Internal: proxy rollout ───────────────────────────────────────────────

    def _rollout_proxy(
        self,
        qs,
        active: list,
        spec: dict,
    ) -> Tuple[float, float, float]:
        """Geometric P(timeout) proxy using shadow pick distribution.

        Same formula as SparseTutorAgent._compute_p_timeout / _compute_p_death.
        """
        probs = self.pick_dist(qs, active, spec)
        if len(probs) == 0:
            return (0.0, 1.0, 0.0)

        # P_death: sum p(j) for lethal non-correct options
        p_death = 0.0
        p_j_star = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_j_star = float(probs[i])
            elif opt.risk_class >= qs.hp:
                p_death += float(probs[i])

        # P_timeout: geometric
        tau_t = max(0, qs.max_rounds - qs.rounds_used)
        if tau_t <= 0:
            p_timeout = 1.0
        else:
            p_timeout = max(0.0, (1.0 - p_j_star) ** tau_t)

        p_success = max(0.0, 1.0 - p_death - p_timeout)
        return (float(p_death), float(p_timeout), float(p_success))

    # ── Internal: shadow Monte Carlo rollout ──────────────────────────────────

    def _rollout_shadow_mc(
        self,
        qs,
        active: list,
        spec: dict,
        n: int,
    ) -> Tuple[float, float, float]:
        """Monte Carlo rollout using cloned shadow model.

        Simulates n full query trajectories from current state.
        Uses mixture pick distribution (no profile sampling — mixture).
        """
        deaths = 0
        timeouts = 0
        successes = 0

        rng = np.random.default_rng(
            seed=hash((qs.query_id, spec.get("action", "WAIT"))) & 0xFFFFFFFF
        )

        for _ in range(n):
            shadow_copy = self._shadow.deep_copy()
            hp = qs.hp
            rounds_used = qs.rounds_used
            max_rounds = qs.max_rounds
            outcome = 'timeout'

            while rounds_used < max_rounds and hp > 0:
                # Use mixture pick probs from shadow copy
                pick_probs = self.pick_dist(qs, active, spec)
                K = len(active)
                if K == 0:
                    break

                pick_i = int(rng.choice(K, p=pick_probs))
                picked = active[pick_i]
                rounds_used += 1

                if picked.is_correct:
                    outcome = 'success'
                    break
                else:
                    hp -= picked.risk_class
                    if hp <= 0:
                        outcome = 'death'
                        break

            if outcome == 'success':
                successes += 1
            elif outcome == 'death':
                deaths += 1
            else:
                timeouts += 1

        return (deaths / n, timeouts / n, successes / n)
