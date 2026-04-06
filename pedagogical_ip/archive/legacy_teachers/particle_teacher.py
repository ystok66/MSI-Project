"""
SIPS-lite Particle Teacher — v1a.

Maintains particles representing hypotheses about the learner's internal
state (beliefs, plan, rationality traits). Updates weights based on
observed learner actions. Does NOT access oracle learner belief at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..agents.belief import (
    BeliefMap, update_belief_cell, apply_rsa_warning, log_det_risk_var,
)
from ..agents.observation_model import generate_observations
from ..agents.planner_astar import (
    bounded_astar, plan_next_action, sample_search_budget, MOVES,
)
from .interventions import Intervention, InterventionType
from .rsa_warning import select_best_warning, UTTERANCE_VOCAB, UTTERANCE_IS_RISKY
from .block_scoring import compute_block_decision


# ── Particle latent state ────────────────────────────────────────────
@dataclass
class Particle:
    """One hypothesis about the learner's internal state."""
    belief: BeliefMap
    current_plan: list[tuple[int, int]]
    plan_step_idx: int
    budget_class: int           # {4, 8, 16}
    warn_sensitivity: float     # {0.25, 0.5, 1.0}
    risk_aversion: float        # {0.5, 1.0, 2.0}  → scales lambda_risk
    weight: float = 1.0

    def copy(self) -> Particle:
        return Particle(
            belief=self.belief.copy(),
            current_plan=list(self.current_plan),
            plan_step_idx=self.plan_step_idx,
            budget_class=self.budget_class,
            warn_sensitivity=self.warn_sensitivity,
            risk_aversion=self.risk_aversion,
            weight=self.weight,
        )


# ── Discrete trait grid ──────────────────────────────────────────────
BUDGET_CLASSES = [4, 8, 16]
WARN_SENSITIVITIES = [0.25, 0.5, 1.0]
RISK_AVERSIONS = [0.5, 1.0, 2.0]


def _init_particles(
    n_particles: int,
    height: int,
    width: int,
    prior_cost_mean: float,
    prior_cost_var: float,
    prior_risk_mean: float,
    prior_risk_var: float,
    rng: np.random.Generator,
) -> list[Particle]:
    """
    Initialize particles with shared prior beliefs but varied traits.

    Sample from the 3×3×3 = 27 trait combos without replacement when possible.
    """
    all_combos = [
        (b, w, r)
        for b in BUDGET_CLASSES
        for w in WARN_SENSITIVITIES
        for r in RISK_AVERSIONS
    ]
    n_combos = len(all_combos)

    if n_particles <= n_combos:
        indices = rng.choice(n_combos, size=n_particles, replace=False)
    else:
        indices = rng.choice(n_combos, size=n_particles, replace=True)

    particles = []
    w0 = 1.0 / n_particles
    for idx in indices:
        b_cls, w_sens, r_aver = all_combos[idx]
        particles.append(Particle(
            belief=BeliefMap.from_prior(
                height, width,
                prior_cost_mean, prior_cost_var,
                prior_risk_mean, prior_risk_var,
            ),
            current_plan=[],
            plan_step_idx=0,
            budget_class=b_cls,
            warn_sensitivity=w_sens,
            risk_aversion=r_aver,
            weight=w0,
        ))
    return particles


def _effective_sample_size(particles: list[Particle]) -> float:
    """ESS = 1 / Σ(w²)."""
    weights = np.array([p.weight for p in particles])
    return 1.0 / np.sum(weights ** 2)


def _normalize_weights(particles: list[Particle]) -> None:
    """Normalize particle weights to sum to 1."""
    total = sum(p.weight for p in particles)
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        w0 = 1.0 / len(particles)
        for p in particles:
            p.weight = w0


def _resample(particles: list[Particle], rng: np.random.Generator) -> list[Particle]:
    """Multinomial resampling."""
    n = len(particles)
    weights = np.array([p.weight for p in particles])
    indices = rng.choice(n, size=n, p=weights)
    new_particles = []
    w0 = 1.0 / n
    for i in indices:
        pp = particles[i].copy()
        pp.weight = w0
        new_particles.append(pp)
    return new_particles


# ── Core: predict learner action for a particle ──────────────────────
def _predict_action(
    particle: Particle,
    agent_pos: tuple[int, int],
    goal: tuple[int, int],
    passable_mask: np.ndarray,
    lambda_uncertainty: float,
    rng: np.random.Generator,
) -> str:
    """
    Predict what action the learner would take under this particle's hypothesis.

    Uses the particle's belief and planner params to run bounded A*.
    """
    # Check if particle plan is still valid
    needs_replan = (
        not particle.current_plan
        or particle.plan_step_idx >= len(particle.current_plan)
        or (particle.plan_step_idx < len(particle.current_plan)
            and particle.current_plan[particle.plan_step_idx] != agent_pos
            and particle.current_plan[0] != agent_pos)
    )

    if needs_replan:
        budget = sample_search_budget(particle.budget_class, rng)
        path = bounded_astar(
            agent_pos, goal,
            particle.belief.cost_mean,
            particle.belief.risk_mean,
            particle.belief.cost_var,
            budget=budget,
            lambda_risk=particle.risk_aversion * 3.0,
            lambda_uncertainty=lambda_uncertainty,
            passable_mask=passable_mask,
        )
        particle.current_plan = path
        particle.plan_step_idx = 0
        if path and path[0] == agent_pos:
            particle.plan_step_idx = 1

    # Extract next move
    if particle.plan_step_idx < len(particle.current_plan):
        next_pos = particle.current_plan[particle.plan_step_idx]
        # Don't advance plan_step_idx here — that happens in propagate
        dr = next_pos[0] - agent_pos[0]
        dc = next_pos[1] - agent_pos[1]
        for mdr, mdc, name in MOVES:
            if dr == mdr and dc == mdc:
                return name
    return "STAY"


# ── Particle teacher ─────────────────────────────────────────────────
class ParticleTeacherPolicy:
    """
    SIPS-lite particle-inference teacher.

    Does NOT access the learner's true belief maps during decision time.
    Maintains particles and infers learner state from observed actions.
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_particles: int = 16,
        prior_cost_mean: float = 1.5,
        prior_cost_var: float = 4.0,
        prior_risk_mean: float = 0.1,
        prior_risk_var: float = 0.25,
        lambda_uncertainty: float = 0.5,
        # Action likelihood
        action_mismatch_penalty: float = 2.0,
        # Utility weights
        w_success: float = 4.0,
        w_ig: float = 1.0,
        w_cost: float = 0.5,
        w_frustration: float = 0.5,
        # Intervention costs
        intervention_costs: dict[str, float] | None = None,
        # Rollout
        rollout_horizon: int = 6,
        rollout_samples: int = 4,
        # RSA params
        rsa_alpha: float = 5.0,
        rsa_beta: float = 0.1,
        rsa_tau: float = 1.0,
        # RNG
        rng: np.random.Generator | None = None,
    ):
        self.height = height
        self.width = width
        self.n_particles = n_particles
        self.lambda_uncertainty = lambda_uncertainty
        self.action_penalty = action_mismatch_penalty

        self.w_s = w_success
        self.w_ig = w_ig
        self.w_c = w_cost
        self.w_f = w_frustration

        self.costs = intervention_costs or {
            "WAIT": 0.0, "WARN": 0.1,
            "UNLOCK_DOOR": 0.3, "DROP_SHIELD": 0.5,
        }
        self.rollout_horizon = rollout_horizon
        self.rollout_samples = rollout_samples
        self.rsa_alpha = rsa_alpha
        self.rsa_beta = rsa_beta
        self.rsa_tau = rsa_tau

        self.rng = rng or np.random.default_rng()

        # Prior params (for init/reset)
        self._prior = (prior_cost_mean, prior_cost_var,
                       prior_risk_mean, prior_risk_var)

        self.particles: list[Particle] = []
        self._step_count = 0

    def reset(self) -> None:
        """Reset particles for a new episode."""
        pcm, pcv, prm, prv = self._prior
        self.particles = _init_particles(
            self.n_particles, self.height, self.width,
            pcm, pcv, prm, prv, self.rng,
        )
        self._step_count = 0

    # ── Per-step update ──────────────────────────────────────────────
    def update(
        self,
        observed_action: str,
        agent_pos: tuple[int, int],
        goal: tuple[int, int],
        passable_mask: np.ndarray,
        true_cost: np.ndarray,
        true_risk: np.ndarray,
        last_robot_action: Intervention | None = None,
    ) -> dict:
        """
        Update particle weights based on observed learner action.

        1. Predict action for each particle
        2. Update weights
        3. Normalize
        4. Propagate beliefs (observation update from new position)
        5. Resample if ESS < N/2
        """
        self._step_count += 1
        predicted_actions = []

        for p in self.particles:
            # Apply last robot action to particle's belief
            if last_robot_action and last_robot_action.type != InterventionType.WAIT:
                if last_robot_action.type == InterventionType.WARN:
                    apply_rsa_warning(
                        p.belief, last_robot_action.param,
                        p.warn_sensitivity,
                    )
                    p.current_plan = []  # invalidate plan
                    p.plan_step_idx = 0
                elif last_robot_action.type == InterventionType.UNLOCK_DOOR:
                    # Door opened — update belief
                    for dr in range(self.height):
                        for dc in range(self.width):
                            if p.belief.cost_mean[dr, dc] > 50:
                                # Might be the door; reduce cost belief
                                pass
                    p.current_plan = []
                    p.plan_step_idx = 0
                elif last_robot_action.type == InterventionType.DROP_SHIELD:
                    p.current_plan = []
                    p.plan_step_idx = 0

            # Predict action
            pred = _predict_action(
                p, agent_pos, goal, passable_mask,
                self.lambda_uncertainty, self.rng,
            )
            predicted_actions.append(pred)

            # Update weight: downweight if prediction doesn't match
            if pred == observed_action:
                p.weight *= 1.0  # match — keep weight
            else:
                p.weight *= np.exp(-self.action_penalty)

            # Advance particle's plan step index
            if p.plan_step_idx < len(p.current_plan):
                p.plan_step_idx += 1

        # Normalize
        _normalize_weights(self.particles)

        # Propagate: update each particle's belief from the new position
        for p in self.particles:
            obs = generate_observations(
                agent_pos, true_cost, true_risk,
                self_noise_var=0.001,
                neighbor_noise_var=1.0,
                neighbor_radius=1,
                rng=self.rng,
            )
            for i, (r, c) in enumerate(obs.positions):
                update_belief_cell(
                    p.belief, r, c,
                    obs.cost_obs[i], obs.risk_obs[i],
                    obs.cost_var[i], obs.risk_var[i],
                )
                if r == agent_pos[0] and c == agent_pos[1]:
                    p.belief.visited_mask[r, c] = True

        # Resample if ESS too low
        ess = _effective_sample_size(self.particles)
        resampled = False
        if ess < self.n_particles / 2:
            self.particles = _resample(self.particles, self.rng)
            resampled = True

        return {
            "ess": ess,
            "resampled": resampled,
            "predicted_actions": predicted_actions,
            "match_frac": sum(
                1 for pa in predicted_actions if pa == observed_action
            ) / len(predicted_actions),
        }

    # ── Weighted belief estimate ─────────────────────────────────────
    def get_estimated_belief(self) -> BeliefMap:
        """Weighted average of particle beliefs — teacher's best estimate."""
        H, W = self.height, self.width
        cost_mean = np.zeros((H, W))
        cost_var = np.zeros((H, W))
        risk_mean = np.zeros((H, W))
        risk_var = np.zeros((H, W))

        for p in self.particles:
            cost_mean += p.weight * p.belief.cost_mean
            cost_var += p.weight * p.belief.cost_var
            risk_mean += p.weight * p.belief.risk_mean
            risk_var += p.weight * p.belief.risk_var

        return BeliefMap(
            height=H, width=W,
            cost_mean=cost_mean, cost_var=cost_var,
            risk_mean=risk_mean, risk_var=risk_var,
            visited_mask=np.zeros((H, W), dtype=bool),
        )

    # ── Action selection (v1d: cause-aware two-stage) ──────────────────
    def select_action(
        self,
        agent_pos: tuple[int, int],
        goal: tuple[int, int],
        true_risk: np.ndarray,
        true_cost: np.ndarray,
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
        locked_doors: set[tuple[int, int]] | None = None,
        door_positions: list[tuple[int, int]] | None = None,
        # v1d cause-aware params
        kappa: float = 3.0,
        theta_warn: float = 0.05,
        theta_unlock: float = 0.05,
        gamma_risk: float = 0.3,
        gamma_time: float = 0.7,
        lambda_fp: float = 0.5,
        rho_warn_threshold: float = 0.15,
        # Allowed modalities (for per-family ablation)
        allow_warn: bool = True,
        allow_unlock: bool = True,
        allow_shield: bool = True,
        allow_block: bool = True,
        search_budget: int = 40,
        # Agent state for BLOCK
        agent_plan: list[tuple[int, int]] | None = None,
        observation_positions_recent: list[tuple[int, int]] | None = None,
    ) -> tuple[Intervention, dict]:
        """
        Select intervention via cause-aware two-stage decision (v1d).

        Stage 1: Compute q(z | h_t) over 4 latent failure causes.
        Stage 2: If safe → select modality by dominant cause.
                 If hazardous → intervene to reduce failure probability.

        No global θ_intervene. Uses modality-specific margins instead.
        """
        locked_doors = locked_doors or set()
        door_positions = door_positions or list(locked_doors)
        est_belief = self.get_estimated_belief()

        # ── Prepare WARN simulation ──
        sim_belief_warn = est_belief.copy()
        best_utt, _ = select_best_warning(
            est_belief.risk_mean, est_belief.risk_var,
            true_risk, agent_pos,
            alpha=self.rsa_alpha, beta=self.rsa_beta, tau=self.rsa_tau,
        )
        apply_rsa_warning(sim_belief_warn, best_utt, 0.5)

        # Get region mask for warning
        from .rsa_warning import _build_region_masks
        masks = _build_region_masks(self.height, self.width)
        warn_region_mask = masks.get(best_utt)

        # ── Compute cause scores ──
        from .cause_scoring import (
            compute_cause_scores, compute_success_prob, compute_survival_prob,
            CauseScores, CAUSES,
        )

        scores = compute_cause_scores(
            est_belief=est_belief,
            agent_pos=agent_pos,
            goal=goal,
            true_risk=true_risk,
            true_cost=true_cost,
            time_left=time_left,
            risk_budget_left=risk_budget_left,
            passable_mask=passable_mask,
            locked_doors=locked_doors,
            door_positions=door_positions,
            best_utterance=best_utt,
            sim_belief_warn=sim_belief_warn,
            warn_region_mask=warn_region_mask,
            lambda_fp=lambda_fp,
            rho_warn_thresh=rho_warn_threshold,
            gamma_risk=gamma_risk,
            gamma_time=gamma_time,
            search_budget=search_budget,
        )

        posterior = scores.posterior(kappa)
        dominant = scores.dominant_cause(kappa)

        # ── Compute P_fatal and P_timeout for safety gate ──
        wait_plan = bounded_astar(
            agent_pos, goal,
            est_belief.cost_mean, est_belief.risk_mean, est_belief.cost_var,
            budget=search_budget, lambda_risk=3.0, passable_mask=passable_mask,
        ) or []
        survival = compute_survival_prob(wait_plan, true_risk)
        p_fatal = max(0.0, 1.0 - survival)
        if wait_plan and wait_plan[-1] == goal:
            plen = len(wait_plan) - 1
            p_timeout = max(0.0, 1.0 - max(0.0, (time_left - plen) / max(time_left, 1)))
        else:
            p_timeout = 0.8

        # ── Stage 1: Safety gate ──
        safety_override = (p_fatal > gamma_risk) or (p_timeout > gamma_time)

        # ── Stage 2: Modality selection ──
        if not safety_override and dominant == "safe_exploration":
            # Wait is best — learner should explore
            chosen = Intervention.wait()
            chosen_reason = "safe_exploration"

        elif dominant == "belief_error" and allow_warn:
            # Check modality-specific margin
            if scores.belief > theta_warn:
                chosen = Intervention.warn(best_utt)
                chosen_reason = "belief_error"
            else:
                chosen = Intervention.wait()
                chosen_reason = "belief_error_below_margin"

        elif dominant == "planning_bottleneck" and allow_unlock and locked_doors:
            if scores.plan > theta_unlock:
                chosen = Intervention.unlock_door("0")
                chosen_reason = "planning_bottleneck"
            else:
                chosen = Intervention.wait()
                chosen_reason = "planning_below_margin"

        elif (dominant == "immediate_hazard" or safety_override):
            # Emergency: pick action that most reduces failure prob
            # Try WARN first (cheap), then UNLOCK, then SHIELD
            best_emergency = Intervention.wait()
            best_reduction = 0.0
            p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)

            if allow_warn:
                p_succ_warn = compute_success_prob(
                    bounded_astar(
                        agent_pos, goal,
                        sim_belief_warn.cost_mean, sim_belief_warn.risk_mean,
                        sim_belief_warn.cost_var,
                        budget=search_budget, lambda_risk=3.0, passable_mask=passable_mask,
                    ) or [], goal, true_risk, time_left,
                )
                if p_succ_warn - p_succ_wait > best_reduction:
                    best_reduction = p_succ_warn - p_succ_wait
                    best_emergency = Intervention.warn(best_utt)

            if allow_unlock and locked_doors:
                unlock_passable = passable_mask.copy()
                for dr, dc in door_positions:
                    if (dr, dc) in locked_doors:
                        unlock_passable[dr, dc] = True
                unlock_plan = bounded_astar(
                    agent_pos, goal,
                    est_belief.cost_mean, est_belief.risk_mean,
                    est_belief.cost_var,
                    budget=search_budget, lambda_risk=3.0, passable_mask=unlock_passable,
                ) or []
                p_succ_unlock = compute_success_prob(
                    unlock_plan, goal, true_risk, time_left,
                )
                p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
                if p_succ_unlock - p_succ_wait > best_reduction:
                    best_reduction = p_succ_unlock - p_succ_wait
                    best_emergency = Intervention.unlock_door("0")

            if allow_shield:
                p_succ_shield = compute_success_prob(
                    wait_plan, goal, true_risk, time_left, shielded=True,
                )
                p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
                if p_succ_shield - p_succ_wait > best_reduction:
                    best_reduction = p_succ_shield - p_succ_wait
                    best_emergency = Intervention.drop_shield(5)

            # BLOCK: uses est_belief.risk_mean, NOT true_risk (no oracle leak)
            if allow_block and observation_positions_recent is not None:
                plan_for_block = agent_plan if agent_plan else wait_plan
                should_blk, block_cell, blk_conds = compute_block_decision(
                    agent_pos=agent_pos,
                    agent_plan=plan_for_block,
                    agent_belief_risk=est_belief.risk_mean,
                    hazard_risk_map=est_belief.risk_mean,  # particle: posterior, NO oracle
                    observation_positions_recent=observation_positions_recent,
                    goal=goal,
                    time_left=time_left,
                    risk_budget_left=risk_budget_left,
                    passable_mask=passable_mask,
                    belief_cost_mean=est_belief.cost_mean,
                    belief_cost_var=est_belief.cost_var,
                    search_budget=search_budget,
                )
                if should_blk and block_cell is not None:
                    block_passable = passable_mask.copy()
                    block_passable[block_cell] = False
                    block_plan = bounded_astar(
                        agent_pos, goal,
                        est_belief.cost_mean, est_belief.risk_mean,
                        est_belief.cost_var,
                        budget=search_budget, lambda_risk=3.0,
                        passable_mask=block_passable,
                    ) or []
                    p_succ_block = compute_success_prob(
                        block_plan, goal, true_risk, time_left,
                    )
                    if p_succ_block - p_succ_wait > best_reduction:
                        best_reduction = p_succ_block - p_succ_wait
                        best_emergency = Intervention.block_path(3)

            chosen = best_emergency
            chosen_reason = "immediate_hazard" if not safety_override else "safety_override"

        else:
            # Fallback: WAIT
            chosen = Intervention.wait()
            chosen_reason = f"{dominant}_unavailable"

        # ── Diagnostics ──
        # Warning precision
        warn_precision = 1.0
        if warn_region_mask is not None:
            mean_true = float(true_risk[warn_region_mask].mean())
            warn_precision = float(mean_true >= rho_warn_threshold)

        info = {
            "dominant_cause": chosen_reason,
            "cause_posterior": {
                CAUSES[i]: round(float(posterior[i]), 4)
                for i in range(4)
            },
            "scores": {
                "s_explore": round(scores.explore, 4),
                "s_belief": round(scores.belief, 4),
                "s_plan": round(scores.plan, 4),
                "s_hazard": round(scores.hazard, 4),
            },
            "p_fatal_wait": round(p_fatal, 4),
            "p_timeout_wait": round(p_timeout, 4),
            "warn_precision": warn_precision,
            "safety_override": safety_override,
            "selected_warning": best_utt if chosen.type == InterventionType.WARN else "",
            "ess": _effective_sample_size(self.particles),
        }

        return chosen, info

