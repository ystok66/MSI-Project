"""
Oracle-Cause Teacher (v1e) — Upper bound for cause-aware decision making.

Uses TRUE agent belief + v1d cause-aware two-stage logic.
This is the fair upper bound: if oracle-cause also fails on PlanningTrap,
the problem is in the S_plan score itself, not particle inference.
"""

from __future__ import annotations

import numpy as np

from ..agents.bounded_agent import BoundedRationalAgent
from ..agents.belief import BeliefMap, apply_rsa_warning
from ..agents.planner_astar import bounded_astar
from .interventions import Intervention, InterventionType
from .rsa_warning import select_best_warning, _build_region_masks
from .cause_scoring import (
    compute_cause_scores, compute_success_prob, compute_survival_prob,
    CauseScores, CAUSES,
)
from .block_scoring import compute_block_decision, compute_timeout_risk


class OracleCauseTeacherPolicy:
    """
    Oracle teacher with v1d cause-aware decision logic.

    Sees the agent's TRUE belief maps but uses the same two-stage
    (safety gate → modality selection by dominant cause) as particle teacher.
    """

    def __init__(
        self,
        rsa_alpha: float = 5.0,
        rsa_beta: float = 0.1,
        rsa_tau: float = 1.0,
    ):
        self.rsa_alpha = rsa_alpha
        self.rsa_beta = rsa_beta
        self.rsa_tau = rsa_tau

    def select_action(
        self,
        agent: BoundedRationalAgent,
        true_cost: np.ndarray,
        true_risk: np.ndarray,
        goal: tuple[int, int],
        time_left: int,
        risk_budget_left: float,
        passable_mask: np.ndarray,
        door_positions: list[tuple[int, int]] | None = None,
        locked_doors: set[tuple[int, int]] | None = None,
        # v1d params
        kappa: float = 3.0,
        theta_warn: float = 0.05,
        theta_unlock: float = 0.05,
        gamma_risk: float = 0.3,
        gamma_time: float = 0.7,
        lambda_fp: float = 0.5,
        rho_warn_threshold: float = 0.15,
        search_budget: int = 40,
        # Allowed modalities
        allow_warn: bool = True,
        allow_unlock: bool = True,
        allow_shield: bool = True,
        allow_block: bool = True,
        # Agent state for BLOCK
        agent_plan: list[tuple[int, int]] | None = None,
        observation_positions_recent: list[tuple[int, int]] | None = None,
    ) -> tuple[Intervention, dict]:
        """
        Select intervention using TRUE belief + v1d cause-aware logic.
        Identical decision logic to particle teacher, but with oracle state.
        """
        locked_doors = locked_doors or set()
        door_positions = door_positions or list(locked_doors)
        H, W = true_risk.shape

        # Use agent's TRUE belief (oracle access)
        est_belief = agent.belief

        # ── Prepare WARN simulation ──
        sim_belief_warn = est_belief.copy()
        best_utt, _ = select_best_warning(
            est_belief.risk_mean, est_belief.risk_var,
            true_risk, agent.pos,
            alpha=self.rsa_alpha, beta=self.rsa_beta, tau=self.rsa_tau,
        )
        apply_rsa_warning(sim_belief_warn, best_utt, 0.5)

        masks = _build_region_masks(H, W)
        warn_region_mask = masks.get(best_utt)

        # ── Compute cause scores ──
        scores = compute_cause_scores(
            est_belief=est_belief,
            agent_pos=agent.pos,
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
        )

        posterior = scores.posterior(kappa)
        dominant = scores.dominant_cause(kappa)

        # ── Safety gate ──
        wait_plan = bounded_astar(
            agent.pos, goal,
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

        safety_override = (p_fatal > gamma_risk) or (p_timeout > gamma_time)

        # ── Modality selection (identical to particle teacher v1d) ──
        if not safety_override and dominant == "safe_exploration":
            chosen = Intervention.wait()
            chosen_reason = "safe_exploration"
        elif dominant == "belief_error" and allow_warn:
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
        elif dominant == "immediate_hazard" or safety_override:
            best_emergency = Intervention.wait()
            best_reduction = 0.0
            p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
            if allow_warn:
                p_succ_warn = compute_success_prob(
                    bounded_astar(
                        agent.pos, goal,
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
                    agent.pos, goal,
                    est_belief.cost_mean, est_belief.risk_mean, est_belief.cost_var,
                    budget=search_budget, lambda_risk=3.0, passable_mask=unlock_passable,
                ) or []
                p_succ_unlock = compute_success_prob(unlock_plan, goal, true_risk, time_left)
                p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
                if p_succ_unlock - p_succ_wait > best_reduction:
                    best_reduction = p_succ_unlock - p_succ_wait
                    best_emergency = Intervention.unlock_door("0")
            if allow_shield:
                p_succ_shield = compute_success_prob(wait_plan, goal, true_risk, time_left, shielded=True)
                p_succ_wait = compute_success_prob(wait_plan, goal, true_risk, time_left)
                if p_succ_shield - p_succ_wait > best_reduction:
                    best_reduction = p_succ_shield - p_succ_wait
                    best_emergency = Intervention.drop_shield(5)

            # BLOCK: only when belief_error or hazard is dominant
            if allow_block and observation_positions_recent is not None:
                plan_for_block = agent_plan if agent_plan else wait_plan
                should_blk, block_cell, blk_conds = compute_block_decision(
                    agent_pos=agent.pos,
                    agent_plan=plan_for_block,
                    agent_belief_risk=est_belief.risk_mean,
                    hazard_risk_map=true_risk,  # oracle: uses true_risk
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
                    # Compare BLOCK vs current best emergency
                    # BLOCK is preferred when it prevents the fatal path
                    block_passable = passable_mask.copy()
                    block_passable[block_cell] = False
                    block_plan = bounded_astar(
                        agent.pos, goal,
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
            chosen = Intervention.wait()
            chosen_reason = f"{dominant}_unavailable"

        info = {
            "dominant_cause": chosen_reason,
            "cause_posterior": {CAUSES[i]: round(float(posterior[i]), 4) for i in range(4)},
            "scores": {
                "s_explore": round(scores.explore, 4),
                "s_belief": round(scores.belief, 4),
                "s_plan": round(scores.plan, 4),
                "s_hazard": round(scores.hazard, 4),
            },
            "p_fatal_wait": round(p_fatal, 4),
            "p_timeout_wait": round(p_timeout, 4),
            "safety_override": safety_override,
        }
        return chosen, info
