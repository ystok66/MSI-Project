"""Joint Tutor v2 — Bounded One-Step Nonmyopic Tutor.

Built on v1.1 repaired decision boundary + coupled q(g,θ).

Key upgrade over JointLatentTutorV1:
  - Explicit one-step lookahead: "what if I WAIT and observe agent's next action?"
  - Decision-aware value: uses branch-decision Bayes Risk, not just entropy
  - 4 gating quantities from v1.1 (confidence, susceptibility, opportunity, slack)
  - Conflict mass: detects when goal × pref push in opposite directions

Core comparison:
  Q_warn = v1.1_base + λ_J·ΔJointInfo + λ_conflict·R_conflict·tempt_risk
  Q_wait = v1.1_base + λ_obs·S_obs·V_obs^(1) - λ_miss·P(miss_window)
           + λ_conf·C(q_joint)·O_wait

Where V_obs^(1) = BR(q_t) - E_a~p(a|h_t)[BR(q_{t+1}^(a))]
  = expected decision Bayes Risk reduction from one more observation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES,
)
from ..agents.joint_posterior_v2 import (
    JointPosteriorV2, compute_joint_likelihood, GOAL_TYPES,
    N_GOALS, N_PREF,
)
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


# Temptation susceptibility table (shared with v1.1)
TEMPT_SUSCEPTIBILITY = {
    "safe": 0.10, "neutral": 0.30, "risky": 0.70,
    "shortcut": 0.60, "shiny": 1.00,
}


@dataclass
class JointTutorV2Config:
    """All config for the bounded nonmyopic joint tutor."""
    # ── v4 base ──
    lambda_s: float = 1.0
    lambda_i: float = 2.0
    lambda_m: float = 1.5
    lambda_c: float = 0.05
    lambda_r: float = 0.3
    lambda_v: float = 2.0
    lambda_f: float = 1.5
    tau_v: float = 1.0
    tau_m: float = 1.0
    confidence_threshold: float = 0.7

    # ── v1.1 gating ──
    lambda_conf: float = 3.0
    tau_conf_margin: float = 0.15
    tau_conf_temp: float = 5.0
    tau_wait_opp: float = 1.0
    tau_wait_opp_temp: float = 1.5
    tau_obs_slack: float = 2.0
    tau_obs_slack_temp: float = 1.5

    # ── v2 joint-specific ──
    lambda_obs: float = 4.0       # one-step observation value weight
    lambda_joint: float = 1.5     # joint info gain from warning
    lambda_conflict: float = 2.5  # conflict mass × temptation
    lambda_miss: float = 1.0      # miss-window penalty for lookahead
    lambda_t: float = 2.0         # temptation risk (gated)
    lambda_p: float = 1.5         # preference info (gated)


class JointTutorV2:
    """Bounded one-step nonmyopic tutor over coupled q(g,θ).

    Explicitly compares:
      WARN_NOW: intervene immediately
      WAIT_1:   observe agent's next action, update q(g,θ), then decide
    """

    def __init__(
        self,
        config: Optional[JointTutorV2Config] = None,
        agent_params: Optional[AgentPolicyParams] = None,
        use_coupled: bool = True,
    ):
        self.cfg = config or JointTutorV2Config()
        self.agent_params = agent_params or AgentPolicyParams()
        self.joint_posterior = JointPosteriorV2()
        self.use_coupled = use_coupled
        self.warn_count = 0
        self.wait_count = 0

    # ══════════════════════════════════════════════
    # One-step nonmyopic core
    # ══════════════════════════════════════════════

    def _predictive_action_probs(
        self, branches: list[BranchAttributes],
    ) -> np.ndarray:
        """p(a_{t+1} | h_t) = Σ_{g,θ} q_t(g,θ) · P_A(a | s,g,θ).

        Returns array of shape [n_branches].
        """
        q = self.joint_posterior.table  # [N_GOALS, N_PREF]
        n_b = len(branches)
        p_action = np.zeros(n_b)
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                w = q[gi, pi]
                if w < 1e-12:
                    continue
                for ai in range(n_b):
                    lik = compute_joint_likelihood(ai, branches, g, p, self.agent_params)
                    p_action[ai] += w * lik
        return p_action

    def _posterior_after_hypothetical(
        self, action_idx: int, branches: list[BranchAttributes],
    ) -> JointPosteriorV2:
        """q_{t+1}^(a)(g,θ) ∝ q_t(g,θ) · P_A(a | s, g, θ)."""
        hypo = JointPosteriorV2(
            log_table=self.joint_posterior.log_table.copy(),
            observation_count=self.joint_posterior.observation_count,
            forgetting_rate=self.joint_posterior.forgetting_rate,
        )
        hypo.update_from_choice(action_idx, branches, self.agent_params)
        return hypo

    def _decision_bayes_risk(
        self, posterior: JointPosteriorV2,
        branches: list[BranchAttributes],
        safe_branch_idx: int = 0,
    ) -> float:
        """BR(q) = 1 - max(p_safe_branch, p_other_branch).

        p_safe_branch = Σ_{g,θ} q(g,θ) · P_A(safe | s,g,θ)
        """
        q = posterior.table
        p_safe = 0.0
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                w = q[gi, pi]
                if w < 1e-12:
                    continue
                lik = compute_joint_likelihood(
                    safe_branch_idx, branches, g, p, self.agent_params)
                p_safe += w * lik
        p_safe = np.clip(p_safe, 0, 1)
        return float(1.0 - max(p_safe, 1.0 - p_safe))

    def _one_step_obs_value(
        self, branches: list[BranchAttributes],
        safe_branch_idx: int = 0,
    ) -> float:
        """V_obs^(1) = BR(q_t) - E_a[BR(q_{t+1}^(a))].

        Expected decision Bayes Risk reduction from observing one more action.
        """
        br_current = self._decision_bayes_risk(
            self.joint_posterior, branches, safe_branch_idx)

        p_action = self._predictive_action_probs(branches)
        expected_br_after = 0.0
        n_b = len(branches)
        for ai in range(n_b):
            if p_action[ai] < 1e-12:
                continue
            hypo = self._posterior_after_hypothetical(ai, branches)
            br_after = self._decision_bayes_risk(hypo, branches, safe_branch_idx)
            expected_br_after += p_action[ai] * br_after

        return max(br_current - expected_br_after, 0.0)

    # ══════════════════════════════════════════════
    # Gating quantities (from v1.1)
    # ══════════════════════════════════════════════

    def _joint_confidence(self) -> float:
        """C(q_joint): entropy + top-2 margin for joint table."""
        t = self.joint_posterior.table.ravel()
        H = self.joint_posterior.entropy
        H_max = self.joint_posterior.max_entropy
        H_norm = H / max(H_max, 1e-6)

        sorted_t = np.sort(t)[::-1]
        margin = float(sorted_t[0] - sorted_t[1]) if len(sorted_t) >= 2 else float(sorted_t[0])

        margin_gate = _sigmoid(
            (margin - self.cfg.tau_conf_margin) * self.cfg.tau_conf_temp)
        return float((1.0 - H_norm) * margin_gate)

    def _temptation_susceptibility(self) -> float:
        """R_tempt(q) = Σ_θ q_marginal(θ) · κ_θ."""
        q_pref = self.joint_posterior.marginal_pref
        r = 0.0
        for i, ptype in enumerate(PREFERENCE_TYPES):
            kappa = TEMPT_SUSCEPTIBILITY.get(ptype, 0.3)
            r += float(q_pref[i]) * kappa
        return r

    def _wait_opportunity(self, p_self: float, delta: int) -> float:
        delta_gate = _sigmoid(
            (delta - self.cfg.tau_wait_opp) / self.cfg.tau_wait_opp_temp)
        return float(p_self * delta_gate)

    def _obs_slack(self, d_commit: int) -> float:
        return _sigmoid(
            (d_commit - self.cfg.tau_obs_slack) / self.cfg.tau_obs_slack_temp)

    def _conflict_mass(self) -> float:
        """R_conflict = Σ_{g,θ} q(g,θ) · 1[goal and pref prefer different branches].

        Uses sign of goal vs pref reward on safety dimension.
        """
        from ..agents.goal_posterior_v1 import GOAL_REWARD
        from ..agents.stochastic_agent_policy import PREF_REWARD

        q = self.joint_posterior.table
        mass = 0.0
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                # Conflict = goal prefers safety but pref prefers temptation, or vice versa
                goal_safety = GOAL_REWARD[g][0]   # safety_bonus
                pref_safety = PREF_REWARD[p][0]   # safety_bonus
                goal_tempt = GOAL_REWARD[g][1]     # tempt_bonus
                pref_tempt = PREF_REWARD[p][1]     # tempt_bonus

                # Conflict if one pushes toward safe and other toward risky
                goal_dir = goal_safety - goal_tempt  # positive = prefers safe
                pref_dir = pref_safety - pref_tempt  # positive = prefers safe

                if goal_dir * pref_dir < 0:  # opposite signs = conflict
                    mass += float(q[gi, pi])
        return mass

    # ══════════════════════════════════════════════
    # Main decision
    # ══════════════════════════════════════════════

    def _build_branch_attrs(self, sc, summary, is_risky: bool) -> BranchAttributes:
        tempt = getattr(sc, 'tempt_score_b' if is_risky else 'tempt_score_a', 0.0)
        return BranchAttributes(
            safety_score=float(summary[0]),
            temptation_score=tempt,
            texture_novelty=float(abs(summary[0] - 0.5)),
            shortcut_bonus=0.0,
            risk_penalty=float(1.0 - summary[0]) if is_risky else 0.0,
        )

    def decide(
        self,
        sc,
        fb_mod: np.ndarray,
        lp,
        lib: BranchConceptLibrary,
        scorer: BranchScorerProbe,
        obs_radius: int = 2,
    ) -> tuple[str, dict]:
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)
        delta = d_commit - d_reveal
        p_self = estimate_self_discovery_prob(d_commit, d_reveal, tau_v=cfg.tau_v)
        p_fail_wait = estimate_failure_if_wait(d_commit, d_reveal, tau_f=cfg.tau_v)

        # Branch summaries
        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs_radius)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs_radius)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        s_a_pre = summarize_branch(vis_a, fb_mod, fv, lp)
        s_b_pre = summarize_branch(vis_b, fb_mod, fv, lp)
        margin_pre = abs(s_a_pre[0] - s_b_pre[0])

        s_a_post = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
        margin_post = abs(s_a_post[0] - s_b_post[0])

        u_pre = _sigmoid(cfg.tau_m * margin_pre)
        u_post = _sigmoid(cfg.tau_m * margin_post)
        dvoi = max(u_post - u_pre, 0)
        delta_s = max(margin_post - margin_pre, 0)

        # Redundancy
        inp_a = build_scorer_input(s_a_pre, lib)
        inp_b = build_scorer_input(s_b_pre, lib)
        sc_a = scorer.score(inp_a)
        sc_b = scorer.score(inp_b)
        redundancy = max(abs(sc_a - sc_b) - cfg.confidence_threshold, 0)

        g_self = delta_s

        # Branch attributes
        is_a_risky = (getattr(sc, 'oracle_safe_branch_id', 0) != 0)
        ba_safe = self._build_branch_attrs(sc, s_a_pre, is_risky=is_a_risky)
        ba_risky = self._build_branch_attrs(sc, s_b_pre, is_risky=not is_a_risky)
        branches = [ba_safe, ba_risky]
        safe_idx = 0

        # ── Gating quantities ──
        C_q = self._joint_confidence()
        R_tempt = self._temptation_susceptibility()
        O_wait = self._wait_opportunity(p_self, delta)
        S_obs = self._obs_slack(d_commit)
        R_conflict = self._conflict_mass()

        # ── One-step nonmyopic observation value ──
        obs_value_1step = self._one_step_obs_value(branches, safe_idx)
        br_current = self._decision_bayes_risk(
            self.joint_posterior, branches, safe_idx)

        # Joint-specific terms
        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        joint_unc = self.joint_posterior.entropy / max(self.joint_posterior.max_entropy, 1e-6)

        raw_tempt_risk = tempt_str * joint_unc
        delta_joint = self.joint_posterior.posterior_predictive_variance(
            branches, self.agent_params) if joint_unc > 0.3 else 0.0

        # ══════════════════════════════════════════════
        # Q_warn
        # ══════════════════════════════════════════════
        Q_warn_base = (cfg.lambda_s * delta_s
                       + cfg.lambda_i * dvoi
                       + cfg.lambda_m * (1.0 - p_self)
                       - cfg.lambda_c * 1.0
                       - cfg.lambda_r * redundancy)

        # Joint info gain + conflict-aware temptation
        joint_bonus = cfg.lambda_joint * delta_joint
        tempt_bonus = cfg.lambda_t * (1.0 - O_wait) * R_tempt * raw_tempt_risk
        conflict_bonus = cfg.lambda_conflict * R_conflict * raw_tempt_risk
        prefinfo_bonus = cfg.lambda_p * (1.0 - C_q) * delta_joint

        Q_warn = Q_warn_base + joint_bonus + tempt_bonus + conflict_bonus + prefinfo_bonus

        # ══════════════════════════════════════════════
        # Q_wait
        # ══════════════════════════════════════════════
        Q_wait_base = (cfg.lambda_v * p_self * g_self
                       - cfg.lambda_f * p_fail_wait)

        # One-step observation value (decision-aware, gated by slack)
        obs_bonus = cfg.lambda_obs * S_obs * obs_value_1step
        # Confidence × opportunity
        conf_bonus = cfg.lambda_conf * C_q * O_wait
        # Miss-window penalty for lookahead
        miss_penalty = cfg.lambda_miss * p_fail_wait * (1.0 - S_obs)

        Q_wait = Q_wait_base + obs_bonus + conf_bonus - miss_penalty

        # ── Decision ──
        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        g_pred, p_pred = self.joint_posterior.predicted_joint
        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "Q_warn_base": round(Q_warn_base, 4), "Q_wait_base": round(Q_wait_base, 4),
            # One-step nonmyopic
            "obs_value_1step": round(obs_value_1step, 4),
            "br_current": round(br_current, 4),
            # Gating
            "C_q": round(C_q, 4), "R_tempt": round(R_tempt, 4),
            "O_wait": round(O_wait, 4), "S_obs": round(S_obs, 4),
            "R_conflict": round(R_conflict, 4),
            # Bonuses
            "joint_bonus": round(joint_bonus, 4),
            "tempt_bonus": round(tempt_bonus, 4),
            "conflict_bonus": round(conflict_bonus, 4),
            "obs_bonus": round(obs_bonus, 4),
            "conf_bonus": round(conf_bonus, 4),
            # State
            "p_self": round(p_self, 4), "delta": delta,
            "joint_entropy": round(self.joint_posterior.entropy, 4),
            "joint_unc": round(joint_unc, 4),
            "predicted_goal": g_pred, "predicted_pref": p_pred,
            "joint_conf": round(self.joint_posterior.joint_confidence, 4),
        }
        return action, diag

    def observe_agent_choice(self, chosen_idx: int, branches: list[BranchAttributes]):
        """Update joint posterior from observed action."""
        self.joint_posterior.update_from_choice(
            chosen_idx, branches, self.agent_params)

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
