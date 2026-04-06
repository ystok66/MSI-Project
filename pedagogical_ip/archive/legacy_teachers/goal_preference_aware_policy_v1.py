"""K3 — Goal + Preference Aware Tutor v1.

Unified tutor that considers:
  - Semantic uncertainty (selectivity law)
  - Preference uncertainty (ΔPrefInfo)
  - Goal uncertainty (ΔGoalInfo)
  - Observation value from both latent dimensions

Q_warn = v4_terms + λ_P·ΔPrefInfo + λ_G·ΔGoalInfo + λ_T·tempt_risk
Q_wait = v4_terms + λ_O·V_obs(pref+goal) - λ_F·P(fail)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from ..agents.joint_latent_belief import JointLatentBelief
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


@dataclass
class GoalPrefTutorConfig:
    """Config for unified goal+preference tutor."""
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
    lambda_p: float = 1.0     # preference info
    lambda_g: float = 1.0     # goal info
    lambda_t: float = 2.0     # temptation risk
    lambda_o: float = 2.0     # observation value
    lambda_a: float = 0.3     # autonomy


class GoalPreferenceAwarePolicyV1:
    """Unified tutor for semantic + preference + goal uncertainty."""

    def __init__(
        self,
        config: Optional[GoalPrefTutorConfig] = None,
        agent_params: Optional[AgentPolicyParams] = None,
    ):
        self.cfg = config or GoalPrefTutorConfig()
        self.agent_params = agent_params or AgentPolicyParams()
        self.joint_belief = JointLatentBelief()
        self.warn_count = 0
        self.wait_count = 0

    @property
    def pref_posterior(self):
        return self.joint_belief.pref_posterior

    @property
    def goal_posterior(self):
        return self.joint_belief.goal_posterior

    def decide(
        self,
        sc,
        fb_mod: np.ndarray,
        lp,
        lib: BranchConceptLibrary,
        scorer: BranchScorerProbe,
        obs_radius: int = 2,
    ) -> tuple[str, dict]:
        """Decide WAIT or WARN with goal + preference + semantic terms."""
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)
        p_self = estimate_self_discovery_prob(d_commit, d_reveal, tau_v=cfg.tau_v)
        p_fail_wait = estimate_failure_if_wait(d_commit, d_reveal, tau_f=cfg.tau_v)

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
        g_self = delta_s

        inp_a = build_scorer_input(s_a_pre, lib)
        inp_b = build_scorer_input(s_b_pre, lib)
        sc_a = scorer.score(inp_a)
        sc_b = scorer.score(inp_b)
        confidence = abs(sc_a - sc_b)
        redundancy = max(confidence - cfg.confidence_threshold, 0)

        # Build branch attributes
        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        is_a_risky = (getattr(sc, 'oracle_safe_branch_id', 0) != 0)
        ba_safe = BranchAttributes(
            safety_score=float(s_a_pre[0]),
            temptation_score=getattr(sc, 'tempt_score_a', 0.1),
            risk_penalty=0.1 if not is_a_risky else 0.4)
        ba_risky = BranchAttributes(
            safety_score=float(s_b_pre[0]),
            temptation_score=getattr(sc, 'tempt_score_b', tempt_str * 0.8),
            risk_penalty=0.4 if not is_a_risky else 0.1)
        branches = [ba_safe, ba_risky]

        # Multi-latent observation value
        obs_value = self.joint_belief.observation_value(branches, self.agent_params)

        # Latent uncertainties
        pref_unc = self.joint_belief.pref_entropy / max(self.pref_posterior.max_entropy, 1e-6)
        goal_unc = self.joint_belief.goal_entropy / max(self.goal_posterior.max_entropy, 1e-6)
        total_unc = self.joint_belief.total_uncertainty

        # Temptation risk
        tempt_risk = tempt_str * total_unc

        # ΔPrefInfo and ΔGoalInfo
        delta_pref = self.pref_posterior.expected_info_gain_from_observation(
            branches, self.agent_params) if pref_unc > 0.3 else 0.0
        delta_goal = self.goal_posterior.posterior_predictive_variance(
            branches, self.agent_params) if goal_unc > 0.3 else 0.0

        Q_warn = (cfg.lambda_s * delta_s
                  + cfg.lambda_i * dvoi
                  + cfg.lambda_m * (1.0 - p_self)
                  + cfg.lambda_p * delta_pref
                  + cfg.lambda_g * delta_goal
                  + cfg.lambda_t * tempt_risk
                  - cfg.lambda_c * 1.0
                  - cfg.lambda_r * redundancy)

        Q_wait = (cfg.lambda_v * p_self * g_self
                  + cfg.lambda_o * obs_value
                  + cfg.lambda_a * total_unc
                  - cfg.lambda_f * p_fail_wait)

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "p_self": round(p_self, 4), "dvoi": round(dvoi, 4),
            "delta_s": round(delta_s, 4),
            "margin_pre": round(margin_pre, 4), "margin_post": round(margin_post, 4),
            "d_commit": d_commit, "d_reveal": d_reveal,
            "obs_value": round(obs_value, 4),
            "delta_pref": round(delta_pref, 4),
            "delta_goal": round(delta_goal, 4),
            "pref_unc": round(pref_unc, 4),
            "goal_unc": round(goal_unc, 4),
            "total_unc": round(total_unc, 4),
            "pref_predicted": self.pref_posterior.predicted_type,
            "goal_predicted": self.goal_posterior.predicted_type,
        }
        return action, diag

    def observe_agent_choice(self, chosen_idx: int, branches: list[BranchAttributes]):
        """Update both posteriors from observed agent choice."""
        self.joint_belief.update_from_choice(chosen_idx, branches, self.agent_params)

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)
