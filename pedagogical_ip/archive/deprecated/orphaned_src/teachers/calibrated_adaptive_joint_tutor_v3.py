"""CAJT-v3: Calibrated Adaptive Joint Tutor.

Upgrades over joint_v2:
  A. Calibration: decision uses q̃_t = softmax(log q_t / T), not raw q_t
  B. Adaptive memory: forgetting ρ_t driven by surprisal-based drift D_t
  C. V_adapt: observation value weighted by change detection potential

Q_wait = v2_base + λ_cal·C_t·O_wait + λ_obs·S_obs·V_obs^(1)
                  + λ_adapt·S_obs·V_adapt^(1) - λ_miss·P(miss)
Q_warn = v2_base + λ_conflict·R_conflict·tempt
                  + λ_unc·(1-C_t)·ΔInfo + λ_shift·D_t·WarnNowGain
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from ..agents.branch_concepts import BranchConceptLibrary
from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES,
)
from ..agents.joint_posterior_v2 import (
    JointPosteriorV2, compute_joint_likelihood,
    GOAL_TYPES, N_GOALS, N_PREF,
)
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from ..metrics.change_detection import (
    compute_surprisal, compute_drift_score, compute_adaptive_rho,
    apply_adaptive_diffusion,
)
from ..metrics.calibrated_confidence import (
    calibrate_posterior, calibrated_confidence, calibrated_entropy,
)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


TEMPT_SUSCEPTIBILITY = {
    "safe": 0.10, "neutral": 0.30, "risky": 0.70,
    "shortcut": 0.60, "shiny": 1.00,
}


@dataclass
class CAJTv3Config:
    # v4 base
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
    # v1.1 gating
    tau_conf_margin: float = 0.15
    tau_conf_temp: float = 5.0
    tau_wait_opp: float = 1.0
    tau_wait_opp_temp: float = 1.5
    tau_obs_slack: float = 2.0
    tau_obs_slack_temp: float = 1.5
    # v2 joint
    lambda_obs: float = 4.0
    lambda_joint: float = 1.5
    lambda_conflict: float = 2.5
    lambda_miss: float = 1.0
    lambda_t: float = 2.0
    lambda_p: float = 1.5
    lambda_conf: float = 3.0
    # v3 calibration + adaptive
    T_joint: float = 0.6          # calibration temperature (< 1 sharpens)
    lambda_cal: float = 3.5       # calibrated confidence → WAIT
    lambda_adapt: float = 2.0     # adaptive observation value
    lambda_shift: float = 1.5     # drift-aware warn bonus
    tau_surprisal: float = 1.5    # surprisal threshold for drift
    tau_drift: float = 2.0        # drift sigmoid sharpness
    rho_min: float = 0.005        # min forgetting
    rho_max: float = 0.15         # max forgetting (during drift)


class CAJTv3:
    """Calibrated Adaptive Joint Tutor v3."""

    def __init__(
        self,
        config: Optional[CAJTv3Config] = None,
        agent_params: Optional[AgentPolicyParams] = None,
        enable_calibration: bool = True,
        enable_adaptive: bool = True,
        enable_drift_warn: bool = True,
    ):
        self.cfg = config or CAJTv3Config()
        self.agent_params = agent_params or AgentPolicyParams()
        self.joint_posterior = JointPosteriorV2(forgetting_rate=0.0)  # we handle forgetting
        self.warn_count = 0
        self.wait_count = 0
        self.last_surprisal = 0.0
        self.last_drift = 0.0
        self.last_rho = self.cfg.rho_min
        # ablation
        self._cal = enable_calibration
        self._adapt = enable_adaptive
        self._drift = enable_drift_warn

    def _calibrated_table(self) -> np.ndarray:
        if self._cal:
            return calibrate_posterior(self.joint_posterior.log_table, self.cfg.T_joint)
        return self.joint_posterior.table

    def _joint_confidence(self) -> float:
        ct = self._calibrated_table()
        return calibrated_confidence(ct, self.cfg.tau_conf_margin, self.cfg.tau_conf_temp)

    def _temptation_susceptibility(self) -> float:
        ct = self._calibrated_table()
        q_pref = ct.sum(axis=0)  # marginal over goals
        r = 0.0
        for i, p in enumerate(PREFERENCE_TYPES):
            r += float(q_pref[i]) * TEMPT_SUSCEPTIBILITY.get(p, 0.3)
        return r

    def _wait_opportunity(self, p_self: float, delta: int) -> float:
        dg = _sigmoid((delta - self.cfg.tau_wait_opp) / self.cfg.tau_wait_opp_temp)
        return float(p_self * dg)

    def _obs_slack(self, d_commit: int) -> float:
        return _sigmoid((d_commit - self.cfg.tau_obs_slack) / self.cfg.tau_obs_slack_temp)

    def _conflict_mass(self) -> float:
        from ..agents.goal_posterior_v1 import GOAL_REWARD
        from ..agents.stochastic_agent_policy import PREF_REWARD
        ct = self._calibrated_table()
        mass = 0.0
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                gd = GOAL_REWARD[g][0] - GOAL_REWARD[g][1]
                pd = PREF_REWARD[p][0] - PREF_REWARD[p][1]
                if gd * pd < 0:
                    mass += float(ct[gi, pi])
        return mass

    def _predictive_action_probs(self, branches):
        ct = self._calibrated_table()
        n_b = len(branches)
        p_action = np.zeros(n_b)
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                w = ct[gi, pi]
                if w < 1e-12:
                    continue
                for ai in range(n_b):
                    lik = compute_joint_likelihood(ai, branches, g, p, self.agent_params)
                    p_action[ai] += w * lik
        return p_action

    def _posterior_after_hypothetical(self, action_idx, branches):
        hypo = JointPosteriorV2(
            log_table=self.joint_posterior.log_table.copy(),
            observation_count=self.joint_posterior.observation_count,
            forgetting_rate=0.0,
        )
        hypo.update_from_choice(action_idx, branches, self.agent_params)
        return hypo

    def _decision_bayes_risk(self, posterior, branches, safe_idx=0):
        q = posterior.table if not self._cal else calibrate_posterior(
            posterior.log_table, self.cfg.T_joint)
        p_safe = 0.0
        for gi, g in enumerate(GOAL_TYPES):
            for pi, p in enumerate(PREFERENCE_TYPES):
                w = q[gi, pi]
                if w < 1e-12:
                    continue
                lik = compute_joint_likelihood(safe_idx, branches, g, p, self.agent_params)
                p_safe += w * lik
        p_safe = np.clip(p_safe, 0, 1)
        return float(1.0 - max(p_safe, 1.0 - p_safe))

    def _one_step_obs_value(self, branches, safe_idx=0):
        br_now = self._decision_bayes_risk(self.joint_posterior, branches, safe_idx)
        p_act = self._predictive_action_probs(branches)
        exp_br = 0.0
        for ai in range(len(branches)):
            if p_act[ai] < 1e-12:
                continue
            hypo = self._posterior_after_hypothetical(ai, branches)
            exp_br += p_act[ai] * self._decision_bayes_risk(hypo, branches, safe_idx)
        return max(br_now - exp_br, 0.0)

    def _one_step_adapt_value(self, branches, safe_idx=0):
        """V_adapt = E_a[D_{t+1}^(a) · (BR(q_t) - BR(q_{t+1}^(a)))]."""
        if not self._adapt:
            return 0.0
        br_now = self._decision_bayes_risk(self.joint_posterior, branches, safe_idx)
        p_act = self._predictive_action_probs(branches)
        ct = self._calibrated_table()
        val = 0.0
        for ai in range(len(branches)):
            if p_act[ai] < 1e-12:
                continue
            # Compute hypothetical surprisal for action ai
            lik_per_cell = np.array([
                compute_joint_likelihood(ai, branches, GOAL_TYPES[gi], PREFERENCE_TYPES[pi],
                                        self.agent_params)
                for gi in range(N_GOALS) for pi in range(N_PREF)
            ])
            hypo_surp = compute_surprisal(ct.ravel(), lik_per_cell)
            hypo_drift = compute_drift_score(hypo_surp, self.cfg.tau_surprisal, self.cfg.tau_drift)
            hypo = self._posterior_after_hypothetical(ai, branches)
            br_after = self._decision_bayes_risk(hypo, branches, safe_idx)
            val += p_act[ai] * hypo_drift * max(br_now - br_after, 0.0)
        return val

    def _build_branch_attrs(self, sc, summary, is_risky):
        tempt = getattr(sc, 'tempt_score_b' if is_risky else 'tempt_score_a', 0.0)
        return BranchAttributes(
            safety_score=float(summary[0]), temptation_score=tempt,
            texture_novelty=float(abs(summary[0] - 0.5)), shortcut_bonus=0.0,
            risk_penalty=float(1.0 - summary[0]) if is_risky else 0.0)

    def decide(self, sc, fb_mod, lp, lib, scorer, obs_radius=2):
        fv = np.full_like(fb_mod, 0.3)
        cfg = self.cfg

        d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
        d_reveal = getattr(sc, 'reveal_depth', 3)
        delta = d_commit - d_reveal
        p_self = estimate_self_discovery_prob(d_commit, d_reveal, tau_v=cfg.tau_v)
        p_fail = estimate_failure_if_wait(d_commit, d_reveal, tau_f=cfg.tau_v)

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

        inp_a = build_scorer_input(s_a_pre, lib)
        inp_b = build_scorer_input(s_b_pre, lib)
        redundancy = max(abs(scorer.score(inp_a) - scorer.score(inp_b)) - cfg.confidence_threshold, 0)

        is_a_risky = (getattr(sc, 'oracle_safe_branch_id', 0) != 0)
        ba_safe = self._build_branch_attrs(sc, s_a_pre, is_risky=is_a_risky)
        ba_risky = self._build_branch_attrs(sc, s_b_pre, is_risky=not is_a_risky)
        branches = [ba_safe, ba_risky]

        # Gating
        C_t = self._joint_confidence()
        R_tempt = self._temptation_susceptibility()
        O_wait = self._wait_opportunity(p_self, delta)
        S_obs = self._obs_slack(d_commit)
        R_conflict = self._conflict_mass()
        D_t = self.last_drift

        # Observation values
        obs_val = self._one_step_obs_value(branches)
        adapt_val = self._one_step_adapt_value(branches)
        br_now = self._decision_bayes_risk(self.joint_posterior, branches)

        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        ct = self._calibrated_table()
        joint_unc = calibrated_entropy(ct) / max(np.log(N_GOALS * N_PREF), 1e-6)
        raw_tempt = tempt_str * joint_unc

        ppv = self.joint_posterior.posterior_predictive_variance(
            branches, self.agent_params) if joint_unc > 0.3 else 0.0

        # Q_warn
        Q_warn_base = (cfg.lambda_s * delta_s + cfg.lambda_i * dvoi
                       + cfg.lambda_m * (1.0 - p_self)
                       - cfg.lambda_c * 1.0 - cfg.lambda_r * redundancy)
        joint_bonus = cfg.lambda_joint * ppv
        tempt_bonus = cfg.lambda_t * (1.0 - O_wait) * R_tempt * raw_tempt
        conflict_bonus = cfg.lambda_conflict * R_conflict * raw_tempt
        unc_bonus = cfg.lambda_p * (1.0 - C_t) * ppv
        shift_bonus = cfg.lambda_shift * D_t * max(delta_s, dvoi) if self._drift else 0.0

        Q_warn = Q_warn_base + joint_bonus + tempt_bonus + conflict_bonus + unc_bonus + shift_bonus

        # Q_wait
        Q_wait_base = cfg.lambda_v * p_self * delta_s - cfg.lambda_f * p_fail
        cal_bonus = cfg.lambda_cal * C_t * O_wait
        obs_bonus = cfg.lambda_obs * S_obs * obs_val
        adapt_bonus = cfg.lambda_adapt * S_obs * adapt_val
        miss_pen = cfg.lambda_miss * p_fail * (1.0 - S_obs)

        Q_wait = Q_wait_base + cal_bonus + obs_bonus + adapt_bonus - miss_pen

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        g_pred, p_pred = self.joint_posterior.predicted_joint
        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "C_t": round(C_t, 4), "D_t": round(D_t, 4), "rho_t": round(self.last_rho, 4),
            "R_tempt": round(R_tempt, 4), "O_wait": round(O_wait, 4),
            "S_obs": round(S_obs, 4), "R_conflict": round(R_conflict, 4),
            "obs_val": round(obs_val, 4), "adapt_val": round(adapt_val, 4),
            "br_now": round(br_now, 4), "surprisal": round(self.last_surprisal, 4),
            "cal_bonus": round(cal_bonus, 4), "obs_bonus": round(obs_bonus, 4),
            "adapt_bonus": round(adapt_bonus, 4), "shift_bonus": round(shift_bonus, 4),
            "predicted_goal": g_pred, "predicted_pref": p_pred,
            "joint_conf": round(self.joint_posterior.joint_confidence, 4),
            "calib_top1": round(float(np.max(ct)), 4),
        }
        return action, diag

    def observe_agent_choice(self, chosen_idx, branches):
        """Update posterior with adaptive diffusion."""
        # Compute surprisal BEFORE update
        ct = self._calibrated_table()
        lik_per_cell = np.array([
            compute_joint_likelihood(chosen_idx, branches,
                                    GOAL_TYPES[gi], PREFERENCE_TYPES[pi], self.agent_params)
            for gi in range(N_GOALS) for pi in range(N_PREF)
        ])
        self.last_surprisal = compute_surprisal(ct.ravel(), lik_per_cell)
        self.last_drift = compute_drift_score(
            self.last_surprisal, self.cfg.tau_surprisal, self.cfg.tau_drift)
        self.last_rho = compute_adaptive_rho(
            self.last_drift, self.cfg.rho_min, self.cfg.rho_max)

        # Adaptive diffusion THEN Bayesian update
        if self._adapt:
            self.joint_posterior.log_table = apply_adaptive_diffusion(
                self.joint_posterior.log_table, self.last_rho)
        self.joint_posterior.update_from_choice(chosen_idx, branches, self.agent_params)

    @property
    def warn_rate(self):
        t = self.warn_count + self.wait_count
        return self.warn_count / max(t, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
