"""PP-v1.1 — Confidence-Gated Persistent Tutor.

Decision-Boundary Repair: translates posterior learning into
actionable selectivity via 4 gating quantities:

  1. C(q):       posterior confidence (entropy × top-2 margin)
  2. R_tempt(q): posterior-weighted temptation susceptibility
  3. O_wait:     wait opportunity (p_self × Δ-gate)
  4. S_obs:      observation slack (time budget for future obs)

Q_warn = Q_warn^v4  + λ_T·(1-O_wait)·R_tempt·tempt_risk
                    + λ_P·(1-C(q))·ΔPrefInfo
Q_wait = Q_wait^v4  + λ_O·S_obs·V_obs
                    + λ_conf·C(q)·O_wait
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
from ..agents.preference_posterior_v2 import PreferencePosteriorV2
from ..envs.observation_mask import make_observation_mask
from ..metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


# Temptation susceptibility table — extensible to new latent types
TEMPT_SUSCEPTIBILITY = {
    "safe":     0.10,
    "neutral":  0.30,
    "risky":    0.70,
    "shortcut": 0.60,
    "shiny":    1.00,
}


@dataclass
class PersistentTutorV1_1Config:
    """All config-driven weights for v1.1."""
    # ── Inherited v4 base ──
    lambda_s: float = 1.0       # success gain
    lambda_i: float = 2.0       # DVOI
    lambda_m: float = 1.5       # missed-window
    lambda_c: float = 0.05      # intervention cost
    lambda_r: float = 0.3       # redundancy penalty
    lambda_v: float = 2.0       # self-discovery value
    lambda_f: float = 1.5       # failure-if-wait penalty
    tau_v: float = 1.0          # p_self temperature
    tau_m: float = 1.0          # DVOI margin sensitivity
    confidence_threshold: float = 0.7

    # ── New v1.1 gating quantities ──
    lambda_conf: float = 3.0    # confidence × opportunity → WAIT bonus
    lambda_t: float = 2.0       # temptation risk (gated)
    lambda_p: float = 1.5       # preference info (gated by uncertainty)
    lambda_o: float = 1.5       # observation value (gated by slack)

    # ── Gating temperatures ──
    tau_conf_margin: float = 0.15   # top-2 margin threshold for C(q)
    tau_conf_temp: float = 5.0      # sigmoid sharpness for margin gate
    tau_wait_opp: float = 1.0       # Δ threshold for wait opportunity
    tau_wait_opp_temp: float = 1.5  # sigmoid sharpness for Δ gate
    tau_obs_slack: float = 2.0      # buffer steps for observation slack
    tau_obs_slack_temp: float = 1.5 # sigmoid sharpness


class PersistentTutorV1_1:
    """Confidence-gated persistent tutor.

    Core principle: posterior learning (q(θ) sharpening) must translate
    into actionable selectivity — warn less on WAIT-favorable episodes,
    warn promptly on WARN-necessary episodes.
    """

    def __init__(
        self,
        config: Optional[PersistentTutorV1_1Config] = None,
        agent_params: Optional[AgentPolicyParams] = None,
        # Ablation flags
        enable_confidence: bool = True,
        enable_susceptibility: bool = True,
        enable_opportunity: bool = True,
        enable_obs_slack: bool = True,
    ):
        self.cfg = config or PersistentTutorV1_1Config()
        self.agent_params = agent_params or AgentPolicyParams()
        self.pref_posterior = PreferencePosteriorV2()
        self.warn_count = 0
        self.wait_count = 0

        # Ablation controls
        self._conf = enable_confidence
        self._sus = enable_susceptibility
        self._opp = enable_opportunity
        self._obs = enable_obs_slack

    # ─── Gating quantity 1: Posterior Confidence C(q) ───

    def _confidence(self) -> float:
        """C(q) = (1 - H̄(q)) · σ((margin - τ_m) / τ_c)"""
        if not self._conf:
            return 0.0

        q = self.pref_posterior.probs
        H_norm = self.pref_posterior.entropy / max(self.pref_posterior.max_entropy, 1e-6)

        # Top-2 margin
        sorted_q = np.sort(q)[::-1]
        margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) >= 2 else float(sorted_q[0])

        margin_gate = _sigmoid(
            (margin - self.cfg.tau_conf_margin) * self.cfg.tau_conf_temp)

        return float((1.0 - H_norm) * margin_gate)

    # ─── Gating quantity 2: Temptation Susceptibility R_tempt(q) ───

    def _temptation_susceptibility(self) -> float:
        """R_tempt(q) = Σ_θ q(θ)·κ_θ"""
        if not self._sus:
            return 0.5  # neutral fallback

        q = self.pref_posterior.probs
        r = 0.0
        for i, ptype in enumerate(PREFERENCE_TYPES):
            kappa = TEMPT_SUSCEPTIBILITY.get(ptype, 0.3)
            r += float(q[i]) * kappa
        return r

    # ─── Gating quantity 3: Wait Opportunity O_wait ───

    def _wait_opportunity(self, p_self: float, delta: int) -> float:
        """O_wait = p_self · σ((Δ - τ_o) / τ_o')"""
        if not self._opp:
            return 0.0

        delta_gate = _sigmoid(
            (delta - self.cfg.tau_wait_opp) / self.cfg.tau_wait_opp_temp)
        return float(p_self * delta_gate)

    # ─── Gating quantity 4: Observation Slack S_obs ───

    def _obs_slack(self, d_commit: int) -> float:
        """S_obs = σ((d_commit - b_obs) / τ_s)"""
        if not self._obs:
            return 1.0  # always on

        return _sigmoid(
            (d_commit - self.cfg.tau_obs_slack) / self.cfg.tau_obs_slack_temp)

    # ─── Main decision ───

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

        # ── Visible vs full summaries ──
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
        confidence_scorer = abs(sc_a - sc_b)
        redundancy = max(confidence_scorer - cfg.confidence_threshold, 0)

        g_self = max(margin_post - margin_pre, 0)

        # ── 4 Gating Quantities ──
        C_q = self._confidence()
        R_tempt = self._temptation_susceptibility()
        O_wait = self._wait_opportunity(p_self, delta)
        S_obs = self._obs_slack(d_commit)

        # ── Preference-aware terms ──
        is_a_risky = (sc.oracle_safe_branch_id != 0)
        ba_safe = self._build_branch_attrs(sc, s_a_pre, is_risky=is_a_risky)
        ba_risky = self._build_branch_attrs(sc, s_b_pre, is_risky=not is_a_risky)
        branches = [ba_safe, ba_risky]

        obs_value = self.pref_posterior.posterior_predictive_variance(
            branches, self.agent_params)

        pref_entropy = self.pref_posterior.entropy
        pref_uncertainty = pref_entropy / max(self.pref_posterior.max_entropy, 1e-6)

        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        raw_tempt_risk = tempt_str * pref_uncertainty

        # ΔPrefInfo
        delta_pref = self.pref_posterior.expected_info_gain_from_observation(
            branches, self.agent_params) if pref_uncertainty > 0.3 else 0.0

        # ══════════════════════════════════════════════
        # Q_warn = v4_base + gated_tempt + gated_prefinfo
        # ══════════════════════════════════════════════
        Q_warn_base = (cfg.lambda_s * delta_s
                       + cfg.lambda_i * dvoi
                       + cfg.lambda_m * (1.0 - p_self)
                       - cfg.lambda_c * 1.0
                       - cfg.lambda_r * redundancy)

        # Temptation: gated by (1-O_wait) × susceptibility
        tempt_bonus = cfg.lambda_t * (1.0 - O_wait) * R_tempt * raw_tempt_risk
        # PrefInfo: gated by (1-C(q)) — only valuable when uncertain
        prefinfo_bonus = cfg.lambda_p * (1.0 - C_q) * delta_pref

        Q_warn = Q_warn_base + tempt_bonus + prefinfo_bonus

        # ══════════════════════════════════════════════
        # Q_wait = v4_base + gated_obs + confidence×opportunity
        # ══════════════════════════════════════════════
        Q_wait_base = (cfg.lambda_v * p_self * g_self
                       - cfg.lambda_f * p_fail_wait)

        # Observation value: gated by slack
        obs_bonus = cfg.lambda_o * S_obs * obs_value
        # Confidence × opportunity: "I know this learner AND it's safe to wait"
        conf_bonus = cfg.lambda_conf * C_q * O_wait

        Q_wait = Q_wait_base + obs_bonus + conf_bonus

        # ── Decision ──
        action = "WARN" if Q_warn > Q_wait else "WAIT"

        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1

        diag = {
            "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
            "Q_warn_base": round(Q_warn_base, 4), "Q_wait_base": round(Q_wait_base, 4),
            # Gating quantities
            "C_q": round(C_q, 4),
            "R_tempt": round(R_tempt, 4),
            "O_wait": round(O_wait, 4),
            "S_obs": round(S_obs, 4),
            # Bonus decomposition
            "tempt_bonus": round(tempt_bonus, 4),
            "prefinfo_bonus": round(prefinfo_bonus, 4),
            "obs_bonus": round(obs_bonus, 4),
            "conf_bonus": round(conf_bonus, 4),
            # Base components
            "p_self": round(p_self, 4), "dvoi": round(dvoi, 4),
            "delta_s": round(delta_s, 4),
            "margin_pre": round(margin_pre, 4), "margin_post": round(margin_post, 4),
            "d_commit": d_commit, "d_reveal": d_reveal, "delta": delta,
            "pref_entropy": round(pref_entropy, 4),
            "pref_predicted": self.pref_posterior.predicted_type,
            "pref_confidence": round(self.pref_posterior.predicted_prob, 4),
        }
        return action, diag

    def observe_agent_choice(
        self, chosen_idx: int, branches: list[BranchAttributes],
    ):
        """Update posterior from observed agent choice (persistent carry)."""
        self.pref_posterior.update_from_choice(
            chosen_idx, branches, self.agent_params)

    @property
    def warn_rate(self) -> float:
        total = self.warn_count + self.wait_count
        return self.warn_count / max(total, 1)

    def reset_stats(self):
        self.warn_count = 0
        self.wait_count = 0
