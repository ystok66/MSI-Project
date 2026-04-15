"""Step-level diagnostic logger for TPM analysis.

Structured per-step logging with 4 temporal phases per timestep:
  1. pre_decision   — agent state BEFORE tutor decides
  2. post_intervention — state AFTER tutor acts, BEFORE agent moves
  3. post_transition   — state AFTER agent moves, BEFORE learning update
  4. post_learning     — state AFTER online learner updates weights

Each record is one JSONL line containing all 4 phases for one timestep.

Metric definitions:
  theta_t = (w_c in R^d_f, b_c in R, w_r in R^d_f, b_r in R)
  dim(theta_t) = 2*d_f + 2  (e.g., d_f=4 => 10 scalars)

  delta_theta   = ||theta_{t+1} - theta_t||_2
  delta_theta_c = ||w_c_{t+1} - w_c_t||_2  (cost head only)
  delta_theta_r = ||w_r_{t+1} - w_r_t||_2  (risk head only)

  delta_B_latent = (1/|D_t|) * sum_i ||mu_{i,t+1} - mu_{i,t}||_2
    where D_t = decision-relevant cells (top-k prefix)

  delta_B_pred = (1/|D_t|) * sum_i (|r_hat_{i,t+1} - r_hat_{i,t}|
                                     + |c_hat_{i,t+1} - c_hat_{i,t}|)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..agents.predictor_protocol import (
    extract_theta, extract_theta_components,
)


@dataclass
class PreDecisionPhase:
    """State before tutor decides (after observe, before apply_tutor)."""
    pos: tuple[int, int] = (0, 0)
    obs_quality: float = 0.0          # best patch quality at current pos
    route_necessity: float = 0.0
    # Belief summary for next cell on path
    c_hat_next: float = 0.0           # predicted cost at next cell
    r_hat_next: float = 0.0           # predicted risk at next cell
    u_c_next: float = 0.0             # cost uncertainty at next cell
    u_r_next: float = 0.0             # risk uncertainty at next cell
    # Learner weights snapshot
    theta_pre: list = field(default_factory=list)  # [w_c..., b_c, w_r..., b_r]
    # Chosen path prefix (first 5 cells of A* path) before intervention
    path_prefix_pre: list = field(default_factory=list)
    # Tutor scoring (all from robot belief)
    q_wait: float = 0.0
    q_warn: float = 0.0
    q_unlock: float = 0.0
    q_item: float = 0.0
    # Bottleneck diagnosis
    s_epi: float = 0.0
    s_str: float = 0.0
    s_out: float = 0.0
    dominant_bottleneck: str = ""


@dataclass
class PostInterventionPhase:
    """State after tutor acts, before agent moves."""
    selected_action: str = "WAIT"
    # Score decomposition
    raw_q: dict = field(default_factory=dict)
    match_bonus: dict = field(default_factory=dict)
    redundancy_penalty: float = 0.0
    warn_damping: float = 0.0
    warn_repeat_penalty: float = 0.0
    unlock_memory_penalty: float = 0.0
    final_q: dict = field(default_factory=dict)
    action_available: dict = field(default_factory=dict)
    decision_margin: float = 0.0
    # Belief after intervention (may differ if WARN updated belief)
    theta_post_intervention: list = field(default_factory=list)


@dataclass
class PostTransitionPhase:
    """State after agent moves, before learning update."""
    next_pos: tuple[int, int] = (0, 0)
    event: str = "safe"               # safe | hazard_survived | shield_used | death | goal | timeout
    true_cost: float = 1.0
    true_risk: float = 0.0
    # Effective observation quality at new position
    effective_obs_quality: float = 0.0


@dataclass
class PostLearningPhase:
    """State after online learner updates weights."""
    theta_post: list = field(default_factory=list)
    # Weight update magnitudes
    delta_theta: float = 0.0            # ||theta_post - theta_pre||_2
    delta_theta_c: float = 0.0          # ||w_c_post - w_c_pre||_2
    delta_theta_r: float = 0.0          # ||w_r_post - w_r_pre||_2
    # Decision-relevant belief change
    delta_B_latent: float = 0.0         # mean ||mu_post - mu_pre|| over prefix cells
    delta_B_pred: float = 0.0           # mean (|r_hat_post-r_hat_pre| + |c_hat_post-c_hat_pre|)


@dataclass
class StepRecord:
    """One complete timestep record with all 4 phases."""
    episode_id: str = ""
    family: str = ""
    difficulty: str = "medium"
    condition: str = ""
    seed: int = 0
    t: int = 0
    pre_decision: PreDecisionPhase = field(default_factory=PreDecisionPhase)
    post_intervention: PostInterventionPhase = field(default_factory=PostInterventionPhase)
    post_transition: PostTransitionPhase = field(default_factory=PostTransitionPhase)
    post_learning: PostLearningPhase = field(default_factory=PostLearningPhase)


class StepLogger:
    """Accumulates step records for one episode. Write to JSONL on close.

    Usage:
        logger = StepLogger("hazard_belt", "medium", "robot_belief_post", seed=7)
        # After each sub-step, call the appropriate record method
        logger.record_pre_decision(s)
        logger.record_post_intervention(s)
        logger.record_post_transition(s)
        logger.record_post_learning(s)
        logger.flush()  # write to JSONL
    """

    def __init__(self, family: str, difficulty: str, condition: str,
                 seed: int, output_dir: str = "results/step_logs"):
        self.family = family
        self.difficulty = difficulty
        self.condition = condition
        self.seed = seed
        self.episode_id = f"{family}_{condition}_s{seed}"
        self.output_dir = Path(output_dir)
        self.records: list[StepRecord] = []
        self._current: Optional[StepRecord] = None
        self._belief_snapshot_pre: Optional[np.ndarray] = None
        self._pred_snapshot_pre: Optional[dict] = None

    def _get_theta(self, lp) -> list:
        """Extract theta vector from any predictor type (dynamic-dim)."""
        return extract_theta(lp)

    def _get_theta_components(self, lp):
        """Return (w_c, b_c, w_r, b_r) as numpy arrays (dynamic-dim)."""
        return extract_theta_components(lp)

    def record_pre_decision(self, s):
        """Phase 1: Before tutor decision. Call after observe(), before apply_tutor()."""
        rec = StepRecord(
            episode_id=self.episode_id,
            family=self.family,
            difficulty=self.difficulty,
            condition=self.condition,
            seed=self.seed,
            t=s.t,
        )

        pre = rec.pre_decision
        pre.pos = tuple(s.agent_pos)

        # Obs quality: 1.0 = self position
        pre.obs_quality = 1.0

        # Theta snapshot
        lp = s.latent_predictor
        pre.theta_pre = self._get_theta(lp)

        # Route necessity
        from ..agents.route_necessity import compute_route_necessity
        if lp is not None and s.t_max > 0:
            unvisited = set()
            for r in range(s.gridmap.height):
                for c in range(s.gridmap.width):
                    if (s.feature_belief.visit_count[r, c] == 0
                            and s.passable[r, c]
                            and s.gridmap.cell_types[r, c] != 0):  # not WALL
                        unvisited.add((r, c))
            pre.route_necessity = compute_route_necessity(
                s.agent_pos, s.goal, s.passable,
                t=s.t, t_max=s.t_max, route_cells=unvisited)
        else:
            pre.route_necessity = 0.0

        # Belief snapshot for delta_B computation
        self._belief_snapshot_pre = s.feature_belief.mean.copy()

        # Predictions at next cell on path (if available)
        if lp is not None and s.last_belief_plan is not None:
            path = s.last_belief_plan.full_path
            if len(path) > 1:
                nr, nc = path[1]
                x = s.feature_belief.get_mean(nr, nc)
                x_var = s.feature_belief.var[nr, nc]
                pre.c_hat_next = round(float(lp.predict_cost(x)), 4)
                pre.r_hat_next = round(float(lp.predict_risk(x)), 4)
                pre.u_c_next = round(float(lp.predict_cost_uncertainty_from_var(x_var)), 4)
                pre.u_r_next = round(float(lp.predict_risk_uncertainty_from_var(x_var)), 4)
                pre.path_prefix_pre = [tuple(p) for p in path[:5]]

        # Prediction snapshot for delta_B_pred
        self._pred_snapshot_pre = {}
        if lp is not None and pre.path_prefix_pre:
            for p in pre.path_prefix_pre:
                x = s.feature_belief.get_mean(p[0], p[1])
                self._pred_snapshot_pre[p] = (
                    float(lp.predict_cost(x)),
                    float(lp.predict_risk(x)),
                )

        self._current = rec

    def record_post_intervention(self, s):
        """Phase 2: After tutor acts, before agent moves."""
        if self._current is None:
            return
        post = self._current.post_intervention

        if s.last_intervention is not None:
            d = s.last_intervention
            post.selected_action = d.action
            post.decision_margin = round(d.decision_margin, 4)

            # Bottleneck scores
            if d.bottleneck is not None:
                self._current.pre_decision.s_epi = round(d.bottleneck.epistemic, 4)
                self._current.pre_decision.s_str = round(d.bottleneck.structural, 4)
                self._current.pre_decision.s_out = round(d.bottleneck.outcome, 4)
                self._current.pre_decision.dominant_bottleneck = d.bottleneck.dominant

            # Score decomposition
            if d.score_decomposition is not None:
                sd = d.score_decomposition
                post.raw_q = {k: round(v, 4) for k, v in sd["raw_q"].items()}
                post.match_bonus = {k: round(v, 4) for k, v in sd["match_bonus"].items()}
                post.redundancy_penalty = round(sd["redundancy_penalty"], 4)
                post.warn_damping = round(sd["warn_damping"], 4)
                post.warn_repeat_penalty = round(sd["warn_repeat_penalty"], 4)
                post.unlock_memory_penalty = round(sd["unlock_memory_penalty"], 4)
                post.final_q = {k: round(v, 4) for k, v in sd["final_q"].items()}
                post.action_available = sd["action_available"]

            # Tutor scores (for pre_decision phase — filled here since tutor runs first)
            if d.scores:
                self._current.pre_decision.q_wait = round(d.scores.get("WAIT", 0), 4)
                self._current.pre_decision.q_warn = round(d.scores.get("WARN", 0), 4)
                self._current.pre_decision.q_unlock = round(d.scores.get("UNLOCK", 0), 4)
                self._current.pre_decision.q_item = round(d.scores.get("ITEM_DROP", 0), 4)

        else:
            post.selected_action = "NONE"

        # Theta after intervention (usually same as pre)
        post.theta_post_intervention = self._get_theta(s.latent_predictor)

    def record_post_transition(self, s, event: str = "safe",
                               true_cost: float = 1.0, true_risk: float = 0.0):
        """Phase 3: After agent moves, before learning update."""
        if self._current is None:
            return
        pt = self._current.post_transition
        pt.next_pos = tuple(s.agent_pos)
        pt.event = event
        pt.true_cost = round(true_cost, 4)
        pt.true_risk = round(true_risk, 4)
        pt.effective_obs_quality = 1.0

    def record_post_learning(self, s):
        """Phase 4: After online learner updates weights."""
        if self._current is None:
            return
        pl = self._current.post_learning
        lp = s.latent_predictor

        # Theta post
        pl.theta_post = self._get_theta(lp)

        # Delta theta
        theta_pre = np.array(self._current.pre_decision.theta_pre)
        theta_post = np.array(pl.theta_post)
        if len(theta_pre) > 0 and len(theta_post) > 0:
            pl.delta_theta = round(float(np.linalg.norm(theta_post - theta_pre)), 6)

            # Component-wise: use actual weight dimensions (dynamic)
            w_c_pre, _, w_r_pre, _ = extract_theta_components(lp)
            # Re-extract post (lp may have been updated)
            theta_post_arr = np.array(pl.theta_post)
            d_c = len(w_c_pre)
            d_r = len(w_r_pre)

            # theta layout: [w_c(d_c), b_c(1), w_r(d_r), b_r(1)]
            if len(theta_pre) >= d_c + 1 + d_r + 1 and len(theta_post_arr) >= d_c + 1 + d_r + 1:
                wc_pre_v = theta_pre[:d_c]
                wc_post_v = theta_post_arr[:d_c]
                wr_pre_v = theta_pre[d_c + 1:d_c + 1 + d_r]
                wr_post_v = theta_post_arr[d_c + 1:d_c + 1 + d_r]
                pl.delta_theta_c = round(float(np.linalg.norm(wc_post_v - wc_pre_v)), 6)
                pl.delta_theta_r = round(float(np.linalg.norm(wr_post_v - wr_pre_v)), 6)

        # Delta B (decision-relevant belief change)
        if self._belief_snapshot_pre is not None:
            prefix = self._current.pre_decision.path_prefix_pre
            if prefix:
                delta_lat = 0.0
                delta_pred = 0.0
                for p in prefix:
                    r, c = p
                    if r < s.feature_belief.mean.shape[0] and c < s.feature_belief.mean.shape[1]:
                        mu_pre = self._belief_snapshot_pre[r, c]
                        mu_post = s.feature_belief.mean[r, c]
                        delta_lat += float(np.linalg.norm(mu_post - mu_pre))

                        if lp is not None and p in self._pred_snapshot_pre:
                            c_pre, r_pre = self._pred_snapshot_pre[p]
                            x_post = s.feature_belief.get_mean(r, c)
                            c_post = float(lp.predict_cost(x_post))
                            r_post = float(lp.predict_risk(x_post))
                            delta_pred += abs(r_post - r_pre) + abs(c_post - c_pre)

                n = max(len(prefix), 1)
                pl.delta_B_latent = round(delta_lat / n, 6)
                pl.delta_B_pred = round(delta_pred / n, 6)

        # Finalize record
        self.records.append(self._current)
        self._current = None

    def flush(self, filepath: Optional[str] = None):
        """Write all records to JSONL."""
        if filepath is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{self.episode_id}.jsonl"

        with open(filepath, "w") as f:
            for rec in self.records:
                d = asdict(rec)
                # Convert numpy arrays that leak through
                f.write(json.dumps(d, default=_json_default) + "\n")

    @property
    def summary(self) -> dict:
        """Episode-level summary from step records."""
        if not self.records:
            return {}
        dthetas = [r.post_learning.delta_theta for r in self.records]
        dthetas_c = [r.post_learning.delta_theta_c for r in self.records]
        dthetas_r = [r.post_learning.delta_theta_r for r in self.records]
        dB_lat = [r.post_learning.delta_B_latent for r in self.records]
        dB_pred = [r.post_learning.delta_B_pred for r in self.records]

        # BAR: bottleneck-action alignment rate
        action_match = {"WARN": "epistemic", "UNLOCK": "structural", "ITEM_DROP": "outcome"}
        non_wait = [r for r in self.records if r.post_intervention.selected_action != "WAIT"
                    and r.post_intervention.selected_action != "NONE"]
        bar = 0.0
        if non_wait:
            matches = sum(1 for r in non_wait
                          if action_match.get(r.post_intervention.selected_action)
                          == r.pre_decision.dominant_bottleneck)
            bar = matches / len(non_wait)

        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "condition": self.condition,
            "seed": self.seed,
            "n_steps": len(self.records),
            "mean_delta_theta": round(float(np.mean(dthetas)), 6) if dthetas else 0.0,
            "max_delta_theta": round(float(np.max(dthetas)), 6) if dthetas else 0.0,
            "mean_delta_theta_c": round(float(np.mean(dthetas_c)), 6) if dthetas_c else 0.0,
            "mean_delta_theta_r": round(float(np.mean(dthetas_r)), 6) if dthetas_r else 0.0,
            "mean_delta_B_latent": round(float(np.mean(dB_lat)), 6) if dB_lat else 0.0,
            "mean_delta_B_pred": round(float(np.mean(dB_pred)), 6) if dB_pred else 0.0,
            "bar": round(bar, 4),
            "n_non_wait": len(non_wait),
        }


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
