"""I1 — Adapter layer: bridges existing runner/planner/tutor to state_types.

Minimal diff — wraps existing ad-hoc dict/tuple patterns into clean
dataclass interface without rewriting core modules.
"""

from __future__ import annotations
from typing import Optional, Any

import numpy as np

from .state_types import (
    WorldState, AgentObservation, AgentBelief,
    RobotBeliefOnAgent, BranchPosterior, TutorDecisionTrace,
)
from ..envs.observation_mask import make_observation_mask
from ..agents.branch_summary import summarize_branch
from ..agents.branch_scorer_probe import build_scorer_input
from ..metrics.self_discovery import estimate_self_discovery_prob


def from_scenario_to_world(
    gm,
    meta,
    sc,
    fb_modified: np.ndarray,
    ww: Any = None,
) -> WorldState:
    """Convert generator output → WorldState."""
    ws = WorldState(
        grid_height=gm.height,
        grid_width=gm.width,
        cell_types=gm.cell_types,
        cell_costs=gm.costs if hasattr(gm, 'costs') else None,
        cell_risks=gm.risks if hasattr(gm, 'risks') else None,
        cell_features=fb_modified,
        world_weights=ww,
        latent_mode=meta.latent_mode,
        oracle_safe_branch_id=sc.oracle_safe_branch_id,
        safe_cells=list(sc.safe_cells),
        risky_cells=list(sc.risky_cells),
        reveal_depth=getattr(sc, 'reveal_depth', 3),
        commit_depth=getattr(sc, 'commit_depth', 3),
        branch_len=getattr(sc, 'branch_len', 10),
    )
    # Hidden preferences hook
    if hasattr(sc, 'latent_preference'):
        ws.latent_preference_vector = sc.latent_preference
    if hasattr(sc, 'temptation_cells'):
        ws.hidden_temptation_cells = sc.temptation_cells
    return ws


def from_world_to_observation(
    world: WorldState,
    sc,
    obs_radius: int = 2,
    warning_received: bool = False,
    warning_content: Optional[dict] = None,
) -> AgentObservation:
    """Generate AgentObservation from world + fork + obs_radius."""
    fork = sc.fork_cell
    mask_a = make_observation_mask(sc.branch_a_cells, fork, obs_radius)
    mask_b = make_observation_mask(sc.branch_b_cells, fork, obs_radius)
    vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
    vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]
    all_vis = list(set(vis_a + vis_b))

    obs_feats = None
    if world.cell_features is not None and len(all_vis) > 0:
        obs_feats = np.array([world.cell_features[r, c] for r, c in all_vis])

    return AgentObservation(
        visible_cells=all_vis,
        observed_features=obs_feats,
        observation_mask_a=mask_a,
        observation_mask_b=mask_b,
        visible_branch_prefix_a=vis_a,
        visible_branch_prefix_b=vis_b,
        obs_radius=obs_radius,
        fork_position=fork,
        warning_received=warning_received,
        warning_content=warning_content,
    )


def update_agent_belief(
    belief: AgentBelief,
    observation: AgentObservation,
    world_features: np.ndarray,
    lp: Any,
) -> AgentBelief:
    """Update agent belief with new observation (in-place mutation + return)."""
    fv = np.full_like(world_features, 0.3)
    if observation.visible_branch_prefix_a:
        s_a = summarize_branch(observation.visible_branch_prefix_a,
                                world_features, fv, lp)
        belief.branch_summaries[0] = s_a
    if observation.visible_branch_prefix_b:
        s_b = summarize_branch(observation.visible_branch_prefix_b,
                                world_features, fv, lp)
        belief.branch_summaries[1] = s_b
    belief.n_observations += 1
    belief.risk_head_params = lp

    if observation.warning_received:
        belief.n_warnings_received += 1

    return belief


def compute_branch_posterior(
    belief: AgentBelief,
) -> BranchPosterior:
    """Derive BranchPosterior from current AgentBelief."""
    s_a = belief.branch_summaries.get(0)
    s_b = belief.branch_summaries.get(1)

    if s_a is None or s_b is None:
        return BranchPosterior()

    margin = abs(s_a[0] - s_b[0])
    # Simple softmax-like safety probability
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    safe_a = float(_sig(s_b[0] - s_a[0]))  # higher risk in b → a is safer
    safe_b = 1.0 - safe_a

    # Scorer scores
    sc_a, sc_b = 0.0, 0.0
    if belief.scorer_probe is not None and belief.concept_library is not None:
        inp_a = build_scorer_input(s_a, belief.concept_library)
        inp_b = build_scorer_input(s_b, belief.concept_library)
        sc_a = belief.scorer_probe.score(inp_a)
        sc_b = belief.scorer_probe.score(inp_b)

    entropy = -(safe_a * np.log(safe_a + 1e-10) + safe_b * np.log(safe_b + 1e-10))

    return BranchPosterior(
        safe_prob_a=round(safe_a, 4),
        safe_prob_b=round(safe_b, 4),
        entropy=round(float(entropy), 4),
        margin=round(float(margin), 4),
        scorer_score_a=round(float(sc_a), 4),
        scorer_score_b=round(float(sc_b), 4),
    )


def infer_robot_belief(
    world: WorldState,
    observation: AgentObservation,
    agent_belief: AgentBelief,
) -> RobotBeliefOnAgent:
    """Robot's inferred belief about agent state."""
    # Fraction visible
    total_branch = len(world.safe_cells) + len(world.risky_cells)
    vis_total = len(observation.visible_branch_prefix_a) + len(observation.visible_branch_prefix_b)
    obs_frac = vis_total / max(total_branch, 1)

    p_self = estimate_self_discovery_prob(
        world.commit_depth, world.reveal_depth, tau_v=1.0)

    bp = compute_branch_posterior(agent_belief)
    confidence = bp.decision_confidence

    return RobotBeliefOnAgent(
        estimated_obs_access=round(obs_frac, 3),
        estimated_branch_posterior=bp,
        estimated_p_self=round(p_self, 4),
        estimated_commitment_horizon=world.commit_depth,
        estimated_agent_confidence=round(confidence, 4),
    )


def build_tutor_trace(
    action: str,
    diag: dict,
) -> TutorDecisionTrace:
    """Convert tutor v4 output to serializable trace."""
    return TutorDecisionTrace(
        selected_action=action,
        Q_warn=diag.get("Q_warn", 0),
        Q_wait=diag.get("Q_wait", 0),
        dvoi=diag.get("dvoi", 0),
        p_self=diag.get("p_self", 0),
        margin_pre=diag.get("margin_pre", 0),
        margin_post=diag.get("margin_post", 0),
        delta_margin=diag.get("delta_s", diag.get("delta_margin", 0)),
        d_commit=diag.get("d_commit", 0),
        d_reveal=diag.get("d_reveal", 0),
        delta=diag.get("delta", 0),
    )
