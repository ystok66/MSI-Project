"""H1 — Core State Types for Pedagogical Decision Framework.

Clean POMDP / nested-belief dataclasses that enforce the data flow:

  WorldState → AgentObservation → AgentBelief
  WorldState + ObsHistory → RobotBeliefOnAgent
  AgentBelief → BranchPosterior → PlannerChoice
  RobotBeliefOnAgent + BranchPosterior → TutorDecisionTrace

These types do NOT replace existing modules — they provide a clean
interface layer that future extensions (hidden preferences,
compositional goals) can hook into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Layer 1: World Truth (not visible to agent)
# ═══════════════════════════════════════════════════════════════

@dataclass
class WorldState:
    """Ground truth — agent MUST NOT access directly."""
    # Grid topology
    grid_height: int = 0
    grid_width: int = 0
    cell_types: Optional[np.ndarray] = None       # (H, W) int
    cell_costs: Optional[np.ndarray] = None        # (H, W) float
    cell_risks: Optional[np.ndarray] = None        # (H, W) float
    cell_features: Optional[np.ndarray] = None     # (H, W, D) float

    # Latent semantic state
    world_weights: Optional[Any] = None            # WorldWeights object
    latent_mode: bool = True

    # Branch ground truth
    oracle_safe_branch_id: int = -1
    safe_cells: list = field(default_factory=list)
    risky_cells: list = field(default_factory=list)

    # Diagnostic vs distractor cue assignment
    diagnostic_cue_dims: list = field(default_factory=lambda: [2, 3])
    distractor_cue_dims: list = field(default_factory=lambda: [1])

    # Timing parameters
    reveal_depth: int = 3
    commit_depth: int = 3
    branch_len: int = 10

    # ── Future hooks (H5) ──
    latent_goal_vector: Optional[np.ndarray] = None
    latent_preference_vector: Optional[np.ndarray] = None
    hidden_temptation_cells: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Layer 2: Agent Observation (what agent actually sees)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentObservation:
    """What the agent can actually observe at current timestep."""
    visible_cells: list = field(default_factory=list)
    observed_features: Optional[np.ndarray] = None   # (N, D) features of visible cells
    observation_mask_a: Optional[np.ndarray] = None   # per-cell visibility for branch A
    observation_mask_b: Optional[np.ndarray] = None   # per-cell visibility for branch B
    visible_branch_prefix_a: list = field(default_factory=list)
    visible_branch_prefix_b: list = field(default_factory=list)
    obs_radius: int = 2
    fork_position: tuple = (0, 0)

    # Warning received this step (if any)
    warning_received: bool = False
    warning_content: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════
# Layer 3: Agent Belief (agent's internal model)
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentBelief:
    """Agent's internal belief state (updated via observations)."""
    risk_head_params: Optional[Any] = None         # LatentCostRiskHead
    cost_head_params: Optional[Any] = None
    branch_summaries: dict = field(default_factory=dict)  # {branch_id: summary_vec}
    concept_library: Optional[Any] = None          # BranchConceptLibrary
    scorer_probe: Optional[Any] = None             # BranchScorerProbe
    n_observations: int = 0
    n_warnings_received: int = 0

    # Branch-level belief
    branch_posterior: Optional["BranchPosterior"] = None

    # ── Future hooks (H5) ──
    goal_hypothesis: Optional[np.ndarray] = None
    preference_hypothesis: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════
# Layer 4: Robot's belief about agent (nested belief)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RobotBeliefOnAgent:
    """Tutor's estimate of agent's mental state."""
    estimated_obs_access: float = 0.0              # fraction of branch visible to agent
    estimated_branch_posterior: Optional["BranchPosterior"] = None
    estimated_p_self: float = 0.0                  # P(agent self-discovers)
    estimated_commitment_horizon: int = 0          # steps until irreversible choice
    estimated_agent_confidence: float = 0.0

    # ── Future hooks (H5) ──
    belief_over_goal_hypothesis: Optional[np.ndarray] = None
    belief_over_preference_hypothesis: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════
# Layer 5: Branch Posterior (shared by planner + tutor)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BranchPosterior:
    """Posterior over branch safety — used by both planner and tutor."""
    safe_prob_a: float = 0.5
    safe_prob_b: float = 0.5
    entropy: float = 0.693    # ln(2) = max binary entropy
    margin: float = 0.0       # |risk_a - risk_b|
    scorer_score_a: float = 0.0
    scorer_score_b: float = 0.0

    @property
    def predicted_safe_branch(self) -> int:
        return 0 if self.safe_prob_a >= self.safe_prob_b else 1

    @property
    def bayes_risk(self) -> float:
        return 1.0 - max(self.safe_prob_a, self.safe_prob_b)

    @property
    def decision_confidence(self) -> float:
        return abs(self.safe_prob_a - self.safe_prob_b)


# ═══════════════════════════════════════════════════════════════
# Layer 6: Tutor Decision Trace (fully serializable)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TutorDecisionTrace:
    """Complete record of why tutor chose WAIT or WARN."""
    # Decision
    selected_action: str = "WAIT"   # "WAIT" or "WARN"
    Q_warn: float = 0.0
    Q_wait: float = 0.0

    # Components
    dvoi: float = 0.0
    p_self: float = 0.0
    urgency: float = 0.0
    missed_window: float = 0.0
    margin_pre: float = 0.0
    margin_post: float = 0.0
    delta_margin: float = 0.0
    redundancy: float = 0.0
    confidence: float = 0.0

    # Timing
    d_commit: int = 0
    d_reveal: int = 0
    delta: int = 0

    # Outcome (filled after episode)
    agent_chose_safe: Optional[bool] = None
    agent_self_discovered: Optional[bool] = None
    warning_flipped_choice: Optional[bool] = None

    def to_dict(self) -> dict:
        """Serialize to dict (JSONL-ready)."""
        return {
            "action": self.selected_action,
            "Q_warn": round(self.Q_warn, 4),
            "Q_wait": round(self.Q_wait, 4),
            "dvoi": round(self.dvoi, 4),
            "p_self": round(self.p_self, 4),
            "urgency": round(self.urgency, 4),
            "missed_window": round(self.missed_window, 4),
            "margin_pre": round(self.margin_pre, 4),
            "margin_post": round(self.margin_post, 4),
            "delta_margin": round(self.delta_margin, 4),
            "d_commit": self.d_commit,
            "d_reveal": self.d_reveal,
            "delta": self.delta,
            "agent_chose_safe": self.agent_chose_safe,
            "agent_self_discovered": self.agent_self_discovered,
        }
