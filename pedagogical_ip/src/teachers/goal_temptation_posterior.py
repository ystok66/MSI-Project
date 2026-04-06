"""Goal/Temptation Posterior — Robot's belief over agent's hidden variables.

Maintains factorized posterior q_t(g, z) over:
  - g: agent's true goal (2 hypotheses: true_goal, decoy_goal)
  - z: hidden temptation level (4 hypotheses: 0.0, 0.3, 0.6, 0.9)

Update rule:
  q_t(g, z) ∝ q_{t-1}(g, z) · P(a_obs | s_world, g, z)

where P(a_obs | ...) comes from ActionPredictor with modified BranchAttributes
that reflect each (g, z) hypothesis.

This is shadow-mode only. Does not modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes
from ..agents.agent_belief_state import AgentBelief


# Default temptation grid (4 levels: none / weak / medium / strong)
DEFAULT_TEMPT_GRID = (0.0, 0.3, 0.6, 0.9)

# Default prior: mild low-temptation bias
DEFAULT_TEMPT_PRIOR = (0.4, 0.3, 0.2, 0.1)


@dataclass(frozen=True)
class HiddenState:
    """Single hypothesis about agent's hidden state."""
    goal: str            # goal label
    z_tempt: float       # hidden temptation level


@dataclass
class PosteriorEntry:
    """Hypothesis with weight."""
    hidden: HiddenState
    log_weight: float = 0.0
    weight: float = 1.0


class GoalTemptationPosterior:
    """Factorized posterior over (goal, temptation) hypotheses.

    Robot-side belief tracker. Updated from observed actions via
    inverse planning through ActionPredictor.

    Usage:
        gtp = GoalTemptationPosterior(goals, action_predictor=ap)
        gtp.update(world_state, branches, observed_action, agent_belief)
        print(gtp.marginal_tempt())
        print(gtp.entropy())
    """

    def __init__(self,
                 goals: Tuple[str, ...] = ("true_goal", "decoy_goal"),
                 tempt_grid: Tuple[float, ...] = DEFAULT_TEMPT_GRID,
                 tempt_prior: Optional[Tuple[float, ...]] = DEFAULT_TEMPT_PRIOR,
                 goal_prior: Optional[Dict[str, float]] = None,
                 action_predictor=None):
        """
        Args:
            goals: goal labels
            tempt_grid: temptation levels
            tempt_prior: prior over temptation (will be normalized)
            goal_prior: prior over goals (uniform if None)
            action_predictor: ActionPredictor for P(a | s, b)
        """
        self._goals = goals
        self._tempt_grid = tempt_grid
        self._predictor = action_predictor
        self._history: List[Dict] = []

        # Build prior
        if tempt_prior is None:
            tempt_prior = tuple(1.0 / len(tempt_grid) for _ in tempt_grid)
        tp = np.array(tempt_prior, dtype=np.float64)
        tp /= tp.sum()

        if goal_prior is None:
            goal_prior = {g: 1.0 / len(goals) for g in goals}
        gp_total = sum(goal_prior.values())

        # Create hypothesis grid
        self._entries: List[PosteriorEntry] = []
        for g in goals:
            p_g = goal_prior.get(g, 1.0 / len(goals)) / gp_total
            for i, z in enumerate(tempt_grid):
                w = p_g * tp[i]
                self._entries.append(PosteriorEntry(
                    hidden=HiddenState(goal=g, z_tempt=z),
                    log_weight=np.log(max(w, 1e-15)),
                    weight=w,
                ))
        self._normalize()

    def update(self,
               world_state,
               branches: list[BranchAttributes],
               observed_action: int,
               agent_belief: Optional[AgentBelief] = None,
               risky_branch_idx: int = 1):
        """Bayesian update from observed action.

        q_t(g,z) ∝ q_{t-1}(g,z) · P(a_obs | s, g, z)

        Args:
            world_state: WorldState
            branches: original BranchAttributes
            observed_action: which branch agent picked
            agent_belief: optional belief hint
            risky_branch_idx: which branch is the temptation-sensitive one
        """
        if self._predictor is None:
            return

        for entry in self._entries:
            # Modify branches to reflect this hypothesis
            mod_branches = self._modify_branches(
                branches, entry.hidden, risky_branch_idx)

            # Create agent belief under this hypothesis
            ab = self._make_belief(agent_belief, entry.hidden)

            # Compute log P(a_obs | s, g, z)
            ll = self._predictor.score(
                world_state, ab, mod_branches, observed_action)

            entry.log_weight += ll

        self._normalize()

        # Log
        self._history.append({
            "step": len(self._history),
            "observed_action": observed_action,
            "entropy": self.entropy(),
            "map": self.map_hypothesis().goal + f"_z={self.map_hypothesis().z_tempt}",
            "marginal_tempt": dict(self.marginal_tempt()),
            "marginal_goal": dict(self.marginal_goal()),
        })

    def marginal_goal(self) -> Dict[str, float]:
        """P(g) = Σ_z q(g, z)."""
        result = {g: 0.0 for g in self._goals}
        for e in self._entries:
            result[e.hidden.goal] += e.weight
        return result

    def marginal_tempt(self) -> Dict[float, float]:
        """P(z) = Σ_g q(g, z)."""
        result = {z: 0.0 for z in self._tempt_grid}
        for e in self._entries:
            result[e.hidden.z_tempt] += e.weight
        return result

    def expected_tempt(self) -> float:
        """E[z_tempt] = Σ z · P(z)."""
        mt = self.marginal_tempt()
        return sum(z * p for z, p in mt.items())

    def entropy(self) -> float:
        """H(q) = -Σ q · log q."""
        weights = np.array([e.weight for e in self._entries])
        weights = weights[weights > 1e-15]
        return -float(np.sum(weights * np.log(weights)))

    def map_hypothesis(self) -> HiddenState:
        """Maximum a posteriori hypothesis."""
        best = max(self._entries, key=lambda e: e.weight)
        return best.hidden

    def get_weights(self) -> Dict[str, float]:
        """Full posterior as {label: weight}."""
        return {f"{e.hidden.goal}_z={e.hidden.z_tempt}": e.weight
                for e in self._entries}

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self):
        """Reset to prior."""
        self.__init__(
            goals=self._goals,
            tempt_grid=self._tempt_grid,
            action_predictor=self._predictor,
        )

    # ─── Internal ────────────────────────────────────────────

    def _normalize(self):
        """Normalize weights (log-sum-exp for stability)."""
        log_ws = np.array([e.log_weight for e in self._entries])
        log_max = np.max(log_ws)
        ws = np.exp(log_ws - log_max)
        total = ws.sum()
        if total > 0:
            ws /= total
        for i, e in enumerate(self._entries):
            e.weight = float(ws[i])
            e.log_weight = float(np.log(max(ws[i], 1e-15)))

    def _modify_branches(self, branches: list[BranchAttributes],
                         hidden: HiddenState,
                         risky_idx: int) -> list[BranchAttributes]:
        """Modify BranchAttributes to reflect hidden state hypothesis.

        For temptation z: boost temptation_score on risky branch by z.
        For goal: if decoy_goal, flip which branch is "safe".
        """
        mod = []
        for i, b in enumerate(branches):
            new_tempt = b.temptation_score
            new_safety = b.safety_score

            # Temptation effect: agent with higher z sees more
            # temptation on the risky branch
            if i == risky_idx:
                new_tempt = b.temptation_score + hidden.z_tempt

            # Goal effect: decoy goal makes the risky branch
            # look like the "goal branch"
            if hidden.goal == "decoy_goal":
                # Swap safety perception for this hypothesis
                new_safety = 1.0 - b.safety_score

            mod.append(BranchAttributes(
                safety_score=new_safety,
                temptation_score=new_tempt,
                texture_novelty=b.texture_novelty,
                shortcut_bonus=b.shortcut_bonus,
                risk_penalty=b.risk_penalty,
            ))
        return mod

    def _make_belief(self, ab: Optional[AgentBelief],
                     hidden: HiddenState) -> AgentBelief:
        """Create AgentBelief for this hypothesis."""
        if ab is not None:
            return AgentBelief(
                belief_mean=ab.belief_mean,
                belief_var=ab.belief_var,
                m_state=dict(ab.m_state),
                theta=ab.theta,
            )
        return AgentBelief(theta="shiny" if hidden.z_tempt > 0.5 else "safe")
