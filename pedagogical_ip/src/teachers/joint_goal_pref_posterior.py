"""Joint Goal-Preference Posterior — q(g, θ) with ActionPredictor.

Bayesian posterior over (goal, preference_type) hypotheses, updated
from observed actions via ActionPredictor-based inverse planning.

Step 4 prior modes:
  structural:    P₀(g|c₀) at init, pure action-likelihood update  [CANONICAL DEFAULT]
  pcfg:          PCFG-based P₀(g) at init, pure action-likelihood update  [paper baseline]
  legacy_bonus:  original exp(β_C · C_t(g)) bonus in update  [DEPRECATED — backward compat only]

Primary metric (Step 4+): subgoal marginals q(u) = Σ_{g∋u} q(g).

Does NOT modify any existing module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, PREFERENCE_TYPES,
)
from ..agents.agent_belief_state import AgentBelief
from .compositional_goal_hypotheses import (
    GoalHypothesisSpace, GoalHypothesis, DEFAULT_GOAL_SPACE,
)
from .compositional_goal_prior import (
    GoalPriorContext, GoalPriorConfig, PCFGPriorConfig,
    compute_normalized_goal_prior, compute_pcfg_log_prior_vector,
    compute_subgoal_marginals,
)


# Canonical 2-type (default)
THETA_2 = ("safe", "shiny")
# Research expanded K-type
THETA_K = ("safe", "shiny", "risky", "shortcut", "neutral")

# Default temptation grid (reused from T5)
DEFAULT_TEMPT_GRID = (0.0, 0.3, 0.6, 0.9)
DEFAULT_TEMPT_PRIOR = (0.4, 0.3, 0.2, 0.1)


@dataclass(frozen=True)
class JointHypothesis:
    """Single (goal, theta, z_tempt) hypothesis."""
    goal_label: str
    theta: str
    z_tempt: float = 0.0


class JointGoalPrefPosterior:
    """Bayesian posterior q(g, θ) or q(g, θ, z) over joint hypotheses.

    Updated from observed actions via goal-conditioned utility computation
    through GoalHypothesisSpace (not custom utility functions).

    Usage:
        jgpp = JointGoalPrefPosterior(goal_space, pref_types=THETA_2)
        jgpp.update(world_state, branches, observed_action=1)
        print(jgpp.marginal_goal())
        print(jgpp.map_hypothesis())
    """

    def __init__(self,
                 goal_space: Optional[GoalHypothesisSpace] = None,
                 pref_types: Tuple[str, ...] = THETA_2,
                 tempt_grid: Optional[Tuple[float, ...]] = None,
                 tempt_prior: Optional[Tuple[float, ...]] = None,
                 goal_prior: Optional[Dict[str, float]] = None,
                 pref_prior: Optional[Dict[str, float]] = None,
                 params: Optional[AgentPolicyParams] = None,
                 forgetting_rate: float = 0.01,
                 compatibility=None,
                 prior_mode: str = "structural",
                 prior_context: Optional[GoalPriorContext] = None,
                 prior_config: Optional[GoalPriorConfig] = None,
                 pcfg_config: Optional[PCFGPriorConfig] = None):
        """
        Args:
            prior_mode: "legacy_bonus" | "structural" | "pcfg"
            prior_context: episode-start context (structural mode)
            prior_config: hyperparams for structural prior
            pcfg_config: hyperparams for PCFG prior
        """
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE
        self._pref_types = pref_types
        self._tempt_grid = tempt_grid or (0.0,)
        self._has_tempt = tempt_grid is not None
        self._params = params or AgentPolicyParams()
        self._forgetting_rate = forgetting_rate
        self._compatibility = compatibility
        self._prior_mode = prior_mode
        self._prior_context = prior_context or GoalPriorContext()
        self._prior_config = prior_config or GoalPriorConfig()
        self._pcfg_config = pcfg_config or PCFGPriorConfig()
        self._history: List[Dict] = []

        # Build hypothesis grid
        n_g = self._goal_space.n_goals
        n_p = len(pref_types)
        n_z = len(self._tempt_grid)

        # ── Build goal prior P₀(g|c₀) ──
        g_labels = self._goal_space.labels
        if self._prior_mode == "structural":
            gp = compute_normalized_goal_prior(
                self._goal_space, self._prior_context, self._prior_config)
        elif self._prior_mode == "pcfg":
            log_pcfg = compute_pcfg_log_prior_vector(
                self._goal_space, self._pcfg_config)
            log_pcfg -= np.max(log_pcfg)
            gp = np.exp(log_pcfg)
            gp /= gp.sum()
        elif goal_prior is not None:
            gp = np.array([goal_prior.get(g, 1.0/n_g) for g in g_labels])
            gp /= gp.sum()
        else:
            gp = np.ones(n_g) / n_g

        # ── Build pref/tempt priors (unchanged) ──
        if pref_prior is None:
            pp = np.ones(n_p) / n_p
        else:
            pp = np.array([pref_prior.get(t, 1.0/n_p) for t in pref_types])
            pp /= pp.sum()

        if tempt_prior is not None and self._has_tempt:
            zp = np.array(tempt_prior[:n_z], dtype=np.float64)
            zp /= zp.sum()
        else:
            zp = np.ones(n_z) / n_z

        # q₀(g,θ,z) = P₀(g|c₀) · P₀(θ) · P₀(z)
        self._log_weights = np.zeros((n_g, n_p, n_z))
        for gi in range(n_g):
            for pi in range(n_p):
                for zi in range(n_z):
                    self._log_weights[gi, pi, zi] = (
                        np.log(max(gp[gi] * pp[pi] * zp[zi], 1e-15)))

        self._normalize()

    def update(self,
               world_state,
               branches: list[BranchAttributes],
               observed_action: int,
               agent_belief: Optional[AgentBelief] = None,
               risky_branch_idx: int = 1):
        """Bayesian update from observed action.

        Step 4 prior modes:
          legacy_bonus: q_t ∝ q_{t-1} · P(a|s,g,θ,z) · exp(β_C·C(g))
          structural/pcfg: q_t ∝ q_{t-1} · P(a|s,g,θ,z)  [pure likelihood]
        """
        n_g = self._goal_space.n_goals
        n_p = len(self._pref_types)
        n_z = len(self._tempt_grid)

        # Update compatibility tracker ONLY in legacy mode
        if self._prior_mode == "legacy_bonus" and self._compatibility is not None:
            theta_est = self.predicted_pref()
            self._compatibility.observe(branches, observed_action, theta_est)

        for gi, gh in enumerate(self._goal_space.hypotheses):
            for pi, theta in enumerate(self._pref_types):
                for zi, z in enumerate(self._tempt_grid):
                    # Modify branches for temptation hypothesis
                    mod_branches = self._tempt_modified_branches(
                        branches, z, risky_branch_idx)

                    # Compute P(a_obs | g, θ, z) via goal-conditioned utility
                    probs = self._goal_space.compute_choice_probs(
                        mod_branches, gh, theta, self._params)
                    ll = np.log(max(probs[observed_action], 1e-15))
                    self._log_weights[gi, pi, zi] += ll

            # Compatibility bonus ONLY in legacy mode
            if self._prior_mode == "legacy_bonus" and self._compatibility is not None:
                bonus = self._compatibility.log_compatibility_bonus(
                    gh, branches, self.predicted_pref())
                self._log_weights[gi, :, :] += bonus

        # Forgetting / diffusion
        if self._forgetting_rate > 0:
            w = self._weights()
            uniform = np.ones_like(w) / w.size
            w_diffused = (1 - self._forgetting_rate) * w + \
                         self._forgetting_rate * uniform
            self._log_weights = np.log(np.clip(w_diffused, 1e-15, None))

        self._normalize()

        # Log
        self._history.append({
            "step": len(self._history),
            "observed_action": observed_action,
            "entropy": self.entropy(),
            "map": str(self.map_hypothesis()),
            "marginal_goal_top1": self.predicted_goal(),
            "marginal_pref_top1": self.predicted_pref(),
        })

    def marginal_goal(self) -> Dict[str, float]:
        """P(g) = Σ_{θ,z} q(g,θ,z)."""
        w = self._weights()
        mg = w.sum(axis=(1, 2))  # sum over pref and tempt
        return {self._goal_space.labels[i]: float(mg[i])
                for i in range(len(mg))}

    def marginal_pref(self) -> Dict[str, float]:
        """P(θ) = Σ_{g,z} q(g,θ,z)."""
        w = self._weights()
        mp = w.sum(axis=(0, 2))  # sum over goal and tempt
        return {self._pref_types[i]: float(mp[i])
                for i in range(len(mp))}

    def marginal_tempt(self) -> Dict[float, float]:
        """P(z) = Σ_{g,θ} q(g,θ,z)."""
        w = self._weights()
        mz = w.sum(axis=(0, 1))
        return {self._tempt_grid[i]: float(mz[i])
                for i in range(len(mz))}

    def entropy(self) -> float:
        """H(q) = -Σ q · log q."""
        w = self._weights().ravel()
        w = w[w > 1e-15]
        return -float(np.sum(w * np.log(w)))

    def max_entropy(self) -> float:
        return float(np.log(self._log_weights.size))

    def map_hypothesis(self) -> JointHypothesis:
        """Maximum a posteriori hypothesis."""
        w = self._weights()
        idx = np.unravel_index(np.argmax(w), w.shape)
        return JointHypothesis(
            goal_label=self._goal_space.labels[idx[0]],
            theta=self._pref_types[idx[1]],
            z_tempt=self._tempt_grid[idx[2]],
        )

    def predicted_goal(self) -> str:
        mg = self.marginal_goal()
        return max(mg, key=mg.get)

    def predicted_pref(self) -> str:
        mp = self.marginal_pref()
        return max(mp, key=mp.get)

    def subgoal_marginals(self) -> Dict[str, float]:
        """q(u) = Σ_{g ∋ u} q(g) — Step 4 primary metric."""
        return compute_subgoal_marginals(
            self.marginal_goal(), self._goal_space)

    @property
    def prior_mode(self) -> str:
        return self._prior_mode

    def goal_conditional_pref(self, goal_label: str) -> Dict[str, float]:
        """P(θ | g) = q(g,θ) / q(g)."""
        gi = self._goal_space.index(goal_label)
        w = self._weights()
        w_g = w[gi, :, :].sum(axis=1)  # sum over z
        total = w_g.sum()
        if total < 1e-15:
            return {t: 1.0 / len(self._pref_types) for t in self._pref_types}
        return {self._pref_types[i]: float(w_g[i] / total)
                for i in range(len(self._pref_types))}

    def joint_confidence(self) -> float:
        return float(np.max(self._weights()))

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self):
        """Reset to prior."""
        if self._compatibility is not None:
            self._compatibility.reset()
        self.__init__(
            goal_space=self._goal_space,
            pref_types=self._pref_types,
            tempt_grid=self._tempt_grid if self._has_tempt else None,
            params=self._params,
            forgetting_rate=self._forgetting_rate,
            compatibility=self._compatibility,
            prior_mode=self._prior_mode,
            prior_context=self._prior_context,
            prior_config=self._prior_config,
            pcfg_config=self._pcfg_config,
        )

    # ─── Internal ────────────────────────────────────────────

    def _weights(self) -> np.ndarray:
        """Normalized weights from log-weights."""
        lw = self._log_weights - np.max(self._log_weights)
        w = np.exp(lw)
        total = w.sum()
        if total > 0:
            w /= total
        return w

    def _normalize(self):
        """Re-normalize log-weights (numerical stability)."""
        self._log_weights -= np.mean(self._log_weights)

    def _tempt_modified_branches(self, branches, z_tempt, risky_idx):
        """Boost temptation on risky branch by z."""
        if z_tempt < 0.01:
            return branches
        mod = []
        for i, b in enumerate(branches):
            new_tempt = b.temptation_score
            if i == risky_idx:
                new_tempt = b.temptation_score + z_tempt
            mod.append(BranchAttributes(
                safety_score=b.safety_score,
                temptation_score=new_tempt,
                texture_novelty=b.texture_novelty,
                shortcut_bonus=b.shortcut_bonus,
                risk_penalty=b.risk_penalty,
            ))
        return mod
