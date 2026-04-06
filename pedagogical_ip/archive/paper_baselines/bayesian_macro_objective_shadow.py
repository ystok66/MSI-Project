"""Bayesian Macro Objective Shadow — Step 5C.

Rewrites the hand-crafted macro curriculum score as a unified
Bayes-style decision objective. Sits on top of the existing
GoalConditionalCurriculumHook.

Formula:
  J(ℓ) = E_q[ΔU_task(ℓ)]
       + β_I · EIG_q(ℓ)
       - β_D · C_dep(ℓ)
       + β_κ · G_κ(ℓ; κ̂)
       - β_R · C_res(ℓ)

Components:
  - task gain: counterfactual success lift
  - info gain: expected posterior entropy reduction
  - dep cost: expected Δν̂ + η·Δγ̂_gen (internalization cost)
  - κ term: additive macro state (unchanged from canonical)
  - resource cost: intervention effort

Does NOT modify any canonical module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from ..agents.agent_belief_state import AgentBelief
from .compositional_goal_hypotheses import GoalHypothesisSpace, DEFAULT_GOAL_SPACE
from .joint_goal_pref_posterior import JointGoalPrefPosterior
from .consequence_grounded_option_rollout import ConsequenceGroundedRollout
from .action_predictor import ActionPredictor


@dataclass
class BayesMacroConfig:
    """Hyperparameters for Bayesian macro objective."""
    beta_info: float = 0.5      # information gain weight
    beta_dep: float = 1.0       # dependence cost weight
    beta_kappa: float = 0.02    # κ̂ macro bonus
    beta_resource: float = 0.1  # intervention resource cost
    eta_gamma: float = 0.5      # γ̂_gen weight in dep cost

    # Resource costs per option type
    resource_costs: Dict[str, float] = field(default_factory=lambda: {
        "NONE": 0.0,
        "WARN": 0.2,
        "UNLOCK": 0.5,
        "ITEM_DROP": 0.8,
    })


@dataclass
class MacroObjectiveBreakdown:
    """Per-option breakdown of the Bayesian macro objective."""
    option: str
    task_gain: float
    info_gain: float
    dep_cost: float
    kappa_term: float
    resource_cost: float
    total: float


@dataclass
class BayesMacroDecision:
    """Result from Bayesian macro objective."""
    chosen_option: str
    breakdowns: Dict[str, MacroObjectiveBreakdown]
    posterior_entropy: float
    agrees_with_baseline: bool
    baseline_option: str


class BayesianMacroObjectiveShadow:
    """Bayesian macro objective shadow evaluator.

    Computes a unified decision-theoretic score for each intervention option,
    decomposed into interpretable components.

    Usage:
        shadow = BayesianMacroObjectiveShadow(ap)
        decision = shadow.evaluate(posterior, branches, ...)
    """

    def __init__(self,
                 action_predictor: ActionPredictor,
                 goal_space: Optional[GoalHypothesisSpace] = None,
                 config: Optional[BayesMacroConfig] = None):
        self._ap = action_predictor
        self._goal_space = goal_space or DEFAULT_GOAL_SPACE
        self.cfg = config or BayesMacroConfig()
        self._cgr = ConsequenceGroundedRollout(action_predictor)

    def evaluate(
        self,
        posterior: JointGoalPrefPosterior,
        branches: List[BranchAttributes],
        agent_belief: AgentBelief,
        kappa_hat: float = 0.0,
        nu_hat: float = 0.0,
        gamma_gen_hat: float = 0.5,
        safe_branch_idx: int = 0,
        baseline_option: Optional[str] = None,
    ) -> BayesMacroDecision:
        """Evaluate all options with Bayesian macro objective.

        Returns:
            BayesMacroDecision with per-option breakdowns
        """
        cfg = self.cfg
        mg = posterior.marginal_goal()
        mp = posterior.marginal_pref()
        h_current = posterior.entropy()

        options = ["NONE", "WARN", "UNLOCK", "ITEM_DROP"]
        breakdowns = {}

        for opt in options:
            # ── Task Gain: E_q[ΔU_task] ──
            task_gain = self._compute_task_gain(
                opt, branches, mg, mp, safe_branch_idx)

            # ── Information Gain: H(q) - E[H(q'|y)] ──
            info_gain = self._compute_info_gain(
                opt, posterior, branches, h_current)

            # ── Dependence Cost ──
            dep_cost = self._compute_dep_cost(
                opt, nu_hat, gamma_gen_hat)

            # ── κ Term ──
            kappa_term = cfg.beta_kappa * max(0.0, kappa_hat)
            if opt == "NONE":
                kappa_term *= 0.5  # less κ bonus for doing nothing

            # ── Resource Cost ──
            resource = cfg.resource_costs.get(opt, 0.0)

            # ── Total ──
            total = (task_gain
                     + cfg.beta_info * info_gain
                     - cfg.beta_dep * dep_cost
                     + kappa_term
                     - cfg.beta_resource * resource)

            breakdowns[opt] = MacroObjectiveBreakdown(
                option=opt,
                task_gain=task_gain,
                info_gain=info_gain,
                dep_cost=dep_cost,
                kappa_term=kappa_term,
                resource_cost=resource,
                total=total,
            )

        chosen = max(breakdowns, key=lambda o: breakdowns[o].total)

        agrees = (chosen == baseline_option) if baseline_option else True

        return BayesMacroDecision(
            chosen_option=chosen,
            breakdowns=breakdowns,
            posterior_entropy=h_current,
            agrees_with_baseline=agrees,
            baseline_option=baseline_option or "N/A",
        )

    def _compute_task_gain(self, opt, branches, mg, mp, safe_idx):
        """E_q[ΔP_safe(opt)] — expected success lift."""
        if opt == "NONE":
            return 0.0

        expected_lift = 0.0
        for gl, gw in mg.items():
            gh = self._goal_space.get(gl)
            for tl, tw in mp.items():
                orig = self._goal_space.compute_choice_probs(
                    branches, gh, tl)
                mod_branches = self._cgr.apply_consequence(opt, branches)
                mod = self._goal_space.compute_choice_probs(
                    mod_branches, gh, tl)
                lift = float(mod[safe_idx] - orig[safe_idx])
                expected_lift += gw * tw * lift

        return expected_lift

    def _compute_info_gain(self, opt, posterior, branches, h_current):
        """Approximate expected information gain.

        One-step approximation: if option reveals information,
        posterior entropy should decrease.

        For NONE: info gain = 0 (no intervention = no extra info)
        For others: approximate via teaching value ratio
        """
        if opt == "NONE":
            return 0.0

        h_max = posterior.max_entropy()
        if h_max < 1e-8:
            return 0.0

        # Info gain proxy: normalized current entropy
        # More uncertain → more potential info gain from intervention
        return h_current / h_max

    def _compute_dep_cost(self, opt, nu_hat, gamma_gen_hat):
        """C_dep = E[Δν̂] + η·E[Δγ̂_gen].

        Intervention increases dependence (ν̂) and may reduce
        generalization (γ̂_gen).
        """
        if opt == "NONE":
            return 0.0

        # Expected dependence increase from intervention
        dep_increase = {
            "WARN": 0.05,
            "UNLOCK": 0.10,
            "ITEM_DROP": 0.15,
        }.get(opt, 0.0)

        # Modulated by current dependence level
        delta_nu = dep_increase * (1.0 + max(0.0, nu_hat))

        # Generalization cost
        delta_gamma = dep_increase * 0.5 * (1.0 - max(0.0, gamma_gen_hat))

        return delta_nu + self.cfg.eta_gamma * delta_gamma

    def component_audit(
        self,
        decisions: List[BayesMacroDecision],
    ) -> Dict[str, float]:
        """Audit which components drive the decision.

        Returns per-component Kendall τ with total score.
        """
        if not decisions:
            return {}

        # Collect per-option vectors across decisions
        components = {"task_gain": [], "info_gain": [], "dep_cost": [],
                      "kappa_term": [], "resource_cost": [], "total": []}

        for d in decisions:
            for opt, bd in d.breakdowns.items():
                components["task_gain"].append(bd.task_gain)
                components["info_gain"].append(bd.info_gain)
                components["dep_cost"].append(bd.dep_cost)
                components["kappa_term"].append(bd.kappa_term)
                components["resource_cost"].append(bd.resource_cost)
                components["total"].append(bd.total)

        # Correlation of each component with total
        total = np.array(components["total"])
        result = {}
        for name in ["task_gain", "info_gain", "dep_cost",
                      "kappa_term", "resource_cost"]:
            vals = np.array(components[name])
            if np.std(vals) > 1e-8 and np.std(total) > 1e-8:
                result[f"corr_{name}"] = float(np.corrcoef(vals, total)[0, 1])
            else:
                result[f"corr_{name}"] = 0.0

        return result
