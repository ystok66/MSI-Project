"""Shadow Integration Bridge — POMDP interface adapter layer.

Bridges existing observer/tutor/policy calls to the new POMDP interface
WITHOUT modifying any existing module. This is the T3-B/C adapter.

Usage in experiment scripts:
    bridge = ShadowBridge(theta="safe")
    bridge.observe_step(world_state, agent_belief, branches, action, observer)
    report = bridge.get_report()

All shadow — does not affect canonical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from ..agents.world_state import WorldState
from ..agents.agent_belief_state import AgentBelief
from ..agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, compute_choice_probs,
)
from .action_predictor import ActionPredictor
from .robot_belief_over_agent import RobotBeliefOverAgent


@dataclass
class ShadowStepLog:
    """One-step shadow log entry."""
    step: int = 0
    old_probs: Optional[np.ndarray] = None
    new_probs: Optional[np.ndarray] = None
    observed_action: int = 0
    old_nll: float = 0.0
    new_nll: float = 0.0
    theta_posterior: Optional[Dict[str, float]] = None
    belief_entropy: float = 0.0
    top1_agree: bool = True


@dataclass
class ShadowReport:
    """Aggregate shadow-mode diagnostics."""
    n_steps: int = 0
    mean_old_nll: float = 0.0
    mean_new_nll: float = 0.0
    nll_parity: float = 0.0       # |new - old| mean
    brier_old: float = 0.0
    brier_new: float = 0.0
    top1_agreement: float = 0.0   # fraction of steps with same top-1
    mean_entropy: float = 0.0
    final_theta_posterior: Optional[Dict[str, float]] = None
    ece_old: float = 0.0
    ece_new: float = 0.0

    # Per-step calibration bins
    calibration_bins: Optional[Dict] = None


class ShadowBridge:
    """Shadow-mode POMDP interface bridge.

    Runs new ActionPredictor + RobotBeliefOverAgent in parallel
    with the old path. Logs NLL, Brier, calibration, entropy.
    Never affects canonical decisions.
    """

    def __init__(self, theta: str = "safe",
                 params: Optional[AgentPolicyParams] = None):
        self.theta = theta
        self.params = params or AgentPolicyParams(
            beta=4.0, epsilon=0.1, lambda_theta=1.0)
        self.predictor = ActionPredictor(params=self.params)
        self.belief_tracker = RobotBeliefOverAgent(
            action_predictor=self.predictor)
        self._log: List[ShadowStepLog] = []
        self._step = 0

        # Calibration bins
        self._cal_bins_old = [[] for _ in range(10)]
        self._cal_bins_new = [[] for _ in range(10)]

    def observe_step(self, world_state: Optional[WorldState],
                     agent_belief: AgentBelief,
                     branches: list[BranchAttributes],
                     observed_action: int,
                     observer_estimate: Optional[Dict] = None,
                     observer_confidence: Optional[Dict] = None):
        """Log one step through both old and new paths."""
        # --- Old path (direct computation) ---
        old_probs = compute_choice_probs(
            branches, self.theta, self.params)
        old_nll = -float(np.log(max(old_probs[observed_action], 1e-10)))

        # --- New path (via ActionPredictor) ---
        dist = self.predictor.predict(world_state, agent_belief, branches)
        new_nll = -float(dist.log_probs[observed_action])

        # Top-1 agreement
        top1_agree = (np.argmax(old_probs) == dist.top1_idx)

        # Brier score components
        n_acts = len(branches)
        old_onehot = np.zeros(n_acts)
        old_onehot[observed_action] = 1.0
        # (accumulated later in report)

        # Robot belief update
        self.belief_tracker.update_from_action(
            world_state, branches, observed_action, agent_belief)
        if observer_estimate:
            self.belief_tracker.update_from_observer(
                observer_estimate, observer_confidence or {})

        state = self.belief_tracker.get_state()

        # Calibration binning
        old_top_p = float(np.max(old_probs))
        new_top_p = dist.top1_prob
        old_correct = (np.argmax(old_probs) == observed_action)
        new_correct = (dist.top1_idx == observed_action)

        old_bin = min(int(old_top_p * 10), 9)
        new_bin = min(int(new_top_p * 10), 9)
        self._cal_bins_old[old_bin].append(1 if old_correct else 0)
        self._cal_bins_new[new_bin].append(1 if new_correct else 0)

        entry = ShadowStepLog(
            step=self._step,
            old_probs=old_probs.copy(),
            new_probs=dist.probs.copy(),
            observed_action=observed_action,
            old_nll=old_nll,
            new_nll=new_nll,
            theta_posterior=dict(state.theta_posterior),
            belief_entropy=state.entropy,
            top1_agree=top1_agree,
        )
        self._log.append(entry)
        self._step += 1

    def get_report(self) -> ShadowReport:
        """Compute aggregate diagnostics."""
        if not self._log:
            return ShadowReport()

        n = len(self._log)
        old_nlls = [e.old_nll for e in self._log]
        new_nlls = [e.new_nll for e in self._log]
        top1s = [e.top1_agree for e in self._log]
        entropies = [e.belief_entropy for e in self._log]

        # Brier scores
        brier_old, brier_new = 0.0, 0.0
        for e in self._log:
            n_a = len(e.old_probs)
            onehot = np.zeros(n_a)
            onehot[e.observed_action] = 1.0
            brier_old += float(np.sum((e.old_probs - onehot) ** 2))
            brier_new += float(np.sum((e.new_probs - onehot) ** 2))
        brier_old /= n
        brier_new /= n

        # ECE
        ece_old = self._compute_ece(self._cal_bins_old)
        ece_new = self._compute_ece(self._cal_bins_new)

        return ShadowReport(
            n_steps=n,
            mean_old_nll=float(np.mean(old_nlls)),
            mean_new_nll=float(np.mean(new_nlls)),
            nll_parity=float(np.mean(np.abs(
                np.array(new_nlls) - np.array(old_nlls)))),
            brier_old=round(brier_old, 6),
            brier_new=round(brier_new, 6),
            top1_agreement=float(np.mean(top1s)),
            mean_entropy=float(np.mean(entropies)),
            final_theta_posterior=self._log[-1].theta_posterior,
            ece_old=round(ece_old, 6),
            ece_new=round(ece_new, 6),
        )

    def _compute_ece(self, bins) -> float:
        """Expected calibration error from binned data."""
        ece = 0.0
        total = sum(len(b) for b in bins)
        if total == 0:
            return 0.0
        for i, b in enumerate(bins):
            if not b:
                continue
            bin_conf = (i + 0.5) / 10.0
            bin_acc = np.mean(b)
            ece += (len(b) / total) * abs(bin_acc - bin_conf)
        return float(ece)

    def reset(self):
        self._log = []
        self._step = 0
        self.predictor.reset_stats()
        self.belief_tracker.reset()
        self._cal_bins_old = [[] for _ in range(10)]
        self._cal_bins_new = [[] for _ in range(10)]
