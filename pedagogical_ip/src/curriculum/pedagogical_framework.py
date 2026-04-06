"""Unified Pedagogical Decision Framework — Runtime API.

Two-layer control:
  Layer 1 (micro):  episode-level tutor actions (WAIT/WARN/UNLOCK/ITEM_DROP)
  Layer 2 (macro):  curriculum-level controller (TEACH/EVAL/STOP)

Integrates all Stage 1–5 components into a single runtime with:
  - Unified config
  - Controller trace (JSONL-compatible)
  - Calibration hooks
  - OTR decomposition
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from .lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from .curriculum_controller_v13 import CurriculumControllerV13, ControllerV13Config
from .pairwise_response_model import PairwiseResponseModel
from .mastery_model import MasteryModel
from .risk_budget_calibration import AdaptiveRiskBudget
from ..agents.internalization_state_v3 import FactoredInternalizationState
from ..agents.behavior_probes import all_probes
from ..agents.stochastic_agent_policy import AgentPolicyParams
from ..agents.trainable_bridge import TrainableBridge
from ..metrics.teaching_zone_v2 import overteach_rate_v2


@dataclass
class FrameworkConfig:
    """Unified configuration."""
    theta: str = "safe"
    total_budget: float = 4.0
    risk_budget_mode: str = "theta"  # "fixed", "theta", "full"
    max_teach: int = 12
    max_eval: int = 3
    agent_beta: float = 4.0
    agent_epsilon: float = 0.1
    min_teach_before_stop: int = 3


@dataclass
class SessionTrace:
    """Complete audit trail for one teaching session."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    mastery_snapshots: List[Dict] = field(default_factory=list)
    state_snapshots: List[Dict] = field(default_factory=list)
    otr_decomp: Dict = field(default_factory=dict)
    stop_audit: Dict = field(default_factory=dict)
    eval_audit: List[Dict] = field(default_factory=list)

    def add_action(self, step, action, lesson_name, J, info):
        self.actions.append({
            "step": step, "action": action, "lesson": lesson_name,
            "J": round(J, 4) if J else 0, **info,
        })

    def add_mastery(self, u):
        self.mastery_snapshots.append(dict(u))

    def add_state(self, m):
        self.state_snapshots.append({
            "nu": round(m.nu, 4), "tau": round(m.tau, 4),
            "gamma_gen": round(m.gamma_gen, 4),
            "gamma_spec": round(m.gamma_spec, 4),
            "kappa": round(m.kappa, 4),
        })


@dataclass
class OTRDecomposition:
    """Decomposes OTR into teach-driven and eval-driven."""
    otr_total: float = 0.0
    otr_teach: float = 0.0
    otr_eval_overhead: float = 0.0
    n_teach: int = 0
    n_eval: int = 0
    family_repeats: int = 0

    def compute(self, m, n_teach, n_eval, history):
        otr = overteach_rate_v2(m)
        self.otr_total = otr["total"]
        self.n_teach = n_teach
        self.n_eval = n_eval
        # Estimate eval overhead: fraction of OTR attributable to eval exposure
        total_steps = n_teach + n_eval
        if total_steps > 0:
            self.otr_eval_overhead = round(self.otr_total * (n_eval / total_steps), 4)
            self.otr_teach = round(self.otr_total - self.otr_eval_overhead, 4)
        # Family repeat count
        families = []
        for h in history:
            les = next((l for l in LESSON_CATALOG_V2 if l.name == h), None)
            if les: families.append(les.family)
        self.family_repeats = sum(1 for i, f in enumerate(families) if f in families[:i])
        return self.to_dict()

    def to_dict(self):
        return {
            "otr_total": self.otr_total, "otr_teach": self.otr_teach,
            "otr_eval_overhead": self.otr_eval_overhead,
            "n_teach": self.n_teach, "n_eval": self.n_eval,
            "family_repeats": self.family_repeats,
        }


@dataclass
class CalibrationAudit:
    """STOP/EVAL calibration metrics."""
    stop_margins: List[float] = field(default_factory=list)
    stop_counterfactuals: List[float] = field(default_factory=list)
    eval_triggers: List[str] = field(default_factory=list)
    eval_rank_changed: List[bool] = field(default_factory=list)
    eval_pre_top1: List[str] = field(default_factory=list)
    eval_post_top1: List[str] = field(default_factory=list)

    def record_stop(self, margin, counterfactual_gain=None):
        self.stop_margins.append(round(margin, 4))
        if counterfactual_gain is not None:
            self.stop_counterfactuals.append(round(counterfactual_gain, 4))

    def record_eval(self, trigger_reason, pre_top1, post_top1):
        self.eval_triggers.append(trigger_reason)
        self.eval_pre_top1.append(pre_top1)
        self.eval_post_top1.append(post_top1)
        self.eval_rank_changed.append(pre_top1 != post_top1)

    def stop_margin_monotonicity(self):
        """Check: larger margin → lower counterfactual gain (desired)."""
        if len(self.stop_margins) < 2 or len(self.stop_counterfactuals) < 2:
            return None
        pairs = list(zip(self.stop_margins, self.stop_counterfactuals))
        concordant = sum(1 for i in range(len(pairs)) for j in range(i+1, len(pairs))
                        if (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1]) < 0)
        total = len(pairs) * (len(pairs) - 1) // 2
        return round(concordant / max(total, 1), 3) if total > 0 else None

    def eval_rank_change_rate(self):
        if not self.eval_rank_changed: return 0.0
        return round(sum(self.eval_rank_changed) / len(self.eval_rank_changed), 3)

    def summary(self):
        return {
            "n_stop_recorded": len(self.stop_margins),
            "avg_stop_margin": round(np.mean(self.stop_margins), 4) if self.stop_margins else None,
            "stop_monotonicity": self.stop_margin_monotonicity(),
            "n_eval_recorded": len(self.eval_triggers),
            "eval_rank_change_rate": self.eval_rank_change_rate(),
            "eval_trigger_counts": {r: self.eval_triggers.count(r) for r in set(self.eval_triggers)} if self.eval_triggers else {},
        }


class PedagogicalFramework:
    """Unified two-layer pedagogical decision framework.

    Usage:
        fw = PedagogicalFramework(FrameworkConfig(theta="shiny"))
        fw.reset_session()
        while True:
            action, lesson, info = fw.macro_step(learner_state)
            if action == "STOP": break
            if action == "EVAL": fw.run_eval(learner_state)
            if action == "TEACH": ...  # run episode
    """

    def __init__(self, config: FrameworkConfig = None):
        self.config = config or FrameworkConfig()
        self.ap = AgentPolicyParams(beta=self.config.agent_beta,
                                     epsilon=self.config.agent_epsilon, lambda_theta=1.0)
        self.controller = None
        self.trace = None
        self.otr_decomp = None
        self.cal_audit = None
        self._step = 0

    def reset_session(self):
        cfg = self.config
        c13cfg = ControllerV13Config(
            total_budget=cfg.total_budget,
            risk_budget_mode=cfg.risk_budget_mode,
            max_eval=cfg.max_eval,
            min_teach_before_stop=cfg.min_teach_before_stop,
        )
        cn = [l.name for l in LESSON_CATALOG_V2]
        self.controller = CurriculumControllerV13(
            cfg=c13cfg, theta=cfg.theta,
            response=PairwiseResponseModel(catalog_names=cn, theta=cfg.theta)
        )
        self.controller.reset_session(cfg.total_budget)
        self.trace = SessionTrace()
        self.otr_decomp = OTRDecomposition()
        self.cal_audit = CalibrationAudit()
        self._step = 0

    def macro_step(self, m: FactoredInternalizationState):
        """One macro decision: TEACH lesson / EVAL / STOP."""
        action, lesson, J, info = self.controller.select_action(m)
        lesson_name = lesson.name if lesson else None
        self.trace.add_action(self._step, action, lesson_name, J, info)
        self.trace.add_mastery(self.controller.mastery.mastery())
        self.trace.add_state(m)
        self._step += 1

        if action == "STOP":
            margin = info.get("margin", 0)
            self.cal_audit.record_stop(margin, info.get("counterfactual_J"))
        return action, lesson, info

    def run_eval(self, m: FactoredInternalizationState):
        """Execute EVAL: probe mastery, record calibration."""
        pre_u = dict(self.controller.mastery.mastery())
        # Get pre-eval top lesson
        pre_top1 = self._current_top_lesson(m)
        probes = all_probes(m, self.ap, self.config.theta)
        self.controller.update_mastery(probes)
        post_top1 = self._current_top_lesson(m)

        # Determine trigger reason from last trace
        last = self.trace.actions[-1] if self.trace.actions else {}
        trigger = "close_gap" if last.get("delta_12", 1.0) < 0.15 else "uncertainty"

        self.cal_audit.record_eval(trigger, pre_top1 or "", post_top1 or "")
        return probes

    def _current_top_lesson(self, m):
        """Get current best lesson name without side effects."""
        u = self.controller.mastery.mastery()
        best_name = None; best_J = -1e9
        for lesson in LESSON_CATALOG_V2:
            J, _ = self.controller._score_lesson(lesson, u, m)
            if J > best_J: best_J = J; best_name = lesson.name
        return best_name

    def update_after_teach(self, lesson_name, mastery_before, mastery_after,
                           nu_before, nu_after, gg_before, gg_after,
                           otr_before, otr_after):
        """Update response model after teaching."""
        self.controller.update_response(lesson_name, mastery_before, mastery_after,
                                        nu_before, nu_after, gg_before, gg_after,
                                        otr_before, otr_after)

    def finalize_session(self, m):
        """Compute final metrics."""
        otr = self.otr_decomp.compute(
            m, self.controller.n_teach, self.controller.eval_count,
            [h for h in self.controller.history if h not in ("EVAL", "STOP")]
        )
        return {
            "controller": self.controller.controller_summary(),
            "audit": self.controller.actionability_audit(),
            "posterior": self.controller.posterior_stats(),
            "otr_decomp": otr,
            "calibration": self.cal_audit.summary(),
        }
