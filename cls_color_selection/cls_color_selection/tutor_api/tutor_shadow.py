"""
tutor_shadow.py — T2: Full shadow learner tutor.

Uses a shadow copy of the learner's internal state to predict
how each tutor action changes the learner's future performance.

Key difference from T1 (proxy):
  - G_eval is computed via probe_eval on the counterfactual shadow state
  - Not just low-dimensional belief summary

Two fidelity modes:
  - exact: full shadow copy
  - compressed: role_counts + emit_stats only
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..interfaces import TutorAction, CandidateBall, Example
from ..constants import TutorActionType, Outcome
from ..environment.state import QueryState
from ..config import TutorConfig, LearnerConfig, FullConfig
from .shadow_snapshot import ShadowLearnerSnapshot
from .shadow_clone import create_shadow_snapshot, write_shadow_to_real_risk
from .shadow_update import (
    shadow_warning_update, shadow_courage_update,
    shadow_safe_observation_update, shadow_death_update,
    shadow_feedback_update,
)
from .shadow_eval import (
    probe_eval_accuracy, estimate_shadow_eval_gain,
)
from .tutor_state import TutorBelief
from .belief_update import compute_timeout_risk, compute_death_risk
from .action_generators import (
    generate_pre_select_actions, generate_post_confirm_actions,
    apply_hint_to_state,
)


class ShadowTutor:
    """T2: Full shadow learner tutor.

    Maintains a shadow snapshot of the learner's grammar + risk state.
    For each candidate action, clones the shadow, simulates the update,
    and estimates counterfactual eval gain.
    """

    def __init__(
        self,
        cfg: TutorConfig,
        learner_cfg: LearnerConfig,
        fidelity: str = 'exact',
        probe_queries: Optional[List[Example]] = None,
        belief: Optional[TutorBelief] = None,
    ):
        self.cfg = cfg
        self.learner_cfg = learner_cfg
        self.fidelity = fidelity
        self.probe_queries = probe_queries or []  # held-out for G_eval
        self.belief = belief

        # Shadow state (set by init_shadow or update_shadow)
        self.shadow: Optional[ShadowLearnerSnapshot] = None

        # Debug log
        self.last_action_log: Optional[dict] = None

    def init_shadow(
        self,
        predictor,
        risk_belief,
        support_examples: list,
    ):
        """Initialize shadow from real learner state."""
        self.shadow = create_shadow_snapshot(
            predictor, risk_belief, self.learner_cfg,
            support_examples, fidelity=self.fidelity,
        )

    def set_belief(self, belief: TutorBelief):
        self.belief = belief

    # ── Pre-select hook ─────────────────────────────────────────

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """Pre-placement: WARN > shadow-Q(COURAGE vs WAIT)."""
        has_danger = any(b.is_danger for b in selected)

        # Safety-first: always warn on danger
        if has_danger:
            if self.belief:
                self.belief.n_warnings_issued += 1
            # Update shadow risk with warning
            if self.shadow:
                shadow_copy = self.shadow.clone()
                shadow_warning_update(shadow_copy, selected)
                self.shadow = shadow_copy
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Your selection contains danger.")

        # No danger: should we COURAGE or WAIT?
        # Update shadow with safe observation
        if self.shadow:
            shadow_after = self.shadow.clone()
            for b in selected:
                shadow_safe_observation_update(shadow_after, b.observed_vec)
            self.shadow = shadow_after

        # Courage check
        if state.consecutive_retries >= self.cfg.n_retry_courage:
            needed = state.needed_colors()
            pool_has_safe_needed = any(
                not b.is_danger and b.color in needed
                for b in state.candidate_pool
            )
            if pool_has_safe_needed:
                # Evaluate courage via shadow: would courage help eval?
                u_courage = self._eval_courage_utility(state)
                if u_courage > 0:
                    if self.belief:
                        self.belief.n_courage_issued += 1
                    return TutorAction(
                        action_type=TutorActionType.COURAGE,
                        message="A safe needed ball exists.")

        return TutorAction(action_type=TutorActionType.WAIT)

    # ── Post-confirm-fail hook ──────────────────────────────────

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """Post-confirm: evaluate WAIT vs HINT_k via shadow counterfactual."""
        if not self.cfg.hint_after_confirm_fail:
            return TutorAction(action_type=TutorActionType.WAIT)

        if not self.shadow or not self.probe_queries:
            # Fall back to rule-based hint
            return self._rule_hint_fallback(state, feedback)

        # Compute G_eval for each HINT option
        candidates = generate_post_confirm_actions(state, feedback, self.cfg)
        if len(candidates) <= 1:
            return candidates[0]

        best_action = candidates[0]  # WAIT
        best_q = 0.0

        for action in candidates:
            if action.action_type == TutorActionType.WAIT:
                continue

            # Counterfactual: simulate hint + feedback update
            shadow_cf = self.shadow.clone()

            # Simulate feedback update on shadow grammar
            submitted = [c if c is not None else '?'
                         for c in feedback.get('submitted', state.completion)]
            shadow_feedback_update(
                shadow_cf, state.query_words, submitted,
                feedback, self.learner_cfg)

            # G_eval
            g_eval = estimate_shadow_eval_gain(
                self.shadow, shadow_cf, self.probe_queries)
            g_eval *= self.cfg.lambda_eval

            # G_teach
            k = len(action.hint_positions) if action.hint_positions else 0
            remaining = state.L - state.filled_count
            g_teach = self.cfg.lambda_teach * (k / max(remaining, 1))

            # C_over
            competence = probe_eval_accuracy(self.shadow, self.probe_queries)
            over_factor = 1.0 / (1.0 + np.exp(-5.0 * (competence - 0.5)))
            c_over = self.cfg.lambda_over * k * over_factor

            # C_int
            c_int = self.cfg.lambda_int

            q_a = g_eval + g_teach - c_over - c_int

            self.last_action_log = {
                'action': action.action_type.name,
                'k': k,
                'g_eval': g_eval,
                'g_teach': g_teach,
                'c_over': c_over,
                'c_int': c_int,
                'q_a': q_a,
            }

            if q_a > best_q:
                best_q = q_a
                best_action = action

        if best_action.action_type == TutorActionType.HINT:
            if self.belief:
                self.belief.n_hints_issued += 1

        return best_action

    # ── Courage check ───────────────────────────────────────────

    def on_courage_check(self, state: QueryState) -> TutorAction:
        """Courage from on_select (already handled there)."""
        return TutorAction(action_type=TutorActionType.WAIT)

    # ── Shadow sync ─────────────────────────────────────────────

    def sync_shadow(self, predictor, risk_belief, support_examples):
        """Re-sync shadow from real learner (optional T3 mechanism)."""
        self.shadow = create_shadow_snapshot(
            predictor, risk_belief, self.learner_cfg,
            support_examples, fidelity=self.fidelity,
        )

    # ── Internal utilities ──────────────────────────────────────

    def _eval_courage_utility(self, state: QueryState) -> float:
        """Estimate utility of courage via shadow."""
        if not self.shadow:
            return 0.1  # mild positive default
        # Courage reduces over-avoidance — check if shadow predicts stuck
        if self.belief:
            return self.belief.risk.overavoid_rate * 0.3
        return 0.1

    def _rule_hint_fallback(self, state, feedback) -> TutorAction:
        """Fallback to rule-based hint when shadow not available."""
        confirms_left = state.n_confirm_max - state.confirm_count
        if confirms_left <= 1:
            gt = state.ground_truth
            for pos in range(min(len(gt), len(state.completion))):
                if state.completion[pos] is None or state.completion[pos] != gt[pos]:
                    if self.belief:
                        self.belief.n_hints_issued += 1
                    return TutorAction(
                        action_type=TutorActionType.HINT,
                        hint_positions=[(pos, gt[pos])],
                        message="Hint: placing 1 correct ball.",
                    )
        return TutorAction(action_type=TutorActionType.WAIT)
