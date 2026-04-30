"""
tutor_inverse.py — T3 Inverse Inference Tutor (Phase 4 / 4.5).

Architecture:
    TutorTaskModel   → ground truth, danger oracle, correct hints
    TutorLearnerModel → inferred learner grammar state via constrained beam
    BehavioralRiskStats → risk competence from behavioral observations

Phase 4.5 upgrades:
    P0: warning = oracle safety gate (always warn on danger)
    P1: hint decision via continuous utility (beam entropy, margin, E_wrong)
    P2: position-specific hint selection (per-position uncertainty)
    P3: adaptive k via greedy marginal gain

Hint source: ALWAYS from TutorTaskModel (ground truth).
Hint decision: gated by beam posterior from TutorLearnerModel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..interfaces import TutorAction, CandidateBall, Example
from ..constants import TutorActionType, Outcome
from ..environment.state import QueryState
from ..config import TutorConfig, LearnerConfig
from .task_model import TutorTaskModel
from .learner_model import TutorLearnerModel
from .observation_v2 import ObservationRecord, ObservationSummaryV2
from .hint_policy import decide_hint, HintPolicyConfig
from .trace_analysis import analyze_trace_salience
from .counterfactual import one_step_counterfactual


# ── Behavioral risk tracker ──────────────────────────────────

@dataclass
class BehavioralRiskStats:
    """Track learner risk competence from behavioral observations.

    Used during TEACH phase (process-level) to estimate whether
    learner has learned to avoid danger on its own.
    """
    n_danger_selected: int = 0
    n_safe_selected: int = 0
    n_deaths: int = 0
    n_warnings_given: int = 0

    @property
    def risk_competence(self) -> float:
        """P(learner avoids danger) estimated from selections."""
        total = self.n_danger_selected + self.n_safe_selected
        if total < 3:
            return 0.0  # too few observations → assume incompetent
        return self.n_safe_selected / total


# ── InverseTutor ─────────────────────────────────────────────

class InverseTutor:
    """T3: Inverse inference tutor (Phase 4 main line).

    Two-model architecture:
    - task_model: knows ground truth → generates correct hints
    - learner_model: infers learner grammar → gates hint/warning decisions

    Unlike T2_oracle_shadow:
    - Does NOT copy learner's grammar library or risk belief
    - Infers learner state from observed outputs via constrained beam
    - Hints are ALWAYS correct (from ground truth)
    - Hint DECISION is based on learner model (does learner need help?)

    Unlike T3_behavioral:
    - Has a real grammar model of the learner (not just competence stats)
    - Can predict what learner will output (PredAcc_next)
    - update_depth controls inference granularity
    """

    def __init__(
        self,
        tutor_cfg: TutorConfig,
        learner_cfg: LearnerConfig,
        task_model: TutorTaskModel,
        update_depth: str = 'full_trace',
        warn_threshold: float = 0.7,
        hint_after_confirm_fail: bool = True,
        hint_policy_cfg: Optional[HintPolicyConfig] = None,
    ):
        self.cfg = tutor_cfg
        self.task_model = task_model
        self.learner_model = TutorLearnerModel(
            learner_cfg, update_depth=update_depth)
        self.risk_stats = BehavioralRiskStats()
        self.warn_threshold = warn_threshold
        self.hint_enabled = hint_after_confirm_fail
        self._hint_cfg = hint_policy_cfg or HintPolicyConfig()

        # Diagnostics
        self._div_log: List[Dict] = []
        self._pred_log: List[Dict] = []
        self._hint_diag_log: List[Dict] = []

        # Counterfactual: available probe queries (set externally)
        self._probe_queries: List[Tuple[List[str], List[str]]] = []
        self._enable_counterfactual: bool = False
        self._lambda_learn: float = 1.0  # weight for ΔLearn in final decision

        # ── Ablation overrides (Group B / C experiments) ──
        self._pos_mode: str = 'policy'     # 'policy' | 'random'
        self._gate_mode: str = 'policy'    # 'policy' | 'random_matched'
        self._matched_hint_rate: float = 0.5  # P(hint) when gate_mode='random_matched'
        self._rng = np.random.default_rng(42)

    def init_learner_model(self, support: List[Example]):
        """[ORACLE] Initialize learner model from support (same as learner)."""
        self.learner_model.init_from_support(support)

    def init_learner_model_cold(self, vocabulary: List[str]):
        """[SCHEME A] Cold start — empty learner model, vocab only."""
        self.learner_model.init_cold(vocabulary)

    def init_learner_model_teacher_prior(self, queries: List[Example]):
        """[SCHEME B] Teacher prior — knows task, not learner's examples."""
        self.learner_model.init_teacher_prior(queries)

    # ── Obs phase: result-level only ─────────────────────────

    def process_observation(self, record: 'ObservationRecord',
                             evidence_mode: str = 'strict'):
        """Update learner model from one obs result.

        This is the obs-phase entry point. Only result-level info is used.

        Args:
            record: one obs record
            evidence_mode:
                'strict'  — only use confirm evidence (submitted_output exists)
                'partial_fallback' — also use partial completion (legacy)
        """
        # Evidence gating
        etype = record.evidence_type
        if evidence_mode == 'strict' and etype == 'partial':
            # Skip: partial-only evidence not used in strict mode
            return

        output = record.best_output if evidence_mode == 'partial_fallback' \
            else record.submitted_output
        if output is None:
            return

        # Step 1: trace posterior fitting from observed output
        self.learner_model.update_from_output(
            record.query_words, output)

        # Step 2: if wrong + mask available, refine with feedback
        if not record.correct and record.wrong_mask and record.submitted_output:
            self.learner_model.update_from_feedback(
                record.query_words,
                record.submitted_output,
                {'mode': 'wrong_positions', 'mask': record.wrong_mask},
            )

    def process_all_observations(self, summary: 'ObservationSummaryV2',
                                  evidence_mode: str = 'strict'):
        """Process all obs records sequentially.

        Args:
            summary: observation summary containing records
            evidence_mode: 'strict' or 'partial_fallback'
        """
        for record in summary.records:
            self.process_observation(record, evidence_mode=evidence_mode)

    # ── Teach phase: per-step decisions (process-level) ──────

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """Warning decision: oracle safety gate.

        P0 (Phase 4.5): warning = safety gate, always warn on danger.
        risk_stats still recorded for diagnostics / eval prediction,
        but NOT used for warning decision.

        This ensures T3 has 0% teach death, same as T0/T1/T2,
        so teaching quality comparisons are clean.
        """
        has_danger = self.task_model.any_danger_in(selected)

        if has_danger:
            self.risk_stats.n_danger_selected += 1
            self.risk_stats.n_warnings_given += 1
            return TutorAction(
                action_type=TutorActionType.WARNING,
                message="Danger detected in selection.")
        else:
            self.risk_stats.n_safe_selected += 1
            return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """Hint decision after confirm failure (Phase 5: P1+P2+P3+P5).

        Hint SOURCE: TutorTaskModel (ground truth — never wrong)
        Hint DECISION: continuous utility + trace salience from beam posterior

        P1: Q_hint computed from beam entropy / margin / E_wrong / P_succ
        P2: positions scored by p_wrong, H_i, impact, and T_i
        P3: k selected greedily until marginal gain ≤ 0
        P5-1: trace salience T_i from per-word structural uncertainty
        P5-2: flat-beam fallback for high-entropy cases
        """
        if not self.hint_enabled:
            return TutorAction(action_type=TutorActionType.WAIT)

        gt = state.ground_truth
        if gt is None:
            return TutorAction(action_type=TutorActionType.WAIT)

        # Get beam posterior from learner model
        beam = self.learner_model.beam_posterior(state.query_words)

        # Wrong mask from feedback
        wrong_mask = feedback.get('mask', [])
        # Pad mask if needed
        while len(wrong_mask) < len(gt):
            wrong_mask.append(False)

        # Confirms remaining
        c_left = max(state.n_confirm_max - state.confirm_count, 0)

        # Competence estimate: use risk_stats safe ratio as proxy
        total_sel = (self.risk_stats.n_safe_selected
                     + self.risk_stats.n_danger_selected)
        competence = (self.risk_stats.n_safe_selected / max(total_sel, 1))

        # P5-1: Compute trace salience T_i
        trace_salience = None
        if beam:
            try:
                trace_result = analyze_trace_salience(
                    beam, state.query_words)
                if len(trace_result.T_i) > 0:
                    trace_salience = trace_result.T_i
            except Exception:
                pass  # Fall back to no trace salience

        # P1+P2+P3+P5: full decision
        should_hint, positions, diag = decide_hint(
            beam, gt, wrong_mask, c_left, competence,
            trace_salience=trace_salience,
            words=state.query_words,
            cfg=self._hint_cfg)

        # ── Hook 1: gating override (Group C ablation) ──
        diag['hint_gate_mode'] = self._gate_mode
        diag['hint_pos_mode'] = self._pos_mode
        diag['hint_decision_policy_should_hint'] = should_hint

        if self._gate_mode == 'random_matched':
            # Override policy gating with random coin flip
            should_hint = (self._rng.random() < self._matched_hint_rate)
            diag['hint_decision_random_gate'] = should_hint
            if should_hint and not positions:
                # Policy said no positions but random gate says hint:
                # generate fallback positions from wrong mask
                wrong_positions = [
                    i for i in range(len(gt))
                    if i < len(wrong_mask) and not wrong_mask[i]
                ]
                if wrong_positions:
                    k = min(self.cfg.max_hint_balls, len(wrong_positions))
                    chosen = self._rng.choice(
                        wrong_positions, size=k, replace=False)
                    positions = [(int(p), gt[p]) for p in chosen]

        # Log diagnostics
        self._hint_diag_log.append(diag)

        if should_hint and positions:
            # ── Hook 2: position override (Group B ablation) ──
            if self._pos_mode == 'random':
                # Keep k from policy, randomize which positions
                k = len(positions)
                wrong_positions = [
                    i for i in range(len(gt))
                    if i < len(wrong_mask) and not wrong_mask[i]
                ]
                if wrong_positions:
                    actual_k = min(k, len(wrong_positions))
                    chosen = self._rng.choice(
                        wrong_positions, size=actual_k, replace=False)
                    positions = [(int(p), gt[p]) for p in chosen]
                diag['hint_positions_randomized'] = True
            else:
                diag['hint_positions_randomized'] = False

            diag['hint_k'] = len(positions)
            diag['hint_decision_final_given'] = True
            # S3: One-step counterfactual (if enabled)
            if self._enable_counterfactual and self._probe_queries:
                try:
                    probe_words = [pq[0] for pq in self._probe_queries]
                    probe_golds = [pq[1] for pq in self._probe_queries]
                    rho = getattr(self.learner_model.cfg, 'rho_assist', 0.3)

                    cf_result = one_step_counterfactual(
                        self.learner_model,
                        state.query_words, gt, positions,
                        wrong_mask, feedback,
                        probe_words, probe_golds,
                        rho_assist=rho)

                    # Log both metrics
                    diag['cf_delta_learn_bin'] = float(cf_result.delta_learn_bin)
                    diag['cf_delta_learn_soft'] = float(cf_result.delta_learn_soft)
                    diag['cf_probe_acc_hint'] = float(cf_result.probe_acc_hint)
                    diag['cf_probe_acc_wait'] = float(cf_result.probe_acc_wait)
                    diag['cf_probe_ll_hint'] = float(cf_result.probe_ll_hint)
                    diag['cf_probe_ll_wait'] = float(cf_result.probe_ll_wait)
                    diag['cf_n_probes'] = cf_result.n_probes

                    # Continuous utility: adjust Q_hint with λ_learn · ΔLearn_soft
                    q_orig = diag.get('Q_after_pollution', diag.get('Q_hint', 0))
                    q_cf = q_orig + self._lambda_learn * cf_result.delta_learn_soft
                    diag['Q_cf_adjusted'] = float(q_cf)

                    # If CF-adjusted Q goes negative, veto
                    if q_cf <= 0:
                        diag['decision'] = 'WAIT'
                        diag['reason'] = (
                            f'CF-adjusted Q<=0 '
                            f'(Q_orig={q_orig:.3f}, '
                            f'dl_soft={cf_result.delta_learn_soft:.4f})')
                        return TutorAction(action_type=TutorActionType.WAIT)

                    # Catastrophic hard veto: extreme learning damage
                    if cf_result.delta_learn_soft < -0.2:
                        diag['decision'] = 'WAIT'
                        diag['reason'] = (
                            f'catastrophic CF veto '
                            f'(dl_soft={cf_result.delta_learn_soft:.4f})')
                        return TutorAction(action_type=TutorActionType.WAIT)

                except Exception:
                    pass  # CF failed → proceed with original decision

            hint_positions = [(pos, color) for pos, color in positions]
            return TutorAction(
                action_type=TutorActionType.HINT,
                hint_positions=hint_positions,
                message=f"Fine-grained hint: {len(positions)} positions.")

        diag['hint_decision_final_given'] = False
        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        """Courage decision for stuck learner."""
        if state.consecutive_retries >= self.cfg.n_retry_courage:
            needed = state.needed_colors()
            for ball in state.candidate_pool:
                if not self.task_model.is_danger(ball) and ball.color in needed:
                    return TutorAction(
                        action_type=TutorActionType.COURAGE,
                        message="A safe option exists for a color you need.")
        return TutorAction(action_type=TutorActionType.WAIT)

    # ── Post-query update ────────────────────────────────────

    def update_after_query(
        self,
        words: List[str],
        submitted: Optional[List[str]],
        outcome: Outcome,
        feedback: Optional[dict] = None,
    ):
        """Update learner model after a teach query completes.

        Called once per teach query after the query loop finishes.
        """
        if outcome == Outcome.SUCCESS:
            # Learner got it right → strong grammar signal
            gt = self.task_model.ground_truth_output(words)
            self.learner_model.update_from_output(words, gt)

        elif outcome == Outcome.TIMEOUT:
            # Learner timed out → use last submitted if available
            if submitted:
                self.learner_model.update_from_output(words, submitted)

        elif outcome == Outcome.DEATH:
            # Death → risk event, no grammar update
            self.risk_stats.n_deaths += 1

        # Feedback refinement (if confirm happened and was wrong)
        if feedback and not feedback.get('correct', True) and submitted:
            self.learner_model.update_from_feedback(
                words, submitted, feedback)

    # ── Behavioral hooks (called by experiment runner) ────────

    def observe_query_start(self):
        """Called at the start of each teach query."""
        pass  # placeholder for future per-query tracking

    def observe_death(self):
        """Called when learner dies."""
        self.risk_stats.n_deaths += 1

    def observe_retry(self):
        """Called when learner retries."""
        pass  # not used for inverse inference

    def observe_confirm_success(self):
        """Called when learner confirms successfully."""
        pass  # handled by update_after_query

    # ── Diagnostics ──────────────────────────────────────────

    def summary_dict(self) -> Dict:
        """Return full diagnostic summary."""
        return {
            'learner_model': self.learner_model.summary_dict(),
            'risk_stats': {
                'risk_competence': self.risk_stats.risk_competence,
                'n_danger_selected': self.risk_stats.n_danger_selected,
                'n_safe_selected': self.risk_stats.n_safe_selected,
                'n_deaths': self.risk_stats.n_deaths,
                'n_warnings_given': self.risk_stats.n_warnings_given,
            },
            'divergence_log': self._div_log,
            'prediction_log': self._pred_log,
        }

    # ── Divergence tracking ──────────────────────────────────

    def compute_divergence(
        self,
        real_predictor,
        probe_words: List[List[str]],
        probe_gold: Optional[List[List[str]]] = None,
        phase: str = 'teach',
        query_idx: int = 0,
    ) -> Dict:
        """Compute divergence between tutor's learner model and real learner.

        Also computes predictive validity (PredAcc_next).
        """
        from .divergence_v3 import compute_inverse_divergence

        rec = compute_inverse_divergence(
            self.learner_model, real_predictor,
            probe_words, probe_gold,
            phase=phase, query_idx=query_idx,
        )
        self._div_log.append(rec)
        return rec

    def compute_predictive_validity(
        self,
        real_predictor,
        next_words: List[str],
    ) -> Dict:
        """Predict what learner will output on next query."""
        from .divergence_v3 import compute_predictive_validity

        rec = compute_predictive_validity(
            self.learner_model, real_predictor, next_words)
        self._pred_log.append(rec)
        return rec
