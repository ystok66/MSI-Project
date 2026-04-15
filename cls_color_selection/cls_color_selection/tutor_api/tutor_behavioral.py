"""
tutor_behavioral.py — T3 Behavioral Tutor (realistic ToM).

Unlike T2_oracle_shadow which deep-copies the learner's internal state,
T3 only observes learner BEHAVIOR and infers competence from it.

Information access:
  ✅ Support examples (studies independently)
  ✅ Which balls the learner selected
  ✅ Whether each ball is danger (oracle knowledge, same as T0/T1)
  ✅ Confirm result: success / fail / timeout
  ✅ Feedback content (wrong positions)
  ✅ Death / retry events
  ❌ Learner's beam posterior / grammar library
  ❌ Learner's risk belief posterior
  ❌ Learner's policy parameters

Key design decisions:
  - T3 builds its OWN CLS grammar from support (independent, not copied)
  - T3 uses behavioral statistics to estimate learner competence
  - Warning: oracle danger + behavioral threshold (fade out when learner learns)
  - Hint: T3 uses its OWN grammar to predict correct output. This means
    hints can be WRONG if T3's grammar is also imperfect. This is a
    DELIBERATE FEATURE — a realistic tutor's hints are only as good as
    its own understanding of the task.
  - Courage: triggered by observed excessive retry patterns
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from ..interfaces import TutorAction, CandidateBall
from ..constants import TutorActionType
from ..environment.state import QueryState
from ..config import TutorConfig, LearnerConfig
from ..learner.cls_wrapper import CLSSequencePredictor
from ..learner.target_predictor import TargetPredictor
from ..interfaces import Example


# ── Behavioral statistics tracker ─────────────────────────────

@dataclass
class BehavioralStats:
    """Track learner behavior from observations only.

    All estimates are derived from observed events, never from
    reading learner internal state.
    """
    # Grammar competence tracking
    n_confirm_success: int = 0
    n_confirm_fail: int = 0
    n_confirm_total: int = 0

    # Risk competence tracking
    n_danger_selected: int = 0      # times learner picked danger
    n_danger_warned: int = 0        # times we warned (and learner avoided)
    n_safe_selected: int = 0        # times learner picked only safe
    n_deaths: int = 0               # observed deaths (shouldn't happen if we warn)

    # Over-caution tracking
    n_retries: int = 0              # total retry count
    n_empty_selections: int = 0     # times learner selected nothing
    n_queries_seen: int = 0         # total queries observed

    # Per-query tracking
    retries_this_query: int = 0

    @property
    def gram_competence(self) -> float:
        """Estimated P(learner grammar correct) from confirm history."""
        if self.n_confirm_total == 0:
            return 0.5  # uninformative prior
        return self.n_confirm_success / self.n_confirm_total

    @property
    def risk_competence(self) -> float:
        """Estimated P(learner avoids danger on own) from selection history.

        Based on: how often does learner select danger when NOT warned?
        We count danger selections as failures to detect.
        """
        total_select_events = self.n_danger_selected + self.n_safe_selected
        if total_select_events < 3:
            return 0.0  # too few observations → assume incompetent (safe default: warn)
        # Fraction of selections that were safe
        return self.n_safe_selected / total_select_events

    @property
    def stuck_tendency(self) -> float:
        """Estimated P(learner is over-cautious) from retry patterns."""
        if self.n_queries_seen == 0:
            return 0.0
        avg_retries = self.n_retries / max(self.n_queries_seen, 1)
        # Normalize: >10 retries per query → stuck tendency ~1.0
        return min(1.0, avg_retries / 10.0)

    def summary(self) -> Dict:
        return {
            'gram_competence': self.gram_competence,
            'risk_competence': self.risk_competence,
            'stuck_tendency': self.stuck_tendency,
            'n_confirm_success': self.n_confirm_success,
            'n_confirm_fail': self.n_confirm_fail,
            'n_danger_selected': self.n_danger_selected,
            'n_safe_selected': self.n_safe_selected,
            'n_retries': self.n_retries,
            'n_queries_seen': self.n_queries_seen,
        }


# ── T3 Behavioral Tutor ──────────────────────────────────────

class BehavioralTutor:
    """T3: Realistic ToM tutor that infers learner state from behavior.

    This tutor builds its OWN grammar model independently and tracks
    learner competence purely from observed behavior (selections,
    confirm results, deaths, retries).

    Unlike T2_oracle_shadow:
    - Does NOT copy learner's grammar library or risk belief
    - Does NOT have access to learner's beam posterior
    - Must infer learner capability from behavioral patterns
    - May give incorrect hints (if its own grammar is wrong)
    """

    def __init__(
        self,
        tutor_cfg: TutorConfig,
        learner_cfg: LearnerConfig,
        # Thresholds (tuned via experiments)
        warn_risk_threshold: float = 0.7,
        hint_gram_threshold: float = 0.4,
        courage_stuck_threshold: float = 0.5,
        # Observation phase
        belief=None,
    ):
        self.tutor_cfg = tutor_cfg
        self.learner_cfg = learner_cfg
        self.warn_risk_threshold = warn_risk_threshold
        self.hint_gram_threshold = hint_gram_threshold
        self.courage_stuck_threshold = courage_stuck_threshold
        self.belief = belief

        # Behavioral tracker
        self.stats = BehavioralStats()

        # T3's own grammar model (NOT learner's copy)
        self._predictor: Optional[CLSSequencePredictor] = None
        self._target_pred: Optional[TargetPredictor] = None
        self._studied = False

    def init_own_grammar(self, support: List[Example]):
        """Build T3's OWN grammar from support.

        This is NOT a copy of the learner's grammar — it's T3 studying
        the same support independently. T3 and learner may reach
        different interpretations, especially with few support examples.
        """
        self._predictor = CLSSequencePredictor(self.learner_cfg)
        self._predictor.fit_support(support)
        self._target_pred = TargetPredictor(self._predictor)
        self._studied = True

    def predict_own(self, words: List[str]) -> Optional[List[str]]:
        """T3's own prediction for a query. May differ from learner's."""
        if not self._studied:
            return None
        try:
            return self._target_pred.predict_target(words)
        except Exception:
            return None

    # ── Core tutor hooks ──────────────────────────────────────

    def on_select(
        self,
        state: QueryState,
        selected: List[CandidateBall],
    ) -> TutorAction:
        """Decide whether to warn about danger.

        T3 is oracle for danger detection (knows which balls are danger,
        same as T0/T1). The difference from T0 is that T3 considers
        WHETHER the learner still needs warning based on behavioral history.

        Policy:
        - If learner's estimated risk_competence < threshold → WARN
        - If learner appears to have learned danger avoidance → WAIT
          (let learner practice independent risk judgment)
        """
        has_danger = any(b.is_danger for b in selected)

        if has_danger:
            # Track: learner selected danger (behavioral signal)
            self.stats.n_danger_selected += 1

            # Decision: does learner still need warning?
            if self.stats.risk_competence < self.warn_risk_threshold:
                # Learner hasn't demonstrated sufficient risk avoidance
                self.stats.n_danger_warned += 1
                return TutorAction(
                    action_type=TutorActionType.WARNING,
                    message="Danger detected in selection.")
            else:
                # Learner seems to have learned — let it face consequences
                # This is a STRATEGIC decision: risk death for learning autonomy
                return TutorAction(action_type=TutorActionType.WAIT)
        else:
            # Safe selection — good behavioral signal
            self.stats.n_safe_selected += 1
            return TutorAction(action_type=TutorActionType.WAIT)

    def on_confirm_fail(
        self,
        state: QueryState,
        feedback: dict,
    ) -> TutorAction:
        """Decide whether to give hint after confirm failure.

        T3 uses its OWN grammar prediction to generate hints.
        NOTE: T3's hints may be WRONG if T3's own grammar is also
        imperfect. This is a deliberate feature — a realistic tutor's
        understanding is limited by its own learning.

        Policy:
        - If learner's grammar competence is low AND we're near timeout → HINT
        - Otherwise → WAIT (let learner figure it out)
        """
        self.stats.n_confirm_fail += 1
        self.stats.n_confirm_total += 1

        if not self.tutor_cfg.hint_after_confirm_fail:
            return TutorAction(action_type=TutorActionType.WAIT)

        # Check if hint is warranted
        near_timeout = state.confirm_count >= state.n_confirm_max - 2
        low_competence = self.stats.gram_competence < self.hint_gram_threshold

        if near_timeout or low_competence:
            # Generate hint from T3's OWN grammar (may be wrong!)
            my_pred = self.predict_own(state.query_words)
            if my_pred is not None:
                # Build hint_positions: (pos, color) tuples where T3
                # disagrees with learner's submission.
                # NOTE: colors come from T3's own prediction, which may
                # be WRONG if T3's grammar is imperfect.
                hint_positions = []
                submitted = feedback.get('submitted', [])
                for i, (sub, mine) in enumerate(
                    zip(submitted, my_pred)
                ):
                    if sub is not None and sub != mine:
                        hint_positions.append((i, mine))

                if hint_positions:
                    return TutorAction(
                        action_type=TutorActionType.HINT,
                        hint_positions=hint_positions,
                        message="Based on my understanding, check these positions.")

        return TutorAction(action_type=TutorActionType.WAIT)

    def on_courage_check(self, state: QueryState) -> TutorAction:
        """Decide whether to encourage a stuck learner.

        Policy:
        - If learner shows high stuck tendency (many retries) → COURAGE
        - Check if safe balls exist for needed colors
        """
        if self.stats.stuck_tendency > self.courage_stuck_threshold:
            needed = state.needed_colors()
            for ball in state.candidate_pool:
                if not ball.is_danger and ball.color in needed:
                    return TutorAction(
                        action_type=TutorActionType.COURAGE,
                        message="A safe option exists for a color you need.")
            return TutorAction(action_type=TutorActionType.WAIT)
        return TutorAction(action_type=TutorActionType.WAIT)

    # ── Behavioral update hooks (called by experiment runner) ──

    def observe_confirm_success(self):
        """Called when learner confirms successfully."""
        self.stats.n_confirm_success += 1
        self.stats.n_confirm_total += 1

    def observe_death(self):
        """Called when learner dies."""
        self.stats.n_deaths += 1

    def observe_retry(self):
        """Called when learner retries."""
        self.stats.n_retries += 1
        self.stats.retries_this_query += 1

    def observe_empty_selection(self):
        """Called when learner selects nothing."""
        self.stats.n_empty_selections += 1

    def observe_query_start(self):
        """Called at the start of each query."""
        self.stats.n_queries_seen += 1
        self.stats.retries_this_query = 0

    def summary_dict(self) -> Dict:
        """Return behavioral stats + divergence log for logging."""
        d = self.stats.summary()
        d['divergence_log'] = self._divergence_log
        return d

    # ── Per-query divergence tracking ─────────────────────────

    _divergence_log: List[Dict] = None

    def _ensure_div_log(self):
        if self._divergence_log is None:
            self._divergence_log = []

    def compute_divergence_vs_real(
        self,
        real_predictor: CLSSequencePredictor,
        probe_words: List[List[str]],
        probe_gold: Optional[List[List[str]]] = None,
        phase: str = 'teach',
        query_idx: int = 0,
    ) -> Dict:
        """Compute T3-vs-real divergence at current query.

        This measures how well T3's independent grammar matches the real
        learner's grammar. Unlike T2's divergence (which measures shadow
        drift from a copy), this measures genuine ToM accuracy.

        Returns dict with per-probe metrics.
        """
        self._ensure_div_log()

        if not self._studied or not probe_words:
            rec = {
                'phase': phase, 'query_idx': query_idx,
                'top1_agreement': None, 'js_divergence': None,
                'my_accuracy': None, 'real_accuracy': None,
            }
            self._divergence_log.append(rec)
            return rec

        n_agree = 0
        my_correct = 0
        real_correct = 0
        js_vals = []

        for pi, words in enumerate(probe_words):
            # T3's prediction
            my_pred = self.predict_own(words)
            # Real learner's prediction
            try:
                real_pred = real_predictor.predict_target(words)
            except Exception:
                real_pred = None

            # Top-1 agreement
            if my_pred is not None and real_pred is not None:
                if my_pred == real_pred:
                    n_agree += 1

            # Accuracy vs gold
            if probe_gold and pi < len(probe_gold):
                gold = probe_gold[pi]
                if my_pred is not None and my_pred == gold:
                    my_correct += 1
                if real_pred is not None and real_pred == gold:
                    real_correct += 1

            # JS divergence over beam posteriors
            try:
                my_beam = self._predictor.beam_posterior(words)
                real_beam = real_predictor.beam_posterior(words)
                if my_beam and real_beam:
                    js = self._compute_js(my_beam, real_beam)
                    js_vals.append(js)
            except Exception:
                pass

        n_probes = len(probe_words)
        rec = {
            'phase': phase,
            'query_idx': query_idx,
            'top1_agreement': n_agree / max(n_probes, 1),
            'js_divergence': float(np.mean(js_vals)) if js_vals else 0.0,
            'my_accuracy': my_correct / max(n_probes, 1),
            'real_accuracy': real_correct / max(n_probes, 1),
            'accuracy_gap': abs(my_correct - real_correct) / max(n_probes, 1),
            'behavioral_gram_competence': self.stats.gram_competence,
            'behavioral_risk_competence': self.stats.risk_competence,
        }
        self._divergence_log.append(rec)
        return rec

    @staticmethod
    def _compute_js(beam_a, beam_b) -> float:
        """Jensen-Shannon divergence between two beam posteriors."""
        from scipy.special import logsumexp

        def beam_to_probs(beam):
            scores = np.array([b[0] for b in beam])
            log_q = scores - logsumexp(scores)
            probs = np.exp(log_q)
            d = {}
            for b, p in zip(beam, probs):
                key = str(b[2] if len(b) > 2 else b[1])
                d[key] = d.get(key, 0.0) + p
            return d

        da = beam_to_probs(beam_a)
        db = beam_to_probs(beam_b)
        all_keys = sorted(set(da.keys()) | set(db.keys()))
        if not all_keys:
            return 0.0

        p = np.array([da.get(k, 1e-10) for k in all_keys])
        q = np.array([db.get(k, 1e-10) for k in all_keys])
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m + 1e-30)) + \
             0.5 * np.sum(q * np.log(q / m + 1e-30))
        return max(0.0, float(js))

