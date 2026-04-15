"""
learner_model.py — TutorLearnerModel: tutor's inverse inference of learner state.

This is the CORE of Phase 4. The tutor maintains its own estimate of what
the learner's grammar looks like, updated via CLS constrained beam search
applied to the learner's OBSERVED outputs.

Key insight: CLS's `infer_top_k_ast(words, target_vecs, library, priors)`
is already an inverse inference engine. When target_vecs = learner's output,
it infers the most likely latent parse that could produce that output.

Evidence separation contract:
    update_from_output()   → trace posterior fitting (constrained beam E-step)
                             Absorbs: "learner produced this output sequence"
    update_from_feedback() → posterior reweight + differential M-step refinement
                             Absorbs: "that output was judged wrong at these positions"
    These two functions handle DISTINCT evidence. Never double-count.

update_depth controls what gets updated:
    'role_only'  → only role_counts (cheapest, Level 1)
    'role_emit'  → role_counts + emit_stats (Level 2)
    'full_trace' → role + emit + repeat + color (Level 3, most complete)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np
from scipy.special import logsumexp

from ..config import LearnerConfig
from ..interfaces import Example


# ── Update depth enum ─────────────────────────────────────────

VALID_UPDATE_DEPTHS = ('role_only', 'role_emit', 'full_trace')


# ── TutorLearnerModel ─────────────────────────────────────────

class TutorLearnerModel:
    """Tutor's MODEL of what the learner knows.

    Maintains independent role/emit posteriors inferred from
    observed learner outputs, NOT copied from learner internals.

    Architecture:
    - Has its OWN CLS agent (initialized from same support as learner)
    - Updates from observed learner outputs via constrained beam search
    - Updates from observed feedback via reweight + differential M-step
    - Can predict what it thinks learner will output next
    """

    def __init__(
        self,
        learner_cfg: LearnerConfig,
        update_depth: str = 'full_trace',
        eta_inv: float = 1.0,
    ):
        """
        Args:
            learner_cfg: learner config (for CLS mode, n_em, etc.)
            update_depth: 'role_only' | 'role_emit' | 'full_trace'
            eta_inv: learning rate for inverse inference M-step
        """
        if update_depth not in VALID_UPDATE_DEPTHS:
            raise ValueError(
                f"update_depth must be one of {VALID_UPDATE_DEPTHS}, "
                f"got {update_depth!r}")

        self.cfg = learner_cfg
        self.update_depth = update_depth
        self.eta_inv = eta_inv

        self._agent = None
        self._predictor = None
        self._studied = False

        # Track updates for diagnostics
        self._n_output_updates = 0
        self._n_feedback_updates = 0

    def init_from_support(self, support: List[Example]):
        """[ORACLE BASELINE] Initialize from same support as learner.

        This is the ORACLE baseline — tutor uses learner's exact training
        data, so they start identical. Only valid for upper-bound comparison.
        """
        from ..learner.cls_wrapper import CLSSequencePredictor

        self._predictor = CLSSequencePredictor(self.cfg)
        self._predictor.fit_support(support)
        self._agent = self._predictor.get_agent()
        self._studied = True
        self._init_mode = 'support'

    def init_cold(self, vocabulary: List[str]):
        """[SCHEME A] Cold start — empty learner model.

        Tutor's learner model starts with:
        - Vocabulary registered (so beam search can enumerate words)
        - Uniform priors on all roles/colors (no training data)
        - All knowledge comes from obs-phase update_from_output() calls

        This is the most information-constrained setting: tutor knows
        NOTHING about the learner's grammar before observing outputs.

        Args:
            vocabulary: list of words the learner might use
        """
        from ..learner.cls_wrapper import CLSSequencePredictor, _ensure_basic_on_path
        _ensure_basic_on_path()
        from cls_learner.agent import CLSAgent
        from cls_learner.config import CLSConfig

        # Create bare CLS agent with vocabulary only — no study()
        cls_cfg = CLSConfig(
            mode=self.cfg.cls_mode,
            use_hpc=self.cfg.use_hpc,
            n_em=self.cfg.n_em,
        )
        agent = CLSAgent(cls_cfg)
        agent.reset_episode()

        # Register vocabulary so beam search can work
        for w in vocabulary:
            agent.cortex._ensure_concept(w)

        # Wrap in predictor for consistent API
        self._predictor = CLSSequencePredictor(self.cfg)
        self._predictor._agent = agent
        self._predictor._studied = True  # allow predict/beam calls
        self._agent = agent
        self._studied = True
        self._init_mode = 'cold'

    def init_teacher_prior(self, queries: List[Example]):
        """[SCHEME B] Teacher prior — tutor knows the TASK, not the learner.

        Tutor builds a grammar model from the TASK's ground-truth
        query-output pairs (which the teacher naturally knows as the
        course designer). But these are NOT the learner's support
        examples — the learner may have seen different examples.

        This is like "a teacher who designed the exam knows what the
        right answers look like, but doesn't know which practice
        problems the student has already done."

        Args:
            queries: task query Examples with ground-truth outputs
                     (from TutorTaskModel's known queries, NOT learner support)
        """
        from ..learner.cls_wrapper import CLSSequencePredictor

        self._predictor = CLSSequencePredictor(self.cfg)
        self._predictor.fit_support(queries)
        self._agent = self._predictor.get_agent()
        self._studied = True
        self._init_mode = 'teacher_prior'

    # ── Level 1-3: Update from observed output ────────────────

    def update_from_output(self, words: List[str], Y_submit: List[str]):
        """Inverse inference from observed learner output.

        EVIDENCE ABSORBED: "learner produced Y_submit for this query"

        Mechanism:
        1. Convert Y_submit to target_vecs
        2. Run CONSTRAINED beam search with Y_submit as target
           → finds traces that could produce Y_submit
        3. Weighted M-step: update stats from these traces
           (depth controlled by self.update_depth)

        This is the SAME algorithm as CLS study()/EM, but with
        Y_submit (learner's output) instead of ground truth.
        """
        if not self._studied or not Y_submit:
            return

        target_vecs = self._color_to_vecs(Y_submit)
        if not target_vecs:
            return

        library = self._agent.cortex.library
        priors = self._agent.priors

        # Ensure vocabulary
        for w in words:
            self._agent.cortex._ensure_concept(w)

        # Constrained beam search (E-step with Y_submit as target)
        try:
            if self.cfg.cls_mode == 'ast':
                from ns_learner.ns_ast import infer_top_k_ast
                traces = infer_top_k_ast(
                    words, target_vecs, library, priors)
            else:
                from ns_learner.ns_inference import infer_top_k_stack
                traces = infer_top_k_stack(
                    words, target_vecs, library, priors)
        except Exception:
            return

        if not traces:
            return

        # Weighted M-step with depth control
        self._depth_controlled_m_step(traces, words)
        self._n_output_updates += 1

    # ── Level 2-3: Update from feedback ───────────────────────

    def update_from_feedback(
        self,
        words: List[str],
        Y_submit: List[str],
        feedback: dict,
    ):
        """Refine learner model from confirm feedback.

        EVIDENCE ABSORBED: "Y_submit was judged wrong, with mask showing
        which positions were correct/wrong"

        This is DIFFERENT evidence from update_from_output().
        output_update says "learner produced this sequence".
        feedback_update says "that sequence was wrong HERE".

        Uses the SAME feedback likelihood + differential M-step
        as the real learner, but applied to tutor's learner model.
        """
        if not self._studied:
            return

        beam = self._predictor.beam_posterior(words)
        if not beam:
            return

        from ..learner.feedback_update import FeedbackUpdater
        updater = FeedbackUpdater(self.cfg)
        q_old, q_new = updater.reweight_beam_posterior(
            beam, Y_submit, feedback)

        if len(q_old) == 0:
            return

        # Apply depth-controlled differential M-step
        library = self._predictor.get_library()
        self._depth_controlled_diff_m_step(
            library, beam, q_old, q_new)

        self._n_feedback_updates += 1

    # ── Prediction ────────────────────────────────────────────

    def predict_learner(self, words: List[str]) -> Optional[List[str]]:
        """Predict what tutor thinks learner will output.

        Returns None if model not initialized.
        """
        if not self._studied:
            return None
        try:
            return self._predictor.predict_target(words)
        except Exception:
            return None

    def beam_posterior(self, words: List[str]) -> list:
        """Get tutor's estimate of learner's beam posterior."""
        if not self._studied:
            return []
        try:
            return self._predictor.beam_posterior(words)
        except Exception:
            return []

    def get_library(self):
        """Return the learner model's concept library (for diagnostics)."""
        if self._agent is None:
            return {}
        return self._agent.cortex.library

    # ── Diagnostics ───────────────────────────────────────────

    def summary_dict(self) -> Dict:
        """Return diagnostic summary."""
        d = {
            'update_depth': self.update_depth,
            'n_output_updates': self._n_output_updates,
            'n_feedback_updates': self._n_feedback_updates,
            'studied': self._studied,
        }
        # Add per-word role MAP for debugging
        if self._studied and self._agent:
            lib = self._agent.cortex.library
            priors = self._agent.priors
            role_map = {}
            for word, concept in lib.items():
                role_map[word] = concept.map_role(priors.alpha)
            d['role_map'] = role_map
        return d

    # ── Internal helpers ──────────────────────────────────────

    def _color_to_vecs(self, colors: List[str]) -> list:
        """Convert color names to target vectors."""
        if self._agent is None:
            return []
        return self._agent._color_to_vecs(colors)

    def _depth_controlled_m_step(self, traces, words):
        """Weighted M-step with update_depth control.

        role_only:  only update role_counts
        role_emit:  update role_counts + emit_stats
        full_trace: update everything (role + emit + repeat + color)
        """
        from ns_learner.ns_concept import NeuroConcept

        # Compute weights from trace scores
        if self.cfg.cls_mode == 'ast':
            scores = np.array([t[0] for t in traces])
        else:
            scores = np.array([t[0] for t in traces])

        if len(scores) > 1:
            log_w = scores - logsumexp(scores)
            weights = np.exp(log_w)
        else:
            weights = np.array([1.0])

        library = self._agent.cortex.library

        for trace_item, weight in zip(traces, weights):
            # Extract trace steps (last element in tuple)
            trace_steps = trace_item[-1]
            w = weight * self.eta_inv

            for step in trace_steps:
                word = step.word
                if word not in library:
                    continue
                concept = library[word]

                # Always update role counts
                concept.role_counts[step.role] = (
                    concept.role_counts.get(step.role, 0.0) + w)

                if self.update_depth in ('role_emit', 'full_trace'):
                    # Update emit stats
                    if step.role == 'EMIT' and hasattr(step, 'emit_vec') \
                            and step.emit_vec is not None:
                        vec = step.emit_vec
                        concept.emit_stats['sum_w'] += w
                        concept.emit_stats['sum_wx'] += w * vec
                        concept.emit_stats['sum_wx2'] += w * (vec ** 2)

                if self.update_depth == 'full_trace':
                    # Update repeat counts
                    if step.role == 'REPEAT' and hasattr(step, 'repeat_k') \
                            and step.repeat_k is not None:
                        k = step.repeat_k
                        if k in concept.repeat_counts:
                            concept.repeat_counts[k] += w

                    # Update color counts (discrete)
                    if step.role == 'EMIT' and hasattr(step, 'emit_vec') \
                            and step.emit_vec is not None:
                        from ns_learner.ns_concept import vec_to_color
                        c = vec_to_color(step.emit_vec)
                        concept.color_counts[c] = (
                            concept.color_counts.get(c, 0.0) + w)

    def _depth_controlled_diff_m_step(
        self, library, beam, q_old, q_new,
    ):
        """Differential M-step with update_depth control.

        Same as FeedbackUpdater.differential_m_step but respects
        update_depth to only update the specified stats.
        """
        eta = self.cfg.eta_fb * self.eta_inv
        K = len(beam)

        for k in range(K):
            delta_q = q_new[k] - q_old[k]
            if abs(delta_q) < 1e-15:
                continue

            trace = beam[k][1]  # trace steps
            weight = eta * delta_q

            for step in trace:
                word = step.word
                if word not in library:
                    continue
                concept = library[word]
                role = step.role

                # Always update role counts
                concept.role_counts[role] = (
                    concept.role_counts.get(role, 0.0) + weight)

                if self.update_depth in ('role_emit', 'full_trace'):
                    if role == 'EMIT' and hasattr(step, 'emit_vec') \
                            and step.emit_vec is not None:
                        vec = step.emit_vec
                        concept.emit_stats['sum_w'] += weight
                        concept.emit_stats['sum_wx'] += weight * vec
                        concept.emit_stats['sum_wx2'] += weight * (vec ** 2)

                if self.update_depth == 'full_trace':
                    if role == 'REPEAT' and hasattr(step, 'repeat_k') \
                            and step.repeat_k is not None:
                        k_rep = step.repeat_k
                        if k_rep in concept.repeat_counts:
                            concept.repeat_counts[k_rep] += weight

                    if role == 'EMIT' and hasattr(step, 'emit_vec') \
                            and step.emit_vec is not None:
                        from ns_learner.ns_concept import vec_to_color
                        c = vec_to_color(step.emit_vec)
                        concept.color_counts[c] = (
                            concept.color_counts.get(c, 0.0) + weight)
