"""
cortex.py — CortexMemory: Layer 1 slow/generalizable concept system.

Wraps the existing NeuroConcept library and NSLearner's bootstrap + EM
logic into a clean interface for CLSAgent.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import logsumexp

from cls_learner.interfaces import Example, TraceSummary, MemoryPayload
from cls_learner.config import CLSConfig
from cls_learner.layer1_cortex.concept_adapter import ConceptAdapter

from ns_learner.ns_concept import (
    NeuroConcept, NIGParams, ROLES, REPEAT_RANGE,
    N_COLORS, COLORS, COLOR_VECS, color_to_vec,
)
from ns_learner.ns_learner import GlobalPriors


class CortexMemory:
    """
    Layer 1: Slow, generalizable concept memory.

    Wraps the concept library (Dict[str, NeuroConcept]) and provides:
      - fit_support(): bootstrap + EM learning
      - score_role/emit(): scoring for beam search
      - replay_update(): soft update from HPC replays
      - decay(): between-EM-iter count decay
    """

    def __init__(self, cfg: CLSConfig, priors: Optional[GlobalPriors] = None):
        self.cfg = cfg
        self.priors = priors or GlobalPriors()
        self.priors.rsa_alpha = cfg.rsa_alpha

        # Concept library: word → NeuroConcept
        self.library: Dict[str, NeuroConcept] = {}

    def _ensure_concept(self, word: str):
        """Create a NeuroConcept for word if it doesn't exist."""
        if word not in self.library:
            d = len(self.priors.nig.mu0)  # 3 for Lab, 6 for one-hot
            self.library[word] = NeuroConcept(word, d=d)

    def ensure_vocabulary(self, examples: List[Example]):
        """Ensure concepts exist for all words in examples."""
        for ex in examples:
            for w in ex.words:
                self._ensure_concept(w)

    def get_adapter(self, word: str) -> ConceptAdapter:
        """Get adapter for a word's concept."""
        self._ensure_concept(word)
        return ConceptAdapter(self.library[word])

    # ── Bootstrap ──────────────────────────────────────────────

    def bootstrap(self, examples: List[Example], verbose: bool = False):
        """
        Phase 0: learn obvious 1:1 mappings (noun detection).

        Delegates to the existing NSLearner bootstrap logic.
        """
        from ns_learner.ns_learner import NSLearner

        # Create a temporary NSLearner to run bootstrap
        learner = NSLearner(priors=self.priors, n_em=0, use_hpc=False)
        learner.library = self.library  # share the library

        examples_dicts = [{'input': ex.words, 'output': ex.output} for ex in examples]
        learner._bootstrap_nouns(examples_dicts, verbose)

    # ── Scoring (for beam search) ──────────────────────────────

    def score_role(self, word: str, role: str) -> float:
        """Log role probability for word."""
        self._ensure_concept(word)
        return self.library[word].log_role_prob(role, self.priors.alpha)

    def score_emit(self, word: str, vec: np.ndarray) -> float:
        """Log emission probability for word."""
        self._ensure_concept(word)
        return self.library[word].log_emit_prob(
            vec, self.priors.nig,
            self.priors.eps_obj, self.priors.tau_inc,
            delta=self.priors.delta,
        )

    # ── M-step: update from traces ─────────────────────────────

    @staticmethod
    def _unpack_trace(t):
        """Unpack trace from either (score, trace) or (score, roots, trace)."""
        return t[0], t[-1]  # score is always first, trace is always last

    def m_step_from_traces(self, traces_per_example: List[Optional[list]],
                           mem_biases: Optional[List[Optional['MemBias']]] = None,
                           n_support: int = 0):
        """
        M-step: accumulate weighted stats from beam traces.

        With IS correction (adjustment 1):
          log_w = (log_p_model - log_q) / T_resp
        where log_q is the HPC proposal contribution.

        With per-step normalization (adjustment 2):
          Scores divided by trace length before softmax.

        With adaptive temperature (adjustment 2b):
          T_resp = T0 * sqrt(1 + n_support/10)
        """
        from cls_learner.interfaces import MemBias

        # Adaptive temperature
        T = self.cfg.T_resp_base
        if self.cfg.T_resp_scale_by_support and n_support > 0:
            T = T * np.sqrt(1.0 + n_support / 10.0)
        T = np.clip(T, self.cfg.T_resp_min, self.cfg.T_resp_max)

        for ex_idx, traces in enumerate(traces_per_example):
            if not traces:
                continue

            # Get mem_bias for this example (if available)
            mb = None
            if mem_biases and ex_idx < len(mem_biases):
                mb = mem_biases[ex_idx]

            # Compute IS-corrected scores
            log_ws = []
            for t in traces:
                score, trace = self._unpack_trace(t)

                if self.cfg.use_is_correction and mb is not None:
                    # Reconstruct proposal contribution
                    log_q = mb.log_q_trace(trace)
                    log_p_model = score - log_q
                else:
                    log_p_model = score

                # Per-step normalization
                if self.cfg.norm_by_steps and len(trace) > 0:
                    log_p_model = log_p_model / len(trace)

                log_ws.append(log_p_model)

            log_ws = np.array(log_ws)

            # Temperature-scaled softmax
            if len(log_ws) > 1:
                scaled = log_ws / T
                scaled_norm = scaled - logsumexp(scaled)
                weights = np.exp(scaled_norm)
            else:
                weights = np.array([1.0])

            for t, weight in zip(traces, weights):
                score, trace = self._unpack_trace(t)
                for step in trace:
                    concept = self.library[step.word]
                    concept.soft_update(
                        weight=weight, role=step.role,
                        vec=step.emit_vec, k=step.repeat_k,
                    )
                    if step.b_word and step.b_vec is not None:
                        b_concept = self.library.get(step.b_word)
                        if b_concept:
                            b_concept.soft_update(
                                weight=weight, role='EMIT', vec=step.b_vec)

    # ── Replay update ──────────────────────────────────────────

    def replay_update(self, payload: MemoryPayload, weight: float = 0.2):
        """
        Soft update from HPC replay (S3: uses trace_roles distribution).
        """
        for word, role_dist in payload.trace_roles.items():
            if word in self.library:
                concept = self.library[word]
                total_r = sum(role_dist.values())
                if total_r > 0:
                    for role, count in role_dist.items():
                        if count > 0 and role in ROLES:
                            concept.soft_update(
                                weight=weight * count / total_r,
                                role=role)

    # ── Decay ──────────────────────────────────────────────────

    def decay(self, rate: Optional[float] = None):
        """Decay role/repeat counts between EM iterations."""
        rate = rate if rate is not None else self.cfg.decay_rate
        for concept in self.library.values():
            for r in ROLES:
                concept.role_counts[r] *= rate
            for k in REPEAT_RANGE:
                concept.repeat_counts[k] *= rate

    # ── Trace summary extraction ───────────────────────────────

    def extract_trace_summary(self, example: Example,
                               traces: Optional[list] = None) -> TraceSummary:
        """
        Build TraceSummary from best trace or current concept MAP.
        Replaces NSLearner._extract_trace_summary.
        """
        per_word_role: Dict[str, str] = {}
        per_word_color: Dict[str, str] = {}
        trace_roles: Dict[str, Dict[str, float]] = {}

        if traces and len(traces) > 0:
            scores = np.array([t[0] for t in traces])
            if len(scores) > 1:
                log_w = scores - logsumexp(scores)
                weights = np.exp(log_w)
            else:
                weights = np.array([1.0])

            for t, w in zip(traces, weights):
                score, trace = self._unpack_trace(t)
                for step in trace:
                    word = step.word
                    role = step.role
                    if word not in trace_roles:
                        trace_roles[word] = {r: 0.0 for r in ROLES}
                    trace_roles[word][role] += w

            for word, role_dist in trace_roles.items():
                best_role = max(role_dist, key=role_dist.get)
                per_word_role[word] = best_role
                if best_role == 'EMIT' and word in self.library:
                    per_word_color[word] = self.library[word].map_color(
                        self.priors.nig, self.priors.eps_obj,
                        self.priors.tau_inc, delta=self.priors.delta)
        else:
            for w in example.words:
                if w in self.library:
                    c = self.library[w]
                    mr = c.map_role(self.priors.alpha)
                    per_word_role[w] = mr
                    trace_roles[w] = {
                        r: c.role_counts.get(r, 0.0) for r in ROLES
                    }
                    if mr == 'EMIT':
                        per_word_color[w] = c.map_color(
                            self.priors.nig, self.priors.eps_obj,
                            self.priors.tau_inc, delta=self.priors.delta)

        return TraceSummary(
            per_word_role=per_word_role,
            per_word_color=per_word_color,
            trace_roles=trace_roles,
        )
