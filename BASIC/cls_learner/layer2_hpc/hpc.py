"""
hpc.py — EpisodeHPC: top-level wrapper for the hippocampal system.

Composes Encoder → DG → CA3 → CA1 → ReplaySampler.
Provides the same API as the original ns_hpc.EpisodeHPC but uses
the split submodules and blockwise Mahalanobis CA1.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import logsumexp as _logsumexp

from cls_learner.interfaces import MemBias, MemoryPayload, TraceSummary
from cls_learner.config import CLSConfig
from cls_learner.layer2_hpc.encoder import EventEncoder
from cls_learner.layer2_hpc.dg import DGEncoder
from cls_learner.layer2_hpc.ca3 import CA3Memory
from cls_learner.layer2_hpc.ca1 import CA1Comparator
from cls_learner.layer2_hpc.replay import ReplaySampler


# ── Utility: aggregate role boost from retrieved payloads ──────

def _aggregate_role_boost(
    retrieved: List[Tuple[float, int, MemoryPayload]],
    query_words: List[str],
    roles: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-word role distributions from retrieved memories.
    Returns {word: {role: log_softmax_prob}}.
    """
    role_boost: Dict[str, Dict[str, float]] = {}

    for w in query_words:
        role_scores = {r: 0.0 for r in roles}
        total_weight = 0.0

        for sim, idx, payload in retrieved:
            if w in payload.trace_roles:
                w_roles = payload.trace_roles[w]
                for r in roles:
                    role_scores[r] += sim * w_roles.get(r, 0.0)
                total_weight += sim

        if total_weight < 1e-12:
            continue

        raw = np.array([role_scores[r] for r in roles])
        raw = np.maximum(raw, 1e-8)
        log_raw = np.log(raw)
        log_Z = _logsumexp(log_raw)
        log_probs = log_raw - log_Z

        role_boost[w] = {r: float(lp) for r, lp in zip(roles, log_probs)}

    return role_boost


class EpisodeHPC:
    """
    Episode-level Hippocampal system (CLS Layer 2).

    Wraps DG → CA3 → CA1 into a single API for CLSAgent integration.
    Reset at the start of each episode (no cross-episode leakage).
    """

    def __init__(self, cfg: Optional[CLSConfig] = None):
        cfg = cfg or CLSConfig()

        self.encoder = EventEncoder(d_bow=cfg.hpc_d_bow, d_bigr=cfg.hpc_d_bigr)
        d_in = self.encoder.d_out

        self.dg = DGEncoder(d_in=d_in, m=cfg.hpc_m, k=cfg.hpc_k,
                            noise_std=cfg.hpc_noise_std, seed=cfg.hpc_seed)
        self.ca3 = CA3Memory(m=cfg.hpc_m, k=cfg.hpc_k,
                             top_r=cfg.hpc_top_r, eta=cfg.hpc_eta,
                             completion_steps=cfg.hpc_completion_steps)
        self.ca1 = CA1Comparator(
            lam_min=cfg.hpc_lam_min, lam_max=cfg.hpc_lam_max,
            default_th=cfg.ca1_default_th, default_temp=cfg.ca1_default_temp,
            eps=cfg.ca1_eps, mix_a=cfg.ca1_mix_a,
        )
        self.ca1.set_block_ranges(self.encoder.blocks)

        self.replay_sampler = ReplaySampler(
            rho=cfg.replay_priority_rho,
            priority_clip=cfg.replay_priority_clip,
        )

        self._event_cache: List[np.ndarray] = []
        self._delta_cache: List[float] = []  # per-memory mismatch for priority
        self._roles: Optional[List[str]] = None

    def _get_roles(self) -> List[str]:
        if self._roles is None:
            from ns_learner.ns_primitives import ROLES
            self._roles = list(ROLES)
        return self._roles

    def reset(self):
        """Clear all memories (call at episode start)."""
        self.ca3.clear()
        self._event_cache.clear()
        self._delta_cache.clear()
        self.ca1._calibrated = False
        self.ca1._inv_var_blocks = None

    def write_example(self, words: List[str], colors: List[str],
                      trace_summary: Optional[TraceSummary] = None) -> int:
        """
        Encode and store one SUPPORT example.
        Returns memory index for later reconsolidation.
        """
        e = self.encoder.encode_utterance(words)
        h = self.dg.encode(e)

        if trace_summary is not None:
            payload = MemoryPayload.from_trace_summary(words, colors, trace_summary)
        else:
            payload = MemoryPayload(
                words=list(words), colors=list(colors),
                per_word_role={}, per_word_color={}, trace_roles={},
            )

        idx = self.ca3.write(h, e, payload)
        self._event_cache.append(e.copy())
        self._delta_cache.append(0.0)
        return idx

    def update_trace(self, idx: int, trace_summary: TraceSummary):
        """Reconsolidation: update payload with improved trace."""
        if idx < 0 or idx >= len(self.ca3.memories):
            return
        h, e, old_payload = self.ca3.memories[idx]
        new_payload = MemoryPayload(
            words=old_payload.words,
            colors=old_payload.colors,
            per_word_role=trace_summary.per_word_role or old_payload.per_word_role,
            per_word_color=trace_summary.per_word_color or old_payload.per_word_color,
            trace_roles=trace_summary.trace_roles or old_payload.trace_roles,
        )
        self.ca3.update_payload(idx, new_payload)

    def calibrate_gate(self):
        """
        Auto-calibrate CA1 thresholds using blockwise Mahalanobis (S1).

        Uses self-retrieval residuals mixed with feature variance.
        """
        if len(self.ca3.memories) < 2:
            return

        residuals = []  # e_cue - e_recon differences
        features = []   # raw event vectors

        for i, (h_i, e_i, _) in enumerate(self.ca3.memories):
            retrieved = self.ca3.retrieve(h_i)
            events = []
            weights = []
            for sim, j, _ in retrieved:
                if j == i:
                    continue
                events.append(self._event_cache[j])
                weights.append(sim)

            if not events:
                continue

            w_sum = sum(weights)
            if w_sum > 1e-12:
                weights = [w / w_sum for w in weights]

            e_recon = sum(w * e for w, e in zip(weights, events))
            residuals.append(e_i - e_recon)
            features.append(e_i)

        self.ca1.calibrate(
            residuals=residuals,
            features=features,
            block_ranges=self.encoder.blocks,
        )

    def get_bias(self, words: List[str]) -> MemBias:
        """
        Retrieve HPC bias for a query word sequence.

        1. Encode → DG sparse code
        2. CA3 completion + top-R retrieval
        3. CA1 blockwise Mahalanobis mismatch → gate
        4. Aggregate role boosts (log-softmax)
        """
        roles = self._get_roles()

        if not self.ca3.memories:
            return MemBias(role_boost={})

        e_cue = self.encoder.encode_utterance(words)
        h_q = self.dg.encode(e_cue)

        h_complete = self.ca3.complete(h_q)
        retrieved = self.ca3.retrieve(h_complete)

        events = []
        weights = []
        for sim, idx, _ in retrieved:
            events.append(self._event_cache[idx])
            weights.append(sim)

        w_sum = sum(weights)
        if w_sum > 1e-12:
            weights_norm = [w / w_sum for w in weights]
        else:
            weights_norm = [1.0 / len(weights)] * len(weights)

        delta = self.ca1.mismatch(e_cue, events, weights_norm)
        lam_mem, mode = self.ca1.gate(delta)

        role_boost = _aggregate_role_boost(retrieved, words, roles)

        return MemBias(
            role_boost=role_boost,
            lam_mem=lam_mem,
            delta=delta,
            mode=mode,
        )

    def sample_replay(self, batch_size: int = 3) -> List[MemoryPayload]:
        """
        Sample memories for replay with mixed uniform+priority.
        Priority based on per-memory mismatch deltas.
        """
        return self.replay_sampler.sample(
            memories=self.ca3.memories,
            deltas=self._delta_cache if self._delta_cache else None,
            batch_size=batch_size,
        )
