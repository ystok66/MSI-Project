"""
interfaces.py — Unified dataclasses for CLS three-layer system.

All inter-layer communication uses these types instead of raw dicts.
MemoryPayload and TraceSummary are kept separate per user requirement (E):
  - TraceSummary: ephemeral, one per inference call (many per episode)
  - MemoryPayload: persistent, stored in HPC (one per written example)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Task-level data ────────────────────────────────────────────

@dataclass
class Example:
    """One input→output pair (support or query)."""
    words: List[str]
    output: List[str]
    meta: dict = field(default_factory=dict)


@dataclass
class Episode:
    """One learning episode: support examples + query examples."""
    support: List[Example]
    query: List[Example]


# ── Inference results ──────────────────────────────────────────

@dataclass
class TraceSummary:
    """
    Summary extracted from one beam-search / AST inference pass.

    Ephemeral — produced by every E-step or predict call.
    May contain weighted distributions from multiple beam candidates.
    """
    per_word_role: Dict[str, str]                   # MAP role per word
    per_word_color: Dict[str, str]                  # MAP emit color (EMIT words)
    trace_roles: Dict[str, Dict[str, float]]        # soft role distribution
    score: float = 0.0                               # log-likelihood
    log_p_model: float = 0.0                         # model score (no HPC)
    log_q: float = 0.0                               # proposal score (HPC only)
    raw_traces: Optional[list] = None                # beam traces (opaque)
    stats: dict = field(default_factory=dict)


# ── HPC types ──────────────────────────────────────────────────

@dataclass
class MemoryPayload:
    """
    What HPC stores per example (bound to DG sparse code).

    Persistent — written once, reconsolidated per EM iteration.
    Separate from TraceSummary to keep lifecycles clean.
    """
    words: List[str]
    colors: List[str]
    per_word_role: Dict[str, str]
    per_word_color: Dict[str, str]
    trace_roles: Dict[str, Dict[str, float]]

    @staticmethod
    def from_trace_summary(words: List[str], colors: List[str],
                           ts: TraceSummary) -> 'MemoryPayload':
        """Convert a TraceSummary to a MemoryPayload for HPC storage."""
        return MemoryPayload(
            words=list(words),
            colors=list(colors),
            per_word_role=dict(ts.per_word_role),
            per_word_color=dict(ts.per_word_color),
            trace_roles={w: dict(d) for w, d in ts.trace_roles.items()},
        )


@dataclass
class MemBias:
    """
    HPC output to beam search: per-word role boosts + gating signal.

    Produced by EpisodeHPC.get_bias(), consumed by PFCPlanner.
    Also provides log_q_trace() for IS correction in E-step.
    """
    role_boost: Dict[str, Dict[str, float]]   # {word: {role: log_softmax_prob}}
    emit_boost: Dict[str, np.ndarray] = field(default_factory=dict)  # optional
    lam_mem: float = 0.0                       # gated memory strength [0, lam_max]
    delta: float = float('inf')                # raw mismatch signal
    mode: str = 'explore'                      # 'retrieve' / 'explore' / 'mixed'

    def log_q_trace(self, trace) -> float:
        """
        Reconstruct the HPC proposal contribution for a trace.

        In beam search, each step adds: lam_mem * role_boost[word].get(role, 0)
        This computes the sum of those additions = log_q.
        IS correction: log_w = log_p_model - log_q
        where log_p_model = trace.score - log_q.
        """
        if self.lam_mem <= 0 or not self.role_boost:
            return 0.0
        total = 0.0
        for step in trace:
            w = step.word
            r = step.role
            if w in self.role_boost:
                total += self.lam_mem * self.role_boost[w].get(r, 0.0)
        return total
