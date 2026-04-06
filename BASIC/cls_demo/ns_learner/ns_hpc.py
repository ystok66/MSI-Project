"""
ns_hpc.py — Hippocampal fast-memory module (CLS Layer 2).

Three-component architecture modelling DG–CA3–CA1:

  DG  (Dentate Gyrus)  : random projection + kWTA → sparse pattern separation
  CA3 (Cornu Ammonis 3): Hopfield auto-associative memory → pattern completion
  CA1 (Cornu Ammonis 1): mismatch / novelty gate → encode vs retrieve mode

The module sits between Cortex (ns_concept, slow/generalizable) and
PFC-BG (ns_ast/ns_inference, beam search selection / planning).

Usage within an episode:
    hpc = EpisodeHPC()
    hpc.reset()                                    # start of episode

    # — encoding phase (SUPPORT) —
    for ex in support:
        idx = hpc.write_example(ex['input'], ex['output'], trace_steps)
    hpc.calibrate_gate()                           # set CA1 thresholds

    # — reconsolidation (after each EM iter) —
    hpc.update_trace(idx, new_trace_summary)

    # — retrieval phase (QUERY) —
    mem_bias = hpc.get_bias(query_words)           # → MemBias
    # pass mem_bias to infer_top_k / infer_top_k_ast

    # — replay —
    replays = hpc.sample_replay(batch_size=3)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib


# ── Data Structures ────────────────────────────────────────────

@dataclass
class MemBias:
    """HPC output to beam search: per-word role boosts + gating signal."""
    role_boost: Dict[str, Dict[str, float]]  # {word: {role: log_softmax_prob}}
    lam_mem: float                            # gated memory strength [0, lam_max]
    delta: float                              # raw mismatch signal
    mode: str                                 # 'retrieve' / 'explore' / 'mixed'


@dataclass
class MemoryPayload:
    """What HPC stores per example (bound to the DG sparse code)."""
    words: List[str]
    colors: List[str]
    per_word_role: Dict[str, str]               # MAP role per word
    per_word_color: Dict[str, str]              # MAP emit color (EMIT words only)
    trace_roles: Dict[str, Dict[str, float]]    # full role distribution per word


# ── Hashing Utilities ──────────────────────────────────────────

def _token_hash(token: str, d: int) -> int:
    """Deterministic hash of a token string to bucket index in [0, d)."""
    h = hashlib.md5(token.encode('utf-8')).hexdigest()
    return int(h, 16) % d


def _bigram_hash(t1: str, t2: str, d: int) -> int:
    """Deterministic hash of a bigram to bucket index in [0, d)."""
    h = hashlib.md5(f"{t1}||{t2}".encode('utf-8')).hexdigest()
    return int(h, 16) % d


# ── EventEncoder ───────────────────────────────────────────────

class EventEncoder:
    """
    Encode utterance (word sequence) into a fixed-size feature vector.

    Only the utterance cue is used for DG indexing (M1: query has no
    output/trace, so including them would cause DG code divergence).

    Features:
      φ_bow(x):  hash-BOW over tokens              (d_bow dims)
      φ_bigr(x): hash over adjacent token bigrams   (d_bigr dims)
    """

    def __init__(self, d_bow: int = 64, d_bigr: int = 64):
        self.d_bow = d_bow
        self.d_bigr = d_bigr
        self.d_out = d_bow + d_bigr

    def encode_utterance(self, words: List[str]) -> np.ndarray:
        """Encode word sequence → 128D L2-normalized feature vector."""
        phi_bow = np.zeros(self.d_bow)
        phi_bigr = np.zeros(self.d_bigr)

        # BOW: hash each token to a bucket and increment
        for w in words:
            idx = _token_hash(w, self.d_bow)
            phi_bow[idx] += 1.0

        # Bigrams: hash adjacent pairs
        for i in range(len(words) - 1):
            idx = _bigram_hash(words[i], words[i + 1], self.d_bigr)
            phi_bigr[idx] += 1.0

        e = np.concatenate([phi_bow, phi_bigr])

        # L2 normalize
        norm = np.linalg.norm(e)
        if norm > 1e-12:
            e = e / norm

        return e


# ── DGEncoder (Dentate Gyrus) ──────────────────────────────────

def _kwta(u: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All: keep top-k activations, zero the rest."""
    if k >= len(u):
        return u.copy()
    # Find the k-th largest value
    threshold = np.partition(u, -k)[-k]
    h = np.where(u >= threshold, u, 0.0)
    # If ties cause more than k active, keep exactly k by zeroing extras
    active = np.nonzero(h)[0]
    if len(active) > k:
        # Sort by value, keep top k
        sorted_idx = active[np.argsort(h[active])]
        to_zero = sorted_idx[:len(active) - k]
        h[to_zero] = 0.0
    return h


class DGEncoder:
    """
    Dentate Gyrus: pattern separation via random projection + kWTA.

    Input:  event vector e ∈ R^{d_in}  (from EventEncoder, 128D)
    Output: sparse code h ∈ R^m        (only k entries non-zero)

    The random projection W is fixed per episode (not learned).
    """

    def __init__(self, d_in: int = 128, m: int = 512, k: int = 30,
                 noise_std: float = 0.01, seed: int = 42):
        self.d_in = d_in
        self.m = m
        self.k = k
        self.noise_std = noise_std
        rng = np.random.RandomState(seed)
        # Xavier-like init for distance preservation
        self.W = rng.randn(m, d_in) / np.sqrt(d_in)

    def encode(self, e: np.ndarray) -> np.ndarray:
        """Project + noise + kWTA → sparse code."""
        u = self.W @ e
        if self.noise_std > 0:
            u += np.random.randn(self.m) * self.noise_std
        return _kwta(u, self.k)


# ── CA3Memory ──────────────────────────────────────────────────

class CA3Memory:
    """
    CA3 auto-associative memory with Hopfield matrix + list storage.

    Write:    store (h, e, payload) and update M += η·outer(h,h)
    Retrieve: iterative completion via M, then find nearest neighbors
              in stored list to recover payloads.
    """

    def __init__(self, m: int = 512, k: int = 30,
                 top_r: int = 5, eta: float = 1.0,
                 completion_steps: int = 3, temp: float = 1.0):
        self.m = m
        self.k = k
        self.top_r = top_r
        self.eta = eta
        self.completion_steps = completion_steps
        self.temp = temp

        # Hopfield weight matrix (M2: true auto-associative)
        self.M = np.zeros((m, m), dtype=np.float32)

        # Explicit list for payload binding
        self.memories: List[Tuple[np.ndarray, np.ndarray, MemoryPayload]] = []

    def clear(self):
        """Reset all memories."""
        self.M[:] = 0.0
        self.memories.clear()

    def write(self, h: np.ndarray, e: np.ndarray,
              payload: MemoryPayload) -> int:
        """
        Store one memory. Returns the memory index (M3).

        Updates Hopfield matrix M and appends to list.
        """
        idx = len(self.memories)
        self.memories.append((h.copy(), e.copy(), payload))

        # Hopfield outer-product learning rule (diagonal removed)
        outer = np.outer(h, h)
        np.fill_diagonal(outer, 0.0)
        self.M += self.eta * outer

        return idx

    def update_payload(self, idx: int, new_payload: MemoryPayload):
        """Reconsolidation: update payload without changing DG code."""
        if 0 <= idx < len(self.memories):
            h, e, _ = self.memories[idx]
            self.memories[idx] = (h, e, new_payload)

    def retrieve(self, h_q: np.ndarray,
                 top_r: Optional[int] = None) -> List[Tuple[float, int, MemoryPayload]]:
        """
        Retrieve top-R most similar memories by sparse dot-product.

        Returns list of (similarity, idx, payload) sorted descending.
        """
        if not self.memories:
            return []

        top_r = top_r or self.top_r
        sims = []
        for i, (h_i, e_i, payload) in enumerate(self.memories):
            sim = float(np.dot(h_q, h_i))
            sims.append((sim, i, payload))

        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_r]

    def complete(self, h_q: np.ndarray) -> np.ndarray:
        """
        Pattern completion via Hopfield iteration:
          h ← kWTA(M @ h, k)  repeated `completion_steps` times.
        """
        h = h_q.copy()
        for _ in range(self.completion_steps):
            h = _kwta(self.M @ h, self.k)
        return h


# ── CA1Comparator ──────────────────────────────────────────────

class CA1Comparator:
    """
    CA1 mismatch / novelty gate.

    Compares the query cue with retrieved memories to determine
    whether to enter retrieval mode (familiar) or exploration mode (novel).

    Thresholds are auto-calibrated from support self-retrieval (S1).
    Output: continuous sigmoid-gated λ_mem with hysteresis.
    """

    def __init__(self, lam_min: float = 0.0, lam_max: float = 1.0,
                 default_th: float = 0.5, default_temp: float = 0.1):
        self.lam_min = lam_min
        self.lam_max = lam_max
        # These get overwritten by calibrate()
        self.th = default_th
        self.temp = default_temp
        self.th_low = default_th * 0.6
        self.th_high = default_th * 1.4
        self._calibrated = False

    def calibrate(self, delta_samples: List[float]):
        """
        Auto-set thresholds from support self-retrieval deltas (S1).

        th_low  = 30th percentile  → below this: strong retrieval
        th_high = 70th percentile  → above this: exploration
        temp    = 0.1 * (th_high - th_low)
        """
        if not delta_samples or len(delta_samples) < 2:
            return

        arr = np.array(delta_samples)
        self.th_low = float(np.percentile(arr, 30))
        self.th_high = float(np.percentile(arr, 70))
        self.th = (self.th_low + self.th_high) / 2.0

        spread = max(self.th_high - self.th_low, 1e-6)
        self.temp = 0.1 * spread
        self._calibrated = True

    def mismatch(self, e_cue: np.ndarray,
                 retrieved_events: List[np.ndarray],
                 weights: List[float]) -> float:
        """
        Compute mismatch between cue and weighted reconstruction.

        delta = ||e_cue - Σ w_i e_i||
        """
        if not retrieved_events:
            return float('inf')

        e_recon = sum(w * e for w, e in zip(weights, retrieved_events))
        delta = float(np.linalg.norm(e_cue - e_recon))
        return delta

    def gate(self, delta: float) -> Tuple[float, str]:
        """
        Continuous sigmoid gating with hysteresis.

        g(δ) = σ((θ - δ) / T)
        λ_mem = λ_min + (λ_max - λ_min) * g(δ)

        Returns (lam_mem, mode).
        """
        if self.temp < 1e-12:
            g = 1.0 if delta < self.th else 0.0
        else:
            z = (self.th - delta) / self.temp
            z = np.clip(z, -20.0, 20.0)
            g = 1.0 / (1.0 + np.exp(-z))

        lam_mem = self.lam_min + (self.lam_max - self.lam_min) * float(g)

        if delta < self.th_low:
            mode = 'retrieve'
        elif delta > self.th_high:
            mode = 'explore'
        else:
            mode = 'mixed'

        return lam_mem, mode


# ── Utility: build role distribution from retrieved payloads ───

def _aggregate_role_boost(
    retrieved: List[Tuple[float, int, MemoryPayload]],
    query_words: List[str],
    roles: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-word role distributions from retrieved memories.

    For each query word that appears in any retrieved memory's trace_roles,
    compute a similarity-weighted role distribution, then log-softmax (M4).

    Returns {word: {role: log_softmax_prob}}.
    """
    from scipy.special import logsumexp as _logsumexp

    role_boost: Dict[str, Dict[str, float]] = {}

    for w in query_words:
        # Gather weighted role counts from all retrieved memories
        role_scores = {r: 0.0 for r in roles}
        total_weight = 0.0

        for sim, idx, payload in retrieved:
            if w in payload.trace_roles:
                w_roles = payload.trace_roles[w]
                for r in roles:
                    role_scores[r] += sim * w_roles.get(r, 0.0)
                total_weight += sim

        if total_weight < 1e-12:
            continue  # no info about this word → no boost

        # Normalize to get a probability, then log-softmax
        raw = np.array([role_scores[r] for r in roles])
        # Add small floor to prevent log(0)
        raw = np.maximum(raw, 1e-8)
        log_raw = np.log(raw)
        log_Z = _logsumexp(log_raw)
        log_probs = log_raw - log_Z

        role_boost[w] = {r: float(lp) for r, lp in zip(roles, log_probs)}

    return role_boost


# ── EpisodeHPC (top-level wrapper) ─────────────────────────────

class EpisodeHPC:
    """
    Episode-level Hippocampal system.

    Wraps DG → CA3 → CA1 into a single API for NSLearner integration.
    Reset at the start of each episode (no cross-episode leakage).
    """

    def __init__(self,
                 d_bow: int = 64, d_bigr: int = 64,
                 m: int = 512, k: int = 30,
                 noise_std: float = 0.01,
                 top_r: int = 5,
                 eta: float = 1.0,
                 completion_steps: int = 3,
                 lam_min: float = 0.0, lam_max: float = 1.0,
                 seed: int = 42):

        self.encoder = EventEncoder(d_bow=d_bow, d_bigr=d_bigr)
        d_in = self.encoder.d_out

        self.dg = DGEncoder(d_in=d_in, m=m, k=k,
                            noise_std=noise_std, seed=seed)
        self.ca3 = CA3Memory(m=m, k=k, top_r=top_r, eta=eta,
                             completion_steps=completion_steps)
        self.ca1 = CA1Comparator(lam_min=lam_min, lam_max=lam_max)

        # Cache event vectors for calibration / replay
        self._event_cache: List[np.ndarray] = []  # parallel to ca3.memories

        # Import roles lazily (avoid circular import)
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
        self.ca1._calibrated = False

    def write_example(self, words: List[str], colors: List[str],
                      trace_summary: Optional[Dict] = None) -> int:
        """
        Encode and store one SUPPORT example.

        Returns memory index for later reconsolidation (M3).

        Args:
            words: input token sequence
            colors: output color names
            trace_summary: dict with 'per_word_role', 'trace_roles',
                           'per_word_color' keys (from MAP trace)
        """
        e = self.encoder.encode_utterance(words)
        h = self.dg.encode(e)

        # Build payload
        if trace_summary is None:
            trace_summary = {}

        payload = MemoryPayload(
            words=list(words),
            colors=list(colors),
            per_word_role=trace_summary.get('per_word_role', {}),
            per_word_color=trace_summary.get('per_word_color', {}),
            trace_roles=trace_summary.get('trace_roles', {}),
        )

        idx = self.ca3.write(h, e, payload)
        self._event_cache.append(e.copy())
        return idx

    def update_trace(self, idx: int, trace_summary: Dict):
        """
        Reconsolidation: update a memory's payload with improved trace
        from a later EM iteration.  DG code (index) stays unchanged.
        """
        if idx < 0 or idx >= len(self.ca3.memories):
            return
        h, e, old_payload = self.ca3.memories[idx]
        new_payload = MemoryPayload(
            words=old_payload.words,
            colors=old_payload.colors,
            per_word_role=trace_summary.get('per_word_role',
                                            old_payload.per_word_role),
            per_word_color=trace_summary.get('per_word_color',
                                             old_payload.per_word_color),
            trace_roles=trace_summary.get('trace_roles',
                                          old_payload.trace_roles),
        )
        self.ca3.update_payload(idx, new_payload)

    def calibrate_gate(self):
        """
        Auto-calibrate CA1 thresholds (S1).

        Use each stored support example as a cue, retrieve, and compute
        mismatch.  The distribution of these "self-retrieval" deltas
        sets th_low/th_high.
        """
        if len(self.ca3.memories) < 2:
            return

        deltas = []
        for i, (h_i, e_i, _) in enumerate(self.ca3.memories):
            retrieved = self.ca3.retrieve(h_i)
            # Exclude self from reconstruction
            events = []
            weights = []
            for sim, j, _ in retrieved:
                if j == i:
                    continue
                events.append(self._event_cache[j])
                weights.append(sim)

            if not events:
                continue

            # Normalize weights
            w_sum = sum(weights)
            if w_sum > 1e-12:
                weights = [w / w_sum for w in weights]

            delta = self.ca1.mismatch(e_i, events, weights)
            deltas.append(delta)

        self.ca1.calibrate(deltas)

    def get_bias(self, words: List[str]) -> MemBias:
        """
        Retrieve HPC bias for a query (or support) word sequence.

        1. Encode utterance → DG sparse code
        2. CA3 pattern completion + top-R retrieval
        3. CA1 mismatch → gate λ_mem, mode
        4. Aggregate role boosts (log-softmax, M4)

        Returns MemBias with per-word role boosts + gating info.
        """
        roles = self._get_roles()

        if not self.ca3.memories:
            return MemBias(
                role_boost={}, lam_mem=0.0,
                delta=float('inf'), mode='explore'
            )

        e_cue = self.encoder.encode_utterance(words)
        h_q = self.dg.encode(e_cue)

        # Pattern completion
        h_complete = self.ca3.complete(h_q)

        # Retrieve top-R by completed code
        retrieved = self.ca3.retrieve(h_complete)

        # Compute mismatch
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

        # Aggregate role boosts
        role_boost = _aggregate_role_boost(retrieved, words, roles)

        return MemBias(
            role_boost=role_boost,
            lam_mem=lam_mem,
            delta=delta,
            mode=mode,
        )

    def sample_replay(self, batch_size: int = 3) -> List[MemoryPayload]:
        """
        Sample memories for replay (S3: includes trace_roles for soft update).

        Uses uniform sampling for v0.  Priority sampling (by mismatch
        or uncertainty) is a v1 enhancement.
        """
        if not self.ca3.memories:
            return []

        n = len(self.ca3.memories)
        k = min(batch_size, n)
        indices = np.random.choice(n, size=k, replace=False)
        return [self.ca3.memories[i][2] for i in indices]
