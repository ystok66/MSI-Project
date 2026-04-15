"""
ns_concept.py — Per-word probabilistic model (NeuroConcept).

Each word carries Dirichlet posteriors over:
  1. Role:   P(role | word)  ∈ Δ{EMIT, REPEAT, SWAP_INFIX, CONCAT_INFIX, OVER_INFIX}
  2. Repeat: P(k | word)    ∈ Δ{1, 2, 3, 4}

And NIG-style (Normal-Inverse-Gamma) sufficient statistics for:
  3. Visual: P(color_vec | word)  — used when role=EMIT

Scoring uses Dirichlet predictive (posterior mean), and plug-in
Gaussian for the visual component.

Soft updates accumulate weighted sufficient statistics (for EM M-step).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
import functools

from .ns_primitives import ROLES, N_ROLES


# ── Constants ───────────────────────────────────────────────────

REPEAT_RANGE = [1, 2, 3, 4]
N_REPEATS = len(REPEAT_RANGE)

# Default 6-color palette for MLC tasks
COLORS = ['BLUE', 'RED', 'GREEN', 'YELLOW', 'PURPLE', 'PINK']
N_COLORS = len(COLORS)
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}


def color_to_vec(color: str, d: int = N_COLORS) -> np.ndarray:
    """One-hot encode a color name to a d-dim vector."""
    v = np.zeros(d)
    if color in COLOR_TO_IDX:
        v[COLOR_TO_IDX[color]] = 1.0
    return v


# Pre-computed one-hot vectors
COLOR_VECS = {c: color_to_vec(c) for c in COLORS}
# Pre-stacked matrix for vectorized scoring (6, 6)
_ALL_COLOR_VECS = np.stack([COLOR_VECS[c] for c in COLORS])  # shape (N_COLORS, D)


def vec_to_color(vec: np.ndarray) -> str:
    """Decode a vector to the nearest color name (argmax)."""
    idx = int(np.argmax(vec[:N_COLORS]))
    if idx < len(COLORS):
        return COLORS[idx]
    return '?'


# ── NIG Prior Parameters ───────────────────────────────────────

class NIGParams:
    """Normal-Inverse-Gamma prior parameters for visual concept."""
    __slots__ = ('mu0', 'kappa0', 'alpha0', 'beta0')

    def __init__(self, d: int = N_COLORS,
                 mu0: Optional[np.ndarray] = None,
                 kappa0: float = 0.1,
                 alpha0: float = 1.0,
                 beta0: float = 1.0):
        self.mu0 = mu0 if mu0 is not None else np.full(d, 1.0 / d)
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0


# ── NeuroConcept ───────────────────────────────────────────────

class NeuroConcept:
    """
    Per-word probabilistic model.
    
    Tracks sufficient statistics accumulated from soft-EM iterations.
    Scoring uses global priors (passed in) combined with local counts.
    """

    def __init__(self, name: str, d: int = N_COLORS):
        self.name = name
        self.d = d

        # Dirichlet sufficient stats for role posterior
        self.role_counts: Dict[str, float] = {r: 0.0 for r in ROLES}

        # Dirichlet sufficient stats for repeat-k posterior
        self.repeat_counts: Dict[int, float] = {k: 0.0 for k in REPEAT_RANGE}

        # Weighted moments for visual (EMIT) posterior — continuous model
        self.emit_stats = {
            'sum_w':   0.0,              # Σ weight
            'sum_wx':  np.zeros(d),      # Σ weight * vec
            'sum_wx2': np.zeros(d),      # Σ weight * vec²
        }

        # Discrete color counts — BPL discrete baseline
        self.color_counts: Dict[str, float] = {c: 0.0 for c in COLORS}

    # ── Predictive Scoring ──────────────────────────────────────

    def log_role_prob(self, role: str, prior_alpha: Dict[str, float]) -> float:
        """
        Log Dirichlet predictive probability for a role.
        
        P(role | w) = (α[role] + count[role]) / Σ(α + count)
        """
        eff = prior_alpha[role] + self.role_counts[role]
        total = sum(prior_alpha[r] + self.role_counts[r] for r in ROLES)
        if total <= 0:
            return -np.log(N_ROLES)
        return np.log(max(eff, 1e-30)) - np.log(total)

    def log_repeat_prob(self, k: int, prior_gamma: Dict[int, float]) -> float:
        """
        Log Dirichlet predictive probability for repeat count k.
        
        P(k | w) = (γ[k] + count[k]) / Σ(γ + count)
        """
        eff = prior_gamma[k] + self.repeat_counts.get(k, 0.0)
        total = sum(prior_gamma[ki] + self.repeat_counts.get(ki, 0.0)
                    for ki in REPEAT_RANGE)
        if total <= 0:
            return -np.log(N_REPEATS)
        return np.log(max(eff, 1e-30)) - np.log(total)

    def log_emit_prob(self, vec: np.ndarray, nig: NIGParams,
                      eps_obj: float = 1e-3, tau_inc: float = 1.0,
                      delta: Optional[Dict[str, float]] = None,
                      gauss: bool = False) -> float:
        """
        Log emission probability.
        
        Three modes:
          delta=dict  → Discrete (Dirichlet): log(δ_c + count_c) - log(Σ)
          gauss=True  → Gaussian log-likelihood: log N(vec | μ_post, σ²_post)
          else        → Continuous (NIG/KL):  -D_KL(P_obj || P_conc) / τ_inc
        """
        if delta is not None:
            # ── Discrete Dirichlet predictive ──────────────────
            c = vec_to_color(vec)
            num = delta.get(c, 1.0) + self.color_counts.get(c, 0.0)
            denom = sum(delta.get(ci, 1.0) + self.color_counts.get(ci, 0.0)
                        for ci in COLORS)
            return float(np.log(max(num, 1e-30)) - np.log(max(denom, 1e-30)))

        # Compute posterior mean and variance (shared by KL and Gauss modes)
        sw = self.emit_stats['sum_w']
        swx = self.emit_stats['sum_wx']

        denom = nig.kappa0 + sw
        if denom < 1e-30:
            mu_post = nig.mu0
            var_post = np.full_like(mu_post, 1.0)
        else:
            mu_post = (nig.kappa0 * nig.mu0 + swx) / denom
            swx2 = self.emit_stats['sum_wx2']
            sse = swx2 - 2 * mu_post * swx + sw * (mu_post ** 2)
            var_post = (2 * nig.beta0 + np.maximum(sse, 0.0)) / (2 * nig.alpha0 + sw)
            var_post = np.maximum(var_post, 1e-6)

        if gauss:
            # ── Gaussian log-likelihood ────────────────────────
            # log N(vec | mu_post, var_post) = -0.5 * Σ [log(2π σ²) + (x-μ)²/σ²]
            d = len(vec)
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * var_post) +
                (vec - mu_post) ** 2 / var_post
            )
            return float(log_lik / tau_inc)

        # ── Continuous NIG/KL ──────────────────────────────────
        kl = 0.5 * np.sum(
            np.log(var_post / eps_obj) +
            (eps_obj + (vec - mu_post) ** 2) / var_post - 1.0
        )
        return float(-kl / tau_inc)

    def map_color(self, nig: NIGParams,
                  eps_obj: float = 1e-3, tau_inc: float = 1.0,
                  delta: Optional[Dict[str, float]] = None,
                  gauss: bool = False) -> str:
        """Return the MAP color name under the current posterior."""
        if delta is not None:
            return max(COLORS,
                       key=lambda c: delta.get(c, 1.0) + self.color_counts.get(c, 0.0))
        best_color = COLORS[0]
        best_score = -np.inf
        # Use Lab vectors when gauss mode is active
        if gauss:
            from ns_learner.ns_colors import lab_vec as _lv
            color_vecs = {c: _lv(c) for c in COLORS}
        else:
            color_vecs = COLOR_VECS
        for c in COLORS:
            s = self.log_emit_prob(color_vecs[c], nig, eps_obj, tau_inc,
                                   gauss=gauss)
            if s > best_score:
                best_score = s
                best_color = c
        return best_color

    def top_k_emit_candidates(self, nig: NIGParams, k_b: int = 3,
                              eps_obj: float = 1e-3,
                              tau_inc: float = 1.0,
                              delta: Optional[Dict[str, float]] = None,
                              gauss: bool = False
                              ) -> List[Tuple[np.ndarray, float]]:
        """
        Return top-K_b candidate color vectors with log-scores.

        OPTIMIZED: compute mu_post/var_post ONCE, then score all 6
        colors in a single vectorized op (no repeated scalar loops).
        Falls back to loop for delta/gauss modes.
        """
        if gauss or delta is not None:
            # Fallback: loop-based (handles Lab vecs / discrete modes)
            if gauss:
                from ns_learner.ns_colors import lab_vec as _lv
                color_vecs = {c: _lv(c) for c in COLORS}
            else:
                color_vecs = COLOR_VECS
            scored = []
            for c in COLORS:
                vec = color_vecs[c]
                s = self.log_emit_prob(vec, nig, eps_obj, tau_inc, delta=delta, gauss=gauss)
                scored.append((vec, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k_b]

        # ── Fast vectorized path (default: NIG/KL, 6-color discrete palette) ──
        sw = self.emit_stats['sum_w']
        denom = nig.kappa0 + sw
        if denom < 1e-30:
            mu_post = nig.mu0
            var_post = np.ones_like(mu_post)
        else:
            swx = self.emit_stats['sum_wx']
            mu_post = (nig.kappa0 * nig.mu0 + swx) / denom
            swx2 = self.emit_stats['sum_wx2']
            sse = swx2 - 2 * mu_post * swx + sw * (mu_post ** 2)
            var_post = (2 * nig.beta0 + np.maximum(sse, 0.0)) / (2 * nig.alpha0 + sw)
            var_post = np.maximum(var_post, 1e-6)

        # _ALL_COLOR_VECS: (N_COLORS, D) — pre-stacked at module load
        diff = _ALL_COLOR_VECS - mu_post[None, :]          # (N_COLORS, D)
        kl = 0.5 * np.sum(
            np.log(var_post / eps_obj)
            + (eps_obj + diff ** 2) / var_post - 1.0,
            axis=1
        )                                                    # (N_COLORS,)
        scores = -kl / tau_inc                              # (N_COLORS,)
        top_idx = np.argsort(scores)[::-1][:k_b]
        return [(_ALL_COLOR_VECS[i], float(scores[i])) for i in top_idx]

    def role_probs(self, prior_alpha: Dict[str, float]) -> Dict[str, float]:
        """Return posterior mean probabilities for all roles (for debugging)."""
        probs = {}
        total = sum(prior_alpha[r] + self.role_counts[r] for r in ROLES)
        for r in ROLES:
            probs[r] = (prior_alpha[r] + self.role_counts[r]) / max(total, 1e-30)
        return probs

    def map_role(self, prior_alpha: Dict[str, float]) -> str:
        """Return the MAP role."""
        return max(ROLES, key=lambda r: self.log_role_prob(r, prior_alpha))

    # ── Soft Update (M-step) ───────────────────────────────────

    def soft_update(self, weight: float, role: str,
                    vec: Optional[np.ndarray] = None,
                    k: Optional[int] = None):
        """
        Accumulate weighted sufficient statistics from one trace step.
        
        Called during the M-step of soft-EM with the posterior weight
        of the trace that generated this step.
        """
        self.role_counts[role] += weight

        if role == 'EMIT' and vec is not None:
            # Continuous stats (NIG)
            self.emit_stats['sum_w'] += weight
            self.emit_stats['sum_wx'] += weight * vec
            self.emit_stats['sum_wx2'] += weight * (vec ** 2)
            # Discrete stats (Dirichlet)  
            c = vec_to_color(vec)
            self.color_counts[c] = self.color_counts.get(c, 0.0) + weight

        if role == 'REPEAT' and k is not None:
            if k in self.repeat_counts:
                self.repeat_counts[k] += weight

    def reset_counts(self):
        """Zero out all counts before a new E-step."""
        for r in ROLES:
            self.role_counts[r] = 0.0
        for k in REPEAT_RANGE:
            self.repeat_counts[k] = 0.0
        self.emit_stats['sum_w'] = 0.0
        self.emit_stats['sum_wx'][:] = 0.0
        self.emit_stats['sum_wx2'][:] = 0.0
        for c in COLORS:
            self.color_counts[c] = 0.0

    # ── Debug ──────────────────────────────────────────────────

    def __repr__(self):
        total_role = sum(self.role_counts.values())
        top_role = max(self.role_counts, key=self.role_counts.get)
        return (f"NeuroConcept({self.name!r}, "
                f"top_role={top_role}, total_obs={total_role:.1f})")

    def emit_entropy(self, delta: Optional[Dict[str, float]] = None) -> float:
        """Shannon entropy of discrete emission posterior H[P(color|word)]."""
        d = delta or {c: 1.0 for c in COLORS}
        counts = [d.get(c, 1.0) + self.color_counts.get(c, 0.0) for c in COLORS]
        total = sum(counts)
        if total <= 0:
            return np.log(N_COLORS)
        probs = [c / total for c in counts]
        return float(-sum(p * np.log(max(p, 1e-30)) for p in probs))

    def snapshot(self, prior_alpha: Dict[str, float]) -> str:
        """Human-readable summary."""
        rp = self.role_probs(prior_alpha)
        parts = [f"{self.name}:"]
        for r in ROLES:
            if rp[r] > 0.05:
                parts.append(f"  {r}={rp[r]:.2f}")
        if self.emit_stats['sum_w'] > 0:
            sw = self.emit_stats['sum_w']
            mu = self.emit_stats['sum_wx'] / sw
            parts.append(f"  emit_μ={vec_to_color(mu)}")
        return ' '.join(parts)
