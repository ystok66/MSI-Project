"""
attention_model.py — Learner attention weights over target cells.

Implements §10.2 HIGHLIGHT boost:
    w_ℓ^(H) ∝ w_ℓ · exp(ρ_H · 1[ℓ ∈ H])

Provides a uniform baseline and highlight-modified weights.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


class AttentionModel:
    """Learner's attention distribution over target output cells.

    Starts uniform. HIGHLIGHT boosts specified cells.
    Attention weights are used by the semantic scorer for
    weighted mismatch computation.

    Persistent-prior mode:
      Maintains block-level highlight_counts across queries.
      New queries start with w_ℓ ∝ 1 + η_attn · c_ℓ
      instead of pure uniform. This allows HIGHLIGHT to transfer.
    """

    def __init__(self, L: int, rho_H: float = 2.0):
        """
        Args:
            L: number of target output cells
            rho_H: highlight attention boost strength (§10.2)
        """
        self.L = L
        self.rho_H = rho_H
        self._weights = np.ones(L) / L  # uniform prior
        self._highlighted: Tuple[int, ...] = ()

        # Block-level persistent state
        self._highlight_counts: Optional[np.ndarray] = None

    def init_for_query(self, L: int,
                       prior_counts: Optional[np.ndarray] = None,
                       eta_attn: float = 0.3) -> None:
        """Initialize attention for a new query.

        Args:
            L: target output length for this query
            prior_counts: accumulated highlight counts from teaching phase
            eta_attn: strength of persistent prior (0 = uniform)
        """
        self.L = L
        self._highlighted = ()

        if prior_counts is not None and len(prior_counts) >= L:
            # Persistent prior: w_ℓ ∝ 1 + η_attn · c_ℓ
            w = 1.0 + eta_attn * prior_counts[:L]
            self._weights = w / w.sum()
        else:
            # Default: uniform
            self._weights = np.ones(L) / L

    def apply_highlight(self, cells: Tuple[int, ...]) -> np.ndarray:
        """Apply HIGHLIGHT boost to specified cells.

        §10.2: w_ℓ^(H) ∝ w_ℓ · exp(ρ_H · 1[ℓ ∈ H])
        Then re-normalize.

        Returns the new weights.
        """
        if not cells:
            return self._weights.copy()

        self._highlighted = cells
        w = self._weights.copy()
        for ell in cells:
            if 0 <= ell < self.L:
                w[ell] *= np.exp(self.rho_H)

        # Re-normalize
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(self.L) / self.L

        self._weights = w
        return self._weights.copy()

    def record_highlight(self, cells: Tuple[int, ...],
                         L_max: int = 8) -> None:
        """Record highlight event for persistent prior accumulation.

        Args:
            cells: highlighted cell indices
            L_max: maximum target output length (for counts array)
        """
        if self._highlight_counts is None:
            self._highlight_counts = np.zeros(L_max)
        # Grow if needed
        if L_max > len(self._highlight_counts):
            new = np.zeros(L_max)
            new[:len(self._highlight_counts)] = self._highlight_counts
            self._highlight_counts = new

        for c in cells:
            if 0 <= c < len(self._highlight_counts):
                self._highlight_counts[c] += 1

    def get_highlight_counts(self) -> Optional[np.ndarray]:
        """Return accumulated highlight counts (block-level)."""
        return self._highlight_counts

    def reset_highlight_counts(self) -> None:
        """Reset block-level highlight counts (start of new block)."""
        self._highlight_counts = None

    def reset(self) -> None:
        """Reset to uniform (e.g., after refresh)."""
        self._weights = np.ones(self.L) / self.L
        self._highlighted = ()

    @property
    def weights(self) -> np.ndarray:
        """Current attention weights (sum to 1)."""
        return self._weights.copy()

    @property
    def highlighted_cells(self) -> Tuple[int, ...]:
        return self._highlighted

    @property
    def is_highlighted(self) -> bool:
        return len(self._highlighted) > 0

    def effective_coverage(self) -> float:
        """Effective number of attended cells (exp of entropy)."""
        p = self._weights[self._weights > 0]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))

    # ─────────────────────────────────────────────────────────────────────
    # RSA meta-attention persistence (new in RSA mode)
    # ─────────────────────────────────────────────────────────────────────

    def init_meta_prior(self, L: int) -> None:
        """Initialize block-level meta-attention prior (uniform).

        Called once per block at LearnerAgent.init_block().
        Must be reset between blocks to prevent state leakage.

        Args:
            L: uniform-weight dimension (use env.L_max or a generous bound)
        """
        self._meta_prior = np.ones(L) / L   # bar_w_0 = uniform

    def update_meta_prior(
        self,
        cells: Tuple[int, ...],
        rho: float = 0.3,
    ) -> None:
        """Update block-level meta-attention prior from HIGHLIGHT(H).

        bar_w_{t+1} = Normalize((1-ρ)*bar_w_t + ρ*u(H))

        where u(H) is a uniform indicator over highlighted cells:
            u(H)_ℓ = 1/|H|  if ℓ ∈ H  else 0

        Args:
            cells: highlighted cell indices H
            rho: update rate ρ (from RSAConfig.rho_attn)
        """
        if not hasattr(self, '_meta_prior') or self._meta_prior is None:
            self.init_meta_prior(self.L)

        L = len(self._meta_prior)
        u_H = np.zeros(L)
        valid = [c for c in cells if 0 <= c < L]
        if valid:
            for c in valid:
                u_H[c] = 1.0 / len(valid)

        new_meta = (1 - rho) * self._meta_prior + rho * u_H
        s = new_meta.sum()
        if s > 0:
            self._meta_prior = new_meta / s
        # else keep old prior

    def effective_attention(self, gamma: float = 0.3) -> np.ndarray:
        """Blend query-level attention with block-level meta prior.

        w_eff = Normalize((1-γ)*w_query + γ*bar_w)

        Used by RSA path as the attention input to pick utility.
        Legacy path uses self._weights directly.

        Args:
            gamma: meta-prior blend ratio γ (from RSAConfig.gamma_attn)

        Returns:
            Normalized effective attention weights (L,)
        """
        if not hasattr(self, '_meta_prior') or self._meta_prior is None:
            return self._weights.copy()

        L = self.L
        meta = self._meta_prior

        # Align lengths (meta may be longer than L for current query)
        if len(meta) >= L:
            meta_slice = meta[:L]
        else:
            # Pad with zeros (shouldn't happen in practice)
            meta_slice = np.zeros(L)
            meta_slice[:len(meta)] = meta

        # Normalize meta_slice
        s = meta_slice.sum()
        if s > 0:
            meta_slice = meta_slice / s
        else:
            meta_slice = np.ones(L) / L

        blended = (1 - gamma) * self._weights + gamma * meta_slice
        s2 = blended.sum()
        if s2 > 0:
            return blended / s2
        return np.ones(L) / L

    def apply_rsa_highlight(
        self,
        cells: Tuple[int, ...],
        rho: float = 0.3,
        gamma: float = 0.3,
    ) -> np.ndarray:
        """Apply HIGHLIGHT in RSA mode (combining legacy boost + meta update).

        Steps:
          1. Apply legacy HIGHLIGHT boost (exp(rho_H) on cells) to _weights
          2. Update meta prior from this HIGHLIGHT event
          3. Recompute _weights as effective_attention(gamma)

        This ensures:
          - Immediate effect: attention shifts to highlighted cells (like legacy)
          - Persistent effect: meta prior accumulates across queries

        Args:
            cells: highlighted cell indices H
            rho: meta-attention update rate (RSAConfig.rho_attn)
            gamma: meta-prior blend ratio (RSAConfig.gamma_attn)

        Returns:
            New effective attention weights
        """
        # Step 1: legacy boost on current query weights
        self.apply_highlight(cells)  # modifies self._weights in place

        # Step 2: update meta prior
        self.update_meta_prior(cells, rho=rho)

        # Step 3: blend with meta prior
        eff = self.effective_attention(gamma=gamma)
        self._weights = eff
        return self._weights.copy()

    @property
    def meta_prior(self) -> Optional[np.ndarray]:
        """Current block-level meta-attention prior (or None if not init)."""
        return getattr(self, '_meta_prior', None)
