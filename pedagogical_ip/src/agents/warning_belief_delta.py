"""
WarningBeliefDelta — canonical output of the unified warning semantic layer.

Phase 1A: This dataclass is the SINGLE output type that all warning variants
produce. It decouples the semantic computation (RSA) from the downstream
adapters (planner penalty, pseudo-label injection, feature belief update).

Semantic source: rsa_warning_channel.compute_warning_belief_delta()
Consumers:
  - Planner adapter: planner_cell_penalties → warned_cell_extra
  - Pseudo-label adapter (legacy compat): pseudo_label_pkg → risk_head.update_from_label
  - Diagnostics: always computed, never affects behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np


@dataclass
class PseudoLabelEntry:
    """One pseudo-label injection entry for legacy compatibility."""
    z_proto: np.ndarray    # prototype feature vector (4D)
    y_label: float         # risk label (0.0–1.0)
    weight: float          # effective injection weight


@dataclass
class WarningBeliefDelta:
    """Canonical output of the unified warning semantic layer.

    All warning variants produce this. The variant determines which
    downstream adapters consume it:

      legacy_bias:      diagnostics only (shadow RSA), original behavior preserved
      rsa_obs_l0/s1:    planner_cell_penalties consumed, pseudo-labels NOT used
      rsa_obs_s1_trust: same as s1 with trust-gated evidence
      rsa_plus_phase10: planner + pseudo-labels (hybrid ablation)
    """
    # ── Semantic layer (RSA-computed) ──
    utterance: str                                 # RSA utterance name
    variant: str                                   # warning variant used
    context: dict                                  # branch availability etc.
    prior_belief: np.ndarray                       # b⁻(r), shape (4,)
    posterior_belief: np.ndarray                    # b⁺(r), shape (4,)

    # ── Risk deltas ──
    delta_rho_inc: float                           # E[ρ|b⁺] - E[ρ|b⁻]
    delta_rho_uniform: float                       # E[ρ|b⁺] - E[ρ|uniform]

    # ── Planner adapter output ──
    planner_cell_penalties: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # ── Optional pseudo-label adapter ──
    pseudo_label_pkg: List[PseudoLabelEntry] = field(default_factory=list)

    # ── Diagnostics (always computed, never affects behavior) ──
    diagnostics: dict = field(default_factory=dict)
