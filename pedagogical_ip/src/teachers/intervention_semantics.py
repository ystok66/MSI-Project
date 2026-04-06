"""Intervention Semantics — Formal update functions for each intervention type.

Formalizes what each intervention type does to agent belief / world state:

  WARN      = belief evidence: updates agent belief, NOT world topology
  UNLOCK    = affordance reveal: changes world topology, NOT risk mean
  ITEM_DROP = traversal mitigation: reduces traversal cost, NOT belief

These are small pure functions used by the option controller for
counterfactual simulation. They do NOT modify existing modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════
# WARN: Belief Evidence
# ═══════════════════════════════════════════════════════════════

@dataclass
class WarnEffect:
    """Result of applying a WARN intervention."""
    belief_delta: np.ndarray     # change in agent's risk belief mean
    uncertainty_reduction: float  # fraction reduction in belief variance
    world_changed: bool = False   # WARN never changes world (invariant)


class WarnSemantics:
    """WARN = belief evidence. Does NOT change world topology.

    μ_i^+ = μ_i + α_warn · v_warn
    Σ_i^+ = (1 - β_warn) · Σ_i

    where v_warn is aligned with risk-head direction.
    """

    def __init__(self, alpha_warn: float = 0.3, beta_warn: float = 0.2):
        self.alpha_warn = alpha_warn
        self.beta_warn = beta_warn

    def apply(self, belief_mean: np.ndarray, belief_var: np.ndarray,
              warn_direction: np.ndarray) -> WarnEffect:
        """Apply WARN to agent's feature belief.

        Args:
            belief_mean: (d,) current belief mean for target cell
            belief_var: (d,) current belief variance
            warn_direction: (d,) direction vector toward risk awareness
        Returns:
            WarnEffect with belief_delta and uncertainty_reduction
        """
        # Normalize direction
        norm = np.linalg.norm(warn_direction)
        if norm > 1e-8:
            v = warn_direction / norm
        else:
            v = np.zeros_like(warn_direction)

        delta = self.alpha_warn * v
        unc_reduction = self.beta_warn

        return WarnEffect(
            belief_delta=delta,
            uncertainty_reduction=unc_reduction,
            world_changed=False,  # invariant
        )

    def predicted_belief_after_warn(self, belief_mean: np.ndarray,
                                     belief_var: np.ndarray,
                                     warn_direction: np.ndarray):
        """Return predicted (mean, var) after WARN."""
        effect = self.apply(belief_mean, belief_var, warn_direction)
        new_mean = belief_mean + effect.belief_delta
        new_var = belief_var * (1.0 - effect.uncertainty_reduction)
        return new_mean, new_var


# ═══════════════════════════════════════════════════════════════
# UNLOCK: Affordance Reveal
# ═══════════════════════════════════════════════════════════════

@dataclass
class UnlockEffect:
    """Result of applying an UNLOCK intervention."""
    cells_unlocked: list       # list of (r, c) cells made passable
    topology_changed: bool = True
    risk_mean_changed: bool = False  # invariant: UNLOCK doesn't change risk


class UnlockSemantics:
    """UNLOCK = affordance/uncertainty semantics. Changes topology, NOT risk mean.

    s_{t+1}^world = Unlock(s_t^world)
    b_{t+1}^{A,env} = AffordanceReveal(b_t^{A,env}, s_{t+1}^world)

    Opens gated cells, making them passable. Does NOT modify risk values.
    """

    def __init__(self, uncertainty_reduction: float = 0.3):
        self.uncertainty_reduction = uncertainty_reduction

    def apply(self, passable: np.ndarray,
              locked_cells: list) -> UnlockEffect:
        """Apply UNLOCK to world state.

        Args:
            passable: (H, W) bool array
            locked_cells: list of (r, c) to unlock
        Returns:
            UnlockEffect
        """
        unlocked = []
        for r, c in locked_cells:
            if not passable[r, c]:
                unlocked.append((r, c))

        return UnlockEffect(
            cells_unlocked=unlocked,
            topology_changed=len(unlocked) > 0,
            risk_mean_changed=False,  # invariant
        )

    def apply_to_passable(self, passable: np.ndarray,
                          locked_cells: list) -> np.ndarray:
        """Return new passable array with unlocked cells."""
        new_passable = passable.copy()
        for r, c in locked_cells:
            new_passable[r, c] = True
        return new_passable

    def affordance_belief_update(self, belief_var: np.ndarray,
                                  unlocked_cells: list) -> np.ndarray:
        """Reduce uncertainty on newly accessible cells."""
        new_var = belief_var.copy()
        for r, c in unlocked_cells:
            new_var[r, c] *= (1.0 - self.uncertainty_reduction)
        return new_var


# ═══════════════════════════════════════════════════════════════
# ITEM_DROP: Traversal Mitigation
# ═══════════════════════════════════════════════════════════════

@dataclass
class ItemDropEffect:
    """Result of applying an ITEM_DROP (shield) intervention."""
    risk_reduction: float        # fractional reduction in traversal risk
    belief_changed: bool = False  # invariant: ITEM_DROP doesn't change belief
    world_topology_changed: bool = False  # invariant


class ItemDropSemantics:
    """ITEM_DROP = traversal mitigation. Does NOT change belief or topology.

    TraversalCost^shield(i) = λ_r · (1 - γ_shield) · φ(r̂_i)
    where φ(r) = -ln(1 - r)

    Shield halves traversal risk cost but doesn't teach the agent
    about the environment.
    """

    def __init__(self, gamma_shield: float = 0.5):
        self.gamma_shield = gamma_shield

    def apply(self, risk_value: float) -> ItemDropEffect:
        """Compute shield effect on a cell's traversal risk."""
        return ItemDropEffect(
            risk_reduction=self.gamma_shield,
            belief_changed=False,   # invariant
            world_topology_changed=False,  # invariant
        )

    def shielded_risk_cost(self, risk_value: float,
                            lambda_r: float = 3.0) -> float:
        """Compute shielded traversal cost.

        φ(r) = -ln(1 - r)
        cost = λ_r · (1 - γ_shield) · φ(r)
        """
        phi = -np.log(max(1.0 - risk_value, 1e-6))
        return lambda_r * (1.0 - self.gamma_shield) * phi

    def unshielded_risk_cost(self, risk_value: float,
                              lambda_r: float = 3.0) -> float:
        """Compute unshielded traversal cost for comparison."""
        phi = -np.log(max(1.0 - risk_value, 1e-6))
        return lambda_r * phi

    def cost_reduction_ratio(self, risk_value: float) -> float:
        """Fraction of cost saved by shield."""
        if risk_value < 1e-6:
            return 0.0
        shielded = self.shielded_risk_cost(risk_value)
        unshielded = self.unshielded_risk_cost(risk_value)
        if unshielded < 1e-6:
            return 0.0
        return 1.0 - (shielded / unshielded)


# ═══════════════════════════════════════════════════════════════
# Convenience: all semantics in one place
# ═══════════════════════════════════════════════════════════════

@dataclass
class InterventionSemantics:
    """Bundle of all intervention semantic modules."""
    warn: WarnSemantics = None
    unlock: UnlockSemantics = None
    item_drop: ItemDropSemantics = None

    def __post_init__(self):
        if self.warn is None:
            self.warn = WarnSemantics()
        if self.unlock is None:
            self.unlock = UnlockSemantics()
        if self.item_drop is None:
            self.item_drop = ItemDropSemantics()
