"""
shadow_snapshot.py — Shadow learner state snapshot.

Captures the complete or compressed state of the real learner for
tutor-side counterfactual simulation. The tutor NEVER shares the
real learner's object references.

Two fidelity levels:
  - exact: full deep copy of grammar library + risk state
  - compressed: role_counts + emit_stats only (skip repeat, color_counts)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np


@dataclass
class ConceptSnapshot:
    """Snapshot of one NeuroConcept's sufficient statistics."""
    name: str
    role_counts: Dict[str, float] = field(default_factory=dict)
    repeat_counts: Dict[int, float] = field(default_factory=dict)
    emit_stats: Dict[str, object] = field(default_factory=dict)  # sum_w, sum_wx, sum_wx2
    color_counts: Dict[str, float] = field(default_factory=dict)

    def clone(self) -> 'ConceptSnapshot':
        return ConceptSnapshot(
            name=self.name,
            role_counts=dict(self.role_counts),
            repeat_counts=dict(self.repeat_counts),
            emit_stats={
                'sum_w': float(self.emit_stats.get('sum_w', 0.0)),
                'sum_wx': np.array(self.emit_stats['sum_wx']).copy()
                    if 'sum_wx' in self.emit_stats else np.zeros(0),
                'sum_wx2': np.array(self.emit_stats['sum_wx2']).copy()
                    if 'sum_wx2' in self.emit_stats else np.zeros(0),
            },
            color_counts=dict(self.color_counts),
        )


@dataclass
class RiskSnapshot:
    """Snapshot of DangerTypeBelief state."""
    n_danger_types: int
    n_types: int
    danger_dim: int
    obs_sigma: float
    type_prior: np.ndarray         # (n_types,)
    proto_mu: np.ndarray           # (n_types, danger_dim)
    proto_var: np.ndarray          # (n_types, danger_dim)
    _counts: np.ndarray            # (n_types,)
    _sum_x: np.ndarray             # (n_types, danger_dim)
    _sum_x2: np.ndarray            # (n_types, danger_dim)

    def clone(self) -> 'RiskSnapshot':
        return RiskSnapshot(
            n_danger_types=self.n_danger_types,
            n_types=self.n_types,
            danger_dim=self.danger_dim,
            obs_sigma=self.obs_sigma,
            type_prior=self.type_prior.copy(),
            proto_mu=self.proto_mu.copy(),
            proto_var=self.proto_var.copy(),
            _counts=self._counts.copy(),
            _sum_x=self._sum_x.copy(),
            _sum_x2=self._sum_x2.copy(),
        )


@dataclass
class PolicySnapshot:
    """Snapshot of policy-relevant learner parameters."""
    alpha_fill: float = 1.0
    alpha_risk: float = 0.5
    alpha_waste: float = 0.3
    confirm_fill_threshold: float = 1.0
    beta_policy: float = 4.0
    epsilon_policy: float = 0.05
    enable_courage: bool = False
    n_retry_courage: int = 5

    def clone(self) -> 'PolicySnapshot':
        return PolicySnapshot(
            alpha_fill=self.alpha_fill,
            alpha_risk=self.alpha_risk,
            alpha_waste=self.alpha_waste,
            confirm_fill_threshold=self.confirm_fill_threshold,
            beta_policy=self.beta_policy,
            epsilon_policy=self.epsilon_policy,
            enable_courage=self.enable_courage,
            n_retry_courage=self.n_retry_courage,
        )


@dataclass
class ShadowLearnerSnapshot:
    """Complete shadow learner state.

    Attributes:
        grammar: Dict[word -> ConceptSnapshot]
        risk: RiskSnapshot
        policy: PolicySnapshot
        fidelity: 'exact' or 'compressed'
        support_words: original support words (for beam search context)
        support_outputs: original support outputs
    """
    grammar: Dict[str, ConceptSnapshot] = field(default_factory=dict)
    risk: Optional[RiskSnapshot] = None
    policy: Optional[PolicySnapshot] = None
    fidelity: str = 'exact'
    support_words: List[List[str]] = field(default_factory=list)
    support_outputs: List[List[str]] = field(default_factory=list)
    # CLS config subset needed for beam search reconstruction
    cls_mode: str = 'ast'
    use_hpc: bool = False
    n_em: int = 3

    def clone(self) -> 'ShadowLearnerSnapshot':
        """Deep clone for counterfactual simulation."""
        return ShadowLearnerSnapshot(
            grammar={w: c.clone() for w, c in self.grammar.items()},
            risk=self.risk.clone() if self.risk else None,
            policy=self.policy.clone() if self.policy else None,
            fidelity=self.fidelity,
            support_words=[list(ws) for ws in self.support_words],
            support_outputs=[list(os) for os in self.support_outputs],
            cls_mode=self.cls_mode,
            use_hpc=self.use_hpc,
            n_em=self.n_em,
        )

    def grammar_hash(self) -> str:
        """Quick hash for divergence tracking."""
        import hashlib
        h = hashlib.md5()
        for w in sorted(self.grammar.keys()):
            c = self.grammar[w]
            h.update(w.encode())
            for r in sorted(c.role_counts.keys()):
                h.update(f"{r}:{c.role_counts[r]:.4f}".encode())
        return h.hexdigest()[:12]

    def risk_hash(self) -> str:
        """Quick hash for risk state."""
        import hashlib
        if self.risk is None:
            return "no_risk"
        h = hashlib.md5()
        h.update(self.risk.proto_mu.tobytes())
        h.update(self.risk.type_prior.tobytes())
        return h.hexdigest()[:12]
