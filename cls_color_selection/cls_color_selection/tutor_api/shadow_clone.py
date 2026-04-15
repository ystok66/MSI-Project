"""
shadow_clone.py — Clone real learner state into ShadowLearnerSnapshot.

Two modes:
  - exact: deep copy all grammar + risk + policy state
  - compressed: only role_counts + emit_stats (skip repeat, color_counts)
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from .shadow_snapshot import (
    ShadowLearnerSnapshot, ConceptSnapshot, RiskSnapshot, PolicySnapshot,
)
from ..learner.cls_wrapper import CLSSequencePredictor
from ..learner.risk_belief import DangerTypeBelief
from ..config import LearnerConfig


def clone_grammar_exact(predictor: CLSSequencePredictor) -> dict:
    """Deep copy full grammar library into ConceptSnapshots."""
    library = predictor.get_library()
    snapshots = {}
    for word, concept in library.items():
        snapshots[word] = ConceptSnapshot(
            name=concept.name,
            role_counts=dict(concept.role_counts),
            repeat_counts=dict(concept.repeat_counts),
            emit_stats={
                'sum_w': float(concept.emit_stats.get('sum_w', 0.0)),
                'sum_wx': np.array(concept.emit_stats['sum_wx']).copy(),
                'sum_wx2': np.array(concept.emit_stats['sum_wx2']).copy(),
            },
            color_counts=dict(concept.color_counts),
        )
    return snapshots


def clone_grammar_compressed(predictor: CLSSequencePredictor) -> dict:
    """Copy only role_counts + emit_stats (skip repeat, color_counts)."""
    library = predictor.get_library()
    snapshots = {}
    for word, concept in library.items():
        snapshots[word] = ConceptSnapshot(
            name=concept.name,
            role_counts=dict(concept.role_counts),
            repeat_counts={},  # skip
            emit_stats={
                'sum_w': float(concept.emit_stats.get('sum_w', 0.0)),
                'sum_wx': np.array(concept.emit_stats['sum_wx']).copy(),
                'sum_wx2': np.array(concept.emit_stats['sum_wx2']).copy(),
            },
            color_counts={},  # skip
        )
    return snapshots


def clone_risk(risk_belief: DangerTypeBelief) -> RiskSnapshot:
    """Deep copy risk belief into RiskSnapshot."""
    return RiskSnapshot(
        n_danger_types=risk_belief.n_danger_types,
        n_types=risk_belief.n_types,
        danger_dim=risk_belief.danger_dim,
        obs_sigma=risk_belief.obs_sigma,
        type_prior=risk_belief.type_prior.copy(),
        proto_mu=risk_belief.proto_mu.copy(),
        proto_var=risk_belief.proto_var.copy(),
        _counts=risk_belief._counts.copy(),
        _sum_x=risk_belief._sum_x.copy(),
        _sum_x2=risk_belief._sum_x2.copy(),
    )


def clone_policy(cfg: LearnerConfig) -> PolicySnapshot:
    """Extract policy-relevant parameters."""
    return PolicySnapshot(
        alpha_fill=cfg.alpha_fill,
        alpha_risk=cfg.alpha_risk,
        alpha_waste=cfg.alpha_waste,
        confirm_fill_threshold=cfg.confirm_fill_threshold,
        beta_policy=cfg.beta_policy,
        epsilon_policy=cfg.epsilon_policy,
        enable_courage=cfg.enable_courage,
        n_retry_courage=cfg.n_retry_courage,
    )


def create_shadow_snapshot(
    predictor: CLSSequencePredictor,
    risk_belief: DangerTypeBelief,
    cfg: LearnerConfig,
    support_examples: list,
    fidelity: str = 'exact',
) -> ShadowLearnerSnapshot:
    """Create a full shadow snapshot from the real learner.

    Args:
        predictor: CLS grammar learner
        risk_belief: current risk belief state
        cfg: learner config
        support_examples: list of Example objects given to learner
        fidelity: 'exact' or 'compressed'

    Returns:
        ShadowLearnerSnapshot independent of the real learner
    """
    if fidelity == 'compressed':
        grammar = clone_grammar_compressed(predictor)
    else:
        grammar = clone_grammar_exact(predictor)

    risk = clone_risk(risk_belief)
    policy = clone_policy(cfg)

    return ShadowLearnerSnapshot(
        grammar=grammar,
        risk=risk,
        policy=policy,
        fidelity=fidelity,
        support_words=[ex.words for ex in support_examples],
        support_outputs=[ex.output for ex in support_examples],
        cls_mode=cfg.cls_mode,
        use_hpc=cfg.use_hpc,
        n_em=cfg.n_em,
    )


def write_shadow_to_real_risk(
    snapshot: ShadowLearnerSnapshot,
) -> DangerTypeBelief:
    """Reconstruct a DangerTypeBelief from a shadow risk snapshot.

    Used for simulation: create a working risk model from shadow state.
    """
    rs = snapshot.risk
    if rs is None:
        return DangerTypeBelief(n_danger_types=3, danger_dim=10)

    belief = DangerTypeBelief(
        n_danger_types=rs.n_danger_types,
        danger_dim=rs.danger_dim,
        obs_sigma=rs.obs_sigma,
    )
    belief.type_prior = rs.type_prior.copy()
    belief.proto_mu = rs.proto_mu.copy()
    belief.proto_var = rs.proto_var.copy()
    belief._counts = rs._counts.copy()
    belief._sum_x = rs._sum_x.copy()
    belief._sum_x2 = rs._sum_x2.copy()
    return belief


def write_shadow_grammar_to_library(
    snapshot: ShadowLearnerSnapshot,
    library: dict,
):
    """Write shadow grammar state into a real NeuroConcept library.

    Used for reconstruction: populate a CLSAgent's library from shadow state.
    """
    for word, csnapshot in snapshot.grammar.items():
        if word not in library:
            continue
        concept = library[word]
        concept.role_counts = dict(csnapshot.role_counts)
        if csnapshot.repeat_counts:
            concept.repeat_counts = dict(csnapshot.repeat_counts)
        if csnapshot.emit_stats:
            concept.emit_stats['sum_w'] = float(csnapshot.emit_stats.get('sum_w', 0.0))
            swx = csnapshot.emit_stats.get('sum_wx')
            if swx is not None:
                concept.emit_stats['sum_wx'] = np.array(swx).copy()
            swx2 = csnapshot.emit_stats.get('sum_wx2')
            if swx2 is not None:
                concept.emit_stats['sum_wx2'] = np.array(swx2).copy()
        if csnapshot.color_counts:
            concept.color_counts = dict(csnapshot.color_counts)
