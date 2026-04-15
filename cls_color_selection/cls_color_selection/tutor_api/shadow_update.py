"""
shadow_update.py — Simulate learner updates on shadow state.

All updates operate on ShadowLearnerSnapshot or its clones,
never touching the real learner's state.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .shadow_snapshot import (
    ShadowLearnerSnapshot, ConceptSnapshot, RiskSnapshot,
)
from .shadow_clone import write_shadow_to_real_risk
from ..interfaces import CandidateBall


# ── Risk updates on shadow ──────────────────────────────────────

def shadow_warning_update(
    snapshot: ShadowLearnerSnapshot,
    selected_balls: List[CandidateBall],
) -> ShadowLearnerSnapshot:
    """Simulate warning_set_bayes_update on shadow risk state.

    Modifies snapshot.risk in place (caller should clone first if needed).
    """
    if snapshot.risk is None or not selected_balls:
        return snapshot

    # Reconstruct working risk model from shadow
    risk = write_shadow_to_real_risk(snapshot)

    # Apply the real warning update
    from ..learner.warning_update import warning_set_bayes_update
    warning_set_bayes_update(risk, selected_balls)

    # Write back to snapshot
    snapshot.risk.type_prior = risk.type_prior.copy()
    snapshot.risk.proto_mu = risk.proto_mu.copy()
    snapshot.risk.proto_var = risk.proto_var.copy()
    snapshot.risk._counts = risk._counts.copy()
    snapshot.risk._sum_x = risk._sum_x.copy()
    snapshot.risk._sum_x2 = risk._sum_x2.copy()

    return snapshot


def shadow_courage_update(
    snapshot: ShadowLearnerSnapshot,
    candidate_pool: List[CandidateBall],
    needed_colors: Set[str],
) -> ShadowLearnerSnapshot:
    """Simulate courage_literal_update on shadow risk state."""
    if snapshot.risk is None or not candidate_pool:
        return snapshot

    risk = write_shadow_to_real_risk(snapshot)

    from ..learner.courage_update import courage_literal_update
    courage_literal_update(risk, candidate_pool, needed_colors)

    snapshot.risk.proto_mu = risk.proto_mu.copy()
    snapshot.risk.proto_var = risk.proto_var.copy()
    snapshot.risk._counts = risk._counts.copy()
    snapshot.risk._sum_x = risk._sum_x.copy()
    snapshot.risk._sum_x2 = risk._sum_x2.copy()

    return snapshot


def shadow_safe_observation_update(
    snapshot: ShadowLearnerSnapshot,
    x: np.ndarray,
) -> ShadowLearnerSnapshot:
    """Simulate update_from_safe_observation on shadow risk."""
    if snapshot.risk is None:
        return snapshot

    risk = write_shadow_to_real_risk(snapshot)
    risk.update_from_safe_observation(x)

    snapshot.risk.proto_mu = risk.proto_mu.copy()
    snapshot.risk.proto_var = risk.proto_var.copy()
    snapshot.risk._counts = risk._counts.copy()
    snapshot.risk._sum_x = risk._sum_x.copy()
    snapshot.risk._sum_x2 = risk._sum_x2.copy()

    return snapshot


def shadow_death_update(
    snapshot: ShadowLearnerSnapshot,
    x: np.ndarray,
) -> ShadowLearnerSnapshot:
    """Simulate update_from_death on shadow risk."""
    if snapshot.risk is None:
        return snapshot

    risk = write_shadow_to_real_risk(snapshot)
    risk.update_from_death(x)

    snapshot.risk.proto_mu = risk.proto_mu.copy()
    snapshot.risk.proto_var = risk.proto_var.copy()
    snapshot.risk._counts = risk._counts.copy()
    snapshot.risk._sum_x = risk._sum_x.copy()
    snapshot.risk._sum_x2 = risk._sum_x2.copy()

    return snapshot


# ── Grammar updates on shadow ───────────────────────────────────

def shadow_feedback_update(
    snapshot: ShadowLearnerSnapshot,
    words: List[str],
    Y_hat: List[str],
    feedback: dict,
    cfg_learner=None,
) -> ShadowLearnerSnapshot:
    """Simulate feedback reweight + differential M-step on shadow grammar.

    Steps:
      1. Reconstruct a temporary CLSAgent with shadow grammar
      2. Run beam search to get candidates
      3. Reweight with feedback likelihood
      4. Differential M-step on the temporary library
      5. Copy updated stats back to snapshot

    This is the most expensive shadow operation — involves beam search.
    """
    if not snapshot.grammar:
        return snapshot

    # Reconstruct temporary CLS agent with shadow state
    agent, predictor = _reconstruct_shadow_cls(snapshot)
    if agent is None:
        return snapshot

    # Get beam posterior
    beam = predictor.beam_posterior(words)
    if not beam:
        return snapshot

    # Feedback reweight
    from ..learner.feedback_update import FeedbackUpdater
    from ..config import LearnerConfig
    lcfg = cfg_learner or LearnerConfig()
    updater = FeedbackUpdater(lcfg)

    q_old, q_new = updater.reweight_beam_posterior(beam, Y_hat, feedback)

    # Differential M-step on the temporary library
    library = predictor.get_library()
    updater.differential_m_step(library, beam, q_old, q_new)

    # Copy updated stats back to snapshot
    for word, concept in library.items():
        if word in snapshot.grammar:
            snap = snapshot.grammar[word]
            snap.role_counts = dict(concept.role_counts)
            snap.repeat_counts = dict(concept.repeat_counts)
            snap.emit_stats = {
                'sum_w': float(concept.emit_stats.get('sum_w', 0.0)),
                'sum_wx': np.array(concept.emit_stats['sum_wx']).copy(),
                'sum_wx2': np.array(concept.emit_stats['sum_wx2']).copy(),
            }
            snap.color_counts = dict(concept.color_counts)

    return snapshot


# ── Shadow CLS reconstruction ──────────────────────────────────

def _reconstruct_shadow_cls(
    snapshot: ShadowLearnerSnapshot,
) -> tuple:
    """Build a temporary CLSAgent populated with shadow grammar state.

    Returns:
        (CLSAgent, CLSSequencePredictor) or (None, None) on failure
    """
    try:
        from ..learner.cls_wrapper import _ensure_basic_on_path, CLSSequencePredictor
        _ensure_basic_on_path()
        from cls_learner.agent import CLSAgent
        from cls_learner.config import CLSConfig
        from cls_learner.interfaces import Example as CLSExample

        # Build config matching snapshot
        cls_cfg = CLSConfig(
            mode=snapshot.cls_mode,
            use_hpc=snapshot.use_hpc,
            n_em=snapshot.n_em,
        )
        agent = CLSAgent(cls_cfg)
        agent.reset_episode()

        # Study support to establish vocabulary and base structure
        cls_examples = [
            CLSExample(words=ws, output=os)
            for ws, os in zip(snapshot.support_words, snapshot.support_outputs)
        ]
        agent.study(cls_examples, verbose=False)

        # Now overwrite library with shadow state
        from .shadow_clone import write_shadow_grammar_to_library
        write_shadow_grammar_to_library(snapshot, agent.cortex.library)

        # Wrap in predictor
        from ..config import LearnerConfig
        lcfg = LearnerConfig(
            cls_mode=snapshot.cls_mode,
            use_hpc=snapshot.use_hpc,
            n_em=snapshot.n_em,
        )
        predictor = CLSSequencePredictor(lcfg)
        predictor._agent = agent
        predictor._studied = True

        return agent, predictor

    except Exception:
        return None, None


def shadow_predict_target(
    snapshot: ShadowLearnerSnapshot,
    words: List[str],
) -> Optional[List[str]]:
    """Predict Y* using shadow grammar state."""
    agent, predictor = _reconstruct_shadow_cls(snapshot)
    if predictor is None:
        return None
    try:
        return predictor.predict_target(words)
    except Exception:
        return None


def shadow_beam_entropy(
    snapshot: ShadowLearnerSnapshot,
    words: List[str],
) -> float:
    """Compute beam entropy H(beam) using shadow grammar."""
    agent, predictor = _reconstruct_shadow_cls(snapshot)
    if predictor is None:
        return 0.0
    try:
        beam = predictor.beam_posterior(words)
        if not beam:
            return 0.0
        scores = np.array([b[0] for b in beam])
        from scipy.special import logsumexp
        log_q = scores - logsumexp(scores)
        q = np.exp(log_q)
        h = -np.sum(q * log_q)
        return float(h)
    except Exception:
        return 0.0
