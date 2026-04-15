"""
joint_debug.py — Divergence + counterfactual diagnostics for shadow tutor.

Tracks:
  - D_gram: grammar divergence between shadow and real learner
  - D_risk: risk belief divergence
  - Counterfactual prediction error
  - Per-step logs with all Q(a) decompositions
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .shadow_snapshot import ShadowLearnerSnapshot
from .shadow_clone import create_shadow_snapshot, write_shadow_to_real_risk
from .shadow_update import shadow_predict_target
from ..interfaces import CandidateBall, Example
from ..learner.cls_wrapper import CLSSequencePredictor
from ..learner.risk_belief import DangerTypeBelief
from ..config import LearnerConfig


@dataclass
class DivergenceRecord:
    """One-step divergence measurement."""
    step: int = 0
    query_id: int = 0

    # Grammar divergence
    top1_agreement: bool = True     # shadow Y* == real Y*
    beam_entropy_gap: float = 0.0   # |H_shadow - H_real|
    probe_acc_gap: float = 0.0      # |acc_shadow - acc_real| on probes

    # Risk divergence
    risk_l1: float = 0.0            # mean |P_real(z|x) - P_shadow(z|x)|_1

    # State hashes
    real_grammar_hash: str = ""
    shadow_grammar_hash: str = ""
    real_risk_hash: str = ""
    shadow_risk_hash: str = ""

    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'query_id': self.query_id,
            'top1_agreement': self.top1_agreement,
            'beam_entropy_gap': self.beam_entropy_gap,
            'probe_acc_gap': self.probe_acc_gap,
            'risk_l1': self.risk_l1,
            'real_grammar_hash': self.real_grammar_hash,
            'shadow_grammar_hash': self.shadow_grammar_hash,
        }


@dataclass
class CounterfactualRecord:
    """Record of tutor's predicted vs realized return."""
    step: int = 0
    query_id: int = 0
    action_chosen: str = "WAIT"
    q_predicted: float = 0.0        # shadow-estimated Q(a*)
    return_realized: float = 0.0    # actual outcome score
    error: float = 0.0              # q_predicted - return_realized

    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'query_id': self.query_id,
            'action': self.action_chosen,
            'q_predicted': self.q_predicted,
            'return_realized': self.return_realized,
            'error': self.error,
        }


@dataclass
class JointDebugLog:
    """Accumulates divergence and counterfactual records across an episode."""
    divergences: List[DivergenceRecord] = field(default_factory=list)
    counterfactuals: List[CounterfactualRecord] = field(default_factory=list)

    def add_divergence(self, rec: DivergenceRecord):
        self.divergences.append(rec)

    def add_counterfactual(self, rec: CounterfactualRecord):
        self.counterfactuals.append(rec)

    def summary(self) -> Dict:
        """Aggregate summary statistics."""
        n_div = len(self.divergences)
        n_cf = len(self.counterfactuals)

        top1_agree = np.mean([d.top1_agreement for d in self.divergences]) if n_div else 0.0
        mean_entropy_gap = np.mean([d.beam_entropy_gap for d in self.divergences]) if n_div else 0.0
        mean_risk_l1 = np.mean([d.risk_l1 for d in self.divergences]) if n_div else 0.0

        mean_cf_error = np.mean([c.error for c in self.counterfactuals]) if n_cf else 0.0
        abs_cf_error = np.mean([abs(c.error) for c in self.counterfactuals]) if n_cf else 0.0

        return {
            'n_divergence_records': n_div,
            'n_counterfactual_records': n_cf,
            'D_gram_top1_agreement': float(top1_agree),
            'D_gram_beam_entropy_gap': float(mean_entropy_gap),
            'D_risk_l1': float(mean_risk_l1),
            'CF_mean_error': float(mean_cf_error),
            'CF_abs_error': float(abs_cf_error),
        }


# ── Divergence measurement ─────────────────────────────────────

def measure_grammar_divergence(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
    probe_words: List[List[str]],
    probe_gold: Optional[List[List[str]]] = None,
) -> Tuple[bool, float, float]:
    """Measure grammar divergence between shadow and real learner.

    Returns:
        (top1_agreement, beam_entropy_gap, probe_accuracy_gap)
    """
    if not probe_words:
        return True, 0.0, 0.0

    # Top-1 agreement
    agreements = []
    entropy_gaps = []
    real_correct = 0
    shadow_correct = 0

    for i, words in enumerate(probe_words):
        real_pred = predictor.predict_target(words)
        shadow_pred = shadow_predict_target(shadow, words)

        if shadow_pred is not None:
            agreements.append(real_pred == shadow_pred)

        # Beam entropy comparison
        try:
            from .shadow_update import shadow_beam_entropy
            h_shadow = shadow_beam_entropy(shadow, words)
            real_beam = predictor.beam_posterior(words)
            if real_beam:
                from scipy.special import logsumexp
                scores = np.array([b[0] for b in real_beam])
                log_q = scores - logsumexp(scores)
                q = np.exp(log_q)
                h_real = float(-np.sum(q * log_q))
            else:
                h_real = 0.0
            entropy_gaps.append(abs(h_shadow - h_real))
        except Exception:
            pass

        # Probe accuracy
        if probe_gold and i < len(probe_gold):
            if real_pred == probe_gold[i]:
                real_correct += 1
            if shadow_pred is not None and shadow_pred == probe_gold[i]:
                shadow_correct += 1

    top1_agree = all(agreements) if agreements else True
    mean_entropy_gap = float(np.mean(entropy_gaps)) if entropy_gaps else 0.0
    n_probes = len(probe_words)
    acc_gap = abs(real_correct - shadow_correct) / max(n_probes, 1)

    return top1_agree, mean_entropy_gap, acc_gap


def measure_risk_divergence(
    shadow: ShadowLearnerSnapshot,
    risk_belief: DangerTypeBelief,
    test_vecs: List[np.ndarray],
) -> float:
    """Measure risk divergence: mean L1 between real and shadow posteriors.

    Args:
        shadow: shadow snapshot
        risk_belief: real learner's risk belief
        test_vecs: observation vectors to test on

    Returns:
        Mean L1 distance between posterior distributions
    """
    if not test_vecs or shadow.risk is None:
        return 0.0

    shadow_risk = write_shadow_to_real_risk(shadow)
    l1_dists = []

    for x in test_vecs:
        real_post = risk_belief.single_ball_posterior(x)
        shadow_post = shadow_risk.single_ball_posterior(x)
        l1 = float(np.sum(np.abs(real_post - shadow_post)))
        l1_dists.append(l1)

    return float(np.mean(l1_dists))


def compute_full_divergence(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
    risk_belief: DangerTypeBelief,
    probe_words: List[List[str]],
    probe_gold: Optional[List[List[str]]] = None,
    test_vecs: Optional[List[np.ndarray]] = None,
    step: int = 0,
    query_id: int = 0,
) -> DivergenceRecord:
    """Compute full divergence record at one timestep."""
    top1, entropy_gap, acc_gap = measure_grammar_divergence(
        shadow, predictor, probe_words, probe_gold)

    risk_l1 = measure_risk_divergence(
        shadow, risk_belief, test_vecs or [])

    return DivergenceRecord(
        step=step,
        query_id=query_id,
        top1_agreement=top1,
        beam_entropy_gap=entropy_gap,
        probe_acc_gap=acc_gap,
        risk_l1=risk_l1,
        shadow_grammar_hash=shadow.grammar_hash(),
        shadow_risk_hash=shadow.risk_hash(),
    )
