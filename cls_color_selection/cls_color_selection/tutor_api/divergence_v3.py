"""
divergence_v3.py — Phase 4 divergence & predictive validity metrics.

Measures how well the tutor's learner model matches the real learner.
Unlike v2 (which measured shadow copy drift), v3 measures INVERSE
INFERENCE ACCURACY — a fundamentally different quantity.

Three metric layers:
    1. Behavioral: top-1 agreement, sequence edit distance
    2. Distributional: JS divergence over beam posteriors
    3. Predictive: PredAcc_next (can tutor predict learner's NEXT output?)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from scipy.special import logsumexp


def compute_inverse_divergence(
    tutor_learner_model,
    real_predictor,
    probe_words: List[List[str]],
    probe_gold: Optional[List[List[str]]] = None,
    phase: str = 'teach',
    query_idx: int = 0,
) -> Dict:
    """Compute tutor-learner-model vs real-learner divergence.

    Args:
        tutor_learner_model: TutorLearnerModel
        real_predictor: CLSSequencePredictor (real learner)
        probe_words: list of probe queries (each is [str, ...])
        probe_gold: optional gold outputs for accuracy
        phase: 'pre_teach' | 'obs' | 'teach' | 'post_teach'
        query_idx: index within phase

    Returns:
        Dict with divergence metrics
    """
    if not probe_words:
        return _empty_record(phase, query_idx)

    n_agree = 0
    tutor_correct = 0
    real_correct = 0
    js_vals = []
    edit_dists = []

    for pi, words in enumerate(probe_words):
        # Tutor's prediction of what learner would output
        tutor_pred = tutor_learner_model.predict_learner(words)
        # Real learner's actual output
        try:
            real_pred = real_predictor.predict_target(words)
        except Exception:
            real_pred = None

        # ── Layer 1: Behavioral ──
        if tutor_pred is not None and real_pred is not None:
            if tutor_pred == real_pred:
                n_agree += 1
            # Sequence edit distance
            ed = _seq_edit_distance(tutor_pred, real_pred)
            edit_dists.append(ed)

        # ── Accuracy vs gold ──
        if probe_gold and pi < len(probe_gold):
            gold = probe_gold[pi]
            if tutor_pred is not None and tutor_pred == gold:
                tutor_correct += 1
            if real_pred is not None and real_pred == gold:
                real_correct += 1

        # ── Layer 2: Distributional (JS divergence) ──
        try:
            tutor_beam = tutor_learner_model.beam_posterior(words)
            real_beam = real_predictor.beam_posterior(words)
            if tutor_beam and real_beam:
                js = _beam_js_divergence(tutor_beam, real_beam)
                js_vals.append(js)
        except Exception:
            pass

    n_probes = len(probe_words)
    return {
        'phase': phase,
        'query_idx': query_idx,
        # Layer 1: Behavioral
        'top1_agreement': n_agree / max(n_probes, 1),
        'mean_edit_dist': float(np.mean(edit_dists)) if edit_dists else 0.0,
        # Layer 2: Distributional
        'js_divergence': float(np.mean(js_vals)) if js_vals else 0.0,
        # Layer 3: Accuracy
        'tutor_model_accuracy': tutor_correct / max(n_probes, 1),
        'real_learner_accuracy': real_correct / max(n_probes, 1),
        'accuracy_gap': abs(tutor_correct - real_correct) / max(n_probes, 1),
        'n_probes': n_probes,
    }


def compute_predictive_validity(
    tutor_learner_model,
    real_predictor,
    next_words: List[str],
) -> Dict:
    """PredAcc_next: can tutor predict learner's output on NEXT query?

    This is the most important metric. It directly tests whether
    the tutor's learner model has practical predictive power.
    """
    tutor_pred = tutor_learner_model.predict_learner(next_words)
    try:
        real_pred = real_predictor.predict_target(next_words)
    except Exception:
        real_pred = None

    match = False
    edit_dist = -1

    if tutor_pred is not None and real_pred is not None:
        match = (tutor_pred == real_pred)
        edit_dist = _seq_edit_distance(tutor_pred, real_pred)

    return {
        'pred_match': match,
        'tutor_pred': tutor_pred,
        'real_pred': real_pred,
        'edit_distance': edit_dist,
    }


# ── Helpers ───────────────────────────────────────────────────

def _empty_record(phase, query_idx):
    return {
        'phase': phase, 'query_idx': query_idx,
        'top1_agreement': None, 'js_divergence': None,
        'mean_edit_dist': None,
        'tutor_model_accuracy': None, 'real_learner_accuracy': None,
        'accuracy_gap': None, 'n_probes': 0,
    }


def _seq_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Simple element-wise edit distance between two sequences."""
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 0
    diffs = 0
    for i in range(max_len):
        a = seq_a[i] if i < len(seq_a) else None
        b = seq_b[i] if i < len(seq_b) else None
        if a != b:
            diffs += 1
    return diffs


def _beam_js_divergence(beam_a, beam_b) -> float:
    """Jensen-Shannon divergence between two beam posteriors."""
    def beam_to_probs(beam):
        scores = np.array([b[0] for b in beam])
        log_q = scores - logsumexp(scores)
        probs = np.exp(log_q)
        d = {}
        for b, p in zip(beam, probs):
            # Key: rendered output (last element of tuple)
            key = str(b[-1]) if len(b) > 1 else str(b)
            d[key] = d.get(key, 0.0) + p
        return d

    da = beam_to_probs(beam_a)
    db = beam_to_probs(beam_b)
    all_keys = sorted(set(da.keys()) | set(db.keys()))
    if not all_keys:
        return 0.0

    p = np.array([da.get(k, 1e-10) for k in all_keys])
    q = np.array([db.get(k, 1e-10) for k in all_keys])
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m + 1e-30)) + \
         0.5 * np.sum(q * np.log(q / m + 1e-30))
    return max(0.0, float(js))
