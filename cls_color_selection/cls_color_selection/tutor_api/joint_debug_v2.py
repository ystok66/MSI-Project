"""
joint_debug_v2.py — Enhanced divergence metrics (Phase 3 fix).

Three layers:
  L1: Behavioral — top-1 agreement, success gap
  L2: Distributional — JS divergence over beam posterior
  L3: Parameter — role/emit stat L1 distance
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .shadow_snapshot import ShadowLearnerSnapshot
from .shadow_update import shadow_predict_target, _reconstruct_shadow_cls
from .shadow_clone import write_shadow_to_real_risk
from .joint_debug import DivergenceRecord, JointDebugLog
from ..learner.cls_wrapper import CLSSequencePredictor
from ..learner.risk_belief import DangerTypeBelief


def compute_full_divergence_v2(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
    risk_belief: DangerTypeBelief,
    probe_words: List[List[str]],
    probe_gold: Optional[List[List[str]]] = None,
    test_vecs: Optional[List[np.ndarray]] = None,
    step: int = 0,
    query_id: int = 0,
) -> DivergenceRecord:
    """Compute multi-layer divergence record."""
    # Layer 1: Behavioral
    top1_agree, acc_gap = _behavioral_divergence(
        shadow, predictor, probe_words, probe_gold)

    # Layer 2: Distributional (JS over beam)
    js_div = _js_divergence(shadow, predictor, probe_words)

    # Layer 3: Parameter-level
    role_l1, emit_l1 = _parameter_divergence(shadow, predictor)

    # Risk L1
    risk_l1 = _risk_divergence(shadow, risk_belief, test_vecs or [])

    rec = DivergenceRecord(
        step=step,
        query_id=query_id,
        top1_agreement=top1_agree,
        beam_entropy_gap=js_div,  # repurpose field for JS
        probe_acc_gap=acc_gap,
        risk_l1=risk_l1,
        shadow_grammar_hash=shadow.grammar_hash(),
        shadow_risk_hash=shadow.risk_hash(),
    )
    # Store extra fields as attributes
    rec._js_div = js_div
    rec._role_l1 = role_l1
    rec._emit_l1 = emit_l1
    return rec


def _behavioral_divergence(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
    probe_words: List[List[str]],
    probe_gold: Optional[List[List[str]]] = None,
) -> Tuple[float, float]:
    """Layer 1: top-1 agreement and probe accuracy gap."""
    if not probe_words:
        return 1.0, 0.0

    n_agree = 0
    real_correct = 0
    shadow_correct = 0

    for i, words in enumerate(probe_words):
        real_pred = predictor.predict_target(words)
        shadow_pred = shadow_predict_target(shadow, words)

        if shadow_pred is not None and real_pred == shadow_pred:
            n_agree += 1

        if probe_gold and i < len(probe_gold):
            if real_pred == probe_gold[i]:
                real_correct += 1
            if shadow_pred is not None and shadow_pred == probe_gold[i]:
                shadow_correct += 1

    top1 = n_agree / len(probe_words)
    acc_gap = abs(real_correct - shadow_correct) / max(len(probe_words), 1)
    return top1, acc_gap


def _js_divergence(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
    probe_words: List[List[str]],
) -> float:
    """Layer 2: Mean Jensen-Shannon divergence over beam posteriors."""
    if not probe_words:
        return 0.0

    from scipy.special import logsumexp

    js_vals = []
    for words in probe_words:
        try:
            # Real beam
            real_beam = predictor.beam_posterior(words)
            if not real_beam:
                continue

            # Shadow beam
            agent, shadow_pred = _reconstruct_shadow_cls(shadow)
            if shadow_pred is None:
                continue
            shadow_beam = shadow_pred.beam_posterior(words)
            if not shadow_beam:
                continue

            # Build probability vectors keyed by output
            real_dict = _beam_to_prob_dict(real_beam)
            shadow_dict = _beam_to_prob_dict(shadow_beam)

            # Align keys
            all_keys = sorted(set(real_dict.keys()) | set(shadow_dict.keys()))
            if not all_keys:
                continue

            p = np.array([real_dict.get(k, 1e-10) for k in all_keys])
            q = np.array([shadow_dict.get(k, 1e-10) for k in all_keys])

            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)

            # JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
            kl_pm = np.sum(p * np.log(p / m + 1e-30))
            kl_qm = np.sum(q * np.log(q / m + 1e-30))
            js = 0.5 * kl_pm + 0.5 * kl_qm
            js_vals.append(max(0.0, js))

        except Exception:
            continue

    return float(np.mean(js_vals)) if js_vals else 0.0


def _beam_to_prob_dict(beam) -> Dict[str, float]:
    """Convert beam list to {output_str: probability} dict."""
    from scipy.special import logsumexp
    if not beam:
        return {}
    scores = np.array([b[0] for b in beam])
    log_q = scores - logsumexp(scores)
    probs = np.exp(log_q)
    d = {}
    for b, prob in zip(beam, probs):
        output = b[2] if len(b) > 2 else b[1]
        key = str(output)
        d[key] = d.get(key, 0.0) + prob
    return d


def _parameter_divergence(
    shadow: ShadowLearnerSnapshot,
    predictor: CLSSequencePredictor,
) -> Tuple[float, float]:
    """Layer 3: Parameter-level divergence (role counts + emit stats)."""
    real_lib = predictor.get_library()
    if not real_lib or not shadow.grammar:
        return 0.0, 0.0

    role_l1s = []
    emit_l1s = []

    for word in real_lib:
        if word not in shadow.grammar:
            continue
        real_c = real_lib[word]
        shadow_c = shadow.grammar[word]

        # Role count L1
        role_diff = 0.0
        for r in real_c.role_counts:
            rv = real_c.role_counts.get(r, 0.0)
            sv = shadow_c.role_counts.get(r, 0.0)
            role_diff += abs(float(rv) - float(sv))
        role_l1s.append(role_diff)

        # Emit stat L1 (sum_wx distance)
        try:
            real_wx = np.array(real_c.emit_stats.get('sum_wx', np.zeros(1)))
            shadow_wx = np.array(shadow_c.emit_stats.get('sum_wx', np.zeros(1)))
            if real_wx.shape == shadow_wx.shape:
                emit_l1s.append(float(np.sum(np.abs(real_wx - shadow_wx))))
        except Exception:
            pass

    role_l1 = float(np.mean(role_l1s)) if role_l1s else 0.0
    emit_l1 = float(np.mean(emit_l1s)) if emit_l1s else 0.0
    return role_l1, emit_l1


def _risk_divergence(
    shadow: ShadowLearnerSnapshot,
    risk_belief: DangerTypeBelief,
    test_vecs: List[np.ndarray],
) -> float:
    """Risk posterior L1 divergence."""
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


# ── Enhanced summary for JointDebugLog ────────────────────────

def enhanced_summary(log: JointDebugLog) -> Dict:
    """Summary with all three divergence layers."""
    base = log.summary()

    # Extract extra fields
    js_vals = [getattr(d, '_js_div', 0.0) for d in log.divergences]
    role_vals = [getattr(d, '_role_l1', 0.0) for d in log.divergences]
    emit_vals = [getattr(d, '_emit_l1', 0.0) for d in log.divergences]

    base['D_gram_JS'] = float(np.mean(js_vals)) if js_vals else 0.0
    base['D_param_role_l1'] = float(np.mean(role_vals)) if role_vals else 0.0
    base['D_param_emit_l1'] = float(np.mean(emit_vals)) if emit_vals else 0.0

    return base
