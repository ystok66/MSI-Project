"""
counterfactual.py — One-step counterfactual learner modeling.

Compare WAIT vs HINT(S) via:
  1. Clone learner model
  2. Simulate each branch through end of current query
  3. Evaluate future competence on small probe set
  4. Return ΔLearn = A(hint_branch) - A(wait_branch)

Dual probe metrics:
  - Binary: top-1 exact match (coarse)
  - Soft: sum of log P(Y* | grammar) via beam score (sensitive)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import numpy as np
from scipy.special import logsumexp


@dataclass
class CounterfactualResult:
    """Result of one-step counterfactual comparison."""
    # Binary probe metric
    delta_learn_bin: float     # top-1 exact-match accuracy diff
    probe_acc_hint: float
    probe_acc_wait: float
    # Soft probe metric (beam log-likelihood)
    delta_learn_soft: float    # sum of log P(Y*) difference
    probe_ll_hint: float       # mean log P(Y* | hint grammar)
    probe_ll_wait: float       # mean log P(Y* | wait grammar)
    # Meta
    n_probes: int
    hint_branch_details: Dict
    wait_branch_details: Dict


def one_step_counterfactual(
    learner_model,
    words: List[str],
    gt: List[str],
    hint_positions: List[Tuple[int, str]],
    wrong_mask: List[bool],
    feedback: dict,
    probe_words: List[List[str]],
    probe_golds: Optional[List[List[str]]] = None,
    rho_assist: float = 0.3,
) -> CounterfactualResult:
    """One-step counterfactual: compare HINT(S) vs WAIT.

    Args:
        learner_model: TutorLearnerModel (will be deep-copied for each branch)
        words: current query words
        gt: ground truth output
        hint_positions: [(pos, color), ...] proposed hints
        wrong_mask: [True=correct, False=wrong] per position
        feedback: feedback dict from confirm fail
        probe_words: list of probe query word lists
        probe_golds: list of probe query ground truths (optional)
        rho_assist: evidence discount for assisted positions

    Returns:
        CounterfactualResult with dual metrics
    """
    # ── Branch A: HINT(S) ──
    hint_model = copy.deepcopy(learner_model)
    hint_details = _simulate_hint_branch(
        hint_model, words, gt, hint_positions,
        wrong_mask, feedback, rho_assist)

    # ── Branch B: WAIT (learner tries again with no help) ──
    wait_model = copy.deepcopy(learner_model)
    wait_details = _simulate_wait_branch(
        wait_model, words, gt, wrong_mask, feedback)

    # ── Evaluate future competence on probes (dual metric) ──
    acc_hint = _evaluate_probes_binary(hint_model, probe_words, probe_golds)
    acc_wait = _evaluate_probes_binary(wait_model, probe_words, probe_golds)
    ll_hint = _evaluate_probes_soft(hint_model, probe_words, probe_golds)
    ll_wait = _evaluate_probes_soft(wait_model, probe_words, probe_golds)

    return CounterfactualResult(
        delta_learn_bin=acc_hint - acc_wait,
        probe_acc_hint=acc_hint,
        probe_acc_wait=acc_wait,
        delta_learn_soft=ll_hint - ll_wait,
        probe_ll_hint=ll_hint,
        probe_ll_wait=ll_wait,
        n_probes=len(probe_words),
        hint_branch_details=hint_details,
        wait_branch_details=wait_details,
    )


def _simulate_hint_branch(
    model,
    words: List[str],
    gt: List[str],
    hint_positions: List[Tuple[int, str]],
    wrong_mask: List[bool],
    feedback: dict,
    rho_assist: float,
) -> Dict:
    """Simulate hint branch: hint fixes positions → submit → feedback update.

    After hint, we assume learner submits a sequence where:
    - hint positions are correct (tutor-provided)
    - other positions retain their current predictions
    Then feedback update happens with assist-aware discounting.
    """
    # Build the "after hint" submitted sequence
    submitted = list(feedback.get('submitted', ['?'] * len(gt)))
    assist_mask = [False] * len(gt)

    for pos, color in hint_positions:
        if 0 <= pos < len(submitted):
            submitted[pos] = color
            assist_mask[pos] = True

    # Simulate feedback update with assist discount
    try:
        beam = model.beam_posterior(words)
        if beam:
            from cls_color_selection.learner.feedback_update import FeedbackUpdater
            from cls_color_selection.config import LearnerConfig
            cfg = LearnerConfig(rho_assist=rho_assist)
            updater = FeedbackUpdater(cfg)

            q_old, q_new = updater.reweight_beam_posterior(
                beam, submitted, feedback, assist_mask=assist_mask)

            if len(q_old) > 0:
                library = model.get_library()
                if library:
                    updater.differential_m_step(library, beam, q_old, q_new)

            shift = float(np.sum(np.abs(q_new - q_old))) if len(q_old) > 0 else 0.0
        else:
            shift = 0.0
    except Exception:
        shift = 0.0

    return {
        'branch': 'hint',
        'n_assisted': sum(assist_mask),
        'posterior_shift': shift,
    }


def _simulate_wait_branch(
    model,
    words: List[str],
    gt: List[str],
    wrong_mask: List[bool],
    feedback: dict,
) -> Dict:
    """Simulate wait branch: learner gets raw feedback → full grammar update.

    No hint → submitted is the original (wrong) submission
    → feedback update without any assist discount
    → learner learns from its own mistakes (uncontaminated)
    """
    submitted = list(feedback.get('submitted', ['?'] * len(gt)))

    try:
        beam = model.beam_posterior(words)
        if beam:
            from cls_color_selection.learner.feedback_update import FeedbackUpdater
            from cls_color_selection.config import LearnerConfig
            cfg = LearnerConfig(rho_assist=1.0)  # No discount
            updater = FeedbackUpdater(cfg)

            q_old, q_new = updater.reweight_beam_posterior(
                beam, submitted, feedback)

            if len(q_old) > 0:
                library = model.get_library()
                if library:
                    updater.differential_m_step(library, beam, q_old, q_new)

            shift = float(np.sum(np.abs(q_new - q_old))) if len(q_old) > 0 else 0.0
        else:
            shift = 0.0
    except Exception:
        shift = 0.0

    return {
        'branch': 'wait',
        'posterior_shift': shift,
    }


# ── Probe evaluation ─────────────────────────────────────────

def _evaluate_probes_binary(
    model,
    probe_words: List[List[str]],
    probe_golds: Optional[List[List[str]]] = None,
) -> float:
    """Binary probe: fraction of probes where top-1 prediction matches gold."""
    if not probe_words:
        return 0.0

    n_correct = 0
    n_total = 0

    for i, words in enumerate(probe_words):
        try:
            pred = model.predict_learner(words)
            if pred is None:
                continue
            n_total += 1
            if probe_golds and i < len(probe_golds):
                if pred == list(probe_golds[i]):
                    n_correct += 1
        except Exception:
            continue

    return n_correct / max(n_total, 1)


def _evaluate_probes_soft(
    model,
    probe_words: List[List[str]],
    probe_golds: Optional[List[List[str]]] = None,
) -> float:
    """Soft probe: mean log P(Y* | grammar) via beam posterior mass on gold.

    For each probe query:
      1. Get beam posterior [(score_k, trace_k, Y_k), ...]
      2. Find traces where Y_k == Y*_gold
      3. Compute log P(Y*) = logsumexp(scores of matching traces) - logsumexp(all)
      4. If no trace matches, assign log P = -10 (floor)

    Returns: mean log P(Y*) across probes (higher = better grammar fit)
    """
    if not probe_words or not probe_golds:
        return -10.0

    log_probs = []

    for i, words in enumerate(probe_words):
        if i >= len(probe_golds):
            break
        gold = list(probe_golds[i])
        try:
            beam = model.beam_posterior(words)
            if not beam:
                log_probs.append(-10.0)
                continue

            all_scores = np.array([b[0] for b in beam])
            log_Z = logsumexp(all_scores)

            # Find traces matching gold
            matching_scores = []
            for score, trace, Y_k in beam:
                if list(Y_k) == gold:
                    matching_scores.append(score)

            if matching_scores:
                log_p_gold = logsumexp(matching_scores) - log_Z
            else:
                log_p_gold = -10.0  # Floor: gold not in beam

            log_probs.append(float(log_p_gold))
        except Exception:
            log_probs.append(-10.0)

    return float(np.mean(log_probs)) if log_probs else -10.0
