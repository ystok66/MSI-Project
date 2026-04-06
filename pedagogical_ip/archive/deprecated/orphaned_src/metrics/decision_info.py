"""C1 — Decision-Aware Information Metrics.

Replaces generic Shannon entropy IG with metrics that directly
measure how much a warning helps the agent make the correct
branch decision.

Three metrics:
  1. Decision Bayes Risk: BR = 1 - max(p_safe_A, p_safe_B)
  2. Margin Gain: ΔM = M_post - M_pre where M = J(risky) - J(safe)
  3. Directional Correctness Gain: DCG = 1[post correct] - 1[pre correct]
"""

from __future__ import annotations

import numpy as np


def decision_bayes_risk(
    p_safe_a: float,
    p_safe_b: float,
) -> float:
    """Bayes risk: probability of making the wrong branch choice.

    BR = 1 - max(p_safe_a, p_safe_b)

    Lower = agent is more certain about which branch is safe.
    """
    return 1.0 - max(p_safe_a, p_safe_b)


def decision_info_gain(
    pre_p_safe_a: float, pre_p_safe_b: float,
    post_p_safe_a: float, post_p_safe_b: float,
) -> float:
    """Decision-aware information gain: reduction in Bayes risk.

    DIG = BR_pre - BR_post

    Positive = warning reduced decision uncertainty.
    """
    br_pre = decision_bayes_risk(pre_p_safe_a, pre_p_safe_b)
    br_post = decision_bayes_risk(post_p_safe_a, post_p_safe_b)
    return br_pre - br_post


def margin_gain(
    pre_score_safe: float, pre_score_risky: float,
    post_score_safe: float, post_score_risky: float,
) -> float:
    """Margin gain in planner space.

    M = score(risky) - score(safe)  [positive = safe is preferred]
    ΔM = M_post - M_pre

    Positive = warning made safe branch relatively more preferred.
    """
    m_pre = pre_score_risky - pre_score_safe
    m_post = post_score_risky - post_score_safe
    return m_post - m_pre


def directional_correctness_gain(
    pre_chose_safe: bool,
    post_chose_safe: bool,
) -> int:
    """Directional correctness gain.

    DCG = 1[post picks safe] - 1[pre picks safe]

    +1 = warning flipped choice from unsafe to safe
     0 = no change
    -1 = warning flipped from safe to unsafe (bad)
    """
    return int(post_chose_safe) - int(pre_chose_safe)


def compute_branch_posteriors(
    summary_a: np.ndarray,
    summary_b: np.ndarray,
    scorer,
    build_input_fn,
    concept_lib,
) -> tuple[float, float]:
    """Compute P(safe|branch) for each branch using scorer.

    Returns (p_safe_a, p_safe_b).
    """
    inp_a = build_input_fn(summary_a, concept_lib)
    inp_b = build_input_fn(summary_b, concept_lib)
    p_a = scorer.predict_safe_prob(inp_a)
    p_b = scorer.predict_safe_prob(inp_b)
    return float(p_a), float(p_b)


def compute_all_decision_metrics(
    pre_p_safe_a: float, pre_p_safe_b: float,
    post_p_safe_a: float, post_p_safe_b: float,
    oracle_safe_id: int,
) -> dict:
    """Compute all decision-aware metrics from pre/post posteriors.

    Returns dict with BR_pre, BR_post, DIG, pre_correct, post_correct, DCG.
    """
    br_pre = decision_bayes_risk(pre_p_safe_a, pre_p_safe_b)
    br_post = decision_bayes_risk(post_p_safe_a, post_p_safe_b)
    dig = br_pre - br_post

    # Which branch would be chosen pre/post
    pre_choice = 0 if pre_p_safe_a >= pre_p_safe_b else 1
    post_choice = 0 if post_p_safe_a >= post_p_safe_b else 1
    pre_correct = (pre_choice == oracle_safe_id)
    post_correct = (post_choice == oracle_safe_id)
    dcg = int(post_correct) - int(pre_correct)

    return {
        "BR_pre": round(br_pre, 4),
        "BR_post": round(br_post, 4),
        "DIG": round(dig, 4),
        "pre_correct": pre_correct,
        "post_correct": post_correct,
        "DCG": dcg,
    }
