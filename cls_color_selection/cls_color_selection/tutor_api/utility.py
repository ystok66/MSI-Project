"""
utility.py — Tutor utility function Q(a) for action selection.

Q(a) = λ_eval·G_eval(a) + λ_teach·G_teach(a) + λ_death·G_death(a)
     + λ_to·G_to(a) - λ_over·C_over(a) - λ_int·C_int(a)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..config import TutorConfig
from ..interfaces import TutorAction
from ..constants import TutorActionType
from ..environment.state import QueryState
from .tutor_state import TutorBelief
from .belief_update import compute_timeout_risk, compute_death_risk


def compute_action_utility(
    action: TutorAction,
    state: QueryState,
    belief: TutorBelief,
    risk_belief,  # learner's DangerTypeBelief
    cfg: TutorConfig,
    context: str = 'pre_select',  #  'pre_select' | 'post_confirm'
) -> float:
    """Compute Q(a) for a candidate action.

    Args:
        action: candidate action
        state: current query state
        belief: tutor belief state
        risk_belief: learner's risk belief (for estimating learner behavior)
        cfg: tutor config
        context: where in the query timeline this decision is

    Returns:
        Scalar utility value
    """
    a_type = action.action_type

    if a_type == TutorActionType.WAIT:
        return _utility_wait(state, belief, risk_belief, cfg, context)
    elif a_type == TutorActionType.WARNING:
        return _utility_warning(state, belief, cfg)
    elif a_type == TutorActionType.HINT:
        k = len(action.hint_positions) if action.hint_positions else 0
        return _utility_hint(state, belief, cfg, k)
    elif a_type == TutorActionType.COURAGE:
        return _utility_courage(state, belief, cfg)
    else:
        return 0.0


def _utility_wait(state, belief, risk_belief, cfg, context):
    """Utility of WAIT = 0 (baseline reference)."""
    return 0.0


def _utility_warning(state, belief, cfg):
    """Utility of WARNING.

    G_death(WARNING): prevents death
    C_int(WARNING): small fixed cost
    """
    p_death_wait = compute_death_risk(belief, state, None)
    g_death = cfg.lambda_death * p_death_wait
    c_int = cfg.lambda_int * 1.0
    return g_death - c_int


def _utility_hint(state, belief, cfg, k):
    """Utility of HINT_k.

    G_teach: placing k correct balls improves P(success this query)
    G_to: reduces timeout risk
    G_eval: minimal (hint doesn't teach the learner generalizable knowledge)
    C_over: k × σ(ρ · a_probe) — more costly if learner is already competent
    C_int: fixed cost per intervention
    """
    # Teaching gain: each correct ball improves fill ratio toward completion
    fill_before = state.fill_ratio
    remaining = state.L - state.filled_count
    if remaining > 0:
        fill_after = min(1.0, fill_before + k / state.L)
        g_teach = cfg.lambda_teach * (fill_after - fill_before)
    else:
        g_teach = 0.0

    # Timeout reduction: more filled → more likely to succeed on next confirm
    p_to_before = compute_timeout_risk(belief, state)
    # Simulate: if k positions filled, what's timeout risk?
    # Approximate: reduce timeout risk proportionally
    if remaining > 0:
        frac_fixed = min(k / remaining, 1.0)
        p_to_after = p_to_before * (1.0 - 0.5 * frac_fixed)
    else:
        p_to_after = p_to_before
    g_to = cfg.lambda_to * (p_to_before - p_to_after)

    # Eval gain: hint doesn't help generalization much
    g_eval = 0.0  # Phase 2: could add small boost if hint teaches learner indirectly

    # Over-help penalty: k × sigmoid(competence)
    competence = belief.sem.a_probe
    over_factor = 1.0 / (1.0 + np.exp(-5.0 * (competence - 0.5)))  # sigmoid
    c_over = cfg.lambda_over * k * over_factor

    # Fixed intervention cost
    c_int = cfg.lambda_int * 1.0

    return g_teach + g_to + g_eval - c_over - c_int


def _utility_courage(state, belief, cfg):
    """Utility of COURAGE.

    G_teach: helps learner unstick from over-avoidance
    C_int: small cost
    """
    overavoid = belief.risk.overavoid_rate
    # Courage is more valuable when learner is over-avoiding
    g_teach = cfg.lambda_teach * overavoid * 0.5
    c_int = cfg.lambda_int * 0.5  # less costly than hint
    return g_teach - c_int


def select_best_action(
    candidates: List[TutorAction],
    state: QueryState,
    belief: TutorBelief,
    risk_belief,
    cfg: TutorConfig,
    context: str = 'pre_select',
) -> Tuple[TutorAction, float]:
    """Select the action with highest utility.

    Args:
        candidates: list of candidate actions
        state: current query state
        belief: tutor belief
        risk_belief: learner's risk belief
        cfg: tutor config
        context: 'pre_select' or 'post_confirm'

    Returns:
        (best_action, best_utility)
    """
    best_action = candidates[0]  # WAIT
    best_util = -np.inf

    for action in candidates:
        u = compute_action_utility(
            action, state, belief, risk_belief, cfg, context)
        if u > best_util:
            best_util = u
            best_action = action

    return best_action, best_util
