from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import FullConfig
from ..env.state import QueryState
from ..interfaces import Example, Option
from ..learner.learner_agent import LearnerAgent


def _highlighted_attention(
    cfg: FullConfig,
    qs: QueryState,
    learner: LearnerAgent,
    action: str,
    hl_cells: Optional[Tuple],
) -> np.ndarray:
    """Return attention weights after optional highlight intervention."""
    lcfg = cfg.learner
    policy = getattr(learner, "policy", None)
    if policy is not None and policy.attention is not None:
        attn = policy.attention.weights.copy()
    else:
        L = len(qs.target_output)
        attn = np.ones(L) / max(L, 1)

    if action in ("HIGHLIGHT", "MIX") and hl_cells:
        attn = attn.copy()
        for ell in hl_cells:
            if 0 <= ell < len(attn):
                attn[ell] *= np.exp(lcfg.rho_H)
        s = attn.sum()
        if s > 0:
            attn = attn / s
    return attn


def compute_direct_pick_probs(
    cfg: FullConfig,
    qs: QueryState,
    active: List[Option],
    spec: Dict[str, any],
    learner: LearnerAgent,
) -> np.ndarray:
    """Learner-consistent pick distribution using direct privileged state."""
    action = spec["action"]
    ban_idx: Optional[int] = spec.get("ban_index")
    hl_cells: Optional[Tuple] = spec.get("highlight_cells")

    K_full = len(active)
    if K_full == 0:
        return np.array([])

    lcfg = cfg.learner
    policy = getattr(learner, "policy", None)
    attn = _highlighted_attention(cfg, qs, learner, action, hl_cells)

    if action in ("BAN", "MIX") and ban_idx is not None:
        active_sub = [o for o in active if o.index != ban_idx]
    else:
        active_sub = list(active)

    K = len(active_sub)
    if K == 0:
        return np.zeros(K_full)

    scorer = getattr(learner, "_scorer", None)
    danger_head = policy.danger_head if policy is not None else None

    sem = np.zeros(K)
    mu_d = np.zeros(K)
    u_d = np.zeros(K)

    for i, opt in enumerate(active_sub):
        if scorer is not None:
            sem[i] = scorer.score_option(
                qs.target_output, opt.text, attention_weights=attn
            )
        if danger_head is not None:
            mu, u = danger_head.predict(opt.danger_vec)
            mu_d[i] = mu
            u_d[i] = u

    U = (lcfg.alpha_sem * sem
         - lcfg.alpha_risk * mu_d
         - lcfg.alpha_unc * u_d)

    U_shifted = U - U.max()
    exp_u = np.exp(lcfg.beta_L * U_shifted)
    probs_sub = exp_u / (exp_u.sum() + 1e-30)
    eps = lcfg.epsilon
    probs_sub = (1 - eps) * probs_sub + eps / K
    probs_sub = np.clip(probs_sub, 0, 1)
    probs_sub /= probs_sub.sum()

    p = np.zeros(K_full)
    sub_idx_map = {o.index: i for i, o in enumerate(active_sub)}
    for j, o in enumerate(active):
        if o.index in sub_idx_map:
            p[j] = probs_sub[sub_idx_map[o.index]]
    return p


def rollout_estimate_direct(
    cfg: FullConfig,
    qs: QueryState,
    active: List[Option],
    spec: Dict[str, any],
    learner: LearnerAgent,
    n_rollout: int,
) -> Tuple[float, float, float]:
    """Short direct learner rollout for calibrated death/timeout/success."""
    import copy as _copy

    action = spec.get("action", "WAIT")
    ban_idx: Optional[int] = spec.get("ban_index")
    hl_cells: Optional[Tuple] = spec.get("highlight_cells")
    lcfg = cfg.learner

    scorer_snap = _copy.deepcopy(getattr(learner, "_scorer", None))
    policy = getattr(learner, "policy", None)
    danger_head_snap = _copy.deepcopy(policy.danger_head) if policy else None

    if policy is not None and policy.attention is not None:
        base_attn = policy.attention.weights.copy()
    else:
        L = len(qs.target_output)
        base_attn = np.ones(L) / max(L, 1)

    attn_action = base_attn.copy()
    if action in ("HIGHLIGHT", "MIX") and hl_cells:
        for ell in hl_cells:
            if 0 <= ell < len(attn_action):
                attn_action[ell] *= np.exp(lcfg.rho_H)
        s = attn_action.sum()
        if s > 0:
            attn_action /= s

    deaths = 0
    timeouts = 0
    successes = 0
    rng_base = np.random.default_rng(seed=hash((qs.query_id, action)) & 0xFFFFFFFF)

    for roll_i in range(n_rollout):
        scorer_roll = _copy.deepcopy(scorer_snap)
        dh_roll = _copy.deepcopy(danger_head_snap)
        rng = np.random.default_rng(int(rng_base.integers(0, 2**31 - 1)) + roll_i)

        hp = qs.hp
        rounds_used = qs.rounds_used
        target = list(qs.target_output)
        outcome = "timeout"

        banned = set(qs.banned_indices)
        if action in ("BAN", "MIX") and ban_idx is not None:
            banned.add(ban_idx)
        active_roll = [o for o in active if o.index not in banned]

        while rounds_used < qs.max_rounds and hp > 0 and active_roll:
            K = len(active_roll)
            sem = np.zeros(K)
            mu_d = np.zeros(K)
            u_d = np.zeros(K)

            for i, opt in enumerate(active_roll):
                if scorer_roll is not None:
                    sem[i] = scorer_roll.score_option(
                        target, opt.text, attention_weights=attn_action
                    )
                if dh_roll is not None:
                    mu, u = dh_roll.predict(opt.danger_vec)
                    mu_d[i] = mu
                    u_d[i] = u

            U = (lcfg.alpha_sem * sem
                 - lcfg.alpha_risk * mu_d
                 - lcfg.alpha_unc * u_d)

            U_shifted = U - U.max()
            exp_u = np.exp(lcfg.beta_L * U_shifted)
            probs = exp_u / (exp_u.sum() + 1e-30)
            eps = lcfg.epsilon
            probs = (1 - eps) * probs + eps / K
            probs /= probs.sum()

            pick_i = int(rng.choice(K, p=probs))
            picked = active_roll[pick_i]
            rounds_used += 1

            if picked.is_correct:
                if (
                    lcfg.correct_pick_learning_mode == "cortex_em"
                    and scorer_roll is not None
                    and hasattr(scorer_roll, "incremental_study")
                ):
                    pos_ex = Example(
                        words=list(picked.text),
                        output=list(target),
                    )
                    n_em_ov = lcfg.correct_pick_n_em_override
                    if lcfg.eta_correct_pick >= 1.0 or rng.random() < lcfg.eta_correct_pick:
                        scorer_roll.incremental_study([pos_ex], n_em_override=n_em_ov)
                outcome = "success"
                break

            damage = picked.risk_class
            hp = max(0, hp - damage)
            if hp <= 0:
                outcome = "death"
                break

            if scorer_roll is not None and hasattr(scorer_roll, "incremental_study"):
                rendered = picked.rendered_output
                if rendered and lcfg.eta_reveal >= 1.0:
                    ex = Example(words=list(picked.text), output=list(rendered))
                    scorer_roll.incremental_study([ex])

            if dh_roll is not None:
                dh_roll.update(picked.danger_vec, damage)

        if outcome == "success":
            successes += 1
        elif outcome == "death":
            deaths += 1
        else:
            timeouts += 1

    p_death = deaths / n_rollout
    p_timeout = timeouts / n_rollout
    p_success = successes / n_rollout
    return float(p_death), float(p_timeout), float(p_success)
