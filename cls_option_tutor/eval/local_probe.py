"""
local_probe.py — Localized contrastive probe for Phase 6.5.

Measures learning signal local to a specific teach event by:
  1. Selecting probe queries from the same grammar
  2. Computing semantic margin, policy margin, and correct-pick probability
  3. Reporting DeltaLocalProbeSR, DeltaSemanticMargin, DeltaPolicyMargin, DeltaCorrectProb

Phase 6.5 changes:
  - Uses prepare_probe_block() instead of init_block() — preserves learned state
  - Reports both semantic-only margin and full-policy margin
  - Reports correct-pick probability from learner policy
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..env.option_env import OptionEnv
from ..env.state import BlockState, QueryState
from ..env.interventions import get_active_menu
from ..learner.learner_agent import LearnerAgent
from ..config import FullConfig
from .autonomous_probe import _clone_learner


@dataclass
class LocalProbeResult:
    """Result from a localized probe evaluation."""
    n_local: int = 0
    sr: float = 0.0
    first_ok: float = 0.0
    # Semantic-only margin (scorer.score_option)
    avg_semantic_margin: float = 0.0
    avg_semantic_rank: float = 0.0
    # Full-policy margin (compute_policy utilities)
    avg_policy_margin: float = 0.0
    avg_policy_rank: float = 0.0
    # Correct-pick probability from learner policy
    avg_correct_pick_prob: float = 0.0


@dataclass
class LocalLearningResult:
    """Local learning increment from pre/post probe comparison."""
    delta_local_sr: float = 0.0
    delta_local_first_ok: float = 0.0
    delta_semantic_margin: float = 0.0
    delta_policy_margin: float = 0.0
    delta_semantic_rank: float = 0.0
    delta_policy_rank: float = 0.0
    delta_correct_prob: float = 0.0
    n_local: int = 0


def run_local_probe(
    learner: LearnerAgent,
    env: OptionEnv,
    task_id: str,
    cfg: FullConfig,
    probe_seed: int = 54321,
    n_local: int = 8,
) -> LocalProbeResult:
    """Run a localized probe evaluation.

    Phase 6.5: Uses prepare_probe_block to preserve learned state.
    Reports semantic margin, policy margin, and correct-pick probability.

    Args:
        learner: The real learner (will be cloned internally)
        env: Environment for generating probes
        task_id: Grammar/task to probe
        cfg: Config
        probe_seed: Seed for reproducible probe generation
        n_local: Number of local probe queries
    """
    # Clone and freeze learner updates so the probe measures retained
    # competence instead of allowing the probe itself to teach the clone.
    learner_clone = _clone_learner(
        learner,
        freeze_semantic=True,
        freeze_risk=True,
        freeze_memory=True,
    )

    # Create a probe env with eval-only queries
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.env.N_obs = 0
    probe_cfg.env.N_teach = 0
    probe_cfg.env.N_eval = n_local
    probe_cfg.env.M_queries = n_local

    probe_env = OptionEnv(cfg=probe_cfg, data_dir=env.data_dir)
    probe_block = probe_env.reset_block(task_id, seed=probe_seed)

    # Phase 6.5: preserve learned state via prepare_probe_block
    learner_clone.prepare_probe_block(probe_block)

    successes = 0
    first_ok_count = 0
    sem_margins = []
    sem_ranks = []
    pol_margins = []
    pol_ranks = []
    correct_probs = []

    for qi, qs in enumerate(probe_block.queries):
        if qs.done:
            continue

        # Compute margins BEFORE action
        sm, sr = _compute_semantic_margin_rank(learner_clone, qs)
        sem_margins.append(sm)
        sem_ranks.append(sr)

        pm, pr, cp = _compute_policy_margin_rank_prob(learner_clone, qs)
        pol_margins.append(pm)
        pol_ranks.append(pr)
        correct_probs.append(cp)

        # Let learner act (frozen — no updates)
        while not qs.done and qs.rounds_used < qs.max_rounds:
            learner_clone.act(probe_block, probe_env)

        if qs.success:
            successes += 1
            if qi == 0 or not any(
                probe_block.queries[j].success for j in range(qi)
            ):
                first_ok_count += 1

    n = len(probe_block.queries)
    return LocalProbeResult(
        n_local=n,
        sr=successes / max(n, 1),
        first_ok=first_ok_count / max(n, 1),
        avg_semantic_margin=float(np.mean(sem_margins)) if sem_margins else 0.0,
        avg_semantic_rank=float(np.mean(sem_ranks)) if sem_ranks else 0.0,
        avg_policy_margin=float(np.mean(pol_margins)) if pol_margins else 0.0,
        avg_policy_rank=float(np.mean(pol_ranks)) if pol_ranks else 0.0,
        avg_correct_pick_prob=float(np.mean(correct_probs)) if correct_probs else 0.0,
    )


def compute_local_learning(
    pre: LocalProbeResult,
    post: LocalProbeResult,
) -> LocalLearningResult:
    """Compute local learning increment from pre/post probes."""
    return LocalLearningResult(
        delta_local_sr=post.sr - pre.sr,
        delta_local_first_ok=post.first_ok - pre.first_ok,
        delta_semantic_margin=post.avg_semantic_margin - pre.avg_semantic_margin,
        delta_policy_margin=post.avg_policy_margin - pre.avg_policy_margin,
        delta_semantic_rank=pre.avg_semantic_rank - post.avg_semantic_rank,  # positive = improved
        delta_policy_rank=pre.avg_policy_rank - post.avg_policy_rank,  # positive = improved
        delta_correct_prob=post.avg_correct_pick_prob - pre.avg_correct_pick_prob,
        n_local=pre.n_local,
    )


def _compute_semantic_margin_rank(
    learner: LearnerAgent,
    qs: QueryState,
) -> Tuple[float, float]:
    """Compute semantic-only margin and rank for a query.

    Uses scorer.score_option with attention weights.

    margin = S_sem(j*) - max_{j!=j*} S_sem(j)
    rank = 1 + #{j!=j* : S_sem(j) > S_sem(j*)}
    """
    scorer = learner._scorer
    if scorer is None:
        return 0.0, float(len(qs.menu))

    target = list(qs.target_output) if qs.target_output else []
    if not target:
        return 0.0, float(len(qs.menu))

    # Use attention weights if available, else uniform
    weights = None
    if learner.policy.attention is not None and learner.policy.attention.L == len(target):
        weights = learner.policy.attention.weights
    else:
        weights = np.ones(len(target)) / len(target)

    active = get_active_menu(qs)
    scores = []
    correct_idx = None

    for i, opt in enumerate(active):
        try:
            s = scorer.score_option(target, list(opt.text), attention_weights=weights)
        except Exception:
            s = 0.0
        scores.append(s)
        if opt.is_correct:
            correct_idx = i

    if correct_idx is None:
        return 0.0, float(len(active))

    correct_score = scores[correct_idx]
    other_scores = [s for j, s in enumerate(scores) if j != correct_idx]

    if not other_scores:
        return correct_score, 1.0

    max_other = max(other_scores)
    margin = correct_score - max_other
    rank = 1 + sum(1 for s in other_scores if s > correct_score)

    return float(margin), float(rank)


def _compute_policy_margin_rank_prob(
    learner: LearnerAgent,
    qs: QueryState,
) -> Tuple[float, float, float]:
    """Compute full-policy margin, rank, and correct-pick probability.

    Uses learner.get_policy_snapshot_for_query() which includes:
    attention, danger head, episodic memory, negative memory, etc.

    Returns: (policy_margin, policy_rank, correct_pick_prob)
    """
    try:
        po = learner.get_policy_snapshot_for_query(qs)
    except Exception:
        return 0.0, float(len(qs.menu)), 0.0

    active = get_active_menu(qs)
    if len(active) == 0:
        return 0.0, 0.0, 0.0

    # Find correct option index in active menu
    correct_idx = None
    for i, opt in enumerate(active):
        if opt.is_correct:
            correct_idx = i
            break

    if correct_idx is None:
        return 0.0, float(len(active)), 0.0

    # Policy utilities (first K elements, excluding refresh)
    K = len(active)
    utils = po.utilities[:K]

    correct_util = utils[correct_idx]
    other_utils = [u for j, u in enumerate(utils) if j != correct_idx]

    if not other_utils:
        margin = correct_util
        rank = 1.0
    else:
        max_other = max(other_utils)
        margin = correct_util - max_other
        rank = 1 + sum(1 for u in other_utils if u > correct_util)

    # Correct-pick probability from policy output
    probs = po.probs[:K]
    correct_prob = float(probs[correct_idx]) if correct_idx < len(probs) else 0.0

    return float(margin), float(rank), correct_prob
