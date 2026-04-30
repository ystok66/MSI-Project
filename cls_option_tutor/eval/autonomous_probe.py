"""
autonomous_probe.py — Tutor-off autonomous competence evaluator.

Measures learner competence on a fixed probe set WITHOUT tutor interventions.
Clones the learner so the real learner is never mutated.
Optionally freezes semantic, risk, and memory updates during the probe
to measure pure pre-existing competence.

Key invariant: run_autonomous_probe() NEVER modifies the input learner.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..env.option_env import OptionEnv
from ..env.state import BlockState, QueryState
from ..env.interventions import get_active_menu
from ..config import FullConfig
from ..learner.learner_agent import LearnerAgent


# ── Probe result ──────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Results from running an autonomous probe set."""
    sr: float              # success rate
    first_ok: float        # first-pick correct rate (no prior wrong picks in query)
    timeout_rate: float
    death_rate: float
    damage_mean: float
    attempts_mean: float
    n: int


# ── Probe runner ──────────────────────────────────────────────────────────────

def run_autonomous_probe(
    learner: LearnerAgent,
    env: OptionEnv,
    task_id: str,
    probe_seed: int,
    cfg: FullConfig,
    *,
    n_probe: int = 5,
    freeze_semantic: bool = True,
    freeze_risk: bool = True,
    freeze_memory: bool = True,
) -> ProbeResult:
    """Run autonomous probe queries with a CLONED learner, no tutor.

    The input learner is NEVER mutated.

    Args:
        learner: the real learner (will be deepcopied)
        env: environment
        task_id: grammar/task to test
        probe_seed: independent seed for probe query generation
        cfg: config
        n_probe: number of probe queries
        freeze_semantic: if True, disable scorer updates during probe
        freeze_risk: if True, disable danger_head updates during probe
        freeze_memory: if True, disable cross-query episodic memory

    Returns:
        ProbeResult with success rate, first-ok rate, etc.
    """
    # Clone learner — real learner is never modified
    probe_learner = _clone_learner(learner, freeze_semantic, freeze_risk, freeze_memory)

    # Generate probe block
    probe_block = _make_probe_block(env, task_id, probe_seed, cfg, n_probe)

    # Attach cloned learner to probe block — preserves learned state
    probe_learner.prepare_probe_block(probe_block)

    # Run all probe queries without tutor
    max_steps = len(probe_block.queries) * 20
    steps = 0
    while not probe_block.done and steps < max_steps:
        steps += 1
        qs = probe_block.current_query
        if qs is None or qs.done:
            break
        # No tutor action — just WAIT
        env.tutor_act(probe_block, "WAIT")
        if qs.done:
            continue
        # Learner acts
        probe_learner.act(probe_block, env)

    if not probe_block.done:
        probe_block.done = True

    # Compute metrics
    return _compute_probe_result(probe_block)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clone_learner(
    learner: LearnerAgent,
    freeze_semantic: bool,
    freeze_risk: bool,
    freeze_memory: bool,
) -> LearnerAgent:
    """Deep-copy learner with optional freezes.

    Freezing is done by replacing update methods with no-ops.
    """
    clone = copy.deepcopy(learner)

    if freeze_semantic and hasattr(clone, '_scorer') and clone._scorer is not None:
        if hasattr(clone._scorer, 'incremental_study'):
            clone._scorer.incremental_study = lambda *args, **kwargs: None

    if freeze_risk and hasattr(clone, 'policy') and hasattr(clone.policy, 'danger_head'):
        dh = clone.policy.danger_head
        if dh is not None:
            dh.update = lambda *args, **kwargs: None
            dh.hazard.update = lambda *args, **kwargs: None
            dh.severity.update = lambda *args, **kwargs: None

    if freeze_memory:
        # Disable cross-query episodic memory accumulation
        if hasattr(clone, 'policy') and hasattr(clone.policy, 'memory'):
            clone.policy.memory = None

    return clone


def _make_probe_block(
    env: OptionEnv,
    task_id: str,
    probe_seed: int,
    cfg: FullConfig,
    n_probe: int,
) -> BlockState:
    """Generate a probe block with independent seed.

    All queries are treated as eval-phase (no teach/obs distinction).
    """
    # Use a separate seed namespace to avoid collision with training
    block = env.reset_block(task_id, seed=probe_seed)

    # Override phase structure: all queries are "eval" (no teaching)
    block.obs_phase_queries = 0
    block.teach_phase_queries = 0

    # Trim to n_probe if block has more queries
    if len(block.queries) > n_probe:
        block.queries = block.queries[:n_probe]

    # Reset all query states for clean probe
    for qs in block.queries:
        qs.done = False
        qs.success = False
        qs.skipped = False
        qs.rounds_used = 0
        qs.reveal_history.clear()
        qs.risk_hint_history.clear()
        qs.banned_indices.clear()
        qs.highlighted_cells = ()
        qs.shortlisted_indices = None

    block.done = False
    return block


def _compute_probe_result(block: BlockState) -> ProbeResult:
    """Compute probe metrics from a completed probe block."""
    n = len(block.queries)
    if n == 0:
        return ProbeResult(
            sr=0.0, first_ok=0.0, timeout_rate=0.0,
            death_rate=0.0, damage_mean=0.0, attempts_mean=0.0, n=0,
        )

    successes = 0
    first_oks = 0
    deaths = 0
    timeouts = 0
    total_damage = 0
    total_attempts = 0

    for qs in block.queries:
        if qs.success:
            successes += 1
        if qs.hp <= 0 and not qs.success:
            deaths += 1
        if qs.hp > 0 and not qs.success and not qs.skipped:
            timeouts += 1

        # First-OK: success with no prior wrong picks
        wrong_picks = sum(
            1 for ls in block.learner_trace
            if ls.query_id == qs.query_id and ls.action == "pick" and not ls.correct
        )
        all_picks = sum(
            1 for ls in block.learner_trace
            if ls.query_id == qs.query_id and ls.action == "pick"
        )
        if qs.success and wrong_picks == 0:
            first_oks += 1

        total_attempts += all_picks

        # Damage
        q_damage = sum(
            ls.damage for ls in block.learner_trace
            if ls.query_id == qs.query_id and ls.damage is not None
        )
        total_damage += q_damage

    return ProbeResult(
        sr=successes / n,
        first_ok=first_oks / n,
        timeout_rate=timeouts / n,
        death_rate=deaths / n,
        damage_mean=total_damage / n,
        attempts_mean=total_attempts / n,
        n=n,
    )
