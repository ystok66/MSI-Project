"""
Transfer Evaluation — Phase 9.

Evaluates agent performance WITHOUT a tutor after tutor-assisted training.

Transfer protocol:
1. COPY: learned predictor weights (frozen snapshot)
2. RESET: episodic belief state, position, inventory, runner state
3. DISABLE: tutor (tutor_mode="none", robot_belief_mode=False)

This separation is critical:
- What transfers: learned model parameters
- What resets: short-term episode state
"""

from __future__ import annotations

from typing import Optional
from copy import deepcopy

import numpy as np

from ..agents.predictor_protocol import (
    snapshot_predictor, restore_predictor,
)
from ..envs.lattice_v2_runner import LatticeV2Runner
from .phase9_metrics import compute_transfer_summary, TransferSummary


def snapshot_learned_params(state) -> dict:
    """Snapshot the learned parameters that should transfer.

    Uses predictor_protocol.snapshot_predictor for shape-agnostic
    deep-copy. Works with any head type (linear, basis, slowfast).

    COPIES: full predictor state (weights + sufficient statistics)
    DOES NOT COPY: episodic belief, position, inventory, runner state
    """
    snapshot = {}
    if hasattr(state, 'latent_predictor') and state.latent_predictor is not None:
        snapshot['predictor'] = snapshot_predictor(state.latent_predictor)
    elif hasattr(state, 'risk_head') and state.risk_head is not None:
        snapshot['risk_head'] = snapshot_predictor(state.risk_head)
    return snapshot


def apply_learned_params(state, snapshot: dict) -> None:
    """Apply snapshotted learned params to a fresh episode state.

    Uses predictor_protocol.restore_predictor for shape-agnostic
    state restoration. Works with any head type.

    Only touches predictor weights. Does NOT touch belief/position/inventory.
    """
    if not snapshot:
        return
    if 'predictor' in snapshot:
        if hasattr(state, 'latent_predictor') and state.latent_predictor is not None:
            restore_predictor(state.latent_predictor, snapshot['predictor'])
    elif 'risk_head' in snapshot:
        if hasattr(state, 'risk_head') and state.risk_head is not None:
            restore_predictor(state.risk_head, snapshot['risk_head'])


def run_transfer_episodes(
    runner: LatticeV2Runner,
    trained_state,
    n_episodes: int = 10,
    seeds: Optional[list[int]] = None,
    agent_level: str = "",
    teacher_condition: str = "",
    env_condition: str = "",
    difficulty: str = "medium",
    latent_mode: bool = True,
) -> list[TransferSummary]:
    """Run transfer evaluation: no-tutor episodes with learned params.

    1. Snapshot learned params from trained_state
    2. For each seed: reset fresh episode (no tutor) → inject learned params → run
    3. Return separate TransferSummary list
    """
    if seeds is None:
        seeds = list(range(1000, 1000 + n_episodes))

    learned = snapshot_learned_params(trained_state)
    summaries = []

    for seed in seeds:
        # Fresh episode with NO tutor
        s = runner.reset(
            seed=seed,
            difficulty=difficulty,
            tutor_mode="none",
            warning_mode="none",
            latent_mode=latent_mode,
            robot_belief_mode=False,
            intervention_family_mode=False,
        )

        # Inject learned params (transfers learned model, not episode state)
        apply_learned_params(s, learned)

        # Run without tutor
        while not s.done:
            runner.step(s)

        # Compute transfer summary
        ts = compute_transfer_summary(
            s, seed=seed,
            agent_level=agent_level,
            teacher_condition=teacher_condition,
            env_condition=env_condition,
        )
        summaries.append(ts)

    return summaries


def run_standard_transfer_protocol(
    runner: LatticeV2Runner,
    family: str = "harder_baseline_v2",
    train_seeds: Optional[list[int]] = None,
    eval_seeds: Optional[list[int]] = None,
    difficulty: str = "medium",
    world_weights_seed_mode: str = "different",
    tutor_kwargs: Optional[dict] = None,
    n_train: int = 5,
    n_eval: int = 5,
) -> dict:
    """Standard train→freeze→eval transfer protocol.

    Phase 1: Train with tutor for n_train episodes.
    Phase 2: Snapshot learned predictor state.
    Phase 3: Eval with tutor_mode="none" for n_eval episodes.

    WorldWeights seed semantics:
        "same"      — eval reuses train seeds (same WorldWeights → retention test)
        "different"  — eval uses separate seeds (different WorldWeights → domain shift)

    IMPORTANT: This uses tutor_mode="none" (true tutor-off), NOT WAIT-only.
    WAIT-only still runs the full tutor scoring pipeline each step;
    tutor-off completely skips tutor dispatch. For fair transfer evaluation,
    always use true tutor-off.

    Returns:
        dict with keys:
            "train_states": list of final train episode states
            "transfer_summaries": list[TransferSummary]
            "train_seeds": list[int]
            "eval_seeds": list[int]
            "world_weights_seed_mode": str
    """
    if train_seeds is None:
        train_seeds = list(range(n_train))
    if eval_seeds is None:
        if world_weights_seed_mode == "same":
            eval_seeds = list(train_seeds)
        else:
            eval_seeds = list(range(1000, 1000 + n_eval))

    _tutor_kw = {
        "scenario_family": family,
        "difficulty": difficulty,
        "latent_mode": True,
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "belief_planning_mode": True,
        "prefix_horizon": 5,
    }
    if tutor_kwargs:
        _tutor_kw.update(tutor_kwargs)

    # Phase 1: Train with tutor
    train_states = []
    for seed in train_seeds:
        s = runner.reset(seed=seed, **_tutor_kw)
        while not s.done:
            runner.step(s)
        train_states.append(s)

    # Phase 2: Snapshot from last training episode
    last_trained = train_states[-1]

    # Phase 3: Eval without tutor
    transfer_results = run_transfer_episodes(
        runner, last_trained,
        seeds=eval_seeds,
        difficulty=difficulty,
        teacher_condition=f"tutor_{family}",
        env_condition=world_weights_seed_mode,
    )

    return {
        "train_states": train_states,
        "transfer_summaries": transfer_results,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "world_weights_seed_mode": world_weights_seed_mode,
    }
