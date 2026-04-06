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

from ..agents.cost_risk_model import LatentCostRiskHead
from ..agents.risk_model import BayesianRiskHead
from ..envs.lattice_v2_runner import LatticeV2Runner
from .phase9_metrics import compute_transfer_summary, TransferSummary


def snapshot_learned_params(state) -> dict:
    """Snapshot the learned parameters that should transfer.

    COPIES: predictor weights / competence params
    DOES NOT COPY: episodic belief, position, inventory, runner state
    """
    snapshot = {}
    if hasattr(state, 'latent_predictor') and state.latent_predictor is not None:
        lp = state.latent_predictor
        snapshot['cost_w'] = lp.cost_head.w.copy()
        snapshot['cost_b'] = float(lp.cost_head.b)      # scalar
        snapshot['risk_w'] = lp.risk_head.w.copy()
        snapshot['risk_b'] = float(lp.risk_head.b)       # scalar
    elif hasattr(state, 'risk_head') and state.risk_head is not None:
        rh = state.risk_head
        snapshot['risk_w'] = rh.w.copy()
        snapshot['risk_b'] = float(rh.b)                 # scalar
    return snapshot


def apply_learned_params(state, snapshot: dict) -> None:
    """Apply snapshotted learned params to a fresh episode state.

    Only touches predictor weights. Does NOT touch belief/position/inventory.
    """
    if not snapshot:
        return
    if hasattr(state, 'latent_predictor') and state.latent_predictor is not None:
        lp = state.latent_predictor
        if 'cost_w' in snapshot:
            lp.cost_head.w[:] = snapshot['cost_w']
            lp.cost_head.b = snapshot['cost_b']          # scalar assign
        if 'risk_w' in snapshot:
            lp.risk_head.w[:] = snapshot['risk_w']
            lp.risk_head.b = snapshot['risk_b']           # scalar assign
    elif hasattr(state, 'risk_head') and state.risk_head is not None:
        rh = state.risk_head
        if 'risk_w' in snapshot:
            rh.w[:] = snapshot['risk_w']
            rh.b = snapshot['risk_b']                     # scalar assign


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
