"""
Pedagogical Regime-Shift Session (PRS) — Family 3 session wrapper.

PRS-2: Transfer Identifiability Repair.
Supports two WorldWeights modes:
  - episode_random: each episode gets fresh WorldWeights (v1 negative control)
  - session_shared: all episodes share session-level WorldWeights (transfer regime)

Session structure (4 blocks):
    Block A: Tutor-ON training (30 episodes)
    Block B: Tutor-OFF IID (15 episodes)
    Block C: Tutor-OFF topology shift (15 episodes)
    Block D: Tutor-OFF semantic shift (15 episodes)

Stateful: LatentCostRiskHead persists across episodes within a session,
representing the agent's learned world model. Grid-specific state
(FeatureBeliefMap, belief_cost, passable) resets per episode since maps change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
from copy import deepcopy

import numpy as np


# ── Families ────────────────────────────────────────────────────────
DTMB_FAMILY = "deep_tree_mixed_bottleneck_lattice"
GTET_FAMILY = "goal_preference_temptation_entanglement_lattice"

VALID_FAMILIES = {DTMB_FAMILY, GTET_FAMILY}


# ── Weight Modes ────────────────────────────────────────────────────

WeightMode = Literal["episode_random", "session_shared", "session_perturbed"]


# ── Shift configurations ────────────────────────────────────────────

def topology_shift_cfg(difficulty: str = "hard") -> dict:
    """Block C: topology shift via user_cfg.

    Changes structural metadata: route_count, merge layout, belt placement,
    commitment points. The generator must produce a structurally different
    map compared to standard params.
    """
    if difficulty == "hard":
        return {
            "route_count_override": 3,
            "merge_column_offset": 2,
            "belt_position_shift": 2,
            "commitment_depth_offset": -1,
            "branch_width_override": 2,
            "has_locked_fast_lane": True,
            "door_column_offset": -2,
            "_prs_shift": "topology",
        }
    else:
        return {
            "route_count_override": 3,
            "merge_column_offset": 1,
            "belt_position_shift": 1,
            "_prs_shift": "topology",
        }


def semantic_shift_cfg(difficulty: str = "hard") -> dict:
    """Block D: semantic/statistical shift via user_cfg.

    Changes cue order, lure strength, risk statistics, deadline.
    Does NOT change map topology.
    """
    if difficulty == "hard":
        return {
            "lure_strength": 0.95,
            "cue_order_swap": True,
            "risk_scale": 1.3,
            "deadline_slack": 0.85,
            "goal_cue_reliability": 0.7,
            "misleading_fraction": 0.3,
            "_prs_shift": "semantic",
        }
    else:
        return {
            "lure_strength": 0.85,
            "risk_scale": 1.15,
            "deadline_slack": 0.90,
            "_prs_shift": "semantic",
        }


# ── Session Config ──────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Configuration for a PRS session."""

    # Block sizes
    block_a_size: int = 30
    block_b_size: int = 15
    block_c_size: int = 15
    block_d_size: int = 15

    # Family mix: "mixed" | "dtmb_only" | "gtet_only"
    curriculum: str = "mixed"

    # Difficulty
    difficulty: str = "hard"

    # Tutor strategy: "selective" | "always_warn" | "no_tutor"
    tutor_strategy: str = "selective"

    # ── PRS-2: WorldWeights mode ──
    weight_mode: WeightMode = "session_shared"  # NEW: the core axis

    # Perturbation strength for session_perturbed mode
    perturb_rho: float = 0.15

    # Session state persistence
    persist_agent_memory: bool = True
    persist_tutor_state: bool = False

    # Topology/semantic shift difficulty
    shift_difficulty: str = "hard"

    # Session-level seed
    session_seed: int = 42


# ── Session State ───────────────────────────────────────────────────

@dataclass
class SessionState:
    """Minimal cross-episode state for transfer measurement."""

    # Agent's learned cost/risk model (persists across episodes)
    latent_predictor: Optional[object] = None

    # Session-level WorldWeights (None for episode_random)
    session_world_weights: Optional[object] = None

    # Session tracking
    episode_index: int = 0
    block_id: str = "A"
    block_episode_index: int = 0

    # Cumulative metrics per block
    block_results: dict = field(default_factory=lambda: {
        "A": [], "B": [], "C": [], "D": []
    })


# ── PRS Session Runner ──────────────────────────────────────────────

class PRSSession:
    """Multi-block session wrapper over DTMB-L v1 / GTET-L v1.

    PRS-2: supports session_shared WorldWeights for transfer identifiability.
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()

    def run_session(self) -> dict:
        """Run a full 4-block session. Returns session results dict."""
        from ..envs.lattice_v2_runner import LatticeV2Runner
        from ..agents.cost_risk_model import (
            LatentCostRiskHead, generate_world_weights, WorldWeights,
        )

        cfg = self.config
        rng = np.random.default_rng(cfg.session_seed)

        runner = LatticeV2Runner()
        state = SessionState()

        # Init persistent agent model
        if cfg.persist_agent_memory:
            state.latent_predictor = LatentCostRiskHead(d=4)

        # PRS-2: Generate session-level WorldWeights if needed
        if cfg.weight_mode in ("session_shared", "session_perturbed"):
            ww_rng = np.random.default_rng(cfg.session_seed * 7 + 3)
            state.session_world_weights = generate_world_weights(ww_rng, d=4)

        # Build episode schedule
        schedule = self._build_schedule(rng)

        # Run all blocks
        for block_id, episodes in schedule.items():
            state.block_id = block_id
            state.block_episode_index = 0

            # Tutor enabled logic
            if cfg.tutor_strategy == "no_tutor":
                tutor_enabled = False
            elif cfg.tutor_strategy == "always_warn":
                tutor_enabled = True
            else:  # selective
                tutor_enabled = (block_id == "A")

            for ep_spec in episodes:
                result = self._run_episode(
                    runner, state, ep_spec, tutor_enabled)
                state.block_results[block_id].append(result)
                state.episode_index += 1
                state.block_episode_index += 1

        # Compute session metrics
        from .prs_metrics import compute_session_metrics
        metrics = compute_session_metrics(state.block_results)
        metrics["config"] = {
            "curriculum": cfg.curriculum,
            "tutor_strategy": cfg.tutor_strategy,
            "persist_agent_memory": cfg.persist_agent_memory,
            "weight_mode": cfg.weight_mode,
            "session_seed": cfg.session_seed,
            "block_sizes": {
                "A": cfg.block_a_size, "B": cfg.block_b_size,
                "C": cfg.block_c_size, "D": cfg.block_d_size,
            },
        }
        metrics["block_results"] = state.block_results

        return metrics

    def _build_schedule(self, rng) -> dict:
        """Build episode schedule for all blocks."""
        cfg = self.config
        schedule = {}

        # Block A: tutor-on training
        a_eps = []
        for i in range(cfg.block_a_size):
            family = self._sample_family(rng, cfg.curriculum)
            seed = int(rng.integers(0, 100000))
            a_eps.append(EpisodeSpec(
                family=family, seed=seed, difficulty=cfg.difficulty,
                block_id="A", user_cfg=None))
        schedule["A"] = a_eps

        # Block B: tutor-off IID (same distribution as A)
        b_eps = []
        for i in range(cfg.block_b_size):
            family = self._sample_family(rng, cfg.curriculum)
            seed = int(rng.integers(0, 100000))
            b_eps.append(EpisodeSpec(
                family=family, seed=seed, difficulty=cfg.difficulty,
                block_id="B", user_cfg=None))
        schedule["B"] = b_eps

        # Block C: tutor-off topology shift
        c_cfg = topology_shift_cfg(cfg.shift_difficulty)
        c_eps = []
        for i in range(cfg.block_c_size):
            family = self._sample_family(rng, cfg.curriculum)
            seed = int(rng.integers(0, 100000))
            c_eps.append(EpisodeSpec(
                family=family, seed=seed, difficulty=cfg.difficulty,
                block_id="C", user_cfg=c_cfg))
        schedule["C"] = c_eps

        # Block D: tutor-off semantic shift
        d_cfg = semantic_shift_cfg(cfg.shift_difficulty)
        d_eps = []
        for i in range(cfg.block_d_size):
            family = self._sample_family(rng, cfg.curriculum)
            seed = int(rng.integers(0, 100000))
            d_eps.append(EpisodeSpec(
                family=family, seed=seed, difficulty=cfg.difficulty,
                block_id="D", user_cfg=d_cfg))
        schedule["D"] = d_eps

        return schedule

    def _sample_family(self, rng, curriculum: str) -> str:
        if curriculum == "dtmb_only":
            return DTMB_FAMILY
        elif curriculum == "gtet_only":
            return GTET_FAMILY
        elif curriculum == "mixed":
            return DTMB_FAMILY if rng.random() < 0.5 else GTET_FAMILY
        else:
            raise ValueError(f"Unknown curriculum: {curriculum}")

    def _get_world_weights_for_episode(
        self, state: SessionState, ep: 'EpisodeSpec'
    ) -> 'Optional[object]':
        """Return WorldWeights override for this episode, or None.

        - episode_random: return None (let generator create fresh weights)
        - session_shared: return session-level weights
        - session_perturbed: return session weights + perturbation for Block D
        """
        cfg = self.config
        if cfg.weight_mode == "episode_random":
            return None
        elif cfg.weight_mode == "session_shared":
            return state.session_world_weights
        elif cfg.weight_mode == "session_perturbed":
            if ep.block_id == "D":
                # Perturb weights for Block D
                return self._perturb_weights(state, ep)
            return state.session_world_weights
        return None

    def _perturb_weights(self, state: SessionState, ep: 'EpisodeSpec'):
        """Create mildly perturbed WorldWeights for Block D semantic shift."""
        from ..agents.cost_risk_model import generate_world_weights, WorldWeights
        cfg = self.config
        rho = cfg.perturb_rho

        # Generate a fresh random set of weights
        perturb_rng = np.random.default_rng(ep.seed * 13 + 7)
        fresh = generate_world_weights(perturb_rng, d=4)
        base = state.session_world_weights

        # Linear interpolation: (1-rho)*base + rho*fresh
        w_cost = (1 - rho) * base.w_cost + rho * fresh.w_cost
        b_cost = (1 - rho) * base.b_cost + rho * fresh.b_cost
        w_risk = (1 - rho) * base.w_risk + rho * fresh.w_risk
        b_risk = (1 - rho) * base.b_risk + rho * fresh.b_risk

        return WorldWeights(w_cost=w_cost, b_cost=b_cost,
                           w_risk=w_risk, b_risk=b_risk)

    def _run_episode(self, runner, state: SessionState,
                     ep: 'EpisodeSpec', tutor_enabled: bool) -> dict:
        """Run a single episode with optional state persistence."""
        cfg = self.config

        # Build user_cfg with WorldWeights override
        ep_user_cfg = dict(ep.user_cfg) if ep.user_cfg else {}
        ww_override = self._get_world_weights_for_episode(state, ep)
        if ww_override is not None:
            ep_user_cfg['world_weights_override'] = ww_override

        # Build reset kwargs
        reset_kwargs = dict(
            seed=ep.seed,
            difficulty=ep.difficulty,
            scenario_family=ep.family,
            latent_mode=True,
            patch_radius=2,
            prefix_horizon=5,
            belief_planning_mode=True,
            user_cfg=ep_user_cfg if ep_user_cfg else None,
        )

        if tutor_enabled:
            reset_kwargs.update(
                robot_belief_mode=True,
                intervention_family_mode=True,
                item_drop_enabled=True,
            )
        else:
            reset_kwargs.update(
                robot_belief_mode=False,
                intervention_family_mode=False,
                item_drop_enabled=False,
            )

        # GTET-specific: use P4 predictor
        if ep.family == GTET_FAMILY:
            reset_kwargs["predictor_mode"] = "P4"
            reset_kwargs["factor_mode"] = "G_THETA"  # Phase 1: z demoted

        # Inject persistent agent model
        if cfg.persist_agent_memory and state.latent_predictor is not None:
            reset_kwargs["latent_predictor"] = state.latent_predictor

        try:
            s = runner.reset(**reset_kwargs)

            while not s.done:
                s = runner.step(s)

            m = runner.get_metrics(s)

            # Persist updated agent model
            if cfg.persist_agent_memory and hasattr(s, 'latent_predictor'):
                lp = getattr(s, 'latent_predictor', None)
                if lp is not None:
                    state.latent_predictor = lp

            return {
                "seed": ep.seed,
                "family": ep.family,
                "block_id": ep.block_id,
                "block_ep_idx": state.block_episode_index,
                "session_ep_idx": state.episode_index,
                "survived": bool(m["survived"]),
                "reached_goal": bool(m["reached_goal"]),
                "success": bool(m["survived"] and m["reached_goal"]),
                "steps": int(m["steps"]),
                "t_max": int(m["t_max"]),
                "warnings": int(m.get("warnings", 0)),
                "tutor_enabled": tutor_enabled,
                "shift": ep.user_cfg.get("_prs_shift", "none") if ep.user_cfg else "none",
                "weight_mode": cfg.weight_mode,
                "predictor_n_updates": (
                    state.latent_predictor.n_updates
                    if state.latent_predictor else 0),
            }

        except Exception as e:
            return {
                "seed": ep.seed,
                "family": ep.family,
                "block_id": ep.block_id,
                "block_ep_idx": state.block_episode_index,
                "session_ep_idx": state.episode_index,
                "survived": False,
                "reached_goal": False,
                "success": False,
                "steps": 0, "t_max": 0, "warnings": 0,
                "tutor_enabled": tutor_enabled,
                "shift": ep.user_cfg.get("_prs_shift", "none") if ep.user_cfg else "none",
                "error": str(e),
                "weight_mode": cfg.weight_mode,
                "predictor_n_updates": 0,
            }


@dataclass
class EpisodeSpec:
    """Specification for a single episode within a session."""
    family: str
    seed: int
    difficulty: str
    block_id: str
    user_cfg: Optional[dict] = None
