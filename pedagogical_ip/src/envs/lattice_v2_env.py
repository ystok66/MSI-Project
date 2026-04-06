"""
Lattice V2 Environment — thin facade over LatticeV2Runner.

Provides a stable public API for the V2 experiment platform.
The env holds a reference to the runner's V2EpisodeState (no duplicate state).
step_full() is the semantic source of truth; step_teacher() + step_agent()
must compose to the exact same result.

Usage:
    env = LatticeV2Env()
    obs = env.reset(seed=42, tutor_mode="time_aware", closure_budget=2)
    while not env.done:
        result = env.step_full()
    metrics = env.get_metrics()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .lattice_v2_runner import LatticeV2Runner, V2EpisodeState
from ..agents.observation_model import observe_features, FeatureObservation
from ..agents.risk_model import BayesianRiskHead
from ..envs.map_generator import CellType


# ── Schema dataclasses ──────────────────────────────────────────────

@dataclass
class Observation:
    """Agent-visible observation at current timestep."""
    agent_pos: tuple[int, int]
    goal: tuple[int, int]
    t: int
    t_max: int
    feature_obs: list[np.ndarray]         # noisy d-dim features per observed cell
    obs_positions: list[tuple[int, int]]   # which cells were observed
    obs_variances: list[float]             # noise variance per cell
    passable: np.ndarray                   # (H, W) bool — current passability


@dataclass
class TeacherInfo:
    """Record of what the teacher did in one step."""
    actions_taken: list[str] = field(default_factory=list)
    gates_closed: list[tuple[int, int]] = field(default_factory=list)
    segments_warned: list[int] = field(default_factory=list)
    closures_this_step: int = 0
    warnings_this_step: int = 0


@dataclass
class StepResult:
    """Result of one agent step (or one full step cycle)."""
    observation: Observation
    done: bool
    survived: bool
    reached_goal: bool
    action_taken: str
    teacher_info: Optional[TeacherInfo]
    info: dict = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Full diagnostic state for logging/debugging."""
    agent_pos: tuple[int, int]
    goal: tuple[int, int]
    t: int
    t_max: int
    survived: bool
    reached_goal: bool
    done: bool
    closures: int          # backward alias for unlock_count
    warn_count: int
    warned_segments: set
    closed_gates: set
    steps: int
    risky_entered: int
    traps_hit: int

    @property
    def warnings_sent(self) -> int:
        """Backward alias for warn_count."""
        return self.warn_count


# ── Environment facade ──────────────────────────────────────────────

class LatticeV2Env:
    """Thin environment facade over LatticeV2Runner.

    Core invariant: the env holds a reference to the runner's V2EpisodeState.
    All episode logic lives in the runner — the env only wraps and exposes.
    """

    def __init__(self):
        self._runner = LatticeV2Runner()
        self._state: Optional[V2EpisodeState] = None

    def reset(self, seed: int, **config) -> Observation:
        """Start new episode. Returns initial observation."""
        self._state = self._runner.reset(seed, **config)
        return self._make_observation()

    def observe_agent(self) -> Observation:
        """Get agent's current observation without advancing time.

        Note: on the first call after reset() or step_agent(), this
        re-reads the same position. It does NOT call runner.observe()
        again — it just returns the last known features at agent_pos.
        """
        return self._make_observation()

    def step_teacher(self) -> TeacherInfo:
        """Execute teacher sub-step (door close / warning).

        Does NOT move the agent. Updates passable, belief_cost, warned state.
        """
        s = self._state
        closures_before = s.unlock_count
        warnings_before = s.warn_count
        warned_segs_before = set(s.warned_segments)

        # Observe first (teacher acts after observation)
        self._runner.observe(s)
        self._runner.apply_tutor(s)

        # Compute diff
        new_closures = s.unlock_count - closures_before
        new_warnings = s.warn_count - warnings_before
        new_warned = s.warned_segments - warned_segs_before

        gates_closed = []
        if s.tutor and new_closures > 0:
            # Get recently closed gates from tutor log
            for a in s.tutor.actions_log[-new_closures:]:
                if a.action == "close_risky_gate":
                    gates_closed.append(a.gate_cell)

        actions = []
        if new_closures > 0:
            actions.extend(["close_risky_gate"] * new_closures)
        if new_warnings > 0:
            actions.extend(["warn_only"] * new_warnings)

        return TeacherInfo(
            actions_taken=actions,
            gates_closed=gates_closed,
            segments_warned=list(new_warned),
            closures_this_step=new_closures,
            warnings_this_step=new_warnings,
        )

    def step_agent(self) -> StepResult:
        """Execute agent sub-step (plan → move → outcome).

        Assumes observe + teacher already happened this timestep.
        Returns step result with new observation.
        """
        s = self._state
        pos_before = s.agent_pos

        self._runner.plan_and_move(s)

        # Determine action taken
        dr = s.agent_pos[0] - pos_before[0]
        dc = s.agent_pos[1] - pos_before[1]
        action_map = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
        action = action_map.get((dr, dc), "STAY")

        return StepResult(
            observation=self._make_observation(),
            done=s.done,
            survived=s.survived,
            reached_goal=s.reached_goal,
            action_taken=action,
            teacher_info=None,
        )

    def step_full(self) -> StepResult:
        """Convenience: observe → step_teacher → step_agent in one call.

        This is the SEMANTIC SOURCE OF TRUTH. It calls runner.step()
        directly, guaranteeing identical behaviour to the original
        sweep script. step_teacher() + step_agent() must compose
        to the same result as this method.
        """
        s = self._state
        pos_before = s.agent_pos
        closures_before = s.unlock_count
        warnings_before = s.warn_count

        self._runner.step(s)

        # Action taken
        dr = s.agent_pos[0] - pos_before[0]
        dc = s.agent_pos[1] - pos_before[1]
        action_map = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
        action = action_map.get((dr, dc), "STAY")

        teacher_info = TeacherInfo(
            closures_this_step=s.unlock_count - closures_before,
            warnings_this_step=s.warn_count - warnings_before,
        )

        return StepResult(
            observation=self._make_observation(),
            done=s.done,
            survived=s.survived,
            reached_goal=s.reached_goal,
            action_taken=action,
            teacher_info=teacher_info,
        )

    def get_state(self) -> StateSnapshot:
        """Full state snapshot for logging/debugging."""
        s = self._state
        closed = set(s.tutor.closed_gates) if s.tutor else set()
        return StateSnapshot(
            agent_pos=s.agent_pos, goal=s.goal,
            t=s.t, t_max=s.t_max,
            survived=s.survived, reached_goal=s.reached_goal,
            done=s.done, closures=s.unlock_count,
            warn_count=s.warn_count,
            warned_segments=set(s.warned_segments),
            closed_gates=closed,
            steps=s.steps,
            risky_entered=s.risky_entered,
            traps_hit=s.traps_hit,
        )

    def get_metrics(self) -> dict:
        """Final episode metrics (compatible with sweep output)."""
        return LatticeV2Runner.get_metrics(self._state)

    @property
    def done(self) -> bool:
        return self._state.done if self._state else True

    @property
    def agent_pos(self) -> tuple[int, int]:
        return self._state.agent_pos

    # ── Internals ────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """Build Observation from current state. No runner calls."""
        s = self._state
        # Get the last observed features from feature_belief
        # (positions that have been observed)
        observed_positions = []
        feature_obs = []
        obs_variances = []
        for r in range(s.gridmap.height):
            for c in range(s.gridmap.width):
                if s.feature_belief.observed[r, c]:
                    observed_positions.append((r, c))
                    feature_obs.append(s.feature_belief.mean[r, c].copy())
                    obs_variances.append(float(s.feature_belief.var[r, c].mean()))

        return Observation(
            agent_pos=s.agent_pos,
            goal=s.goal,
            t=s.t,
            t_max=s.t_max,
            feature_obs=feature_obs,
            obs_positions=observed_positions,
            obs_variances=obs_variances,
            passable=s.passable.copy(),
        )
