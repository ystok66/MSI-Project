"""
DEPRECATED — V0 environment. Replaced by LatticeV2Env / LatticeV2Runner.

PedagogicalGridEnv — Gymnasium environment.

External action = robot/teacher intervention.
Internal agent = bounded-rational learner (runs inside step).

The V2 system uses lattice_v2_env.py and lattice_v2_runner.py.
This file is kept for backward compatibility and reference only.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .map_generator import CellType, GridMap, generate_default_map
from ..agents.bounded_agent import BoundedRationalAgent
from ..teachers.interventions import InterventionType


class PedagogicalGridEnv(gym.Env):
    """
    Pedagogical Grid World environment.

    Action space: Discrete(4) — WAIT=0, WARN=1, UNLOCK_DOOR=2, DROP_SHIELD=3
    Observation: Dict with agent state, belief summary, grid info.
    """

    metadata = {"render_modes": ["ansi"]}

    # Map action indices to InterventionType
    ACTION_MAP = {
        0: InterventionType.WAIT,
        1: InterventionType.WARN,
        2: InterventionType.UNLOCK_DOOR,
        3: InterventionType.DROP_SHIELD,
        4: InterventionType.BLOCK_PATH,
    }

    def __init__(
        self,
        grid_map: Optional[GridMap] = None,
        # Env config
        max_steps: int = 60,
        initial_risk_budget: float = 1.0,
        shield_duration: int = 5,
        risk_trigger_prob: float = 0.3,
        risk_trigger_prob_shield: float = 0.02,
        # Agent config
        prior_cost_mean: float = 1.5,
        prior_cost_var: float = 4.0,
        prior_risk_mean: float = 0.1,
        prior_risk_var: float = 0.25,
        self_noise_var: float = 0.001,
        neighbor_noise_var: float = 1.0,
        neighbor_radius: int = 1,
        search_budget: int = 30,
        lambda_risk: float = 3.0,
        lambda_uncertainty: float = 0.5,
        # Warning param for WARN action
        warn_message: str = "CURRENT_PLAN_RISKY",
        # RNG
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.grid_map = grid_map or generate_default_map()
        self.H = self.grid_map.height
        self.W = self.grid_map.width

        # Env params
        self.max_steps = max_steps
        self.initial_risk_budget = initial_risk_budget
        self.shield_duration = shield_duration
        self.risk_trigger_prob = risk_trigger_prob
        self.risk_trigger_prob_shield = risk_trigger_prob_shield
        self.warn_message = warn_message
        self.render_mode = render_mode

        # Agent config (passed through)
        self._agent_cfg = dict(
            prior_cost_mean=prior_cost_mean,
            prior_cost_var=prior_cost_var,
            prior_risk_mean=prior_risk_mean,
            prior_risk_var=prior_risk_var,
            self_noise_var=self_noise_var,
            neighbor_noise_var=neighbor_noise_var,
            neighbor_radius=neighbor_radius,
            search_budget=search_budget,
            lambda_risk=lambda_risk,
            lambda_uncertainty=lambda_uncertainty,
        )

        # RNG
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

        # Create internal agent
        self.agent = BoundedRationalAgent(
            height=self.H,
            width=self.W,
            start_pos=self.grid_map.agent_start,
            rng=self.rng,
            **self._agent_cfg,
        )

        # Dynamic state
        self.step_count = 0
        self.risk_budget_left = initial_risk_budget
        self.object_pos: tuple[int, int] = self.grid_map.object_spawn
        self.object_picked = False
        self.object_delivered = False
        self.locked_doors: set[tuple[int, int]] = set(self.grid_map.door_positions)
        # Pedagogical blocking: cell → remaining TTL (steps)
        self.blocked_cells: dict[tuple[int, int], int] = {}
        self.block_target: tuple[int, int] | None = None  # set by teacher before step()

        # Spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(0, max(self.H, self.W), shape=(2,), dtype=np.int32),
            "object_pos": spaces.Box(0, max(self.H, self.W), shape=(2,), dtype=np.int32),
            "goal_pos": spaces.Box(0, max(self.H, self.W), shape=(2,), dtype=np.int32),
            "has_object": spaces.Discrete(2),
            "has_shield": spaces.Discrete(2),
            "time_left": spaces.Box(0, max_steps, shape=(1,), dtype=np.int32),
            "risk_budget_left": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "belief_cost_mean": spaces.Box(-10, 100, shape=(self.H, self.W), dtype=np.float32),
            "belief_risk_mean": spaces.Box(0, 1, shape=(self.H, self.W), dtype=np.float32),
            "belief_risk_var": spaces.Box(0, 10, shape=(self.H, self.W), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.risk_budget_left = self.initial_risk_budget
        self.object_pos = self.grid_map.object_spawn
        self.object_picked = False
        self.object_delivered = False
        self.locked_doors = set(self.grid_map.door_positions)
        self.blocked_cells = {}
        self.block_target = None

        # Reset cost map (re-lock doors)
        self._true_cost_dynamic = self.grid_map.true_cost.copy()
        for dr, dc in self.locked_doors:
            self._true_cost_dynamic[dr, dc] = np.inf

        self.agent.reset(
            start_pos=self.grid_map.agent_start,
            **{k: v for k, v in self._agent_cfg.items()
               if k.startswith("prior")},
        )
        self.agent.rng = self.rng

        # Initial observation for agent
        self.agent.observe_and_update(
            self._true_cost_dynamic, self.grid_map.true_risk,
        )

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Step the environment.

        Flow:
        1. Apply robot action (warn / unlock / shield / wait)
        2. Agent observes and updates belief
        3. Agent plans and executes one move
        4. Evaluate movement outcome (cost, risk, pickup, delivery)
        5. Return observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        intervention_type = self.ACTION_MAP[action]

        belief_before = self.agent.belief.copy()
        agent_pos_before = self.agent.pos

        # 1. Apply robot action
        warn_msg = ""
        if intervention_type == InterventionType.WARN:
            warn_msg = self.warn_message
            self.agent.process_teacher_action("WARN", warn_msg, self.shield_duration)
        elif intervention_type == InterventionType.UNLOCK_DOOR:
            if self.locked_doors:
                door = next(iter(self.locked_doors))
                self.locked_doors.discard(door)
                self._true_cost_dynamic[door[0], door[1]] = 1.0
                # Agent gets informed that door is open
                self.agent.belief.cost_mean[door[0], door[1]] = 1.0
                self.agent.belief.cost_var[door[0], door[1]] = 0.1
        elif intervention_type == InterventionType.DROP_SHIELD:
            self.agent.process_teacher_action("DROP_SHIELD", "", self.shield_duration)
        elif intervention_type == InterventionType.BLOCK_PATH:
            if self.block_target is not None:
                cell = self.block_target
                duration = 3  # default TTL
                self.blocked_cells[cell] = duration
                self.agent.process_teacher_action("BLOCK_PATH", "")
                # Do NOT change cost_mean or risk_mean
                self.block_target = None  # consumed

        # Tick blocked cell TTLs
        expired = [c for c, ttl in self.blocked_cells.items() if ttl <= 0]
        for c in expired:
            del self.blocked_cells[c]
        for c in self.blocked_cells:
            self.blocked_cells[c] -= 1

        # 2. Agent observes current surroundings
        self.agent.observe_and_update(
            self._true_cost_dynamic, self.grid_map.true_risk,
        )

        # 3. Agent plans and acts
        goal = self._current_goal()
        passable = self._passable_mask()
        agent_action, next_pos = self.agent.plan_and_act(goal, passable)

        # 4. Execute movement
        reward = 0.0
        terminated = False
        truncated = False

        if agent_action != "STAY" and self._is_passable(next_pos):
            self.agent.move_to(next_pos)

            # Cost
            r, c = next_pos
            move_cost = self._true_cost_dynamic[r, c]
            if not np.isinf(move_cost):
                reward -= move_cost * 0.01  # small step penalty

            # Risk check
            true_risk = self.grid_map.true_risk[r, c]
            if true_risk > 0:
                p = self.risk_trigger_prob_shield if self.agent.has_shield else self.risk_trigger_prob
                if self.rng.random() < true_risk * p / 0.3:  # normalize by base prob
                    damage = true_risk
                    self.risk_budget_left -= damage
                    reward -= damage * 0.5
                    if self.risk_budget_left <= 0:
                        terminated = True
                        reward -= 5.0  # death penalty

            # Pickup check
            if not self.object_picked and next_pos == self.object_pos:
                self.object_picked = True
                self.agent.has_object = True
                reward += 1.0

            # Delivery check
            if self.object_picked and next_pos == self.grid_map.target_pos:
                self.object_delivered = True
                terminated = True
                reward += 10.0

        # Truncation (timeout)
        time_left = self.max_steps - self.step_count
        if time_left <= 0 and not terminated:
            truncated = True
            reward -= 2.0

        # Build info
        info = self._get_info()
        info.update({
            "agent_action": agent_action,
            "agent_pos_before": list(agent_pos_before),
            "agent_pos_after": list(self.agent.pos),
            "robot_action": {"type": intervention_type.value, "param": warn_msg},
            "true_cost": float(self._true_cost_dynamic[self.agent.pos[0], self.agent.pos[1]])
                         if not np.isinf(self._true_cost_dynamic[self.agent.pos[0], self.agent.pos[1]]) else 999.0,
            "true_risk": float(self.grid_map.true_risk[self.agent.pos[0], self.agent.pos[1]]),
            "belief_snapshot": self.agent.belief.snapshot(),
            "belief_before_snapshot": belief_before.snapshot(),
        })

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_goal(self) -> tuple[int, int]:
        """Phase 1: go to object; Phase 2: go to target."""
        if not self.object_picked:
            return self.object_pos
        return self.grid_map.target_pos

    def _passable_mask(self) -> np.ndarray:
        """Boolean mask: True if cell is passable."""
        mask = np.ones((self.H, self.W), dtype=bool)
        for r in range(self.H):
            for c in range(self.W):
                if self.grid_map.cell_types[r, c] == CellType.WALL:
                    mask[r, c] = False
                if (r, c) in self.locked_doors:
                    mask[r, c] = False
                if (r, c) in self.blocked_cells:
                    mask[r, c] = False
        return mask

    def _is_passable(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if not (0 <= r < self.H and 0 <= c < self.W):
            return False
        if self.grid_map.cell_types[r, c] == CellType.WALL:
            return False
        if (r, c) in self.locked_doors:
            return False
        if (r, c) in self.blocked_cells:
            return False
        return True

    def _get_obs(self) -> dict:
        """Build observation dict for the teacher/robot."""
        time_left = max(0, self.max_steps - self.step_count)
        return {
            "agent_pos": np.array(self.agent.pos, dtype=np.int32),
            "object_pos": np.array(
                self.object_pos if not self.object_picked
                else self.agent.pos, dtype=np.int32
            ),
            "goal_pos": np.array(self.grid_map.target_pos, dtype=np.int32),
            "has_object": int(self.object_picked),
            "has_shield": int(self.agent.has_shield),
            "time_left": np.array([time_left], dtype=np.int32),
            "risk_budget_left": np.array([self.risk_budget_left], dtype=np.float32),
            "belief_cost_mean": self.agent.belief.cost_mean.astype(np.float32),
            "belief_risk_mean": self.agent.belief.risk_mean.astype(np.float32),
            "belief_risk_var": self.agent.belief.risk_var.astype(np.float32),
        }

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "time_left": self.max_steps - self.step_count,
            "risk_budget_left": self.risk_budget_left,
            "object_picked": self.object_picked,
            "object_delivered": self.object_delivered,
            "locked_doors": list(self.locked_doors),
            "agent_pos": self.agent.pos,
        }

    def render(self) -> Optional[str]:
        """ASCII rendering for debugging."""
        if self.render_mode != "ansi":
            return None

        lines = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                if (r, c) == self.agent.pos:
                    row.append("A")
                elif (r, c) == self.grid_map.target_pos:
                    row.append("T")
                elif not self.object_picked and (r, c) == self.object_pos:
                    row.append("O")
                elif (r, c) in self.locked_doors:
                    row.append("D")
                elif self.grid_map.cell_types[r, c] == CellType.WALL:
                    row.append("#")
                elif self.grid_map.cell_types[r, c] == CellType.HIGH_COST:
                    row.append("H")
                elif self.grid_map.cell_types[r, c] == CellType.RISKY:
                    row.append("!")
                else:
                    row.append(".")
            lines.append(" ".join(row))

        header = f"Step {self.step_count}/{self.max_steps}  Risk: {self.risk_budget_left:.2f}  Obj: {'✓' if self.object_picked else '✗'}"
        return header + "\n" + "\n".join(lines)
