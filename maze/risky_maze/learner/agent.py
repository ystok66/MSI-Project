from __future__ import annotations

import numpy as np

from ..config import MazeScenarioConfig
from ..core import ACTION_DELTAS, Action, Observation, PolicySnapshot, Pos, astar_path, manhattan
from .memory import MapMemory
from .risk_belief import GaussianRiskBelief


class LearnerAgent:
    def __init__(self, cfg: MazeScenarioConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.risk_belief = GaussianRiskBelief(
            risk_dim=cfg.risk_dim,
            n_trap_types=cfg.n_trap_types,
            seed=seed,
        )
        self.memory = MapMemory()

    def clone_for_new_map(self) -> "LearnerAgent":
        other = LearnerAgent(self.cfg)
        other.risk_belief = self.risk_belief.copy()
        other.memory = MapMemory()
        return other

    def clone(self) -> "LearnerAgent":
        other = LearnerAgent(self.cfg)
        other.risk_belief = self.risk_belief.copy()
        other.memory = self.memory.copy()
        return other

    def observe(self, obs: Observation) -> None:
        self.memory.observe(obs)

    def mark_transition(
        self,
        pos: Pos,
        observed_feature: np.ndarray,
        trap_type: int,
        learn: bool,
    ) -> None:
        self.memory.mark_visit(pos)
        self.memory.observed_vectors[pos] = observed_feature.copy()
        self.memory.confirm_trap(pos, trap_type)
        if learn:
            self.risk_belief.update_labeled(observed_feature, trap_type)

    def apply_warning(self, cells: list[Pos] | tuple[Pos, ...]) -> None:
        self.memory.add_warning_suspicion(cells)
        features: list[np.ndarray] = []
        for pos in cells:
            feat = self.memory.observed_feature(pos)
            if feat is not None:
                features.append(feat)
        self.risk_belief.warning_update(features)

    def target_for(self, obs: Observation) -> Pos:
        return obs.exit_pos if obs.has_gem else obs.gem_pos

    def policy_snapshot(self, obs: Observation) -> PolicySnapshot:
        self.observe(obs)
        target = self.target_for(obs)
        path = self._plan_path(obs.agent_pos, target)
        return PolicySnapshot(target=target, planned_path=tuple(path))

    def choose_action(self, obs: Observation) -> tuple[Action, PolicySnapshot]:
        snapshot = self.policy_snapshot(obs)
        if len(snapshot.planned_path) <= 1:
            return Action.STAY, snapshot
        nxt = snapshot.planned_path[1]
        dr = nxt[0] - obs.agent_pos[0]
        dc = nxt[1] - obs.agent_pos[1]
        for action, (adr, adc) in ACTION_DELTAS.items():
            if (dr, dc) == (adr, adc):
                return action, snapshot
        return Action.STAY, snapshot

    def _plan_path(self, start: Pos, goal: Pos) -> list[Pos]:
        h = max(
            max(pos[0] for pos in self.memory.seen_kind.keys() | {start, goal}) + 1,
            self.cfg.height,
        )
        w = max(
            max(pos[1] for pos in self.memory.seen_kind.keys() | {start, goal}) + 1,
            self.cfg.width,
        )

        def in_bounds(pos: Pos) -> bool:
            return 0 <= pos[0] < h and 0 <= pos[1] < w

        def neighbors(pos: Pos) -> list[Pos]:
            out: list[Pos] = []
            for dr, dc in ACTION_DELTAS.values():
                if (dr, dc) == (0, 0):
                    continue
                nxt = (pos[0] + dr, pos[1] + dc)
                if not in_bounds(nxt):
                    continue
                if self.memory.is_known_wall(nxt):
                    continue
                out.append(nxt)
            return out

        def cell_cost(pos: Pos) -> float:
            base = 1.0
            danger = self._danger_probability(pos)
            revisit = self.cfg.learner_revisit_penalty * self.memory.visited(pos)
            unknown = 0.0
            if not self.memory.is_known_walkable(pos):
                unknown = self.cfg.learner_unknown_penalty
            info_bonus = self.cfg.learner_info_bonus if self.memory.visited(pos) == 0 else 0.0
            cost = base + self.cfg.learner_risk_weight * danger + revisit + unknown - info_bonus
            return max(0.2, cost)

        return astar_path(start, goal, neighbors, cell_cost, manhattan)

    def _danger_probability(self, pos: Pos) -> float:
        if pos in self.memory.confirmed_traps:
            trap_type = self.memory.confirmed_traps[pos]
            return 0.0 if trap_type == 0 else 1.0
        feat = self.memory.observed_feature(pos)
        suspicion = self.memory.warning_suspicion.get(pos, 0.0)
        if feat is None:
            return max(0.35, suspicion)
        return max(self.risk_belief.danger_probability(feat), suspicion)
