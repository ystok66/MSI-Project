from __future__ import annotations

import numpy as np

from ..core import ACTION_DELTAS, Action, CellKind, Observation, Pos, StepOutcome, VisibleCell
from .layout import MazeLayout


class MazeEpisode:
    def __init__(
        self,
        layout: MazeLayout,
        start: Pos | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.layout = layout
        self.rng = rng or np.random.default_rng()
        self.agent_pos = start or layout.start
        self.has_gem = False
        self.hp = layout.cfg.hp
        self.step_count = 0
        self.total_damage = 0
        self.success = False
        self.died = False
        self.timeout = False
        path_budget = (
            layout.shortest_path_length(self.agent_pos, layout.gem)
            + layout.shortest_path_length(layout.gem, layout.exit)
        )
        self.time_limit = max(6, int(np.ceil(layout.cfg.time_limit_scale * path_budget)))

    def _visible_cells(self) -> tuple[VisibleCell, ...]:
        out: list[VisibleCell] = []
        r0, c0 = self.agent_pos
        vr = self.layout.cfg.view_radius
        for r in range(r0 - vr, r0 + vr + 1):
            for c in range(c0 - vr, c0 + vr + 1):
                pos = (r, c)
                if not self.layout.in_bounds(pos):
                    continue
                kind = self.layout.kind_at(pos)
                if kind == CellKind.WALL:
                    out.append(
                        VisibleCell(
                            pos=pos,
                            kind=kind,
                            walkable=False,
                            observed_vec=None,
                        )
                    )
                    continue
                observed_vec = self.layout.bank.observe(self.layout.feature_at(pos), self.rng)
                out.append(
                    VisibleCell(
                        pos=pos,
                        kind=kind,
                        walkable=True,
                        observed_vec=observed_vec,
                    )
                )
        return tuple(out)

    def observe(self) -> Observation:
        return Observation(
            agent_pos=self.agent_pos,
            gem_pos=self.layout.gem,
            exit_pos=self.layout.exit,
            has_gem=self.has_gem,
            hp=self.hp,
            time_remaining=max(0, self.time_limit - self.step_count),
            visible_cells=self._visible_cells(),
        )

    def step(self, action: Action) -> StepOutcome:
        if self.success or self.died or self.timeout:
            return StepOutcome(
                observation=self.observe(),
                moved_to=self.agent_pos,
                damage=0,
                trap_type=0,
                success=self.success,
                died=self.died,
                timeout=self.timeout,
            )

        dr, dc = ACTION_DELTAS[action]
        candidate = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        if self.layout.in_bounds(candidate) and self.layout.is_walkable(candidate):
            self.agent_pos = candidate

        self.step_count += 1
        if self.agent_pos == self.layout.gem:
            self.has_gem = True

        trap_type = self.layout.trap_type_at(self.agent_pos)
        damage = trap_type
        if damage > 0:
            self.hp -= damage
            self.total_damage += damage

        if self.hp <= 0:
            self.died = True
        elif self.has_gem and self.agent_pos == self.layout.exit:
            self.success = True
        elif self.step_count >= self.time_limit:
            self.timeout = True

        return StepOutcome(
            observation=self.observe(),
            moved_to=self.agent_pos,
            damage=damage,
            trap_type=trap_type,
            success=self.success,
            died=self.died,
            timeout=self.timeout,
        )
