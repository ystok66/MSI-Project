"""POMDP-style fixed risky-maze runtime.

This module implements the formal runtime missing from the old random-maze
prototype: hidden state, local stochastic observations, HP/time dynamics,
objective progression, fixed map support, and baseline modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

import numpy as np

from .fixed_loader import FixedRuntimeLayout, MazeTask
from .objectives import Coord, Inventory, LandmarkLearningEvent, Objective, ObjectiveEvent, ObjectiveState


class RuntimeAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


@dataclass(slots=True)
class MazeState:
    pos: Coord
    hp: int
    step_count: int
    time_limit: int
    has_key: bool
    collected_gems: set[Coord]
    completed_objective_index: int
    terminated: bool = False
    truncated: bool = False
    success: bool = False


@dataclass(slots=True)
class ObservedCell:
    coord: Coord
    visible_kind: str
    risk_vector: Optional[np.ndarray]
    is_walkable_observed: bool

    @property
    def kind(self) -> str:
        return self.visible_kind


@dataclass(slots=True)
class Observation:
    pos: Coord
    hp: int
    step_count: int
    time_limit: int
    view_radius: int
    visible_cells: dict[Coord, ObservedCell]
    has_key: bool
    objective_index: int
    current_objective: Optional[Objective]
    objective_sequence: list[Objective]
    has_gem_or_collected: Any
    map_shape: tuple[int, int]

    @property
    def visible(self) -> dict[Coord, ObservedCell]:
        return self.visible_cells

    @property
    def has_gem(self) -> bool:
        return bool(self.has_gem_or_collected)


@dataclass(slots=True)
class StepOutcome:
    action: Any
    from_pos: Coord
    to_pos: Coord
    attempted_pos: Coord
    moved: bool
    blocked: bool
    damage: int
    died: bool
    timeout: bool
    success: bool
    objective_event: ObjectiveEvent
    learning_events: tuple[LandmarkLearningEvent, ...] = ()
    trap_coord: Optional[Coord] = None
    trap_type: int = 0
    immortal_danger_event: bool = False
    danger_feature: Optional[np.ndarray] = None
    discovered_new_cells: int = 0
    repeated_step: bool = False
    no_info_step: bool = False


@dataclass(slots=True)
class POMDPConfigView:
    """Small adapter around the existing MazeScenarioConfig.

    The old project already has a config object.  This adapter reads attributes
    by name and provides safe defaults when fixed-runtime fields are absent.
    """

    risk_dim: int = 6
    n_safe_types: int = 3
    n_trap_types: int = 3
    cluster_std: float = 0.30
    obs_noise: float = 0.15
    view_radius: int = 2
    hp: int = 5

    @classmethod
    def from_any(cls, config: Any | None) -> "POMDPConfigView":
        if config is None:
            return cls()
        defaults = cls()
        return cls(
            risk_dim=int(getattr(config, "risk_dim", defaults.risk_dim)),
            n_safe_types=int(getattr(config, "n_safe_types", defaults.n_safe_types)),
            n_trap_types=int(getattr(config, "n_trap_types", defaults.n_trap_types)),
            cluster_std=float(getattr(config, "cluster_std", defaults.cluster_std)),
            obs_noise=float(getattr(config, "obs_noise", defaults.obs_noise)),
            view_radius=int(getattr(config, "view_radius", defaults.view_radius)),
            hp=int(getattr(config, "hp", defaults.hp)),
        )


_LATENT_WORLD_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, dict[Coord, np.ndarray]]] = {}


class RiskyMazePOMDPEnv:
    """A fixed-layout POMDP runtime with Gymnasium-like reset/step shape.

    Hidden true labels remain inside the runtime env/layout and are never copied into
    ``Observation``.  Observation risk vectors are stochastic draws around fixed
    latent per-cell vectors.
    """

    def __init__(
        self,
        *,
        layout: FixedRuntimeLayout,
        task: MazeTask,
        config: Any | None = None,
        seed: int | None = None,
        prototype_seed: int | None = None,
        baseline_mode: str = "mortal",
        phase: str = "teach",
    ) -> None:
        self.layout = layout
        self.task = task
        self.config = POMDPConfigView.from_any(config)
        self.baseline_mode = baseline_mode
        self.phase = phase
        self.prototype_seed = int(prototype_seed if prototype_seed is not None else seed if seed is not None else 0)
        self._prototype_rng = np.random.default_rng(self.prototype_seed)
        self._obs_rng = np.random.default_rng(seed)
        self._class_means, self.risk_latent_features = self._get_or_create_latent_world()
        attach_fixed_layout_cell_features = (
            bool(getattr(config, "attach_fixed_layout_cell_features", True)) if config is not None else True
        )
        if attach_fixed_layout_cell_features and getattr(self.layout, "cell_features", None) is not self.risk_latent_features:
            # Fixed layouts are immutable, so attach oracle latent features by
            # replacing the dataclass instance once at env construction time.
            self.layout = replace(self.layout, cell_features=self.risk_latent_features)
        self.trap_types = {coord: layout.trap_type(coord) for coord in layout.trap_cells()}
        self.objective_state = ObjectiveState(list(task.objectives), index=0)
        self.inventory = Inventory()
        self.state: MazeState | None = None
        self._ever_visible: set[Coord] = set()
        self._visited: dict[Coord, int] = {}
        self._last_distance_to_objective: Optional[int] = None

    def reset(self, task: MazeTask | None = None, seed: int | None = None) -> Observation:
        if task is not None:
            self.task = task
        if seed is not None:
            # Reset only the observation process; latent world remains fixed.
            self._obs_rng = np.random.default_rng(seed)
        self.objective_state = ObjectiveState(list(self.task.objectives), index=0)
        self.inventory = Inventory()
        self.state = MazeState(
            pos=self.task.start,
            hp=self.config.hp,
            step_count=0,
            time_limit=self.task.time_limit,
            has_key=False,
            collected_gems=set(),
            completed_objective_index=0,
        )
        self._ever_visible = set()
        self._visited = {self.task.start: 1}
        self._last_distance_to_objective = self._distance_to_current_objective(self.task.start)
        return self._observe()

    def step(self, action: Any) -> tuple[Observation, StepOutcome, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.state.terminated or self.state.truncated:
            obs = self._observe()
            outcome = StepOutcome(
                action=action,
                from_pos=self.state.pos,
                to_pos=self.state.pos,
                attempted_pos=self.state.pos,
                moved=False,
                blocked=False,
                damage=0,
                died=self.state.terminated and not self.state.success,
                timeout=self.state.truncated,
                success=self.state.success,
                objective_event=ObjectiveEvent(message="episode already done"),
            )
            return obs, outcome, self.state.terminated, self.state.truncated, {"already_done": True}

        old_pos = self.state.pos
        delta = _action_delta(action)
        attempted = (old_pos[0] + delta[0], old_pos[1] + delta[1])
        blocked = not self.layout.is_walkable(attempted)
        new_pos = old_pos if blocked else attempted
        moved = new_pos != old_pos

        self.state.step_count += 1

        damage = 0
        trap_coord: Optional[Coord] = None
        trap_type = 0
        danger_feature: Optional[np.ndarray] = None
        immortal_danger_event = False
        if moved and self.layout.is_trap(new_pos):
            trap_coord = new_pos
            trap_type = self.layout.trap_type(new_pos)
            latent = self.risk_latent_features.get(new_pos)
            danger_feature = None if latent is None else self._noisy_feature(latent)
            if self._ignore_trap_damage():
                immortal_danger_event = True
            else:
                damage = self.layout.trap_damage(new_pos)
                self.state.hp -= damage

        self.inventory.has_key = self.state.has_key
        self.inventory.collected_gems = self.state.collected_gems
        objective_event = self.objective_state.update(new_pos, self.layout, self.inventory)
        learning_events = self._landmark_learning_events(new_pos, objective_event)
        self.state.has_key = self.inventory.has_key
        self.state.collected_gems = self.inventory.collected_gems
        self.state.completed_objective_index = self.objective_state.index
        self.state.pos = new_pos

        died = self.state.hp <= 0 and not self._ignore_death()
        if died:
            self.state.terminated = True
        if objective_event.completed_all:
            self.state.success = True
            self.state.terminated = True
        timeout = self.state.step_count >= self.state.time_limit and not self._ignore_timeout()
        if timeout and not self.state.terminated:
            self.state.truncated = True

        repeated_step = self._visited.get(new_pos, 0) > 0
        self._visited[new_pos] = self._visited.get(new_pos, 0) + 1

        before_visible = set(self._ever_visible)
        obs = self._observe()
        discovered_new_cells = len(self._ever_visible - before_visible)

        new_distance = self._distance_to_current_objective(new_pos)
        closer = (
            self._last_distance_to_objective is not None
            and new_distance is not None
            and new_distance < self._last_distance_to_objective
        )
        self._last_distance_to_objective = new_distance
        no_info_step = discovered_new_cells == 0 and not objective_event.advanced and not closer

        outcome = StepOutcome(
            action=action,
            from_pos=old_pos,
            to_pos=new_pos,
            attempted_pos=attempted,
            moved=moved,
            blocked=blocked,
            damage=damage,
            died=died,
            timeout=timeout,
            success=self.state.success,
            objective_event=objective_event,
            learning_events=learning_events,
            trap_coord=trap_coord,
            trap_type=trap_type,
            immortal_danger_event=immortal_danger_event,
            danger_feature=None if danger_feature is None else np.array(danger_feature, copy=True),
            discovered_new_cells=discovered_new_cells,
            repeated_step=repeated_step,
            no_info_step=no_info_step,
        )
        info = {
            "baseline_mode": self.baseline_mode,
            "phase": self.phase,
            "objective_index": self.objective_state.index,
            "current_objective": self.objective_state.current(),
            "immortal_danger_event": immortal_danger_event,
        }
        return obs, outcome, self.state.terminated, self.state.truncated, info

    def current_observation(self) -> Observation:
        if self.state is None:
            raise RuntimeError("Call reset() before current_observation().")
        return self._observe()

    def _landmark_learning_events(
        self,
        coord: Coord,
        objective_event: ObjectiveEvent,
    ) -> tuple[LandmarkLearningEvent, ...]:
        ch = self.layout.char_at(coord)
        kind_map = {
            "K": "pickup",
            "D": "pass",
            "g": "collect_gem",
            "E": "exit",
        }
        kind = kind_map.get(ch)
        if kind is None:
            return ()
        objective_advanced = bool(
            objective_event.advanced
            and objective_event.completed_objective is not None
            and str(objective_event.completed_objective.kind) == kind
        )
        return (
            LandmarkLearningEvent(
                kind=kind,  # type: ignore[arg-type]
                coord=coord,
                objective_advanced=objective_advanced,
            ),
        )

    def true_is_trap(self, coord: Coord) -> bool:
        return self.layout.is_trap(coord)

    def true_trap_prefix(self, path: list[Coord], prefix_len: int) -> list[Coord]:
        return [p for p in path[:prefix_len] if self.layout.is_trap(p)]

    def features_for(self, coords: list[Coord], *, observed_noise: bool = False) -> list[np.ndarray]:
        feats: list[np.ndarray] = []
        for coord in coords:
            base = self.risk_latent_features.get(coord)
            if base is None:
                continue
            if observed_noise:
                feats.append(self._noisy_feature(base))
            else:
                feats.append(np.array(base, copy=True))
        return feats

    def risk_eval_dataset(self, *, observed_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for coord in self.layout.walkable_cells():
            feat = self.risk_latent_features[coord]
            xs.append(self._noisy_feature(feat) if observed_noise else np.array(feat, copy=True))
            ys.append(1 if self.layout.is_trap(coord) else 0)
        return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

    def _observe(self) -> Observation:
        assert self.state is not None
        visible: dict[Coord, ObservedCell] = {}
        r0, c0 = self.state.pos
        rad = self.config.view_radius
        for r in range(r0 - rad, r0 + rad + 1):
            for c in range(c0 - rad, c0 + rad + 1):
                coord = (r, c)
                if not self.layout.in_bounds(coord):
                    continue
                kind = self.layout.visible_kind(coord)
                walkable = self.layout.is_walkable(coord)
                risk_vector = None
                if walkable:
                    risk_vector = self._noisy_feature(self.risk_latent_features[coord])
                visible[coord] = ObservedCell(
                    coord=coord,
                    visible_kind=kind,
                    risk_vector=risk_vector,
                    is_walkable_observed=walkable,
                )
        self._ever_visible.update(visible.keys())
        return Observation(
            pos=self.state.pos,
            hp=self.state.hp,
            step_count=self.state.step_count,
            time_limit=self.state.time_limit,
            view_radius=self.config.view_radius,
            visible_cells=visible,
            has_key=self.state.has_key,
            objective_index=self.objective_state.index,
            current_objective=self.objective_state.current(),
            objective_sequence=self.task.objectives,
            has_gem_or_collected=frozenset(self.state.collected_gems),
            map_shape=self.layout.shape,
        )

    def _get_or_create_latent_world(self) -> tuple[np.ndarray, dict[Coord, np.ndarray]]:
        key = (
            tuple(self.layout.rows),
            self.prototype_seed,
            self.config.risk_dim,
            self.config.n_safe_types,
            self.config.n_trap_types,
            round(float(self.config.cluster_std), 8),
        )
        cached = _LATENT_WORLD_CACHE.get(key)
        if cached is not None:
            return cached
        class_means = self._make_class_means()
        risk_latent_features = self._make_latent_features(class_means)
        _LATENT_WORLD_CACHE[key] = (class_means, risk_latent_features)
        return class_means, risk_latent_features

    def _make_class_means(self) -> np.ndarray:
        n_classes = self.config.n_safe_types + self.config.n_trap_types
        means = self._prototype_rng.normal(0.0, 1.0, size=(n_classes, self.config.risk_dim))
        # Separate danger prototypes a little from safe prototypes to make the
        # concept learnable but still noisy.
        if n_classes >= 6:
            means[3:] += 2.0
        return means

    def _make_latent_features(self, class_means: np.ndarray | None = None) -> dict[Coord, np.ndarray]:
        means = self._class_means if class_means is None else class_means
        feats: dict[Coord, np.ndarray] = {}
        for coord in self.layout.walkable_cells():
            class_id = min(self.layout.latent_class_id(coord), len(means) - 1)
            feats[coord] = means[class_id] + self._prototype_rng.normal(
                0.0, self.config.cluster_std, size=(self.config.risk_dim,)
            )
        return feats

    def _noisy_feature(self, latent: np.ndarray) -> np.ndarray:
        return np.array(latent, copy=False) + self._obs_rng.normal(0.0, self.config.obs_noise, size=latent.shape)

    def _ignore_trap_damage(self) -> bool:
        return self.phase == "teach" and self.baseline_mode in {"immortal_warnlike", "immortal_no_timeout"}

    def _ignore_death(self) -> bool:
        return self.phase == "teach" and self.baseline_mode in {"immortal_warnlike", "immortal_no_timeout"}

    def _ignore_timeout(self) -> bool:
        return self.phase == "teach" and self.baseline_mode == "immortal_no_timeout"

    def _distance_to_current_objective(self, pos: Coord) -> Optional[int]:
        obj = self.objective_state.current()
        if obj is None:
            return None
        return abs(pos[0] - obj.coord[0]) + abs(pos[1] - obj.coord[1])


def _action_delta(action: Any) -> tuple[int, int]:
    name = action
    if not isinstance(name, str):
        name = getattr(action, "name", str(action))
    name = str(name).upper()
    if name.endswith(".UP") or name == "UP":
        return (-1, 0)
    if name.endswith(".DOWN") or name == "DOWN":
        return (1, 0)
    if name.endswith(".LEFT") or name == "LEFT":
        return (0, -1)
    if name.endswith(".RIGHT") or name == "RIGHT":
        return (0, 1)
    if name.endswith(".STAY") or name in {"STAY", "WAIT", "NOOP"}:
        return (0, 0)
    # Integer fallback compatible with RuntimeAction values.
    try:
        return {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0),
        }[int(action)]
    except Exception as exc:
        raise ValueError(f"Unsupported action: {action!r}") from exc
