from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import unittest

from risky_maze.tutor.context import TutorDecisionContext
from risky_maze.tutor.inverse_planner import FullInverseTutor, TutorConfig, WarningOnlyInverseTutor
from risky_maze.tutor.baselines import RiskThresholdWarnTutor, RiskThresholdWarnConfig


@dataclass
class FakeLayout:
    height: int = 5
    width: int = 8
    start: tuple[int, int] = (2, 1)
    gem: tuple[int, int] = (2, 5)
    exit: tuple[int, int] = (2, 6)
    walls: set[tuple[int, int]] = field(default_factory=set)
    traps: dict[tuple[int, int], float] = field(default_factory=lambda: {(2, 3): 1.0})
    cell_features: dict[tuple[int, int], tuple[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        open_cells = {(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (1, 2), (1, 3), (1, 4)}
        self.walls = {(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in open_cells}
        for c in open_cells:
            self.cell_features[c] = (0.2,) if c == (2, 3) else (0.0,)

    def in_bounds(self, coord: tuple[int, int]) -> bool:
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def is_walkable(self, coord: tuple[int, int]) -> bool:
        return self.in_bounds(coord) and coord not in self.walls

    def trap_damage(self, coord: tuple[int, int]) -> float:
        return float(self.traps.get(coord, 0.0))


@dataclass
class FakeMemory:
    known_walkable: set[tuple[int, int]]
    known_walls: set[tuple[int, int]] = field(default_factory=set)
    observed_vectors: dict[tuple[int, int], tuple[float]] = field(default_factory=dict)
    visited_count: dict[tuple[int, int], int] = field(default_factory=dict)


class FakeBelief:
    def __init__(self):
        self.warned = set()

    def danger_probability(self, x):
        key = tuple(x)
        if key in self.warned:
            return 0.95
        return float(x[0])

    def warning_update(self, features):
        for f in features:
            self.warned.add(tuple(f))

    def update_labeled(self, feature, label):
        if label:
            self.warned.add(tuple(feature))


def make_context(phase="teach"):
    layout = FakeLayout()
    open_cells = {c for c, f in layout.cell_features.items()}
    memory = FakeMemory(
        known_walkable=set(open_cells),
        known_walls=set(layout.walls),
        observed_vectors=dict(layout.cell_features),
        visited_count={(2, 1): 1},
    )
    state = SimpleNamespace(pos=(2, 1), has_gem=False, hp=1.0, step_count=0, time_limit=12, current_objective=SimpleNamespace(coord=layout.gem))
    obs = SimpleNamespace(visible_cells=set(open_cells), observed_vectors=dict(layout.cell_features), current_objective=SimpleNamespace(coord=layout.gem))
    policy = SimpleNamespace(action="RIGHT")
    return TutorDecisionContext(
        true_env_state=state,
        true_layout=layout,
        learner_observation=obs,
        learner_memory_snapshot=memory,
        learner_risk_belief_snapshot=FakeBelief(),
        learner_policy_snapshot=policy,
        history=None,
        phase=phase,
        remaining_time=12,
    )


class InverseTutorOverlayTests(unittest.TestCase):
    def test_eval_phase_disabled(self) -> None:
        tutor = FullInverseTutor()
        action = tutor.act(make_context(phase="eval"))
        self.assertEqual(action.kind, "WAIT")

    def test_warning_only_inverse_warns_on_deadly_predicted_prefix(self) -> None:
        cfg = TutorConfig(mode="warning_only", rollout_horizon=8)
        tutor = WarningOnlyInverseTutor(cfg)
        action = tutor.act(make_context())
        self.assertIn(action.kind, {"WARNING", "WAIT"})
        self.assertTrue(hasattr(action, "diagnostics"))
        self.assertGreaterEqual(action.diagnostics["candidate_count"], 1)
        self.assertEqual(action.kind, "WARNING")

    def test_risk_threshold_baseline_warns(self) -> None:
        tutor = RiskThresholdWarnTutor(RiskThresholdWarnConfig(threshold=0.1, prefix_len=4))
        action = tutor.act(make_context())
        self.assertEqual(action.kind, "WARNING")
        self.assertIn((2, 3), action.cells)

if __name__ == "__main__":
    unittest.main()
