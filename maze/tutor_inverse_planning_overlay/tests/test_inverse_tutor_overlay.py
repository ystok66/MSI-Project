from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from risky_maze.tutor.context import TutorDecisionContext
from risky_maze.tutor.inverse_planner import FullInverseTutor, TutorConfig, WarningOnlyInverseTutor
from risky_maze.tutor.baselines import RiskThresholdWarnTutor, RiskThresholdWarnConfig


@dataclass
class FakeLayout:
    width: int = 8
    height: int = 5
    start: tuple[int, int] = (1, 2)
    gem: tuple[int, int] = (5, 2)
    exit: tuple[int, int] = (6, 2)
    walls: set[tuple[int, int]] = field(default_factory=set)
    traps: dict[tuple[int, int], float] = field(default_factory=lambda: {(3, 2): 1.0})
    cell_features: dict[tuple[int, int], tuple[float]] = field(default_factory=dict)

    def __post_init__(self):
        open_cells = {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (2, 1), (3, 1), (4, 1)}
        self.walls = {(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in open_cells}
        for c in open_cells:
            self.cell_features[c] = (0.2,) if c == (3, 2) else (0.0,)


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
        if x in self.warned:
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
    memory = FakeMemory(known_walkable=set(open_cells), known_walls=set(layout.walls), observed_vectors=dict(layout.cell_features), visited_count={(1, 2): 1})
    state = SimpleNamespace(pos=(1, 2), has_gem=False, hp=1.0, step_count=0, time_limit=12)
    obs = SimpleNamespace(visible_cells=set(open_cells), observed_vectors=dict(layout.cell_features))
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


def test_eval_phase_disabled():
    tutor = FullInverseTutor()
    action = tutor.act(make_context(phase="eval"))
    assert action.kind == "WAIT"


def test_warning_only_inverse_warns_on_deadly_predicted_prefix():
    cfg = TutorConfig(mode="warning_only", rollout_horizon=8)
    tutor = WarningOnlyInverseTutor(cfg)
    action = tutor.act(make_context())
    assert action.kind in {"WARNING", "WAIT"}
    assert hasattr(action, "diagnostics")
    assert action.diagnostics["candidate_count"] >= 1
    # In this toy map warning should be valuable because it redirects the shadow
    # learner from the fatal shortcut to the known detour.
    assert action.kind == "WARNING"


def test_risk_threshold_baseline_warns():
    tutor = RiskThresholdWarnTutor(RiskThresholdWarnConfig(threshold=0.1, prefix_len=4))
    action = tutor.act(make_context())
    assert action.kind == "WARNING"
    assert (3, 2) in action.cells
