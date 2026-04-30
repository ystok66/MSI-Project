from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from risky_maze.env.fixed_loader import FixedRuntimeLayout, MazeTask
from risky_maze.env.objectives import Objective
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv, RuntimeAction
from risky_maze.learner.objective_agent import ObjectiveAwareLearner, SimpleMapMemory
from risky_maze.runner.fixed_episode_runner import run_fixed_episode
from risky_maze.tutor.compat import make_tutor_action
from risky_maze.tutor.context import TutorDecisionContext
from risky_maze.tutor.profiles import default_profiles
from risky_maze.tutor.rollout import CounterfactualRolloutEvaluator
from risky_maze.tutor.shadow import clone_from_snapshots
from risky_maze.tutor.world_model import cells_in_radius, feature_at, known_walkable, known_walls, mark_memory_observed, observed_vector


class _FeatureLayout:
    def __init__(self) -> None:
        self.height = 3
        self.width = 3
        self.cell_features = {(1, 1): np.array([1.0, 3.0], dtype=float)}

    def in_bounds(self, coord: tuple[int, int]) -> bool:
        row, col = coord
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, coord: tuple[int, int]) -> bool:
        return self.in_bounds(coord)

    def is_wall(self, coord: tuple[int, int]) -> bool:
        return not self.is_walkable(coord)


def _tiny_config(**overrides: object) -> SimpleNamespace:
    base = {
        "risk_dim": 4,
        "hp": 1,
        "view_radius": 1,
        "n_safe_types": 3,
        "n_trap_types": 3,
        "cluster_std": 0.0,
        "obs_noise": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FixedRuntimeRegressionTests(unittest.TestCase):
    def test_observed_vector_aggregates_list_values(self) -> None:
        memory = SimpleMapMemory(
            observed_vectors={
                (1, 1): [
                    np.array([0.0, 1.0], dtype=float),
                    np.array([2.0, 3.0], dtype=float),
                ]
            }
        )
        feat = observed_vector(memory, SimpleNamespace(), (1, 1), allow_oracle=False)
        self.assertIsNotNone(feat)
        self.assertTrue(np.allclose(feat, np.array([1.0, 2.0], dtype=float)))

    def test_mark_memory_observed_preserves_list_storage_for_simple_map_memory(self) -> None:
        layout = _FeatureLayout()
        memory = SimpleMapMemory()
        mark_memory_observed(memory, layout, (1, 1), allow_oracle_feature=True)
        self.assertIn((1, 1), memory.observed_vectors)
        self.assertIsInstance(memory.observed_vectors[(1, 1)], list)
        self.assertEqual(len(memory.observed_vectors[(1, 1)]), 1)
        self.assertTrue(np.allclose(memory.observed_vectors[(1, 1)][0], np.array([1.0, 3.0], dtype=float)))

    def test_simple_map_memory_clone_isolates_containers(self) -> None:
        arr = np.array([1.0, 2.0], dtype=float)
        memory = SimpleMapMemory(
            known_walkable={(1, 1)},
            observed_vectors={(1, 1): [arr]},
            visited_count={(1, 1): 1},
        )
        cloned = memory.clone()
        self.assertIs(cloned.observed_vectors[(1, 1)][0], arr)
        cloned.known_walkable.add((1, 2))
        cloned.observed_vectors[(1, 1)].append(np.array([3.0, 4.0], dtype=float))
        cloned.visited_count[(1, 1)] = 9
        self.assertNotIn((1, 2), memory.known_walkable)
        self.assertEqual(len(memory.observed_vectors[(1, 1)]), 1)
        self.assertEqual(memory.visited_count[(1, 1)], 1)

    def test_mean_vector_cache_invalidates_on_new_observation(self) -> None:
        memory = SimpleMapMemory(observed_vectors={(1, 1): [np.array([1.0, 3.0], dtype=float)]})
        first = memory.mean_vector((1, 1))
        self.assertTrue(np.allclose(first, np.array([1.0, 3.0], dtype=float)))
        obs = SimpleNamespace(
            pos=(1, 1),
            visible_cells={
                (1, 1): SimpleNamespace(
                    visible_kind="walkable",
                    is_walkable_observed=True,
                    risk_vector=np.array([3.0, 5.0], dtype=float),
                )
            },
        )
        memory.update_from_observation(obs)
        second = memory.mean_vector((1, 1))
        self.assertTrue(np.allclose(second, np.array([2.0, 4.0], dtype=float)))

    def test_known_coord_sets_use_fast_path_for_native_sets(self) -> None:
        memory = SimpleNamespace(known_walkable={(1, 1)}, known_walls={(0, 0)})
        self.assertIs(known_walkable(memory), memory.known_walkable)
        self.assertIs(known_walls(memory), memory.known_walls)

    def test_observe_is_idempotent_for_same_observation_object(self) -> None:
        learner = ObjectiveAwareLearner(risk_dim=4)
        obs = SimpleNamespace(
            pos=(1, 1),
            visible_cells={
                (1, 1): SimpleNamespace(
                    visible_kind="walkable",
                    is_walkable_observed=True,
                    risk_vector=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
                )
            },
        )
        learner.observe(obs)
        learner.observe(obs)
        self.assertEqual(len(learner.memory.observed_vectors[(1, 1)]), 1)

    def test_cells_in_radius_matches_square_observation_window(self) -> None:
        layout = SimpleNamespace(height=5, width=5)
        cells = cells_in_radius(layout, (2, 2), 2)
        self.assertEqual(len(cells), 25)
        self.assertIn((0, 0), cells)
        self.assertIn((4, 4), cells)

    def test_rollout_success_uses_fixed_objective_sequence(self) -> None:
        layout = FixedRuntimeLayout(rows=(".E",), name="tiny")
        state = SimpleNamespace(
            pos=(0, 0),
            has_gem=False,
            has_key=False,
            hp=1.0,
            step_count=0,
            time_limit=4,
            current_objective=Objective("exit", (0, 1)),
            objective_index=0,
            objective_sequence=[Objective("exit", (0, 1))],
        )
        memory = SimpleNamespace(
            known_walkable={(0, 0), (0, 1)},
            known_walls=set(),
            observed_vectors={(0, 0): np.array([0.0], dtype=float), (0, 1): np.array([0.0], dtype=float)},
            visited_count={(0, 0): 1},
        )
        belief = SimpleNamespace(danger_probability=lambda _: 0.0, update_labeled=lambda *_args, **_kwargs: None)
        shadow = clone_from_snapshots(memory, belief, default_profiles()[0], env_state=state, layout=layout)
        context = TutorDecisionContext(
            true_env_state=state,
            true_layout=layout,
            learner_memory_snapshot=memory,
            learner_risk_belief_snapshot=belief,
            phase="teach",
            remaining_time=4,
        )
        evaluator = CounterfactualRolloutEvaluator()
        value = evaluator.evaluate_candidate(make_tutor_action("WAIT"), context, [(shadow, 1.0)])
        self.assertGreater(value.p_success, 0.5)

    def test_immortal_no_timeout_restores_damage_and_death_in_eval(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="trap_eval")
        task = MazeTask(
            task_id="trap_eval",
            split="eval_same_map",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(),
            seed=0,
            prototype_seed=0,
            baseline_mode="immortal_no_timeout",
            phase="eval_same_map",
        )
        env.reset(seed=0)
        _obs, outcome, terminated, _truncated, _info = env.step(RuntimeAction.RIGHT)
        self.assertEqual(outcome.damage, 1)
        self.assertTrue(outcome.died)
        self.assertTrue(terminated)

    def test_immortal_warnlike_trap_step_updates_danger_once(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="trap_teach")
        task = MazeTask(
            task_id="trap_teach",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        learner = ObjectiveAwareLearner(risk_dim=4)
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(hp=3),
            seed=0,
            prototype_seed=0,
            baseline_mode="immortal_warnlike",
            phase="teach",
        )
        metrics = run_fixed_episode(env, learner, tutor_name="no_tutor", tutor_off=True, seed=0)
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.immortal_danger_events, 1)
        self.assertAlmostEqual(learner.risk_belief.danger_count, 2.0)

    def test_trap_outcome_uses_noisy_danger_feature_not_latent_oracle(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="trap_noise")
        task = MazeTask(
            task_id="trap_noise",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(obs_noise=0.75),
            seed=7,
            prototype_seed=11,
            baseline_mode="mortal",
            phase="teach",
        )
        env.reset(seed=7)
        _obs, outcome, _terminated, _truncated, _info = env.step(RuntimeAction.RIGHT)
        latent = env.risk_latent_features[(0, 1)]
        self.assertIsNotNone(outcome.danger_feature)
        self.assertFalse(np.allclose(outcome.danger_feature, latent))

    def test_env_reuses_latent_world_for_same_layout_and_seed(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="latent_cache")
        task = MazeTask(
            task_id="latent_cache",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env1 = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(),
            seed=0,
            prototype_seed=123,
            baseline_mode="mortal",
            phase="teach",
        )
        env2 = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(),
            seed=1,
            prototype_seed=123,
            baseline_mode="mortal",
            phase="teach",
        )
        self.assertIs(env1.risk_latent_features, env2.risk_latent_features)
        self.assertIs(env1._class_means, env2._class_means)

    def test_feature_at_reads_fixed_layout_cell_features_for_tutor_rollout(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="layout_features")
        task = MazeTask(
            task_id="layout_features",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(),
            seed=0,
            prototype_seed=5,
            baseline_mode="mortal",
            phase="teach",
        )
        feat = feature_at(env.layout, (0, 1))
        self.assertIsNotNone(feat)
        self.assertTrue(np.allclose(feat, env.risk_latent_features[(0, 1)]))

    def test_mark_memory_observed_uses_fixed_runtime_layout_features(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="layout_mark_features")
        task = MazeTask(
            task_id="layout_mark_features",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(),
            seed=0,
            prototype_seed=9,
            baseline_mode="mortal",
            phase="teach",
        )
        memory = SimpleMapMemory()
        mark_memory_observed(memory, env.layout, (0, 1), allow_oracle_feature=True)
        feat = memory.mean_vector((0, 1))
        self.assertIsNotNone(feat)
        self.assertTrue(np.allclose(feat, env.risk_latent_features[(0, 1)]))

    def test_fixed_layout_feature_attachment_can_be_disabled_for_d4_ablation(self) -> None:
        layout = FixedRuntimeLayout(rows=(".rE",), name="layout_features_off")
        task = MazeTask(
            task_id="layout_features_off",
            split="teach",
            start=(0, 0),
            objectives=[Objective("exit", (0, 2))],
            time_limit=5,
        )
        env = RiskyMazePOMDPEnv(
            layout=layout,
            task=task,
            config=_tiny_config(attach_fixed_layout_cell_features=False),
            seed=0,
            prototype_seed=13,
            baseline_mode="mortal",
            phase="teach",
        )
        self.assertIsNone(feature_at(env.layout, (0, 1)))


if __name__ == "__main__":
    unittest.main()
