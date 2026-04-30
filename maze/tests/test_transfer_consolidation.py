from __future__ import annotations

import unittest

from risky_maze.env.fixed_loader import build_layout_from_spec, build_task_from_spec, load_fixed_spec
from risky_maze.env.pomdp_episode import RiskyMazePOMDPEnv, RuntimeAction
from risky_maze.experiments.run_didactic_scaffold_ablation import scaffold_ablation_conditions
from risky_maze.experiments.run_didactic_tutor_suite import didactic_maps
from risky_maze.learner.objective_agent import ObjectiveAwareLearner


def _action_from_to(a: tuple[int, int], b: tuple[int, int]) -> RuntimeAction:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return RuntimeAction.UP
    if dr == 1 and dc == 0:
        return RuntimeAction.DOWN
    if dr == 0 and dc == -1:
        return RuntimeAction.LEFT
    if dr == 0 and dc == 1:
        return RuntimeAction.RIGHT
    return RuntimeAction.STAY


class TransferConsolidationTests(unittest.TestCase):
    def test_success_gated_commit_requires_teach_success(self) -> None:
        learner = ObjectiveAwareLearner(consolidation_mode="success_gated")
        path = [(1, 1), (1, 2), (1, 3)]
        learning_events = [{"kind": "pass", "coord": (1, 3), "objective_advanced": False}]

        learner.finalize_episode(
            phase="teach",
            success=False,
            path=path,
            learning_events=learning_events,
            assist_leakage=0.0,
        )
        self.assertEqual(learner.memory.transfer_graph.total_commits, 0)

        learner.finalize_episode(
            phase="teach",
            success=True,
            path=path,
            learning_events=learning_events,
            assist_leakage=0.0,
        )
        tg = learner.memory.transfer_graph
        self.assertEqual(tg.total_commits, 1)
        self.assertEqual(tg.successful_commits, 1)
        self.assertTrue(tg.route_node_confidence)
        self.assertIn((1, 3), tg.landmark_confidence)

    def test_always_commit_commits_even_on_failure(self) -> None:
        learner = ObjectiveAwareLearner(consolidation_mode="always_commit")
        learner.finalize_episode(
            phase="teach",
            success=False,
            path=[(1, 1), (1, 2)],
            learning_events=[{"kind": "pickup", "coord": (1, 2), "objective_advanced": False}],
            assist_leakage=1.5,
        )
        tg = learner.memory.transfer_graph
        self.assertEqual(tg.total_commits, 1)
        self.assertEqual(tg.successful_commits, 0)
        self.assertTrue(tg.route_node_confidence)

    def test_assist_discount_reduces_autonomy_credit_and_commit_strength(self) -> None:
        low_assist = ObjectiveAwareLearner(
            consolidation_mode="success_gated_assist_discounted",
            autonomy_assist_discount=0.35,
        )
        high_assist = ObjectiveAwareLearner(
            consolidation_mode="success_gated_assist_discounted",
            autonomy_assist_discount=0.35,
        )
        path = [(2, 2), (2, 3), (2, 4), (2, 5)]

        low_assist.finalize_episode(phase="teach", success=True, path=path, assist_leakage=0.0)
        high_assist.finalize_episode(phase="teach", success=True, path=path, assist_leakage=4.0)

        low_credit = low_assist.memory.transfer_graph.mean_autonomy_credit()
        high_credit = high_assist.memory.transfer_graph.mean_autonomy_credit()
        self.assertGreater(low_credit, high_credit)
        self.assertGreater(
            low_assist.memory.transfer_graph.route_graph_confidence_score(),
            high_assist.memory.transfer_graph.route_graph_confidence_score(),
        )

    def test_clone_for_eval_can_clear_long_term_memory(self) -> None:
        learner = ObjectiveAwareLearner(consolidation_mode="always_commit")
        learner.finalize_episode(
            phase="teach",
            success=True,
            path=[(3, 3), (3, 4), (3, 5)],
            learning_events=[{"kind": "collect_gem", "coord": (3, 5), "objective_advanced": True}],
            assist_leakage=0.0,
        )
        self.assertTrue(learner.memory.transfer_graph.route_node_confidence)

        cloned = learner.clone_for_eval(clear_long_term_memory=True)
        self.assertFalse(cloned.memory.transfer_graph.route_node_confidence)
        self.assertFalse(cloned.memory.transfer_graph.landmark_confidence)
        self.assertEqual(cloned.memory.transfer_graph.total_commits, 0)
        self.assertTrue(learner.memory.transfer_graph.route_node_confidence)

    def test_step_on_door_emits_learning_event_without_objective_advance(self) -> None:
        spec = load_fixed_spec("TutorAutonomyLoop_v1")
        layout = build_layout_from_spec(spec)
        task = build_task_from_spec(spec, "teach", "S2_T01_west_gem_to_ne_exit")
        env = RiskyMazePOMDPEnv(layout=layout, task=task, seed=0, prototype_seed=0)
        env.reset(seed=0)

        door = None
        neighbor = None
        for coord in layout.walkable_cells():
            if layout.char_at(coord) != "D":
                continue
            for nb in layout.neighbors4(coord):
                door = coord
                neighbor = nb
                break
            if door is not None:
                break
        self.assertIsNotNone(door)
        self.assertIsNotNone(neighbor)
        assert env.state is not None
        env.state.pos = neighbor  # type: ignore[assignment]
        env._visited = {neighbor: 1}  # type: ignore[attr-defined]
        env._last_distance_to_objective = env._distance_to_current_objective(neighbor)  # type: ignore[attr-defined]

        _, outcome, _, _, _ = env.step(_action_from_to(neighbor, door))  # type: ignore[arg-type]
        self.assertFalse(outcome.objective_event.advanced)
        self.assertTrue(any(evt.kind == "pass" and evt.coord == door for evt in outcome.learning_events))
        self.assertTrue(any(not evt.objective_advanced for evt in outcome.learning_events))

    def test_didactic_maps_enable_transfer_learning_defaults(self) -> None:
        cfg = didactic_maps()[0]["recommended_config"]
        self.assertEqual(cfg["learner_consolidation_mode"], "success_gated_assist_discounted")
        self.assertTrue(cfg["learner_enable_objective_learning_events"])
        self.assertTrue(cfg["learner_use_long_term_route_graph"])
        self.assertTrue(cfg["learner_use_landmark_graph"])

    def test_scaffold_ablation_conditions_cover_key_memory_switches(self) -> None:
        names = {row["condition_name"] for row in scaffold_ablation_conditions()}
        self.assertIn("random_frontier_scaffold_success_gated_assist_discounted", names)
        self.assertIn("minimal_scaffold_success_gated_assist_discounted", names)
        self.assertIn("minimal_scaffold_clear_eval_long_term_memory", names)
        self.assertIn("minimal_scaffold_no_route_graph", names)
        self.assertIn("minimal_scaffold_no_landmark_graph", names)
        self.assertIn("minimal_scaffold_no_objective_learning_events", names)


if __name__ == "__main__":
    unittest.main()
