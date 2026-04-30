from __future__ import annotations

from ..env import MazeEpisode
from ..learner import LearnerAgent
from ..tutor import TutorPolicy
from .metrics import EpisodeMetrics


def run_episode(
    episode: MazeEpisode,
    learner: LearnerAgent,
    tutor: TutorPolicy,
    learn: bool,
) -> EpisodeMetrics:
    warnings = 0
    visited_this_episode: set[tuple[int, int]] = set()
    repeated_steps = 0

    while True:
        obs = episode.observe()
        learner.observe(obs)
        action, snapshot = learner.choose_action(obs)
        assist = tutor.maybe_intervene(episode, learner, obs, snapshot)
        if assist.kind == "WARNING":
            learner.apply_warning(assist.cells)
            warnings += 1
            action, snapshot = learner.choose_action(obs)

        outcome = episode.step(action)
        learner.observe(outcome.observation)
        feat = learner.memory.observed_feature(outcome.moved_to)
        if feat is None:
            feat = episode.layout.feature_at(outcome.moved_to)
        learner.mark_transition(
            pos=outcome.moved_to,
            observed_feature=feat,
            trap_type=outcome.trap_type,
            learn=learn,
        )

        if outcome.moved_to in visited_this_episode:
            repeated_steps += 1
        visited_this_episode.add(outcome.moved_to)

        if outcome.success or outcome.died or outcome.timeout:
            break

    return EpisodeMetrics(
        success=float(outcome.success),
        died=float(outcome.died),
        timeout=float(outcome.timeout),
        steps=float(episode.step_count),
        damage=float(episode.total_damage),
        warnings=float(warnings),
        discovered_cells=float(len(visited_this_episode)),
        repeated_steps=float(repeated_steps),
    )
