from __future__ import annotations

from ..core import Observation, PolicySnapshot, TutorAction
from ..env import MazeEpisode
from ..learner import LearnerAgent


class TutorPolicy:
    name = "base"

    def maybe_intervene(
        self,
        episode: MazeEpisode,
        learner: LearnerAgent,
        obs: Observation,
        snapshot: PolicySnapshot,
    ) -> TutorAction:
        raise NotImplementedError
