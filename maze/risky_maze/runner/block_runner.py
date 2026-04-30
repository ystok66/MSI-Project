from __future__ import annotations

import numpy as np

from ..config import MazeScenarioConfig
from ..env import MazeEpisode, PrototypeBank, generate_layout, sample_starts
from ..learner import LearnerAgent
from ..tutor import NoTutor, build_tutor
from .episode_runner import run_episode
from .metrics import merge_episode_metrics


def run_block(
    cfg: MazeScenarioConfig,
    tutor_name: str = "inverse_warn",
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    bank = PrototypeBank.random(cfg, rng)
    teach_layout = generate_layout(cfg, rng, bank=bank)
    learner = LearnerAgent(cfg, seed=cfg.seed)
    tutor = build_tutor(tutor_name, cfg)

    teach_starts = sample_starts(
        teach_layout,
        cfg.teach_episodes,
        rng,
        forbid=(teach_layout.gem, teach_layout.exit),
    )
    teach_metrics = [
        run_episode(MazeEpisode(teach_layout, start=start, rng=rng), learner, tutor, learn=True)
        for start in teach_starts
    ]

    eval_tutor = NoTutor()
    same_map_starts = sample_starts(
        teach_layout,
        cfg.eval_same_map_episodes,
        rng,
        forbid=(teach_layout.gem, teach_layout.exit),
    )
    same_map_metrics = [
        run_episode(
            MazeEpisode(teach_layout, start=start, rng=rng),
            learner.clone(),
            eval_tutor,
            learn=False,
        )
        for start in same_map_starts
    ]

    transfer_layout = generate_layout(cfg, rng, bank=bank)
    new_map_starts = sample_starts(
        transfer_layout,
        cfg.eval_new_map_episodes,
        rng,
        forbid=(transfer_layout.gem, transfer_layout.exit),
    )
    new_map_metrics = [
        run_episode(
            MazeEpisode(transfer_layout, start=start, rng=rng),
            learner.clone_for_new_map(),
            eval_tutor,
            learn=False,
        )
        for start in new_map_starts
    ]

    results: dict[str, float] = {}
    for prefix, stats in (
        ("teach", merge_episode_metrics(teach_metrics)),
        ("eval_same_map", merge_episode_metrics(same_map_metrics)),
        ("eval_new_map", merge_episode_metrics(new_map_metrics)),
    ):
        for key, value in stats.items():
            results[f"{prefix}_{key}"] = value
    return results
