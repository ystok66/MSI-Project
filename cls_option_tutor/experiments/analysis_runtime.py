"""Shared runtime helpers for narrow Phase 6 analysis scripts.

These helpers intentionally keep analysis scripts small and consistent:

- one place to build the current micro-benchmark config
- one place to run a teach block for a condition
- one place for small formatting helpers
"""

from __future__ import annotations

import copy
from statistics import mean
from typing import Iterable, Sequence

from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.experiments.condition_overrides import (
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.run_learning_increment_micro import (
    DATA_DIR,
    _apply_condition_overrides,
    make_cfg,
)
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent

DEFAULT_TASKS = ("000001", "000002", "000003", "000004")
DEFAULT_SEEDS = (42, 43, 44)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return mean(vals) if vals else 0.0


def rate(rows: Sequence[dict], key: str) -> float:
    if not rows:
        return 0.0
    return mean_or_zero([1.0 if bool(row.get(key, False)) else 0.0 for row in rows])


def run_teach_block(task_id: str, seed: int, condition: str, *, rho: float, generator: str):
    """Run one teach block for a single condition using current micro defaults."""
    cfg = make_cfg(
        n_sup=4,
        rho_assist=rho,
        generator_mode=generator,
        tutor_lg_mode="off",
        highlight_mode="diagnostic",
    )
    condition_eff = resolve_condition_alias(condition)
    cfg = _apply_condition_overrides(copy.deepcopy(cfg), condition_eff)

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=seed)
    support, _, grammar = env.adapter.load_task(task_id)
    init_block = env.reset_block(task_id, seed=seed)
    learner.init_block(init_block, grammar, support)

    if condition_eff.startswith("script_") or condition_eff.startswith("no_tutor_"):
        teach_cfg = copy.deepcopy(cfg)
        protocol = extract_scripted_protocol_name(condition_eff)
        runner = ScriptedProtocolRunner(cfg=teach_cfg, protocol=protocol)
        result = runner.run_block(
            OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
            learner,
            task_id,
            seed=seed,
        )
        return result.block

    teach_cfg = copy.deepcopy(cfg)
    tutor = SparseTutorAgent(cfg=teach_cfg)
    return tutor.run_block(
        OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
        learner,
        task_id,
        seed=seed,
    )
