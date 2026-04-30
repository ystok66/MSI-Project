from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .baselines import run_condition_suite
from .config import OneHintConfig
from .experiment_presets import apply_named_presets, seed_bundle
from .hint_planner import select_hint
from .interfaces import ExperimentResult
from .inverse_particles import fit_inverse_posterior
from .learner_runner import bootstrap_env, build_base_learner, probe_teach_case_difficulty, run_observation_case
from .menu_builder import (
    build_eval_items_from_teach_menu,
    build_observation_cases,
    build_task_context,
    build_teach_case,
    sample_prelearn_examples,
)


@dataclass
class PreparedOneHintExperiment:
    task_id: str
    seed: int
    cfg: object
    rng: object
    context: object
    prelearn_examples: list
    observation_runs: list
    posterior: object
    teach_case: object
    eval_items: list
    base_learner: object
    seed_bundle: dict


def _teach_case_matches_difficulty(probe: dict, cfg) -> bool:
    mode = str(getattr(cfg, "menu_difficulty_mode", "default"))
    if mode != "rank_stratified":
        return True
    probe_mode = str(getattr(cfg, "teach_probe_mode", "initial_rank"))
    if probe_mode == "unlimited_tau":
        tau = probe.get("probe_no_tutor_unlimited_tau")
        if tau is None:
            return False
        return int(getattr(cfg, "target_no_tutor_unlimited_tau_min", 5)) <= int(tau) <= int(
            getattr(cfg, "target_no_tutor_unlimited_tau_max", 10)
        )
    rank = probe.get("probe_initial_correct_rank")
    if rank is None:
        return False
    return int(getattr(cfg, "target_initial_rank_min", 5)) <= int(rank) <= int(
        getattr(cfg, "target_initial_rank_max", 12)
    )


def _teach_case_probe_distance(probe: dict, cfg) -> float:
    probe_mode = str(getattr(cfg, "teach_probe_mode", "initial_rank"))
    if probe_mode == "unlimited_tau":
        tau = probe.get("probe_no_tutor_unlimited_tau")
        if tau is None:
            return float("inf")
        lo = int(getattr(cfg, "target_no_tutor_unlimited_tau_min", 5))
        hi = int(getattr(cfg, "target_no_tutor_unlimited_tau_max", 10))
        tau_val = int(tau)
        if lo <= tau_val <= hi:
            return 0.0
        return float(min(abs(tau_val - lo), abs(tau_val - hi)))
    rank = probe.get("probe_initial_correct_rank")
    if rank is None:
        return float("inf")
    lo = int(getattr(cfg, "target_initial_rank_min", 5))
    hi = int(getattr(cfg, "target_initial_rank_max", 12))
    rank_val = int(rank)
    if lo <= rank_val <= hi:
        return 0.0
    return float(min(abs(rank_val - lo), abs(rank_val - hi)))


def _annotated_teach_case(teach_case, probe: dict, *, matched: bool, trial_index: int):
    teach_case.metadata.update(
        {
            "difficulty_filter_mode": "rank_stratified",
            "difficulty_filter_matched": bool(matched),
            "difficulty_filter_trial": int(trial_index),
            **dict(probe),
        }
    )
    return teach_case


def _build_prepared_teach_case(
    context,
    cfg,
    rng_teach,
    base_learner,
    exclude_words,
):
    if str(getattr(cfg, "menu_difficulty_mode", "default")) != "rank_stratified":
        teach_case = build_teach_case(context, cfg, rng_teach, exclude_words=exclude_words)
        teach_case.metadata.update(
            probe_teach_case_difficulty(
                base_learner=base_learner,
                context=context,
                teach_case=teach_case,
                probe_mode=str(getattr(cfg, "teach_probe_mode", "initial_rank")),
            )
        )
        teach_case.metadata["difficulty_filter_mode"] = "default"
        teach_case.metadata["difficulty_filter_matched"] = True
        teach_case.metadata["difficulty_filter_trial"] = 1
        return teach_case

    trials = max(1, int(getattr(cfg, "teach_menu_build_trials", 24)))
    best_case = None
    best_probe = None
    best_distance = float("inf")
    probe_mode = str(getattr(cfg, "teach_probe_mode", "initial_rank"))
    for trial_idx in range(1, trials + 1):
        teach_case = build_teach_case(context, cfg, rng_teach, exclude_words=exclude_words)
        probe = probe_teach_case_difficulty(
            base_learner=base_learner,
            context=context,
            teach_case=teach_case,
            probe_mode=probe_mode,
        )
        distance = _teach_case_probe_distance(probe, cfg)
        if distance < best_distance:
            best_distance = distance
            best_case = teach_case
            best_probe = probe
        if _teach_case_matches_difficulty(probe, cfg):
            return _annotated_teach_case(
                teach_case,
                probe,
                matched=True,
                trial_index=trial_idx,
            )

    assert best_case is not None and best_probe is not None
    return _annotated_teach_case(
        best_case,
        best_probe,
        matched=False,
        trial_index=trials,
    )


def prepare_one_hint_experiment(
    task_id: str,
    cfg: Optional[OneHintConfig] = None,
    seed: Optional[int] = None,
) -> PreparedOneHintExperiment:
    cfg = copy.deepcopy(cfg or OneHintConfig())
    if seed is not None:
        cfg.seed = int(seed)
    apply_named_presets(cfg)
    stage_seeds = seed_bundle(cfg.seed, cfg)
    rng_prelearn = np.random.default_rng(stage_seeds["prelearn"])
    rng_obs = np.random.default_rng(stage_seeds["obs"])
    rng_teach = np.random.default_rng(stage_seeds["teach"])
    rng_eval = np.random.default_rng(stage_seeds["eval"])

    cfg._env = bootstrap_env(cfg, task_id, stage_seeds["context"])
    context = build_task_context(task_id, cfg, stage_seeds["context"])
    prelearn_examples = sample_prelearn_examples(context, cfg, rng_prelearn)
    if not prelearn_examples:
        raise ValueError(f"Failed to sample prelearn examples for task {task_id}")
    used_words = {tuple(ex.words) for ex in prelearn_examples}

    base_learner = build_base_learner(context, prelearn_examples, seed=stage_seeds["learner"])
    observation_cases = build_observation_cases(context, cfg, rng_obs, exclude_words=used_words)
    used_words.update(tuple(case.example.words) for case in observation_cases)
    observation_runs = [
        run_observation_case(base_learner, context, case)
        for case in observation_cases
    ]

    posterior = fit_inverse_posterior(
        context=context,
        prelearn_examples=prelearn_examples,
        observation_runs=observation_runs,
        cfg=cfg,
    )

    teach_case = _build_prepared_teach_case(
        context=context,
        cfg=cfg,
        rng_teach=rng_teach,
        base_learner=base_learner,
        exclude_words=used_words,
    )
    eval_items = build_eval_items_from_teach_menu(context, teach_case, cfg, rng_eval)
    return PreparedOneHintExperiment(
        task_id=task_id,
        seed=cfg.seed,
        cfg=cfg,
        rng=np.random.default_rng(stage_seeds["plan"]),
        context=context,
        prelearn_examples=prelearn_examples,
        observation_runs=observation_runs,
        posterior=posterior,
        teach_case=teach_case,
        eval_items=eval_items,
        base_learner=base_learner,
        seed_bundle=stage_seeds,
    )


def finalize_prepared_experiment(
    prepared: PreparedOneHintExperiment,
    plan,
    rng: Optional[np.random.Generator] = None,
) -> ExperimentResult:
    rng = rng or np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed)))
    conditions = run_condition_suite(
        base_learner=prepared.base_learner,
        context=prepared.context,
        teach_case=prepared.teach_case,
        eval_items=prepared.eval_items,
        plan=plan,
        cfg=prepared.cfg,
        rng=rng,
    )

    return ExperimentResult(
        task_id=prepared.task_id,
        seed=prepared.seed,
        prelearn_examples=prepared.prelearn_examples,
        observation_examples=[run.case.example for run in prepared.observation_runs],
        teach_example=prepared.teach_case.example,
        teach_case_metadata=dict(getattr(prepared.teach_case, "metadata", {}) or {}),
        eval_items=prepared.eval_items,
        posterior=prepared.posterior.summary,
        plan=plan,
        conditions=conditions,
    )


def run_one_hint_experiment(
    task_id: str,
    cfg: Optional[OneHintConfig] = None,
    seed: Optional[int] = None,
) -> ExperimentResult:
    prepared = prepare_one_hint_experiment(task_id=task_id, cfg=cfg, seed=seed)
    plan = select_hint(
        posterior=prepared.posterior,
        context=prepared.context,
        teach_case=prepared.teach_case,
        eval_items=prepared.eval_items,
        cfg=prepared.cfg,
        rng=np.random.default_rng(int(prepared.seed_bundle.get("plan", prepared.seed))),
    )
    return finalize_prepared_experiment(
        prepared,
        plan,
        rng=np.random.default_rng(int(prepared.seed_bundle.get("baseline", prepared.seed))),
    )
