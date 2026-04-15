"""
run_phase1.py — Main experiment runner for Phase 1.

Orchestrates: load task → support learning → teach queries → eval queries.
Supports parallel execution via ProcessPoolExecutor.

Usage:
    python -m cls_color_selection.experiments.run_phase1 --conditions sanity_basic
    python -m cls_color_selection.experiments.run_phase1 --conditions ALL
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is on path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cls_color_selection.config import FullConfig
from cls_color_selection.interfaces import EpisodeResult, QueryResult, TutorAction, Example
from cls_color_selection.constants import Outcome, TutorActionType
from cls_color_selection.environment.grammar_task_env import GrammarTaskEnv
from cls_color_selection.environment.generator import parse_task_file, DangerModel
from cls_color_selection.environment.state import QueryState
from cls_color_selection.environment.metrics import compute_metrics
from cls_color_selection.learner.cls_wrapper import CLSSequencePredictor
from cls_color_selection.learner.target_predictor import TargetPredictor
from cls_color_selection.learner.risk_belief import DangerTypeBelief
from cls_color_selection.learner.warning_update import warning_set_bayes_update
from cls_color_selection.learner.courage_update import courage_literal_update
from cls_color_selection.learner.feedback_update import FeedbackUpdater
from cls_color_selection.learner.policy import ColorSelectionPolicy
from cls_color_selection.learner.memory import QueryMemory
from cls_color_selection.tutor_api.dummy_tutor import (
    NoTutor, NoTutorImmortalWarnlike, NoTutorImmortalNoTimeout, OracleWarningTutor,
)
from cls_color_selection.experiments.registry import REGISTRY, apply_overrides
from cls_color_selection.experiments.seeds import generate_seeds


# ── Tutor class lookup ─────────────────────────────────────────

TUTOR_CLASSES = {
    'NoTutor': NoTutor,
    'NoTutorImmortalWarnlike': NoTutorImmortalWarnlike,
    'NoTutorImmortalNoTimeout': NoTutorImmortalNoTimeout,
    'OracleWarningTutor': OracleWarningTutor,
}


def get_tutor(condition_overrides: dict):
    """Instantiate the tutor from condition overrides."""
    cls_name = condition_overrides.get('_tutor_class', 'OracleWarningTutor')
    return TUTOR_CLASSES.get(cls_name, OracleWarningTutor)()


def is_immortal(tutor) -> bool:
    """Check if tutor mode is immortal (no death)."""
    return isinstance(tutor, (NoTutorImmortalWarnlike, NoTutorImmortalNoTimeout))


def has_timeout(tutor) -> bool:
    """Check if tutor mode has timeout."""
    return not isinstance(tutor, NoTutorImmortalNoTimeout)


# ── Single Episode Runner ─────────────────────────────────────

MAX_STEPS_PER_QUERY = 200  # hard safety limit


def run_single_query(
    env: GrammarTaskEnv,
    state: QueryState,
    policy: ColorSelectionPolicy,
    risk_belief: DangerTypeBelief,
    feedback_updater: FeedbackUpdater,
    predictor: CLSSequencePredictor,
    target_pred: TargetPredictor,
    tutor,
    memory: QueryMemory,
    rng: np.random.Generator,
    cfg: FullConfig,
    immortal: bool = False,
    enable_feedback: bool = True,
) -> QueryResult:
    """Run one query through the full SELECT/WARN/PLACE/CONFIRM loop."""

    step_count = 0
    empty_retry_count = 0
    MAX_EMPTY_RETRIES = 20  # force confirm after this many empty selections

    while not state.is_terminal and step_count < MAX_STEPS_PER_QUERY:
        step_count += 1

        # 1. Learner selects balls
        selected_indices = policy.select_set(state, risk_belief, rng)

        if not selected_indices:
            empty_retry_count += 1
            # After too many empty retries, force a confirm with whatever we have
            if empty_retry_count >= MAX_EMPTY_RETRIES and state.filled_count > 0:
                state, success, feedback = env.step_confirm(state, cfg.learner.feedback_mode)
                if success or state.is_terminal:
                    break
                empty_retry_count = 0
                continue
            # No good balls — retry immediately
            from cls_color_selection.environment.generator import generate_candidate_pool
            new_pool = generate_candidate_pool(
                grammar_colors=state.grammar_colors,
                target_output=state.target_output,
                n_candidates=cfg.env.n_candidates,
                danger_model=env.danger_model,
                cfg=cfg.env,
                rng=rng,
            )
            from cls_color_selection.environment.transition import retry_refresh
            state = retry_refresh(state, new_pool)
            memory.record_retry()
            continue
        else:
            empty_retry_count = 0  # reset on successful selection

        # Get selected balls
        from cls_color_selection.environment.transition import select_balls
        selected = select_balls(state, selected_indices)

        # 2. Tutor hook: WARNING / WAIT
        tutor_action = tutor.on_select(state, selected)

        # 3. Process
        state, step_info = env.step_select(
            state, selected_indices, tutor_action, immortal=immortal)

        # 4. Handle warning — update risk belief
        if tutor_action.action_type == TutorActionType.WARNING:
            X = np.stack([b.observed_vec for b in selected])
            warning_set_bayes_update(risk_belief, selected)
            memory.record_warning(selected_indices, X)
            # Courage check
            if policy.should_courage_trigger(state):
                courage_action = tutor.on_courage_check(state)
                if courage_action.action_type == TutorActionType.COURAGE:
                    courage_literal_update(
                        risk_belief, state.candidate_pool, state.needed_colors())
            continue

        # 5. Handle death
        if state.outcome == Outcome.DEATH:
            # Learn from death: the last selected ball was danger
            for b in selected:
                if b.is_danger:
                    risk_belief.update_from_death(b.observed_vec)
            break

        # 6. Safe placement — update risk belief with safe observations
        for b in selected:
            if not b.is_danger:
                risk_belief.update_from_safe_observation(b.observed_vec)
        memory.record_safe_placement([b.observed_vec for b in selected])

        # 7. Check confirm
        if policy.should_confirm(state):
            state, success, feedback = env.step_confirm(state, cfg.learner.feedback_mode)

            if success:
                break

            if state.outcome == Outcome.TIMEOUT:
                break

            # 8. Failed confirm — grammar feedback update
            if enable_feedback and cfg.learner.feedback_mode != 'none':
                submitted = [c if c is not None else '?' for c in feedback['submitted']]
                q_old, q_new = feedback_updater.apply_feedback(
                    predictor, state.query_words, submitted, feedback)
                # Invalidate target cache (grammar updated)
                target_pred.invalidate_cache(state.query_words)
                # Re-predict target (may have changed)
                new_target = target_pred.predict_target(state.query_words)
                if new_target != list(state.target_output):
                    state.target_output = new_target
                    state.completion = [None] * len(new_target)

            # Tutor: on_confirm_fail hook (Phase 1: WAIT)
            tutor.on_confirm_fail(state, feedback)

    # If we hit step limit without resolution, force timeout
    if not state.is_terminal:
        state.outcome = Outcome.TIMEOUT
        state.step_log.append({'event': 'step_limit_timeout', 'steps': step_count})

    return env.to_query_result(state)


def run_episode(
    task_path: str,
    task_id: str,
    seed: int,
    cfg: FullConfig,
    condition_overrides: dict,
) -> dict:
    """Run one full episode: support → teach → eval.

    Returns serializable dict of results.
    """
    rng = np.random.default_rng(seed)

    # Apply condition overrides
    if cfg is None:
        cfg = FullConfig()
    cfg = copy.deepcopy(cfg)
    apply_overrides(cfg, condition_overrides)

    # Initialize environment
    env = GrammarTaskEnv(cfg, rng)
    support, queries, grammar = env.load_task(task_path)

    # Initialize CLS grammar learner
    predictor = CLSSequencePredictor(cfg.learner)
    sub_support = support[:cfg.learner.n_sup]
    predictor.fit_support(sub_support)
    target_pred = TargetPredictor(predictor)

    # Initialize risk belief
    risk_belief = DangerTypeBelief(
        n_danger_types=cfg.env.n_danger_types,
        danger_dim=cfg.env.danger_dim,
        obs_sigma=cfg.env.obs_sigma,
        prior_safe=cfg.learner.risk_prior_safe,
    )
    # Initialize prototypes from danger model
    risk_belief.set_prototypes(
        env.danger_model.prototypes,
        np.ones_like(env.danger_model.prototypes) * cfg.env.cluster_sigma**2,
    )

    # Initialize policy and feedback
    policy = ColorSelectionPolicy(cfg.learner)
    feedback_updater = FeedbackUpdater(cfg.learner)
    tutor = get_tutor(condition_overrides)
    immortal = is_immortal(tutor)

    # Split queries into teach and eval
    n_teach = min(cfg.exp.n_teach_queries, len(queries))
    n_eval = min(cfg.exp.n_eval_queries, len(queries) - n_teach)
    teach_queries = queries[:n_teach]
    eval_queries = queries[n_teach:n_teach + n_eval]

    # ── Teach phase ──
    teach_results = []
    for qi, query in enumerate(teach_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=qi, target_output=y_star)
        memory = QueryMemory()

        result = run_single_query(
            env, state, policy, risk_belief, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            immortal=immortal, enable_feedback=True,
        )
        teach_results.append(result)

    # ── Eval phase (freeze grammar + risk) ──
    eval_results = []
    # Deep-copy risk_belief to freeze (no further updates)
    import copy as _copy
    frozen_risk = _copy.deepcopy(risk_belief)

    for qi, query in enumerate(eval_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(
            query, query_id=n_teach + qi, target_output=y_star)
        memory = QueryMemory()

        result = run_single_query(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            immortal=False,  # eval always mortal
            enable_feedback=False,  # eval: no learning
        )
        eval_results.append(result)

    # ── Compute metrics ──
    teach_metrics = compute_metrics(teach_results, prefix='Teach')
    eval_metrics = compute_metrics(eval_results, prefix='Eval')

    # ── Serialize results ──
    def _qr_to_dict(qr: QueryResult) -> dict:
        return {
            'query_id': qr.query_id,
            'query_words': qr.query_words,
            'target_output': qr.target_output,
            'ground_truth': qr.ground_truth,
            'outcome': qr.outcome.name,
            'confirm_count': qr.confirm_count,
            'retry_count': qr.retry_count,
            'death_count': qr.death_count,
            'danger_select_count': qr.danger_select_count,
            'stuck_retry_events': qr.stuck_retry_events,
        }

    return {
        'task_id': task_id,
        'seed': seed,
        'teach_metrics': teach_metrics,
        'eval_metrics': eval_metrics,
        'teach_details': [_qr_to_dict(r) for r in teach_results],
        'eval_details': [_qr_to_dict(r) for r in eval_results],
    }


# ── Worker function for parallel execution ─────────────────────

def _worker(args):
    """Picklable worker for ProcessPoolExecutor."""
    task_path, task_id, seed, cfg_dict, condition_name, condition_overrides = args
    # Reconstruct config
    cfg = FullConfig()
    from cls_color_selection.experiments.registry import apply_overrides as _ao
    # Apply base config values
    for section_name in ['env', 'learner', 'tutor', 'exp']:
        section_dict = cfg_dict.get(section_name, {})
        section = getattr(cfg, section_name)
        for k, v in section_dict.items():
            if hasattr(section, k):
                setattr(section, k, v)
    cfg.cls_data_dir = cfg_dict.get('cls_data_dir', '')

    try:
        result = run_episode(task_path, task_id, seed, cfg, condition_overrides)
        result['condition'] = condition_name
        result['status'] = 'ok'
    except Exception as e:
        result = {
            'task_id': task_id,
            'seed': seed,
            'condition': condition_name,
            'status': 'error',
            'error': str(e),
        }
    return result


# ── Main entry point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 1 experiments')
    parser.add_argument('--conditions', nargs='+', default=['sanity_basic'],
                        help='Condition names from registry (or ALL)')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Task IDs (default: all 20)')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=16)
    parser.add_argument('--output-dir', type=str,
                        default='cls_color_selection/results/phase1')
    parser.add_argument('--data-dir', type=str, default='')
    args = parser.parse_args()

    # Resolve data dir
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.normpath(os.path.join(
            _project_root, 'BASIC', 'cls_learner', 'data'))

    # Resolve conditions
    if 'ALL' in args.conditions:
        conditions = list(REGISTRY.keys())
    else:
        conditions = args.conditions

    # Resolve task IDs
    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = [f'{i:06d}' for i in range(1, 21)]

    # Generate seeds
    seeds = generate_seeds(42, args.n_seeds)

    # Build job list
    cfg = FullConfig()
    cfg.cls_data_dir = data_dir
    cfg_dict = {
        'env': {k: getattr(cfg.env, k) for k in cfg.env.__dataclass_fields__},
        'learner': {k: getattr(cfg.learner, k) for k in cfg.learner.__dataclass_fields__},
        'tutor': {k: getattr(cfg.tutor, k) for k in cfg.tutor.__dataclass_fields__},
        'exp': {k: getattr(cfg.exp, k) for k in cfg.exp.__dataclass_fields__},
        'cls_data_dir': data_dir,
    }

    jobs = []
    for cond_name in conditions:
        overrides = REGISTRY.get(cond_name, {})
        for task_id in task_ids:
            task_path = os.path.join(data_dir, f'{task_id}.txt')
            if not os.path.exists(task_path):
                print(f"  [SKIP] Task file not found: {task_path}")
                continue
            for seed in seeds:
                jobs.append((task_path, task_id, seed, cfg_dict, cond_name, overrides))

    print(f"Running {len(jobs)} jobs across {len(conditions)} conditions, "
          f"{len(task_ids)} tasks, {args.n_seeds} seeds")
    print(f"Workers: {args.n_workers}")

    # Run
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    t0 = time.time()

    if args.n_workers <= 1:
        for job in jobs:
            r = _worker(job)
            all_results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Save results
    output_path = os.path.join(args.output_dir, 'raw_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Print summary per condition
    summary_lines = []
    for cond_name in conditions:
        cond_results = [r for r in all_results
                        if r.get('condition') == cond_name and r.get('status') == 'ok']
        if not cond_results:
            summary_lines.append(f"{cond_name}: NO RESULTS")
            continue

        # Aggregate metrics
        teach_keys = [k for k in cond_results[0].get('teach_metrics', {}).keys()]
        eval_keys = [k for k in cond_results[0].get('eval_metrics', {}).keys()]

        teach_agg = {}
        for k in teach_keys:
            vals = [r['teach_metrics'][k] for r in cond_results if k in r.get('teach_metrics', {})]
            teach_agg[k] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else 'N/A'

        eval_agg = {}
        for k in eval_keys:
            vals = [r['eval_metrics'][k] for r in cond_results if k in r.get('eval_metrics', {})]
            eval_agg[k] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else 'N/A'

        summary_lines.append(f"\n=== {cond_name} ({len(cond_results)} runs) ===")
        for k, v in teach_agg.items():
            summary_lines.append(f"  {k}: {v}")
        for k, v in eval_agg.items():
            summary_lines.append(f"  {k}: {v}")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Phase 1 Results Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total jobs: {len(jobs)}, Elapsed: {elapsed:.1f}s\n\n")
        f.write(summary_text)

    print(f"\nResults saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
