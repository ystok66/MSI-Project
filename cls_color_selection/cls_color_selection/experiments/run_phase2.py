"""
run_phase2.py — Main experiment runner for Phase 2.

Episode lifecycle: support → observation → teaching → eval
Supports parallel execution via ProcessPoolExecutor.

Usage:
    python -m cls_color_selection.experiments.run_phase2 --conditions tutor_rule
    python -m cls_color_selection.experiments.run_phase2 --conditions ALL
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is on path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cls_color_selection.config import FullConfig
from cls_color_selection.interfaces import QueryResult, TutorAction, Example
from cls_color_selection.constants import Outcome, TutorActionType
from cls_color_selection.environment.grammar_task_env import GrammarTaskEnv
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
from cls_color_selection.tutor_api.tutor_state import TutorBelief
from cls_color_selection.tutor_api.observation import run_observation_phase
from cls_color_selection.tutor_api.belief_update import (
    initialize_belief_from_observation, update_belief_from_query_result,
)
from cls_color_selection.tutor_api.action_generators import apply_hint_to_state
from cls_color_selection.tutor_api.dummy_tutor import (
    NoTutor, NoTutorImmortalWarnlike, OracleWarningTutor,
)
from cls_color_selection.tutor_api.tutor_rule import RuleTutor
from cls_color_selection.tutor_api.tutor_proxy import ProxyTutor
from cls_color_selection.experiments.run_phase1 import run_single_query
from cls_color_selection.experiments.registry_phase2 import REGISTRY_P2, apply_overrides


# ── Tutor factory ──────────────────────────────────────────────

def create_tutor(cfg: FullConfig, belief: TutorBelief, risk_belief=None):
    """Create tutor instance from config."""
    mode = cfg.tutor.tutor_policy_mode

    if mode == 'rule':
        tutor = RuleTutor(cfg.tutor, belief=belief)
        return tutor
    elif mode == 'proxy':
        tutor = ProxyTutor(cfg.tutor, belief=belief, risk_belief=risk_belief)
        return tutor
    elif mode == 'none':
        return NoTutor()
    elif mode == 'oracle_warning':
        return OracleWarningTutor()
    else:
        return RuleTutor(cfg.tutor, belief=belief)


def is_immortal_tutor(tutor) -> bool:
    return isinstance(tutor, (NoTutorImmortalWarnlike,))


# ── Run single query with hint support ─────────────────────────

MAX_STEPS_PER_QUERY = 200


def run_query_with_tutor(
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
    belief: Optional[TutorBelief] = None,
    immortal: bool = False,
    enable_feedback: bool = True,
) -> Tuple[QueryResult, dict]:
    """Run one query with full Phase 2 tutor support including hints.

    Returns:
        (QueryResult, query_diagnostics)
    """
    from cls_color_selection.environment.generator import generate_candidate_pool
    from cls_color_selection.environment.transition import select_balls, retry_refresh

    step_count = 0
    empty_retry_count = 0
    MAX_EMPTY_RETRIES = 20
    diag = {
        'had_warning': False,
        'had_hint': False,
        'had_courage': False,
        'danger_encountered': False,
        'danger_avoided': False,
        'safe_skipped': False,
    }

    while not state.is_terminal and step_count < MAX_STEPS_PER_QUERY:
        step_count += 1

        # 1. Learner selects balls
        selected_indices = policy.select_set(state, risk_belief, rng)

        if not selected_indices:
            empty_retry_count += 1
            if empty_retry_count >= MAX_EMPTY_RETRIES and state.filled_count > 0:
                state, success, feedback = env.step_confirm(state, cfg.learner.feedback_mode)
                if success or state.is_terminal:
                    break
                # Post-confirm fail: tutor hint hook
                hint_action = tutor.on_confirm_fail(state, feedback)
                if hint_action.action_type == TutorActionType.HINT:
                    state = apply_hint_to_state(state, hint_action)
                    diag['had_hint'] = True
                empty_retry_count = 0
                continue

            diag['safe_skipped'] = True  # couldn't find usable balls
            new_pool = generate_candidate_pool(
                grammar_colors=state.grammar_colors,
                target_output=state.target_output,
                n_candidates=cfg.env.n_candidates,
                danger_model=env.danger_model,
                cfg=cfg.env,
                rng=rng,
            )
            state = retry_refresh(state, new_pool)
            memory.record_retry()
            continue
        else:
            empty_retry_count = 0

        # Get selected balls
        selected = select_balls(state, selected_indices)

        # 2. Tutor hook: on_select (WARNING / COURAGE / WAIT)
        tutor_action = tutor.on_select(state, selected)

        # 3. Process selection
        state, step_info = env.step_select(
            state, selected_indices, tutor_action, immortal=immortal)

        # 4. Handle warning
        if tutor_action.action_type == TutorActionType.WARNING:
            diag['had_warning'] = True
            diag['danger_encountered'] = True
            diag['danger_avoided'] = True
            X = np.stack([b.observed_vec for b in selected])
            warning_set_bayes_update(risk_belief, selected)
            memory.record_warning(selected_indices, X)
            # Courage check
            if policy.should_courage_trigger(state):
                courage_action = tutor.on_courage_check(state)
                if courage_action.action_type == TutorActionType.COURAGE:
                    diag['had_courage'] = True
                    courage_literal_update(
                        risk_belief, state.candidate_pool, state.needed_colors())
            continue

        # 5. Handle death
        if state.outcome == Outcome.DEATH:
            diag['danger_encountered'] = True
            for b in selected:
                if b.is_danger:
                    risk_belief.update_from_death(b.observed_vec)
            break

        # 6. Safe placement: update risk belief
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

            # Post-confirm fail: tutor hint hook
            hint_action = tutor.on_confirm_fail(state, feedback)
            if hint_action.action_type == TutorActionType.HINT:
                state = apply_hint_to_state(state, hint_action)
                diag['had_hint'] = True

            # Grammar feedback update
            if enable_feedback and cfg.learner.feedback_mode != 'none':
                submitted = [c if c is not None else '?' for c in feedback['submitted']]
                q_old, q_new = feedback_updater.apply_feedback(
                    predictor, state.query_words, submitted, feedback)
                target_pred.invalidate_cache(state.query_words)
                new_target = target_pred.predict_target(state.query_words)
                if new_target != list(state.target_output):
                    state.target_output = new_target
                    state.completion = [None] * len(new_target)

    # Force timeout at step limit
    if not state.is_terminal:
        state.outcome = Outcome.TIMEOUT
        state.step_log.append({'event': 'step_limit_timeout', 'steps': step_count})

    result = env.to_query_result(state)
    return result, diag


# ── Full episode runner ────────────────────────────────────────

def run_episode_phase2(
    task_path: str,
    task_id: str,
    seed: int,
    cfg: FullConfig,
    condition_overrides: dict,
) -> dict:
    """Run one full Phase 2 episode: support → observation → teach → eval."""
    rng = np.random.default_rng(seed)

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
    risk_belief.set_prototypes(
        env.danger_model.prototypes,
        np.ones_like(env.danger_model.prototypes) * cfg.env.cluster_sigma**2,
    )

    policy = ColorSelectionPolicy(cfg.learner)
    feedback_updater = FeedbackUpdater(cfg.learner)

    # ── Split queries: obs / teach / eval ──
    n_obs = cfg.exp.n_obs_queries if cfg.tutor.use_observation_phase else 0
    n_teach = cfg.exp.n_teach_queries
    n_eval = cfg.exp.n_eval_queries

    total_needed = n_obs + n_teach + n_eval
    available = len(queries)
    if total_needed > available:
        # Proportionally reduce
        scale = available / total_needed
        n_obs = int(n_obs * scale)
        n_teach = int(n_teach * scale)
        n_eval = available - n_obs - n_teach

    obs_queries = queries[:n_obs]
    teach_queries = queries[n_obs:n_obs + n_teach]
    eval_queries = queries[n_obs + n_teach:n_obs + n_teach + n_eval]

    # ── Initialize tutor belief ──
    belief = TutorBelief.from_config(cfg.belief)

    # ── Phase 1: Observation ──
    obs_summary = None
    if cfg.tutor.use_observation_phase and n_obs > 0:
        obs_summary = run_observation_phase(
            env, obs_queries, policy, risk_belief,
            feedback_updater, predictor, target_pred, rng, cfg,
        )
        initialize_belief_from_observation(belief, obs_summary, cfg.belief)

    # ── Create tutor ──
    tutor = create_tutor(cfg, belief, risk_belief)
    immortal = is_immortal_tutor(tutor)

    # ── Phase 2: Teaching ──
    teach_results = []
    for qi, query in enumerate(teach_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=n_obs + qi, target_output=y_star)
        memory = QueryMemory()

        result, diag = run_query_with_tutor(
            env, state, policy, risk_belief, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            belief=belief, immortal=immortal, enable_feedback=True,
        )
        teach_results.append(result)

        # Update belief from result
        update_belief_from_query_result(belief, result, **diag)

    # ── Phase 3: Eval (freeze) ──
    eval_results = []
    frozen_risk = copy.deepcopy(risk_belief)

    # Eval uses NoTutor (no intervention)
    eval_tutor = NoTutor()

    for qi, query in enumerate(eval_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(
            query, query_id=n_obs + n_teach + qi, target_output=y_star)
        memory = QueryMemory()

        result = run_single_query(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, eval_tutor, memory, rng, cfg,
            immortal=False, enable_feedback=False,
        )
        eval_results.append(result)

    # ── Compute metrics ──
    teach_metrics = compute_metrics(teach_results, prefix='Teach')
    eval_metrics = compute_metrics(eval_results, prefix='Eval')

    # ── Serialize ──
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

    result_dict = {
        'task_id': task_id,
        'seed': seed,
        'teach_metrics': teach_metrics,
        'eval_metrics': eval_metrics,
        'teach_details': [_qr_to_dict(r) for r in teach_results],
        'eval_details': [_qr_to_dict(r) for r in eval_results],
        'belief_summary': belief.summary_dict() if belief else {},
    }

    if obs_summary:
        result_dict['obs_summary'] = obs_summary.to_dict()

    return result_dict


# ── Worker ─────────────────────────────────────────────────────

def _worker(args):
    task_path, task_id, seed, cfg_dict, cond_name, cond_overrides = args
    cfg = FullConfig()
    for section_name in ['env', 'learner', 'tutor', 'belief', 'exp']:
        section_dict = cfg_dict.get(section_name, {})
        section = getattr(cfg, section_name, None)
        if section is None:
            continue
        for k, v in section_dict.items():
            if hasattr(section, k):
                setattr(section, k, v)
    cfg.cls_data_dir = cfg_dict.get('cls_data_dir', '')

    try:
        result = run_episode_phase2(task_path, task_id, seed, cfg, cond_overrides)
        result['condition'] = cond_name
        result['status'] = 'ok'
    except Exception as e:
        import traceback
        result = {
            'task_id': task_id, 'seed': seed,
            'condition': cond_name, 'status': 'error',
            'error': str(e), 'traceback': traceback.format_exc(),
        }
    return result


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 2 experiments')
    parser.add_argument('--conditions', nargs='+', default=['tutor_rule'])
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=16)
    parser.add_argument('--output-dir', type=str,
                        default='cls_color_selection/results/phase2')
    parser.add_argument('--data-dir', type=str, default='')
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.normpath(
        os.path.join(_project_root, 'BASIC', 'cls_learner', 'data'))

    if 'ALL' in args.conditions:
        conditions = list(REGISTRY_P2.keys())
    else:
        conditions = args.conditions

    task_ids = args.tasks or [f'{i:06d}' for i in range(1, 21)]

    from cls_color_selection.experiments.seeds import generate_seeds
    seeds = generate_seeds(42, args.n_seeds)

    # Build config dict for serialization
    cfg = FullConfig()
    cfg.cls_data_dir = data_dir
    cfg_dict = {}
    for section_name in ['env', 'learner', 'tutor', 'belief', 'exp']:
        section = getattr(cfg, section_name)
        cfg_dict[section_name] = {
            k: getattr(section, k)
            for k in section.__dataclass_fields__
        }
    cfg_dict['cls_data_dir'] = data_dir

    jobs = []
    for cond_name in conditions:
        overrides = REGISTRY_P2.get(cond_name, {})
        for task_id in task_ids:
            task_path = os.path.join(data_dir, f'{task_id}.txt')
            if not os.path.exists(task_path):
                print(f"  [SKIP] {task_path}")
                continue
            for seed in seeds:
                jobs.append((task_path, task_id, seed, cfg_dict, cond_name, overrides))

    print(f"Phase2: {len(jobs)} jobs, {len(conditions)} conditions, "
          f"{len(task_ids)} tasks, {args.n_seeds} seeds, workers={args.n_workers}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    t0 = time.time()

    if args.n_workers <= 1:
        for job in jobs:
            all_results.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for f in as_completed(futures):
                all_results.append(f.result())

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Save
    output_path = os.path.join(args.output_dir, 'raw_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + '\n')

    # Summary
    summary_lines = []
    for cond_name in conditions:
        cond_results = [r for r in all_results
                        if r.get('condition') == cond_name and r.get('status') == 'ok']
        if not cond_results:
            summary_lines.append(f"{cond_name}: NO RESULTS")
            errors = [r for r in all_results
                      if r.get('condition') == cond_name and r.get('status') == 'error']
            if errors:
                summary_lines.append(f"  ERRORS: {errors[0].get('error', '?')}")
            continue

        t_keys = list(cond_results[0].get('teach_metrics', {}).keys())
        e_keys = list(cond_results[0].get('eval_metrics', {}).keys())

        summary_lines.append(f"\n=== {cond_name} ({len(cond_results)} runs) ===")
        for k in t_keys:
            vals = [r['teach_metrics'].get(k, 0) for r in cond_results]
            summary_lines.append(f"  {k}: {np.mean(vals):.3f}±{np.std(vals):.3f}")
        for k in e_keys:
            vals = [r['eval_metrics'].get(k, 0) for r in cond_results]
            summary_lines.append(f"  {k}: {np.mean(vals):.3f}±{np.std(vals):.3f}")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Phase 2 Results Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Jobs: {len(jobs)}, Elapsed: {elapsed:.1f}s\n\n")
        f.write(summary_text)

    print(f"\nResults: {output_path}\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
