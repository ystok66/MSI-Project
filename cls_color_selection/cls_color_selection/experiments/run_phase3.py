"""
run_phase3.py — Main experiment runner for Phase 3 (FIXED).

Fixes from diagnostic:
  1. Query source: supports txt_only, txt_resample, generated, hybrid
  2. Shadow grammar sync: shadow gets feedback update at same checkpoints as real
  3. Divergence: multi-probe, JS divergence over beam, parameter-level metrics
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
from cls_color_selection.environment.query_generator import (
    generate_episode_queries, resample_queries,
)
from cls_color_selection.learner.cls_wrapper import CLSSequencePredictor
from cls_color_selection.learner.target_predictor import TargetPredictor
from cls_color_selection.learner.risk_belief import DangerTypeBelief
from cls_color_selection.learner.warning_update import warning_set_bayes_update
from cls_color_selection.learner.feedback_update import FeedbackUpdater
from cls_color_selection.learner.policy import ColorSelectionPolicy
from cls_color_selection.learner.memory import QueryMemory
from cls_color_selection.tutor_api.tutor_state import TutorBelief
from cls_color_selection.tutor_api.observation import run_observation_phase
from cls_color_selection.tutor_api.belief_update import (
    initialize_belief_from_observation, update_belief_from_query_result,
)
from cls_color_selection.tutor_api.tutor_shadow import ShadowTutor
from cls_color_selection.tutor_api.shadow_update import shadow_feedback_update
from cls_color_selection.tutor_api.joint_debug import (
    JointDebugLog, DivergenceRecord, CounterfactualRecord,
)
from cls_color_selection.tutor_api.joint_debug_v2 import (
    compute_full_divergence_v2,
)
from cls_color_selection.tutor_api.dummy_tutor import (
    NoTutor, NoTutorImmortalWarnlike, NoTutorImmortalNoTimeout, OracleWarningTutor,
)
from cls_color_selection.tutor_api.tutor_rule import RuleTutor
from cls_color_selection.tutor_api.tutor_proxy import ProxyTutor
from cls_color_selection.tutor_api.tutor_behavioral import BehavioralTutor
from cls_color_selection.tutor_api.action_generators import apply_hint_to_state
from cls_color_selection.experiments.registry_phase3 import REGISTRY_P3, apply_overrides


# ── Helper: create tutor ──────────────────────────────────────

def create_tutor_p3(cfg, belief, risk_belief):
    mode = cfg.tutor.tutor_policy_mode
    if mode == 'rule':
        return RuleTutor(cfg.tutor, belief=belief)
    elif mode == 'proxy':
        return ProxyTutor(cfg.tutor, belief=belief, risk_belief=risk_belief)
    elif mode == 'behavioral':
        # T3: will be fully initialized in episode runner (needs support)
        return BehavioralTutor(cfg.tutor, cfg.learner, belief=belief)
    elif mode == 'none':
        return NoTutor()
    elif mode == 'immortal_warnlike':
        return NoTutorImmortalWarnlike()
    elif mode == 'immortal_no_timeout':
        return NoTutorImmortalNoTimeout()
    elif mode == 'oracle_warning':
        return OracleWarningTutor()
    return RuleTutor(cfg.tutor, belief=belief)


def is_immortal_tutor_p3(tutor) -> bool:
    return isinstance(tutor, (NoTutorImmortalWarnlike, NoTutorImmortalNoTimeout))


# ── Main query loop with shadow sync ──────────────────────────

MAX_STEPS = 200


def run_query_with_shadow_sync(
    env, state, policy, risk_belief, feedback_updater,
    predictor, target_pred, tutor, memory, rng, cfg,
    belief=None, immortal=False, enable_feedback=True,
):
    """Run one query with proper shadow grammar sync.

    Key fix: after every feedback update on real grammar,
    also apply shadow_feedback_update on the tutor's shadow.
    """
    from cls_color_selection.environment.generator import generate_candidate_pool
    from cls_color_selection.environment.transition import select_balls, retry_refresh

    step_count = 0
    empty_retry_count = 0
    MAX_EMPTY = 20
    diag = {
        'had_warning': False, 'had_hint': False, 'had_courage': False,
        'danger_encountered': False, 'danger_avoided': False, 'safe_skipped': False,
        'n_feedback_updates': 0,
    }

    is_shadow = isinstance(tutor, ShadowTutor)
    is_behavioral = isinstance(tutor, BehavioralTutor)

    # T3: signal query start
    if is_behavioral:
        tutor.observe_query_start()

    while not state.is_terminal and step_count < MAX_STEPS:
        step_count += 1

        selected_indices = policy.select_set(state, risk_belief, rng)

        if not selected_indices:
            empty_retry_count += 1
            if empty_retry_count >= MAX_EMPTY and state.filled_count > 0:
                state, success, feedback = env.step_confirm(state, cfg.learner.feedback_mode)
                if success or state.is_terminal:
                    break
                hint_action = tutor.on_confirm_fail(state, feedback)
                if hint_action.action_type == TutorActionType.HINT:
                    state = apply_hint_to_state(state, hint_action)
                    diag['had_hint'] = True
                # Grammar feedback on confirm fail
                if enable_feedback and cfg.learner.feedback_mode != 'none':
                    _do_grammar_update(
                        state, feedback, predictor, target_pred,
                        feedback_updater, tutor, is_shadow, cfg, diag)
                empty_retry_count = 0
                continue

            diag['safe_skipped'] = True
            new_pool = generate_candidate_pool(
                grammar_colors=state.grammar_colors,
                target_output=state.target_output,
                n_candidates=cfg.env.n_candidates,
                danger_model=env.danger_model,
                cfg=cfg.env, rng=rng,
            )
            state = retry_refresh(state, new_pool)
            memory.record_retry()
            if is_behavioral:
                tutor.observe_retry()
            continue
        else:
            empty_retry_count = 0

        selected = select_balls(state, selected_indices)
        tutor_action = tutor.on_select(state, selected)
        state, step_info = env.step_select(
            state, selected_indices, tutor_action, immortal=immortal)

        if tutor_action.action_type == TutorActionType.WARNING:
            diag['had_warning'] = True
            diag['danger_encountered'] = True
            diag['danger_avoided'] = True
            warning_set_bayes_update(risk_belief, selected)
            X = np.stack([b.observed_vec for b in selected])
            memory.record_warning(selected_indices, X)
            if policy.should_courage_trigger(state):
                from cls_color_selection.learner.courage_update import courage_literal_update
                courage_action = tutor.on_courage_check(state)
                if courage_action.action_type == TutorActionType.COURAGE:
                    diag['had_courage'] = True
                    courage_literal_update(
                        risk_belief, state.candidate_pool, state.needed_colors())
            continue

        if state.outcome == Outcome.DEATH:
            diag['danger_encountered'] = True
            for b in selected:
                if b.is_danger:
                    risk_belief.update_from_death(b.observed_vec)
            if is_behavioral:
                tutor.observe_death()
            break

        for b in selected:
            if not b.is_danger:
                risk_belief.update_from_safe_observation(b.observed_vec)
        memory.record_safe_placement([b.observed_vec for b in selected])

        if policy.should_confirm(state):
            state, success, feedback = env.step_confirm(state, cfg.learner.feedback_mode)

            if success:
                if is_behavioral:
                    tutor.observe_confirm_success()
                break
            if state.outcome == Outcome.TIMEOUT:
                break

            hint_action = tutor.on_confirm_fail(state, feedback)
            if hint_action.action_type == TutorActionType.HINT:
                state = apply_hint_to_state(state, hint_action)
                diag['had_hint'] = True

            # ★ Grammar feedback update (Rule G1: confirm fail → update both)
            if enable_feedback and cfg.learner.feedback_mode != 'none':
                _do_grammar_update(
                    state, feedback, predictor, target_pred,
                    feedback_updater, tutor, is_shadow, cfg, diag)

    if not state.is_terminal:
        state.outcome = Outcome.TIMEOUT
        state.step_log.append({'event': 'step_limit_timeout', 'steps': step_count})

    return env.to_query_result(state), diag


def _do_grammar_update(
    state, feedback, predictor, target_pred,
    feedback_updater, tutor, is_shadow, cfg, diag,
):
    """Apply grammar feedback to real learner AND shadow (if shadow tutor).

    Rule G1: confirm fail → update both real and shadow grammar.
    Shadow uses its OWN beam posterior, not the real learner's copy.
    """
    submitted = [c if c is not None else '?' for c in feedback['submitted']]

    # Real learner update
    q_old, q_new = feedback_updater.apply_feedback(
        predictor, state.query_words, submitted, feedback)
    target_pred.invalidate_cache(state.query_words)
    new_target = target_pred.predict_target(state.query_words)
    if new_target != list(state.target_output):
        state.target_output = new_target
        state.completion = [None] * len(new_target)
    diag['n_feedback_updates'] = diag.get('n_feedback_updates', 0) + 1

    # ★ Shadow grammar update (same checkpoint, own posterior)
    if is_shadow and hasattr(tutor, 'shadow') and tutor.shadow is not None:
        shadow_feedback_update(
            tutor.shadow, state.query_words, submitted,
            feedback, cfg.learner)


# ── Query split with source mode ──────────────────────────────

def split_queries(
    support, queries, grammar, cfg, rng,
):
    """Split queries based on query_source_mode."""
    n_obs = cfg.exp.n_obs_queries if cfg.tutor.use_observation_phase else 0
    n_teach = cfg.exp.n_teach_queries
    n_eval = cfg.exp.n_eval_queries
    mode = cfg.exp.query_source_mode
    tags = []

    if mode == 'txt_only':
        # Old behavior: proportional scaling if overflow
        available = len(queries)
        total = n_obs + n_teach + n_eval
        if total > available:
            scale = available / total
            n_obs = int(n_obs * scale)
            n_teach = int(n_teach * scale)
            n_eval = available - n_obs - n_teach
        obs_q = queries[:n_obs]
        teach_q = queries[n_obs:n_obs + n_teach]
        eval_q = queries[n_obs + n_teach:n_obs + n_teach + n_eval]
        tags = ['txt'] * (n_obs + n_teach + n_eval)

    elif mode == 'txt_resample':
        obs_q, teach_q, eval_q = resample_queries(
            support, queries, n_obs, n_teach, n_eval, rng)
        tags = ['resample'] * (n_obs + n_teach + n_eval)

    elif mode == 'generated':
        obs_q, teach_q, eval_q, tags = generate_episode_queries(
            grammar, support, rng, n_obs, n_teach, n_eval)

    elif mode == 'hybrid':
        # Half from txt, half generated
        n_txt_teach = n_teach // 2
        n_gen_teach = n_teach - n_txt_teach
        n_txt_eval = n_eval // 2
        n_gen_eval = n_eval - n_txt_eval

        obs_q, _, _ = resample_queries(
            support, queries, n_obs, 0, 0, rng)
        _, txt_teach, txt_eval = resample_queries(
            support, queries, 0, n_txt_teach, n_txt_eval, rng)

        gen_obs, gen_teach, gen_eval, gen_tags = generate_episode_queries(
            grammar, support, rng, 0, n_gen_teach, n_gen_eval)
        teach_q = txt_teach + gen_teach
        eval_q = txt_eval + gen_eval
        rng.shuffle(teach_q)
        rng.shuffle(eval_q)
        tags = ['resample'] * n_obs + ['txt'] * n_txt_teach + gen_tags[:n_gen_teach] + \
               ['txt'] * n_txt_eval + gen_tags[n_gen_teach:]

    else:
        obs_q, teach_q, eval_q, tags = generate_episode_queries(
            grammar, support, rng, n_obs, n_teach, n_eval)

    return obs_q, teach_q, eval_q, tags


# ── Main episode runner ───────────────────────────────────────

def run_episode_phase3(
    task_path: str,
    task_id: str,
    seed: int,
    cfg: FullConfig,
    condition_overrides: dict,
) -> dict:
    """Run one full Phase 3 episode with all fixes."""
    rng = np.random.default_rng(seed)
    if cfg is None:
        cfg = FullConfig()
    cfg = copy.deepcopy(cfg)

    shadow_fidelity = condition_overrides.pop('_shadow_fidelity', 'exact')
    apply_overrides(cfg, condition_overrides)

    # Init environment
    env = GrammarTaskEnv(cfg, rng)
    support, queries, grammar = env.load_task(task_path)

    # Init CLS learner
    predictor = CLSSequencePredictor(cfg.learner)
    sub_support = support[:cfg.learner.n_sup]
    predictor.fit_support(sub_support)
    target_pred = TargetPredictor(predictor)

    # Init risk belief
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

    # ★ FIX 1: Query split with source mode (no overflow)
    obs_queries, teach_queries, eval_queries, query_tags = split_queries(
        support, queries, grammar, cfg, rng)

    n_obs = len(obs_queries)
    n_teach = len(teach_queries)
    n_eval = len(eval_queries)

    # Probe queries for divergence (use ALL eval queries)
    probe_queries = [Example(words=q.words, output=q.output) for q in eval_queries]

    # Init belief
    belief = TutorBelief.from_config(cfg.belief)

    # Observation phase
    obs_summary = None
    if cfg.tutor.use_observation_phase and n_obs > 0:
        obs_summary = run_observation_phase(
            env, obs_queries, policy, risk_belief,
            feedback_updater, predictor, target_pred, rng, cfg,
        )
        initialize_belief_from_observation(belief, obs_summary, cfg.belief)

    # Create tutor
    mode = cfg.tutor.tutor_policy_mode
    is_shadow = (mode == 'shadow')
    is_behavioral = (mode == 'behavioral')

    if is_shadow:
        tutor = ShadowTutor(
            cfg.tutor, cfg.learner,
            fidelity=shadow_fidelity,
            probe_queries=probe_queries,
            belief=belief,
        )
        tutor.init_shadow(predictor, risk_belief, sub_support)
    else:
        tutor = create_tutor_p3(cfg, belief, risk_belief)
        # T3: init independent grammar (NOT a copy of learner)
        if is_behavioral and isinstance(tutor, BehavioralTutor):
            tutor.init_own_grammar(sub_support)

    immortal = is_immortal_tutor_p3(tutor)

    # Joint debug log
    debug_log = JointDebugLog()

    # T3: baseline divergence before teaching (after obs)
    if is_behavioral and isinstance(tutor, BehavioralTutor):
        probe_words = [q.words for q in probe_queries]
        probe_gold = [q.output for q in probe_queries]
        tutor.compute_divergence_vs_real(
            predictor, probe_words, probe_gold,
            phase='pre_teach', query_idx=-1,
        )

    # ── Teaching ──
    teach_results = []
    for qi, query in enumerate(teach_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=n_obs + qi, target_output=y_star)
        memory = QueryMemory()

        # ★ FIX 3: Multi-probe divergence BEFORE this query
        if is_shadow and tutor.shadow:
            probe_words = [q.words for q in probe_queries]
            probe_gold = [q.output for q in probe_queries]
            test_vecs = [b.observed_vec for b in state.candidate_pool[:5]]
            div_rec = compute_full_divergence_v2(
                tutor.shadow, predictor, risk_belief,
                probe_words, probe_gold, test_vecs,
                step=qi, query_id=n_obs + qi,
            )
            debug_log.add_divergence(div_rec)

        # T3: per-query divergence (T3's grammar vs real learner's grammar)
        if is_behavioral and isinstance(tutor, BehavioralTutor):
            probe_words = [q.words for q in probe_queries]
            probe_gold = [q.output for q in probe_queries]
            tutor.compute_divergence_vs_real(
                predictor, probe_words, probe_gold,
                phase='teach', query_idx=qi,
            )

        # ★ FIX 2: Run query with shadow grammar sync
        result, diag = run_query_with_shadow_sync(
            env, state, policy, risk_belief, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            belief=belief, immortal=immortal, enable_feedback=True,
        )
        teach_results.append(result)
        # Filter diag to only accepted kwargs
        belief_diag = {k: v for k, v in diag.items()
                       if k in ('had_warning', 'had_hint', 'had_courage',
                                'danger_encountered', 'danger_avoided', 'safe_skipped')}
        update_belief_from_query_result(belief, result, **belief_diag)

    # ★ FIX 3: Divergence AFTER teaching
    if is_shadow and tutor.shadow:
        test_vecs_final = []
        if teach_queries:
            last_state = env.init_query(
                teach_queries[-1], query_id=n_obs + n_teach - 1,
                target_output=target_pred.predict_target(teach_queries[-1].words))
            test_vecs_final = [b.observed_vec for b in last_state.candidate_pool[:5]]
        div_final = compute_full_divergence_v2(
            tutor.shadow, predictor, risk_belief,
            [q.words for q in probe_queries],
            [q.output for q in probe_queries],
            test_vecs_final,
            step=n_teach, query_id=-1,
        )
        debug_log.add_divergence(div_final)

    # ── Eval (freeze) ──
    eval_results = []
    frozen_risk = copy.deepcopy(risk_belief)
    eval_tutor = NoTutor()

    for qi, query in enumerate(eval_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(
            query, query_id=n_obs + n_teach + qi, target_output=y_star)
        memory = QueryMemory()
        result, _ = run_query_with_shadow_sync(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, eval_tutor, memory, rng, cfg,
            belief=None, immortal=False, enable_feedback=False,
        )
        eval_results.append(result)

    # Metrics
    teach_metrics = compute_metrics(teach_results, prefix='Teach')
    eval_metrics = compute_metrics(eval_results, prefix='Eval')

    def _qr_dict(qr):
        return {
            'query_id': qr.query_id,
            'outcome': qr.outcome.name,
            'confirm_count': qr.confirm_count,
            'retry_count': qr.retry_count,
            'death_count': qr.death_count,
            'danger_select_count': qr.danger_select_count,
        }

    result_dict = {
        'task_id': task_id,
        'seed': seed,
        'shadow_fidelity': shadow_fidelity if is_shadow else 'none',
        'query_source': cfg.exp.query_source_mode,
        'n_obs': n_obs,
        'n_teach': n_teach,
        'n_eval': n_eval,
        'teach_metrics': teach_metrics,
        'eval_metrics': eval_metrics,
        'teach_details': [_qr_dict(r) for r in teach_results],
        'eval_details': [_qr_dict(r) for r in eval_results],
        'belief_summary': belief.summary_dict() if belief else {},
    }
    if obs_summary:
        result_dict['obs_summary'] = obs_summary.to_dict()
    if is_shadow:
        from cls_color_selection.tutor_api.joint_debug_v2 import enhanced_summary
        result_dict['joint_debug'] = enhanced_summary(debug_log)
    if is_behavioral and isinstance(tutor, BehavioralTutor):
        result_dict['behavioral_stats'] = tutor.summary_dict()

    return result_dict


# ── Worker + Main ──────────────────────────────────────────────

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
        result = run_episode_phase3(task_path, task_id, seed, cfg, dict(cond_overrides))
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


def main():
    parser = argparse.ArgumentParser(description='Phase 3 experiments (fixed)')
    parser.add_argument('--conditions', nargs='+', default=['T2_exact'])
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=16)
    parser.add_argument('--output-dir', type=str,
                        default='cls_color_selection/results/phase3')
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--query-source', type=str, default='generated',
                        choices=['txt_only', 'txt_resample', 'generated', 'hybrid'])
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.normpath(
        os.path.join(_project_root, 'BASIC', 'cls_learner', 'data'))

    if 'ALL' in args.conditions:
        conditions = list(REGISTRY_P3.keys())
    else:
        conditions = args.conditions

    task_ids = args.tasks or [f'{i:06d}' for i in range(1, 21)]

    from cls_color_selection.experiments.seeds import generate_seeds
    seeds = generate_seeds(42, args.n_seeds)

    cfg = FullConfig()
    cfg.cls_data_dir = data_dir
    cfg.exp.query_source_mode = args.query_source
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
        overrides = dict(REGISTRY_P3.get(cond_name, {}))
        for task_id in task_ids:
            task_path = os.path.join(data_dir, f'{task_id}.txt')
            if not os.path.exists(task_path):
                continue
            for seed in seeds:
                jobs.append((task_path, task_id, seed, cfg_dict, cond_name, overrides))

    print(f"Phase3: {len(jobs)} jobs, {len(conditions)} conditions, "
          f"{len(task_ids)} tasks, {args.n_seeds} seeds, workers={args.n_workers}")
    print(f"  query_source={args.query_source}")

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

    output_path = os.path.join(args.output_dir, 'raw_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + '\n')

    summary_lines = []
    for cond_name in conditions:
        cond_results = [r for r in all_results
                        if r.get('condition') == cond_name and r.get('status') == 'ok']
        if not cond_results:
            errors = [r for r in all_results
                      if r.get('condition') == cond_name and r.get('status') == 'error']
            summary_lines.append(f"\n=== {cond_name}: NO OK RESULTS ===")
            if errors:
                summary_lines.append(f"  ERROR: {errors[0].get('error', '?')[:200]}")
            continue

        summary_lines.append(f"\n=== {cond_name} ({len(cond_results)} runs) ===")
        for prefix in ['teach_metrics', 'eval_metrics']:
            keys = list(cond_results[0].get(prefix, {}).keys())
            for k in keys:
                vals = [r[prefix].get(k, 0) for r in cond_results]
                summary_lines.append(f"  {k}: {np.mean(vals):.3f}±{np.std(vals):.3f}")

        debug_results = [r.get('joint_debug', {}) for r in cond_results
                         if 'joint_debug' in r]
        if debug_results:
            for dk in ['D_gram_top1_agreement', 'D_gram_JS', 'D_risk_l1',
                        'D_param_role_l1', 'CF_abs_error']:
                vals = [d.get(dk, 0) for d in debug_results if dk in d]
                if vals:
                    summary_lines.append(f"  {dk}: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Phase 3 Results Summary (FIXED)\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Jobs: {len(jobs)}, Elapsed: {elapsed:.1f}s\n")
        f.write(f"Query source: {args.query_source}\n\n")
        f.write(summary_text)

    print(f"\nResults: {output_path}\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
