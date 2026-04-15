"""
run_phase4.py — Phase 4 experiment runner: Inverse Inference Tutor.

Key differences from run_phase3.py:
  1. Observation = result-level (observation_v2): tutor sees (words, submitted, outcome)
  2. TutorTaskModel + TutorLearnerModel separation
  3. InverseTutor with update_depth control (role_only / role_emit / full_trace)
  4. Per-query divergence + predictive validity timeline
  5. Baselines (T0/T1/T2) reuse Phase 3 query loop unchanged

Architecture:
  Obs  phase → result-level inverse inference (learner model updated)
  Teach phase → process-level with per-query learner model updates
  Eval  phase → frozen, no tutor, no updates
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
from cls_color_selection.tutor_api.belief_update import (
    initialize_belief_from_observation, update_belief_from_query_result,
)
from cls_color_selection.tutor_api.dummy_tutor import (
    NoTutor, NoTutorImmortalWarnlike,
)
from cls_color_selection.tutor_api.tutor_rule import RuleTutor
from cls_color_selection.tutor_api.tutor_proxy import ProxyTutor
from cls_color_selection.tutor_api.action_generators import apply_hint_to_state

# Phase 4 new imports
from cls_color_selection.tutor_api.observation_v2 import (
    run_observation_phase_v2, ObservationSummaryV2,
)
from cls_color_selection.tutor_api.task_model import TutorTaskModel
from cls_color_selection.tutor_api.tutor_inverse import InverseTutor
from cls_color_selection.tutor_api.divergence_v3 import (
    compute_inverse_divergence, compute_predictive_validity,
)
from cls_color_selection.experiments.registry_phase4 import (
    REGISTRY_P4, EXPERIMENT_SETS, apply_overrides,
)

# Also import Phase 3 pieces for baseline conditions
from cls_color_selection.tutor_api.tutor_shadow import ShadowTutor
from cls_color_selection.tutor_api.shadow_update import shadow_feedback_update
from cls_color_selection.tutor_api.joint_debug import JointDebugLog
from cls_color_selection.tutor_api.joint_debug_v2 import (
    compute_full_divergence_v2,
)


MAX_STEPS = 200


# ── Query loop (shared by all tutors) ─────────────────────────

def run_query_loop(
    env, state, policy, risk_belief, feedback_updater,
    predictor, target_pred, tutor, memory, rng, cfg,
    belief=None, immortal=False, enable_feedback=True,
):
    """Run one query loop. Works for ALL tutor types.

    For inverse tutor: tutor.on_select / on_confirm_fail are called
    per step. Post-query update handled separately by caller.
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
        'last_submitted': None,
        'last_feedback': None,
    }

    is_shadow = isinstance(tutor, ShadowTutor)
    is_inverse = isinstance(tutor, InverseTutor)

    # Signal query start
    if is_inverse:
        tutor.observe_query_start()

    while not state.is_terminal and step_count < MAX_STEPS:
        step_count += 1

        selected_indices = policy.select_set(state, risk_belief, rng)

        if not selected_indices:
            empty_retry_count += 1
            if empty_retry_count >= MAX_EMPTY and state.filled_count > 0:
                state, success, feedback = env.step_confirm(
                    state, cfg.learner.feedback_mode)
                if success or state.is_terminal:
                    break
                # Record submitted for inverse tutor
                diag['last_submitted'] = feedback.get('submitted')
                diag['last_feedback'] = feedback

                hint_action = tutor.on_confirm_fail(state, feedback)
                if hint_action.action_type == TutorActionType.HINT:
                    state = apply_hint_to_state(state, hint_action)
                    diag['had_hint'] = True
                # Grammar feedback
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
            if is_inverse:
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
                from cls_color_selection.learner.courage_update import \
                    courage_literal_update
                courage_action = tutor.on_courage_check(state)
                if courage_action.action_type == TutorActionType.COURAGE:
                    diag['had_courage'] = True
                    courage_literal_update(
                        risk_belief, state.candidate_pool,
                        state.needed_colors())
            continue

        if state.outcome == Outcome.DEATH:
            diag['danger_encountered'] = True
            for b in selected:
                if b.is_danger:
                    risk_belief.update_from_death(b.observed_vec)
            if is_inverse:
                tutor.observe_death()
            break

        for b in selected:
            if not b.is_danger:
                risk_belief.update_from_safe_observation(b.observed_vec)
        memory.record_safe_placement([b.observed_vec for b in selected])

        if policy.should_confirm(state):
            state, success, feedback = env.step_confirm(
                state, cfg.learner.feedback_mode)

            if success:
                if is_inverse:
                    tutor.observe_confirm_success()
                break
            if state.outcome == Outcome.TIMEOUT:
                diag['last_submitted'] = feedback.get('submitted')
                diag['last_feedback'] = feedback
                break

            # Record submitted for inverse tutor
            diag['last_submitted'] = feedback.get('submitted')
            diag['last_feedback'] = feedback

            hint_action = tutor.on_confirm_fail(state, feedback)
            if hint_action.action_type == TutorActionType.HINT:
                state = apply_hint_to_state(state, hint_action)
                diag['had_hint'] = True

            # Grammar feedback
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
    """Apply grammar feedback to real learner (and shadow if applicable)."""
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

    # Shadow tutor: sync shadow grammar
    if is_shadow and hasattr(tutor, 'shadow') and tutor.shadow is not None:
        shadow_feedback_update(
            tutor.shadow, state.query_words, submitted,
            feedback, cfg.learner)


# ── Query split ────────────────────────────────────────────────

def split_queries(support, queries, grammar, cfg, rng):
    """Split queries based on query_source_mode."""
    n_obs = cfg.exp.n_obs_queries
    n_teach = cfg.exp.n_teach_queries
    n_eval = cfg.exp.n_eval_queries
    mode = cfg.exp.query_source_mode

    if mode == 'txt_only':
        all_q = list(queries)
        rng.shuffle(all_q)
        total = n_obs + n_teach + n_eval
        if len(all_q) < total:
            all_q = all_q * ((total // len(all_q)) + 1)
        obs_q = all_q[:n_obs]
        teach_q = all_q[n_obs:n_obs + n_teach]
        eval_q = all_q[n_obs + n_teach:n_obs + n_teach + n_eval]
        tags = ['txt'] * (n_obs + n_teach + n_eval)
    elif mode == 'txt_resample':
        obs_q, teach_q, eval_q, tags = resample_queries(
            support, queries, grammar, rng, n_obs, n_teach, n_eval)
    elif mode == 'hybrid':
        obs_q, teach_q, eval_q, tags = generate_episode_queries(
            grammar, support, rng, n_obs, n_teach, n_eval)
        txt_q = list(queries)
        rng.shuffle(txt_q)
        n_inject = min(len(txt_q), n_teach // 2)
        for i in range(n_inject):
            teach_q[i] = txt_q[i]
            tags[n_obs + i] = 'txt'
    else:
        obs_q, teach_q, eval_q, tags = generate_episode_queries(
            grammar, support, rng, n_obs, n_teach, n_eval)

    return obs_q, teach_q, eval_q, tags


# ── Baseline tutor factory ─────────────────────────────────────

def create_baseline_tutor(cfg, belief, risk_belief):
    """Create non-inverse tutor (T0/T1)."""
    mode = cfg.tutor.tutor_policy_mode
    if mode == 'rule':
        return RuleTutor(cfg.tutor, belief=belief)
    elif mode == 'proxy':
        return ProxyTutor(cfg.tutor, belief=belief, risk_belief=risk_belief)
    elif mode == 'none':
        return NoTutor()
    return RuleTutor(cfg.tutor, belief=belief)


# ── Main episode runner ────────────────────────────────────────

def run_episode_phase4(
    task_path: str,
    task_id: str,
    seed: int,
    cfg: FullConfig,
    condition_overrides: dict,
) -> dict:
    """Run one full Phase 4 episode.

    Supports:
    - T0_rule, T1_proxy: baseline tutors (same logic as Phase 3)
    - T2_oracle: shadow tutor (same logic as Phase 3)
    - T3_*Infer: inverse inference tutors (NEW Phase 4 logic)
    """
    rng = np.random.default_rng(seed)
    if cfg is None:
        cfg = FullConfig()
    cfg = copy.deepcopy(cfg)

    shadow_fidelity = condition_overrides.pop('_shadow_fidelity', 'none')
    inverse_depth = condition_overrides.pop('_inverse_update_depth', 'full_trace')
    inverse_init_mode = condition_overrides.pop('_inverse_init_mode', 'support')
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

    # Query split
    obs_queries, teach_queries, eval_queries, query_tags = split_queries(
        support, queries, grammar, cfg, rng)

    n_obs = len(obs_queries)
    n_teach = len(teach_queries)
    n_eval = len(eval_queries)

    # Probe queries for divergence (use ALL eval queries)
    probe_queries = [Example(words=q.words, output=q.output) for q in eval_queries]
    probe_words = [q.words for q in probe_queries]
    probe_gold = [q.output for q in probe_queries]

    # Init belief
    belief = TutorBelief.from_config(cfg.belief)

    mode = cfg.tutor.tutor_policy_mode
    is_inverse = (mode == 'inverse')
    is_shadow = (mode == 'shadow')

    # ══════════════════════════════════════════════════════════════
    # PATH A: Inverse tutor (T3 new main line)
    # ══════════════════════════════════════════════════════════════
    if is_inverse:
        return _run_inverse_episode(
            env, cfg, rng, sub_support, support, queries,
            predictor, target_pred, risk_belief, policy, feedback_updater,
            belief, obs_queries, teach_queries, eval_queries,
            probe_words, probe_gold, inverse_depth, inverse_init_mode,
            task_id, seed, n_obs, n_teach, n_eval,
        )

    # ══════════════════════════════════════════════════════════════
    # PATH B: Baseline tutors (T0/T1/T2) — same as Phase 3
    # ══════════════════════════════════════════════════════════════
    return _run_baseline_episode(
        env, cfg, rng, sub_support, support, queries,
        predictor, target_pred, risk_belief, policy, feedback_updater,
        belief, obs_queries, teach_queries, eval_queries,
        probe_words, probe_gold, shadow_fidelity,
        task_id, seed, n_obs, n_teach, n_eval,
    )


# ── Inverse episode ───────────────────────────────────────────

def _run_inverse_episode(
    env, cfg, rng, sub_support, support, queries,
    predictor, target_pred, risk_belief, policy, feedback_updater,
    belief, obs_queries, teach_queries, eval_queries,
    probe_words, probe_gold, inverse_depth, inverse_init_mode,
    task_id, seed, n_obs, n_teach, n_eval,
):
    """Run episode with inverse inference tutor."""

    # Create TutorTaskModel (ground truth knowledge)
    task_model = TutorTaskModel(env,
                                queries=list(support) + list(queries))
    # Also register generated queries
    all_q = list(obs_queries) + list(teach_queries) + list(eval_queries)
    task_model.register_queries(all_q)

    # Create InverseTutor
    tutor = InverseTutor(
        cfg.tutor, cfg.learner, task_model,
        update_depth=inverse_depth,
        hint_after_confirm_fail=cfg.tutor.hint_after_confirm_fail,
    )

    # ── Init learner model according to init_mode ──
    if inverse_init_mode == 'cold':
        # SCHEME A: empty start, vocabulary only
        vocab = set()
        for q in list(support) + list(queries):
            vocab.update(q.words)
        for q in all_q:
            vocab.update(q.words)
        tutor.init_learner_model_cold(sorted(vocab))

    elif inverse_init_mode == 'teacher_prior':
        # SCHEME B: teacher knows task queries, not learner's support
        # Use obs+teach queries as teacher's generic grammar knowledge
        teacher_examples = list(obs_queries) + list(teach_queries)
        if not teacher_examples:
            teacher_examples = list(queries)
        tutor.init_learner_model_teacher_prior(teacher_examples)

    else:
        # ORACLE: same support as learner (default, old behavior)
        tutor.init_learner_model(sub_support)

    # ── Observation phase (result-level) ──
    obs_summary = None
    if cfg.tutor.use_observation_phase and n_obs > 0:
        obs_summary = run_observation_phase_v2(
            env, obs_queries, policy, risk_belief,
            feedback_updater, predictor, target_pred, rng, cfg,
        )
        # Feed result-level observations to inverse tutor
        tutor.process_all_observations(obs_summary)

        # Also init belief from obs stats (v2: result-level only)
        obs_stats = obs_summary.to_dict()
        n_ok = obs_summary.n_correct
        n_fail = obs_summary.n_wrong + obs_summary.n_timeout + obs_summary.n_death
        belief.sem.success_rate.update_success(n_ok)
        belief.sem.success_rate.update_failure(n_fail)
        belief.sem.confirm_timeout_rate = obs_stats.get('ObsTimeoutRate', 0.0)

    # ── Divergence timeline ──
    div_timeline = []
    pred_timeline = []

    # Pre-teach divergence
    div_rec = compute_inverse_divergence(
        tutor.learner_model, predictor,
        probe_words, probe_gold,
        phase='pre_teach', query_idx=-1,
    )
    div_timeline.append(div_rec)

    # ── Teaching ──
    teach_results = []
    for qi, query in enumerate(teach_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=n_obs + qi, target_output=y_star)
        memory = QueryMemory()

        # Predictive validity BEFORE this query
        if qi < len(teach_queries):
            pred_rec = compute_predictive_validity(
                tutor.learner_model, predictor, query.words)
            pred_rec['phase'] = 'teach'
            pred_rec['query_idx'] = qi
            pred_timeline.append(pred_rec)

        # Divergence BEFORE this query
        div_rec = compute_inverse_divergence(
            tutor.learner_model, predictor,
            probe_words, probe_gold,
            phase='teach', query_idx=qi,
        )
        div_timeline.append(div_rec)

        # Run query
        result, diag = run_query_loop(
            env, state, policy, risk_belief, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            belief=belief, immortal=False, enable_feedback=True,
        )
        teach_results.append(result)

        # Post-query: update inverse tutor's learner model
        submitted = diag.get('last_submitted')
        if submitted is None:
            # Try to get from result
            if result.outcome == Outcome.SUCCESS:
                submitted = list(state.ground_truth)
            else:
                submitted = [c for c in state.completion if c is not None]

        tutor.update_after_query(
            query.words, submitted, result.outcome,
            feedback=diag.get('last_feedback'),
        )

        # Update belief
        belief_diag = {k: v for k, v in diag.items()
                       if k in ('had_warning', 'had_hint', 'had_courage',
                                'danger_encountered', 'danger_avoided',
                                'safe_skipped')}
        update_belief_from_query_result(belief, result, **belief_diag)

    # Post-teach divergence
    div_rec = compute_inverse_divergence(
        tutor.learner_model, predictor,
        probe_words, probe_gold,
        phase='post_teach', query_idx=n_teach,
    )
    div_timeline.append(div_rec)

    # ── Eval (freeze) ──
    eval_results = []
    frozen_risk = copy.deepcopy(risk_belief)
    eval_tutor = NoTutor()

    for qi, query in enumerate(eval_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(
            query, query_id=n_obs + n_teach + qi, target_output=y_star)
        memory = QueryMemory()
        result, _ = run_query_loop(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, eval_tutor, memory, rng, cfg,
            belief=None, immortal=False, enable_feedback=False,
        )
        eval_results.append(result)

    # ── Assemble results ──
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
        'tutor_mode': 'inverse',
        'inverse_depth': inverse_depth,
        'inverse_init_mode': inverse_init_mode,
        'query_source': cfg.exp.query_source_mode,
        'n_obs': n_obs, 'n_teach': n_teach, 'n_eval': n_eval,
        'teach_metrics': teach_metrics,
        'eval_metrics': eval_metrics,
        'teach_details': [_qr_dict(r) for r in teach_results],
        'eval_details': [_qr_dict(r) for r in eval_results],
        'belief_summary': belief.summary_dict() if belief else {},
        # Phase 4 specific: inverse inference diagnostics
        'inverse_stats': tutor.summary_dict(),
        'divergence_timeline': div_timeline,
        'prediction_timeline': pred_timeline,
        # Phase 4.5: hint decision diagnostics
        'hint_diag_log': tutor._hint_diag_log,
    }

    # Phase 4.5: hint summary stats
    hint_log = tutor._hint_diag_log
    if hint_log:
        n_hint = sum(1 for d in hint_log if d.get('decision') == 'HINT')
        n_wait = sum(1 for d in hint_log if d.get('decision') == 'WAIT')
        avg_Q = float(np.mean([d.get('Q_hint', 0) for d in hint_log]))
        avg_H = float(np.mean([d.get('H_beam_norm', 0) for d in hint_log]))
        avg_margin = float(np.mean([d.get('margin', 1) for d in hint_log]))
        avg_k = float(np.mean([d.get('n_positions', 0)
                                for d in hint_log if d.get('decision') == 'HINT'])
                       ) if n_hint > 0 else 0
        result_dict['hint_summary'] = {
            'n_hint_decisions': len(hint_log),
            'n_hint': n_hint,
            'n_wait': n_wait,
            'hint_rate': n_hint / max(len(hint_log), 1),
            'avg_Q_hint': avg_Q,
            'avg_H_beam_norm': avg_H,
            'avg_margin': avg_margin,
            'avg_k': avg_k,
        }

    if obs_summary:
        result_dict['obs_summary'] = obs_summary.to_dict()

    return result_dict


# ── Baseline episode (T0/T1/T2) ──────────────────────────────

def _run_baseline_episode(
    env, cfg, rng, sub_support, support, queries,
    predictor, target_pred, risk_belief, policy, feedback_updater,
    belief, obs_queries, teach_queries, eval_queries,
    probe_words, probe_gold, shadow_fidelity,
    task_id, seed, n_obs, n_teach, n_eval,
):
    """Run episode for baseline tutors (same as Phase 3)."""

    # Observation phase (v1 for baselines)
    obs_summary = None
    if cfg.tutor.use_observation_phase and n_obs > 0:
        from cls_color_selection.tutor_api.observation import run_observation_phase
        obs_summary = run_observation_phase(
            env, obs_queries, policy, risk_belief,
            feedback_updater, predictor, target_pred, rng, cfg,
        )
        initialize_belief_from_observation(belief, obs_summary, cfg.belief)

    mode = cfg.tutor.tutor_policy_mode
    is_shadow = (mode == 'shadow')

    if is_shadow:
        tutor = ShadowTutor(
            cfg.tutor, cfg.learner,
            fidelity=shadow_fidelity,
            probe_queries=[Example(words=w, output=g)
                           for w, g in zip(probe_words, probe_gold)],
            belief=belief,
        )
        tutor.init_shadow(predictor, risk_belief, sub_support)
    else:
        tutor = create_baseline_tutor(cfg, belief, risk_belief)

    debug_log = JointDebugLog()

    # Teaching
    teach_results = []
    for qi, query in enumerate(teach_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(query, query_id=n_obs + qi, target_output=y_star)
        memory = QueryMemory()

        # Shadow divergence
        if is_shadow and tutor.shadow:
            test_vecs = [b.observed_vec for b in state.candidate_pool[:5]]
            div_rec = compute_full_divergence_v2(
                tutor.shadow, predictor, risk_belief,
                probe_words, probe_gold, test_vecs,
                step=qi, query_id=n_obs + qi,
            )
            debug_log.add_divergence(div_rec)

        result, diag = run_query_loop(
            env, state, policy, risk_belief, feedback_updater,
            predictor, target_pred, tutor, memory, rng, cfg,
            belief=belief, immortal=False, enable_feedback=True,
        )
        teach_results.append(result)
        belief_diag = {k: v for k, v in diag.items()
                       if k in ('had_warning', 'had_hint', 'had_courage',
                                'danger_encountered', 'danger_avoided',
                                'safe_skipped')}
        update_belief_from_query_result(belief, result, **belief_diag)

    # Eval
    eval_results = []
    frozen_risk = copy.deepcopy(risk_belief)
    eval_tutor = NoTutor()

    for qi, query in enumerate(eval_queries):
        y_star = target_pred.predict_target(query.words)
        state = env.init_query(
            query, query_id=n_obs + n_teach + qi, target_output=y_star)
        memory = QueryMemory()
        result, _ = run_query_loop(
            env, state, policy, frozen_risk, feedback_updater,
            predictor, target_pred, eval_tutor, memory, rng, cfg,
            belief=None, immortal=False, enable_feedback=False,
        )
        eval_results.append(result)

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
        'tutor_mode': mode,
        'shadow_fidelity': shadow_fidelity if is_shadow else 'none',
        'query_source': cfg.exp.query_source_mode,
        'n_obs': n_obs, 'n_teach': n_teach, 'n_eval': n_eval,
        'teach_metrics': teach_metrics,
        'eval_metrics': eval_metrics,
        'teach_details': [_qr_dict(r) for r in teach_results],
        'eval_details': [_qr_dict(r) for r in eval_results],
        'belief_summary': belief.summary_dict() if belief else {},
    }
    if obs_summary:
        result_dict['obs_summary'] = (
            obs_summary.to_dict() if hasattr(obs_summary, 'to_dict')
            else obs_summary)
    if is_shadow:
        from cls_color_selection.tutor_api.joint_debug_v2 import enhanced_summary
        result_dict['joint_debug'] = enhanced_summary(debug_log)

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
        result = run_episode_phase4(
            task_path, task_id, seed, cfg, dict(cond_overrides))
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
    parser = argparse.ArgumentParser(
        description='Phase 4 experiments: Inverse Inference Tutor')
    parser.add_argument('--conditions', nargs='+', default=['T3_traceInfer'])
    parser.add_argument('--experiment-set', type=str, default=None,
                        choices=list(EXPERIMENT_SETS.keys()),
                        help='Run a predefined experiment set')
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=16)
    parser.add_argument('--output-dir', type=str,
                        default='cls_color_selection/results/phase4')
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--query-source', type=str, default='generated',
                        choices=['txt_only', 'txt_resample', 'generated', 'hybrid'])
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.normpath(
        os.path.join(_project_root, 'BASIC', 'cls_learner', 'data'))

    # Determine conditions
    if args.experiment_set:
        conditions = EXPERIMENT_SETS[args.experiment_set]
    elif 'ALL' in args.conditions:
        conditions = list(REGISTRY_P4.keys())
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
        overrides = dict(REGISTRY_P4.get(cond_name, {}))
        for task_id in task_ids:
            task_path = os.path.join(data_dir, f'{task_id}.txt')
            if not os.path.exists(task_path):
                continue
            for seed in seeds:
                jobs.append((
                    task_path, task_id, seed, cfg_dict, cond_name, overrides))

    print(f"Phase4: {len(jobs)} jobs, {len(conditions)} conditions, "
          f"{len(task_ids)} tasks, {args.n_seeds} seeds, "
          f"workers={args.n_workers}")
    print(f"  conditions: {conditions}")
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

    # Save raw results
    output_path = os.path.join(args.output_dir, 'raw_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + '\n')

    # Summary
    _print_summary(all_results, conditions, args, elapsed, jobs)

    print(f"\nResults: {output_path}")


def _print_summary(all_results, conditions, args, elapsed, jobs):
    """Print and save summary."""
    summary_lines = []

    for cond_name in conditions:
        cond_results = [r for r in all_results
                        if r.get('condition') == cond_name
                        and r.get('status') == 'ok']
        if not cond_results:
            errors = [r for r in all_results
                      if r.get('condition') == cond_name
                      and r.get('status') == 'error']
            summary_lines.append(f"\n=== {cond_name}: NO OK RESULTS ===")
            if errors:
                summary_lines.append(
                    f"  ERROR: {errors[0].get('error', '?')[:200]}")
                summary_lines.append(
                    f"  TRACEBACK: {errors[0].get('traceback', '')[:500]}")
            continue

        summary_lines.append(f"\n=== {cond_name} ({len(cond_results)} runs) ===")

        # Teaching + eval metrics
        for prefix in ['teach_metrics', 'eval_metrics']:
            keys = list(cond_results[0].get(prefix, {}).keys())
            for k in keys:
                vals = [r[prefix].get(k, 0) for r in cond_results]
                summary_lines.append(
                    f"  {k}: {np.mean(vals):.3f}±{np.std(vals):.3f}")

        # Inverse-specific: divergence timeline
        div_results = [r.get('divergence_timeline', []) for r in cond_results
                       if 'divergence_timeline' in r]
        if div_results:
            # Show pre_teach and post_teach divergence
            for phase_label in ['pre_teach', 'post_teach']:
                js_vals = []
                agr_vals = []
                for timeline in div_results:
                    for rec in timeline:
                        if rec.get('phase') == phase_label:
                            if rec.get('js_divergence') is not None:
                                js_vals.append(rec['js_divergence'])
                            if rec.get('top1_agreement') is not None:
                                agr_vals.append(rec['top1_agreement'])
                if js_vals:
                    summary_lines.append(
                        f"  {phase_label}_JS: "
                        f"{np.mean(js_vals):.4f}±{np.std(js_vals):.4f}")
                if agr_vals:
                    summary_lines.append(
                        f"  {phase_label}_Top1Agree: "
                        f"{np.mean(agr_vals):.4f}±{np.std(agr_vals):.4f}")

        # Prediction accuracy
        pred_results = [r.get('prediction_timeline', []) for r in cond_results
                        if 'prediction_timeline' in r]
        if pred_results:
            match_vals = []
            for timeline in pred_results:
                for rec in timeline:
                    if rec.get('pred_match') is not None:
                        match_vals.append(float(rec['pred_match']))
            if match_vals:
                summary_lines.append(
                    f"  PredAcc_next: "
                    f"{np.mean(match_vals):.4f}±{np.std(match_vals):.4f}")

        # Shadow divergence (for T2)
        debug_results = [r.get('joint_debug', {}) for r in cond_results
                         if 'joint_debug' in r]
        if debug_results:
            for dk in ['D_gram_top1_agreement', 'D_gram_JS', 'D_risk_l1']:
                vals = [d.get(dk, 0) for d in debug_results if dk in d]
                if vals:
                    summary_lines.append(
                        f"  {dk}: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Phase 4 Results Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Jobs: {len(jobs)}, Elapsed: {elapsed:.1f}s\n")
        f.write(f"Query source: {args.query_source}\n\n")
        f.write(summary_text)

    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
