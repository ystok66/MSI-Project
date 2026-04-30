from __future__ import annotations

from dataclasses import asdict
import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import FullConfig
from ..env.interventions import get_active_menu
from ..env.option_env import OptionEnv
from ..env.state import BlockState, QueryState
from ..eval.autonomous_probe import _clone_learner
from ..interfaces import Example
from ..learner.learner_agent import LearnerAgent
from ..tutor.observation_adapter import ObservationAdapter
from .interfaces import ConditionResult, EvalItem, EvalMetrics, HintCandidate, ObservationCase, ObservationRun, TaskContext, TeachCase, TeachTraceSummary


def build_full_config(cfg) -> FullConfig:
    full = FullConfig()
    full.env.K = max(cfg.obs_menu_size, cfg.teach_menu_size)
    full.env.T_max = max(cfg.max_attempts_main, cfg.max_attempts_no_tutor_extra)
    full.env.H_0 = cfg.hp_0
    full.env.n_risky = max(0, cfg.n_risk_options if cfg.use_risk else 0)
    full.env.feedback_mode = cfg.feedback_mode
    full.env.danger_dim = cfg.danger_dim
    full.env.cluster_sigma = cfg.cluster_sigma
    full.env.max_refreshes = 0
    full.env.enforce_max_refreshes = True
    full.learner.use_cls = cfg.use_cls
    full.learner.n_sup = max(1, cfg.n_pre_easy + cfg.n_pre_medium + cfg.n_pre_hard)
    full.learner.n_em = cfg.n_em
    full.learner.use_hpc = cfg.use_hpc
    full.learner.tau_sem = cfg.tau_sem
    reveal_mode = str(getattr(cfg, "reveal_learning_mode", "cortex_em"))
    full.learner.reveal_learning_mode = "off" if reveal_mode == "delayed_study" else reveal_mode
    full.learner.negative_evidence_mode = str(getattr(cfg, "negative_evidence_mode", "off"))
    full.learner.eta_reveal = cfg.eta_reveal
    full.learner.eta_correct_pick = cfg.eta_correct_pick
    full.learner.correct_pick_learning_mode = cfg.correct_pick_learning_mode
    full.learner.rho_assist = cfg.rho_assist
    full.learner.pedagogical_feedback_mode = cfg.pedagogical_feedback_mode
    full.cls_data_dir = cfg.data_dir
    full.seed = cfg.seed
    return full


def bootstrap_env(cfg, task_id: str, seed: int) -> OptionEnv:
    full_cfg = build_full_config(cfg)
    env = OptionEnv(cfg=full_cfg, data_dir=cfg.data_dir)
    env.reset_block(task_id, seed=seed)
    return env


def build_base_learner(context: TaskContext, prelearn_examples: List[Example], seed: int) -> LearnerAgent:
    learner_cfg = copy.deepcopy(context.env.cfg)
    learner_cfg.learner.n_sup = len(prelearn_examples)
    learner = LearnerAgent(cfg=learner_cfg, seed=seed)
    block = BlockState(
        block_id=0,
        support_examples=list(prelearn_examples),
        queries=[],
        obs_phase_queries=0,
        teach_phase_queries=0,
        eval_phase_queries=0,
    )
    learner.init_block(block, context.grammar, list(prelearn_examples))
    return learner


def apply_hint(learner: LearnerAgent, hint: Optional[HintCandidate]) -> None:
    if hint is None:
        return
    scorer = getattr(learner, "_scorer", None)
    if scorer is None or not hasattr(scorer, "incremental_study"):
        return
    scorer.incremental_study([Example(words=list(hint.example.words), output=list(hint.example.output))])


def apply_hints(learner: LearnerAgent, hints: Sequence[HintCandidate]) -> None:
    for hint in hints:
        apply_hint(learner, hint)


def _single_query_block(
    query: QueryState,
    phase: str,
) -> BlockState:
    if phase == "obs":
        return BlockState(
            block_id=0,
            support_examples=[],
            queries=[query],
            obs_phase_queries=1,
            teach_phase_queries=0,
            eval_phase_queries=0,
        )
    if phase == "teach":
        return BlockState(
            block_id=0,
            support_examples=[],
            queries=[query],
            obs_phase_queries=0,
            teach_phase_queries=1,
            eval_phase_queries=0,
        )
    return BlockState(
        block_id=0,
        support_examples=[],
        queries=[query],
        obs_phase_queries=0,
        teach_phase_queries=0,
        eval_phase_queries=1,
    )


def _run_single_query(
    learner: LearnerAgent,
    context: TaskContext,
    example: Example,
    menu,
    max_attempts: int,
    phase: str,
    remove_wrong_after_reveal: bool,
    post_first_wrong_hint: Optional[HintCandidate] = None,
) -> BlockState:
    query = QueryState(
        query_id=0,
        target_output=list(example.output),
        true_program=list(example.words),
        hp=int(context.cfg.hp_0),
        max_rounds=int(max_attempts),
        max_refreshes=0,
        enforce_max_refreshes=True,
        menu=copy.deepcopy(menu),
    )
    block = _single_query_block(query, phase)

    steps = 0
    max_steps = max(1, max_attempts * 3)
    post_hint_used = False
    while not block.done and steps < max_steps:
        steps += 1
        qs_before = block.current_query
        if qs_before is None:
            break
        context.env.tutor_act(block, "WAIT")
        if block.done:
            break
        learner.act(block, context.env)
        if not block.learner_trace:
            break
        last = block.learner_trace[-1]
        if (
            last.action == "pick"
            and last.correct is False
            and not block.done
            and qs_before is block.current_query
        ):
            if not post_hint_used and post_first_wrong_hint is not None:
                apply_hint(learner, post_first_wrong_hint)
                post_hint_used = True
            if remove_wrong_after_reveal:
                qs_before.banned_indices.add(int(last.pick_index))

    return block


def _delayed_reveal_examples(block: BlockState, teach_case: TeachCase) -> List[Example]:
    query = block.queries[0] if block.queries else None
    if query is None:
        return []
    option_by_index = {int(opt.index): opt for opt in teach_case.menu}
    delayed: List[Example] = []
    for event in list(getattr(query, "reveal_history", []) or []):
        opt = option_by_index.get(int(getattr(event, "option_index", -1)))
        if opt is None:
            continue
        delayed.append(
            Example(
                words=list(opt.text),
                output=list(getattr(event, "revealed_output", []) or []),
            )
        )
    return delayed


def apply_delayed_reveal_updates(
    learner: LearnerAgent,
    block: BlockState,
    teach_case: TeachCase,
    cfg,
) -> None:
    scorer = getattr(learner, "_scorer", None)
    if scorer is None or not hasattr(scorer, "incremental_study"):
        return
    delayed = _delayed_reveal_examples(block, teach_case)
    if not delayed:
        return
    eta = float(getattr(cfg, "eta_reveal", 1.0))
    rng = getattr(learner, "rng", None)
    for ex in delayed:
        if eta < 1.0 and rng is not None and rng.random() >= eta:
            continue
        scorer.incremental_study([ex])


def run_observation_case(
    base_learner: LearnerAgent,
    context: TaskContext,
    case: ObservationCase,
) -> ObservationRun:
    learner = _clone_learner(
        base_learner,
        freeze_semantic=True,
        freeze_risk=True,
        freeze_memory=True,
    )
    block = _run_single_query(
        learner,
        context,
        case.example,
        case.menu,
        max_attempts=int(context.cfg.max_attempts_main),
        phase="obs",
        remove_wrong_after_reveal=context.cfg.remove_wrong_after_reveal,
    )
    adapter = ObservationAdapter()
    steps = adapter.extract_steps(block)
    return ObservationRun(case=case, block=block, steps=steps)


def run_teach_condition(
    base_learner: LearnerAgent,
    context: TaskContext,
    teach_case: TeachCase,
    max_attempts: int,
    hint: Optional[HintCandidate],
    eval_items: Optional[Iterable[EvalItem]],
    condition_name: str,
    hints: Optional[Sequence[HintCandidate]] = None,
    post_first_wrong_hint: Optional[HintCandidate] = None,
) -> ConditionResult:
    learner = copy.deepcopy(base_learner)
    pre_hints: List[HintCandidate] = list(hints or [])
    if hint is not None:
        pre_hints.append(hint)
    apply_hints(learner, pre_hints)
    initial_trace = compute_initial_policy_trace_summary(
        learner=learner,
        context=context,
        teach_case=teach_case,
    )
    block = _run_single_query(
        learner,
        context,
        teach_case.example,
        teach_case.menu,
        max_attempts=max_attempts,
        phase="teach",
        remove_wrong_after_reveal=context.cfg.remove_wrong_after_reveal,
        post_first_wrong_hint=post_first_wrong_hint,
    )
    if str(getattr(context.cfg, "reveal_learning_mode", "cortex_em")) == "delayed_study":
        # delayed_study is intentionally "batch after query": teach-time wrong reveals
        # are consolidated once the search episode ends, then eval reads that state.
        apply_delayed_reveal_updates(learner, block, teach_case, context.cfg)
    metrics = summarize_teach_block(block)
    trace_summary = extract_teach_trace_summary(block, teach_case, initial_trace)
    eval_metrics = evaluate_direct(learner, eval_items) if eval_items is not None else None
    hint_kind_parts = [h.kind for h in pre_hints]
    hint_diff_parts = [h.difficulty for h in pre_hints]
    if post_first_wrong_hint is not None:
        hint_kind_parts.append(f"post:{post_first_wrong_hint.kind}")
        hint_diff_parts.append(f"post:{post_first_wrong_hint.difficulty}")
    return ConditionResult(
        condition=condition_name,
        first_correct_attempt=metrics["first_correct_attempt"],
        success_within_limit=metrics["success_within_limit"],
        n_wrong_before_correct=metrics["n_wrong_before_correct"],
        safe_wrong_count=metrics["safe_wrong_count"],
        risky_wrong_count=metrics["risky_wrong_count"],
        risk_any=metrics["risk_any"],
        risk_count=metrics["risk_count"],
        damage_sum=metrics["damage_sum"],
        eval_metrics=eval_metrics,
        hint_used=bool(pre_hints or post_first_wrong_hint is not None),
        hint_kind="none" if not hint_kind_parts else "+".join(hint_kind_parts),
        hint_difficulty="none" if not hint_diff_parts else "+".join(hint_diff_parts),
        hint_source_index=None if not pre_hints else pre_hints[0].source_index,
        teach_trace_summary=trace_summary,
    )


def summarize_teach_block(block: BlockState) -> Dict[str, object]:
    picks = [step for step in block.learner_trace if step.action == "pick"]
    first_correct = None
    wrong_before_correct = 0
    risk_count = 0
    safe_wrong_count = 0
    risky_wrong_count = 0
    damage_sum = 0
    for idx, step in enumerate(picks, start=1):
        if step.correct:
            first_correct = idx
            break
        wrong_before_correct += 1
        damage = int(step.damage or 0)
        damage_sum += damage
        if damage > 0:
            risk_count += 1
            risky_wrong_count += 1
        else:
            safe_wrong_count += 1
    return {
        "first_correct_attempt": first_correct,
        "success_within_limit": first_correct is not None,
        "n_wrong_before_correct": wrong_before_correct,
        "risk_any": risk_count > 0,
        "risk_count": risk_count,
        "safe_wrong_count": safe_wrong_count,
        "risky_wrong_count": risky_wrong_count,
        "damage_sum": damage_sum,
    }


def compute_initial_policy_trace_summary(
    learner: LearnerAgent,
    context: TaskContext,
    teach_case: TeachCase,
) -> TeachTraceSummary:
    query = QueryState(
        query_id=0,
        target_output=list(teach_case.example.output),
        true_program=list(teach_case.example.words),
        hp=int(context.cfg.hp_0),
        max_rounds=int(context.cfg.max_attempts_main),
        menu=copy.deepcopy(teach_case.menu),
        max_refreshes=0,
        enforce_max_refreshes=True,
    )
    policy_out = learner.get_policy_snapshot_for_query(query, rng_seed=0)
    active_menu = get_active_menu(query)
    pick_probs = np.asarray(policy_out.probs[:-1], dtype=float) if len(policy_out.probs) > 0 else np.array([])

    correct_pos = next((idx for idx, opt in enumerate(active_menu) if opt.is_correct), None)
    correct_prob = None
    correct_rank = None
    if correct_pos is not None and correct_pos < len(pick_probs):
        correct_prob = float(pick_probs[correct_pos])
        correct_rank = 1 + sum(1 for value in pick_probs if value > correct_prob + 1e-12)

    top_positions = list(np.argsort(-pick_probs)[: min(5, len(pick_probs))]) if len(pick_probs) else []
    return TeachTraceSummary(
        correct_option_index=teach_case.correct_index,
        actual_initial_correct_prob=correct_prob,
        actual_initial_correct_rank=int(correct_rank) if correct_rank is not None else None,
        actual_initial_top_option_indices=[int(active_menu[pos].index) for pos in top_positions],
        actual_initial_top_option_probs=[float(pick_probs[pos]) for pos in top_positions],
    )


def probe_teach_case_difficulty(
    base_learner: LearnerAgent,
    context: TaskContext,
    teach_case: TeachCase,
    probe_mode: str = "initial_rank",
) -> Dict[str, object]:
    learner = copy.deepcopy(base_learner)
    initial = compute_initial_policy_trace_summary(
        learner=learner,
        context=context,
        teach_case=teach_case,
    )
    payload: Dict[str, object] = {
        "probe_initial_correct_prob": initial.actual_initial_correct_prob,
        "probe_initial_correct_rank": initial.actual_initial_correct_rank,
        "probe_initial_top_option_indices": list(initial.actual_initial_top_option_indices),
        "probe_initial_top_option_probs": list(initial.actual_initial_top_option_probs),
        "probe_no_tutor_unlimited_tau": None,
    }
    if str(probe_mode) != "unlimited_tau":
        return payload

    learner = copy.deepcopy(base_learner)
    block = _run_single_query(
        learner,
        context,
        teach_case.example,
        teach_case.menu,
        max_attempts=int(context.cfg.teach_menu_size),
        phase="teach",
        remove_wrong_after_reveal=context.cfg.remove_wrong_after_reveal,
    )
    metrics = summarize_teach_block(block)
    payload["probe_no_tutor_unlimited_tau"] = metrics.get("first_correct_attempt")
    return payload


def extract_attempt_policy_trace(
    block: BlockState,
    teach_case: TeachCase,
) -> List[Dict[str, object]]:
    snapshots = list(getattr(block, "_policy_snapshots", []) or [])
    pick_steps = [step for step in block.learner_trace if step.action == "pick"]
    trace: List[Dict[str, object]] = []
    pick_idx = 0
    for snap in snapshots:
        if getattr(snap, "learner_action", None) != "pick":
            continue
        probs = np.asarray(getattr(snap, "pick_probs", None))
        if probs.size == 0:
            continue
        pick_probs = probs[:-1] if len(probs) == len(snap.option_indices) + 1 else probs
        option_indices = [int(x) for x in snap.option_indices]
        correct_pos = next((idx for idx, opt_idx in enumerate(option_indices) if opt_idx == teach_case.correct_index), None)
        correct_prob = None
        correct_rank = None
        if correct_pos is not None and correct_pos < len(pick_probs):
            correct_prob = float(pick_probs[correct_pos])
            correct_rank = 1 + sum(1 for value in pick_probs if value > correct_prob + 1e-12)
        top_positions = list(np.argsort(-pick_probs)[: min(5, len(pick_probs))]) if len(pick_probs) else []
        step = pick_steps[pick_idx] if pick_idx < len(pick_steps) else None
        trace.append(
            {
                "attempt": len(trace) + 1,
                "correct_prob": correct_prob,
                "correct_rank": None if correct_rank is None else int(correct_rank),
                "top_option_indices": [option_indices[pos] for pos in top_positions],
                "top_option_probs": [float(pick_probs[pos]) for pos in top_positions],
                "chosen_option_index": None if step is None or step.pick_index is None else int(step.pick_index),
                "chosen_correct": None if step is None or step.correct is None else bool(step.correct),
            }
        )
        pick_idx += 1
    return trace


def extract_teach_trace_summary(
    block: BlockState,
    teach_case: TeachCase,
    initial_trace: Optional[TeachTraceSummary] = None,
) -> TeachTraceSummary:
    picks = [step for step in block.learner_trace if step.action == "pick"]
    semantic_updates_attempted = sum(
        1 for step in block.learner_trace
        if bool(getattr(step, "semantic_update_attempted", False))
    )
    semantic_updates_applied = sum(
        1 for step in block.learner_trace
        if bool(getattr(step, "semantic_update_applied", False))
    )
    actual_picks = [int(step.pick_index) for step in picks if step.pick_index is not None]
    pick_correct_flags = [bool(step.correct) for step in picks]
    query = block.queries[0] if block.queries else None
    wrong_outputs: List[List[str]] = []
    if query is not None:
        wrong_outputs = [list(event.revealed_output) for event in query.reveal_history]
    first_correct = None
    for idx, step in enumerate(picks, start=1):
        if step.correct and first_correct is None:
            first_correct = idx
    return TeachTraceSummary(
        correct_option_index=teach_case.correct_index,
        actual_initial_correct_prob=None if initial_trace is None else initial_trace.actual_initial_correct_prob,
        actual_initial_correct_rank=None if initial_trace is None else initial_trace.actual_initial_correct_rank,
        actual_initial_top_option_indices=[] if initial_trace is None else list(initial_trace.actual_initial_top_option_indices),
        actual_initial_top_option_probs=[] if initial_trace is None else list(initial_trace.actual_initial_top_option_probs),
        attempt_policy_trace=extract_attempt_policy_trace(block, teach_case),
        actual_picks=actual_picks,
        pick_correct_flags=pick_correct_flags,
        selected_wrong_outputs=wrong_outputs,
        actual_first_correct_attempt=first_correct,
        semantic_updates_attempted=semantic_updates_attempted,
        semantic_updates_applied=semantic_updates_applied,
    )


def evaluate_direct(learner: LearnerAgent, eval_items: Optional[Iterable[EvalItem]]) -> EvalMetrics:
    items = list(eval_items or [])
    if not items:
        return EvalMetrics()
    scorer = getattr(learner, "_scorer", None)
    if scorer is None:
        return EvalMetrics(n_items=len(items))

    exact = 0
    total_cells = 0
    correct_cells = 0
    by_diff_counts: Dict[str, int] = {}
    by_diff_correct: Dict[str, int] = {}
    by_group_counts: Dict[str, int] = {}
    by_group_exact: Dict[str, int] = {}
    by_group_total_cells: Dict[str, int] = {}
    by_group_correct_cells: Dict[str, int] = {}
    for item in items:
        pred = scorer.predict_output(list(item.words))
        source = str(getattr(item, "source", "") or "default")
        group = source.split(":", 1)[0] if ":" in source else source
        by_diff_counts[item.difficulty] = by_diff_counts.get(item.difficulty, 0) + 1
        by_group_counts[group] = by_group_counts.get(group, 0) + 1
        if pred == list(item.output):
            exact += 1
            by_diff_correct[item.difficulty] = by_diff_correct.get(item.difficulty, 0) + 1
            by_group_exact[group] = by_group_exact.get(group, 0) + 1
        total_cells += max(len(item.output), len(pred))
        overlap = min(len(item.output), len(pred))
        matched_cells = sum(1 for i in range(overlap) if pred[i] == item.output[i])
        correct_cells += matched_cells
        by_group_total_cells[group] = by_group_total_cells.get(group, 0) + max(len(item.output), len(pred))
        by_group_correct_cells[group] = by_group_correct_cells.get(group, 0) + matched_cells
    by_diff = {
        diff: by_diff_correct.get(diff, 0) / max(count, 1)
        for diff, count in by_diff_counts.items()
    }
    exact_by_group = {
        group: by_group_exact.get(group, 0) / max(count, 1)
        for group, count in by_group_counts.items()
    }
    cell_by_group = {
        group: by_group_correct_cells.get(group, 0) / max(by_group_total_cells.get(group, 0), 1)
        for group in by_group_counts
    }
    return EvalMetrics(
        exact_acc=exact / max(len(items), 1),
        cell_acc=correct_cells / max(total_cells, 1),
        by_difficulty=by_diff,
        exact_by_group=exact_by_group,
        cell_by_group=cell_by_group,
        n_items_by_group=by_group_counts,
        cell_correct_by_group=by_group_correct_cells,
        cell_total_by_group=by_group_total_cells,
        n_items=len(items),
    )
