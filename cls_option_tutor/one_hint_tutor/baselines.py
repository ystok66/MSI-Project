from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional

import numpy as np

from .experiment_presets import resolved_no_tutor_tplush_limit
from .hint_space import build_random_same_pool_candidates, sample_random_hint
from .interfaces import ConditionResult, HintPlanResult, TaskContext, TeachCase
from .learner_runner import run_teach_condition


def _condition_attempt_limit(condition_name: str, cfg) -> int:
    if condition_name in {"no_tutor_unlimited", "tutor_unlimited"}:
        return int(cfg.teach_menu_size)
    if condition_name in {"no_tutor_T", "tutor_T", "tutor_T6", "random_hard_hint_T", "random_hard_hint_T6", "random_same_pool_hint_T"}:
        return int(cfg.max_attempts_main)
    if condition_name in {"no_tutor_TplusH", "no_tutor_T7"}:
        return int(resolved_no_tutor_tplush_limit(cfg))
    return int(cfg.max_attempts_main)


def _alias_condition(condition: ConditionResult, name: str) -> ConditionResult:
    return replace(condition, condition=name)


def _first_wrong_dynamics(condition: ConditionResult) -> Dict[str, object]:
    summary = condition.teach_trace_summary
    trace = [] if summary is None else list(summary.attempt_policy_trace)
    if len(trace) < 2:
        return {
            "first_wrong_delta_prob": None,
            "first_wrong_ratio": None,
            "first_wrong_rank_jump": None,
        }
    first = trace[0]
    second = trace[1]
    if first.get("chosen_correct") is not False:
        return {
            "first_wrong_delta_prob": None,
            "first_wrong_ratio": None,
            "first_wrong_rank_jump": None,
        }
    p0 = first.get("correct_prob")
    p1 = second.get("correct_prob")
    r0 = first.get("correct_rank")
    r1 = second.get("correct_rank")
    delta = None
    ratio = None
    rank_jump = None
    if p0 is not None and p1 is not None:
        delta = float(p1) - float(p0)
        if float(p0) > 1e-12:
            ratio = float(p1) / float(p0)
    if r0 is not None and r1 is not None:
        rank_jump = int(r1) - int(r0)
    return {
        "first_wrong_delta_prob": delta,
        "first_wrong_ratio": ratio,
        "first_wrong_rank_jump": rank_jump,
    }


def classify_condition_failure(condition: ConditionResult, cfg) -> tuple[str, Dict[str, object]]:
    summary = condition.teach_trace_summary
    limit = _condition_attempt_limit(condition.condition, cfg)
    initial_rank = None if summary is None else summary.actual_initial_correct_rank
    eval_exact = None if condition.eval_metrics is None else float(condition.eval_metrics.exact_acc)
    dynamics = _first_wrong_dynamics(condition)
    details = {
        "attempt_limit": int(limit),
        "initial_correct_rank": initial_rank,
        "first_correct_attempt": condition.first_correct_attempt,
        "eval_exact_acc": eval_exact,
        **dynamics,
    }

    if condition.first_correct_attempt is not None:
        if (
            condition.first_correct_attempt <= 2
            and eval_exact is not None
            and eval_exact < float(getattr(cfg, "premature_success_eval_threshold", 0.05))
        ):
            return "premature_success_no_transfer", details
        return "success", details

    if initial_rank is not None and int(initial_rank) > int(limit):
        return "hint_did_not_make_correct_reachable", details

    delta = dynamics.get("first_wrong_delta_prob")
    ratio = dynamics.get("first_wrong_ratio")
    rank_jump = dynamics.get("first_wrong_rank_jump")
    if (
        (delta is not None and float(delta) <= -float(getattr(cfg, "failure_collapse_prob_drop_threshold", 0.02)))
        or (ratio is not None and float(ratio) <= float(getattr(cfg, "failure_collapse_ratio_threshold", 0.863)))
        or (rank_jump is not None and int(rank_jump) >= int(getattr(cfg, "failure_collapse_rank_jump", 5)))
    ):
        return "post_reveal_collapse", details

    return "search_failure", details


def attach_failure_taxonomy(conditions: Dict[str, ConditionResult], cfg) -> None:
    for condition in conditions.values():
        failure_type, failure_details = classify_condition_failure(condition, cfg)
        condition.failure_type = failure_type
        condition.failure_details = failure_details


def _repeated_condition_names(prefix: str, n: int) -> List[str]:
    count = max(1, int(n))
    return [prefix] + [f"{prefix}_rep_{idx}" for idx in range(2, count + 1)]


def _sample_random_same_pool_hints(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
    n: int,
) -> List[Optional[object]]:
    count = max(1, int(n))
    pool_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    pool = build_random_same_pool_candidates(context, teach_case, cfg, pool_rng)
    if not pool:
        return [None] * count
    hints: List[Optional[object]] = []
    for _ in range(count):
        sample_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hints.append(pool[int(sample_rng.integers(0, len(pool)))])
    return hints


def _sample_random_hints(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
    n: int,
) -> List[Optional[object]]:
    count = max(1, int(n))
    hints: List[Optional[object]] = []
    for _ in range(count):
        sample_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hints.append(sample_random_hint(context, cfg, sample_rng))
    return hints


def _run_repeated_hint_condition(
    base_learner,
    context: TaskContext,
    teach_case: TeachCase,
    eval_items,
    max_attempts: int,
    hints: Iterable[object],
    condition_prefix: str,
) -> Dict[str, ConditionResult]:
    hints_list = list(hints)
    conditions: Dict[str, ConditionResult] = {}
    names = _repeated_condition_names(condition_prefix, len(hints_list))
    for idx, hint in enumerate(hints_list):
        conditions[names[idx]] = run_teach_condition(
            base_learner,
            context,
            teach_case,
            max_attempts=max_attempts,
            hint=hint,
            eval_items=eval_items,
            condition_name=names[idx],
        )
    return conditions


def run_condition_suite(
    base_learner,
    context: TaskContext,
    teach_case: TeachCase,
    eval_items,
    plan: HintPlanResult,
    cfg,
    rng: np.random.Generator,
) -> Dict[str, ConditionResult]:
    # Keep hint-sampler randomness on dedicated branches so future changes in
    # candidate generation cannot perturb the rest of the baseline suite.
    random_hint_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    random_same_pool_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    random_hints = _sample_random_hints(
        context=context,
        cfg=cfg,
        rng=random_hint_rng,
        n=int(getattr(cfg, "random_hard_n", 1)),
    )
    random_same_pool_hints = _sample_random_same_pool_hints(
        context=context,
        teach_case=teach_case,
        cfg=cfg,
        rng=random_same_pool_rng,
        n=int(getattr(cfg, "random_same_pool_n", 1)),
    )
    selected_hint = plan.selected_hint
    no_tutor_t = run_teach_condition(
        base_learner,
        context,
        teach_case,
        max_attempts=int(cfg.max_attempts_main),
        hint=None,
        eval_items=eval_items,
        condition_name="no_tutor_T",
    )
    no_tutor_tplush = run_teach_condition(
        base_learner,
        context,
        teach_case,
        max_attempts=int(resolved_no_tutor_tplush_limit(cfg)),
        hint=None,
        eval_items=eval_items,
        condition_name="no_tutor_TplusH",
    )
    random_hard_conditions = _run_repeated_hint_condition(
        base_learner=base_learner,
        context=context,
        teach_case=teach_case,
        eval_items=eval_items,
        max_attempts=int(cfg.max_attempts_main),
        hints=random_hints,
        condition_prefix="random_hard_hint_T",
    )
    random_same_pool_conditions = _run_repeated_hint_condition(
        base_learner=base_learner,
        context=context,
        teach_case=teach_case,
        eval_items=eval_items,
        max_attempts=int(cfg.max_attempts_main),
        hints=random_same_pool_hints,
        condition_prefix="random_same_pool_hint_T",
    )
    conditions = {
        "no_tutor_unlimited": run_teach_condition(
            base_learner,
            context,
            teach_case,
            max_attempts=int(cfg.teach_menu_size),
            hint=None,
            eval_items=None,
            condition_name="no_tutor_unlimited",
        ),
        "no_tutor_T": no_tutor_t,
        "no_tutor_TplusH": no_tutor_tplush,
    }
    conditions.update(random_hard_conditions)
    conditions.update(random_same_pool_conditions)
    if selected_hint is None:
        no_hint_unlimited = conditions["no_tutor_unlimited"]
        conditions["tutor_unlimited"] = ConditionResult(
            condition="tutor_unlimited",
            first_correct_attempt=no_hint_unlimited.first_correct_attempt,
            success_within_limit=no_hint_unlimited.success_within_limit,
            n_wrong_before_correct=no_hint_unlimited.n_wrong_before_correct,
            safe_wrong_count=no_hint_unlimited.safe_wrong_count,
            risky_wrong_count=no_hint_unlimited.risky_wrong_count,
            risk_any=no_hint_unlimited.risk_any,
            risk_count=no_hint_unlimited.risk_count,
            damage_sum=no_hint_unlimited.damage_sum,
            eval_metrics=None,
            hint_used=False,
            hint_kind="none",
            hint_difficulty="none",
            hint_source_index=None,
            teach_trace_summary=no_hint_unlimited.teach_trace_summary,
        )
        conditions["tutor_T"] = ConditionResult(
            condition="tutor_T",
            first_correct_attempt=no_tutor_t.first_correct_attempt,
            success_within_limit=no_tutor_t.success_within_limit,
            n_wrong_before_correct=no_tutor_t.n_wrong_before_correct,
            safe_wrong_count=no_tutor_t.safe_wrong_count,
            risky_wrong_count=no_tutor_t.risky_wrong_count,
            risk_any=no_tutor_t.risk_any,
            risk_count=no_tutor_t.risk_count,
            damage_sum=no_tutor_t.damage_sum,
            eval_metrics=no_tutor_t.eval_metrics,
            hint_used=False,
            hint_kind="none",
            hint_difficulty="none",
            hint_source_index=None,
            teach_trace_summary=no_tutor_t.teach_trace_summary,
        )
    else:
        conditions["tutor_unlimited"] = run_teach_condition(
            base_learner,
            context,
            teach_case,
            max_attempts=int(cfg.teach_menu_size),
            hint=selected_hint,
            eval_items=None,
            condition_name="tutor_unlimited",
        )
        conditions["tutor_T"] = run_teach_condition(
            base_learner,
            context,
            teach_case,
            max_attempts=int(cfg.max_attempts_main),
            hint=selected_hint,
            eval_items=eval_items,
            condition_name="tutor_T",
        )

    if int(cfg.max_attempts_main) == 6:
        conditions["tutor_T6"] = _alias_condition(conditions["tutor_T"], "tutor_T6")
        conditions["random_hard_hint_T6"] = _alias_condition(conditions["random_hard_hint_T"], "random_hard_hint_T6")
    if int(resolved_no_tutor_tplush_limit(cfg)) == int(getattr(cfg, "max_attempts_no_tutor_extra", 7)):
        conditions["no_tutor_T7"] = _alias_condition(conditions["no_tutor_TplusH"], "no_tutor_T7")
    attach_failure_taxonomy(conditions, cfg)
    return conditions
