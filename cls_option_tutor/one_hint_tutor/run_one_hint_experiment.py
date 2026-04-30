from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import OneHintConfig
from .experiment_matrix import run_experiment_scenario
from .protocol import run_one_hint_experiment


def _condition_summary(condition) -> dict | None:
    if condition is None:
        return None
    payload = {
        "first_correct_attempt": condition.first_correct_attempt,
        "success_within_limit": condition.success_within_limit,
        "eval_exact_acc": None if condition.eval_metrics is None else condition.eval_metrics.exact_acc,
        "eval_cell_acc": None if condition.eval_metrics is None else condition.eval_metrics.cell_acc,
        "failure_type": condition.failure_type,
        "failure_details": dict(condition.failure_details),
    }
    if condition.teach_trace_summary is not None:
        payload["teach_trace_summary"] = {
            "correct_option_index": condition.teach_trace_summary.correct_option_index,
            "actual_initial_correct_prob": condition.teach_trace_summary.actual_initial_correct_prob,
            "actual_initial_correct_rank": condition.teach_trace_summary.actual_initial_correct_rank,
            "actual_initial_top_option_indices": list(condition.teach_trace_summary.actual_initial_top_option_indices),
            "actual_initial_top_option_probs": list(condition.teach_trace_summary.actual_initial_top_option_probs),
            "attempt_policy_trace": list(condition.teach_trace_summary.attempt_policy_trace),
            "actual_picks": list(condition.teach_trace_summary.actual_picks),
            "pick_correct_flags": list(condition.teach_trace_summary.pick_correct_flags),
            "selected_wrong_outputs": [list(x) for x in condition.teach_trace_summary.selected_wrong_outputs],
            "actual_first_correct_attempt": condition.teach_trace_summary.actual_first_correct_attempt,
        }
    return payload


def _summary_payload(result) -> dict:
    selected_hint = None
    if result.plan.selected_hint is not None:
        selected_hint = {
            "kind": result.plan.selected_hint.kind,
            "difficulty": result.plan.selected_hint.difficulty,
            "words": result.plan.selected_hint.example.words,
            "metadata": dict(result.plan.selected_hint.metadata),
        }

    planner_prediction = None
    if result.plan.planner_prediction is not None:
        planner_prediction = {
            "pred_p_success_T6": result.plan.planner_prediction.pred_p_success_T6,
            "pred_tau_mean": result.plan.planner_prediction.pred_tau_mean,
            "pred_tau_mode": result.plan.planner_prediction.pred_tau_mode,
            "pred_p_tau_1_to_6": list(result.plan.planner_prediction.pred_p_tau_1_to_6),
            "pred_p_tau_band": result.plan.planner_prediction.pred_p_tau_band,
            "pred_p_tau_early": result.plan.planner_prediction.pred_p_tau_early,
            "pred_attempt_correct_prob_mean": list(result.plan.planner_prediction.pred_attempt_correct_prob_mean),
            "pred_attempt_correct_rank_mean": list(result.plan.planner_prediction.pred_attempt_correct_rank_mean),
            "pred_correct_prob_no_hint_mean": result.plan.planner_prediction.pred_correct_prob_no_hint_mean,
            "pred_correct_prob_after_hint_mean": result.plan.planner_prediction.pred_correct_prob_after_hint_mean,
            "pred_correct_rank_no_hint_mean": result.plan.planner_prediction.pred_correct_rank_no_hint_mean,
            "pred_correct_rank_after_hint_mean": result.plan.planner_prediction.pred_correct_rank_after_hint_mean,
            "abstained": result.plan.planner_prediction.abstained,
            "abstain_reason": result.plan.planner_prediction.abstain_reason,
            "hint_quality_tags": dict(result.plan.planner_prediction.hint_quality_tags),
            "kept_profiles": list(result.plan.planner_prediction.kept_profiles),
        }

    planner_counters = None
    if result.plan.planner_counters is not None:
        planner_counters = asdict(result.plan.planner_counters)

    top_candidates = []
    for item in result.plan.candidate_scores[:3]:
        top_candidates.append(
            {
                "kind": item.get("kind"),
                "difficulty": item.get("difficulty"),
                "source_index": item.get("source_index"),
                "metadata": dict(item.get("metadata", {})),
                "stage": item.get("stage"),
                "utility": item.get("utility"),
                "selection_score": item.get("selection_score"),
                "reranker_score": item.get("reranker_score"),
                "band_success_prob": item.get("band_success_prob"),
                "early_success_prob": item.get("early_success_prob"),
                "conservative_reveal_penalty": item.get("conservative_reveal_penalty"),
                "success_prob": item.get("success_prob"),
                "eval_exact_acc": item.get("eval_exact_acc"),
                "eval_cell_acc": item.get("eval_cell_acc"),
                "initial_correct_prob_mean": item.get("initial_correct_prob_mean"),
                "mean_first_correct_attempt": item.get("mean_first_correct_attempt"),
                "initial_correct_rank_mean": item.get("initial_correct_rank_mean"),
                "collapse_adjustment_mean": item.get("collapse_adjustment_mean"),
                "first_reveal_cache_hit_prob": item.get("first_reveal_cache_hit_prob"),
            }
        )

    return {
        "task_id": result.task_id,
        "seed": result.seed,
        "n_prelearn_examples": len(result.prelearn_examples),
        "n_observation_examples": len(result.observation_examples),
        "n_eval_items": len(result.eval_items),
        "teach_case_metadata": dict(result.teach_case_metadata),
        "posterior": {
            "profile_entropy": result.posterior.profile_entropy,
            "pick_nll": result.posterior.pick_nll,
            "profile_posterior": result.posterior.profile_posterior,
        },
        "plan": {
            "selected_hint": selected_hint,
            "selected_utility": result.plan.selected_utility,
            "no_hint_utility": result.plan.no_hint_utility,
            "delta_vs_no_hint": result.plan.delta_vs_no_hint,
            "n_candidates_reported": len(result.plan.candidate_scores),
            "planner_prediction": planner_prediction,
            "planner_counters": planner_counters,
            "top_candidates": top_candidates,
        },
        "conditions": {
            name: _condition_summary(condition)
            for name, condition in result.conditions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot hint tutor experiment.")
    parser.add_argument("--task", required=True, help="Task id, e.g. 000001")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hint-mode",
        default="free",
        choices=[
            "free",
            "menu_wrong",
            "menu_all",
            "operator_probe",
            "target_neighborhood",
            "target_neighborhood_loose",
            "target_neighborhood_rank_filtered",
            "target_neighborhood_robust_filtered",
            "combined",
            "menu_correct_ceiling",
            "none",
        ],
    )
    parser.add_argument("--hint-families", default="free", help="Comma-separated families for combined mode.")
    parser.add_argument("--ceiling-mode", default="none", choices=["none", "menu_correct", "oracle_best_candidate", "oracle_best_family", "two_hint_pre", "after_first_wrong_hint"])
    parser.add_argument("--teach-difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--teach-menu-size", type=int, default=20)
    parser.add_argument("--max-attempts-main", type=int, default=6)
    parser.add_argument("--max-attempts-no-tutor-extra", type=int, default=7)
    parser.add_argument("--menu-difficulty-mode", default="default", choices=["default", "rank_stratified"])
    parser.add_argument("--teach-probe-mode", default="initial_rank", choices=["initial_rank", "unlimited_tau"])
    parser.add_argument("--target-initial-rank-min", type=int, default=5)
    parser.add_argument("--target-initial-rank-max", type=int, default=12)
    parser.add_argument("--target-no-tutor-unlimited-tau-min", type=int, default=5)
    parser.add_argument("--target-no-tutor-unlimited-tau-max", type=int, default=10)
    parser.add_argument("--teach-menu-build-trials", type=int, default=24)
    parser.add_argument("--n-obs", type=int, default=4)
    parser.add_argument("--obs-difficulty", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--prelearn-profile", default="custom", choices=["custom", "4", "6", "8", "12"])
    parser.add_argument("--prefilter-top-k", type=int, default=4)
    parser.add_argument("--prefilter-min-per-family", type=int, default=1)
    parser.add_argument("--disable-family-aware-prefilter", action="store_true")
    parser.add_argument("--n-pre-easy", type=int, default=1)
    parser.add_argument("--n-pre-medium", type=int, default=2)
    parser.add_argument("--n-pre-hard", type=int, default=1)
    parser.add_argument("--free-hint-pool-easy", type=int, default=2)
    parser.add_argument("--free-hint-pool-medium", type=int, default=3)
    parser.add_argument("--free-hint-pool-hard", type=int, default=3)
    parser.add_argument("--operator-probe-limit", type=int, default=6)
    parser.add_argument("--target-neighborhood-limit", type=int, default=8)
    parser.add_argument("--target-neighborhood-atom-replacements", type=int, default=4)
    parser.add_argument("--target-neighborhood-min-words", type=int, default=2)
    parser.add_argument("--target-neighborhood-rank-min", type=int, default=3)
    parser.add_argument("--target-neighborhood-rank-max", type=int, default=10)
    parser.add_argument("--target-neighborhood-robust-rank-max", type=int, default=10)
    parser.add_argument("--target-neighborhood-robust-collapse-ratio", type=float, default=0.863)
    parser.add_argument("--menu-wrong-limit", type=int, default=6)
    parser.add_argument("--menu-correct-ceiling-limit", type=int, default=1)
    parser.add_argument("--random-hard-n", type=int, default=1)
    parser.add_argument("--random-same-pool-n", type=int, default=1)
    parser.add_argument(
        "--reveal-learning-mode",
        default="cortex_em",
        choices=["cortex_em", "negative_memory", "delayed_study", "off", "nonreveal_negative"],
    )
    parser.add_argument("--negative-evidence-mode", default="off", choices=["off", "exact_program_target"])
    parser.add_argument("--eta-reveal", type=float, default=1.0)
    parser.add_argument("--correct-pick-learning-mode", default="cortex_em", choices=["off", "cortex_em"])
    parser.add_argument("--no-abstain", action="store_true", help="Force planner to pick the best candidate instead of abstaining.")
    parser.add_argument("--use-risk", action="store_true")
    parser.add_argument("--rollout-mode", default="mc", choices=["beam", "mc"])
    parser.add_argument(
        "--utility-mode",
        default="band_delta",
        choices=["legacy", "success_gated", "band_delta", "delta_vs_no_tutor_bonus", "advantage_delta"],
    )
    parser.add_argument("--oracle-metric", default="composite", choices=["success", "band", "eval_cell", "composite"])
    parser.add_argument("--lambda-eval-cell", type=float, default=2.0)
    parser.add_argument("--lambda-soft-tau", type=float, default=2.0)
    parser.add_argument("--soft-tau-center", type=float, default=4.5)
    parser.add_argument("--soft-tau-sigma", type=float, default=1.5)
    parser.add_argument("--advantage-delta-min", type=float, default=0.0)
    parser.add_argument("--early-no-transfer-penalty", type=float, default=4.0)
    parser.add_argument("--reranker-enabled", action="store_true")
    parser.add_argument("--reranker-model-path", default="")
    parser.add_argument("--reranker-alpha", type=float, default=1.0)
    parser.add_argument("--reranker-target", default="search", choices=["search", "transfer"])
    parser.add_argument("--eval-aware", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = OneHintConfig(
        seed=args.seed,
        hint_mode=args.hint_mode,
        hint_families=tuple(part.strip() for part in args.hint_families.split(",") if part.strip()),
        ceiling_mode=args.ceiling_mode,
        teach_difficulty=args.teach_difficulty,
        teach_menu_size=args.teach_menu_size,
        max_attempts_main=args.max_attempts_main,
        max_attempts_no_tutor_extra=args.max_attempts_no_tutor_extra,
        menu_difficulty_mode=args.menu_difficulty_mode,
        teach_probe_mode=args.teach_probe_mode,
        target_initial_rank_min=args.target_initial_rank_min,
        target_initial_rank_max=args.target_initial_rank_max,
        target_no_tutor_unlimited_tau_min=args.target_no_tutor_unlimited_tau_min,
        target_no_tutor_unlimited_tau_max=args.target_no_tutor_unlimited_tau_max,
        teach_menu_build_trials=args.teach_menu_build_trials,
        n_obs=args.n_obs,
        obs_difficulty=args.obs_difficulty,
        prelearn_profile=args.prelearn_profile,
        prefilter_top_k=args.prefilter_top_k,
        prefilter_family_aware=not args.disable_family_aware_prefilter,
        prefilter_min_per_family=args.prefilter_min_per_family,
        n_pre_easy=args.n_pre_easy,
        n_pre_medium=args.n_pre_medium,
        n_pre_hard=args.n_pre_hard,
        free_hint_pool_easy=args.free_hint_pool_easy,
        free_hint_pool_medium=args.free_hint_pool_medium,
        free_hint_pool_hard=args.free_hint_pool_hard,
        operator_probe_limit=args.operator_probe_limit,
        target_neighborhood_limit=args.target_neighborhood_limit,
        target_neighborhood_atom_replacements=args.target_neighborhood_atom_replacements,
        target_neighborhood_min_words=args.target_neighborhood_min_words,
        target_neighborhood_rank_min=args.target_neighborhood_rank_min,
        target_neighborhood_rank_max=args.target_neighborhood_rank_max,
        target_neighborhood_robust_rank_max=args.target_neighborhood_robust_rank_max,
        target_neighborhood_robust_collapse_ratio=args.target_neighborhood_robust_collapse_ratio,
        menu_wrong_limit=args.menu_wrong_limit,
        menu_correct_ceiling_limit=args.menu_correct_ceiling_limit,
        random_hard_n=args.random_hard_n,
        random_same_pool_n=args.random_same_pool_n,
        reveal_learning_mode=args.reveal_learning_mode,
        negative_evidence_mode=args.negative_evidence_mode,
        eta_reveal=args.eta_reveal,
        correct_pick_learning_mode=args.correct_pick_learning_mode,
        allow_abstain=not args.no_abstain,
        use_risk=args.use_risk,
        rollout_mode=args.rollout_mode,
        utility_mode=args.utility_mode,
        oracle_metric=args.oracle_metric,
        lambda_eval_cell=args.lambda_eval_cell,
        lambda_soft_tau=args.lambda_soft_tau,
        soft_tau_center=args.soft_tau_center,
        soft_tau_sigma=args.soft_tau_sigma,
        advantage_delta_min=args.advantage_delta_min,
        early_no_transfer_penalty=args.early_no_transfer_penalty,
        reranker_enabled=args.reranker_enabled,
        reranker_model_path=args.reranker_model_path,
        reranker_alpha=args.reranker_alpha,
        reranker_target=args.reranker_target,
        eval_aware=args.eval_aware,
    )
    if str(cfg.ceiling_mode) != "none":
        payload = run_experiment_scenario(task_id=args.task, cfg=cfg, seed=args.seed)
    else:
        result = run_one_hint_experiment(task_id=args.task, cfg=cfg, seed=args.seed)
        payload = _summary_payload(result) if args.summary_only else asdict(result)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
