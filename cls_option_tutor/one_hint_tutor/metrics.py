from __future__ import annotations

import math


def attempt_time_reward(first_correct_attempt, target_attempt: int, sigma_tau: float) -> float:
    if first_correct_attempt is None:
        return 0.0
    sigma = max(float(sigma_tau), 1e-6)
    return math.exp(-((float(first_correct_attempt) - float(target_attempt)) ** 2) / (2.0 * sigma * sigma))


def _attempt_limit_from_stats(aggregate: dict, cfg, attempt_limit: int | None = None) -> int:
    if attempt_limit is not None:
        return max(1, int(attempt_limit))
    raw = [float(x) for x in aggregate.get("pred_p_tau_1_to_6", [])]
    if raw:
        return max(1, len(raw))
    return max(1, int(getattr(cfg, "max_attempts_main", 6)))


def _tau_vector(aggregate: dict, cfg, attempt_limit: int | None = None) -> list[float]:
    T = _attempt_limit_from_stats(aggregate, cfg, attempt_limit=attempt_limit)
    raw = [float(x) for x in aggregate.get("pred_p_tau_1_to_6", [])]
    if len(raw) < T:
        raw = raw + [0.0] * (T - len(raw))
    return raw[:T]


def band_success_prob(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    tau_min = max(1, int(getattr(cfg, "target_tau_min", 3)))
    # Intentionally use the evaluated rollout horizon, not cfg.target_tau_max,
    # so a no-tutor bonus baseline with T+H attempts gets credit over 3..T+H.
    tau_max = max(tau_min, _attempt_limit_from_stats(aggregate, cfg, attempt_limit=attempt_limit))
    return float(sum(tau[tau_min - 1 : tau_max]))


def early_success_prob(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    tau_min = max(1, int(getattr(cfg, "target_tau_min", 3)))
    if tau_min <= 1:
        return 0.0
    return float(sum(tau[: tau_min - 1]))


def exact_target_prob(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    target = int(getattr(cfg, "target_attempt", len(tau)))
    if target < 1 or target > len(tau):
        return 0.0
    return float(tau[target - 1])


def soft_tau_score(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    center = float(getattr(cfg, "soft_tau_center", getattr(cfg, "target_attempt", len(tau))))
    sigma = max(float(getattr(cfg, "soft_tau_sigma", getattr(cfg, "sigma_tau", 1.0))), 1e-6)
    score = 0.0
    for idx, prob in enumerate(tau, start=1):
        score += float(prob) * math.exp(-((float(idx) - center) ** 2) / (2.0 * sigma * sigma))
    return float(score)


def tau_le2_prob(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    if "pred_tau_le2_exact" in aggregate:
        return float(aggregate.get("pred_tau_le2_exact", 0.0))
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    return float(sum(tau[:2]))


def wrong_before_correct_mean(aggregate: dict) -> float:
    if "wrong_before_correct_mean" in aggregate:
        return float(aggregate.get("wrong_before_correct_mean", 0.0))
    return float(aggregate.get("safe_wrong_mean", 0.0)) + float(aggregate.get("risk_count_mean", 0.0))


def expected_attempt_cost(aggregate: dict, cfg, attempt_limit: int | None = None) -> float:
    tau = _tau_vector(aggregate, cfg, attempt_limit=attempt_limit)
    T = len(tau)
    succ_mass = max(0.0, min(1.0, float(sum(tau))))
    fail_mass = max(0.0, 1.0 - succ_mass)
    expected = 0.0
    for idx, prob in enumerate(tau, start=1):
        expected += float(idx) * float(prob)
    # Treat failure as using the whole budget and still not solving the item.
    expected += float(T + 1) * fail_mass
    return float(expected)


def compute_absolute_hint_utility(
    aggregate: dict,
    cfg,
    attempt_limit: int | None = None,
) -> float:
    succ_prob = float(aggregate.get("success_prob", 0.0))
    fail_prob = max(0.0, 1.0 - succ_prob)
    expected_eval_exact = float(aggregate.get("eval_exact_acc", 0.0)) if cfg.eval_aware else 0.0
    expected_eval_cell = float(aggregate.get("eval_cell_acc", 0.0)) if cfg.eval_aware else 0.0
    expected_safe_wrong = float(aggregate.get("safe_wrong_mean", 0.0))
    time_reward = float(aggregate.get("time_reward_mean", 0.0))
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    band_prob = band_success_prob(aggregate, cfg, attempt_limit=attempt_limit)
    early_prob = early_success_prob(aggregate, cfg, attempt_limit=attempt_limit)
    exact_prob = exact_target_prob(aggregate, cfg, attempt_limit=attempt_limit)
    soft_tau = soft_tau_score(aggregate, cfg, attempt_limit=attempt_limit)
    tau2_prob = tau_le2_prob(aggregate, cfg, attempt_limit=attempt_limit)
    expected_tau_cost = expected_attempt_cost(aggregate, cfg, attempt_limit=attempt_limit)
    wrong_before_correct = wrong_before_correct_mean(aggregate)
    initial_margin = float(aggregate.get("initial_correct_margin_mean", 0.0))
    collapse_penalty = conservative_reveal_penalty(aggregate, cfg)

    if utility_mode == "delta_vs_no_tutor_bonus":
        # This mode intentionally rewards two distinct things:
        # 1) overall bounded success within the available horizon;
        # 2) landing inside the target pedagogical band rather than succeeding
        #    too early or too late.
        utility = (
            cfg.lambda_success * succ_prob
            + float(getattr(cfg, "lambda_band", cfg.lambda_success)) * band_prob
            + float(getattr(cfg, "lambda_exact6", 0.0)) * exact_prob
            + cfg.lambda_exposure * expected_safe_wrong
            + cfg.lambda_eval * expected_eval_exact
            - cfg.lambda_fail * fail_prob
            - cfg.lambda_early * early_prob
            - float(getattr(cfg, "lambda_collapse", 1.0)) * collapse_penalty
        )
    elif utility_mode == "advantage_delta":
        utility = (
            cfg.lambda_success * succ_prob
            + float(getattr(cfg, "lambda_eval_cell", cfg.lambda_eval)) * expected_eval_cell
            + float(getattr(cfg, "lambda_soft_tau", cfg.lambda_time)) * soft_tau
            + cfg.lambda_exposure * expected_safe_wrong
            - cfg.lambda_fail * fail_prob
            - float(getattr(cfg, "lambda_collapse", 1.0)) * collapse_penalty
        )
    elif utility_mode == "advantage_fast_success":
        utility = (
            float(getattr(cfg, "lambda_fast_success_T", 8.0)) * succ_prob
            + float(getattr(cfg, "lambda_fast_tau2", 10.0)) * tau2_prob
            - float(getattr(cfg, "lambda_fast_wrong", 4.0)) * wrong_before_correct
            + float(getattr(cfg, "lambda_fast_margin", 2.0)) * initial_margin
            - float(getattr(cfg, "lambda_fast_collapse", 0.0)) * collapse_penalty
        )
    elif utility_mode == "advantage_transfer":
        utility = (
            float(getattr(cfg, "lambda_transfer_success", 2.0)) * succ_prob
            + float(getattr(cfg, "lambda_transfer_eval_cell", 6.0)) * expected_eval_cell
            + float(getattr(cfg, "lambda_transfer_band", 4.0)) * band_prob
            + float(getattr(cfg, "lambda_transfer_exposure", 2.0)) * expected_safe_wrong
            - float(getattr(cfg, "lambda_transfer_early", 4.0)) * early_prob
            - float(getattr(cfg, "lambda_transfer_collapse", 3.0)) * collapse_penalty
        )
    elif utility_mode == "advantage_mix":
        fast_utility = (
            float(getattr(cfg, "lambda_fast_success_T", 8.0)) * succ_prob
            + float(getattr(cfg, "lambda_fast_tau2", 10.0)) * tau2_prob
            - float(getattr(cfg, "lambda_fast_wrong", 4.0)) * wrong_before_correct
            + float(getattr(cfg, "lambda_fast_margin", 2.0)) * initial_margin
            - float(getattr(cfg, "lambda_fast_collapse", 0.0)) * collapse_penalty
        )
        transfer_utility = (
            float(getattr(cfg, "lambda_transfer_success", 2.0)) * succ_prob
            + float(getattr(cfg, "lambda_transfer_eval_cell", 6.0)) * expected_eval_cell
            + float(getattr(cfg, "lambda_transfer_band", 4.0)) * band_prob
            + float(getattr(cfg, "lambda_transfer_exposure", 2.0)) * expected_safe_wrong
            - float(getattr(cfg, "lambda_transfer_early", 4.0)) * early_prob
            - float(getattr(cfg, "lambda_transfer_collapse", 3.0)) * collapse_penalty
        )
        eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
        utility = eta * fast_utility + (1.0 - eta) * transfer_utility
    elif utility_mode == "fast_success":
        utility = (
            float(getattr(cfg, "lambda_fast_success", 20.0)) * succ_prob
            + float(getattr(cfg, "lambda_fast_early", 5.0)) * early_prob
        )
    elif utility_mode == "min_updates":
        utility = (
            float(getattr(cfg, "lambda_min_updates_success", 20.0)) * succ_prob
            - float(getattr(cfg, "lambda_min_updates_wrong", 3.0)) * expected_safe_wrong
            - float(getattr(cfg, "lambda_min_updates_tau", 1.5)) * expected_tau_cost
            - float(getattr(cfg, "lambda_min_updates_collapse", 3.0)) * collapse_penalty
        )
    elif utility_mode == "band_delta":
        utility = (
            float(getattr(cfg, "lambda_band", cfg.lambda_success)) * band_prob
            + float(getattr(cfg, "lambda_exact6", 0.0)) * exact_prob
            + cfg.lambda_exposure * expected_safe_wrong
            + cfg.lambda_eval * expected_eval_exact
            - cfg.lambda_fail * fail_prob
            - cfg.lambda_early * early_prob
        )
    elif utility_mode == "success_gated":
        utility = (
            cfg.lambda_success * succ_prob
            - cfg.lambda_fail * fail_prob
            + cfg.lambda_time * time_reward
            + cfg.lambda_exposure * expected_safe_wrong
            + cfg.lambda_eval * expected_eval_exact
        )
    else:
        utility = (
            cfg.lambda_success * succ_prob
            + cfg.lambda_eval * expected_eval_exact
            + cfg.lambda_exposure * expected_safe_wrong
            + 0.5 * time_reward
            - cfg.lambda_fail * fail_prob
        )

    if utility_mode not in {
        "band_delta",
        "delta_vs_no_tutor_bonus",
        "advantage_delta",
        "advantage_fast_success",
        "advantage_transfer",
        "advantage_mix",
    } and aggregate.get("mean_first_correct_attempt") is not None:
        mean_tau = float(aggregate["mean_first_correct_attempt"])
        utility -= cfg.lambda_early * max(0.0, float(cfg.target_attempt) - mean_tau)
    utility -= cfg.hint_cost
    if utility_mode not in {
        "delta_vs_no_tutor_bonus",
        "advantage_delta",
        "advantage_fast_success",
        "advantage_transfer",
        "advantage_mix",
        "fast_success",
    }:
        utility -= collapse_penalty

    if cfg.use_risk:
        utility -= cfg.beta_any_risk * float(aggregate.get("risk_any_prob", 0.0))
        utility -= cfg.beta_risk_count * float(aggregate.get("risk_count_mean", 0.0))
        utility -= cfg.beta_damage * float(aggregate.get("damage_mean", 0.0))
    return float(utility)


def compute_hint_utility(
    aggregate: dict,
    cfg,
    *,
    no_hint_stats: dict | None = None,
    bonus_baseline_stats: dict | None = None,
) -> float:
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    absolute = compute_absolute_hint_utility(aggregate, cfg)
    if utility_mode not in {
        "delta_vs_no_tutor_bonus",
        "advantage_delta",
        "advantage_fast_success",
        "advantage_transfer",
        "advantage_mix",
        "fast_success",
        "min_updates",
    }:
        return absolute
    baseline = bonus_baseline_stats if bonus_baseline_stats is not None else no_hint_stats
    if baseline is None:
        return absolute
    delta = float(absolute - compute_absolute_hint_utility(baseline, cfg))
    if utility_mode in {"advantage_delta", "advantage_transfer", "advantage_mix"}:
        if utility_mode == "advantage_mix":
            eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
            # Fast-dominant mix modes intentionally allow aggressive early success.
            # Do not reintroduce the balanced/transfer early-no-transfer penalty.
            if eta >= 0.75:
                return delta
        eval_cell_delta = float(aggregate.get("eval_cell_acc", 0.0)) - float(baseline.get("eval_cell_acc", 0.0))
        if (
            early_success_prob(aggregate, cfg) > float(getattr(cfg, "early_success_reject_prob", 0.40))
            and eval_cell_delta < float(getattr(cfg, "early_success_eval_margin", 0.01))
        ):
            delta -= float(getattr(cfg, "early_no_transfer_penalty", 4.0)) * float(
                early_success_prob(aggregate, cfg)
            )
    return delta


def conservative_reveal_penalty(aggregate: dict, cfg) -> float:
    weight = float(getattr(cfg, "conservative_reveal_penalty_weight", 0.0))
    if weight <= 0.0:
        return 0.0
    probs = [p for p in aggregate.get("pred_attempt_correct_prob_mean", [])]
    vals = [float(p) for p in probs if p is not None]
    if len(vals) < 2:
        return 0.0

    margin = float(getattr(cfg, "conservative_reveal_monotone_margin", 0.0))
    first_jump_weight = float(getattr(cfg, "conservative_reveal_first_jump_weight", 1.0))
    raw_penalty = 0.0

    first_gain = vals[1] - vals[0] - margin
    if first_gain > 0.0:
        raw_penalty += first_jump_weight * first_gain

    prev = vals[1]
    for cur in vals[2:]:
        gain = cur - prev - margin
        if gain > 0.0:
            raw_penalty += gain
        prev = cur
    return weight * raw_penalty


def success_gate_threshold(no_hint_stats: dict, cfg) -> float:
    return max(
        float(no_hint_stats.get("success_prob", 0.0)) + float(getattr(cfg, "success_delta_min", 0.0)),
        float(getattr(cfg, "success_abs_min", 0.0)),
    )


def passes_success_gate(
    candidate_stats: dict,
    no_hint_stats: dict,
    cfg,
    reference_stats: dict | None = None,
) -> bool:
    utility_mode = str(getattr(cfg, "utility_mode", "legacy"))
    ref = reference_stats or no_hint_stats
    transfer_like = utility_mode == "advantage_transfer"
    if utility_mode == "advantage_mix":
        eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
        transfer_like = eta <= 0.25
    if transfer_like and str(getattr(cfg, "transfer_gate_mode", "default")) == "eval_delta":
        eval_cell_delta = float(candidate_stats.get("eval_cell_acc", 0.0)) - float(ref.get("eval_cell_acc", 0.0))
        success_floor = max(
            float(getattr(cfg, "transfer_success_floor", 0.15)),
            float(ref.get("success_prob", 0.0)) - float(getattr(cfg, "transfer_success_slack", 0.05)),
        )
        return (
            eval_cell_delta >= float(getattr(cfg, "transfer_delta_eval_min", 0.005))
            and float(candidate_stats.get("success_prob", 0.0)) >= success_floor
        )
    if utility_mode in {"band_delta", "delta_vs_no_tutor_bonus"}:
        if utility_mode == "delta_vs_no_tutor_bonus" and bool(getattr(cfg, "eval_aware", False)):
            eval_delta = float(candidate_stats.get("eval_exact_acc", 0.0)) - float(ref.get("eval_exact_acc", 0.0))
            if (
                early_success_prob(candidate_stats, cfg) > float(getattr(cfg, "early_success_reject_prob", 0.40))
                and eval_delta < float(getattr(cfg, "early_success_eval_margin", 0.01))
            ):
                return False
        return band_success_prob(candidate_stats, cfg) >= (
            band_success_prob(ref, cfg) + float(getattr(cfg, "delta_band_min", 0.0))
        )
    if utility_mode == "advantage_delta":
        eval_cell_delta = float(candidate_stats.get("eval_cell_acc", 0.0)) - float(ref.get("eval_cell_acc", 0.0))
        if (
            early_success_prob(candidate_stats, cfg) > float(getattr(cfg, "early_success_reject_prob", 0.40))
            and eval_cell_delta < float(getattr(cfg, "early_success_eval_margin", 0.01))
        ):
            return False
        return True
    if utility_mode == "advantage_transfer":
        eval_cell_delta = float(candidate_stats.get("eval_cell_acc", 0.0)) - float(ref.get("eval_cell_acc", 0.0))
        if (
            early_success_prob(candidate_stats, cfg) > float(getattr(cfg, "early_success_reject_prob", 0.40))
            and eval_cell_delta < float(getattr(cfg, "early_success_eval_margin", 0.01))
        ):
            return False
        return True
    if utility_mode == "advantage_mix":
        eta = max(0.0, min(1.0, float(getattr(cfg, "utility_mix_eta", 0.5))))
        if eta >= 0.75:
            return True
        eval_cell_delta = float(candidate_stats.get("eval_cell_acc", 0.0)) - float(ref.get("eval_cell_acc", 0.0))
        if (
            early_success_prob(candidate_stats, cfg) > float(getattr(cfg, "early_success_reject_prob", 0.40))
            and eval_cell_delta < float(getattr(cfg, "early_success_eval_margin", 0.01))
        ):
            return False
        return True
    if utility_mode == "advantage_fast_success":
        return True
    if utility_mode in {"fast_success", "min_updates"}:
        return True
    if utility_mode != "success_gated":
        return True
    return float(candidate_stats.get("success_prob", 0.0)) >= success_gate_threshold(ref, cfg)
