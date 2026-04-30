from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class OneHintConfig:
    """Config for the one-shot hint tutor protocol."""

    data_dir: str = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "BASIC",
            "cls_learner",
            "data",
        )
    )

    # Prelearn support
    prelearn_profile: str = "custom"  # custom | 4 | 6 | 8 | 12
    n_pre_easy: int = 1
    n_pre_medium: int = 2
    n_pre_hard: int = 1

    # Observation
    n_obs: int = 4
    obs_menu_size: int = 6
    obs_difficulty: str = "hard"
    obs_freeze_long_term: bool = True

    # Teach
    teach_difficulty: str = "hard"
    teach_menu_size: int = 20
    max_attempts_main: int = 6
    max_attempts_no_tutor_extra: int = 7
    menu_difficulty_mode: str = "default"  # default | rank_stratified
    teach_probe_mode: str = "initial_rank"  # initial_rank | unlimited_tau
    target_initial_rank_min: int = 5
    target_initial_rank_max: int = 12
    target_no_tutor_unlimited_tau_min: int = 5
    target_no_tutor_unlimited_tau_max: int = 10
    teach_menu_build_trials: int = 24
    remove_wrong_after_reveal: bool = True
    hint_count_budget: int = 1
    no_tutor_bonus_attempts: int = 1

    # Eval
    eval_n_per_diff: int = 10
    add_out_of_menu_eval: bool = False
    eval_aware: bool = False
    plan_candidate_limit: int = 8
    transfer_eval_proxy_mode: str = "off"  # off | static_subset | beam_leaf_subset
    transfer_eval_proxy_n_per_diff: int = 3
    transfer_eval_proxy_max_items: int = 9
    transfer_eval_proxy_beam_top_b: int = 3
    transfer_eval_proxy_beam_keep_l: int = 24
    transfer_eval_proxy_refine_top_k: int = 0  # <=0 means use all refined candidates

    # Hint space
    hint_mode: str = "free"  # free | menu_wrong | menu_all | operator_probe | target_neighborhood | target_neighborhood_loose | target_neighborhood_rank_filtered | target_neighborhood_robust_filtered | answer_neighbor_nonanswer | combined | menu_correct_ceiling | none
    hint_families: tuple[str, ...] = ("free",)
    allow_correct_hint: bool = False
    free_hint_pool_easy: int = 2
    free_hint_pool_medium: int = 3
    free_hint_pool_hard: int = 3
    operator_probe_limit: int = 6
    target_neighborhood_limit: int = 16
    target_neighborhood_atom_replacements: int = 8
    target_neighborhood_min_words: int = 2
    target_neighborhood_rank_min: int = 3
    target_neighborhood_rank_max: int = 10
    target_neighborhood_robust_rank_max: int = 10
    target_neighborhood_robust_collapse_ratio: float = 0.863
    answer_neighbor_limit: int = 8
    menu_wrong_limit: int = 6
    menu_correct_ceiling_limit: int = 1
    hint_cost: float = 0.0
    random_hint_difficulty: str = "hard"
    random_hard_n: int = 1
    random_same_pool_n: int = 1

    # Structured experiment / ceiling modes
    ceiling_mode: str = "none"  # none | menu_correct | oracle_best_candidate | oracle_best_family | two_hint_pre | after_first_wrong_hint
    oracle_metric: str = "composite"  # success | band | eval_cell | fast_success | composite
    oracle_candidate_limit: int = 12
    oracle_pair_limit: int = 8
    oracle_include_planner_reference: bool = True
    two_hint_pair_limit: int = 8
    after_first_wrong_pre_limit: int = 6
    after_first_wrong_post_limit: int = 6

    # Learner backend
    use_cls: bool = True
    n_em: int = 1
    use_hpc: bool = False
    tau_sem: float = 1.0
    reveal_learning_mode: str = "cortex_em"  # cortex_em | negative_memory | delayed_study | off | nonreveal_negative
    negative_evidence_mode: str = "off"
    correct_pick_learning_mode: str = "cortex_em"
    eta_reveal: float = 1.0
    eta_correct_pick: float = 1.0
    rho_assist: float = 1.0
    feedback_mode: str = "reveal"
    pedagogical_feedback_mode: str = "raw"

    # Inverse / shadow
    eta_prof: float = 1.0
    shadow_use_cls: bool = True
    shadow_n_em: int = 1
    shadow_use_hpc: bool = False
    planning_lambda_neg: float = 2.0
    planning_update_mode: str = "proxy"  # proxy | full_cls | lazy_cls
    profile_top_mass: float = 0.90
    profile_min_keep: int = 2
    refine_profile_top_mass: float = 0.75
    refine_profile_min_keep: int = 2

    # Planning
    planner_mode: str = "cascade"  # single_stage | cascade
    prefilter_enabled: bool = True
    prefilter_top_k: int = 4
    prefilter_family_aware: bool = True
    prefilter_min_per_family: int = 1
    prefilter_use_cached_scores: bool = True
    objective_bucketed_prefilter: bool = False
    prefilter_keep_fast: int = 0
    prefilter_keep_transfer: int = 0
    prefilter_keep_balanced: int = 0

    proxy_rollout_top_k: int = 4
    rollout_mode: str = "mc"  # mc | beam
    n_rollouts: int = 6
    beam_top_b: int = 3
    beam_keep_l: int = 24
    proxy_rollout_mode: str = "mc"  # mc | score_table_beam
    proxy_n_rollouts: int = 6
    proxy_beam_top_b: int = 3
    proxy_beam_keep_l: int = 24

    refine_enabled: bool = True
    refine_top_k: int = 2
    refine_update_mode: str = "first_reveal_cached_cls"  # proxy | first_reveal_cached_cls | lazy_cls | full_cls
    refine_n_rollouts: int = 4
    refine_beam_top_b: int = 3
    refine_beam_keep_l: int = 24
    objective_bucketed_refine: bool = False
    refine_keep_fast: int = 0
    refine_keep_transfer: int = 0
    refine_keep_balanced: int = 0
    lazy_cls_n_em_override: int = 1
    first_reveal_top_b: int = 3
    first_reveal_refine_enabled: bool = True

    cache_hint_profile_scores: bool = True
    cache_eval_proxy: bool = True
    rollout_state_mode: str = "delta"  # delta | full_clone
    planner_text_reveal_penalty: float = 2.0
    planner_output_reveal_penalty: float = 0.5
    use_reveal_collapse_prior: bool = True
    collapse_ratio_median: float = 0.863
    collapse_prior_strength: float = 1.0
    conservative_reveal_penalty_weight: float = 3.0
    conservative_reveal_first_jump_weight: float = 2.0
    conservative_reveal_monotone_margin: float = 0.01

    # Utility
    utility_mode: str = "band_delta"  # legacy | success_gated | band_delta | delta_vs_no_tutor_bonus | advantage_delta | advantage_fast_success | advantage_transfer | advantage_mix | fast_success | min_updates
    lambda_success: float = 3.0
    lambda_eval: float = 1.0
    lambda_eval_cell: float = 2.0
    lambda_exposure: float = 0.5
    lambda_fail: float = 8.0
    lambda_time: float = 2.0
    lambda_early: float = 3.0
    lambda_band: float = 8.0
    lambda_exact6: float = 2.0
    lambda_soft_tau: float = 2.0
    lambda_collapse: float = 3.0
    lambda_fast_success: float = 20.0
    lambda_fast_early: float = 5.0
    lambda_fast_success_T: float = 8.0
    lambda_fast_tau2: float = 10.0
    lambda_fast_wrong: float = 4.0
    lambda_fast_margin: float = 2.0
    lambda_fast_collapse: float = 0.0
    lambda_transfer_success: float = 2.0
    lambda_transfer_eval_cell: float = 6.0
    lambda_transfer_band: float = 4.0
    lambda_transfer_exposure: float = 2.0
    lambda_transfer_early: float = 4.0
    lambda_transfer_collapse: float = 3.0
    utility_mix_eta: float = 0.5
    utility_mix_normalize_components: bool = False
    utility_mix_normalize_eps: float = 1e-6
    lambda_min_updates_success: float = 20.0
    lambda_min_updates_wrong: float = 3.0
    lambda_min_updates_tau: float = 1.5
    lambda_min_updates_collapse: float = 3.0
    target_attempt: int = 6
    target_tau_min: int = 3
    target_tau_max: int = 6
    sigma_tau: float = 1.0
    soft_tau_center: float = 4.5
    soft_tau_sigma: float = 1.5
    delta_min_use_hint: float = 0.25
    advantage_delta_min: float = 0.0
    success_delta_min: float = 0.05
    success_abs_min: float = 0.20
    delta_band_min: float = 0.05
    allow_abstain: bool = True
    early_success_reject_prob: float = 0.40
    early_success_eval_margin: float = 0.01
    early_no_transfer_penalty: float = 4.0
    fast_pred_threshold: float = 0.25
    transfer_gate_mode: str = "default"  # default | eval_delta
    transfer_delta_eval_min: float = 0.005
    transfer_success_floor: float = 0.15
    transfer_success_slack: float = 0.05

    # Eval split diagnostics
    exposure_sensitive_eval_enabled: bool = False
    exposure_sensitive_eval_n_per_diff: int = 6

    # Regime discovery
    regime_no_tutor_success_min: float = 0.35
    regime_no_tutor_success_max: float = 0.75
    regime_oracle_success_headroom_min: float = 0.15
    regime_oracle_eval_cell_headroom_min: float = 0.03
    search_success_delta_vs_no_tutor_min: float = 0.20
    search_success_delta_vs_random_min: float = 0.10
    search_early_no_transfer_max: float = 0.10
    transfer_eval_cell_delta_vs_no_tutor_min: float = 0.03
    transfer_eval_cell_delta_vs_random_min: float = 0.03
    transfer_success_delta_vs_no_tutor_min: float = 0.0

    # Oracle-distilled reranker
    reranker_enabled: bool = False
    reranker_model_path: str = ""
    reranker_alpha: float = 1.0
    reranker_target: str = "search"

    # Failure taxonomy
    premature_success_eval_threshold: float = 0.05
    failure_collapse_prob_drop_threshold: float = 0.02
    failure_collapse_ratio_threshold: float = 0.863
    failure_collapse_rank_jump: int = 5

    # Diagnostics
    collect_planner_calibration: bool = True
    collect_cost_counters: bool = True

    # Risk
    use_risk: bool = False
    n_risk_options: int = 4
    beta_any_risk: float = 2.0
    beta_risk_count: float = 1.0
    beta_damage: float = 1.0
    hp_0: int = 5

    # Environment defaults reused from current project
    danger_dim: int = 16
    cluster_sigma: float = 0.5

    # Repro
    seed: int = 0
    common_randomness: bool = True
    seed_context_offset: int = 11
    seed_learner_offset: int = 23
    seed_prelearn_offset: int = 101
    seed_obs_offset: int = 211
    seed_teach_offset: int = 307
    seed_eval_offset: int = 401
    seed_plan_offset: int = 503
    seed_baseline_offset: int = 601
    seed_oracle_offset: int = 701
