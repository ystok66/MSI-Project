from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MazeScenarioConfig:
    width: int = 15
    height: int = 15
    risk_dim: int = 8
    n_safe_types: int = 3
    n_trap_types: int = 2
    trap_density: float = 0.12
    cluster_std: float = 0.35
    obs_noise: float = 0.25
    view_radius: int = 2
    hp: int = 3
    time_limit_scale: float = 2.5
    extra_loop_prob: float = 0.08
    teach_episodes: int = 4
    eval_same_map_episodes: int = 4
    eval_new_map_episodes: int = 4
    warning_horizon: int = 3
    learner_risk_weight: float = 4.0
    learner_unknown_penalty: float = 0.8
    learner_info_bonus: float = 0.25
    learner_revisit_penalty: float = 0.15
    tutor_risk_budget: float = 1.0
    tutor_info_credit: float = 0.4
    tutor_warning_cost: float = 0.1
    tutor_rollout_horizon: int = 6
    tutor_top_k_paths: int = 2
    tutor_max_candidates: int = 8
    tutor_waypoint_cooldown_steps: int = 6
    tutor_max_waypoints_per_episode: int = 3
    tutor_profile_count: int = 3
    tutor_safety_shield_enabled: bool = False
    tutor_catastrophe_threshold: float = 0.35
    tutor_catastrophe_damage_threshold: float = 2.0
    tutor_frontier_only_waypoint: bool = False
    tutor_warning_actionability_threshold: float = 0.0
    tutor_waypoint_damage_veto_margin: float = float("inf")
    learner_warning_suspicion_weight: float = 2.0
    learner_warning_suspicion_mode: str = "persistent"
    learner_warning_suspicion_decay: float = 1.0
    learner_consolidation_mode: str = "none"
    learner_long_term_memory_weight: float = 0.35
    learner_autonomy_assist_discount: float = 0.05
    learner_enable_objective_learning_events: bool = True
    learner_use_long_term_route_graph: bool = True
    learner_use_landmark_graph: bool = True
    warning_update_mode: str = "effective_sample"
    warning_eta0: float = 0.35
    warning_kl_epsilon: float = 1e-6
    teach_time_limit_scale: float = 1.0
    eval_time_limit_scale: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.width < 7 or self.height < 7:
            raise ValueError("maze must be at least 7x7")
        if self.width % 2 == 0 or self.height % 2 == 0:
            raise ValueError("maze width and height must be odd for DFS carving")
        if self.n_trap_types < 1:
            raise ValueError("need at least one trap type")
        if self.risk_dim < 2:
            raise ValueError("risk_dim must be >= 2")
