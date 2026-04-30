from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..interfaces import Example, Option


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    placement: str  # prefix | postfix | infix
    pattern: Tuple[str, ...]
    template: Tuple[str, ...]
    score: int


@dataclass
class TaskContext:
    task_id: str
    grammar: object
    support_examples: List[Example]
    query_examples: List[Example]
    synthetic_examples: List[Example]
    operator_specs: List[OperatorSpec]
    example_pools: Dict[str, List[Example]]
    env: object
    cfg: object


@dataclass
class ObservationCase:
    example: Example
    menu: List[Option]
    difficulty: str


@dataclass
class ObservationRun:
    case: ObservationCase
    block: object
    steps: List[object]


@dataclass
class TeachCase:
    example: Example
    menu: List[Option]
    difficulty: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def correct_index(self) -> Optional[int]:
        for opt in self.menu:
            if opt.is_correct:
                return opt.index
        return None


@dataclass
class EvalItem:
    words: List[str]
    output: List[str]
    difficulty: str
    source: str


@dataclass
class HintCandidate:
    example: Example
    difficulty: str
    kind: str  # free | menu_wrong | menu_correct_ceiling | operator_probe | target_neighborhood | direct_answer
    source_index: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    exact_acc: float = 0.0
    cell_acc: float = 0.0
    by_difficulty: Dict[str, float] = field(default_factory=dict)
    exact_by_group: Dict[str, float] = field(default_factory=dict)
    cell_by_group: Dict[str, float] = field(default_factory=dict)
    n_items_by_group: Dict[str, int] = field(default_factory=dict)
    cell_correct_by_group: Dict[str, int] = field(default_factory=dict)
    cell_total_by_group: Dict[str, int] = field(default_factory=dict)
    n_items: int = 0


@dataclass
class TeachTraceSummary:
    correct_option_index: Optional[int] = None
    actual_initial_correct_prob: Optional[float] = None
    actual_initial_correct_rank: Optional[int] = None
    actual_initial_top_option_indices: List[int] = field(default_factory=list)
    actual_initial_top_option_probs: List[float] = field(default_factory=list)
    attempt_policy_trace: List[Dict[str, object]] = field(default_factory=list)
    actual_picks: List[int] = field(default_factory=list)
    pick_correct_flags: List[bool] = field(default_factory=list)
    selected_wrong_outputs: List[List[str]] = field(default_factory=list)
    actual_first_correct_attempt: Optional[int] = None
    semantic_updates_attempted: int = 0
    semantic_updates_applied: int = 0


@dataclass
class ConditionResult:
    condition: str
    first_correct_attempt: Optional[int]
    success_within_limit: bool
    n_wrong_before_correct: int
    safe_wrong_count: int
    risky_wrong_count: int
    risk_any: bool
    risk_count: int
    damage_sum: int
    eval_metrics: Optional[EvalMetrics] = None
    hint_used: bool = False
    hint_kind: str = "none"
    hint_difficulty: str = "none"
    hint_source_index: Optional[int] = None
    teach_trace_summary: Optional[TeachTraceSummary] = None
    failure_type: Optional[str] = None
    failure_details: Dict[str, object] = field(default_factory=dict)


@dataclass
class PosteriorSummary:
    profile_posterior: Dict[str, float]
    profile_entropy: float
    full_action_nll: Optional[float]
    pick_nll: Optional[float]


@dataclass
class PlannerCounters:
    n_cls_predict_calls: int = 0
    n_cls_score_calls: int = 0
    n_cls_deepcopy_calls: int = 0
    n_incremental_study_calls: int = 0
    n_score_table_hits: int = 0
    n_score_table_misses: int = 0
    n_rollout_paths: int = 0
    n_terminal_paths: int = 0
    n_candidates_prefiltered: int = 0
    n_candidates_proxy_evaluated: int = 0
    n_candidates_refined: int = 0
    stage0_wall_time: float = 0.0
    stage1_wall_time: float = 0.0
    stage2_wall_time: float = 0.0
    select_hint_wall_time: float = 0.0


@dataclass
class PlannerPrediction:
    pred_p_success_T6: float = 0.0
    pred_tau_mean: Optional[float] = None
    pred_tau_mode: Optional[int] = None
    pred_p_tau_1_to_6: List[float] = field(default_factory=list)
    pred_p_tau_band: float = 0.0
    pred_p_tau_early: float = 0.0
    pred_attempt_correct_prob_mean: List[Optional[float]] = field(default_factory=list)
    pred_attempt_correct_rank_mean: List[Optional[float]] = field(default_factory=list)
    pred_correct_prob_no_hint_mean: Optional[float] = None
    pred_correct_prob_after_hint_mean: Optional[float] = None
    pred_correct_rank_no_hint_mean: Optional[float] = None
    pred_correct_rank_after_hint_mean: Optional[float] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    hint_quality_tags: Dict[str, object] = field(default_factory=dict)
    kept_profiles: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class PlannerStageScore:
    stage: str
    hint_kind: str
    hint_difficulty: str
    source_index: Optional[int]
    score: float


@dataclass
class HintPlanResult:
    selected_hint: Optional[HintCandidate]
    selected_utility: float
    no_hint_utility: float
    delta_vs_no_hint: float
    candidate_scores: List[Dict[str, object]] = field(default_factory=list)
    planner_prediction: Optional[PlannerPrediction] = None
    planner_counters: Optional[PlannerCounters] = None
    stage_scores: List[PlannerStageScore] = field(default_factory=list)


@dataclass
class ExperimentResult:
    task_id: str
    seed: int
    prelearn_examples: List[Example]
    observation_examples: List[Example]
    teach_example: Example
    teach_case_metadata: Dict[str, object]
    eval_items: List[EvalItem]
    posterior: PosteriorSummary
    plan: HintPlanResult
    conditions: Dict[str, ConditionResult]
