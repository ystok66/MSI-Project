import numpy as np

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.state import QueryState
from cls_option_tutor.experiments.condition_overrides import (
    apply_condition_overrides,
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.interfaces import Example, LearnerStep, RevealEvent
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.experiments.metrics_extractors import compute_6fg_metrics


class _DummyScorer:
    def __init__(self):
        self.incremental_calls = 0
        self.negative_calls = 0

    def incremental_study(self, examples, n_em_override=None):
        self.incremental_calls += 1

    def score_option(self, target_output, option_text):
        return 0.0

    def add_negative_evidence(self, words, target_output, weight=1.0):
        self.negative_calls += 1


def _make_cfg(mode="budgeted_v1"):
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.reveal_learning_mode = "cortex_em"
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    cfg.learner.pedagogical_feedback_mode = mode
    cfg.learner.eta_reveal = 1.0
    cfg.learner.eta_correct_pick = 1.0
    cfg.learner.rho_assist = 1.0
    cfg.env.H_0 = 5
    return cfg


def _make_qs():
    return QueryState(
        query_id=0,
        target_output=["red", "blue"],
        true_program=["correct_prog"],
        hp=5,
        max_rounds=3,
    )


def _make_reveal(index, *, risk_class=0):
    return RevealEvent(
        round_t=0,
        option_index=index,
        option_text=["wrong", str(index)],
        revealed_output=["out", str(index)],
        damage=risk_class,
        expected_damage=float(risk_class),
        danger_vec=np.zeros(4),
        risk_class=risk_class,
    )


def _make_learner(mode="budgeted_v1"):
    learner = LearnerAgent(cfg=_make_cfg(mode=mode), seed=0)
    learner._scorer = _DummyScorer()
    return learner


def test_budgeted_condition_suffix_routes_and_strips_for_scripted_protocols():
    class Dummy:
        pass

    cfg = Dummy()
    cfg.env = Dummy()
    cfg.learner = Dummy()
    cfg.tutor = Dummy()
    cfg.env.highlight_mode = "diagnostic"
    cfg.env.highlight_strength = 1.0
    cfg.env.diagnostic_quota_strict = False
    cfg.env.T_max = 3
    cfg.learner.eta_reveal = 1.0
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    cfg.learner.pedagogical_feedback_mode = "raw"
    cfg.tutor.tutor_lg_mode = "off"
    cfg.tutor.postreveal_value_mode = "legacy"
    cfg.tutor.mix_target_mode = "current"
    cfg.tutor.postreveal_info_weight = 0.0
    cfg.tutor.use_bayesian_postreveal_value = False
    cfg.tutor.joint_mix_replay_gate = False
    cfg.tutor.direct_mix_selector = False
    cfg.tutor.productive_allow_planning = False
    cfg.tutor.max_highlight_cells = 2
    cfg.tutor.force_postreveal_action = "none"
    cfg.tutor.protect_safe_diag_hard_guard = False

    apply_condition_overrides(cfg, "no_tutor_reveal_budgeted_tmax4")
    assert cfg.learner.pedagogical_feedback_mode == "budgeted_v1"
    assert cfg.env.T_max == 4

    resolved = resolve_condition_alias("w1_sc_diagnostic_safe_budgeted")
    assert resolved == "script_wrong1_self_correct_diagnostic_safe_budgeted"
    protocol = extract_scripted_protocol_name("script_wrong1_self_correct_diagnostic_safe_budgeted_tmax5")
    assert protocol == "script_wrong1_self_correct_diagnostic_safe"


def test_budgeted_safe_diag_reveal_consumes_contrastive_ticket():
    learner = _make_learner(mode="budgeted_v1")
    qs = _make_qs()
    qs.option_diag_labels = {1: "safe_diagnostic_wrong"}
    qs.reveal_history = [_make_reveal(1, risk_class=1)]

    meta = learner._handle_reveal(Example(words=["wrong", "1"], output=["out", "1"]), qs=qs)

    assert meta["semantic_credit"] > 0.0
    assert meta["semantic_credit_reason"] == "safe_diag"
    assert meta["semantic_update_attempted"] is True
    assert meta["semantic_update_applied"] is True
    assert meta["contrastive_ticket_consumed"] is True
    assert qs.contrastive_update_used is True
    assert learner._scorer.incremental_calls == 1


def test_budgeted_same_wrong_reveal_gets_zero_credit():
    learner = _make_learner(mode="budgeted_v1")
    qs = _make_qs()
    qs.option_diag_labels = {1: "safe_diagnostic_wrong"}
    qs.reveal_history = [_make_reveal(1, risk_class=1), _make_reveal(1, risk_class=1)]

    meta = learner._handle_reveal(Example(words=["wrong", "1"], output=["out", "1"]), qs=qs)

    assert meta["feedback_category"] == "same_wrong"
    assert meta["semantic_credit"] == 0.0
    assert meta["semantic_credit_reason"] == "same_wrong"
    assert meta["semantic_update_attempted"] is False
    assert qs.contrastive_update_used is False
    assert learner._scorer.incremental_calls == 0


def test_budgeted_far_wrong_keeps_raw_feedback_but_zero_semantic_credit():
    learner = _make_learner(mode="budgeted_v1")
    qs = _make_qs()
    qs.option_diag_labels = {2: "safe_far"}
    qs.reveal_history = [_make_reveal(2, risk_class=0)]
    step = LearnerStep(round_t=0, query_id=0, action="pick", pick_index=2, correct=False, hp_before=5, hp_after=5)

    meta = learner._handle_reveal(Example(words=["wrong", "2"], output=["out", "2"]), qs=qs)
    learner._apply_feedback_meta(qs, step, meta)

    assert meta["feedback_category"] == "far_wrong"
    assert meta["semantic_credit"] == 0.0
    assert step.raw_feedback_kind == "wrong_reveal"
    assert step.feedback_category == "far_wrong"
    assert step.semantic_update_applied is False
    assert learner._scorer.incremental_calls == 0


def test_budgeted_correct_pick_consumes_positive_ticket_once():
    learner = _make_learner(mode="budgeted_v1")
    qs = _make_qs()

    meta_first = learner._handle_correct_pick(qs)
    meta_second = learner._handle_correct_pick(qs)

    assert meta_first["semantic_credit"] == 0.5
    assert meta_first["positive_ticket_consumed"] is True
    assert meta_first["semantic_update_attempted"] is True
    assert qs.positive_update_used is True
    assert meta_second["semantic_credit"] == 0.0
    assert meta_second["semantic_credit_reason"] == "ticket_spent"
    assert learner._scorer.incremental_calls in (0, 1)


def test_budgeted_self_correct_path_can_use_both_tickets():
    learner = _make_learner(mode="budgeted_v1")
    qs = _make_qs()
    qs.option_diag_labels = {1: "safe_diagnostic_wrong"}
    qs.reveal_history = [_make_reveal(1, risk_class=1)]
    qs.learning_event_source = "scripted_self_correct"

    wrong_meta = learner._handle_reveal(Example(words=["wrong", "1"], output=["out", "1"]), qs=qs)
    correct_meta = learner._handle_correct_pick(qs)

    assert wrong_meta["contrastive_ticket_consumed"] is True
    assert correct_meta["positive_ticket_consumed"] is True
    assert qs.contrastive_update_used is True
    assert qs.positive_update_used is True
    assert correct_meta["semantic_credit"] == 1.0
    assert learner._scorer.incremental_calls == 2


def test_raw_mode_preserves_full_reveal_credit():
    learner = _make_learner(mode="raw")
    qs = _make_qs()
    qs.option_diag_labels = {2: "safe_far"}
    qs.reveal_history = [_make_reveal(2, risk_class=0)]

    meta = learner._handle_reveal(Example(words=["wrong", "2"], output=["out", "2"]), qs=qs)

    assert meta["semantic_credit"] == 1.0
    assert meta["semantic_credit_reason"] == "raw_mode"
    assert meta["contrastive_ticket_consumed"] is False
    assert learner._scorer.incremental_calls == 1


def test_budget_metrics_follow_trace_annotations():
    qs = _make_qs()
    qs.contrastive_update_used = True
    qs.positive_update_used = True
    block = type("Block", (), {})()
    block.obs_phase_queries = 0
    block.teach_phase_queries = 1
    block.eval_phase_queries = 0
    block.queries = [qs]
    block.tutor_trace = []
    block._decision_trace = []
    block._grace_metrics = {}
    block.learner_trace = [
        LearnerStep(
            round_t=0,
            query_id=0,
            action="pick",
            pick_index=1,
            correct=False,
            hp_before=5,
            hp_after=4,
            raw_feedback_kind="wrong_reveal",
            feedback_category="safe_diag",
            semantic_credit=0.8,
            semantic_credit_type="contrastive",
            semantic_update_attempted=True,
            semantic_update_applied=True,
            contrastive_ticket_consumed=True,
        ),
        LearnerStep(
            round_t=1,
            query_id=0,
            action="pick",
            pick_index=0,
            correct=True,
            hp_before=4,
            hp_after=4,
            raw_feedback_kind="correct_pick",
            feedback_category="correct_after_feedback",
            semantic_credit=1.0,
            semantic_credit_type="positive",
            semantic_update_attempted=True,
            semantic_update_applied=True,
            positive_ticket_consumed=True,
        ),
    ]

    metrics = compute_6fg_metrics(block)
    assert metrics["RawWrongRevealCount"] == 1
    assert metrics["SemanticWrongUpdateCount"] == 1
    assert metrics["RawCorrectPickCount"] == 1
    assert metrics["SemanticCorrectUpdateCount"] == 1
    assert metrics["ContrastiveTicketUsedRate"] == 1.0
    assert metrics["PositiveTicketUsedRate"] == 1.0
    assert metrics["ProductiveRevealRate"] == 1.0
