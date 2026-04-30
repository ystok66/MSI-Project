import numpy as np
import pytest

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.state import BlockState, QueryState
from cls_option_tutor.experiments.condition_overrides import (
    apply_condition_overrides,
    extract_scripted_protocol_name,
)
from cls_option_tutor.interfaces import Option
from cls_option_tutor.learner.policy import LearnerPolicy
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
from cls_option_tutor.tutor.sparse_tutor_horizon import compute_pre_reveal_allow_value


def _make_option(index, *, is_correct=False, risk_class=0):
    return Option(
        index=index,
        text=["tok", str(index)],
        danger_vec=np.zeros(4, dtype=float),
        is_correct=is_correct,
        risk_class=risk_class,
        rendered_output=["out", str(index)],
    )


def _make_query():
    qs = QueryState(
        query_id=0,
        target_output=["red", "blue"],
        true_program=["prog"],
        hp=4,
        max_rounds=3,
        max_refreshes=2,
    )
    qs.menu = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=4),
        _make_option(2, is_correct=False, risk_class=1),
    ]
    qs.option_diag_labels = {
        0: "",
        1: "safe_far",
        2: "safe_diagnostic_wrong",
    }
    return qs


class _ConstScorer:
    def score_option(self, target_output, option_text, attention_weights=None):
        return 0.0


class _HighDangerHead:
    def predict(self, v):
        return 10.0, 0.0


def test_condition_overrides_support_refreshcap_and_controlled_allow():
    cfg = FullConfig()
    apply_condition_overrides(
        cfg,
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_budgeted_rfcap_allowctl",
    )

    assert cfg.learner.pedagogical_feedback_mode == "budgeted_v1"
    assert cfg.env.enforce_max_refreshes is True
    assert cfg.tutor.direct_mix_selector is True
    assert cfg.tutor.productive_allow_planning is True
    assert cfg.tutor.productive_allow_mode == "controlled_v1"


def test_condition_overrides_support_consolidation_and_controlled_allow_v2():
    cfg = FullConfig()
    apply_condition_overrides(
        cfg,
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_budgeted_allowctl2_consolidate_tmax5",
    )

    assert cfg.learner.pedagogical_feedback_mode == "budgeted_v1"
    assert cfg.env.T_max == 5
    assert cfg.tutor.productive_allow_planning is True
    assert cfg.tutor.productive_allow_mode == "controlled_v2"
    assert cfg.tutor.use_postreveal_consolidation_value is True
    assert cfg.tutor.direct_mix_selector is True


def test_condition_overrides_support_native_like_allow():
    cfg = FullConfig()
    apply_condition_overrides(
        cfg,
        "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_budgeted_allowctl2_consolidate_tmax5_nativeallow",
    )

    assert cfg.learner.pedagogical_feedback_mode == "budgeted_v1"
    assert cfg.env.T_max == 5
    assert cfg.tutor.productive_allow_planning is True
    assert cfg.tutor.productive_allow_mode == "native_like_v1"
    assert cfg.tutor.direct_mix_selector is True


def test_extract_scripted_protocol_name_strips_refreshcap_suffix():
    protocol = extract_scripted_protocol_name(
        "no_tutor_reveal_budgeted_rfcap"
    )
    assert protocol == "no_tutor_reveal"


def test_policy_refresh_cap_blocks_refresh_only_when_enforced():
    cfg = FullConfig()
    cfg.learner.beta_L = 10.0
    cfg.learner.epsilon = 0.0
    policy = LearnerPolicy(cfg.learner)
    policy.scorer = _ConstScorer()
    policy.danger_head = _HighDangerHead()

    qs = _make_query()
    qs.refreshes_used = qs.max_refreshes

    out_legacy = policy.compute_policy(qs, np.random.default_rng(0))
    assert out_legacy.action == "refresh"

    qs.enforce_max_refreshes = True
    out_capped = policy.compute_policy(qs, np.random.default_rng(0))
    assert out_capped.action == "pick"


def test_env_refresh_cap_raises_when_refresh_attempt_exceeds_limit():
    cfg = FullConfig()
    cfg.env.enforce_max_refreshes = True
    env = OptionEnv(cfg=cfg)

    qs = _make_query()
    qs.enforce_max_refreshes = True
    qs.refreshes_used = qs.max_refreshes
    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )

    with pytest.raises(ValueError, match="Refresh cap exceeded"):
        env.learner_act(block, "refresh")


def test_controlled_productive_allow_preserves_safe_diag_when_info_dominates(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v1"

    qs = _make_query()
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is True


def test_controlled_productive_allow_does_not_preserve_when_harm_dominates(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v1"

    qs = _make_query()
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.65, 0.25], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False


def test_controlled_productive_allow_accepts_info_harm_boundary(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v1"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.20, 0.30, 0.50], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is True


def test_controlled_productive_allow_rejects_when_info_just_below_harm(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v1"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.20, 0.31, 0.49], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False


def test_controlled_productive_allow_diagnostic_failure_is_counted(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v1"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )
    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.build_option_mass_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is True
    assert tutor._productive_allow_diagnostic_failures == 1


def test_controlled_v2_requires_both_tickets_available(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.positive_update_used = True
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False


def test_controlled_v2_requires_rounds_left_ge_3(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.rounds_used = 1  # rounds_left = 2 under max_rounds = 3
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False


def test_controlled_v2_blocks_allow_when_predicted_loop_value_is_zero(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor.cfg.tutor._best_cue_cate_estimate = 0.0
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False


def test_controlled_v2_preserves_when_all_conditions_met(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor.cfg.tutor._best_cue_cate_estimate = 0.1
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is True


def test_native_like_allow_preserves_clean_state(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "native_like_v1"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is True
    meta = tutor._productive_allow_meta_cache[(qs.query_id, qs.rounds_used)]
    assert meta["allow_family_split"] == "NATIVE_LIKE_ALLOW"


def test_native_like_allow_blocks_mixed_prod_harm_state(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "native_like_v1"
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.55, 0.35], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False
    meta = tutor._productive_allow_meta_cache[(qs.query_id, qs.rounds_used)]
    assert meta["allow_family_split"] == "MIXED_PROD_HARM"
    assert meta["allow_reject_reason"] == "FAMILY_MIXED_PROD_HARM"


def test_controlled_v2_allow_loop_uses_expected_damage_survival(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor.cfg.tutor._best_cue_cate_estimate = 0.1
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    qs.last_reveal_option_index = 2
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False
    meta = tutor._productive_allow_meta_cache[(qs.query_id, qs.rounds_used)]
    expected_damage = 0.20 * 4.0 + 0.70 * 1.0
    expected_survive = max(0.0, (qs.hp - expected_damage) / qs.hp)
    expected_value = compute_pre_reveal_allow_value(
        0.70,
        0.0,
        expected_survive,
        1.0,
        0.1,
        0.10,
        harm_mass_wait=meta["harm_mass"],
        contrastive_ticket_available=True,
        positive_ticket_available=True,
    )
    assert meta["expected_damage_wait"] == pytest.approx(expected_damage)
    assert meta["p_survive"] == pytest.approx(expected_survive)
    assert meta["allow_loop_value"] == pytest.approx(expected_value)


def test_controlled_v2_records_allow_reject_reason_for_missing_positive_ticket(monkeypatch):
    tutor = SparseTutorAgent.__new__(SparseTutorAgent)
    tutor.cfg = FullConfig()
    tutor.cfg.tutor.productive_allow_planning = True
    tutor.cfg.tutor.productive_allow_mode = "controlled_v2"
    tutor.cfg.tutor._best_cue_cate_estimate = 0.1
    tutor._productive_allow_diagnostic_failures = 0

    qs = _make_query()
    qs.max_rounds = 5
    qs.positive_update_used = True
    active = list(qs.menu)

    monkeypatch.setattr(
        "cls_option_tutor.tutor.sparse_tutor.infer_pedagogical_phase",
        lambda *args, **kwargs: "PRE_REVEAL_ALLOW",
    )
    monkeypatch.setattr(
        tutor,
        "_compute_learner_probs",
        lambda *args, **kwargs: np.array([0.10, 0.20, 0.70], dtype=float),
    )

    assert tutor._should_preserve_productive_allow(qs, active, learner=None) is False
    meta = tutor._productive_allow_meta_cache[(qs.query_id, qs.rounds_used)]
    assert meta["allow_reject_reason"] == "NO_POSITIVE_TICKET"
    assert meta["eligible"] is False
