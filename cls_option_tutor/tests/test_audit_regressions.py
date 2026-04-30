import numpy as np

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.state import BlockState, QueryState
from cls_option_tutor.eval.autonomous_probe import _clone_learner
from cls_option_tutor.interfaces import Example, LearnerStep, Option, RiskHintEvent, TutorStep
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.learner.policy import PolicyOutput
from cls_option_tutor.tutor.sparse_tutor_phase import (
    PHASE_POST_REVEAL_PROTECT_AND_CUE,
    PHASE_PRE_REVEAL_ALLOW,
    infer_pedagogical_phase,
)
from cls_option_tutor.tutor.sparse_tutor_scoring import compute_postreveal_shift_decomp


def _make_option(index, *, is_correct=False, risk_class=0, danger=None):
    return Option(
        index=index,
        text=["tok", str(index)],
        danger_vec=np.array(danger if danger is not None else [0.0, 0.0, 0.0, 0.0], dtype=float),
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
    )
    qs.menu = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=4, danger=[1.0, 0.0, 0.0, 0.0]),
        _make_option(2, is_correct=False, risk_class=1, danger=[0.0, 1.0, 0.0, 0.0]),
    ]
    qs.option_diag_labels = {
        0: "",
        1: "high_risk_lure",
        2: "safe_diagnostic_wrong",
    }
    return qs


class _EnvStub:
    def learner_act(self, block, action, pick_index=None):
        qs = block.current_query
        return LearnerStep(
            round_t=qs.rounds_used,
            query_id=qs.query_id,
            action=action,
            pick_index=pick_index,
            hp_before=qs.hp,
            hp_after=qs.hp,
            menu_size=len(qs.menu),
        )


def _make_refresh_policy_output(qs):
    k = len(qs.menu)
    return PolicyOutput(
        action="refresh",
        pick_index=None,
        utilities=np.zeros(k + 1, dtype=float),
        probs=np.ones(k + 1, dtype=float) / float(k + 1),
        semantic_scores=np.zeros(k, dtype=float),
        danger_preds=np.zeros(k, dtype=float),
        danger_uncs=np.zeros(k, dtype=float),
    )


def test_clone_learner_freeze_memory_disables_policy_memory():
    learner = LearnerAgent(cfg=FullConfig(), seed=0)
    learner.policy.memory = object()

    clone = _clone_learner(
        learner,
        freeze_semantic=False,
        freeze_risk=False,
        freeze_memory=True,
    )

    assert clone.policy.memory is None


def test_samewrong_drop_requires_specific_last_wrong_index():
    qs = _make_query()
    active = qs.menu
    wait_probs = np.array([0.2, 0.2, 0.4, 0.2], dtype=float)
    action_probs = np.array([0.2, 0.2, 0.0, 0.6], dtype=float)

    decomp = compute_postreveal_shift_decomp(
        active,
        wait_probs,
        action_probs,
        qs.option_diag_labels,
        last_wrong_index=None,
        hp_scale=qs.hp,
        ban_target_index=2,
    )

    assert decomp["same_wrong_drop"] == 0.0


def test_phase_inference_prefers_post_reveal_protect_and_cue():
    cfg = FullConfig()
    cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    qs = _make_query()
    qs.post_reveal_phase = True
    qs.n_safe_diag_wrong_reveals = 1
    probs = np.array([0.2, 0.5, 0.3], dtype=float)

    phase = infer_pedagogical_phase(qs, qs.menu, probs, cfg)

    assert phase == PHASE_POST_REVEAL_PROTECT_AND_CUE


def test_phase_inference_phasecalib_surfaces_missed_prereveal_allow():
    cfg = FullConfig()
    cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    cfg.tutor.phase_allow_family_override = True

    qs = QueryState(
        query_id=0,
        target_output=["red", "blue"],
        true_program=["prog"],
        hp=5,
        max_rounds=5,
    )
    qs.option_diag_labels = {
        0: "",
        1: "safe_diagnostic_wrong",
        2: "bounded_diagnostic_wrong",
        3: "safe_far",
    }
    qs.menu = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=1),
        _make_option(2, is_correct=False, risk_class=1),
        _make_option(3, is_correct=False, risk_class=1),
    ]
    probs = np.array([0.2, 0.3, 0.2, 0.05], dtype=float)

    phase = infer_pedagogical_phase(qs, qs.menu, probs, cfg)

    assert phase == PHASE_PRE_REVEAL_ALLOW


def test_phase_inference_phasecalib_does_not_override_protect():
    cfg = FullConfig()
    cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    cfg.tutor.phase_allow_family_override = True
    qs = _make_query()
    probs = np.array([0.2, 0.5, 0.3], dtype=float)

    phase = infer_pedagogical_phase(qs, qs.menu, probs, cfg)

    assert phase == "PROTECT"


def test_handle_reveal_posterior_shift_uses_query_target_output():
    learner = LearnerAgent(cfg=FullConfig(), seed=0)
    learner.cfg.learner.reveal_learning_mode = "cortex_em"
    learner.cfg.learner.eta_reveal = 1.0

    class DummyScorer:
        def __init__(self):
            self.targets = []

        def score_option(self, target_output, option_text):
            self.targets.append(list(target_output))
            return 0.0

        def incremental_study(self, examples):
            return None

    learner._scorer = DummyScorer()

    qs = _make_query()
    qs.target_output = ["target", "cells"]
    example = Example(words=["wrong"], output=["revealed", "cells"])

    learner._handle_reveal(example, qs=qs)

    assert learner._scorer.targets
    assert all(target == ["target", "cells"] for target in learner._scorer.targets)


def test_risk_hint_history_is_processed_once_per_hint():
    cfg = FullConfig()
    learner = LearnerAgent(cfg=cfg, seed=0)
    qs = _make_query()
    qs.risk_hint_history.append(RiskHintEvent(round_t=0, option_index=1, eta=0.8))
    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )

    class DummyDangerHead:
        def __init__(self):
            self.calls = []

        def __call__(self, v, eta=0.8):
            self.calls.append((tuple(np.asarray(v).tolist()), eta))

    hint_recorder = DummyDangerHead()
    learner.policy.observe_risk_hint = hint_recorder
    learner.policy.compute_policy = lambda *args, **kwargs: _make_refresh_policy_output(qs)

    env = _EnvStub()
    learner.act(block, env)
    learner.act(block, env)

    assert len(hint_recorder.calls) == 1


def test_persistent_ban_trace_is_processed_once():
    cfg = FullConfig()
    cfg.learner.rho_ban_prior = 0.5
    cfg.env.danger_dim = 4
    learner = LearnerAgent(cfg=cfg, seed=0)
    qs = _make_query()
    block = BlockState(
        queries=[qs],
        current_query_idx=0,
        obs_phase_queries=0,
        teach_phase_queries=1,
        eval_phase_queries=0,
    )
    block.tutor_trace.append(TutorStep(round_t=0, query_id=qs.query_id, action="BAN", ban_index=1))

    learner.policy.compute_policy = lambda *args, **kwargs: _make_refresh_policy_output(qs)

    env = _EnvStub()
    learner.act(block, env)
    first = learner._persistent_ban.copy()
    learner.act(block, env)
    second = learner._persistent_ban.copy()

    assert np.allclose(first, second)
