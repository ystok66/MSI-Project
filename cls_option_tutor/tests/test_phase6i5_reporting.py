import shutil
from pathlib import Path

import numpy as np

from cls_option_tutor.env.state import BlockState, QueryState
from cls_option_tutor.interfaces import LearnerStep, Option
from cls_option_tutor.experiments.reporting import write_summary
from cls_option_tutor.experiments.run_learning_increment_micro import _extract_observed_wrong_reveal_stats


def _make_option(index, is_correct=False, risk_class=0):
    return Option(
        index=index,
        text=["tok"],
        danger_vec=np.zeros(4),
        is_correct=is_correct,
        risk_class=risk_class,
        rendered_output=["red"],
    )


def _make_qs(query_id=0):
    qs = QueryState(
        query_id=query_id,
        target_output=["red"],
        true_program=["prog"],
        hp=5,
    )
    qs.menu = [
        _make_option(0, is_correct=True, risk_class=0),
        _make_option(1, is_correct=False, risk_class=0),
        _make_option(2, is_correct=False, risk_class=3),
    ]
    return qs


def test_extract_observed_wrong_reveal_stats_uses_actual_pick_risk():
    qs = _make_qs()
    qs.success = False
    block = BlockState(queries=[qs], obs_phase_queries=0, teach_phase_queries=1, eval_phase_queries=0)
    block.learner_trace = [
        LearnerStep(
            round_t=0,
            query_id=0,
            action="pick",
            pick_index=1,
            correct=False,
            damage=0,
            hp_before=5,
            hp_after=5,
            menu_size=3,
        ),
        LearnerStep(
            round_t=1,
            query_id=0,
            action="pick",
            pick_index=2,
            correct=False,
            damage=3,
            hp_before=5,
            hp_after=2,
            menu_size=3,
        ),
    ]

    stats = _extract_observed_wrong_reveal_stats(block, [qs])
    assert stats["wrong_reveal_count"] == 2
    assert stats["wr_risk"][0] == 1
    assert stats["wr_risk"][3] == 1
    assert stats["safe_wrong_count"] == 1
    assert stats["risky_wrong_count"] == 1


def test_write_summary_uses_semantic_and_policy_rank_columns():
    rows = [{
        "condition": "demo",
        "rho_assist": 0.3,
        "DeltaProbeSR": 0.1,
        "DeltaLocalSR": 0.2,
        "DeltaSemanticMargin": 0.3,
        "DeltaSemanticRank": 1.5,
        "DeltaPolicyRank": 0.5,
        "TeachDamage": 1.0,
        "ScriptedDamage": 0.0,
        "DeathRate": 0.0,
        "Protocol_SelfCorrectCount": 2,
        "Protocol_ThenAnswerCount": 1,
        "PostCueGuidedSCCount": 3,
        "PostCueStructProtectCount": 4,
        "CueTrajectorySuccessWithin2RoundsRate": 0.25,
        "MIXJointGateAppliedRate": 1.0,
        "MIXJointGateReplacedRate": 0.5,
        "MIXJointRegretMean": 0.12,
        "MIXJointInteractionRegretMean": 0.03,
        "WR_Risk0": 1,
        "WR_Risk1": 0,
        "WR_Risk2": 0,
        "WR_Risk3": 0,
        "WR_Risk4": 0,
        "DeathBeforeCorrect": 0,
        "Protocol_WR_Risk0": 2,
        "Protocol_WR_Risk1": 0,
        "Protocol_WR_Risk2": 0,
        "Protocol_WR_Risk3": 0,
        "Protocol_WR_Risk4": 0,
        "Protocol_DeathBeforeCorrect": 0,
    }]
    out_dir = Path("tmp_summary_test")
    out_dir.mkdir(exist_ok=True)
    try:
        path = write_summary(rows, str(out_dir), "ts", ["demo"], 20, 8)
        text = Path(path).read_text(encoding="utf-8")
        assert "dSemRank" in text
        assert "dPolRank" in text
        assert "+1.5000" in text
        assert "+0.5000" in text
        assert "Observed Wrong Reveal Risk Histogram" in text
        assert "Protocol Wrong Reveal Risk Histogram" in text
        assert "JointRegret" in text
        assert "JointIntReg" in text
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
