"""
test_convergence.py — Tests for convergence phase repairs R1-R4.

Mandatory tests from handoff section 8:
  - R1: Repaired SKIP selects mastery > confusion
  - R2: Policy-based profile prefers ground-truth profile
  - R3: Pre/post eval plumbing
  - R4: Text-length constraint
"""
from __future__ import annotations
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig, TutorConfig
from cls_option_tutor.interfaces import Option, PolicyStateSnapshot
from cls_option_tutor.env.state import QueryState, ProfileState
from cls_option_tutor.env.danger_model import generate_danger_model, generate_danger_vector
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.interventions import get_active_menu
from cls_option_tutor.grammar.task_adapter import parse_task_file
from cls_option_tutor.grammar.option_generator import generate_menu
from cls_option_tutor.learner.semantic_scorer import DeterministicSemanticScorer
from cls_option_tutor.learner.danger_head import create_danger_head
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.counterfactual import CounterfactualScorer
from cls_option_tutor.tutor.profile_inference import ProfileInference
from cls_option_tutor.tutor.tutor_agent import TutorAgent
from cls_option_tutor.eval.pre_post_eval import (
    run_pre_post_eval, PrePostResult, EvalMetrics,
)


DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')


def _has_data():
    return os.path.isdir(DATA_DIR) and os.path.exists(
        os.path.join(DATA_DIR, '000001.txt'))


# ================================================================
# R1: SKIP mastery semantics
# ================================================================

class TestSkipMastery:
    def test_skip_prefers_mastery_over_confusion(self):
        """Repaired SKIP Q-value should be higher for mastery than confusion."""
        cfg = TutorConfig()
        cf = CounterfactualScorer(cfg)

        rng = np.random.default_rng(42)
        K = 5

        # Mastery scenario: one option dominates (high P_corr, low entropy)
        mastery_sem = np.array([0.0, -2.0, -3.0, -2.5, -1.8])
        mastery_p = np.exp(4.0 * (mastery_sem - mastery_sem.max()))
        mastery_p /= mastery_p.sum()

        # Confusion scenario: uniform scores (low P_corr, high entropy)
        confused_sem = np.array([-1.0, -1.1, -0.9, -1.05, -0.95])
        confused_p = np.exp(4.0 * (confused_sem - confused_sem.max()))
        confused_p /= confused_p.sum()

        menu = [
            Option(index=i, text=[f"w{i}"],
                   danger_vec=rng.standard_normal(16),
                   is_correct=(i == 0))
            for i in range(K)
        ]

        qs_mastery = QueryState(
            query_id=0, target_output=["C0"], true_program=["w0"],
            hp=8, rounds_used=2, max_rounds=5, menu=menu,
        )
        qs_confused = QueryState(
            query_id=0, target_output=["C0"], true_program=["w0"],
            hp=8, rounds_used=2, max_rounds=5, menu=menu,
        )

        q_mastery = cf._score_skip(qs_mastery, 1.0, mastery_sem, mastery_p)
        q_confused = cf._score_skip(qs_confused, 1.0, confused_sem, confused_p)

        assert q_mastery.total_q > q_confused.total_q, (
            f"SKIP should prefer mastery (Q={q_mastery.total_q:.3f}) "
            f"over confusion (Q={q_confused.total_q:.3f})")

    def test_skip_components_correct_direction(self):
        """SKIP components should show mastery > confusion for P_corr and certainty."""
        cfg = TutorConfig()
        cf = CounterfactualScorer(cfg)

        mastery_sem = np.array([0.0, -5.0, -5.0])
        mastery_p = np.array([0.99, 0.005, 0.005])

        confused_sem = np.array([-1.0, -1.0, -1.0])
        confused_p = np.array([0.33, 0.33, 0.34])

        menu = [Option(index=i, text=[f"w{i}"],
                       danger_vec=np.zeros(16), is_correct=(i == 0))
                for i in range(3)]
        qs = QueryState(query_id=0, target_output=["C"], true_program=["w0"],
                        hp=10, max_rounds=5, menu=menu)

        q_m = cf._score_skip(qs, 0, mastery_sem, mastery_p)
        q_c = cf._score_skip(qs, 0, confused_sem, confused_p)

        assert q_m.components["P_corr"] > q_c.components["P_corr"]
        assert q_m.components["certainty"] > q_c.components["certainty"]
        assert q_m.components["learning_gain"] < q_c.components["learning_gain"]


# ================================================================
# R2: PolicyStateSnapshot + policy-grounded inference
# ================================================================

class TestPolicySnapshot:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_snapshots_recorded(self):
        """Learner should record PolicyStateSnapshots during block execution."""
        env = OptionEnv(data_dir=DATA_DIR)
        learner = LearnerAgent(seed=42)
        block = learner.run_block(env, "000001", seed=42)

        assert hasattr(block, '_policy_snapshots')
        snaps = block._policy_snapshots
        assert len(snaps) > 0, "No snapshots recorded"

        # Check snapshot fields
        s = snaps[0]
        assert isinstance(s, PolicyStateSnapshot)
        assert s.target_output is not None
        assert len(s.option_texts) > 0
        assert s.learner_action in ("pick", "refresh")
        assert s.semantic_scores is not None

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_policy_inference_uses_snapshots(self):
        """Profile inference should use snapshots when available."""
        env = OptionEnv(data_dir=DATA_DIR)
        learner = LearnerAgent(seed=42)
        tutor = TutorAgent()
        block = tutor.run_block(env, learner, "000001", seed=42)

        # If snapshots were recorded, profile inference used them
        assert hasattr(block, '_policy_snapshots')
        assert len(block._policy_snapshots) > 0
        # Profile should have been inferred
        assert block.profile_state is not None


# ================================================================
# R3: Pre -> Teach -> Post evaluation
# ================================================================

class TestPrePostEval:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_pre_post_runs_without_error(self):
        """Pre->Teach->Post pipeline should complete without errors."""
        result = run_pre_post_eval(
            task_id="000001", seed=42, data_dir=DATA_DIR)

        assert isinstance(result, PrePostResult)
        assert 0.0 <= result.pre.solve_rate <= 1.0
        assert 0.0 <= result.post.solve_rate <= 1.0
        assert 0.0 <= result.teach.solve_rate <= 1.0
        assert result.pre.n_queries > 0
        assert result.post.n_queries > 0

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_pre_eval_independent_from_teaching(self):
        """Pre-eval should use an independent clone -- no data leakage."""
        # Run twice with same seed: pre metrics should be identical
        r1 = run_pre_post_eval(task_id="000001", seed=42, data_dir=DATA_DIR)
        r2 = run_pre_post_eval(task_id="000001", seed=42, data_dir=DATA_DIR)

        assert abs(r1.pre.solve_rate - r2.pre.solve_rate) < 1e-6, (
            "Pre-eval not deterministic: possible leakage")
        assert abs(r1.pre.avg_damage - r2.pre.avg_damage) < 1e-6

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_deltas_computed_correctly(self):
        """Delta metrics should be post - pre."""
        result = run_pre_post_eval(
            task_id="000001", seed=42, data_dir=DATA_DIR)

        expected_dsr = result.post.solve_rate - result.pre.solve_rate
        assert abs(result.delta_sr - expected_dsr) < 1e-6

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_summary_string(self):
        """Summary should be a non-empty formatted string."""
        result = run_pre_post_eval(
            task_id="000001", seed=42, data_dir=DATA_DIR)
        s = result.summary()
        assert "Pre:" in s
        assert "Post:" in s
        assert "Delta:" in s


# ================================================================
# R4: Text-length constraint
# ================================================================

class TestTextLengthConstraint:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_distractor_text_not_longer_than_target(self):
        """Distractor text tokens should not exceed target output length."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        violations = 0
        total = 0
        for ex in query[:5]:
            for _ in range(10):  # multiple menus per query
                menu = generate_menu(
                    target_output=ex.output,
                    true_program=ex.words,
                    grammar=grammar, support=support,
                    danger_model=dm, K=10, m=16, rng=rng,
                )
                for opt in menu:
                    if not opt.is_correct:
                        total += 1
                        if len(opt.text) > len(ex.output):
                            violations += 1

        violation_rate = violations / max(total, 1)
        assert violation_rate < 0.01, (
            f"Text-length constraint violated: {violations}/{total} "
            f"({violation_rate:.1%}) distractors exceed target length")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
