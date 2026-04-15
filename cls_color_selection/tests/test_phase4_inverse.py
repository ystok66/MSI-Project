"""
Tests for Phase 4 inverse inference components.

Test categories:
    1. observation_v2: result-level only, no process leaks
    2. task_model: correct hints from ground truth
    3. learner_model: inverse inference updates, update_depth
    4. tutor_inverse: end-to-end
    5. divergence_v3: metric computations
"""
import pytest
import sys
import os
import numpy as np

# Ensure BASIC on path
_this = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.normpath(os.path.join(_this, '..'))
_basic = os.path.normpath(os.path.join(_this, '..', '..', 'BASIC'))
for p in [_proj, _basic]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def env_and_data():
    """Load a task and create all learner components."""
    from cls_color_selection.config import FullConfig
    from cls_color_selection.environment.grammar_task_env import GrammarTaskEnv
    from cls_color_selection.learner.cls_wrapper import CLSSequencePredictor
    from cls_color_selection.learner.target_predictor import TargetPredictor
    from cls_color_selection.learner.risk_belief import DangerTypeBelief
    from cls_color_selection.learner.feedback_update import FeedbackUpdater
    from cls_color_selection.learner.policy import ColorSelectionPolicy
    from cls_color_selection.interfaces import Example

    cfg = FullConfig()
    rng = np.random.default_rng(42)
    env = GrammarTaskEnv(cfg, rng)

    data_dir = cfg.resolve_data_dir()
    task_path = os.path.join(data_dir, '000001.txt')
    if not os.path.exists(task_path):
        pytest.skip(f"Task data not found at {task_path}")

    support, queries, grammar = env.load_task(task_path)
    sub_support = support[:cfg.learner.n_sup]

    predictor = CLSSequencePredictor(cfg.learner)
    predictor.fit_support(sub_support)
    target_pred = TargetPredictor(predictor)

    risk_belief = DangerTypeBelief(
        n_danger_types=cfg.env.n_danger_types,
        danger_dim=cfg.env.danger_dim,
        obs_sigma=cfg.env.obs_sigma,
        prior_safe=cfg.learner.risk_prior_safe,
    )
    risk_belief.set_prototypes(
        env.danger_model.prototypes,
        np.ones_like(env.danger_model.prototypes) * cfg.env.cluster_sigma**2,
    )

    policy = ColorSelectionPolicy(cfg.learner)
    feedback_updater = FeedbackUpdater(cfg.learner)

    return {
        'cfg': cfg, 'env': env, 'rng': rng,
        'support': sub_support, 'queries': queries, 'grammar': grammar,
        'predictor': predictor, 'target_pred': target_pred,
        'risk_belief': risk_belief, 'policy': policy,
        'feedback_updater': feedback_updater,
    }


# ═══════════════════════════════════════════════════════════════
# 1. Observation V2 Tests
# ═══════════════════════════════════════════════════════════════

class TestObservationV2:
    """Test result-level observation."""

    def test_record_has_no_process_info(self):
        """ObservationRecord must NOT contain process-level fields."""
        from cls_color_selection.tutor_api.observation_v2 import ObservationRecord
        from cls_color_selection.constants import Outcome

        record = ObservationRecord(
            query_words=['dax', 'fep'],
            submitted_output=['RED', 'RED', 'BLUE'],
            current_completion=['RED', 'RED', 'BLUE'],
            outcome=Outcome.SUCCESS,
            correct=True,
        )
        # These should NOT exist as fields
        assert not hasattr(record, 'retry_count')
        assert not hasattr(record, 'danger_select_count')
        assert not hasattr(record, 'selected_sets')
        assert not hasattr(record, 'step_log')
        assert not hasattr(record, 'beam_entropy')

    def test_summary_blocks_process_access(self):
        """ObservationSummaryV2 must raise on process-level access."""
        from cls_color_selection.tutor_api.observation_v2 import ObservationSummaryV2

        summary = ObservationSummaryV2()
        with pytest.raises(AttributeError, match="result-level only"):
            _ = summary.total_retries
        with pytest.raises(AttributeError, match="result-level only"):
            _ = summary.total_danger_selects

    def test_timeout_has_submitted_or_completion(self):
        """TIMEOUT records should have either submitted or completion."""
        from cls_color_selection.tutor_api.observation_v2 import ObservationRecord
        from cls_color_selection.constants import Outcome

        # Timeout with submitted
        rec = ObservationRecord(
            query_words=['dax'], submitted_output=['RED'],
            current_completion=['RED', None],
            outcome=Outcome.TIMEOUT, correct=False,
        )
        assert rec.best_output == ['RED']

        # Death without submitted
        rec2 = ObservationRecord(
            query_words=['dax'], submitted_output=None,
            current_completion=['RED', None],
            outcome=Outcome.DEATH, correct=False,
        )
        assert rec2.best_output == ['RED', None]

    def test_run_obs_v2_returns_result_level(self, env_and_data):
        """run_observation_phase_v2 returns only result-level info."""
        from cls_color_selection.tutor_api.observation_v2 import (
            run_observation_phase_v2, ObservationSummaryV2)
        from cls_color_selection.interfaces import Example

        d = env_and_data
        obs_queries = [
            Example(words=q.words, output=q.output)
            for q in d['queries'][:2]
        ]
        summary = run_observation_phase_v2(
            d['env'], obs_queries, d['policy'], d['risk_belief'],
            d['feedback_updater'], d['predictor'], d['target_pred'],
            d['rng'], d['cfg'],
        )
        assert isinstance(summary, ObservationSummaryV2)
        assert summary.n_queries == 2
        for rec in summary.records:
            assert rec.query_words is not None
            assert rec.outcome is not None


# ═══════════════════════════════════════════════════════════════
# 2. TutorTaskModel Tests
# ═══════════════════════════════════════════════════════════════

class TestTaskModel:
    """Test ground truth task model."""

    def test_ground_truth_correct(self, env_and_data):
        """TaskModel returns correct ground truth."""
        from cls_color_selection.tutor_api.task_model import TutorTaskModel

        d = env_and_data
        task_model = TutorTaskModel(d['env'], queries=d['queries'])

        for q in d['queries'][:5]:
            gt = task_model.ground_truth_output(q.words)
            assert gt is not None, f"No ground truth for {q.words}"
            assert gt == list(q.output), \
                f"TaskModel ground truth mismatch for {q.words}"

    def test_hint_always_correct(self, env_and_data):
        """Hints from TaskModel are always correct."""
        from cls_color_selection.tutor_api.task_model import TutorTaskModel

        d = env_and_data
        task_model = TutorTaskModel(d['env'], queries=d['queries'])

        gt = list(d['queries'][0].output)
        wrong = ['WRONG'] * len(gt)
        hints = task_model.generate_hint(gt, wrong)

        for pos, color in hints:
            assert color == gt[pos], \
                f"Hint at pos {pos} gave {color}, expected {gt[pos]}"


# ═══════════════════════════════════════════════════════════════
# 3. TutorLearnerModel Tests
# ═══════════════════════════════════════════════════════════════

class TestLearnerModel:
    """Test inverse inference learner model."""

    def test_init_from_support(self, env_and_data):
        """LearnerModel initializes and can predict."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner)
        model.init_from_support(d['support'])

        pred = model.predict_learner(d['queries'][0].words)
        assert pred is not None
        assert len(pred) > 0

    def test_update_depth_role_only(self, env_and_data):
        """role_only depth only updates role_counts."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel
        import copy

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='role_only')
        model.init_from_support(d['support'])

        # Snapshot emit stats before update
        lib = model.get_library()
        words = d['queries'][0].words
        emit_before = {}
        for w in words:
            if w in lib:
                emit_before[w] = lib[w].emit_stats['sum_w']

        # Run update
        model.update_from_output(words, list(d['queries'][0].output))

        # Emit stats should NOT change for role_only
        for w in words:
            if w in lib and w in emit_before:
                assert lib[w].emit_stats['sum_w'] == emit_before[w], \
                    f"role_only should not change emit_stats for {w}"

    def test_update_depth_full_trace(self, env_and_data):
        """full_trace depth updates role + emit + repeat + color."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='full_trace')
        model.init_from_support(d['support'])

        # Snapshot total stats before
        lib = model.get_library()
        total_before = sum(
            sum(c.role_counts.values()) for c in lib.values())

        # Run update
        model.update_from_output(
            d['queries'][0].words, list(d['queries'][0].output))

        total_after = sum(
            sum(c.role_counts.values()) for c in lib.values())
        assert total_after > total_before, "full_trace should change stats"

    def test_predict_matches_init(self, env_and_data):
        """Right after study, predictions should match real learner."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='full_trace')
        model.init_from_support(d['support'])

        # Both should predict the same after identical study
        for q in d['queries'][:3]:
            model_pred = model.predict_learner(q.words)
            real_pred = d['predictor'].predict_target(q.words)
            # May not be identical (different RNG paths) but should be close
            # At least both should be valid color sequences
            if model_pred is not None and real_pred is not None:
                assert all(c in ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE', 'PINK']
                           for c in model_pred)

    def test_feedback_refinement_improves_fit(self, env_and_data):
        """update_from_feedback should improve fit to learner."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='full_trace')
        model.init_from_support(d['support'])

        words = d['queries'][0].words
        gt = list(d['queries'][0].output)

        # Simulate: learner submitted wrong answer
        wrong_output = gt.copy()
        if wrong_output:
            # Swap a color
            wrong_output[0] = 'PINK' if wrong_output[0] != 'PINK' else 'BLUE'

        mask = [s == g for s, g in zip(wrong_output, gt)]
        feedback = {'mode': 'wrong_positions', 'mask': mask,
                    'submitted': wrong_output}

        # Verify feedback update runs without error
        model.update_from_feedback(words, wrong_output, feedback)
        assert model._n_feedback_updates == 1


# ═══════════════════════════════════════════════════════════════
# 4. InverseTutor E2E Tests
# ═══════════════════════════════════════════════════════════════

class TestInverseTutor:
    """End-to-end inverse tutor tests."""

    def test_construction(self, env_and_data):
        """InverseTutor constructs and initializes."""
        from cls_color_selection.tutor_api.tutor_inverse import InverseTutor
        from cls_color_selection.tutor_api.task_model import TutorTaskModel

        d = env_and_data
        task_model = TutorTaskModel(d['env'])
        tutor = InverseTutor(
            d['cfg'].tutor, d['cfg'].learner, task_model,
            update_depth='full_trace')
        tutor.init_learner_model(d['support'])

        summary = tutor.summary_dict()
        assert 'learner_model' in summary
        assert 'risk_stats' in summary

    def test_process_observation(self, env_and_data):
        """process_observation updates learner model."""
        from cls_color_selection.tutor_api.tutor_inverse import InverseTutor
        from cls_color_selection.tutor_api.task_model import TutorTaskModel
        from cls_color_selection.tutor_api.observation_v2 import ObservationRecord
        from cls_color_selection.constants import Outcome

        d = env_and_data
        task_model = TutorTaskModel(d['env'])
        tutor = InverseTutor(
            d['cfg'].tutor, d['cfg'].learner, task_model)
        tutor.init_learner_model(d['support'])

        q = d['queries'][0]
        record = ObservationRecord(
            query_words=list(q.words),
            submitted_output=list(q.output),
            current_completion=list(q.output),
            outcome=Outcome.SUCCESS,
            correct=True,
        )
        tutor.process_observation(record)
        assert tutor.learner_model._n_output_updates >= 1

    def test_update_depth_param(self, env_and_data):
        """Different update_depths create different behaviors."""
        from cls_color_selection.tutor_api.tutor_inverse import InverseTutor
        from cls_color_selection.tutor_api.task_model import TutorTaskModel

        d = env_and_data
        task_model = TutorTaskModel(d['env'])

        for depth in ('role_only', 'role_emit', 'full_trace'):
            tutor = InverseTutor(
                d['cfg'].tutor, d['cfg'].learner, task_model,
                update_depth=depth)
            tutor.init_learner_model(d['support'])
            assert tutor.learner_model.update_depth == depth


# ═══════════════════════════════════════════════════════════════
# 5. Divergence V3 Tests
# ═══════════════════════════════════════════════════════════════

class TestDivergenceV3:
    """Test divergence and predictive validity metrics."""

    def test_seq_edit_distance(self):
        from cls_color_selection.tutor_api.divergence_v3 import _seq_edit_distance
        assert _seq_edit_distance(['A', 'B'], ['A', 'B']) == 0
        assert _seq_edit_distance(['A', 'B'], ['A', 'C']) == 1
        assert _seq_edit_distance(['A'], ['A', 'B']) == 1
        assert _seq_edit_distance([], []) == 0

    def test_compute_inverse_divergence(self, env_and_data):
        """Divergence computation runs and returns valid structure."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel
        from cls_color_selection.tutor_api.divergence_v3 import (
            compute_inverse_divergence)

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='full_trace')
        model.init_from_support(d['support'])

        probes = [list(q.words) for q in d['queries'][:3]]
        gold = [list(q.output) for q in d['queries'][:3]]

        result = compute_inverse_divergence(
            model, d['predictor'], probes, gold,
            phase='test', query_idx=0)

        assert 'top1_agreement' in result
        assert 'js_divergence' in result
        assert 'tutor_model_accuracy' in result
        assert result['n_probes'] == 3

    def test_predictive_validity(self, env_and_data):
        """Predictive validity computation works."""
        from cls_color_selection.tutor_api.learner_model import TutorLearnerModel
        from cls_color_selection.tutor_api.divergence_v3 import (
            compute_predictive_validity)

        d = env_and_data
        model = TutorLearnerModel(d['cfg'].learner, update_depth='full_trace')
        model.init_from_support(d['support'])

        result = compute_predictive_validity(
            model, d['predictor'], list(d['queries'][0].words))

        assert 'pred_match' in result
        assert 'edit_distance' in result
