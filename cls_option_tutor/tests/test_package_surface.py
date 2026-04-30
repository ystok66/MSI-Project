from cls_option_tutor import ACTIVE_MAINLINE_ALIAS, ACTIVE_MAINLINE_CANONICAL
from cls_option_tutor.experiments import resolve_condition_alias
from cls_option_tutor.learner import (
    DeterministicSemanticScorer,
    LearnerAgent,
    SemanticScorer,
)
from cls_option_tutor.tutor import SparseTutorAgent


def test_root_exports_frozen_mainline_alias():
    assert ACTIVE_MAINLINE_ALIAS == "SIS_cf_mix_loop_v1"
    assert resolve_condition_alias(ACTIVE_MAINLINE_ALIAS) == ACTIVE_MAINLINE_CANONICAL


def test_package_surface_exports_active_runtime_classes():
    assert LearnerAgent.__name__ == "LearnerAgent"
    assert SparseTutorAgent.__name__ == "SparseTutorAgent"


def test_semantic_scorer_package_alias_points_to_active_deterministic_scorer():
    assert SemanticScorer is DeterministicSemanticScorer
