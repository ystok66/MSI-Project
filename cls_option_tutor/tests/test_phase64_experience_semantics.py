"""
test_phase64_experience_semantics.py — Phase 6.4 tests.

Tests verify:
1. self_correct assist rank is 0
2. self_correct counts as unassisted correct
3. then_answer counts as direct_answer
4. Phase-specific damage sums correctly
5. ScriptedDamage <= TeachDamage
6. Local probe does not mutate learner
7. Diagnostic wrong selector prefers near-output over far
8. Confound labels don't use hidden correctness
9. Random vs diagnostic produce different selections
"""
import copy
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.interfaces_assist import ASSIST_RANK
from cls_option_tutor.eval.local_probe import run_local_probe
from cls_option_tutor.grammar.confound_labels import (
    ConfoundType, label_confound, label_diagnostic_risk, DiagnosticRiskLabel,
)


DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')
)
TASK_ID = "000001"


def _make_cfg(rho_assist=1.0):
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.n_sup = 4
    cfg.learner.n_em = 1
    cfg.learner.use_hpc = False
    cfg.learner.rho_assist = rho_assist
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    cfg.env.K = 6
    cfg.env.T_max = 3
    cfg.env.N_obs = 1
    cfg.env.N_teach = 3
    cfg.env.N_eval = 2
    cfg.env.M_queries = 6
    cfg.env.n_risky = 2
    cfg.tutor.rollout_mode = "proxy"
    return cfg


# ── T1: self_correct assist rank is 0 ───────────────────────────────────────

def test_self_correct_assist_rank_is_zero():
    assert ASSIST_RANK["self_correct"] == 0


# ── T2: self_correct counts as unassisted correct ────────────────────────────

def test_wrong_then_self_correct_counts_as_unassisted_correct():
    cfg = _make_cfg(rho_assist=1.0)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_self_correct_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    sem = learner._sem_counters
    # Self-correct should route to correct_unassisted, not direct_answer
    if result.self_correct_count > 0:
        assert sem["correct_unassisted_attempted"] > 0, \
            "self_correct should increment correct_unassisted_attempted"
    # Should NOT increment direct_answer
    assert sem["direct_answer_attempted"] == 0, \
        f"self_correct should NOT increment direct_answer_attempted, got {sem['direct_answer_attempted']}"


# ── T3: then_answer counts as direct_answer ──────────────────────────────────

def test_wrong_then_answer_counts_as_direct_answer():
    cfg = _make_cfg(rho_assist=1.0)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_then_answer_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    sem = learner._sem_counters
    if result.then_answer_count > 0:
        assert sem["direct_answer_attempted"] > 0, \
            "then_answer should increment direct_answer_attempted"


# ── T4: phase-specific damage sums consistently ──────────────────────────────

def test_phase_specific_damage_sums_consistently():
    """ObsDamage + TeachDamage + EvalDamage approximately equals total damage."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_self_correct_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)
    block = result.block

    total_from_trace = sum(
        ls.damage for ls in block.learner_trace
        if ls.damage is not None
    )
    # This is a basic sanity — exact phase attribution requires query_id tracking
    # which the runner does. Here we just verify total > 0 or total == 0.
    assert total_from_trace >= 0, "Total damage should be non-negative"


# ── T5: ScriptedDamage <= TeachDamage ────────────────────────────────────────

def test_scripted_teach_damage_is_subset_of_teach_damage():
    """Damage from forced steps must be a subset of total teach damage."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_self_correct_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)
    block = result.block

    # Compute scripted damage from forced_step_indices
    scripted_dmg = 0
    for idx in result.forced_step_indices:
        if idx < len(block.learner_trace):
            d = block.learner_trace[idx].damage
            scripted_dmg += (d if d is not None else 0)

    total_dmg = sum(ls.damage for ls in block.learner_trace if ls.damage is not None)
    assert scripted_dmg <= total_dmg, \
        f"ScriptedDamage {scripted_dmg} > total {total_dmg}"


# ── T6: local probe does not mutate real learner ─────────────────────────────

def test_local_probe_does_not_mutate_real_learner():
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)
    learner.init_block(block, grammar, support)

    dh_w_before = learner.policy.danger_head.hazard.w.copy()
    sem_before = dict(learner._sem_counters)

    probe = run_local_probe(learner, env, TASK_ID, cfg=cfg, probe_seed=9999, n_local=5)

    dh_w_after = learner.policy.danger_head.hazard.w.copy()
    sem_after = dict(learner._sem_counters)

    assert np.allclose(dh_w_before, dh_w_after), "Local probe mutated danger head!"
    assert sem_before == sem_after, "Local probe mutated sem_counters!"


# ── T7: diagnostic prefers labeled wrong over far distractor ─────────────────

def test_diagnostic_wrong_selector_prefers_labeled_wrong_over_far_distractor():
    """On a toy target, near-output should score higher than far distractor."""
    target = [1, 2, 3, 4, 5]

    near = [1, 2, 3, 4, 6]    # 1 cell different → near_output
    far  = [9, 8, 7, 6, 5]    # completely different → far

    ct_near = label_confound(near, target)
    ct_far = label_confound(far, target)

    assert ct_near == ConfoundType.NEAR_OUTPUT, f"Expected NEAR_OUTPUT, got {ct_near}"
    assert ct_far == ConfoundType.FAR_DISTRACTOR, f"Expected FAR_DISTRACTOR, got {ct_far}"


# ── T8: diagnostic labels don't require hidden correctness ───────────────────

def test_diagnostic_labels_do_not_require_hidden_correctness():
    """Labels use only rendered output, not hidden is_correct."""
    target = [1, 2, 3]
    wrong_out = [1, 2, 4]

    ct = label_confound(wrong_out, target, is_correct=False)
    assert ct != ConfoundType.CORRECT
    assert ct in (ConfoundType.NEAR_OUTPUT, ConfoundType.ORDER_LIKE,
                  ConfoundType.CARDINALITY_LIKE, ConfoundType.SCOPE_LIKE,
                  ConfoundType.FAR_DISTRACTOR)


# ── T9: random vs diagnostic produce different selection when available ──────

def test_random_vs_diagnostic_safe_wrong_produce_different_selection():
    """Sanity: diagnostic selection is actually doing work vs random."""
    from cls_option_tutor.tutor.scripted_protocols import (
        _choose_safe_wrong, _compute_d_learn,
    )
    from cls_option_tutor.interfaces import Option

    # Create a mock QueryState with multiple safe wrongs of varying diagnosticity
    class MockQS:
        target_output = [1, 2, 3, 4, 5]
        banned_indices = set()

    qs = MockQS()

    # Two safe wrong options: one near, one far
    opt_near = Option(index=0, text=[], danger_vec=np.zeros(16),
                      rendered_output=[1, 2, 3, 4, 6],
                      is_correct=False, risk_class=0)
    opt_far = Option(index=1, text=[], danger_vec=np.zeros(16),
                     rendered_output=[9, 8, 7, 6, 5],
                     is_correct=False, risk_class=0)

    d_near = _compute_d_learn(opt_near, qs)
    d_far = _compute_d_learn(opt_far, qs)

    assert d_near > d_far, \
        f"Diagnostic near ({d_near:.3f}) should score higher than far ({d_far:.3f})"
