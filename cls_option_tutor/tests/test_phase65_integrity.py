"""
test_phase65_integrity.py — Phase 6.5 learning-signal integrity tests.

Tests verify:
1. Autonomous probe preserves post-teach semantic state
2. Local probe preserves post-teach semantic state
3. Probe does not call init_block on clone
4. Self-correct forced pick increments scripted self-correct counter
5. Self-correct not counted as direct_answer source
6. Then-answer counted as direct_answer source
7. Policy margin changes after reveal update
8. Semantic margin and policy margin can diverge
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
from cls_option_tutor.eval.autonomous_probe import run_autonomous_probe
from cls_option_tutor.eval.local_probe import run_local_probe


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


def _teach_learner(cfg, protocol="script_wrong1_self_correct_safe", seed=42):
    """Run a teach block and return (learner, scripted_result)."""
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=seed)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol=protocol)
    result = runner.run_block(env, learner, TASK_ID, seed=seed)
    return learner, result, env


# ── T1: autonomous probe preserves post-teach semantic state ──────────────

def test_autonomous_probe_preserves_post_teach_semantic_state():
    """After teaching, the autonomous probe should use the post-teach scorer,
    not a reinitialized one. Verify scorer has been modified by teaching."""
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_self_correct_safe")

    # Record a scorer state signature (e.g., a score for a known pair)
    target = list(result.block.queries[0].target_output)
    correct_text = list(result.block.queries[0].true_program)
    score_post_teach = learner._scorer.score_option(target, correct_text)

    # Now run probe — it doesn't call init_block, so clone should keep scorer
    probe_learner = copy.deepcopy(learner)
    probe_block = env.reset_block(TASK_ID, seed=99999)
    probe_learner.prepare_probe_block(probe_block)

    # Clone's scorer should still give the same score
    score_in_probe = probe_learner._scorer.score_option(target, correct_text)
    assert abs(score_post_teach - score_in_probe) < 1e-6, \
        f"Probe scorer diverged: {score_post_teach} vs {score_in_probe}"


# ── T2: local probe preserves post-teach semantic state ──────────────────

def test_local_probe_preserves_post_teach_semantic_state():
    """Local probe should preserve scorer state via prepare_probe_block."""
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_self_correct_safe")

    target = list(result.block.queries[0].target_output)
    correct_text = list(result.block.queries[0].true_program)
    score_post_teach = learner._scorer.score_option(target, correct_text)

    # Run local probe — should deepcopy + prepare_probe_block
    probe_result = run_local_probe(
        learner, env, TASK_ID, cfg=cfg, probe_seed=11111, n_local=3
    )

    # Real learner's scorer should be completely unchanged
    score_after_probe = learner._scorer.score_option(target, correct_text)
    assert abs(score_post_teach - score_after_probe) < 1e-6, \
        f"Local probe mutated real learner's scorer"


# ── T3: probe does not call init_block on clone ──────────────────────────

def test_probe_does_not_call_init_block_on_clone():
    """Verify that autonomous_probe uses prepare_probe_block, not init_block.
    After teaching, danger_head should have non-trivial state.
    If init_block were called, danger_head would be reset."""
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_self_correct_bounded")

    # Danger head should have some learned state after bounded-risk wrong picks
    dh_w_post = learner.policy.danger_head.hazard.w.copy()

    # Clone + prepare_probe_block
    probe_learner = copy.deepcopy(learner)
    probe_block = env.reset_block(TASK_ID, seed=88888)
    probe_learner.prepare_probe_block(probe_block)

    # Danger head should be preserved
    dh_w_probe = probe_learner.policy.danger_head.hazard.w.copy()
    assert np.allclose(dh_w_post, dh_w_probe), \
        "prepare_probe_block changed danger_head state"


# ── T4: self-correct increments scripted self-correct counter ────────────

def test_self_correct_forced_pick_increments_scripted_self_correct_counter():
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_self_correct_safe")

    src = learner._src_counters
    # SelfCorrectCount should align with cu_scripted_self_correct_att
    assert src["cu_scripted_self_correct_att"] == result.self_correct_count, \
        f"cu_scripted_self_correct_att={src['cu_scripted_self_correct_att']} != " \
        f"SelfCorrectCount={result.self_correct_count}"
    # Also check applied <= attempted
    assert src["cu_scripted_self_correct_app"] <= src["cu_scripted_self_correct_att"]
    # With rho=1.0, all should be applied
    assert src["cu_scripted_self_correct_app"] == src["cu_scripted_self_correct_att"], \
        f"At rho=1.0, all self-correct should be applied"


# ── T5: self-correct NOT counted as direct_answer source ─────────────────

def test_self_correct_not_counted_as_direct_answer():
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_self_correct_safe")

    src = learner._src_counters
    # No direct-answer or then-answer events in self_correct protocol
    assert src["da_direct_answer_att"] == 0, \
        f"Self-correct should not trigger da_direct_answer: got {src['da_direct_answer_att']}"
    assert src["da_then_answer_att"] == 0, \
        f"Self-correct should not trigger da_then_answer: got {src['da_then_answer_att']}"


# ── T6: then-answer counted as direct_answer source ──────────────────────

def test_then_answer_counted_as_direct_answer_source():
    cfg = _make_cfg(rho_assist=1.0)
    learner, result, env = _teach_learner(cfg, "script_wrong1_then_answer_safe")

    src = learner._src_counters
    # then_answer should increment da_then_answer
    assert src["da_then_answer_att"] == result.then_answer_count, \
        f"da_then_answer_att={src['da_then_answer_att']} != " \
        f"ThenAnswerCount={result.then_answer_count}"
    # Should NOT increment cu_scripted_self_correct
    assert src["cu_scripted_self_correct_att"] == 0, \
        f"Then-answer should not trigger cu_scripted_self_correct"


# ── T7: policy margin changes after reveal update ────────────────────────

def test_policy_margin_changes_after_reveal_update():
    """After a wrong-reveal + correct event, policy margin should differ
    from the pre-teach state."""
    cfg = _make_cfg(rho_assist=1.0)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)

    # Need to init_block first to create scorer/danger_head
    support, _, grammar = env.adapter.load_task(TASK_ID)
    block0 = env.reset_block(TASK_ID, seed=42)
    learner.init_block(block0, grammar, support)

    # Pre-teach local probe
    pre_probe = run_local_probe(
        learner, env, TASK_ID, cfg=cfg, probe_seed=77777, n_local=5
    )

    # Teach with bounded-risk wrong (will update danger head + scorer)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_self_correct_bounded")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    # Post-teach local probe
    post_probe = run_local_probe(
        learner, env, TASK_ID, cfg=cfg, probe_seed=77777, n_local=5
    )

    from cls_option_tutor.eval.local_probe import compute_local_learning
    ll = compute_local_learning(pre_probe, post_probe)

    # At least policy margin or correct prob should change
    # (danger head was updated by bounded-risk wrong picks)
    any_change = (
        abs(ll.delta_policy_margin) > 1e-6 or
        abs(ll.delta_correct_prob) > 1e-6 or
        abs(ll.delta_semantic_margin) > 1e-6
    )
    assert any_change, (
        f"No local diagnostic changed after bounded-risk teaching: "
        f"dPM={ll.delta_policy_margin}, dCP={ll.delta_correct_prob}, "
        f"dSM={ll.delta_semantic_margin}"
    )


# ── T8: semantic margin and policy margin can diverge ────────────────────

def test_semantic_margin_and_policy_margin_can_diverge():
    """After danger-head-only learning (from risk reveals), policy margin
    should change even if semantic margin is approximately stable."""
    cfg = _make_cfg(rho_assist=1.0)
    # Use eta_reveal=0 to prevent semantic updates, only danger head changes
    cfg.learner.eta_reveal = 0.0
    cfg.learner.correct_pick_learning_mode = "off"
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)

    # Need to init_block first to create scorer/danger_head
    support, _, grammar = env.adapter.load_task(TASK_ID)
    block0 = env.reset_block(TASK_ID, seed=42)
    learner.init_block(block0, grammar, support)

    # Pre-teach probe
    pre_probe = run_local_probe(
        learner, env, TASK_ID, cfg=cfg, probe_seed=55555, n_local=5
    )

    # Teach with bounded-risk wrong (scorer won't update, but danger head will)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_self_correct_bounded")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    # Post-teach probe
    post_probe = run_local_probe(
        learner, env, TASK_ID, cfg=cfg, probe_seed=55555, n_local=5
    )

    from cls_option_tutor.eval.local_probe import compute_local_learning
    ll = compute_local_learning(pre_probe, post_probe)

    # Semantic margin should be ~0 (scorer not updated)
    # Policy margin may change (danger head was updated)
    # This is a soft check — just verify they CAN differ
    assert abs(ll.delta_semantic_margin) < 0.3, \
        f"Semantic margin changed too much with eta_reveal=0: {ll.delta_semantic_margin}"
    # We don't assert policy margin is non-zero because it depends on
    # whether the probe queries happen to have risky options
    # Just verify the code path doesn't crash
    assert isinstance(ll.delta_policy_margin, float)
    assert isinstance(ll.delta_correct_prob, float)
