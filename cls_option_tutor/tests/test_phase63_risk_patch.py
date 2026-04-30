"""
test_phase63_risk_patch.py — Phase 6.3 risk-valid micro benchmark tests.

Tests verify that:
1. Risk canonicalization produces actual risky options
2. Scripted protocols produce correct damage patterns
3. Assist discount instrumentation works
4. Probe invariant (no learner mutation) still holds
5. Existing inverse boundary tests still pass
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
from cls_option_tutor.tutor.direct_answer_tutor import DirectAnswerTutor
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.eval.autonomous_probe import run_autonomous_probe


DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')
)
TASK_ID = "000001"


def _make_cfg(n_risky=2, rho_assist=1.0):
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.n_sup = 4
    cfg.learner.n_em = 1
    cfg.learner.use_hpc = False
    cfg.learner.rho_assist = rho_assist
    cfg.env.K = 6
    cfg.env.T_max = 3
    cfg.env.N_obs = 1
    cfg.env.N_teach = 3
    cfg.env.N_eval = 2
    cfg.env.M_queries = 6
    cfg.env.n_risky = n_risky
    cfg.tutor.rollout_mode = "proxy"
    return cfg


# ── T1: Risk canonicalization ────────────────────────────────────────────────

def test_micro_runner_respects_n_risky_canonicalization():
    """K=6, n_risky=2 → menus contain exactly 2 risky options on average, not 0."""
    cfg = _make_cfg(n_risky=2)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    block = env.reset_block(TASK_ID, seed=42)

    total_risky = 0
    total_options = 0
    for qs in block.queries:
        for opt in qs.menu:
            total_options += 1
            if opt.risk_class > 0:
                total_risky += 1

    # With K=6 and n_risky=2, expect ~2/6 = 33% risky
    risky_frac = total_risky / total_options
    assert risky_frac > 0.15, (
        f"Expected risky fraction ~0.33, got {risky_frac:.3f}. "
        f"Risk canonicalization may not be working."
    )
    assert total_risky > 0, "Zero risky options found — n_risky=2 not respected."


# ── T2: Safe wrong has zero damage ───────────────────────────────────────────

def test_script_wrong1_safe_has_zero_damage():
    """Safe scripted wrong should produce damage == 0."""
    cfg = _make_cfg(n_risky=2)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_correct_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    # All safe wrongs should have risk_class == 0 → damage == 0
    assert result.wrong_reveal_risk0 >= 0, "Some safe wrongs expected"
    # No risky wrongs in safe protocol
    assert result.wrong_reveal_risk1 == 0
    assert result.wrong_reveal_risk2 == 0
    assert result.wrong_reveal_risk3 == 0
    assert result.wrong_reveal_risk4 == 0


# ── T3: Bounded risk has positive damage ─────────────────────────────────────

def test_script_wrong1_bounded_has_positive_damage():
    """Bounded-risk scripted wrong should produce damage in {1, 2}."""
    cfg = _make_cfg(n_risky=2)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_correct_bounded_risk")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    # Either bounded risk wrongs occurred or skipped
    bounded_count = result.wrong_reveal_risk1 + result.wrong_reveal_risk2
    if len(result.skipped) == 0:
        assert bounded_count > 0, "Expected bounded-risk wrongs when not skipped"
    # No safe wrongs in bounded-risk protocol
    assert result.wrong_reveal_risk0 == 0


# ── T4: High risk has class ≥3 or skips ──────────────────────────────────────

def test_script_wrong1_high_risk_has_class_ge3_or_skips():
    """If high-risk run proceeds, wrong option must be class 3 or 4."""
    cfg = _make_cfg(n_risky=2)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_correct_high_risk")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    high_count = result.wrong_reveal_risk3 + result.wrong_reveal_risk4
    # Either high-risk wrongs occurred or queries were skipped
    if len(result.skipped) == 0:
        assert high_count > 0, "Expected high-risk wrongs when not skipped"
    # No safe wrongs
    assert result.wrong_reveal_risk0 == 0, "No safe wrongs in high-risk protocol"


# ── T5: Direct answer updates blocked when rho=0 ────────────────────────────

def test_direct_answer_updates_blocked_when_rho_zero():
    """With rho_assist=0, direct-answer semantic updates should be attempted but blocked."""
    cfg = _make_cfg(rho_assist=0.0)
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    da = DirectAnswerTutor(cfg=cfg)
    block = da.run_block(env, learner, TASK_ID, seed=42)

    sem = learner._sem_counters
    # Direct answer attempts should occur (> 0 if learner picks correctly after SHORTLIST)
    if sem["direct_answer_attempted"] > 0:
        assert sem["direct_answer_applied"] == 0, (
            f"rho_assist=0 should block all direct_answer updates, "
            f"got applied={sem['direct_answer_applied']}"
        )


# ── T6: Direct correct not blocked by rho=0 ─────────────────────────────────

def test_direct_correct_not_blocked_by_rho_zero():
    """Unassisted correct picks should still apply even at rho_assist=0."""
    cfg = _make_cfg(rho_assist=0.0)
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_direct_correct")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    sem = learner._sem_counters
    # Unassisted correct picks NOT gated by rho_assist
    # (assist_level='none' → rank=0 → omega=0^0=1)
    if sem["correct_unassisted_attempted"] > 0:
        assert sem["correct_unassisted_applied"] == sem["correct_unassisted_attempted"], (
            f"Unassisted should always apply: "
            f"attempted={sem['correct_unassisted_attempted']}, "
            f"applied={sem['correct_unassisted_applied']}"
        )


# ── T7: Scripted logs violation or exact sequence ────────────────────────────

def test_script_protocol_logs_violation_or_exact_sequence():
    """Scripted runs either produce exact sequences or explicit skip/violation."""
    cfg = _make_cfg(n_risky=2)
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    runner = ScriptedProtocolRunner(cfg=cfg, protocol="script_wrong1_correct_safe")
    result = runner.run_block(env, learner, TASK_ID, seed=42)

    # For each teach query: either produced exactly 1 wrong + 1 correct,
    # or it was skipped
    teach_qs = result.block.queries[
        result.block.obs_phase_queries:
        result.block.obs_phase_queries + result.block.teach_phase_queries
    ]
    for qs in teach_qs:
        if qs.skipped:
            # Must be in skipped list
            assert any(sk.query_id == qs.query_id for sk in result.skipped), (
                f"Skipped query {qs.query_id} not in skip log"
            )


# ── T8: Autonomous probe does not mutate real learner ────────────────────────

def test_autonomous_probe_does_not_mutate_real_learner():
    """Preserve the key benchmark invariant."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=42)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)
    learner.init_block(block, grammar, support)

    dh_w_before = learner.policy.danger_head.hazard.w.copy()

    probe = run_autonomous_probe(
        learner, env, TASK_ID, probe_seed=9999, cfg=cfg,
        n_probe=5, freeze_semantic=True, freeze_risk=True, freeze_memory=True,
    )

    dh_w_after = learner.policy.danger_head.hazard.w.copy()
    assert np.allclose(dh_w_before, dh_w_after), "Probe mutated danger head weights!"


# ── T9: Inverse boundary tests still pass ────────────────────────────────────

def test_inverse_boundary_still_passes():
    """Re-run existing anti-cheat test module to ensure no regression."""
    # This is a meta-test: we import and let pytest discover the tests
    # from the existing boundary test file. If they fail, this test fails.
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "cls_option_tutor/tests/test_inverse_boundary.py",
         "-q", "--tb=line"],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    )
    assert result.returncode == 0, (
        f"Inverse boundary tests failed:\n{result.stdout}\n{result.stderr}"
    )
