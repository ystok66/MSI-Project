"""
test_option_level.py — Invariant tests for the Option-Level Strong Tutor.

Five mandatory invariants:
  I1: j* in S                         (correct always reachable)
  I2: |S| = min(tau_t, K_safe)        (completable in remaining rounds)
  I3: final_choice in S               (env-enforced via get_active_menu)
  I4: no lethal option in S           (safety by construction)
  I5: final_choice not in banned      (env-enforced via _do_pick)

Scenarios A-D are smoke tests for block-level behaviors.

Run:
    python cls_option_tutor/tests/test_option_level.py
    python -m pytest cls_option_tutor/tests/test_option_level.py -v
"""
from __future__ import annotations
import sys
import os
import traceback
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.interventions import get_active_menu
from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.grammar.task_adapter import TaskAdapter

DATA_DIR = os.path.join(ROOT, "BASIC", "cls_learner", "data")


# ── Helpers ──────────────────────────────────────────────────────

def _get_task_id():
    """Get a valid grammar/task ID from BASIC/cls_learner/data/*.txt."""
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")
    grammar_ids = sorted(set(
        os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith('.txt')
    ))
    if not grammar_ids:
        raise RuntimeError(f"No .txt grammar files in {DATA_DIR}")
    return grammar_ids[0]  # e.g. "000001"


def _make_cfg(
    K=10, T_max=5, H_0=5,
    N_obs=2, N_teach=2, N_eval=3,
    n_sup=4, n_risky=4,
):
    cfg = FullConfig()
    cfg.env.K = K
    cfg.env.T_max = T_max
    cfg.env.H_0 = H_0
    cfg.env.N_obs = N_obs
    cfg.env.N_teach = N_teach
    cfg.env.N_eval = N_eval
    cfg.env.M_queries = N_obs + N_teach + N_eval
    cfg.env.n_risky = min(n_risky, K - 1)
    cfg.learner.use_cls = True
    cfg.learner.n_sup = n_sup
    cfg.learner.n_em = 2
    cfg.learner.use_hpc = True
    cfg.rsa.use_rsa = False
    return cfg


def _run_block(cfg, task_id, seed=42, mode="confusion"):
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor = OptionLevelTutorAgent(cfg=cfg, shortlist_mode=mode)
    learner = LearnerAgent(cfg=cfg)
    block = tutor.run_block(env, learner, task_id=task_id, seed=seed)
    return block, tutor, learner, env


def _step_block(cfg, task_id, seed, callback_fn):
    """Step through a block, calling callback_fn(qs, tutor_step) after each tutor act."""
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor = OptionLevelTutorAgent(cfg=cfg)
    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(task_id, seed=seed)
    support, _, grammar = env.adapter.load_task(task_id)
    tutor.init_block(block, grammar, support)
    learner.init_block(block, grammar, support)

    max_steps = len(block.queries) * 25
    steps = 0
    while not block.done and steps < max_steps:
        steps += 1
        qs = block.current_query
        if qs is None or qs.done:
            break
        ts = tutor.act(block, env, learner_agent=learner)
        callback_fn(qs, ts)
        if qs.done:
            continue
        learner.act(block, env)
    return block


# ════════════════════════════════════════════════════════════════
# Invariant I1: j* ∈ S
# ════════════════════════════════════════════════════════════════

def test_I1_correct_in_shortlist():
    """j* must always be in the shortlist."""
    task_id = _get_task_id()
    violations = []

    def check(qs, ts):
        if ts.action != "SHORTLIST":
            return
        correct_idx = next((o.index for o in qs.menu if o.is_correct), None)
        if correct_idx is None:
            return
        if correct_idx not in ts.shortlist_indices:
            violations.append(
                f"query {qs.query_id}: j*={correct_idx} not in S={ts.shortlist_indices}"
            )

    cfg = _make_cfg()
    _step_block(cfg, task_id, seed=7, callback_fn=check)
    assert not violations, f"I1 violations:\n" + "\n".join(violations)
    print(f"  PASS  I1: j* always in shortlist")


# ════════════════════════════════════════════════════════════════
# Invariant I2: |S| = min(tau_t, K_safe)
# ════════════════════════════════════════════════════════════════

def test_I2_shortlist_size():
    """Shortlist size must equal min(tau_t, K_safe) at time of issuance."""
    task_id = _get_task_id()
    violations = []

    env = OptionEnv(cfg=_make_cfg(), data_dir=DATA_DIR)
    tutor = OptionLevelTutorAgent(cfg=_make_cfg())
    learner = LearnerAgent(cfg=_make_cfg())
    cfg = _make_cfg()
    block = env.reset_block(task_id, seed=13)
    support, _, grammar = env.adapter.load_task(task_id)
    tutor.init_block(block, grammar, support)
    learner.init_block(block, grammar, support)

    max_steps = len(block.queries) * 25
    steps = 0
    while not block.done and steps < max_steps:
        steps += 1
        qs = block.current_query
        if qs is None or qs.done:
            break

        tau_t = max(0, qs.max_rounds - qs.rounds_used)
        active = get_active_menu(qs)
        lethal = tutor._get_lethal_indices(qs, active, learner)
        safe = [o for o in active if o.index not in lethal]
        expected = min(tau_t, len(safe))

        ts = tutor.act(block, env, learner_agent=learner)

        if ts.action == "SHORTLIST":
            actual = len(ts.shortlist_indices)
            if actual != expected:
                violations.append(
                    f"query {qs.query_id}: |S|={actual} != expected={expected} "
                    f"(tau_t={tau_t}, K_safe={len(safe)})"
                )
        if qs.done:
            continue
        learner.act(block, env)

    assert not violations, f"I2 violations:\n" + "\n".join(violations)
    print(f"  PASS  I2: |S| = min(tau_t, K_safe)")


# ════════════════════════════════════════════════════════════════
# Invariant I3: final_choice ∈ S (env raises ValueError if violated)
# ════════════════════════════════════════════════════════════════

def test_I3_choice_in_shortlist():
    """All picks must be within the active shortlist."""
    task_id = _get_task_id()
    try:
        block, _, _, _ = _run_block(_make_cfg(), task_id, seed=21)
    except ValueError as e:
        if "not in active shortlist" in str(e):
            raise AssertionError(f"I3 violated: {e}")
        raise

    # Double-check via trace
    violations = []
    for lstep in block.learner_trace:
        if lstep.action != "pick" or lstep.pick_index is None:
            continue
        qs = block.queries[lstep.query_id]
        if qs.shortlisted_indices is not None:
            if lstep.pick_index not in qs.shortlisted_indices:
                violations.append(
                    f"query {lstep.query_id}: pick {lstep.pick_index} "
                    f"not in shortlist {qs.shortlisted_indices}"
                )
    assert not violations, f"I3 violations:\n" + "\n".join(violations)
    print(f"  PASS  I3: all picks within shortlist")


# ════════════════════════════════════════════════════════════════
# Invariant I4: no lethal option in S
# ════════════════════════════════════════════════════════════════

def test_I4_no_lethal_in_shortlist():
    """Shortlist must not contain options with risk >= HP_t."""
    task_id = _get_task_id()
    cfg = _make_cfg(n_risky=6)
    violations = []

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor = OptionLevelTutorAgent(cfg=cfg)
    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(task_id, seed=31)
    support, _, grammar = env.adapter.load_task(task_id)
    tutor.init_block(block, grammar, support)
    learner.init_block(block, grammar, support)

    max_steps = len(block.queries) * 25
    steps = 0
    while not block.done and steps < max_steps:
        steps += 1
        qs = block.current_query
        if qs is None or qs.done:
            break

        hp = qs.hp
        active = get_active_menu(qs)
        lethal = tutor._get_lethal_indices(qs, active, learner)

        ts = tutor.act(block, env, learner_agent=learner)

        if ts.action == "SHORTLIST":
            overlap = set(ts.shortlist_indices) & lethal
            if overlap:
                violations.append(
                    f"query {qs.query_id}: lethal {overlap} in shortlist "
                    f"{ts.shortlist_indices} (HP={hp})"
                )
        if qs.done:
            continue
        learner.act(block, env)

    assert not violations, f"I4 violations:\n" + "\n".join(violations)
    print(f"  PASS  I4: no lethal options in shortlist")


# ════════════════════════════════════════════════════════════════
# Invariant I5: final_choice ∉ banned
# ════════════════════════════════════════════════════════════════

def test_I5_no_pick_from_banned():
    """Learner must never pick a banned option."""
    task_id = _get_task_id()
    try:
        block, _, _, _ = _run_block(_make_cfg(), task_id, seed=42)
    except ValueError as e:
        if "is banned" in str(e):
            raise AssertionError(f"I5 violated: {e}")
        raise

    violations = []
    for lstep in block.learner_trace:
        if lstep.action != "pick" or lstep.pick_index is None:
            continue
        qs = block.queries[lstep.query_id]
        if lstep.pick_index in qs.banned_indices:
            violations.append(
                f"query {lstep.query_id}: picked banned option {lstep.pick_index}"
            )
    assert not violations, f"I5 violations:\n" + "\n".join(violations)
    print(f"  PASS  I5: no picks from banned set")


# ════════════════════════════════════════════════════════════════
# Scenario Tests
# ════════════════════════════════════════════════════════════════

def test_scenario_A_shortlist_fires():
    """Scenario A: K=10, T_max=3 → K>tau_t → shortlist must fire."""
    task_id = _get_task_id()
    cfg = _make_cfg(K=10, T_max=3, N_teach=2)
    block, _, _, _ = _run_block(cfg, task_id, seed=17)
    n_shortlist = sum(1 for t in block.tutor_trace if t.action == "SHORTLIST")
    assert n_shortlist >= 1, (
        f"Scenario A: expected SHORTLIST but got 0. "
        f"Tutor actions: {[t.action for t in block.tutor_trace]}"
    )
    print(f"  PASS  Scenario A: {n_shortlist} SHORTLIST actions fired (K>tau_t)")


def test_scenario_A_shortlist_size_bound():
    """Scenario A: all shortlist sizes must be <= T_max."""
    task_id = _get_task_id()
    cfg = _make_cfg(K=10, T_max=3, N_teach=2)
    block, _, _, _ = _run_block(cfg, task_id, seed=17)
    for ts in block.tutor_trace:
        if ts.action == "SHORTLIST":
            assert len(ts.shortlist_indices) <= cfg.env.T_max, (
                f"Scenario A: |S|={len(ts.shortlist_indices)} > T_max={cfg.env.T_max}"
            )
    print(f"  PASS  Scenario A size bound: all shortlists <= T_max")


def test_scenario_B_safety_excludes_lethal():
    """Scenario B: high n_risky → lethal options excluded from all shortlists."""
    task_id = _get_task_id()
    cfg = _make_cfg(K=10, H_0=3, n_risky=7, N_teach=2)
    violations = []

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor = OptionLevelTutorAgent(cfg=cfg)
    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(task_id, seed=23)
    support, _, grammar = env.adapter.load_task(task_id)
    tutor.init_block(block, grammar, support)
    learner.init_block(block, grammar, support)

    max_steps = len(block.queries) * 25
    steps = 0
    while not block.done and steps < max_steps:
        steps += 1
        qs = block.current_query
        if qs is None or qs.done:
            break
        hp = qs.hp
        active = get_active_menu(qs)
        lethal = tutor._get_lethal_indices(qs, active, learner)
        ts = tutor.act(block, env, learner_agent=learner)
        if ts.action == "SHORTLIST":
            overlap = set(ts.shortlist_indices) & lethal
            if overlap:
                violations.append(f"HP={hp}: lethal {overlap} in S={ts.shortlist_indices}")
        if qs.done:
            continue
        learner.act(block, env)

    assert not violations, f"Scenario B safety violations:\n" + "\n".join(violations)
    print(f"  PASS  Scenario B: no lethal options in shortlists (n_risky=7)")


def test_scenario_D_ample_time_no_shortlist():
    """Scenario D: T_max >> K, no risky options → tutor should WAIT."""
    task_id = _get_task_id()
    cfg = _make_cfg(K=10, T_max=20, H_0=20, n_risky=0, N_teach=2)
    block, _, _, _ = _run_block(cfg, task_id, seed=43)
    n_shortlist = sum(1 for t in block.tutor_trace if t.action == "SHORTLIST")
    assert n_shortlist == 0, (
        f"Scenario D: expected 0 SHORTLIST (ample time) but got {n_shortlist}"
    )
    print(f"  PASS  Scenario D: no shortlists when K <= tau_t (ample time)")


# ════════════════════════════════════════════════════════════════
# Main runner (without pytest)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    invariants = [
        test_I1_correct_in_shortlist,
        test_I2_shortlist_size,
        test_I3_choice_in_shortlist,
        test_I4_no_lethal_in_shortlist,
        test_I5_no_pick_from_banned,
    ]
    scenarios = [
        test_scenario_A_shortlist_fires,
        test_scenario_A_shortlist_size_bound,
        test_scenario_B_safety_excludes_lethal,
        test_scenario_D_ample_time_no_shortlist,
    ]

    print("=" * 60)
    print("Option-Level Tutor: Invariant Tests")
    print("=" * 60)
    passed = 0
    total = len(invariants) + len(scenarios)
    for t in invariants + scenarios:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()

    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
