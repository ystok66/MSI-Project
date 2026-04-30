"""
test_inverse_boundary.py — Anti-cheat boundary tests for inverse predictor.

7 tests ensuring the inverse predictor scaffold is epistemically clean:

1. inverse predictor cannot access real scorer
2. inverse predictor cannot access real danger_head
3. same public history → same inverse action (even if private state differs)
4. ObservedStep contains no hidden correct flag for unchosen options
5. inverse predictor has no LearnerAgent reference
6. oracle-forward matches legacy sparse (seeded)
7. inverse action_dist does not call direct-access methods (Bomb test)
"""
import sys
import os
import copy
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
from cls_option_tutor.tutor.observation_adapter import ObservationAdapter, ObservedStep
from cls_option_tutor.tutor.learner_model import (
    ShadowLearnerModel, PROFILE_GRID,
)
from cls_option_tutor.tutor.inverse_predictor import InverseShadowPredictor
from cls_option_tutor.tutor.oracle_predictor import OracleForwardPredictor
from cls_option_tutor.interfaces import Option


DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')
)
TASK_ID = "000001"


def _make_cfg():
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.n_sup = 4
    cfg.learner.n_em = 1
    cfg.learner.use_hpc = False
    cfg.env.K = 6
    cfg.env.T_max = 3
    cfg.env.N_obs = 1
    cfg.env.N_teach = 1
    cfg.env.N_eval = 1
    cfg.env.M_queries = 3
    cfg.env.n_risky = 2
    cfg.tutor.rollout_mode = "proxy"
    return cfg


def _make_shadow_model(cfg, env, task_id):
    """Create an independent shadow model from public support data."""
    from cls_option_tutor.learner.cls_adapter import create_scorer
    from cls_option_tutor.learner.danger_head import DangerHead

    support, _, grammar = env.adapter.load_task(task_id)

    shadow_scorer = create_scorer(
        grammar, support,
        use_cls=True,
        n_sup=cfg.learner.n_sup,
        n_em=cfg.learner.n_em,
        use_hpc=cfg.learner.use_hpc,
    )

    shadow_dh = DangerHead(m=cfg.env.danger_dim)

    return ShadowLearnerModel(
        scorer=shadow_scorer,
        danger_head=shadow_dh,
        attention_L=4,
        rho_H=cfg.learner.rho_H,
    )


# ── Test 1: Cannot access real scorer ─────────────────────────────────────────

def test_inverse_predictor_cannot_access_real_scorer():
    """InverseShadowPredictor must not hold any reference to a real CLS scorer."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    pred = InverseShadowPredictor(shadow_model=shadow)

    # Check that no attribute holds a scorer from a real LearnerAgent
    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)
    learner.init_block(block, grammar, support)

    real_scorer = learner._scorer

    # Walk predictor attributes
    for name in dir(pred):
        val = getattr(pred, name, None)
        if val is real_scorer:
            pytest.fail(f"InverseShadowPredictor.{name} IS the real scorer object")


# ── Test 2: Cannot access real danger_head ────────────────────────────────────

def test_inverse_predictor_cannot_access_real_danger_head():
    """InverseShadowPredictor must not hold any reference to real danger_head."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    pred = InverseShadowPredictor(shadow_model=shadow)

    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)
    learner.init_block(block, grammar, support)

    real_dh = learner.policy.danger_head

    for name in dir(pred):
        val = getattr(pred, name, None)
        if val is real_dh:
            pytest.fail(f"InverseShadowPredictor.{name} IS the real danger_head")


# ── Test 3: Same public history → same inverse prediction ─────────────────────

def test_same_public_history_same_prediction():
    """Two learners with identical public traces but different private states
    must produce identical inverse predictions."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow1 = _make_shadow_model(cfg, env, TASK_ID)
    shadow2 = _make_shadow_model(cfg, env, TASK_ID)

    pred1 = InverseShadowPredictor(shadow_model=shadow1)
    pred2 = InverseShadowPredictor(shadow_model=shadow2)

    # Build a synthetic public step
    step = ObservedStep(
        step_id=0,
        phase="teach",
        query_id=0,
        round_t=0,
        option_texts=(("red", "big"), ("blue", "small")),
        option_danger_vecs=(np.zeros(16), np.ones(16) * 0.5),
        option_indices=(0, 1),
        target_output=("red", "big"),
        active_bans=(),
        active_highlights=(),
        active_risk_hints=(),
        hp_before=5,
        hp_after=5,
        rounds_before=0,
        rounds_after=1,
        learner_action="pick",
        learner_pick_index=0,
        pick_correct=True,
        pick_damage=None,
        revealed_output=None,
        revealed_danger_vec=None,
        outcome="pick_correct",
        assist_level="none",
    )

    pred1.observe(step)
    pred2.observe(step)

    # Both should have identical profile posteriors
    np.testing.assert_allclose(
        pred1._log_weights, pred2._log_weights, atol=1e-10,
        err_msg="Same public history should produce identical profile posteriors"
    )


# ── Test 4: ObservedStep has no hidden correct flag ───────────────────────────

def test_observed_step_no_hidden_correct():
    """ObservedStep must not expose is_correct for unchosen options."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)

    learner = LearnerAgent(cfg=cfg)
    learner.init_block(block, grammar, support)

    # Run one step
    qs = block.current_query
    if qs and not qs.done:
        env.tutor_act(block, "WAIT")
        if not qs.done:
            learner.act(block, env)

    adapter = ObservationAdapter()
    steps = adapter.extract_steps(block)

    for step in steps:
        # Verify no field contains is_correct info
        assert not hasattr(step, 'is_correct'), "ObservedStep has is_correct!"
        assert not hasattr(step, 'options'), "ObservedStep has Option objects!"
        # Check field names
        for field_name in step.__dataclass_fields__:
            assert 'correct' not in field_name.lower() or field_name == 'pick_correct', \
                f"Suspicious field: {field_name}"


# ── Test 5: Inverse predictor has no LearnerAgent reference ───────────────────

def test_inverse_predictor_has_no_learner_reference():
    """InverseShadowPredictor must not hold a LearnerAgent instance anywhere."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    pred = InverseShadowPredictor(shadow_model=shadow)

    def _check_no_learner(obj, path="pred", depth=0):
        if depth > 5:
            return
        if isinstance(obj, LearnerAgent):
            pytest.fail(f"Found LearnerAgent at {path}")
        if hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                _check_no_learner(v, f"{path}.{k}", depth + 1)

    _check_no_learner(pred)


# ── Test 6: Oracle-forward matches legacy sparse ──────────────────────────────

def test_sparse_oracle_forward_matches_legacy():
    """sparse with predictor=None vs predictor=OracleForwardPredictor
    should produce the same tutor actions on the same seeded block."""
    cfg = _make_cfg()
    cfg.tutor.rollout_mode = "proxy"  # avoid rollout randomness
    cfg.tutor.sparse_g_learn_mode = "none"

    env1 = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    env2 = OptionEnv(cfg=cfg, data_dir=DATA_DIR)

    # Legacy: no predictor
    tutor1 = SparseTutorAgent(cfg=cfg, predictor=None)
    learner1 = LearnerAgent(cfg=cfg, seed=42)
    block1 = tutor1.run_block(env1, learner1, TASK_ID, seed=42)

    # Oracle predictor
    tutor2 = SparseTutorAgent(cfg=cfg)
    oracle_pred = OracleForwardPredictor(tutor2)
    tutor2._predictor = oracle_pred
    learner2 = LearnerAgent(cfg=cfg, seed=42)
    block2 = tutor2.run_block(env2, learner2, TASK_ID, seed=42)

    # Compare tutor action sequences
    actions1 = [ts.action for ts in block1.tutor_trace]
    actions2 = [ts.action for ts in block2.tutor_trace]
    assert actions1 == actions2, (
        f"Oracle predictor changed tutor actions!\n"
        f"Legacy:  {actions1}\n"
        f"Oracle:  {actions2}"
    )


# ── Test 7: Inverse action_dist doesn't call direct-access (Bomb test) ────────

def test_inverse_does_not_call_direct_access():
    """Monkeypatch real learner private fields to raise on access.
    Verify inverse predictor still works (uses shadow, not real learner)."""

    class Bomb:
        """Explodes on any attribute access."""
        def __getattr__(self, name):
            raise RuntimeError(f"BOMB: direct access to private field .{name}!")

    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    pred = InverseShadowPredictor(shadow_model=shadow, rollout_mode="proxy")

    # Create a real learner and nuke its internals
    learner = LearnerAgent(cfg=cfg)
    block = env.reset_block(TASK_ID, seed=42)
    support, _, grammar = env.adapter.load_task(TASK_ID)
    learner.init_block(block, grammar, support)
    qs = block.current_query

    # Build active menu from public data
    from cls_option_tutor.env.interventions import get_active_menu
    active = get_active_menu(qs)

    # Now plant bombs — inverse predictor should NOT touch these
    learner._scorer = Bomb()
    learner.policy.danger_head = Bomb()
    learner.policy.attention = Bomb()

    # These should NOT crash (they use shadow, not real learner)
    try:
        result = pred.pick_dist(qs, active, {"action": "WAIT"})
        assert isinstance(result, np.ndarray)
    except RuntimeError as e:
        if "BOMB" in str(e):
            pytest.fail(f"Inverse predictor accessed real learner internals: {e}")
        raise

    try:
        p_d, p_t, p_s = pred.rollout(qs, active, {"action": "WAIT"}, n=2)
    except RuntimeError as e:
        if "BOMB" in str(e):
            pytest.fail(f"Inverse rollout accessed real learner internals: {e}")
        raise

# ── Test 8: BAN removes banned option mass ────────────────────────────────────

def test_inverse_ban_removes_banned_option_mass():
    """Inverse predictor must give 0 probability to banned options."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    from cls_option_tutor.tutor.learner_model import PROFILE_GRID

    block = env.reset_block(TASK_ID, seed=42)
    qs = block.current_query

    target = list(qs.target_output)
    from cls_option_tutor.env.interventions import get_active_menu
    active = get_active_menu(qs)
    texts = [list(o.text) for o in active]
    dvecs = [o.danger_vec for o in active]
    option_indices = [o.index for o in active]

    # Pick a non-correct option to ban
    ban_target = None
    for o in active:
        if not o.is_correct:
            ban_target = o.index
            break
    assert ban_target is not None, "No non-correct option found for ban test"

    profile = PROFILE_GRID[0]
    spec_ban = {"action": "BAN", "ban_index": ban_target}

    probs = shadow.predict_pick_probs(
        target_output=target,
        option_texts=texts,
        option_danger_vecs=dvecs,
        profile=profile,
        spec=spec_ban,
        option_indices=option_indices,
    )

    # Find the position of the banned option
    banned_pos = option_indices.index(ban_target)
    assert probs[banned_pos] == 0.0, (
        f"Banned option index {ban_target} has prob {probs[banned_pos]}, expected 0.0"
    )
    assert abs(probs.sum() - 1.0) < 1e-6, f"Probs do not sum to 1: {probs.sum()}"


# ── Test 9: MIX removes ban AND applies highlight ─────────────────────────────

def test_inverse_mix_removes_ban_and_applies_highlight():
    """MIX = BAN + HIGHLIGHT. Banned option must have 0 mass."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    from cls_option_tutor.tutor.learner_model import PROFILE_GRID

    block = env.reset_block(TASK_ID, seed=42)
    qs = block.current_query

    from cls_option_tutor.env.interventions import get_active_menu
    active = get_active_menu(qs)
    texts = [list(o.text) for o in active]
    dvecs = [o.danger_vec for o in active]
    option_indices = [o.index for o in active]

    ban_target = None
    for o in active:
        if not o.is_correct:
            ban_target = o.index
            break

    profile = PROFILE_GRID[0]
    L = len(qs.target_output)
    hl_cells = (0,) if L > 0 else ()

    spec_mix = {
        "action": "MIX",
        "ban_index": ban_target,
        "highlight_cells": hl_cells,
    }

    probs = shadow.predict_pick_probs(
        target_output=list(qs.target_output),
        option_texts=texts,
        option_danger_vecs=dvecs,
        profile=profile,
        spec=spec_mix,
        option_indices=option_indices,
    )

    banned_pos = option_indices.index(ban_target)
    assert probs[banned_pos] == 0.0, f"MIX banned option has prob {probs[banned_pos]}"
    assert abs(probs.sum() - 1.0) < 1e-6


# ── Test 10: Risk update applied exactly once per wrong pick ──────────────────

def test_inverse_risk_update_applied_once_per_wrong_pick():
    """Shadow danger_head should be updated exactly once per wrong-pick event."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)

    pred = InverseShadowPredictor(shadow_model=shadow)

    dv = np.ones(cfg.env.danger_dim) * 0.5
    n_before = shadow.danger_head.hazard._n_updates

    step = ObservedStep(
        step_id=0, phase="teach", query_id=0, round_t=0,
        option_texts=(("red", "big"), ("blue", "small")),
        option_danger_vecs=(np.zeros(cfg.env.danger_dim), dv.copy()),
        option_indices=(0, 1),
        target_output=("red", "big"),
        active_bans=(), active_highlights=(), active_risk_hints=(),
        hp_before=5, hp_after=3, rounds_before=0, rounds_after=1,
        learner_action="pick", learner_pick_index=1,
        pick_correct=False, pick_damage=2,
        revealed_output=("red", "big"),
        revealed_danger_vec=dv.copy(),
        outcome="pick_wrong", assist_level="none",
    )

    pred.observe(step)
    n_after = shadow.danger_head.hazard._n_updates

    # Exactly one hazard update (not two from double-count)
    assert n_after == n_before + 1, (
        f"Expected 1 risk update, got {n_after - n_before}"
    )


# ── Test 11: Round-0 highlight persists after observe ─────────────────────────

def test_inverse_round0_highlight_persists_after_observe():
    """A highlight on round 0 must not be erased by the attention reset."""
    cfg = _make_cfg()
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    shadow = _make_shadow_model(cfg, env, TASK_ID)
    shadow.reset_attention(3)

    pred = InverseShadowPredictor(shadow_model=shadow)

    # Uniform attention before
    attn_before = shadow.attention.copy()
    assert abs(attn_before[0] - attn_before[1]) < 1e-6, "Not uniform before"

    step = ObservedStep(
        step_id=0, phase="teach", query_id=0, round_t=0,
        option_texts=(("a", "b", "c"),),
        option_danger_vecs=(np.zeros(cfg.env.danger_dim),),
        option_indices=(0,),
        target_output=("a", "b", "c"),
        active_bans=(), active_highlights=(1,), active_risk_hints=(),
        hp_before=5, hp_after=5, rounds_before=0, rounds_after=1,
        learner_action="pick", learner_pick_index=0,
        pick_correct=True, pick_damage=None,
        revealed_output=None, revealed_danger_vec=None,
        outcome="pick_correct", assist_level="highlight",
    )

    pred.observe(step)

    attn_after = shadow.attention
    # Cell 1 should be boosted relative to cells 0 and 2
    assert attn_after[1] > attn_after[0], (
        f"Highlight on cell 1 was erased: attn={attn_after}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
