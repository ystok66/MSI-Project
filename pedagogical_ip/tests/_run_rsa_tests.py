"""Quick test runner for RSA integration tests."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traceback
import numpy as np

PASS = 0
FAIL = 0

def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS: {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        FAIL += 1

# ── 1. Context mapping helpers ──

def test_map_segment_context():
    from src.envs.lattice_v2 import SegmentMeta
    from src.agents.warning_update import map_segment_to_rsa_context

    seg = SegmentMeta(
        index=0, col_start=2, col_end=7,
        risky_row=1, safe_row=3,
        L_risky=5, L_safe=7, detour_len=1,
        risky_cells=[], safe_cells=[],
        risky_entry_gate=(1, 2), safe_entry_gate=(3, 2),
        trap_cell=None, weak_cue_cells=[],
    )
    ctx = map_segment_to_rsa_context(seg)
    assert ctx["risky_side"] == "left", f"Expected left, got {ctx['risky_side']}"

    seg2 = SegmentMeta(
        index=0, col_start=2, col_end=7,
        risky_row=3, safe_row=1,
        L_risky=5, L_safe=7, detour_len=1,
        risky_cells=[], safe_cells=[],
        risky_entry_gate=(3, 2), safe_entry_gate=(1, 2),
        trap_cell=None, weak_cue_cells=[],
    )
    ctx2 = map_segment_to_rsa_context(seg2)
    assert ctx2["risky_side"] == "right", f"Expected right, got {ctx2['risky_side']}"

def test_map_legacy_utterance():
    from src.agents.warning_update import map_legacy_to_rsa_utterance, Utterance
    from src.agents.rsa_warning_channel import RSAUtterance

    assert map_legacy_to_rsa_utterance(Utterance.UPPER_LANE_RISKY, "left") == RSAUtterance.WARN_LEFT
    assert map_legacy_to_rsa_utterance(Utterance.UPPER_LANE_RISKY, "right") == RSAUtterance.WARN_RIGHT
    assert map_legacy_to_rsa_utterance(Utterance.RISKY_TEXTURE_AHEAD, "left") == RSAUtterance.WARN_AHEAD

# ── 2. Runner integration ──

def test_legacy_variant_unchanged():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="legacy_bias")
    while not s.done:
        s = runner.step(s)
    m = runner.get_extended_metrics(s)
    assert m["warning_variant"] == "legacy_bias"
    assert s.warn_count >= 1, f"Expected warnings, got {s.warn_count}"
    assert len(s.warned_lane_bias) > 0, "Legacy should populate warned_lane_bias"
    assert s.rsa_channel is None
    assert s.rsa_belief_state is None

def test_rsa_s1_no_lane_bias():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_obs_s1")
    while not s.done:
        s = runner.step(s)
    assert s.warn_count >= 1, f"Expected warnings, got {s.warn_count}"
    assert len(s.warned_lane_bias) == 0, f"RSA should NOT populate warned_lane_bias, got {s.warned_lane_bias}"

def test_rsa_belief_updates():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.agents.rsa_warning_channel import N_HYPOTHESES
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_obs_s1")
    while not s.done:
        s = runner.step(s)
    assert s.rsa_belief_state is not None
    assert s.rsa_belief_state.n_updates >= 1
    uniform = np.ones(N_HYPOTHESES) / N_HYPOTHESES
    assert not np.allclose(s.rsa_belief_state.belief, uniform, atol=1e-3), \
        f"Belief should differ from uniform: {s.rsa_belief_state.belief}"

def test_rsa_planner_adapter():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_obs_s1")
    while not s.done:
        s = runner.step(s)
    assert s.warn_count >= 1
    assert len(s.warned_cell_extra) > 0, "RSA should feed warned_cell_extra via adapter"
    assert len(s.warned_lane_bias) == 0, "But not via legacy lane bias"

def test_no_warning_parity():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    results = {}
    for variant in ["legacy_bias", "rsa_obs_l0", "rsa_obs_s1"]:
        runner = LatticeV2Runner()
        s = runner.reset(seed=42, latent_mode=True, warning_mode="none",
                         scenario_family="baseline_v2", warning_variant=variant)
        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        results[variant] = (m["survived"], m["reached_goal"], m["steps"], m["risky_entered"])
    legacy = results["legacy_bias"]
    for v in ["rsa_obs_l0", "rsa_obs_s1"]:
        assert results[v] == legacy, f"No-warning parity: {v} {results[v]} != legacy {legacy}"

def test_rsa_no_phase10():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_obs_s1")
    while not s.done:
        s = runner.step(s)
    for seg in s.meta.segments:
        for rc in seg.risky_cells:
            mem = s.feature_belief.memory[rc[0], rc[1]]
            assert "warned" not in mem.intervention_tags, \
                f"RSA should NOT set 'warned' tag on cell {rc}"

def test_extended_metrics_rsa():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.agents.rsa_warning_channel import N_HYPOTHESES
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_obs_s1")
    while not s.done:
        s = runner.step(s)
    m = runner.get_extended_metrics(s)
    assert m["warning_variant"] == "rsa_obs_s1"
    assert "rsa_belief_final" in m
    assert len(m["rsa_belief_final"]) == N_HYPOTHESES
    assert "rsa_entropy_final" in m

def test_hybrid_variant():
    from src.envs.lattice_v2_runner import LatticeV2Runner
    runner = LatticeV2Runner()
    s = runner.reset(seed=42, latent_mode=True, warning_mode="fixed",
                     scenario_family="fork_trap", warning_variant="rsa_plus_phase10")
    while not s.done:
        s = runner.step(s)
    assert s.warn_count >= 1
    assert s.rsa_belief_state is not None
    assert s.rsa_belief_state.n_updates >= 1
    # Hybrid: legacy lane_bias should ALSO be populated
    assert len(s.warned_lane_bias) > 0, "Hybrid should populate warned_lane_bias"

# ── Run all ──

print("=" * 60)
print("RSA Runner Integration Tests")
print("=" * 60)

run("map_segment_context", test_map_segment_context)
run("map_legacy_utterance", test_map_legacy_utterance)
run("legacy_variant_unchanged", test_legacy_variant_unchanged)
run("rsa_s1_no_lane_bias", test_rsa_s1_no_lane_bias)
run("rsa_belief_updates", test_rsa_belief_updates)
run("rsa_planner_adapter", test_rsa_planner_adapter)
run("no_warning_parity", test_no_warning_parity)
run("rsa_no_phase10", test_rsa_no_phase10)
run("extended_metrics_rsa", test_extended_metrics_rsa)
run("hybrid_variant", test_hybrid_variant)

print("=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
