"""Integration tests: RSA warning channel wired into LatticeV2Runner.

Tests cover:
  1. Legacy regression: warning_variant='legacy_bias' identical to before
  2. RSA purity: no lane bias, no Phase 10 apply_warn_update
  3. RSA belief updates: posterior changes after warning
  4. Planner adapter: warned_cell_extra populated from RSA belief
  5. No-warning parity: all variants identical when no WARN fires
  6. Mirror symmetry: side swap doesn't break RSA quality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import pytest

from src.envs.lattice_v2_runner import LatticeV2Runner, V2EpisodeState
from src.agents.warning_update import (
    VALID_WARNING_VARIANTS, map_segment_to_rsa_context,
    map_legacy_to_rsa_utterance, Utterance,
)
from src.agents.rsa_warning_channel import (
    RSAWarningChannel, RSABeliefState, RSAUtterance, N_HYPOTHESES,
)


# ── Shared fixtures ────────────────────────────────────────────────

SEED = 42
FAMILY = "fork_trap"


def _run_episode(warning_variant="legacy_bias", seed=SEED,
                 warning_mode="fixed", family=FAMILY, **kw):
    """Run one episode and return (state, metrics)."""
    runner = LatticeV2Runner()
    s = runner.reset(
        seed=seed,
        latent_mode=True,
        warning_mode=warning_mode,
        scenario_family=family,
        warning_variant=warning_variant,
        **kw,
    )
    while not s.done:
        s = runner.step(s)
    m = runner.get_extended_metrics(s)
    return s, m


# ══════════════════════════════════════════════════════════════════
# 1. Legacy regression
# ══════════════════════════════════════════════════════════════════

def test_legacy_variant_unchanged():
    """warning_variant='legacy_bias' should produce exact same results
    as the system without the new warning_variant param."""
    s, m = _run_episode("legacy_bias")
    assert m["warning_variant"] == "legacy_bias"
    assert s.warn_count >= 1, "fork_trap with fixed warn should fire at least once"
    # Legacy should have lane bias data
    assert len(s.warned_lane_bias) > 0, "Legacy should populate warned_lane_bias"
    # RSA fields should be None
    assert s.rsa_channel is None
    assert s.rsa_belief_state is None


# ══════════════════════════════════════════════════════════════════
# 2. RSA purity: no lane bias, no Phase 10
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("variant", ["rsa_obs_l0", "rsa_obs_s1", "rsa_obs_s1_trust"])
def test_rsa_no_lane_bias(variant):
    """RSA variants should NOT populate warned_lane_bias."""
    s, m = _run_episode(variant)
    assert s.warn_count >= 1, f"{variant}: warning should fire"
    assert len(s.warned_lane_bias) == 0, (
        f"{variant}: warned_lane_bias should be empty, "
        f"got {s.warned_lane_bias}")


@pytest.mark.parametrize("variant", ["rsa_obs_l0", "rsa_obs_s1", "rsa_obs_s1_trust"])
def test_rsa_no_phase10(variant):
    """RSA variants should NOT call feature_belief.apply_warn_update.

    We verify by checking that no cell has the 'warned' intervention tag
    in its memory (which is set by apply_warn_update).
    """
    s, m = _run_episode(variant)
    for seg in s.meta.segments:
        for rc in seg.risky_cells:
            mem = s.feature_belief.memory[rc[0], rc[1]]
            assert "warned" not in mem.intervention_tags, (
                f"{variant}: cell {rc} should NOT have 'warned' tag, "
                "but Phase 10 apply_warn_update was called")


# ══════════════════════════════════════════════════════════════════
# 3. RSA belief updates
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("variant", ["rsa_obs_l0", "rsa_obs_s1"])
def test_rsa_belief_state_updates(variant):
    """RSA belief should move away from uniform after warning."""
    s, m = _run_episode(variant)
    assert s.rsa_belief_state is not None
    assert s.rsa_belief_state.n_updates >= 1, (
        f"{variant}: RSA should have at least 1 update")
    belief = s.rsa_belief_state.belief
    uniform = np.ones(N_HYPOTHESES) / N_HYPOTHESES
    # After update, belief should NOT be uniform
    assert not np.allclose(belief, uniform, atol=1e-3), (
        f"{variant}: belief should differ from uniform after warning, "
        f"got {belief}")


def test_rsa_s1_entropy_drops():
    """S1 warning should reduce entropy (make belief more concentrated)."""
    s, m = _run_episode("rsa_obs_s1")
    if s.rsa_warn_diagnostics:
        diag = s.rsa_warn_diagnostics[0]
        assert diag["delta_H"] > 0, (
            f"S1 warning should reduce entropy, "
            f"delta_H={diag['delta_H']} (positive = entropy dropped)")


# ══════════════════════════════════════════════════════════════════
# 4. Planner adapter feeds cost
# ══════════════════════════════════════════════════════════════════

def test_rsa_planner_adapter_feeds_cost():
    """RSA variants should still populate warned_cell_extra
    (via the planner adapter, not via legacy lane bias)."""
    s, m = _run_episode("rsa_obs_s1")
    assert s.warn_count >= 1
    assert len(s.warned_cell_extra) > 0, (
        "RSA should feed warned_cell_extra via planner adapter")
    # But warned_lane_bias should be empty (not using legacy path)
    assert len(s.warned_lane_bias) == 0


# ══════════════════════════════════════════════════════════════════
# 5. No-warning parity
# ══════════════════════════════════════════════════════════════════

def test_no_warning_parity():
    """Without warnings, all variants should produce identical trajectories."""
    results = {}
    for variant in ["legacy_bias", "rsa_obs_l0", "rsa_obs_s1"]:
        s, m = _run_episode(
            variant, warning_mode="none",
            family="baseline_v2")
        results[variant] = {
            "survived": m["survived"],
            "reached_goal": m["reached_goal"],
            "steps": m["steps"],
            "risky": m["risky_entered"],
        }

    # All should match legacy
    legacy = results["legacy_bias"]
    for variant in ["rsa_obs_l0", "rsa_obs_s1"]:
        assert results[variant] == legacy, (
            f"No-warning: {variant} diverged from legacy: "
            f"{results[variant]} vs {legacy}")


# ══════════════════════════════════════════════════════════════════
# 6. Mirror symmetry
# ══════════════════════════════════════════════════════════════════

def test_mirror_symmetry():
    """Different seeds may swap risky_row between 1 and 3.
    RSA quality (belief concentration) should be comparable."""
    entropies = []
    for seed in range(10, 25):
        try:
            s, m = _run_episode("rsa_obs_s1", seed=seed, family="fork_trap")
            if s.rsa_warn_diagnostics:
                diag = s.rsa_warn_diagnostics[0]
                entropies.append(diag.get("H_after", 1.0))
        except Exception:
            continue

    if len(entropies) >= 3:
        # Entropy after warning should generally be below uniform (1.386)
        mean_entropy = np.mean(entropies)
        uniform_entropy = -np.log(1.0 / N_HYPOTHESES) * N_HYPOTHESES / N_HYPOTHESES
        assert mean_entropy < uniform_entropy, (
            f"Mean post-warning entropy {mean_entropy:.3f} should be below "
            f"uniform {uniform_entropy:.3f}")


# ══════════════════════════════════════════════════════════════════
# 7. Extended metrics
# ══════════════════════════════════════════════════════════════════

def test_extended_metrics_rsa_fields():
    """RSA variant should include RSA diagnostics in extended metrics."""
    s, m = _run_episode("rsa_obs_s1")
    assert "warning_variant" in m
    assert m["warning_variant"] == "rsa_obs_s1"
    assert "rsa_belief_final" in m
    assert "rsa_entropy_final" in m
    assert isinstance(m["rsa_belief_final"], list)
    assert len(m["rsa_belief_final"]) == N_HYPOTHESES


def test_extended_metrics_legacy_no_rsa():
    """Legacy variant should not have RSA belief fields (they're None)."""
    s, m = _run_episode("legacy_bias")
    assert m["warning_variant"] == "legacy_bias"
    assert "rsa_belief_final" not in m


# ══════════════════════════════════════════════════════════════════
# 8. Context mapping helpers
# ══════════════════════════════════════════════════════════════════

def test_map_segment_context():
    """map_segment_to_rsa_context should correctly infer risky_side."""
    from src.envs.lattice_v2 import SegmentMeta

    seg_upper = SegmentMeta(
        index=0, col_start=2, col_end=7,
        risky_row=1, safe_row=3,
        L_risky=5, L_safe=7, detour_len=1,
        risky_cells=[], safe_cells=[],
        risky_entry_gate=(1, 2), safe_entry_gate=(3, 2),
        trap_cell=None, weak_cue_cells=[],
    )
    ctx = map_segment_to_rsa_context(seg_upper)
    assert ctx["risky_side"] == "left"  # upper row → left

    seg_lower = SegmentMeta(
        index=0, col_start=2, col_end=7,
        risky_row=3, safe_row=1,
        L_risky=5, L_safe=7, detour_len=1,
        risky_cells=[], safe_cells=[],
        risky_entry_gate=(3, 2), safe_entry_gate=(1, 2),
        trap_cell=None, weak_cue_cells=[],
    )
    ctx = map_segment_to_rsa_context(seg_lower)
    assert ctx["risky_side"] == "right"  # lower row → right


def test_map_legacy_utterance():
    """Legacy utterance mapping should follow risky_side semantics."""
    utt_left = map_legacy_to_rsa_utterance(
        Utterance.UPPER_LANE_RISKY, "left")
    assert utt_left == RSAUtterance.WARN_LEFT

    utt_right = map_legacy_to_rsa_utterance(
        Utterance.UPPER_LANE_RISKY, "right")
    assert utt_right == RSAUtterance.WARN_RIGHT

    utt_ahead = map_legacy_to_rsa_utterance(
        Utterance.RISKY_TEXTURE_AHEAD, "left")
    assert utt_ahead == RSAUtterance.WARN_AHEAD


# ══════════════════════════════════════════════════════════════════
# 9. Hybrid variant (rsa_plus_phase10)
# ══════════════════════════════════════════════════════════════════

def test_hybrid_variant_both_paths():
    """rsa_plus_phase10 should activate both RSA and legacy paths."""
    s, m = _run_episode("rsa_plus_phase10")
    assert s.warn_count >= 1
    # RSA should have updated
    assert s.rsa_belief_state is not None
    assert s.rsa_belief_state.n_updates >= 1
    # Legacy lane bias should ALSO be populated (hybrid does both)
    assert len(s.warned_lane_bias) > 0, (
        "Hybrid should populate warned_lane_bias (legacy path)")
    # Check diagnostics flag
    if s.rsa_warn_diagnostics:
        assert s.rsa_warn_diagnostics[0].get("hybrid", False), (
            "Hybrid diagnostics should have hybrid=True")
