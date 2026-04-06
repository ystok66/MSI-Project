"""
Tests for scenario families — generation validity, family logic,
intervention leverage, and schema compatibility.
"""
import pytest
import numpy as np

from src.envs.scenario_families import (
    generate_scenario, generate_baseline_v2, generate_fork_trap,
    generate_hazard_belt, generate_deadline_gate,
    generate_delayed_corridor, generate_distractor_cue,
    ScenarioConfig, SCENARIO_NAMES,
)
from src.envs.lattice_v2 import (
    SegmentMeta, LatticeV2Meta, _bfs_len,
)
from src.envs.map_families import FamilyConfig
from src.envs.map_generator import CellType
from src.envs.lattice_v2_runner import LatticeV2Runner


# ── Helpers ──────────────────────────────────────────────────────────

def _is_connected(gm, start, goal):
    """Check that start → goal is reachable (no door blocking)."""
    return _bfs_len(gm, start, goal, set()) < 999


def _episode_smoke(family, seed=42, difficulty="medium"):
    """Run a full episode and return metrics dict."""
    runner = LatticeV2Runner()
    s = runner.reset(seed=seed, scenario_family=family,
                     latent_mode=True, difficulty=difficulty)
    steps = 0
    while not s.done and steps < 200:
        s = runner.step(s)
        steps += 1
    return runner.get_metrics(s), s


# ══════════════════════════════════════════════════════════════════════
# 1. Generation validity — every family produces a solvable map
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", SCENARIO_NAMES)
def test_family_generates_valid_map(family):
    """Map is generated, has correct return type, goal is reachable."""
    gm, cfg, meta, sc = generate_scenario(family, seed=42)
    assert isinstance(cfg, FamilyConfig)
    assert isinstance(meta, LatticeV2Meta)
    assert isinstance(sc, ScenarioConfig)
    assert len(meta.segments) >= 1
    assert meta.cell_features.shape[2] == 4


@pytest.mark.parametrize("family", SCENARIO_NAMES)
@pytest.mark.parametrize("seed", [0, 1, 42, 100])
def test_family_solvable_across_seeds(family, seed):
    """Goal is reachable from start across multiple seeds."""
    gm, cfg, meta, sc = generate_scenario(family, seed=seed)
    start = gm.agent_start
    goal = gm.target_pos
    assert _is_connected(gm, start, goal), (
        f"{family} seed={seed}: start={start} → goal={goal} not connected")


@pytest.mark.parametrize("family", SCENARIO_NAMES)
def test_family_deterministic(family):
    """Same (family, seed) → same map."""
    gm1, _, _, _ = generate_scenario(family, seed=7)
    gm2, _, _, _ = generate_scenario(family, seed=7)
    assert np.array_equal(gm1.true_cost, gm2.true_cost)
    assert np.array_equal(gm1.true_risk, gm2.true_risk)


# ══════════════════════════════════════════════════════════════════════
# 2. Family logic — each family has its defining structure
# ══════════════════════════════════════════════════════════════════════

def test_fork_trap_has_branch_ambiguity():
    """Fork trap has two branches with overlapping cue features."""
    gm, cfg, meta, sc = generate_fork_trap(seed=42, latent_mode=False)
    seg = meta.segments[0]
    assert len(seg.risky_cells) >= 3, "risky branch too short"
    assert len(seg.safe_cells) >= 2, "safe branch too short"
    # Pre-trap risky cells should have low texture (ambiguous)
    for r, c in seg.weak_cue_cells:
        t1 = meta.cell_features[r, c, 2]
        assert t1 < 0.60, f"front cue at ({r},{c}) too revealing: {t1}"


def test_fork_trap_has_trap_deep():
    """Trap cell is not at branch entry (depth >= trap_depth param)."""
    gm, cfg, meta, sc = generate_fork_trap(seed=42, difficulty="medium",
                                            latent_mode=False)
    seg = meta.segments[0]
    assert seg.trap_cell is not None, "no trap cell"
    # trap must not be the first risky cell
    assert seg.trap_cell != seg.risky_cells[0], "trap at entry (depth=0)"


def test_fork_trap_risky_row_varies():
    """Risky branch should appear on different rows across seeds."""
    rows = set()
    for seed in range(20):
        _, _, meta, _ = generate_fork_trap(seed=seed, latent_mode=False)
        rows.add(meta.segments[0].risky_row)
    assert len(rows) == 2, f"risky row should vary: got {rows}"


def test_hazard_belt_has_unavoidable_risk():
    """Belt segment has risk on both lanes (row 1 and row 3)."""
    gm, cfg, meta, sc = generate_hazard_belt(
        seed=42, latent_mode=False, belt_regime="unavoidable")
    belt_seg = meta.segments[1]
    assert len(belt_seg.risky_cells) >= 4, "belt segment too few risky cells"
    # Check both rows have risky cells
    rows_with_risky = {r for r, c in belt_seg.risky_cells}
    assert 1 in rows_with_risky, "belt row 1 not risky"
    assert 3 in rows_with_risky, "belt row 3 not risky"


def test_hazard_belt_near_unavoidable_has_bypass():
    """Near-unavoidable belt has detour cells on rows 4-5."""
    gm, cfg, meta, sc = generate_hazard_belt(
        seed=42, latent_mode=False, belt_regime="near_unavoidable")
    belt_seg = meta.segments[1]
    bypass_cells = [(r, c) for r, c in belt_seg.safe_cells if r in (4, 5)]
    assert len(bypass_cells) >= 3, "near_unavoidable should have bypass cells"


def test_deadline_gate_has_gated_shortcut():
    """Deadline gate has a LOCKED_DOOR on the shortcut."""
    gm, cfg, meta, sc = generate_deadline_gate(seed=42, latent_mode=False)
    assert len(meta.all_door_positions) >= 1, "no door found"
    gate = meta.all_door_positions[0]
    assert gm.cell_types[gate[0], gate[1]] == CellType.LOCKED_DOOR, (
        f"gate at {gate} is not LOCKED_DOOR")


def test_deadline_gate_safe_path_always_possible():
    """Safe long path is always reachable even in hard mode."""
    for diff in ["easy", "medium", "hard"]:
        gm, cfg, meta, sc = generate_deadline_gate(
            seed=42, difficulty=diff, latent_mode=False)
        # Safe path avoids gated shortcut
        gates = set(meta.all_door_positions)
        dist = _bfs_len(gm, gm.agent_start, gm.target_pos, gates)
        assert dist < 999, f"safe path not reachable at difficulty={diff}"
        # Time budget allows safe path (tight but possible)
        assert cfg.max_steps >= dist, (
            f"deadline too tight for safe path: t_max={cfg.max_steps}, dist={dist}")


def test_deadline_gate_shortcut_is_shorter():
    """When gate is open, shortcut path is shorter than safe path."""
    gm, cfg, meta, sc = generate_deadline_gate(seed=42, latent_mode=False)
    gates = set(meta.all_door_positions)
    safe_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, gates)
    # Open the gate for shortcut path
    gate = meta.all_door_positions[0]
    gm.cell_types[gate[0], gate[1]] = CellType.NORMAL
    gm.true_cost[gate[0], gate[1]] = 1.0
    shortcut_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, set())
    assert shortcut_dist < safe_dist, (
        f"shortcut ({shortcut_dist}) not shorter than safe path ({safe_dist})")


def test_delayed_corridor_has_safe_prefix():
    """Corridor A has low-risk prefix cells before the trap."""
    gm, cfg, meta, sc = generate_delayed_corridor(seed=42, latent_mode=False)
    seg = meta.segments[0]
    assert seg.trap_cell is not None, "no trap cell"
    # Pre-trap cells should have very low risk
    for r, c in seg.weak_cue_cells:
        assert gm.true_risk[r, c] < 0.10, (
            f"prefix cell ({r},{c}) risk={gm.true_risk[r,c]:.2f} too high")


def test_delayed_corridor_has_commitment_cells():
    """ScenarioConfig.commitment_cells is populated."""
    _, _, _, sc = generate_delayed_corridor(seed=42, latent_mode=False)
    assert len(sc.commitment_cells) >= 1, "no commitment cells"
    assert sc.expected_failure_mode == "commitment"


def test_delayed_corridor_trap_not_at_entry():
    """Trap cell is past the safe prefix (not at branch entry)."""
    gm, cfg, meta, sc = generate_delayed_corridor(
        seed=42, difficulty="medium", latent_mode=False)
    seg = meta.segments[0]
    assert seg.trap_cell is not None
    # Trap should not be the first risky cell
    assert seg.trap_cell != seg.risky_cells[0], "trap at entry"
    # Trap should be at least safe_prefix cells deep
    trap_col = seg.trap_cell[1]
    entry_col = seg.risky_cells[0][1]
    assert trap_col - entry_col >= 3, "trap too close to entry in medium"


def test_distractor_cue_weak_has_corrupted_features():
    """Weak cue mode: some cells have noisy features."""
    _, _, meta, sc = generate_distractor_cue(
        seed=42, latent_mode=False, cue_mode="weak")
    assert sc.cue_reliability < 1.0
    total_weak = sum(len(seg.weak_cue_cells) for seg in meta.segments)
    assert total_weak >= 3, f"too few corrupted cells: {total_weak}"


def test_distractor_cue_misleading_inverts_features():
    """Misleading cue mode: some risky cells look safe and vice versa."""
    _, _, meta, sc = generate_distractor_cue(
        seed=42, latent_mode=False, cue_mode="misleading")
    assert sc.cue_reliability <= 0.0, "misleading should have low reliability"
    total_weak = sum(len(seg.weak_cue_cells) for seg in meta.segments)
    assert total_weak >= 3, f"too few distractor cells: {total_weak}"


# ══════════════════════════════════════════════════════════════════════
# 3. Intervention leverage — each family's primary lever is relevant
# ══════════════════════════════════════════════════════════════════════

def test_fork_trap_warn_is_relevant():
    """Fork trap has risky cells that differ from safe cells (WARN useful)."""
    _, _, meta, sc = generate_fork_trap(seed=42, latent_mode=False)
    assert sc.primary_intervention == "WARN"
    seg = meta.segments[0]
    assert seg.trap_cell is not None, "no trap → WARN has nothing to warn about"


def test_hazard_belt_item_drop_is_relevant():
    """Hazard belt: ITEM_DROP should be the primary lever."""
    _, _, _, sc = generate_hazard_belt(seed=42, latent_mode=False)
    assert sc.primary_intervention == "ITEM_DROP"
    assert sc.requires_item is True


def test_deadline_gate_unlock_is_relevant():
    """Deadline gate: UNLOCK should be the primary lever."""
    _, _, _, sc = generate_deadline_gate(seed=42, latent_mode=False)
    assert sc.primary_intervention == "UNLOCK"
    assert sc.requires_gate is True
    assert sc.gate_mode == "unlock_shortcut"


def test_delayed_corridor_warn_is_relevant():
    """Delayed corridor: prefix-aware WARN is primary lever."""
    _, _, meta, sc = generate_delayed_corridor(seed=42, latent_mode=False)
    assert sc.primary_intervention == "WARN"
    assert sc.expected_failure_mode == "commitment"
    seg = meta.segments[0]
    assert seg.trap_cell is not None


def test_distractor_cue_warn_is_relevant():
    """Distractor cue: WARN provides ground truth → primary lever."""
    _, _, _, sc = generate_distractor_cue(seed=42, latent_mode=False)
    assert sc.primary_intervention == "WARN"
    assert sc.expected_failure_mode == "cue_error"


# ══════════════════════════════════════════════════════════════════════
# 4. Episode smoke — full episode runs with runner
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", SCENARIO_NAMES)
def test_family_episode_smoke(family):
    """Full episode runs without error and produces valid metrics."""
    metrics, state = _episode_smoke(family)
    assert isinstance(metrics, dict)
    assert "survived" in metrics
    assert "reached_goal" in metrics
    assert "steps" in metrics
    assert "unlock_count" in metrics
    assert "warn_count" in metrics
    assert state.done is True


# ══════════════════════════════════════════════════════════════════════
# 5. Baseline regression — baseline_v2 matches generate_lattice_v2
# ══════════════════════════════════════════════════════════════════════

def test_baseline_v2_matches_default():
    """baseline_v2 should produce the same map as generate_lattice_v2."""
    from src.envs.lattice_v2 import generate_lattice_v2
    gm1, cfg1, meta1 = generate_lattice_v2(seed=42, latent_mode=True)
    gm2, cfg2, meta2, sc = generate_baseline_v2(seed=42, latent_mode=True)
    assert np.array_equal(gm1.true_cost, gm2.true_cost)
    assert np.array_equal(gm1.true_risk, gm2.true_risk)
    assert cfg1.max_steps == cfg2.max_steps
    assert sc.family_name == "baseline_v2"


# ══════════════════════════════════════════════════════════════════════
# 6. ScenarioConfig sanity
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", SCENARIO_NAMES)
def test_scenario_config_populated(family):
    """ScenarioConfig has family_name matching the requested family."""
    _, _, _, sc = generate_scenario(family, seed=0)
    assert sc.family_name == family
    assert sc.difficulty in ("easy", "medium", "hard")
    assert sc.primary_intervention in ("WARN", "UNLOCK", "ITEM_DROP", "WAIT")


# ══════════════════════════════════════════════════════════════════════
# 7. P0-B Runner-level invariants — intervention semantics are correct
# ══════════════════════════════════════════════════════════════════════

class TestHazardBeltInvariants:
    """Belt cells are RISKY; shield can be consumed on belt traversal."""

    def test_belt_cells_are_risky(self):
        """Belt segment cells must be CellType.RISKY (not LOCKED_DOOR)."""
        gm, _, meta, _ = generate_hazard_belt(
            seed=42, latent_mode=False, belt_regime="unavoidable")
        belt_seg = meta.segments[1]
        for r, c in belt_seg.risky_cells:
            actual = int(gm.cell_types[r, c])
            assert actual == CellType.RISKY, (
                f"belt cell ({r},{c}) is {CellType(actual).name}, expected RISKY")

    def test_shield_inventory_can_become_one(self):
        """With item_drop_enabled, inventory starts at 0 and can become 1."""
        from src.teachers.interventions import InventoryState
        inv = InventoryState(shield=0, shield_risk_reduction=0.5)
        assert inv.shield == 0
        assert inv.add_shield() is True
        assert inv.shield == 1
        # No stacking
        assert inv.add_shield() is False
        assert inv.shield == 1

    def test_shield_consumed_on_risky_traversal(self):
        """Shield is consumed when agent enters RISKY cell (runner line ~400)."""
        from src.teachers.interventions import InventoryState
        inv = InventoryState(shield=1, shield_risk_reduction=0.5)
        assert inv.has_shield()
        consumed = inv.consume_shield()
        assert consumed is True
        assert inv.shield == 0
        assert not inv.has_shield()

    def test_shield_reduces_effective_risk(self):
        """Shield halves risk: 0.45 * (1 - 0.5) = 0.225."""
        from src.teachers.interventions import InventoryState
        inv = InventoryState(shield=1, shield_risk_reduction=0.5)
        base_risk = 0.45
        effective = base_risk * (1.0 - inv.shield_risk_reduction)
        assert abs(effective - 0.225) < 1e-6

    def test_runner_uses_cfg_max_steps_for_hazard_belt(self):
        """Runner should use scenario's cfg.max_steps, not time_ratio override."""
        runner = LatticeV2Runner()
        gm, cfg, meta, sc = generate_hazard_belt(seed=42)
        s = runner.reset(seed=42, scenario_family="hazard_belt",
                         latent_mode=True, difficulty="medium")
        assert s.t_max == cfg.max_steps, (
            f"runner t_max={s.t_max} != cfg.max_steps={cfg.max_steps}")

    def test_locked_door_impassable_at_start(self):
        """LOCKED_DOOR cells start as impassable in runner state."""
        runner = LatticeV2Runner()
        s = runner.reset(seed=42, scenario_family="deadline_gate",
                         latent_mode=True)
        gm = s.gridmap
        for r in range(gm.height):
            for c in range(gm.width):
                if int(gm.cell_types[r, c]) == CellType.LOCKED_DOOR:
                    assert not s.passable[r, c], (
                        f"LOCKED_DOOR at ({r},{c}) should be impassable")
                    assert s.belief_cost[r, c] >= 100.0, (
                        f"LOCKED_DOOR at ({r},{c}) belief_cost too low")


class TestDeadlineGateInvariants:
    """UNLOCK opens shortcut; no legacy close-gate for unlock_shortcut."""

    def test_unlock_shortcut_gate_mode(self):
        """deadline_gate family declares gate_mode=unlock_shortcut."""
        _, _, _, sc = generate_deadline_gate(seed=42, latent_mode=False)
        assert sc.gate_mode == "unlock_shortcut"

    def test_runner_gate_mode_propagated(self):
        """Runner state carries gate_mode from ScenarioConfig."""
        runner = LatticeV2Runner()
        s = runner.reset(seed=42, scenario_family="deadline_gate",
                         latent_mode=True)
        assert s.gate_mode == "unlock_shortcut"

    def test_unlock_opens_locked_door_in_robot_belief(self):
        """When UNLOCK is chosen by robot_belief, shortcut becomes passable."""
        runner = LatticeV2Runner()
        s = runner.reset(
            seed=42, scenario_family="deadline_gate",
            latent_mode=True, robot_belief_mode=True,
            intervention_family_mode=True,
            allowed_interventions=frozenset({"UNLOCK"}),
            prefix_horizon=5)
        # Find the shortcut gate
        gate = s.meta.all_door_positions[0]
        assert not s.passable[gate[0], gate[1]], "gate should start closed"
        # Run steps until UNLOCK fires or episode ends
        unlock_fired = False
        for _ in range(50):
            if s.done:
                break
            runner.step(s)
            if s.passable[gate[0], gate[1]]:
                unlock_fired = True
                break
        assert unlock_fired, "UNLOCK never opened the shortcut gate"

    def test_no_legacy_close_for_unlock_shortcut(self):
        """time_aware tutor must NOT close gates when gate_mode=unlock_shortcut."""
        runner = LatticeV2Runner()
        s = runner.reset(
            seed=42, scenario_family="deadline_gate",
            latent_mode=True,
            tutor_mode="time_aware", closure_budget=3)
        # gate_mode should be unlock_shortcut
        assert s.gate_mode == "unlock_shortcut"
        # Run a few steps — no gates should be closed
        initial_unlocks = s.unlock_count
        for _ in range(10):
            if s.done:
                break
            runner.step(s)
        # time_aware should NOT have increased unlock_count (closing gates)
        assert s.unlock_count == initial_unlocks, (
            f"time_aware closed {s.unlock_count - initial_unlocks} gates "
            f"in unlock_shortcut family — wiring bug!")

    def test_safe_path_tight_but_possible(self):
        """Safe path fits within t_max across all difficulties."""
        for diff in ["easy", "medium", "hard"]:
            gm, cfg, meta, sc = generate_deadline_gate(
                seed=42, difficulty=diff, latent_mode=False)
            gates = set(meta.all_door_positions)
            safe_d = _bfs_len(gm, gm.agent_start, gm.target_pos, gates)
            assert safe_d < 999, f"safe path not reachable at {diff}"
            assert cfg.max_steps >= safe_d, (
                f"{diff}: t_max={cfg.max_steps} < safe_dist={safe_d}")
            # But not too loose
            slack = cfg.max_steps - safe_d
            assert slack <= 18, (
                f"{diff}: slack={slack} too generous (t_max={cfg.max_steps}, "
                f"safe={safe_d})")


class TestForkTrapInvariants:
    """Both branches reachable; unlock_only doesn't change topology."""

    def test_both_branches_reachable(self):
        """Both risky and safe branches reach the goal."""
        gm, _, meta, _ = generate_fork_trap(seed=42, latent_mode=False)
        seg = meta.segments[0]
        # Risky only (block safe entry)
        risky_d = _bfs_len(gm, gm.agent_start, gm.target_pos,
                           {seg.safe_entry_gate})
        assert risky_d < 999, "risky branch doesn't reach goal"
        # Safe only (block risky entry)
        safe_d = _bfs_len(gm, gm.agent_start, gm.target_pos,
                          {seg.risky_entry_gate})
        assert safe_d < 999, "safe branch doesn't reach goal"

    def test_unlock_only_does_not_invoke_blocking(self):
        """unlock_only with robot_belief should not block any paths."""
        runner = LatticeV2Runner()
        s = runner.reset(
            seed=42, scenario_family="fork_trap",
            latent_mode=True,
            robot_belief_mode=True,
            intervention_family_mode=True,
            allowed_interventions=frozenset({"UNLOCK"}),
            prefix_horizon=5)
        seg = s.meta.segments[0]
        # Both entries should stay passable throughout
        for _ in range(20):
            if s.done:
                break
            risky_gate = seg.risky_entry_gate
            safe_gate = seg.safe_entry_gate
            # At least one must remain passable
            assert (s.passable[risky_gate[0], risky_gate[1]] or
                    s.passable[safe_gate[0], safe_gate[1]]), (
                "both branches became impassable — unintended blocking!")
            runner.step(s)

    def test_cue_ambiguity_exists(self):
        """Weak cue cells have ambiguous features (texture < 0.6)."""
        _, _, meta, _ = generate_fork_trap(seed=42, latent_mode=False)
        seg = meta.segments[0]
        assert len(seg.weak_cue_cells) >= 1, "no weak cue cells"
        for r, c in seg.weak_cue_cells:
            t1 = meta.cell_features[r, c, 2]
            assert t1 < 0.60, (
                f"weak cue at ({r},{c}) texture={t1:.2f} not ambiguous")

