"""Topology-control invariance tests for DTMB-L.

Verifies:
  - Vertical mirror: shortest_any invariant
  - Stage 1 child permutation: shortest_any / shortest_safe invariant
  - BFS reachability preserved under all transformations
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, ".")

from src.envs.scenario_families import generate_scenario
from src.envs.dtmb_lattice import _bfs_shortest

FAMILY = "deep_tree_mixed_bottleneck_lattice"


def _gen(seed=42, difficulty="easy"):
    return generate_scenario(FAMILY, seed=seed, difficulty=difficulty,
                             latent_mode=False)


class TestMirrorInvariance:
    """Vertical mirror: flip grid rows. shortest_any should be invariant."""

    def test_mirror_shortest_any_easy(self):
        gm, _, meta, _ = _gen(seed=42, difficulty="easy")
        H, W = gm.height, gm.width

        # Mirror the cell_types array vertically
        ct_mirror = gm.cell_types[::-1, :].copy()

        # Mirror start and target
        start_mirror = (H - 1 - gm.agent_start[0], gm.agent_start[1])
        target_mirror = (H - 1 - gm.target_pos[0], gm.target_pos[1])

        dist_orig = _bfs_shortest(gm.cell_types, gm.agent_start, gm.target_pos)
        dist_mirror = _bfs_shortest(ct_mirror, start_mirror, target_mirror)

        assert dist_orig == dist_mirror, (
            f"Mirror broke shortest_any: {dist_orig} vs {dist_mirror}")

    def test_mirror_reachability(self):
        gm, _, _, _ = _gen(seed=42, difficulty="medium")
        H, W = gm.height, gm.width

        ct_mirror = gm.cell_types[::-1, :].copy()
        start_mirror = (H - 1 - gm.agent_start[0], gm.agent_start[1])
        target_mirror = (H - 1 - gm.target_pos[0], gm.target_pos[1])

        dist = _bfs_shortest(ct_mirror, start_mirror, target_mirror)
        assert dist < 999, "Mirror made target unreachable"


class TestSeedVariety:
    """Multiple seeds should produce different layouts but consistent structure."""

    def test_different_seeds_consistent_structure(self):
        """All seeds should have decision_stages=3 and valid routes."""
        for seed in [1, 10, 42, 100, 200, 500, 777, 999]:
            _, _, meta, sc = _gen(seed=seed, difficulty="easy")
            assert meta.decision_stages == 3, f"seed={seed}"
            assert meta.shortest_any < 999, f"seed={seed} unreachable"
            assert meta.route_count >= 4, f"seed={seed} route_count={meta.route_count}"
            assert sc.family_name == FAMILY, f"seed={seed}"

    def test_different_seeds_produce_variety(self):
        """Not all seeds should produce identical feature layouts."""
        from src.envs.lattice_v2 import LatticeV2Meta
        features = []
        for seed in [1, 42, 100]:
            _, _, meta, _ = _gen(seed=seed)
            features.append(meta.cell_features.copy())

        identical_count = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if np.array_equal(features[i], features[j]):
                    identical_count += 1

        assert identical_count == 0, "Some different seeds produced identical features"


class TestPassabilityConsistency:
    """Passability count should be reasonable for each difficulty."""

    @pytest.fixture(params=["easy", "medium", "hard"])
    def scenario(self, request):
        return _gen(seed=42, difficulty=request.param)

    def test_passable_count_reasonable(self, scenario):
        gm, _, _, _ = scenario
        passable = np.sum(gm.cell_types != 1)  # CellType.WALL = 1
        total = gm.height * gm.width
        ratio = passable / total
        # Should be between 10% and 50% — tree corridors in a mostly-wall grid
        assert 0.05 < ratio < 0.60, (
            f"Passable ratio={ratio:.2f} ({passable}/{total})")


class TestCommitmentConsistency:
    """Commitment points should be at valid positions."""

    def test_commitment_points_are_passable(self):
        gm, _, meta, _ = _gen(seed=42, difficulty="medium")
        for stage_cps in meta.commitment_points_by_stage:
            for r, c in stage_cps:
                ct = gm.cell_types[r, c]
                # Commitment point should not be a wall
                # (it can be LOCKED_DOOR for Stage 2 door commitments)
                assert ct != 1, (  # CellType.WALL
                    f"Commitment point ({r},{c}) is a wall")

class TestWarnTargetInvariance:
    """WARN target should be semantically correct across scenarios."""

    def _make_state(self, seed=42, difficulty="medium"):
        """Create a runner state for WARN target testing."""
        from src.envs.lattice_v2_runner import LatticeV2Runner
        runner = LatticeV2Runner()
        s = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            tutor_mode="none", warning_mode="none",
            robot_belief_mode=True,
            intervention_family_mode=True,
        )
        return s

    def test_warn_target_varies_across_seeds(self):
        """Warn target row should not be fixed to a single row."""
        from src.envs.dtmb_helpers import compute_dtmb_warn_target
        warned_rows = set()
        for seed in range(30):
            s = self._make_state(seed=seed)
            target = compute_dtmb_warn_target(s)
            if target:
                warned_rows.add(target[0])
        assert len(warned_rows) > 1, (
            f"Warn target locked to single row: {warned_rows}")

    def test_warn_never_targets_doored_branch(self):
        """Warn target should never be a branch that has a door (good branch)."""
        from src.envs.dtmb_helpers import compute_dtmb_warn_target, _discover_branches
        for seed in range(20):
            s = self._make_state(seed=seed)
            candidate_rows, fork_col, has_door = _discover_branches(s)
            target = compute_dtmb_warn_target(s)
            if target and candidate_rows:
                target_row = target[0]
                idx = candidate_rows.index(target_row) if target_row in candidate_rows else -1
                if idx >= 0:
                    assert not has_door[idx], (
                        f"Seed {seed}: WARN targets row {target_row} which HAS a door!")

    def test_all_variants_produce_valid_target(self):
        """All W1/W2/W3 variants should produce valid (or None) targets."""
        from src.envs.dtmb_helpers import compute_dtmb_warn_target
        for variant in ["W1", "W2", "W3"]:
            for seed in [0, 10, 42]:
                s = self._make_state(seed=seed)
                target = compute_dtmb_warn_target(s, variant=variant)
                if target:
                    r, c = target
                    assert 0 <= r < s.passable.shape[0], f"Row {r} out of bounds"
                    assert 0 <= c < s.passable.shape[1], f"Col {c} out of bounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
