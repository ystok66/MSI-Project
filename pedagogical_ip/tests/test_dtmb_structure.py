"""Structural sanity tests for DTMB-L (Deep Tree Mixed-Bottleneck Lattice).

Verifies:
  - Return type correctness
  - Metadata contract compliance
  - Route count targets
  - BFS reachability
  - Decision/commit/reveal consistency
  - Multi-seed determinism and variety
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, ".")

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType, GridMap
from src.envs.map_families import FamilyConfig
from src.envs.lattice_v2 import LatticeV2Meta
from src.envs.dtmb_lattice import _bfs_shortest, _bfs_reachable

FAMILY = "deep_tree_mixed_bottleneck_lattice"


# ── Helpers ──────────────────────────────────────────────────────────

def _gen(seed=42, difficulty="easy", latent_mode=False):
    return generate_scenario(FAMILY, seed=seed, difficulty=difficulty,
                             latent_mode=latent_mode)


# ── Type and structure tests ─────────────────────────────────────────

class TestReturnTypes:
    def test_returns_tuple_of_four(self):
        result = _gen()
        assert isinstance(result, tuple) and len(result) == 4

    def test_gridmap_type(self):
        gm, _, _, _ = _gen()
        assert isinstance(gm, GridMap)

    def test_familycfg_type(self):
        _, cfg, _, _ = _gen()
        assert isinstance(cfg, FamilyConfig)

    def test_meta_type(self):
        _, _, meta, _ = _gen()
        assert isinstance(meta, LatticeV2Meta)

    def test_family_name(self):
        _, _, _, sc = _gen()
        assert sc.family_name == FAMILY


# ── Metadata contract ────────────────────────────────────────────────

class TestMetadataContract:
    @pytest.fixture(params=["easy", "medium", "hard"])
    def scenario(self, request):
        return _gen(seed=42, difficulty=request.param)

    def test_decision_stages_is_3(self, scenario):
        _, _, meta, _ = scenario
        assert meta.decision_stages == 3

    def test_commitment_points_stage1_nonempty(self, scenario):
        _, _, meta, _ = scenario
        assert len(meta.commitment_points_by_stage[0]) > 0

    def test_reveal_events_stage1_nonempty(self, scenario):
        _, _, meta, _ = scenario
        assert len(meta.reveal_events_by_stage[0]) > 0

    def test_gt_bottlenecks_has_3_entries(self, scenario):
        _, _, meta, _ = scenario
        assert len(meta.dominant_bottleneck_gt_by_stage) == 3

    def test_gt_bottleneck_stage1_is_epistemic(self, scenario):
        _, _, meta, _ = scenario
        assert meta.dominant_bottleneck_gt_by_stage[0] == "epistemic"

    def test_gt_bottleneck_stage3_is_outcome(self, scenario):
        _, _, meta, _ = scenario
        assert meta.dominant_bottleneck_gt_by_stage[2] == "outcome"

    def test_belt_cells_stage3_nonempty(self, scenario):
        _, _, meta, _ = scenario
        assert len(meta.belt_cells_by_stage[2]) > 0

    def test_recommended_levers(self, scenario):
        _, _, meta, _ = scenario
        assert meta.recommended_primary_lever_by_stage == [
            "WAIT/WARN", "UNLOCK", "ITEM_DROP"]


# ── Route count targets ─────────────────────────────────────────────

class TestRouteCounts:
    """Smoke thresholds: easy>=4, medium>=4, hard>=4.
    Final acceptance: easy>=6, medium>=6, hard>=8 (tested across seeds).
    """
    def test_easy_route_count_smoke(self):
        _, _, meta, _ = _gen(seed=42, difficulty="easy")
        assert meta.route_count >= 4, f"easy route_count={meta.route_count}"

    def test_medium_route_count_smoke(self):
        _, _, meta, _ = _gen(seed=42, difficulty="medium")
        assert meta.route_count >= 4, f"medium route_count={meta.route_count}"

    def test_hard_route_count_smoke(self):
        _, _, meta, _ = _gen(seed=42, difficulty="hard")
        assert meta.route_count >= 8, f"hard route_count={meta.route_count}"

    def test_easy_route_count_final(self):
        """Final acceptance: easy must reach >=6 across multiple seeds."""
        for seed in [42, 100, 200, 300, 999]:
            _, _, meta, _ = _gen(seed=seed, difficulty="easy")
            assert meta.route_count >= 6, (
                f"easy seed={seed} route_count={meta.route_count} < 6")

    def test_medium_route_count_final(self):
        """Final acceptance: medium must reach >=6 across multiple seeds."""
        for seed in [42, 100, 200, 300, 999]:
            _, _, meta, _ = _gen(seed=seed, difficulty="medium")
            assert meta.route_count >= 6, (
                f"medium seed={seed} route_count={meta.route_count} < 6")

    def test_hard_route_count_final(self):
        """Final acceptance: hard must reach >=8 across multiple seeds."""
        for seed in [42, 100, 200, 300, 999]:
            _, _, meta, _ = _gen(seed=seed, difficulty="hard")
            assert meta.route_count >= 8, (
                f"hard seed={seed} route_count={meta.route_count} < 8")


# ── BFS reachability ─────────────────────────────────────────────────

class TestReachability:
    @pytest.fixture(params=["easy", "medium", "hard"])
    def scenario(self, request):
        return _gen(seed=42, difficulty=request.param)

    def test_target_reachable(self, scenario):
        gm, _, meta, _ = scenario
        dist = _bfs_shortest(gm.cell_types, gm.agent_start, gm.target_pos)
        assert dist < 999, "Target not reachable from start"

    def test_shortest_any_finite(self, scenario):
        _, _, meta, _ = scenario
        assert meta.shortest_any < 999

    def test_shortest_safe_finite(self, scenario):
        _, _, meta, _ = scenario
        assert meta.shortest_safe < 999

    def test_shortest_any_leq_safe(self, scenario):
        _, _, meta, _ = scenario
        assert meta.shortest_any <= meta.shortest_safe


# ── Grid dimensions ──────────────────────────────────────────────────

class TestGridDimensions:
    def test_easy_dimensions(self):
        gm, _, _, _ = _gen(difficulty="easy")
        assert gm.height == 13 and gm.width == 35

    def test_medium_dimensions(self):
        gm, _, _, _ = _gen(difficulty="medium")
        assert gm.height == 15 and gm.width == 45

    def test_hard_dimensions(self):
        gm, _, _, _ = _gen(difficulty="hard")
        assert gm.height == 17 and gm.width == 60


# ── Search budget contract ───────────────────────────────────────────

class TestSearchBudget:
    def test_easy_budget(self):
        _, cfg, _, _ = _gen(difficulty="easy")
        assert cfg.search_budget == 30

    def test_medium_budget(self):
        _, cfg, _, _ = _gen(difficulty="medium")
        assert cfg.search_budget == 35

    def test_hard_budget(self):
        _, cfg, _, _ = _gen(difficulty="hard")
        assert cfg.search_budget == 40


# ── Determinism and variety ──────────────────────────────────────────

class TestDeterminismAndVariety:
    def test_same_seed_same_output(self):
        gm1, _, m1, _ = _gen(seed=123)
        gm2, _, m2, _ = _gen(seed=123)
        assert np.array_equal(gm1.cell_types, gm2.cell_types)
        assert m1.shortest_any == m2.shortest_any

    def test_different_seeds_different_output(self):
        gm1, _, m1, _ = _gen(seed=42)
        gm2, _, m2, _ = _gen(seed=99)
        # Features vary by seed (topology is config-determined but features are RNG)
        assert not np.array_equal(m1.cell_features, m2.cell_features)


# ── Door and belt presence (medium/hard) ─────────────────────────────

class TestInterventionPresence:
    def test_medium_has_doors_across_seeds(self):
        """At least some seeds should produce doors for medium."""
        has_door = False
        for seed in range(42, 62):
            _, _, meta, _ = _gen(seed=seed, difficulty="medium")
            if len(meta.all_door_positions) > 0:
                has_door = True
                break
        assert has_door, "No seed in [42,62) produced doors for medium"

    def test_medium_has_belt(self):
        _, _, meta, _ = _gen(seed=42, difficulty="medium")
        assert len(meta.belt_cells_by_stage[2]) > 0

    def test_hard_has_belt(self):
        _, _, meta, _ = _gen(seed=42, difficulty="hard")
        assert len(meta.belt_cells_by_stage[2]) > 0


# ── Latent mode ──────────────────────────────────────────────────────

class TestLatentMode:
    def test_latent_mode_produces_world_weights(self):
        _, _, meta, _ = _gen(seed=42, difficulty="easy", latent_mode=True)
        assert meta.world_weights is not None
        assert meta.latent_mode is True

    def test_non_latent_mode_no_world_weights(self):
        _, _, meta, _ = _gen(seed=42, difficulty="easy", latent_mode=False)
        assert meta.world_weights is None
        assert meta.latent_mode is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
