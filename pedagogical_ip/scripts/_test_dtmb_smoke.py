"""Quick verify: hard route count after [4,2] change."""
import sys
sys.path.insert(0, ".")

from src.envs.scenario_families import generate_scenario
from src.envs.dtmb_lattice import print_dtmb_ascii

# Verify hard across multiple seeds
print("=== HARD route count verification ===")
for seed in [0, 1, 42, 100, 200, 500, 777, 999]:
    gm, cfg, meta, sc = generate_scenario(
        "deep_tree_mixed_bottleneck_lattice",
        seed=seed, difficulty="hard", latent_mode=False)
    print(f"  seed={seed:4d}: routes={meta.route_count}, "
          f"doors={len(meta.all_door_positions)}, "
          f"belt={len(meta.belt_cells_by_stage[2])}, "
          f"s_any={meta.shortest_any}, s_safe={meta.shortest_safe}, "
          f"gt={meta.dominant_bottleneck_gt_by_stage}")

# Also check easy/medium still fine
for diff in ["easy", "medium"]:
    gm, cfg, meta, sc = generate_scenario(
        "deep_tree_mixed_bottleneck_lattice",
        seed=42, difficulty=diff, latent_mode=False)
    print(f"\n  {diff}: routes={meta.route_count}, grid={gm.height}x{gm.width}")

# Print ASCII for hard seed=42
gm, cfg, meta, sc = generate_scenario(
    "deep_tree_mixed_bottleneck_lattice",
    seed=42, difficulty="hard", latent_mode=False)
print(f"\n=== HARD seed=42 ASCII ({gm.height}x{gm.width}) ===")
print(print_dtmb_ascii(gm, meta))
