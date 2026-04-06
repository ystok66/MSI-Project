"""DTMB-L Smoke Evaluation — multi-seed multi-difficulty validation.

Generates N_SEEDS × 3 difficulties and prints aggregate statistics.
Verifies structure, route counts, and bottleneck distributions.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from collections import Counter
from src.envs.scenario_families import generate_scenario
from src.envs.dtmb_lattice import _bfs_shortest, print_dtmb_ascii

FAMILY = "deep_tree_mixed_bottleneck_lattice"
N_SEEDS = 20


def main():
    print(f"DTMB-L Smoke Evaluation — {N_SEEDS} seeds × 3 difficulties")
    print("=" * 70)

    for diff in ["easy", "medium", "hard"]:
        route_counts = []
        shortest_any_list = []
        shortest_safe_list = []
        door_counts = []
        belt_counts = []
        bottleneck_s2 = Counter()
        pass_count = 0
        fail_count = 0

        for seed in range(N_SEEDS):
            try:
                gm, cfg, meta, sc = generate_scenario(
                    FAMILY, seed=seed, difficulty=diff, latent_mode=False)

                # Basic validation
                assert meta.decision_stages == 3
                assert meta.shortest_any < 999
                assert meta.route_count >= 4  # smoke threshold
                assert len(meta.dominant_bottleneck_gt_by_stage) == 3

                route_counts.append(meta.route_count)
                shortest_any_list.append(meta.shortest_any)
                shortest_safe_list.append(meta.shortest_safe)
                door_counts.append(len(meta.all_door_positions))
                belt_counts.append(len(meta.belt_cells_by_stage[2]))
                bottleneck_s2[meta.dominant_bottleneck_gt_by_stage[1]] += 1
                pass_count += 1

            except Exception as e:
                fail_count += 1
                print(f"  FAIL seed={seed}: {e}")

        print(f"\n--- {diff.upper()} ({pass_count}/{pass_count + fail_count} passed) ---")
        if route_counts:
            print(f"  route_count:   min={min(route_counts)} avg={np.mean(route_counts):.1f} max={max(route_counts)}")
            print(f"  shortest_any:  min={min(shortest_any_list)} avg={np.mean(shortest_any_list):.1f} max={max(shortest_any_list)}")
            print(f"  shortest_safe: min={min(shortest_safe_list)} avg={np.mean(shortest_safe_list):.1f} max={max(shortest_safe_list)}")
            print(f"  doors:         {Counter(door_counts)}")
            print(f"  belt_cells:    min={min(belt_counts)} avg={np.mean(belt_counts):.1f} max={max(belt_counts)}")
            print(f"  S2 bottleneck: {dict(bottleneck_s2)}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
