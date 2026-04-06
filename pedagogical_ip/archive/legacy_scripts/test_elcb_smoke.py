"""ELCB smoke test + sanity checks."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.scenario_families import generate_scenario
from src.envs.lattice_v2 import CellType

print("=== ELCB Smoke Test ===")
for diff in ["easy", "medium", "hard"]:
    for seed in [0, 1, 42]:
        gm, cfg, meta, sc = generate_scenario("elcb", seed, diff, latent_mode=True)
        # SC4: Passability audit
        for cell in sc.branch_a_cells + sc.branch_b_cells:
            r, c = cell
            assert gm.cell_types[r, c] != CellType.WALL, f"WALL in branch at {cell}"

        # Length equality
        assert len(sc.branch_a_cells) == len(sc.branch_b_cells), "Branch lengths differ!"
        print(f"{diff} s{seed}: {gm.height}x{gm.width} t_max={cfg.max_steps} "
              f"safe_branch={sc.oracle_safe_branch_id} "
              f"branch_len={sc.branch_len} "
              f"safe={meta.shortest_safe} any={meta.shortest_any}")

# Run one episode
from src.envs.lattice_v2_runner import LatticeV2Runner
runner = LatticeV2Runner()
s = runner.reset(seed=42, scenario_family="elcb", latent_mode=True,
                 difficulty="medium", tutor_mode="none", warning_mode="none")
while not s.done:
    runner.step(s)
m = runner.get_metrics(s)
print(f"\nEpisode: steps={m['steps']} goal={m['reached_goal']} survived={m['survived']}")
print("All sanity checks passed.")
