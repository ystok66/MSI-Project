"""Quick check: what risk does latent mode produce for hazard_belt cells?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.scenario_families import generate_hazard_belt

gm, cfg, meta, sc = generate_hazard_belt(seed=0, difficulty="medium", latent_mode=True)
print("=== Latent mode risk map ===")
belt = meta.segments[1]
s0 = meta.segments[0]

print(f"Seg[0] risky cells (non-belt):")
for r, c in s0.risky_cells:
    print(f"  ({r},{c}): risk={gm.true_risk[r,c]:.4f}, type={int(gm.cell_types[r,c])}")

print(f"\nBelt risky cells:")
for r, c in belt.risky_cells:
    print(f"  ({r},{c}): risk={gm.true_risk[r,c]:.4f}, type={int(gm.cell_types[r,c])}")

print(f"\nCorridor (row 2):")
for c in range(1, gm.width - 1):
    if int(gm.cell_types[2, c]) != 1:
        print(f"  (2,{c}): risk={gm.true_risk[2,c]:.4f}, type={int(gm.cell_types[2,c])}")

print(f"\nSeg[0] safe cells (first 5):")
for r, c in s0.safe_cells[:5]:
    print(f"  ({r},{c}): risk={gm.true_risk[r,c]:.4f}, type={int(gm.cell_types[r,c])}")
