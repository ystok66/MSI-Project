"""Check connectivity: what cells on row 1 are passable?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.scenario_families import generate_deadline_gate
from src.envs.map_generator import CellType

gm, cfg, meta, sc = generate_deadline_gate(seed=0, difficulty="easy", latent_mode=True)
print(f"Map: {gm.height}x{gm.width}")
print(f"Gate: {meta.all_door_positions}")

TYPE_NAMES = {0: "NORM", 1: "WALL", 2: "GOAL", 3: "RISK", 4: "LOCK", 5: "TARG"}

for r in range(gm.height):
    row_str = f"Row {r}: "
    for c in range(gm.width):
        ct_val = int(gm.cell_types[r, c])
        name = TYPE_NAMES.get(ct_val, f"?{ct_val}")
        row_str += f"{name:5s}"
    print(row_str)

print(f"\nRow 1 passability:")
for c in range(gm.width):
    ct_val = int(gm.cell_types[1, c])
    passable = ct_val not in (1, 4)  # not WALL or LOCKED_DOOR
    print(f"  (1,{c}): type={TYPE_NAMES.get(ct_val, ct_val)} passable={passable}")
