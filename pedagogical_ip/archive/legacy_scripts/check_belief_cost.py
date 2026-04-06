"""Check: how does belief_cost treat shortcut cells after UNLOCK?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    robot_belief_mode=True, intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

gate = s.meta.all_door_positions[0]
print(f"Gate: {gate}, passable: {s.passable[gate[0], gate[1]]}")
print(f"t_max: {s.t_max}")

# Step once to trigger UNLOCK
runner.step(s)
print(f"\nAfter step 1: gate passable: {s.passable[gate[0], gate[1]]}")

# Print belief_cost for shortcut cells (row 1)
print("\nRow 1 (shortcut) belief_cost after UNLOCK:")
for c in range(s.gridmap.width):
    bc = s.belief_cost[1, c]
    pa = s.passable[1, c]
    ct = int(s.gridmap.cell_types[1, c])
    tr = s.gridmap.true_risk[1, c]
    print(f"  (1,{c}): belief_cost={bc:6.2f}  passable={pa}  type={ct}  true_risk={tr:.3f}")

print("\nRow 2 (corridor) belief_cost:")
for c in range(s.gridmap.width):
    bc = s.belief_cost[2, c]
    pa = s.passable[2, c]
    ct = int(s.gridmap.cell_types[2, c])
    print(f"  (2,{c}): belief_cost={bc:6.2f}  passable={pa}  type={ct}")

print("\nRow 3 (safe long path) belief_cost:")
for c in range(s.gridmap.width):
    bc = s.belief_cost[3, c]
    pa = s.passable[3, c]
    ct = int(s.gridmap.cell_types[3, c])
    print(f"  (3,{c}): belief_cost={bc:6.2f}  passable={pa}  type={ct}")
