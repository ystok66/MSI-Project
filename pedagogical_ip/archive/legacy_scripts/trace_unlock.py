"""Step-by-step trace: where does the agent go on unlock_only?"""
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
print(f"Gate: {gate}, t_max: {s.t_max}, map: {s.gridmap.height}x{s.gridmap.width}")

for step_i in range(50):
    if s.done:
        break
    gate_open = s.passable[gate[0], gate[1]]
    print(f"  t={step_i+1:3d} pos={s.agent_pos} gate={'OPEN' if gate_open else 'CLOSED'}")
    runner.step(s)

m = runner.get_metrics(s)
result = "GOAL" if m["reached_goal"] else ("DEATH" if not m["survived"] else "TIMEOUT")
print(f"\nResult: {result} after {m['steps']} steps")
print(f"Goal: {s.goal}")
