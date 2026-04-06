"""Debug: what does the planner return at (1,6)?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.planner_astar import plan_next_action_v2

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    tutor_mode="none", robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

# Walk to step 7 to reach (1,6)
for i in range(7):
    runner.step(s)
    print(f"Step {i+1}: pos={s.agent_pos}")

print(f"\nNow at {s.agent_pos}, t={s.t}")

# Compute necessity manually
from src.agents.route_necessity import compute_route_necessity
unvisited = set()
for r in range(s.gridmap.height):
    for c in range(s.gridmap.width):
        if s.passable[r, c] and not s.feature_belief.memory[r, c].ever_traversed:
            unvisited.add((r, c))
n = compute_route_necessity(s.agent_pos, s.goal, s.passable, s.t, s.t_max, unvisited)
print(f"Necessity: {n}")
print(f"n_updates: {s.latent_predictor.n_updates}")
print(f"Unvisited: {len(unvisited)} cells")

# Manually plan from (1,6)
action, next_pos, path = plan_next_action_v2(
    s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
    s.risk_head, budget=30, passable_mask=s.passable,
    latent_predictor=s.latent_predictor,
    feature_belief_var=s.feature_belief.var,
    route_necessity=n,
)
print(f"Manual plan from {s.agent_pos}: action={action}, next={next_pos}")
print(f"Path: {path[:10]}... (len={len(path)})")
