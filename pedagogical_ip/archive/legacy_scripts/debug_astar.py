"""Deep debug: manually run plan_next_action_v2 and trace what A* does."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.planner_astar import plan_next_action_v2, cell_cost_v2_latent, _astar_core

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    tutor_mode="none", robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

# Do 2 steps to get UNLOCK fired and agent onto shortcut
runner.observe(s)
runner.apply_tutor(s)  # UNLOCK fires here
# Agent should now be at (2,1)

print(f"Agent at: {s.agent_pos}, goal: {s.goal}")
print(f"Gate passable: {s.passable[1,2]}")

# Compute necessity
from src.agents.route_necessity import compute_route_necessity
unvisited = set()
for r in range(s.gridmap.height):
    for c in range(s.gridmap.width):
        if s.passable[r, c] and not s.feature_belief.memory[r, c].ever_traversed:
            unvisited.add((r, c))
n = compute_route_necessity(s.agent_pos, s.goal, s.passable, s.t, s.t_max, unvisited)
print(f"Route necessity: {n}")

# Manually call plan with necessity
print("\nManual plan_next_action_v2:")
action, next_pos, path = plan_next_action_v2(
    s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
    s.risk_head, budget=30, passable_mask=s.passable,
    latent_predictor=s.latent_predictor,
    feature_belief_var=s.feature_belief.var,
    route_necessity=n,
)
print(f"  Action: {action}, next_pos: {next_pos}")
print(f"  Path: {path[:10]}... (len={len(path)})")

# Now simulate the full plan_from_belief with debug
# Try manually calling A* with an instrumented cost_fn
H, W = s.belief_cost.shape
passable = s.passable
call_count = [0]
def debug_cost_fn(r, c):
    cost = cell_cost_v2_latent(
        r, c, s.feature_belief.mean, s.latent_predictor, passable,
        feature_belief_var=s.feature_belief.var, route_necessity=n,
    )
    call_count[0] += 1
    if call_count[0] <= 30:
        print(f"  A* eval ({r},{c}): cost={cost:.3f}")
    return cost

path2 = _astar_core(s.agent_pos, s.goal, debug_cost_fn, H, W, 30, s.passable)
print(f"\nDebug A* path: {path2[:10]}... (len={len(path2)})")
