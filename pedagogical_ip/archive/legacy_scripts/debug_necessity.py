"""Debug: check if necessity is actually reaching the planner."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.route_necessity import compute_route_necessity

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    tutor_mode="none", robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

# Step 1: observe
runner.observe(s)
runner.apply_tutor(s)

# Check necessity manually
unvisited = set()
for r in range(s.gridmap.height):
    for c in range(s.gridmap.width):
        if s.passable[r, c] and not s.feature_belief.memory[r, c].ever_traversed:
            unvisited.add((r, c))

n = compute_route_necessity(s.agent_pos, s.goal, s.passable, s.t, s.t_max, unvisited)
print(f"Necessity: {n:.4f}")
print(f"n_updates: {s.latent_predictor.n_updates}")
print(f"Agent pos: {s.agent_pos}")
print(f"Unvisited cells: {len(unvisited)}")
print(f"Passable cells: {int(s.passable.sum())}")

# Check what cell_cost returns for a shortcut cell
from src.agents.planner_astar import cell_cost_v2_latent
cost_with_n = cell_cost_v2_latent(
    1, 7, s.feature_belief.mean, s.latent_predictor, s.passable,
    feature_belief_var=s.feature_belief.var, route_necessity=n,
)
cost_without_n = cell_cost_v2_latent(
    1, 7, s.feature_belief.mean, s.latent_predictor, s.passable,
    feature_belief_var=s.feature_belief.var, route_necessity=0.0,
)
print(f"\nCell (1,7) cost WITH necessity={n:.2f}: {cost_with_n:.4f}")
print(f"Cell (1,7) cost WITHOUT necessity:    {cost_without_n:.4f}")

# Check a safe-path corridor cell
cost_safe = cell_cost_v2_latent(
    2, 6, s.feature_belief.mean, s.latent_predictor, s.passable,
    feature_belief_var=s.feature_belief.var, route_necessity=n,
)
print(f"Cell (2,6) cost WITH necessity={n:.2f}: {cost_safe:.4f}")
