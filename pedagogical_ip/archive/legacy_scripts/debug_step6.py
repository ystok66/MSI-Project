"""Debug: check n_updates and cost at step 6."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.planner_astar import cell_cost_v2_latent

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    tutor_mode="none", robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

for i in range(6):
    runner.observe(s)
    runner.apply_tutor(s)
    
    from src.agents.route_necessity import compute_route_necessity
    unvisited = set()
    for r in range(s.gridmap.height):
        for c in range(s.gridmap.width):
            if s.passable[r, c] and not s.feature_belief.memory[r, c].ever_traversed:
                unvisited.add((r, c))
    n = compute_route_necessity(s.agent_pos, s.goal, s.passable, s.t, s.t_max, unvisited)
    
    nu = s.latent_predictor.n_updates
    lf = min(1.0, nu / 10.0)
    cost_right = cell_cost_v2_latent(
        s.agent_pos[0], s.agent_pos[1]+1,
        s.feature_belief.mean, s.latent_predictor, s.passable,
        feature_belief_var=s.feature_belief.var, route_necessity=n) if s.agent_pos[1]+1 < 20 else 999
    cost_down = cell_cost_v2_latent(
        s.agent_pos[0]+1, s.agent_pos[1],
        s.feature_belief.mean, s.latent_predictor, s.passable,
        feature_belief_var=s.feature_belief.var, route_necessity=n) if s.agent_pos[0]+1 < 5 else 999
    print(f"Step {i+1}: pos={s.agent_pos}, n_up={nu}, lf={lf:.2f}, "
          f"n={n:.2f}, cost_right={cost_right:.3f}, cost_down={cost_down:.3f}")
    
    runner.plan_and_move(s)
print(f"Final pos: {s.agent_pos}")
