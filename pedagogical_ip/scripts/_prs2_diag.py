"""Diagnose PRS-2 session_shared: why TBSR=0 and updates=0."""
import sys; sys.path.insert(0, ".")
import numpy as np

from src.agents.cost_risk_model import generate_world_weights, WorldWeights
from src.envs.lattice_v2_runner import LatticeV2Runner

# Generate session-level WorldWeights
ww_rng = np.random.default_rng(42 * 7 + 3)
ww = generate_world_weights(ww_rng, d=4)
print("=== Session WorldWeights ===")
print(f"  w_cost = {ww.w_cost}")
print(f"  b_cost = {ww.b_cost}")
print(f"  w_risk = {ww.w_risk}")
print(f"  b_risk = {ww.b_risk}")

# Test with a GTET episode
runner = LatticeV2Runner()
from src.agents.cost_risk_model import LatentCostRiskHead

lp = LatentCostRiskHead(d=4)

# Without override
s1 = runner.reset(
    seed=42, difficulty="medium",
    scenario_family="goal_preference_temptation_entanglement_lattice",
    latent_mode=True, patch_radius=2, prefix_horizon=5,
    belief_planning_mode=True,
    predictor_mode="P4", factor_mode="FULL",
    robot_belief_mode=True, intervention_family_mode=True,
    item_drop_enabled=True,
    latent_predictor=lp,
)
print(f"\n=== WITHOUT override ===")
print(f"  Grid: {s1.gridmap.height}x{s1.gridmap.width}")
print(f"  t_max: {s1.t_max}")
print(f"  Cost range: [{s1.gridmap.true_cost.min():.3f}, {s1.gridmap.true_cost.max():.3f}]")
print(f"  Risk range: [{s1.gridmap.true_risk.min():.3f}, {s1.gridmap.true_risk.max():.3f}]")
print(f"  Mean cost (non-wall): {s1.gridmap.true_cost[s1.gridmap.true_cost < 50].mean():.3f}")
print(f"  Mean risk (non-wall): {s1.gridmap.true_risk[s1.gridmap.true_risk < 50].mean():.3f}")
print(f"  Meta world_weights: {s1.meta.world_weights}")

# Run it
while not s1.done:
    s1 = runner.step(s1)
m1 = runner.get_metrics(s1)
print(f"  survived={m1['survived']} goal={m1['reached_goal']} steps={m1['steps']}")
print(f"  predictor updates: {lp.n_updates}")

# With override
lp2 = LatentCostRiskHead(d=4)
user_cfg = {"world_weights_override": ww}
s2 = runner.reset(
    seed=42, difficulty="medium",
    scenario_family="goal_preference_temptation_entanglement_lattice",
    latent_mode=True, patch_radius=2, prefix_horizon=5,
    belief_planning_mode=True, user_cfg=user_cfg,
    predictor_mode="P4", factor_mode="FULL",
    robot_belief_mode=True, intervention_family_mode=True,
    item_drop_enabled=True,
    latent_predictor=lp2,
)
print(f"\n=== WITH override ===")
print(f"  Grid: {s2.gridmap.height}x{s2.gridmap.width}")
print(f"  t_max: {s2.t_max}")
print(f"  Cost range: [{s2.gridmap.true_cost.min():.3f}, {s2.gridmap.true_cost.max():.3f}]")
print(f"  Risk range: [{s2.gridmap.true_risk.min():.3f}, {s2.gridmap.true_risk.max():.3f}]")
print(f"  Mean cost (non-wall): {s2.gridmap.true_cost[s2.gridmap.true_cost < 50].mean():.3f}")
print(f"  Mean risk (non-wall): {s2.gridmap.true_risk[s2.gridmap.true_risk < 50].mean():.3f}")
print(f"  Meta world_weights: {s2.meta.world_weights}")

# Sample features
H, W = s2.gridmap.height, s2.gridmap.width
for r in range(H):
    for c in range(W):
        if s2.gridmap.true_cost[r, c] < 50:
            z = s2.meta.cell_features[r, c]
            tc = ww.true_cost(z)
            tr = ww.true_risk(z)
            if r <= 2 and c <= 5:
                print(f"  Cell ({r},{c}): z={z} → cost={tc:.3f}, risk={tr:.3f}")
            break

# Run it
while not s2.done:
    s2 = runner.step(s2)
m2 = runner.get_metrics(s2)
print(f"  survived={m2['survived']} goal={m2['reached_goal']} steps={m2['steps']}")
print(f"  predictor updates: {lp2.n_updates}")
