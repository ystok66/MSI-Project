"""Check: what does latent_predictor.predict_risk return for shortcut features?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.scenario_families import generate_deadline_gate
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, scenario_family="deadline_gate",
    latent_mode=True, difficulty="easy",
    robot_belief_mode=True, intervention_family_mode=True,
    item_drop_enabled=False, prefix_horizon=5,
    allowed_interventions=frozenset({"UNLOCK"}))

# After step 1 (UNLOCK fires), check what latent_predictor predicts
runner.step(s)  # observe + tutor(UNLOCK) + plan_and_move

lp = s.latent_predictor
print(f"latent_predictor type: {type(lp).__name__}")
print(f"\nRow 1 shortcut - latent predictions:")
for c in range(s.gridmap.width):
    if int(s.gridmap.cell_types[1, c]) not in (1, 4):  # not wall/locked
        x = s.feature_belief.get_mean(1, c)
        pred_cost = lp.predict_cost(x)
        pred_risk = lp.predict_risk(x)
        true_risk = s.gridmap.true_risk[1, c]
        print(f"  (1,{c}): pred_risk={pred_risk:.3f} true_risk={true_risk:.3f} "
              f"pred_cost={pred_cost:.3f}")

print(f"\nRow 2 corridor - latent predictions:")
for c in [1, 2, 6, 7, 12, 13, 18, 19]:
    if c < s.gridmap.width and int(s.gridmap.cell_types[2, c]) not in (1, 4):
        x = s.feature_belief.get_mean(2, c)
        pred_risk = lp.predict_risk(x)
        true_risk = s.gridmap.true_risk[2, c]
        print(f"  (2,{c}): pred_risk={pred_risk:.3f} true_risk={true_risk:.3f}")
