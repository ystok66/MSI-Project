import sys
sys.path.insert(0, ".")
from src.envs.lattice_v2_runner import LatticeV2Runner

r = LatticeV2Runner()
s = r.reset(
    seed=0, difficulty='medium', scenario_family='deep_tree_mixed_bottleneck_lattice', 
    robot_belief_mode=True, intervention_family_mode=True
)
for _ in range(20):
    s = r.step(s)
print("Finished!")
