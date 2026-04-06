import sys
sys.path.insert(0, ".")
from src.envs.scenario_families import generate_scenario
from src.envs.dtmb_lattice import print_dtmb_ascii
gm, cfg, meta, sc = generate_scenario('deep_tree_mixed_bottleneck_lattice', seed=42, difficulty='medium', latent_mode=False)
print("Medium")
print(print_dtmb_ascii(gm, meta))
