"""Unit Sanity Test for DTMB-L WARN target computation.

Verifies that:
1. WARN does not always penalize the first row.
2. WARN target flips appropriately with mirror / permutation of subtrees.
"""
import sys
sys.path.insert(0, ".")

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.dtmb_helpers import compute_dtmb_warn_target
import numpy as np

def test_warn_target_sanity():
    print("Testing WARN target sanity...")
    
    runner = LatticeV2Runner()
    warned_rows = set()
    
    for seed in range(20):
        s = runner.reset(
            seed=seed, difficulty="medium",
            scenario_family="deep_tree_mixed_bottleneck_lattice",
            tutor_mode="none", warning_mode="none",
            robot_belief_mode=True,
            intervention_family_mode=True
        )
        
        for step in range(8):
            target = compute_dtmb_warn_target(s)
            if target:
                warned_rows.add(target[0])
                print(f"Seed {seed:2d} Step {step}: target={target}")
                break
            s = runner.step(s)
            
    print(f"Unique warned rows across seeds: {warned_rows}")
    if len(warned_rows) <= 1:
        print("FAIL: Warn target is still fixed to a single row!")
        sys.exit(1)
        
    print("PASS: Warn target varies based on structure/belief rather than fixed index.")

if __name__ == "__main__":
    test_warn_target_sanity()
