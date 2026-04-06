"""DTMB-L Parameter Sweep — sensitivity analysis entry point.

Sweeps over:
  - search_budget: {30, 35, 40, 50}
  - belt_risk: {0.2, 0.3, 0.45, 0.6}
  - mid_door_fraction: {0.0, 0.15, 0.25, 0.35, 0.50}
  - cue_reliability: {0.45, 0.65, 0.85}

Scaffolding-only.

Usage:
    python scripts/sweep_dtmb.py [--seeds 10] [--param search_budget]
"""
import sys
sys.path.insert(0, ".")

import argparse
from src.envs.scenario_families import generate_scenario

FAMILY = "deep_tree_mixed_bottleneck_lattice"

SWEEP_PARAMS = {
    "search_budget": [30, 35, 40, 50],
    "belt_risk": [0.20, 0.30, 0.45, 0.60],
    "mid_door_fraction": [0.0, 0.15, 0.25, 0.35, 0.50],
    "stage1_cue_reliability": [0.45, 0.65, 0.85],
}


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L parameter sweep")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--param", choices=list(SWEEP_PARAMS.keys()),
                   default="search_budget")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--output-dir", default="results/dtmb_sweep")
    return p.parse_args()


def main():
    args = parse_args()
    values = SWEEP_PARAMS[args.param]
    print(f"DTMB-L Sweep: {args.param} ∈ {values}")
    print(f"  {args.seeds} seeds × {len(values)} values, difficulty={args.difficulty}")
    print("STATUS: scaffolding only — episode runner not yet connected")

    for val in values:
        for seed in range(args.seeds):
            user_cfg = {args.param: val}
            gm, cfg, meta, sc = generate_scenario(
                FAMILY, seed=seed, difficulty=args.difficulty,
                latent_mode=True, user_cfg=user_cfg)
            # TODO: run_episode and collect metrics

    print("Scaffolding complete. Connect episode runner to proceed.")


if __name__ == "__main__":
    main()
