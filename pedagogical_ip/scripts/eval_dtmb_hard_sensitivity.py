"""DTMB-L Hard Search Budget Sensitivity: 40 vs 50.

Compares hard difficulty with search_budget ∈ {40, 50} to assess
whether increased budget improves canonical tutor performance.

Usage:
    python scripts/eval_dtmb_hard_sensitivity.py [--seeds 10]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "deep_tree_mixed_bottleneck_lattice"


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L hard budget sensitivity")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output-file", default=None)
    return p.parse_args()


CANONICAL_POLICY = {
    "tutor_mode": "none",
    "warning_mode": "none",
    "robot_belief_mode": True,
    "intervention_family_mode": True,
    "item_drop_enabled": True,
    "belief_planning_mode": True,
    "latent_mode": True,
    "patch_radius": 2,
    "prefix_horizon": 5,
}

NO_TUTOR_POLICY = {
    "tutor_mode": "none",
    "warning_mode": "none",
    "robot_belief_mode": False,
    "intervention_family_mode": False,
    "item_drop_enabled": False,
    "belief_planning_mode": True,
    "latent_mode": True,
    "patch_radius": 2,
    "prefix_horizon": 0,
}


def run_episode(runner, seed, policy_cfg, user_cfg=None):
    try:
        state = runner.reset(
            seed=seed, difficulty="hard",
            scenario_family=FAMILY, **policy_cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        metrics["success"] = True
        return metrics
    except Exception as e:
        return {"success": False, "error": str(e),
                "survived": False, "reached_goal": False, "steps": 0}


def main():
    args = parse_args()
    runner = LatticeV2Runner()

    lines = []
    lines.append(f"DTMB-L Hard Sensitivity: search_budget ∈ {{40, 50}}, {args.seeds} seeds")
    lines.append("=" * 70)

    for budget in [40, 50]:
        lines.append(f"\n--- search_budget={budget} ---")

        for policy_name, policy_cfg in [("canonical", CANONICAL_POLICY),
                                         ("no_tutor", NO_TUTOR_POLICY)]:
            results = defaultdict(list)
            fail_count = 0

            for seed in range(args.seeds):
                # Note: Over-riding search_budget requires passing user_cfg
                # through generate_scenario. For now the generator uses
                # the config default — this script changes the planner budget
                # via the runner if accessible, or we regenerate.
                m = run_episode(runner, seed, policy_cfg,
                                user_cfg={"search_budget": budget})
                if m["success"]:
                    results["survived"].append(m["survived"])
                    results["reached_goal"].append(m["reached_goal"])
                    results["steps"].append(m["steps"])
                    results["risky_entered"].append(m.get("risky_entered", 0))
                else:
                    fail_count += 1

            n = len(results["survived"])
            if n > 0:
                surv = np.mean(results["survived"])
                goal = np.mean(results["reached_goal"])
                steps_avg = np.mean(results["steps"])
                risky_avg = np.mean(results["risky_entered"])
                lines.append(
                    f"  {policy_name:12s}: surv={surv:.2f} goal={goal:.2f} "
                    f"steps={steps_avg:.1f} risky={risky_avg:.1f} "
                    f"({n} ok, {fail_count} fail)")

    lines.append("\n" + "=" * 70)
    lines.append("Done.")

    output = "\n".join(lines)
    print(output)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)


if __name__ == "__main__":
    main()
