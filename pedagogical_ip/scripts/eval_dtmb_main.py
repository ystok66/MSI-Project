"""DTMB-L Main Evaluation — oracle vs no-tutor vs canonical tutor.

Runs full episodes through LatticeV2Runner with three tutor conditions:
  - canonical:  robot_belief_mode + intervention_family_mode + item_drop
  - no_tutor:   no interventions (baseline)
  - oracle:     always_close mode (oracle gate management)

Outputs per-seed metrics and aggregate statistics.

Usage:
    python scripts/eval_dtmb_main.py [--seeds 20] [--difficulty medium]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "deep_tree_mixed_bottleneck_lattice"


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L main evaluation")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--difficulty", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--output-file", default=None)
    return p.parse_args()


POLICIES = {
    "canonical": {
        "tutor_mode": "none",            # no legacy gate tutor
        "warning_mode": "none",           # warnings handled by intervention policy
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
    },
    "no_tutor": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": False,
        "intervention_family_mode": False,
        "item_drop_enabled": False,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 0,
    },
    "oracle": {
        "tutor_mode": "dtmb_oracle",
        "warning_mode": "none",
        "robot_belief_mode": False,
        "intervention_family_mode": False,
        "item_drop_enabled": False,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 0,
    },
}


def run_episode(runner: LatticeV2Runner,
                seed: int, difficulty: str,
                policy_cfg: dict) -> dict:
    """Run a single episode and return metrics."""
    try:
        state = runner.reset(
            seed=seed,
            difficulty=difficulty,
            scenario_family=FAMILY,
            **policy_cfg,
        )
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        metrics["success"] = True
        return metrics
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "survived": False,
            "reached_goal": False,
            "steps": 0,
        }


def main():
    args = parse_args()
    runner = LatticeV2Runner()

    lines = []
    lines.append(f"DTMB-L Main Evaluation: {args.seeds} seeds × {args.difficulty}")
    lines.append("=" * 70)

    for diff in args.difficulty:
        lines.append(f"\n--- {diff.upper()} ---")
        all_results = {}

        for policy_name, policy_cfg in POLICIES.items():
            results = defaultdict(list)
            fail_count = 0

            for seed in range(args.seeds):
                m = run_episode(runner, seed, diff, policy_cfg)
                if m["success"]:
                    results["survived"].append(m["survived"])
                    results["reached_goal"].append(m["reached_goal"])
                    results["steps"].append(m["steps"])
                    results["risky_entered"].append(m.get("risky_entered", 0))
                    results["warn_count"].append(m.get("warn_count", 0))
                    results["unlock_count"].append(m.get("unlock_count", 0))
                else:
                    fail_count += 1

            n = len(results["survived"])
            if n > 0:
                surv = np.mean(results["survived"])
                goal = np.mean(results["reached_goal"])
                steps_avg = np.mean(results["steps"])
                risky_avg = np.mean(results["risky_entered"])
                warn_avg = np.mean(results["warn_count"])
                unlock_avg = np.mean(results["unlock_count"])
                lines.append(
                    f"  {policy_name:12s}: surv={surv:.2f} goal={goal:.2f} "
                    f"steps={steps_avg:.1f} risky={risky_avg:.1f} "
                    f"warn={warn_avg:.1f} unlock={unlock_avg:.1f} "
                    f"({n} ok, {fail_count} fail)")
            else:
                lines.append(f"  {policy_name:12s}: ALL FAILED ({fail_count} errors)")

            all_results[policy_name] = results

        # Canonical vs no_tutor lift
        if all_results.get("canonical") and all_results.get("no_tutor"):
            c_surv = np.mean(all_results["canonical"]["survived"]) if all_results["canonical"]["survived"] else 0
            n_surv = np.mean(all_results["no_tutor"]["survived"]) if all_results["no_tutor"]["survived"] else 0
            c_goal = np.mean(all_results["canonical"]["reached_goal"]) if all_results["canonical"]["reached_goal"] else 0
            n_goal = np.mean(all_results["no_tutor"]["reached_goal"]) if all_results["no_tutor"]["reached_goal"] else 0
            lines.append(f"  LIFT surv: {c_surv - n_surv:+.2f}  goal: {c_goal - n_goal:+.2f}")

    lines.append("\n" + "=" * 70)
    lines.append("Done.")

    output = "\n".join(lines)
    print(output)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
