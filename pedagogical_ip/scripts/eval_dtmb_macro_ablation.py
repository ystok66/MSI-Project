"""DTMB-L Macro Ablation — single-lever ablation studies.

Tests the necessity of each macro lever by restricting allowed interventions:
  - canonical:    all levers active {WAIT, WARN, UNLOCK, ITEM_DROP}
  - no_warn:      {WAIT, UNLOCK, ITEM_DROP} — WARN disabled
  - no_unlock:    {WAIT, WARN, ITEM_DROP} — UNLOCK disabled
  - no_item_drop: {WAIT, WARN, UNLOCK} — ITEM_DROP disabled
  - no_tutor:     no interventions (baseline)

Usage:
    python scripts/eval_dtmb_macro_ablation.py [--seeds 10] [--difficulty medium]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "deep_tree_mixed_bottleneck_lattice"


ABLATION_POLICIES = {
    "canonical": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": None,  # all allowed
    },
    "no_warn": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "UNLOCK", "ITEM_DROP"}),
    },
    "no_unlock": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": True,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "WARN", "ITEM_DROP"}),
    },
    "no_item_drop": {
        "tutor_mode": "none",
        "warning_mode": "none",
        "robot_belief_mode": True,
        "intervention_family_mode": True,
        "item_drop_enabled": False,  # no shield drops
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 5,
        "allowed_interventions": frozenset({"WAIT", "WARN", "UNLOCK"}),
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
}


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L macro ablation")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--output-file", default=None)
    return p.parse_args()


def run_episode(runner, seed, difficulty, policy_cfg):
    try:
        state = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY, **policy_cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        # Get intervention counts
        metrics["success"] = True
        return metrics
    except Exception as e:
        return {"success": False, "error": str(e),
                "survived": False, "reached_goal": False, "steps": 0}


def main():
    args = parse_args()
    runner = LatticeV2Runner()
    diff = args.difficulty

    lines = []
    lines.append(f"DTMB-L Macro Ablation: {args.seeds} seeds, difficulty={diff}")
    lines.append(f"Policies: {list(ABLATION_POLICIES.keys())}")
    lines.append("=" * 70)

    for policy_name, policy_cfg in ABLATION_POLICIES.items():
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
                lines.append(f"  {policy_name} seed={seed}: {m.get('error', 'unknown')}")

        n = len(results["survived"])
        if n > 0:
            surv = np.mean(results["survived"])
            goal = np.mean(results["reached_goal"])
            steps_avg = np.mean(results["steps"])
            risky_avg = np.mean(results["risky_entered"])
            warn_avg = np.mean(results["warn_count"])
            unlock_avg = np.mean(results["unlock_count"])
            lines.append(
                f"  {policy_name:14s}: surv={surv:.2f} goal={goal:.2f} "
                f"steps={steps_avg:.1f} risky={risky_avg:.1f} "
                f"warn={warn_avg:.1f} unlock={unlock_avg:.1f} "
                f"({n} ok, {fail_count} fail)")
        else:
            lines.append(f"  {policy_name:14s}: ALL FAILED ({fail_count} errors)")

    lines.append("\n" + "=" * 70)
    lines.append("Done.")

    output = "\n".join(lines)
    print(output)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)


if __name__ == "__main__":
    main()
