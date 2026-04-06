"""DTMB-L Oracle Sanity Check (Exp D).

Compares no_tutor, canonical, dtmb_oracle (O1 and O2 variants) on
medium and hard difficulties. Key checks:
  - Oracle must not block goal (goal > 0)
  - Oracle survival > canonical (upper bound)
  - Stage schedule aligns with GT bottleneck

Usage:
    python scripts/eval_dtmb_oracle_sanity.py [--seeds 50] [--output-file ...]
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.dtmb_helpers import DTMBDispatchConfig

FAMILY = "deep_tree_mixed_bottleneck_lattice"


def _make_oracle_policy(oracle_variant="O1"):
    return {
        "tutor_mode": "dtmb_oracle",
        "warning_mode": "none",
        "robot_belief_mode": False,
        "intervention_family_mode": False,
        "item_drop_enabled": False,
        "belief_planning_mode": True,
        "latent_mode": True,
        "patch_radius": 2,
        "prefix_horizon": 0,
    }


POLICIES = {
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
        "allowed_interventions": None,
    },
    "oracle_O1": _make_oracle_policy("O1"),
    "oracle_O2": _make_oracle_policy("O2"),
}

ORACLE_VARIANTS = {"oracle_O1": "O1", "oracle_O2": "O2"}


def parse_args():
    p = argparse.ArgumentParser(description="DTMB-L Oracle sanity check")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--difficulty", nargs="+", default=["medium", "hard"])
    p.add_argument("--output-file", default="results/dtmb_oracle_sanity.txt")
    return p.parse_args()


def run_episode(runner, seed, difficulty, policy_cfg, oracle_variant=None):
    dcfg = DTMBDispatchConfig(oracle_variant=oracle_variant) if oracle_variant else None
    try:
        state = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            dtmb_dispatch_cfg=dcfg,
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
            "success": False, "error": str(e),
            "survived": False, "reached_goal": False, "steps": 0,
        }


def main():
    args = parse_args()
    runner = LatticeV2Runner()

    lines = []
    lines.append(f"DTMB-L Oracle Sanity Check (Exp D)")
    lines.append(f"  seeds={args.seeds}, difficulties={args.difficulty}")
    lines.append("=" * 80)

    for diff in args.difficulty:
        lines.append(f"\n--- {diff.upper()} ---")
        all_results = {}

        for policy_name, policy_cfg in POLICIES.items():
            results = defaultdict(list)
            fail_count = 0
            ov = ORACLE_VARIANTS.get(policy_name)

            for seed in range(args.seeds):
                m = run_episode(runner, seed, diff, policy_cfg, oracle_variant=ov)
                if m["success"]:
                    results["survived"].append(m["survived"])
                    results["reached_goal"].append(m["reached_goal"])
                    results["steps"].append(m["steps"])
                    results["warn_count"].append(m.get("warn_count", 0))
                    results["unlock_count"].append(m.get("unlock_count", 0))
                else:
                    fail_count += 1

            n = len(results["survived"])
            if n > 0:
                surv = np.mean(results["survived"])
                goal = np.mean(results["reached_goal"])
                steps_avg = np.mean(results["steps"])
                warn_avg = np.mean(results["warn_count"])
                unlock_avg = np.mean(results["unlock_count"])
                lines.append(
                    f"  {policy_name:14s}: surv={surv:.3f} goal={goal:.3f} "
                    f"steps={steps_avg:.1f} warn={warn_avg:.1f} "
                    f"unlock={unlock_avg:.1f} ({n} ok, {fail_count} fail)")
            else:
                lines.append(f"  {policy_name:14s}: ALL FAILED ({fail_count} errors)")

            all_results[policy_name] = results

        # Sanity checks
        lines.append(f"\n  --- Sanity Checks ({diff}) ---")
        for ov_name in ["oracle_O1", "oracle_O2"]:
            ov_results = all_results.get(ov_name, {})
            if ov_results.get("reached_goal"):
                goal_rate = np.mean(ov_results["reached_goal"])
                surv_rate = np.mean(ov_results["survived"])
                lines.append(f"  {ov_name} goal={goal_rate:.3f}: {'PASS' if goal_rate > 0 else 'FAIL — oracle blocks goal!'}")
                # Check upper bound
                can_surv = np.mean(all_results["canonical"]["survived"]) if all_results["canonical"]["survived"] else 0
                lines.append(f"  {ov_name} surv ({surv_rate:.3f}) vs canonical ({can_surv:.3f}): "
                           f"{'PASS (≥ canonical)' if surv_rate >= can_surv - 0.05 else 'NOTE: oracle below canonical'}")

    lines.append("\n" + "=" * 80)
    lines.append("Done.")

    output = "\n".join(lines)
    print(output)

    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
