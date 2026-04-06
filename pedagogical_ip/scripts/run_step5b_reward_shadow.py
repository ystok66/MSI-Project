"""Step 5B: Continuous Reward Weight Shadow Experiment.

Tests rigid table vs B1 (1D) vs B2 (2D) vs B3 (4D) residuals.
Also compares against Θ_K discrete shadow.

Metrics: held-out action NLL, Brier, predictive calibration.

Usage:
  python scripts/run_step5b_reward_shadow.py --n_seeds 30
"""

from __future__ import annotations
import sys, os, time, argparse
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, compute_choice_probs,
)
from src.agents.continuous_reward_shadow import (
    ContinuousRewardShadow, RewardShadowConfig,
)
from src.teachers.compositional_goal_hypotheses import (
    DEFAULT_GOAL_SPACE, ATOMIC_GOALS,
)

AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

SCENARIOS = {
    "safe_agent": {
        "branches": [
            BranchAttributes(safety_score=0.8, temptation_score=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.5, risk_penalty=0.4),
        ],
        "true_goal": "avoid_blue", "true_theta": "safe",
    },
    "shiny_agent": {
        "branches": [
            BranchAttributes(safety_score=0.6, temptation_score=0.3, risk_penalty=0.2),
            BranchAttributes(safety_score=0.5, temptation_score=0.7, risk_penalty=0.3),
        ],
        "true_goal": "collect_red", "true_theta": "shiny",
    },
    "composite_safe": {
        "branches": [
            BranchAttributes(safety_score=0.7, temptation_score=0.1, risk_penalty=0.15),
            BranchAttributes(safety_score=0.3, temptation_score=0.8, risk_penalty=0.35),
        ],
        "true_goal": "avoid_blue+use_safe", "true_theta": "safe",
    },
    "shortcut_agent": {
        "branches": [
            BranchAttributes(safety_score=0.5, temptation_score=0.0,
                             shortcut_bonus=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, temptation_score=0.0,
                             shortcut_bonus=0.6, risk_penalty=0.2),
        ],
        "true_goal": "reach_fast", "true_theta": "shortcut",
    },
}


def run_episode(mode, scenario, seed, n_train=8, n_test=5):
    """Train on n_train, evaluate on n_test held-out."""
    sc = SCENARIOS[scenario]
    branches = sc["branches"]
    true_goal = sc["true_goal"]
    true_theta = sc["true_theta"]
    true_gh = DEFAULT_GOAL_SPACE.get(true_goal)
    rng = np.random.default_rng(seed)

    if mode == "rigid":
        # No shadow — just use discrete table
        train_nlls, test_nlls = [], []
        for step in range(n_train + n_test):
            probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
                branches, true_gh, true_theta, AP)
            action = int(rng.choice(len(branches), p=probs))

            # Prediction from discrete table
            pred_probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
                branches, true_gh, true_theta, AP)
            nll = -np.log(max(float(pred_probs[action]), 1e-15))

            if step < n_train:
                train_nlls.append(nll)
            else:
                test_nlls.append(nll)

        return {
            "train_nll": np.mean(train_nlls),
            "test_nll": np.mean(test_nlls),
            "residual_norm": 0.0,
            "n_params": 0,
        }

    # Shadow mode
    shadow = ContinuousRewardShadow(
        mode=mode,
        config=RewardShadowConfig(learning_rate=0.02, prior_var=1.0))

    train_nlls, test_nlls = [], []
    for step in range(n_train + n_test):
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
            branches, true_gh, true_theta, AP)
        action = int(rng.choice(len(branches), p=probs))

        nll = shadow.predictive_nll(branches, action, true_goal, true_theta)

        if step < n_train:
            shadow.observe(branches, action, true_goal, true_theta)
            train_nlls.append(nll)
        else:
            test_nlls.append(nll)

    return {
        "train_nll": np.mean(train_nlls),
        "test_nll": np.mean(test_nlls),
        "residual_norm": shadow.residual_norm(true_goal, true_theta),
        "n_params": shadow.n_params(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    args = parser.parse_args()

    out = Path("results/step5b_reward")
    out.mkdir(parents=True, exist_ok=True)

    modes = ["rigid", "B1", "B2", "B3"]
    scenarios = list(SCENARIOS.keys())
    t0 = time.time()

    lines = ["# Step 5B: Continuous Reward Shadow Results\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds}\n\n")

    lines.append("## Headline Metrics\n\n")
    lines.append("| Mode | Scenario | Train_NLL↓ | Test_NLL↓ | Resid_Norm | Params |\n")
    lines.append("|------|----------|-----------|----------|------------|--------|\n")

    for mode in modes:
        for scenario in scenarios:
            rs = [run_episode(mode, scenario, s) for s in range(args.n_seeds)]
            tr_nll = np.mean([r["train_nll"] for r in rs])
            te_nll = np.mean([r["test_nll"] for r in rs])
            rn = np.mean([r["residual_norm"] for r in rs])
            np_val = rs[0]["n_params"]

            lines.append(f"| {mode} | {scenario} | {tr_nll:.4f} | "
                         f"{te_nll:.4f} | {rn:.4f} | {np_val} |\n")
            print(f"  {mode}/{scenario}: train={tr_nll:.4f} test={te_nll:.4f} "
                  f"resid={rn:.4f}", file=sys.stderr)

    # Promotion analysis
    lines.append("\n## Promotion Analysis\n\n")
    for scenario in scenarios:
        rs_rigid = [run_episode("rigid", scenario, s) for s in range(args.n_seeds)]
        rs_b1 = [run_episode("B1", scenario, s) for s in range(args.n_seeds)]
        rs_b2 = [run_episode("B2", scenario, s) for s in range(args.n_seeds)]

        rig_nll = np.mean([r["test_nll"] for r in rs_rigid])
        b1_nll = np.mean([r["test_nll"] for r in rs_b1])
        b2_nll = np.mean([r["test_nll"] for r in rs_b2])

        delta_b1 = b1_nll - rig_nll
        delta_b2 = b2_nll - rig_nll

        lines.append(f"### {scenario}\n")
        lines.append(f"- B1 vs rigid: Δ test NLL = {delta_b1:+.4f} "
                     f"{'BETTER' if delta_b1 < -0.001 else ('WORSE' if delta_b1 > 0.001 else 'PARITY')}\n")
        lines.append(f"- B2 vs rigid: Δ test NLL = {delta_b2:+.4f} "
                     f"{'BETTER' if delta_b2 < -0.001 else ('WORSE' if delta_b2 > 0.001 else 'PARITY')}\n\n")

    elapsed = time.time() - t0
    lines[1] = f"**Seeds**: {args.n_seeds} | **Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step5b_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
