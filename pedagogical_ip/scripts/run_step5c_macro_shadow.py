"""Step 5C: Bayesian Macro Objective Shadow Experiment.

Tests baseline hand-crafted score vs unified Bayes objective.
Ablations: task-only, task+info, task+info-dep, full, full+κ, full-κ.

Usage:
  python scripts/run_step5c_macro_shadow.py --n_seeds 30
"""

from __future__ import annotations
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams,
)
from src.agents.agent_belief_state import AgentBelief
from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
from src.teachers.joint_goal_pref_posterior import JointGoalPrefPosterior
from src.teachers.compositional_goal_prior import GoalPriorContext, GoalPriorConfig
from src.teachers.action_predictor import ActionPredictor
from src.teachers.goal_conditional_curriculum_hook import (
    GoalConditionalCurriculumHook, CurriculumConfig,
)
from src.teachers.bayesian_macro_objective_shadow import (
    BayesianMacroObjectiveShadow, BayesMacroConfig,
)

AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

SCENARIOS = {
    "low_risk": {
        "branches": [
            BranchAttributes(safety_score=0.8, temptation_score=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, temptation_score=0.4, risk_penalty=0.3),
        ],
        "true_goal": "avoid_blue", "true_theta": "safe",
        "kappa_hat": 0.3, "nu_hat": 0.1, "gamma_gen_hat": 0.7,
    },
    "high_risk": {
        "branches": [
            BranchAttributes(safety_score=0.5, temptation_score=0.1, risk_penalty=0.15),
            BranchAttributes(safety_score=0.2, temptation_score=0.8, risk_penalty=0.5),
        ],
        "true_goal": "collect_red+avoid_blue", "true_theta": "shiny",
        "kappa_hat": 0.1, "nu_hat": 0.5, "gamma_gen_hat": 0.3,
    },
    "dependent_agent": {
        "branches": [
            BranchAttributes(safety_score=0.6, temptation_score=0.2, risk_penalty=0.2),
            BranchAttributes(safety_score=0.4, temptation_score=0.6, risk_penalty=0.35),
        ],
        "true_goal": "avoid_blue+use_safe", "true_theta": "safe",
        "kappa_hat": 0.5, "nu_hat": 0.8, "gamma_gen_hat": 0.2,
    },
    "well_calibrated": {
        "branches": [
            BranchAttributes(safety_score=0.7, temptation_score=0.1, risk_penalty=0.1),
            BranchAttributes(safety_score=0.5, temptation_score=0.3, risk_penalty=0.2),
        ],
        "true_goal": "use_safe", "true_theta": "safe",
        "kappa_hat": 0.6, "nu_hat": 0.2, "gamma_gen_hat": 0.8,
    },
}


def run_comparison(scenario_name, seed, ablation_configs):
    """Compare baseline and shadow macro objectives."""
    sc = SCENARIOS[scenario_name]
    branches = sc["branches"]
    true_goal = sc["true_goal"]
    true_theta = sc["true_theta"]
    kh, nh, gh = sc["kappa_hat"], sc["nu_hat"], sc["gamma_gen_hat"]
    rng = np.random.default_rng(seed)

    # Build posterior with some evidence
    post = JointGoalPrefPosterior(
        params=AP, prior_mode="structural",
        prior_context=GoalPriorContext(),
        prior_config=GoalPriorConfig())
    true_gh = DEFAULT_GOAL_SPACE.get(true_goal)

    # Feed some observations
    for _ in range(5):
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
            branches, true_gh, true_theta, AP)
        action = int(rng.choice(len(branches), p=probs))
        post.update(None, branches, action)

    ab = AgentBelief()
    ap = ActionPredictor(params=AP)

    results = {}

    # Baseline
    hook = GoalConditionalCurriculumHook(ap)
    base_dec = hook.decide(post, branches, ab, kappa_hat=kh, nu_hat=nh)
    results["baseline"] = {
        "chosen": base_dec.chosen_option,
        "scores": dict(base_dec.scores),
    }

    # Shadow variants
    for abl_name, abl_cfg in ablation_configs.items():
        shadow = BayesianMacroObjectiveShadow(ap, config=abl_cfg)
        dec = shadow.evaluate(
            post, branches, ab,
            kappa_hat=kh, nu_hat=nh, gamma_gen_hat=gh,
            baseline_option=base_dec.chosen_option)

        results[abl_name] = {
            "chosen": dec.chosen_option,
            "agrees": dec.agrees_with_baseline,
            "breakdowns": {
                opt: {
                    "task": bd.task_gain,
                    "info": bd.info_gain,
                    "dep": bd.dep_cost,
                    "kappa": bd.kappa_term,
                    "res": bd.resource_cost,
                    "total": bd.total,
                }
                for opt, bd in dec.breakdowns.items()
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    args = parser.parse_args()

    out = Path("results/step5c_macro")
    out.mkdir(parents=True, exist_ok=True)

    # Define ablation configs
    ablations = {
        "task_only": BayesMacroConfig(beta_info=0.0, beta_dep=0.0,
                                       beta_kappa=0.0, beta_resource=0.0),
        "task+info": BayesMacroConfig(beta_info=0.5, beta_dep=0.0,
                                       beta_kappa=0.0, beta_resource=0.0),
        "task+info-dep": BayesMacroConfig(beta_info=0.5, beta_dep=1.0,
                                           beta_kappa=0.0, beta_resource=0.0),
        "full": BayesMacroConfig(),
        "full_no_kappa": BayesMacroConfig(beta_kappa=0.0),
    }

    t0 = time.time()
    scenarios = list(SCENARIOS.keys())

    lines = ["# Step 5C: Bayesian Macro Objective Shadow Results\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds}\n\n")

    # Agreement table
    lines.append("## Agreement with Baseline\n\n")
    lines.append("| Ablation | Scenario | Baseline | Shadow | Agrees |\n")
    lines.append("|----------|----------|----------|--------|--------|\n")

    agree_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for scenario in scenarios:
        for seed in range(args.n_seeds):
            r = run_comparison(scenario, seed, ablations)
            for abl_name in ablations:
                a = r[abl_name]["agrees"]
                agree_counts[abl_name] += int(a)
                total_counts[abl_name] += 1

                if seed == 0:  # Report first seed as example
                    lines.append(
                        f"| {abl_name} | {scenario} | "
                        f"{r['baseline']['chosen']} | "
                        f"{r[abl_name]['chosen']} | "
                        f"{'✓' if a else '✗'} |\n")

    # Summary
    lines.append("\n## Agreement Summary\n\n")
    lines.append("| Ablation | Agreement Rate |\n")
    lines.append("|----------|----------------|\n")
    for abl_name in ablations:
        rate = agree_counts[abl_name] / max(total_counts[abl_name], 1)
        lines.append(f"| {abl_name} | {rate:.3f} ({agree_counts[abl_name]}/{total_counts[abl_name]}) |\n")

    # Component audit
    lines.append("\n## Component Breakdown (first seed)\n\n")
    for scenario in scenarios:
        r = run_comparison(scenario, 0, ablations)
        lines.append(f"### {scenario}\n\n")
        full = r.get("full", {})
        if "breakdowns" in full:
            lines.append("| Option | Task | Info | Dep | κ | Res | Total |\n")
            lines.append("|--------|------|------|-----|---|-----|-------|\n")
            for opt, bd in full["breakdowns"].items():
                lines.append(f"| {opt} | {bd['task']:.3f} | {bd['info']:.3f} | "
                             f"{bd['dep']:.3f} | {bd['kappa']:.3f} | "
                             f"{bd['res']:.2f} | {bd['total']:.3f} |\n")
            lines.append(f"\n**Baseline chose**: {r['baseline']['chosen']} | "
                         f"**Shadow chose**: {full['chosen']}\n\n")

    elapsed = time.time() - t0
    lines[1] = f"**Seeds**: {args.n_seeds} | **Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step5c_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
