"""Step 4: Prior Refactor Experiment — baseline vs structural vs pcfg.

Four conditions:
  A. legacy_bonus  — old exp(β_C·C_t(g)) in update
  B. no_bonus      — legacy init, no compatibility bonus (ablation: remove progress)
  C. structural    — P₀(g|c₀) structural prior, pure likelihood update
  D. pcfg          — PCFG prior, pure likelihood update

Primary metric: SubgoalMarginalAcc, NLL, entropy, q(u) calibration.
Secondary: composite top-1 accuracy (sanity only), false composite inflation.

Usage:
  python scripts/run_step4_prior_experiment.py
  python scripts/run_step4_prior_experiment.py --n_seeds 30 --theta_mode k
"""

from __future__ import annotations
import sys, os, argparse, time, csv
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.teachers.compositional_goal_hypotheses import (
    DEFAULT_GOAL_SPACE, GoalHypothesisSpace, ATOMIC_GOALS,
)
from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, THETA_2, THETA_K,
)
from src.teachers.compositional_goal_prior import (
    GoalPriorContext, GoalPriorConfig, PCFGPriorConfig,
    compute_subgoal_marginals,
)


AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Synthetic branch setups
SCENARIOS = {
    "goal_aligned": {
        "branches": [
            BranchAttributes(safety_score=0.8, temptation_score=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.5, risk_penalty=0.4),
        ],
        "true_goal": "avoid_blue",
        "true_theta": "safe",
    },
    "goal_conflict": {
        "branches": [
            BranchAttributes(safety_score=0.6, temptation_score=0.3, risk_penalty=0.2),
            BranchAttributes(safety_score=0.5, temptation_score=0.4, risk_penalty=0.3),
        ],
        "true_goal": "collect_red+avoid_blue",
        "true_theta": "safe",
    },
    "temptation_hard": {
        "branches": [
            BranchAttributes(safety_score=0.7, temptation_score=0.1, risk_penalty=0.15),
            BranchAttributes(safety_score=0.3, temptation_score=0.8, risk_penalty=0.35),
        ],
        "true_goal": "avoid_blue+use_safe",
        "true_theta": "shiny",
    },
    "shortcut": {
        "branches": [
            BranchAttributes(safety_score=0.5, temptation_score=0.0,
                             shortcut_bonus=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, temptation_score=0.2,
                             shortcut_bonus=0.6, risk_penalty=0.2),
        ],
        "true_goal": "reach_fast+avoid_blue",
        "true_theta": "safe",
    },
}


def make_posterior(variant, theta_mode="2"):
    pref_types = THETA_2 if theta_mode == "2" else THETA_K
    ctx = GoalPriorContext()
    cfg = GoalPriorConfig(beta_len=1.0, beta_red=0.5)
    pcfg_cfg = PCFGPriorConfig(p_atomic=0.7, p_compose=0.3)

    if variant == "legacy_bonus":
        return JointGoalPrefPosterior(
            pref_types=pref_types, params=AP, prior_mode="legacy_bonus")
    elif variant == "no_bonus":
        # Legacy init (uniform), but no compatibility bonus in update
        return JointGoalPrefPosterior(
            pref_types=pref_types, params=AP, prior_mode="structural",
            prior_context=ctx, prior_config=GoalPriorConfig(beta_len=0.0, beta_red=0.0))
    elif variant == "structural":
        return JointGoalPrefPosterior(
            pref_types=pref_types, params=AP, prior_mode="structural",
            prior_context=ctx, prior_config=cfg)
    elif variant == "pcfg":
        return JointGoalPrefPosterior(
            pref_types=pref_types, params=AP, prior_mode="pcfg",
            pcfg_config=pcfg_cfg)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_episode(variant, scenario_name, seed, n_steps=10, theta_mode="2"):
    sc = SCENARIOS[scenario_name]
    branches = sc["branches"]
    true_goal = sc["true_goal"]
    true_theta = sc["true_theta"]
    rng = np.random.default_rng(seed)

    post = make_posterior(variant, theta_mode)
    true_gh = DEFAULT_GOAL_SPACE.get(true_goal)

    total_nll = 0.0
    for step in range(n_steps):
        # Sample action from true goal+theta
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
            branches, true_gh, true_theta, AP)
        action = int(rng.choice(len(branches), p=probs))

        # NLL
        mg = post.marginal_goal()
        mp = post.marginal_pref()
        pred_prob = 0.0
        for gl, gw in mg.items():
            gh = DEFAULT_GOAL_SPACE.get(gl)
            for th, tw in mp.items():
                cp = DEFAULT_GOAL_SPACE.compute_choice_probs(
                    branches, gh, th, AP)
                pred_prob += gw * tw * cp[action]
        nll = -np.log(max(pred_prob, 1e-15))
        total_nll += nll

        # Update
        post.update(None, branches, action)

    # Final metrics
    mg = post.marginal_goal()
    sm = post.subgoal_marginals()

    # Subgoal marginal accuracy
    true_components = set(true_gh.components)
    sm_acc = sum(sm.get(u, 0.0) for u in true_components) / len(true_components)

    # Atomic top-1 accuracy
    pred_goal = post.predicted_goal()
    goal_acc = float(pred_goal == true_goal)

    # Entropy
    entropy = post.entropy()

    # False composite inflation: mass on composites that are wrong
    false_comp_mass = 0.0
    for gl, gw in mg.items():
        gh = DEFAULT_GOAL_SPACE.get(gl)
        if gh.is_composite and gl != true_goal:
            false_comp_mass += gw

    return {
        "variant": variant,
        "scenario": scenario_name,
        "seed": seed,
        "theta_mode": theta_mode,
        "true_goal": true_goal,
        "true_theta": true_theta,
        "nll": total_nll / n_steps,
        "sm_acc": sm_acc,
        "goal_acc": goal_acc,
        "entropy": entropy,
        "false_comp_mass": false_comp_mass,
        "pred_goal": pred_goal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--theta_mode", type=str, default="2",
                        choices=["2", "k"])
    parser.add_argument("--outdir", type=str, default="results/step4_prior")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    variants = ["legacy_bonus", "no_bonus", "structural", "pcfg"]
    scenarios = list(SCENARIOS.keys())

    total = len(variants) * len(scenarios) * args.n_seeds
    print(f"Step 4: {total} episodes "
          f"({len(variants)} variants × {len(scenarios)} scenarios × "
          f"{args.n_seeds} seeds, Θ={args.theta_mode})", file=sys.stderr)

    t0 = time.time()
    results = []
    done = 0

    for variant in variants:
        for scenario in scenarios:
            for seed in range(args.n_seeds):
                try:
                    r = run_episode(variant, scenario, seed,
                                    args.n_steps, args.theta_mode)
                    results.append(r)
                except Exception as e:
                    print(f"  ERROR: {variant}/{scenario}/s{seed}: {e}",
                          file=sys.stderr)
                done += 1
                if done % 40 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s",
                          file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)}/{total} in {elapsed:.1f}s",
          file=sys.stderr)

    # ── Aggregate ──────────────────────────────────────────────
    groups = defaultdict(list)
    for r in results:
        groups[(r["variant"], r["scenario"])].append(r)

    # Print headline table
    print("\n" + "=" * 90)
    print("Step 4: Prior Refactor — Headline Results")
    print("=" * 90)
    hdr = (f"{'Variant':<16} {'Scenario':<18} {'NLL':>6} {'SM_Acc':>7} "
           f"{'GoalAcc':>8} {'H':>6} {'FCI':>6}")
    print(hdr)
    print("-" * 70)

    lines = [f"# Step 4: Prior Refactor Results (Θ{args.theta_mode})\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds} | **Steps**: {args.n_steps} | "
                 f"**Elapsed**: {elapsed:.1f}s\n\n")
    lines.append("## Headline Metrics\n\n")
    lines.append("| Variant | Scenario | NLL | SM_Acc | GoalAcc | H | FCI |\n")
    lines.append("|---------|----------|-----|--------|---------|---|-----|\n")

    for variant in variants:
        for scenario in scenarios:
            key = (variant, scenario)
            rs = groups.get(key, [])
            if not rs:
                continue
            nll = np.mean([r["nll"] for r in rs])
            sm_acc = np.mean([r["sm_acc"] for r in rs])
            goal_acc = np.mean([r["goal_acc"] for r in rs])
            entropy = np.mean([r["entropy"] for r in rs])
            fci = np.mean([r["false_comp_mass"] for r in rs])

            row = (f"{variant:<16} {scenario:<18} {nll:>6.3f} {sm_acc:>7.3f} "
                   f"{goal_acc:>8.3f} {entropy:>6.3f} {fci:>6.3f}")
            print(row)
            lines.append(f"| {variant} | {scenario} | {nll:.3f} | "
                         f"{sm_acc:.3f} | {goal_acc:.3f} | {entropy:.3f} | "
                         f"{fci:.3f} |\n")

    # ── Promotion analysis ────────────────────────────────────
    lines.append("\n## Promotion Analysis\n\n")
    for scenario in scenarios:
        lines.append(f"### {scenario}\n\n")
        legacy = groups.get(("legacy_bonus", scenario), [])
        struct = groups.get(("structural", scenario), [])
        if legacy and struct:
            l_nll = np.mean([r["nll"] for r in legacy])
            s_nll = np.mean([r["nll"] for r in struct])
            l_sm = np.mean([r["sm_acc"] for r in legacy])
            s_sm = np.mean([r["sm_acc"] for r in struct])
            lines.append(f"- NLL: {s_nll:.3f} vs {l_nll:.3f} "
                         f"{'BETTER' if s_nll < l_nll else 'WORSE'}\n")
            lines.append(f"- SM_Acc: {s_sm:.3f} vs {l_sm:.3f} "
                         f"{'BETTER' if s_sm > l_sm else 'WORSE'}\n")
        lines.append("\n")

    rpt = out / f"step4_prior_t{args.theta_mode}.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt}", file=sys.stderr)

    # CSV
    csv_path = out / f"step4_episodes_t{args.theta_mode}.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"CSV -> {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
