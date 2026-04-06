"""K1-K6: Unified Multi-Latent Framework Experiment.

Uses pipeline.py config-driven runner to validate:
  K1: Goal posterior convergence (5 goal types × 50 trials)
  K3: Goal+preference tutor vs v4 vs pref_v2 vs oracle
  K4: Cross-family with temptation corridor (stochastic agent)
  K6: Multi-latent robustness suite
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.evals.pipeline import ExperimentConfig, run_experiment, write_report
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.agents.goal_posterior_v1 import GoalPosteriorV1, GOAL_TYPES, compute_goal_likelihood
from src.agents.preference_posterior_v2 import PreferencePosteriorV2
from src.agents.joint_latent_belief import JointLatentBelief

out = Path("results")
out.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# K1: Goal posterior convergence test
# ══════════════════════════════════════════════════════════════
def k1_goal_convergence():
    print("K1: Goal Posterior Convergence", file=sys.stderr)
    params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
    results = []

    for true_goal in GOAL_TYPES:
        correct = 0
        n_trials = 50
        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 2000)
            gp = GoalPosteriorV1()
            for obs_i in range(20):
                safe_br = BranchAttributes(
                    safety_score=0.7 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.1 + rng.uniform(0, 0.1),
                    shortcut_bonus=rng.uniform(0, 0.5),
                    texture_novelty=rng.uniform(0, 0.3),
                    risk_penalty=0.1)
                tempt_br = BranchAttributes(
                    safety_score=0.3 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.8 + rng.uniform(-0.1, 0.1),
                    shortcut_bonus=rng.uniform(0, 0.3),
                    texture_novelty=rng.uniform(0.2, 0.6),
                    risk_penalty=0.4)
                branches = [safe_br, tempt_br]
                # Goal-driven branch choice
                liks = [compute_goal_likelihood(0, branches, true_goal, params),
                        compute_goal_likelihood(1, branches, true_goal, params)]
                chosen = int(rng.choice(2, p=[liks[0]/(sum(liks)), liks[1]/(sum(liks))]))
                gp.update_from_choice(chosen, branches, params)
            if gp.predicted_type == true_goal:
                correct += 1
        results.append({"true_goal": true_goal, "GoalAcc": round(correct / n_trials, 3)})
    return results


# ══════════════════════════════════════════════════════════════
# K2: Joint latent convergence
# ══════════════════════════════════════════════════════════════
def k2_joint_convergence():
    print("K2: Joint Latent Convergence", file=sys.stderr)
    from src.agents.stochastic_agent_policy import PREFERENCE_TYPES, compute_likelihood
    params = AgentPolicyParams(beta=4.0, epsilon=0.1)
    results = []

    for true_theta in ["safe", "shiny"]:
        for true_goal in ["goal_safe_short", "goal_collect"]:
            correct_pref = 0
            correct_goal = 0
            n_trials = 50
            for trial in range(n_trials):
                rng = np.random.default_rng(trial + 3000)
                jb = JointLatentBelief()
                for obs_i in range(25):
                    safe_br = BranchAttributes(
                        safety_score=0.7 + rng.uniform(-0.1, 0.1),
                        temptation_score=0.1, shortcut_bonus=0.3,
                        texture_novelty=0.1, risk_penalty=0.1)
                    tempt_br = BranchAttributes(
                        safety_score=0.3 + rng.uniform(-0.1, 0.1),
                        temptation_score=0.85, shortcut_bonus=0.1,
                        texture_novelty=0.4, risk_penalty=0.4)
                    branches = [safe_br, tempt_br]
                    # Choice driven by both pref AND goal
                    u_pref = compute_likelihood(0, branches, true_theta, params)
                    u_goal = compute_goal_likelihood(0, branches, true_goal, params)
                    p_safe = (u_pref + u_goal) / 2
                    p_safe = np.clip(p_safe, 0.05, 0.95)
                    chosen = int(rng.random() > p_safe)
                    jb.update_from_choice(chosen, branches, params)
                if jb.pref_posterior.predicted_type == true_theta:
                    correct_pref += 1
                if jb.goal_posterior.predicted_type == true_goal:
                    correct_goal += 1
            results.append({
                "theta": true_theta, "goal": true_goal,
                "PrefAcc": round(correct_pref / n_trials, 3),
                "GoalAcc": round(correct_goal / n_trials, 3),
            })
    return results


# ══════════════════════════════════════════════════════════════
# K3+K6: Unified cross-family experiment via pipeline
# ══════════════════════════════════════════════════════════════
def k3_k6_unified():
    print("K3+K6: Unified Multi-Latent Experiment", file=sys.stderr)
    configs = []

    families = {
        "elcb_po":      {"family_kwargs": {}, "obs_radius": 2},
        "delayed_Δ=-2": {"family_kwargs": {"commit_depth": 1, "reveal_depth": 3}, "obs_radius": 1},
        "delayed_Δ=2":  {"family_kwargs": {"commit_depth": 5, "reveal_depth": 3}, "obs_radius": 5},
        "tempt_high":   {"family_kwargs": {"temptation_strength": 0.9}, "obs_radius": 2},
    }
    fam_keys = {
        "elcb_po": "elcb_po",
        "delayed_Δ=-2": "delayed_corridor",
        "delayed_Δ=2": "delayed_corridor",
        "tempt_high": "temptation_corridor",
    }
    tutors = ["always_wait", "v4", "pref_v2", "goal_v1", "oracle"]

    for fname, fcfg in families.items():
        for t in tutors:
            is_tempt = "tempt" in fname
            configs.append(ExperimentConfig(
                name=f"{fname}/{t}",
                family=fam_keys[fname],
                family_kwargs=fcfg["family_kwargs"],
                tutor_type=t,
                agent_type="stochastic" if is_tempt else "deterministic",
                obs_radius=fcfg["obs_radius"],
            ))

    results = []
    for cfg in configs:
        r = run_experiment(cfg)
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    k1 = k1_goal_convergence()
    k2 = k2_joint_convergence()
    k3k6 = k3_k6_unified()

    with open(out / "multi_latent_report.md", "w") as f:
        f.write("# Multi-Latent Framework Report\n\n")

        # K1
        f.write("## K1: Goal Posterior Convergence (20 obs, 50 trials)\n\n")
        f.write("| True Goal | GoalAcc |\n")
        f.write("|-----------|--------|\n")
        for r in k1:
            f.write("| {} | {:.0%} |\n".format(r["true_goal"], r["GoalAcc"]))
        mean_g = np.mean([r["GoalAcc"] for r in k1])
        f.write("\n**Mean GoalAcc: {:.1%}** (chance = {:.1%})\n".format(mean_g, 1/5))

        # K2
        f.write("\n## K2: Joint Latent Convergence (25 obs, 50 trials)\n\n")
        f.write("| θ | Goal | PrefAcc | GoalAcc |\n")
        f.write("|---|------|---------|--------|\n")
        for r in k2:
            f.write("| {} | {} | {:.0%} | {:.0%} |\n".format(
                r["theta"], r["goal"], r["PrefAcc"], r["GoalAcc"]))

        # K3+K6
        f.write("\n## K3+K6: Cross-Family Multi-Latent Robustness\n\n")
        write_report(k3k6, out / "_tmp_k3k6.md", "tmp")
        # Inline the table
        with open(out / "_tmp_k3k6.md") as tmp:
            lines = tmp.readlines()[2:]  # skip title
            f.writelines(lines)

        # Summary comparison
        f.write("\n### Tutor Comparison Summary\n\n")
        f.write("| Family | v4 SBCR | pref_v2 | goal_v1 | Oracle |\n")
        f.write("|--------|---------|---------|---------|--------|\n")
        for fname in ["elcb_po", "delayed_Δ=-2", "delayed_Δ=2", "tempt_high"]:
            v4 = [r for r in k3k6 if r["name"] == f"{fname}/v4"]
            pv = [r for r in k3k6 if r["name"] == f"{fname}/pref_v2"]
            gv = [r for r in k3k6 if r["name"] == f"{fname}/goal_v1"]
            orc = [r for r in k3k6 if r["name"] == f"{fname}/oracle"]
            if v4 and pv and gv and orc:
                f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |\n".format(
                    fname, v4[0]["SBCR"], pv[0]["SBCR"],
                    gv[0]["SBCR"], orc[0]["SBCR"]))

    # Cleanup
    (out / "_tmp_k3k6.md").unlink(missing_ok=True)

    print("Report -> results/multi_latent_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
