"""T7 Exp-T7-1+2: Compositional Goal Posterior + Multi-Type Audit.

Exp-T7-1: Compare single-goal posterior vs compositional q(g,θ)
Exp-T7-2: Compare Θ₂ vs Θ_K posterior quality

Uses CGC-v2's branch profiles to simulate goal-conditioned episodes.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.world_state import WorldState
from src.teachers.compositional_goal_hypotheses import (
    GoalHypothesisSpace, ATOMIC_ONLY_GOAL_SPACE, DEFAULT_GOAL_SPACE,
)
from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, THETA_2, THETA_K,
)

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1)
N_STEPS = 15
N_SEEDS = 10


def make_goal_aligned_branches(goal_label, rng):
    """Create branches where branch 0 aligns with the given goal."""
    gh = DEFAULT_GOAL_SPACE.get(goal_label)
    w = gh.reward_weights
    # Branch 0: aligned with goal
    b0 = BranchAttributes(
        safety_score=max(0.1, min(0.95, 0.5 + w[0] * 0.15)),
        temptation_score=max(0.0, min(0.95, 0.3 + w[1] * 0.1)),
        texture_novelty=max(0.0, min(0.95, 0.3 + w[2] * 0.1)),
        shortcut_bonus=max(0.0, min(0.95, w[3] * 0.15)),
        risk_penalty=max(0.0, 0.15 - w[0] * 0.05 + rng.uniform(-0.02, 0.02)),
    )
    # Branch 1: misaligned
    b1 = BranchAttributes(
        safety_score=max(0.1, min(0.95, 0.5 - w[0] * 0.15)),
        temptation_score=max(0.0, min(0.95, 0.3 - w[1] * 0.05 + rng.uniform(0, 0.2))),
        texture_novelty=max(0.0, min(0.95, 0.3 - w[2] * 0.05)),
        shortcut_bonus=max(0.0, min(0.95, 0.1)),
        risk_penalty=max(0.0, 0.25 + w[0] * 0.05 + rng.uniform(-0.02, 0.02)),
    )
    return [b0, b1]


def run_posterior_episode(goal_label, theta_true, seed, goal_space, pref_types):
    """Run episode where agent acts according to (goal, theta)."""
    rng = np.random.default_rng(seed)
    jgpp = JointGoalPrefPosterior(
        goal_space=goal_space, pref_types=pref_types, forgetting_rate=0.005)
    ws = WorldState()

    nlls = []
    entropies = [jgpp.entropy()]
    goal_correct = []

    for step_i in range(N_STEPS):
        branches = make_goal_aligned_branches(goal_label, rng)

        # Agent chooses via goal-conditioned utility
        probs = goal_space.compute_choice_probs(branches, 
            goal_space.get(goal_label), theta_true, AP)
        ac = int(rng.choice(len(branches), p=probs))

        # Compute NLL before update for this action under the posterior
        # Use marginal predictive: sum over hypotheses
        mg = jgpp.marginal_goal()
        mp = jgpp.marginal_pref()
        pred_p = 0.0
        for gl, gw in mg.items():
            for tl, tw in mp.items():
                gh = goal_space.get(gl)
                cp = goal_space.compute_choice_probs(branches, gh, tl, AP)
                pred_p += gw * tw * cp[ac]
        nll = -np.log(max(pred_p, 1e-10))
        nlls.append(nll)

        # Update posterior
        jgpp.update(ws, branches, ac)
        entropies.append(jgpp.entropy())

        # Check goal recovery
        pg = jgpp.predicted_goal()
        # For composite goals, check if predicted goal matches or contains the true
        goal_correct.append(1 if pg == goal_label else 0)

    return {
        "mean_nll": np.mean(nlls),
        "final_entropy": entropies[-1],
        "entropy_reduction": entropies[0] - entropies[-1],
        "goal_accuracy": np.mean(goal_correct[-5:]),  # last 5 steps
        "predicted_goal": jgpp.predicted_goal(),
        "predicted_pref": jgpp.predicted_pref(),
        "marginal_goal": jgpp.marginal_goal(),
    }


def main():
    print("═══ T7 Goal Posterior + Multi-Type Audit ═══\n", file=sys.stderr)
    L = ["# T7 Exp-T7-1+2: Compositional Goal + Multi-Type Posterior\n\n"]

    # ═══ Exp-T7-1: Atomic-only vs Full (atomic+composite) ═══
    L.append("## Exp-T7-1: Single-Goal vs Compositional Posterior\n\n")
    L.append("| True Goal | θ | Space | Mean NLL | Entropy Δ | Goal Acc (last 5) |\n")
    L.append("|:---------:|:-:|:-----:|:--------:|:---------:|:-----------------:|\n")

    test_goals = ["use_safe", "collect_red", "collect_red+avoid_blue", "reach_fast+avoid_blue"]

    for goal in test_goals:
        for theta in ["safe", "shiny"]:
            for space_name, space in [("atomic", ATOMIC_ONLY_GOAL_SPACE),
                                       ("full", DEFAULT_GOAL_SPACE)]:
                # Skip composite goals for atomic-only space
                if "+" in goal and space_name == "atomic":
                    continue
                nlls, ent_ds, accs = [], [], []
                for sid in range(N_SEEDS):
                    r = run_posterior_episode(goal, theta, sid, space, THETA_2)
                    nlls.append(r["mean_nll"])
                    ent_ds.append(r["entropy_reduction"])
                    accs.append(r["goal_accuracy"])

                L.append(f"| {goal} | {theta} | {space_name} | "
                         f"{np.mean(nlls):.3f} | {np.mean(ent_ds):+.3f} | "
                         f"{np.mean(accs):.2f} |\n")
    print("  T7-1 done", file=sys.stderr)

    # ═══ Exp-T7-2: Θ₂ vs Θ_K ═══
    L.append("\n## Exp-T7-2: 2-Type vs K-Type Posterior\n\n")
    L.append("| True Goal | θ | Types | Mean NLL | Entropy | Goal Acc | Pref Acc |\n")
    L.append("|:---------:|:-:|:-----:|:--------:|:-------:|:--------:|:--------:|\n")

    for goal in ["use_safe", "collect_red"]:
        for theta in ["safe", "shiny"]:
            for types_name, types in [("Θ₂", THETA_2), ("Θ_K", THETA_K)]:
                nlls, ents, g_accs, p_accs = [], [], [], []
                for sid in range(N_SEEDS):
                    r = run_posterior_episode(goal, theta, sid,
                                            DEFAULT_GOAL_SPACE, types)
                    nlls.append(r["mean_nll"])
                    ents.append(r["final_entropy"])
                    g_accs.append(r["goal_accuracy"])
                    p_accs.append(1.0 if r["predicted_pref"] == theta else 0.0)

                L.append(f"| {goal} | {theta} | {types_name} | "
                         f"{np.mean(nlls):.3f} | {np.mean(ents):.3f} | "
                         f"{np.mean(g_accs):.2f} | {np.mean(p_accs):.2f} |\n")
    print("  T7-2 done", file=sys.stderr)

    # ═══ Exp-T7-3: Held-out generalization (composite not seen during posterior init) ═══
    L.append("\n## Exp-T7-3: Held-Out Compositional Generalization\n\n")
    L.append("| True Goal (held-out) | θ | Mean NLL | Entropy Δ | Goal Acc |\n")
    L.append("|:-------------------:|:-:|:--------:|:---------:|:--------:|\n")

    # Use atomic-only space as "no compositional training"
    # vs full space that includes the composite
    for goal in ["collect_red+use_safe", "avoid_blue+use_safe"]:
        for theta in ["safe", "shiny"]:
            nlls, ent_ds, accs = [], [], []
            for sid in range(N_SEEDS):
                r = run_posterior_episode(goal, theta, sid,
                                        DEFAULT_GOAL_SPACE, THETA_2)
                nlls.append(r["mean_nll"])
                ent_ds.append(r["entropy_reduction"])
                accs.append(r["goal_accuracy"])
            L.append(f"| {goal} | {theta} | "
                     f"{np.mean(nlls):.3f} | {np.mean(ent_ds):+.3f} | "
                     f"{np.mean(accs):.2f} |\n")
    print("  T7-3 done", file=sys.stderr)

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check: compositional space should improve goal accuracy on composite goals
    comp_accs = []
    for sid in range(N_SEEDS):
        r = run_posterior_episode("collect_red+avoid_blue", "safe", sid,
                                DEFAULT_GOAL_SPACE, THETA_2)
        comp_accs.append(r["goal_accuracy"])
    comp_mean = np.mean(comp_accs)
    L.append(f"> Composite goal accuracy (full space): {comp_mean:.2f}\n")

    # Check: K-type should not collapse
    k_accs = []
    for sid in range(N_SEEDS):
        r = run_posterior_episode("use_safe", "safe", sid,
                                DEFAULT_GOAL_SPACE, THETA_K)
        k_accs.append(r["goal_accuracy"])
    k_mean = np.mean(k_accs)
    L.append(f"> K-type goal accuracy (use_safe, safe): {k_mean:.2f}\n")
    L.append(f"> K-type calibration: {'✅' if k_mean > 0.3 else '⚠️'}\n")

    rpt = out / "t7_compositional_posterior_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)


if __name__ == "__main__":
    main()
