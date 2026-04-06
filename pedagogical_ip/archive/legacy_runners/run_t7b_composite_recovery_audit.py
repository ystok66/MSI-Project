"""T7-B Exp-T7-B1: Composite Recovery with Compatibility Prior.

Compares three configurations:
1. baseline q(g,θ) — no compatibility
2. q(g,θ) + compatibility prior
3. q(g,θ) + compatibility + complexity penalty

Also reports subgoal marginal accuracy.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.world_state import WorldState
from src.teachers.compositional_goal_hypotheses import (
    GoalHypothesisSpace, DEFAULT_GOAL_SPACE,
)
from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, THETA_2,
)
from src.teachers.composite_goal_compatibility import (
    CompositeGoalCompatibility, CompatibilityConfig,
)

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1)
N_STEPS = 20
N_SEEDS = 15


def make_goal_branches(goal_label, rng):
    gh = DEFAULT_GOAL_SPACE.get(goal_label)
    w = gh.reward_weights
    b0 = BranchAttributes(
        safety_score=max(0.1, min(0.95, 0.5 + w[0] * 0.15)),
        temptation_score=max(0.0, min(0.95, 0.3 + w[1] * 0.1)),
        texture_novelty=max(0.0, min(0.95, 0.3 + w[2] * 0.1)),
        shortcut_bonus=max(0.0, min(0.95, w[3] * 0.15)),
        risk_penalty=max(0.0, 0.15 - w[0] * 0.05 + rng.uniform(-0.02, 0.02)),
    )
    b1 = BranchAttributes(
        safety_score=max(0.1, min(0.95, 0.5 - w[0] * 0.15)),
        temptation_score=max(0.0, min(0.95, 0.3 - w[1] * 0.05 + rng.uniform(0, 0.2))),
        texture_novelty=max(0.0, min(0.95, 0.3 - w[2] * 0.05)),
        shortcut_bonus=max(0.0, min(0.95, 0.1)),
        risk_penalty=max(0.0, 0.25 + w[0] * 0.05 + rng.uniform(-0.02, 0.02)),
    )
    return [b0, b1]


def run_episode(goal_label, theta, seed, compat_config=None):
    rng = np.random.default_rng(seed)
    compat = None
    if compat_config is not None:
        compat = CompositeGoalCompatibility(config=compat_config)
    jgpp = JointGoalPrefPosterior(
        pref_types=THETA_2, forgetting_rate=0.005, compatibility=compat)
    ws = WorldState()

    goal_correct = []
    subgoal_hits = []
    nlls = []
    entropies = [jgpp.entropy()]

    true_gh = DEFAULT_GOAL_SPACE.get(goal_label)
    true_components = set(true_gh.components)

    for step_i in range(N_STEPS):
        branches = make_goal_branches(goal_label, rng)
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
            branches, true_gh, theta, AP)
        ac = int(rng.choice(len(branches), p=probs))

        # NLL
        mg = jgpp.marginal_goal()
        mp = jgpp.marginal_pref()
        pred_p = 0.0
        for gl, gw in mg.items():
            for tl, tw in mp.items():
                gh2 = DEFAULT_GOAL_SPACE.get(gl)
                cp = DEFAULT_GOAL_SPACE.compute_choice_probs(branches, gh2, tl, AP)
                pred_p += gw * tw * cp[ac]
        nlls.append(-np.log(max(pred_p, 1e-10)))

        jgpp.update(ws, branches, ac)
        entropies.append(jgpp.entropy())
        goal_correct.append(1 if jgpp.predicted_goal() == goal_label else 0)

        # Subgoal marginal recovery
        if compat is not None:
            sm = compat.subgoal_marginals(jgpp.marginal_goal())
        else:
            # Manual subgoal marginal
            sm = {}
            for agh in DEFAULT_GOAL_SPACE.atomic_goals:
                total = 0.0
                for gh2 in DEFAULT_GOAL_SPACE.hypotheses:
                    if agh.label in gh2.components:
                        total += mg.get(gh2.label, 0.0)
                sm[agh.label] = total

        top2_subgoals = sorted(sm.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_labels = {x[0] for x in top2_subgoals}
        hit = len(true_components & top2_labels) / len(true_components)
        subgoal_hits.append(hit)

    return {
        "goal_acc": np.mean(goal_correct[-5:]),
        "subgoal_acc": np.mean(subgoal_hits[-5:]),
        "mean_nll": np.mean(nlls),
        "entropy_delta": entropies[0] - entropies[-1],
        "final_entropy": entropies[-1],
    }


def main():
    print("═══ T7-B1: Composite Recovery Audit ═══\n", file=sys.stderr)
    L = ["# T7-B Exp-T7-B1: Composite Recovery with Compatibility Prior\n\n"]

    configs = {
        "baseline": None,
        "compat": CompatibilityConfig(beta_compat=0.5, lambda_comp=0.0, lambda_redund=0.0),
        "compat+penalty": CompatibilityConfig(beta_compat=0.5, lambda_comp=0.3, lambda_redund=0.2),
        "strong_compat": CompatibilityConfig(beta_compat=1.0, lambda_comp=0.4, lambda_redund=0.3),
    }

    test_goals = ["collect_red", "use_safe",
                  "collect_red+avoid_blue", "collect_red+use_safe",
                  "avoid_blue+use_safe", "reach_fast+avoid_blue"]

    L.append("## Exact Goal Recovery + Subgoal Marginal\n\n")
    L.append("| Goal | θ | Config | Goal Acc | Subgoal Acc | NLL | Entropy Δ |\n")
    L.append("|:----:|:-:|:------:|:--------:|:----------:|:---:|:---------:|\n")

    for goal in test_goals:
        for theta in ["safe", "shiny"]:
            for cfg_name, cfg in configs.items():
                gas, sas, nlls, eds = [], [], [], []
                for sid in range(N_SEEDS):
                    r = run_episode(goal, theta, sid, cfg)
                    gas.append(r["goal_acc"])
                    sas.append(r["subgoal_acc"])
                    nlls.append(r["mean_nll"])
                    eds.append(r["entropy_delta"])

                is_best = cfg_name in ("compat+penalty", "strong_compat")
                style = "**" if is_best else ""
                L.append(
                    f"| {goal} | {theta} | {style}{cfg_name}{style} | "
                    f"{np.mean(gas):.2f} | {np.mean(sas):.2f} | "
                    f"{np.mean(nlls):.3f} | {np.mean(eds):+.3f} |\n"
                )
        print(f"  {goal} done", file=sys.stderr)

    # Verdict
    L.append("\n## Verdict\n\n")

    # Compare baseline vs compat+penalty on composites
    comp_goals = [g for g in test_goals if "+" in g]
    for cfg_name in ["baseline", "compat+penalty"]:
        cfg = configs[cfg_name]
        accs = []
        for goal in comp_goals:
            for sid in range(N_SEEDS):
                r = run_episode(goal, "safe", sid, cfg)
                accs.append(r["goal_acc"])
        L.append(f"> {cfg_name} composite goal_acc: {np.mean(accs):.3f}\n")

    # Subgoal marginals
    for cfg_name in ["baseline", "compat+penalty"]:
        cfg = configs[cfg_name]
        sas = []
        for goal in comp_goals:
            for sid in range(N_SEEDS):
                r = run_episode(goal, "safe", sid, cfg)
                sas.append(r["subgoal_acc"])
        L.append(f"> {cfg_name} composite subgoal_acc: {np.mean(sas):.3f}\n")

    rpt = out / "t7b_composite_recovery_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)


if __name__ == "__main__":
    main()
