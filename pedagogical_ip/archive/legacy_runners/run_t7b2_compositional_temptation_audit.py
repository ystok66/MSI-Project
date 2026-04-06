"""T7-B2: Compositional + Temptation Posterior Audit.

Exp-T7-B2.1: q(g,θ,z) stability under compositional ambiguity
Exp-T7-B2.2: q(g,θ) vs q(g,θ,z) ablation
Exp-T7-B2.3: Θ₂ vs Θ_K under compositional + temptation

Uses subgoal marginals as primary metric, exact composite top-1 as secondary.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.world_state import WorldState
from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, THETA_2, THETA_K,
    DEFAULT_TEMPT_GRID, DEFAULT_TEMPT_PRIOR,
)
from src.teachers.composite_goal_compatibility import (
    CompositeGoalCompatibility, CompatibilityConfig,
)

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1)
COMPAT_CFG = CompatibilityConfig(beta_compat=0.5, lambda_comp=0.3, lambda_redund=0.2)
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


def simulate_tempt_action(goal_label, theta, z_true, branches, rng):
    """Agent acts: goal-conditioned utility + local temptation bias."""
    gh = DEFAULT_GOAL_SPACE.get(goal_label)
    # Boost temptation on risky branch (branch 1)
    mod = list(branches)
    if z_true > 0.01:
        mod[1] = BranchAttributes(
            safety_score=branches[1].safety_score,
            temptation_score=branches[1].temptation_score + z_true,
            texture_novelty=branches[1].texture_novelty,
            shortcut_bonus=branches[1].shortcut_bonus,
            risk_penalty=branches[1].risk_penalty,
        )
    probs = DEFAULT_GOAL_SPACE.compute_choice_probs(mod, gh, theta, AP)
    return int(rng.choice(len(branches), p=probs))


def compute_subgoal_marginals(posterior, compat=None):
    """Compute q(u) = Σ_{g∋u} q(g)."""
    mg = posterior.marginal_goal()
    if compat is not None:
        return compat.subgoal_marginals(mg)
    sm = {}
    for agh in DEFAULT_GOAL_SPACE.atomic_goals:
        total = sum(w for gl, w in mg.items()
                    if agh.label in DEFAULT_GOAL_SPACE.get(gl).components)
        sm[agh.label] = total
    return sm


def run_episode(goal_label, theta, z_true, seed,
                use_tempt=False, use_compat=True, pref_types=THETA_2):
    rng = np.random.default_rng(seed)
    compat = CompositeGoalCompatibility(config=COMPAT_CFG) if use_compat else None
    tempt_grid = DEFAULT_TEMPT_GRID if use_tempt else None
    tempt_prior = DEFAULT_TEMPT_PRIOR if use_tempt else None

    jgpp = JointGoalPrefPosterior(
        pref_types=pref_types,
        tempt_grid=tempt_grid,
        tempt_prior=tempt_prior,
        forgetting_rate=0.005,
        compatibility=compat,
    )
    ws = WorldState()
    true_gh = DEFAULT_GOAL_SPACE.get(goal_label)
    true_comp = set(true_gh.components)

    nlls, goal_corr, subgoal_accs, tempt_ests = [], [], [], []
    entropies = [jgpp.entropy()]

    for step_i in range(N_STEPS):
        branches = make_goal_branches(goal_label, rng)
        ac = simulate_tempt_action(goal_label, theta, z_true, branches, rng)

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
        goal_corr.append(1 if jgpp.predicted_goal() == goal_label else 0)

        # Subgoal marginal
        sm = compute_subgoal_marginals(jgpp, compat)
        top2 = sorted(sm.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_l = {x[0] for x in top2}
        hit = len(true_comp & top2_l) / len(true_comp)
        subgoal_accs.append(hit)

        # Temptation estimate
        if use_tempt:
            mt = jgpp.marginal_tempt()
            ez = sum(z * p for z, p in mt.items())
            tempt_ests.append(ez)

    result = {
        "goal_acc": np.mean(goal_corr[-5:]),
        "subgoal_acc": np.mean(subgoal_accs[-5:]),
        "mean_nll": np.mean(nlls),
        "entropy_delta": entropies[0] - entropies[-1],
        "final_entropy": entropies[-1],
        "predicted_pref": jgpp.predicted_pref(),
    }
    if use_tempt and tempt_ests:
        result["E_z"] = tempt_ests[-1]
        result["MAP_z"] = max(jgpp.marginal_tempt().items(), key=lambda x: x[1])[0]
    return result


def main():
    print("═══ T7-B2: Compositional + Temptation Audit ═══\n", file=sys.stderr)
    L = ["# T7-B2: Compositional + Temptation Posterior\n\n"]

    test_goals = ["collect_red", "use_safe",
                  "collect_red+avoid_blue", "avoid_blue+use_safe",
                  "reach_fast+avoid_blue"]

    # ═══ B2.1: q(g,θ,z) basic stability ═══
    L.append("## Exp-T7-B2.1: q(g,θ,z) Stability\n\n")
    L.append("| Goal | θ | z_true | NLL | Subgoal Acc | Goal Acc | E[z] | Entropy Δ |\n")
    L.append("|:----:|:-:|:------:|:---:|:----------:|:--------:|:----:|:---------:|\n")

    for goal in test_goals:
        for theta in ["safe", "shiny"]:
            for z_true in [0.0, 0.6]:
                nlls, sas, gas, ezs, eds = [], [], [], [], []
                for sid in range(N_SEEDS):
                    r = run_episode(goal, theta, z_true, sid, use_tempt=True)
                    nlls.append(r["mean_nll"])
                    sas.append(r["subgoal_acc"])
                    gas.append(r["goal_acc"])
                    ezs.append(r.get("E_z", 0.0))
                    eds.append(r["entropy_delta"])
                L.append(f"| {goal} | {theta} | {z_true} | "
                         f"{np.mean(nlls):.3f} | {np.mean(sas):.2f} | "
                         f"{np.mean(gas):.2f} | {np.mean(ezs):.2f} | "
                         f"{np.mean(eds):+.3f} |\n")
    print("  B2.1 done", file=sys.stderr)

    # ═══ B2.2: q(g,θ) vs q(g,θ,z) ablation ═══
    L.append("\n## Exp-T7-B2.2: q(g,θ) vs q(g,θ,z) Ablation\n\n")
    L.append("| Goal | θ | z_true | Tempt? | NLL | Subgoal Acc | Goal Acc |\n")
    L.append("|:----:|:-:|:------:|:------:|:---:|:----------:|:--------:|\n")

    for goal in ["collect_red+avoid_blue", "reach_fast+avoid_blue"]:
        for theta in ["shiny"]:
            for z_true in [0.0, 0.6, 0.9]:
                for use_t, label in [(False, "No"), (True, "Yes")]:
                    nlls, sas, gas = [], [], []
                    for sid in range(N_SEEDS):
                        r = run_episode(goal, theta, z_true, sid, use_tempt=use_t)
                        nlls.append(r["mean_nll"])
                        sas.append(r["subgoal_acc"])
                        gas.append(r["goal_acc"])
                    L.append(f"| {goal} | {theta} | {z_true} | {label} | "
                             f"{np.mean(nlls):.3f} | {np.mean(sas):.2f} | "
                             f"{np.mean(gas):.2f} |\n")
    print("  B2.2 done", file=sys.stderr)

    # ═══ B2.3: Θ₂ vs Θ_K under compositional + temptation ═══
    L.append("\n## Exp-T7-B2.3: Θ₂ vs Θ_K under Compositional + Temptation\n\n")
    L.append("| Goal | θ | z | Types | NLL | Subgoal | Goal | E[z] |\n")
    L.append("|:----:|:-:|:-:|:-----:|:---:|:------:|:----:|:----:|\n")

    for goal in ["collect_red", "collect_red+avoid_blue"]:
        for theta in ["safe", "shiny"]:
            z_true = 0.3
            for types_name, types in [("Θ₂", THETA_2), ("Θ_K", THETA_K)]:
                nlls, sas, gas, ezs = [], [], [], []
                for sid in range(N_SEEDS):
                    r = run_episode(goal, theta, z_true, sid,
                                   use_tempt=True, pref_types=types)
                    nlls.append(r["mean_nll"])
                    sas.append(r["subgoal_acc"])
                    gas.append(r["goal_acc"])
                    ezs.append(r.get("E_z", 0.0))
                L.append(f"| {goal} | {theta} | {z_true} | {types_name} | "
                         f"{np.mean(nlls):.3f} | {np.mean(sas):.2f} | "
                         f"{np.mean(gas):.2f} | {np.mean(ezs):.2f} |\n")
    print("  B2.3 done", file=sys.stderr)

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check: temptation helps explain deviations
    for z_true in [0.0, 0.6]:
        nlls_noT, nlls_T = [], []
        for sid in range(N_SEEDS):
            r0 = run_episode("collect_red+avoid_blue", "shiny", z_true, sid, use_tempt=False)
            r1 = run_episode("collect_red+avoid_blue", "shiny", z_true, sid, use_tempt=True)
            nlls_noT.append(r0["mean_nll"])
            nlls_T.append(r1["mean_nll"])
        L.append(f"> z={z_true}: q(g,θ) NLL={np.mean(nlls_noT):.3f} | "
                 f"q(g,θ,z) NLL={np.mean(nlls_T):.3f} | "
                 f"Δ={np.mean(nlls_T)-np.mean(nlls_noT):+.3f}\n")

    # Subgoal stability
    sas_all = []
    for goal in test_goals:
        for sid in range(N_SEEDS):
            r = run_episode(goal, "safe", 0.3, sid, use_tempt=True)
            sas_all.append(r["subgoal_acc"])
    L.append(f"> Overall subgoal marginal acc (q(g,θ,z)): {np.mean(sas_all):.3f}\n")

    # Collapse check
    entropies = []
    for goal in test_goals:
        for sid in range(5):
            r = run_episode(goal, "safe", 0.6, sid, use_tempt=True)
            entropies.append(r["final_entropy"])
    L.append(f"> Mean final entropy (should be >0): {np.mean(entropies):.3f}\n")
    L.append(f"> Collapse check: {'✅ No collapse' if np.mean(entropies) > 0.1 else '⚠️ Possible collapse'}\n")

    rpt = out / "t7b2_compositional_temptation_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)


if __name__ == "__main__":
    main()
