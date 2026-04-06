"""Step 4.5: Prior Promotion Audit — 5 decisive experiments.

Exp A: Longer-horizon recovery (n_steps=5,10,20,30,50)
Exp B: Leave-one-feature-out prior ablation
Exp C: Full CGC-v2 integration test
Exp D: Θ₂ vs Θ_K held-out promotion test
Exp E: Subgoal calibration audit (ECE/Brier/reliability)

Usage:
  python scripts/run_step4_5_promotion_audit.py
  python scripts/run_step4_5_promotion_audit.py --n_seeds 50
"""

from __future__ import annotations
import sys, os, argparse, time
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
from src.teachers.composite_goal_compatibility import (
    CompositeGoalCompatibility, CompatibilityConfig,
)
from src.teachers.compositional_goal_bridge import CompositionalGoalBridge
from src.teachers.action_predictor import ActionPredictor
from src.agents.agent_belief_state import AgentBelief

AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# ═════════════════════════════════════════════════════════
# Scenarios (reused from Step 4)
# ═════════════════════════════════════════════════════════

SCENARIOS = {
    "goal_aligned": {
        "branches": [
            BranchAttributes(safety_score=0.8, temptation_score=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.2, temptation_score=0.5, risk_penalty=0.4),
        ],
        "true_goal": "avoid_blue", "true_theta": "safe",
    },
    "goal_conflict": {
        "branches": [
            BranchAttributes(safety_score=0.6, temptation_score=0.3, risk_penalty=0.2),
            BranchAttributes(safety_score=0.5, temptation_score=0.4, risk_penalty=0.3),
        ],
        "true_goal": "collect_red+avoid_blue", "true_theta": "safe",
    },
    "temptation_hard": {
        "branches": [
            BranchAttributes(safety_score=0.7, temptation_score=0.1, risk_penalty=0.15),
            BranchAttributes(safety_score=0.3, temptation_score=0.8, risk_penalty=0.35),
        ],
        "true_goal": "avoid_blue+use_safe", "true_theta": "shiny",
    },
    "shortcut": {
        "branches": [
            BranchAttributes(safety_score=0.5, temptation_score=0.0,
                             shortcut_bonus=0.0, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, temptation_score=0.2,
                             shortcut_bonus=0.6, risk_penalty=0.2),
        ],
        "true_goal": "reach_fast+avoid_blue", "true_theta": "safe",
    },
}


def make_posterior(prior_mode, theta_mode="2", beta_len=1.0, beta_red=0.5,
                   compat=None):
    pref_types = THETA_2 if theta_mode == "2" else THETA_K
    ctx = GoalPriorContext()
    cfg = GoalPriorConfig(beta_len=beta_len, beta_red=beta_red)
    return JointGoalPrefPosterior(
        pref_types=pref_types, params=AP, prior_mode=prior_mode,
        prior_context=ctx, prior_config=cfg, compatibility=compat)


def run_episode_tracked(prior_mode, scenario_name, seed, n_steps=10,
                        theta_mode="2", beta_len=1.0, beta_red=0.5):
    """Run episode and track per-step metrics."""
    sc = SCENARIOS[scenario_name]
    branches = sc["branches"]
    true_goal, true_theta = sc["true_goal"], sc["true_theta"]
    rng = np.random.default_rng(seed)
    true_gh = DEFAULT_GOAL_SPACE.get(true_goal)
    true_comps = set(true_gh.components)

    # Legacy+bonus needs compatibility tracker
    compat = None
    if prior_mode == "legacy_bonus":
        compat = CompositeGoalCompatibility(params=AP)

    post = make_posterior(prior_mode, theta_mode, beta_len, beta_red, compat)

    per_step = []
    for step in range(n_steps):
        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
            branches, true_gh, true_theta, AP)
        action = int(rng.choice(len(branches), p=probs))

        # Pre-update metrics
        mg = post.marginal_goal()
        sm = post.subgoal_marginals()
        sm_acc = sum(sm.get(u, 0.0) for u in true_comps) / len(true_comps)

        # Predictive NLL
        mp = post.marginal_pref()
        pred_p = 0.0
        for gl, gw in mg.items():
            gh = DEFAULT_GOAL_SPACE.get(gl)
            for th, tw in mp.items():
                cp = DEFAULT_GOAL_SPACE.compute_choice_probs(
                    branches, gh, th, AP)
                pred_p += gw * tw * cp[action]
        nll = -np.log(max(pred_p, 1e-15))

        # FCI
        fci = sum(gw for gl, gw in mg.items()
                  if DEFAULT_GOAL_SPACE.get(gl).is_composite and gl != true_goal)

        per_step.append({
            "step": step, "sm_acc": sm_acc, "nll": nll,
            "fci": fci, "entropy": post.entropy(),
            "subgoal_marginals": dict(sm),
        })

        post.update(None, branches, action)

    # Final step
    mg = post.marginal_goal()
    sm = post.subgoal_marginals()
    sm_acc = sum(sm.get(u, 0.0) for u in true_comps) / len(true_comps)
    fci = sum(gw for gl, gw in mg.items()
              if DEFAULT_GOAL_SPACE.get(gl).is_composite and gl != true_goal)
    per_step.append({
        "step": n_steps, "sm_acc": sm_acc, "nll": 0.0,
        "fci": fci, "entropy": post.entropy(),
        "subgoal_marginals": dict(sm),
    })

    return per_step


# ═════════════════════════════════════════════════════════
# Exp A: Longer Horizon Recovery
# ═════════════════════════════════════════════════════════

def exp_a_longer_horizon(n_seeds, lines):
    lines.append("## Exp A: Longer-Horizon Recovery\n\n")
    print("Exp A: Longer-horizon recovery...", file=sys.stderr)

    horizons = [5, 10, 20, 30, 50]
    variants = ["legacy_bonus", "structural", "pcfg"]
    scenarios = list(SCENARIOS.keys())

    lines.append("| Variant | Scenario | t | SM_Acc | FCI | Entropy |\n")
    lines.append("|---------|----------|---|--------|-----|--------|\n")

    for variant in variants:
        for scenario in scenarios:
            for n_steps in horizons:
                sm_accs, fcis, entropies = [], [], []
                for seed in range(n_seeds):
                    trace = run_episode_tracked(
                        variant, scenario, seed, n_steps)
                    final = trace[-1]
                    sm_accs.append(final["sm_acc"])
                    fcis.append(final["fci"])
                    entropies.append(final["entropy"])

                sm_a = np.mean(sm_accs)
                fci_m = np.mean(fcis)
                ent_m = np.mean(entropies)
                lines.append(f"| {variant} | {scenario} | {n_steps} | "
                             f"{sm_a:.3f} | {fci_m:.3f} | {ent_m:.3f} |\n")
                if n_steps in (10, 50):
                    print(f"  {variant}/{scenario} t={n_steps}: "
                          f"SM={sm_a:.3f} FCI={fci_m:.3f} H={ent_m:.3f}",
                          file=sys.stderr)

    lines.append("\n")


# ═════════════════════════════════════════════════════════
# Exp B: Leave-One-Feature-Out Prior Ablation
# ═════════════════════════════════════════════════════════

def exp_b_feature_ablation(n_seeds, lines):
    lines.append("## Exp B: Leave-One-Feature-Out Prior Ablation\n\n")
    print("\nExp B: Feature ablation...", file=sys.stderr)

    configs = {
        "complexity_only":    (1.0, 0.0),
        "redundancy_only":    (0.0, 0.5),
        "complexity+redund":  (1.0, 0.5),
        "neither (uniform)":  (0.0, 0.0),
    }
    scenarios = list(SCENARIOS.keys())

    lines.append("| Config | Scenario | NLL | SM_Acc | FCI | Entropy |\n")
    lines.append("|--------|----------|-----|--------|-----|--------|\n")

    for cfg_name, (bl, br) in configs.items():
        for scenario in scenarios:
            nlls, sm_accs, fcis, ents = [], [], [], []
            for seed in range(n_seeds):
                trace = run_episode_tracked(
                    "structural", scenario, seed, 10, "2", bl, br)
                # Average NLL across steps
                step_nlls = [s["nll"] for s in trace[:-1]]
                nlls.append(np.mean(step_nlls))
                final = trace[-1]
                sm_accs.append(final["sm_acc"])
                fcis.append(final["fci"])
                ents.append(final["entropy"])

            lines.append(f"| {cfg_name} | {scenario} | {np.mean(nlls):.3f} | "
                         f"{np.mean(sm_accs):.3f} | {np.mean(fcis):.3f} | "
                         f"{np.mean(ents):.3f} |\n")
            print(f"  {cfg_name}/{scenario}: NLL={np.mean(nlls):.3f} "
                  f"SM={np.mean(sm_accs):.3f} FCI={np.mean(fcis):.3f}",
                  file=sys.stderr)

    lines.append("\n")


# ═════════════════════════════════════════════════════════
# Exp C: Full CGC-v2 Integration Test
# ═════════════════════════════════════════════════════════

def exp_c_cgc_integration(n_seeds, lines):
    lines.append("## Exp C: Full CGC-v2 Integration Test\n\n")
    print("\nExp C: CGC-v2 integration...", file=sys.stderr)

    ap = ActionPredictor(params=AP)
    bridge = CompositionalGoalBridge(action_predictor=ap)

    # Simulate CGC-v2 episodes through bridge
    scenarios = list(SCENARIOS.keys())
    variants = ["legacy_bonus", "structural"]

    lines.append("| Variant | Scenario | SM_Acc | FCI | Bridge_stable |\n")
    lines.append("|---------|----------|--------|-----|---------------|\n")

    for variant in variants:
        for scenario in scenarios:
            sc = SCENARIOS[scenario]
            branches = sc["branches"]
            true_goal, true_theta = sc["true_goal"], sc["true_theta"]
            true_gh = DEFAULT_GOAL_SPACE.get(true_goal)
            true_comps = set(true_gh.components)

            sm_accs, fcis = [], []
            bridge_ok = True

            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                try:
                    compat = None
                    if variant == "legacy_bonus":
                        compat = CompositeGoalCompatibility(params=AP)

                    post = make_posterior(variant, "2", compat=compat)
                    ab = AgentBelief()

                    for step in range(10):
                        probs = DEFAULT_GOAL_SPACE.compute_choice_probs(
                            branches, true_gh, true_theta, AP)
                        action = int(rng.choice(len(branches), p=probs))
                        bridge.update_posterior(post, None, branches, action, ab)

                    mg = post.marginal_goal()
                    sm = post.subgoal_marginals()
                    sm_acc = sum(sm.get(u, 0.0) for u in true_comps) / len(true_comps)
                    fci = sum(gw for gl, gw in mg.items()
                              if DEFAULT_GOAL_SPACE.get(gl).is_composite and gl != true_goal)
                    sm_accs.append(sm_acc)
                    fcis.append(fci)
                except Exception as e:
                    bridge_ok = False
                    print(f"  Bridge ERROR: {variant}/{scenario}/s{seed}: {e}",
                          file=sys.stderr)

            sm_a = np.mean(sm_accs) if sm_accs else 0.0
            fci_m = np.mean(fcis) if fcis else 0.0
            stable = "YES" if bridge_ok and len(sm_accs) == n_seeds else "NO"
            lines.append(f"| {variant} | {scenario} | {sm_a:.3f} | "
                         f"{fci_m:.3f} | {stable} |\n")
            print(f"  {variant}/{scenario}: SM={sm_a:.3f} FCI={fci_m:.3f} "
                  f"bridge={stable}", file=sys.stderr)

    lines.append("\n")


# ═════════════════════════════════════════════════════════
# Exp D: Θ₂ vs Θ_K Promotion Test
# ═════════════════════════════════════════════════════════

def exp_d_theta_promotion(n_seeds, lines):
    lines.append("## Exp D: Θ₂ vs Θ_K Promotion Test (structural prior)\n\n")
    print("\nExp D: Θ₂ vs Θ_K promotion...", file=sys.stderr)

    scenarios = list(SCENARIOS.keys())

    lines.append("| Θ-mode | Scenario | NLL | SM_Acc | FCI | Entropy |\n")
    lines.append("|--------|----------|-----|--------|-----|--------|\n")

    for theta_mode in ["2", "k"]:
        for scenario in scenarios:
            nlls, sm_accs, fcis, ents = [], [], [], []
            for seed in range(n_seeds):
                trace = run_episode_tracked(
                    "structural", scenario, seed, 10, theta_mode)
                step_nlls = [s["nll"] for s in trace[:-1]]
                nlls.append(np.mean(step_nlls))
                final = trace[-1]
                sm_accs.append(final["sm_acc"])
                fcis.append(final["fci"])
                ents.append(final["entropy"])

            lines.append(f"| Θ{theta_mode} | {scenario} | {np.mean(nlls):.3f} | "
                         f"{np.mean(sm_accs):.3f} | {np.mean(fcis):.3f} | "
                         f"{np.mean(ents):.3f} |\n")
            print(f"  Θ{theta_mode}/{scenario}: NLL={np.mean(nlls):.3f} "
                  f"SM={np.mean(sm_accs):.3f}", file=sys.stderr)

    lines.append("\n")


# ═════════════════════════════════════════════════════════
# Exp E: Subgoal Calibration Audit
# ═════════════════════════════════════════════════════════

def exp_e_calibration(n_seeds, lines):
    lines.append("## Exp E: Subgoal Calibration Audit\n\n")
    print("\nExp E: Calibration audit...", file=sys.stderr)

    variants = ["legacy_bonus", "structural", "pcfg"]
    scenarios = list(SCENARIOS.keys())

    # Collect (predicted_prob, correct) pairs for each variant
    for variant in variants:
        all_preds = []
        all_correct = []

        for scenario in scenarios:
            sc = SCENARIOS[scenario]
            true_goal = sc["true_goal"]
            true_gh = DEFAULT_GOAL_SPACE.get(true_goal)
            true_comps = set(true_gh.components)

            for seed in range(n_seeds):
                trace = run_episode_tracked(
                    variant, scenario, seed, 10)
                final = trace[-1]
                sm = final["subgoal_marginals"]

                for u in ATOMIC_GOALS:
                    p = sm.get(u, 0.0)
                    c = 1.0 if u in true_comps else 0.0
                    all_preds.append(p)
                    all_correct.append(c)

        preds = np.array(all_preds)
        correct = np.array(all_correct)

        # Brier score
        brier = float(np.mean((preds - correct) ** 2))

        # ECE (10 bins)
        n_bins = 10
        ece = 0.0
        reliability_rows = []
        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 0:
                avg_pred = float(preds[mask].mean())
                avg_correct = float(correct[mask].mean())
                n_in_bin = int(mask.sum())
                ece += abs(avg_pred - avg_correct) * n_in_bin / len(preds)
                reliability_rows.append((lo, hi, avg_pred, avg_correct, n_in_bin))

        lines.append(f"### {variant}\n\n")
        lines.append(f"**Brier**: {brier:.4f} | **ECE**: {ece:.4f}\n\n")

        lines.append("| Bin | Avg Pred | Avg Correct | N |\n")
        lines.append("|-----|----------|-------------|---|\n")
        for lo, hi, ap, ac, n in reliability_rows:
            lines.append(f"| [{lo:.1f},{hi:.1f}) | {ap:.3f} | {ac:.3f} | {n} |\n")
        lines.append("\n")

        print(f"  {variant}: Brier={brier:.4f} ECE={ece:.4f}",
              file=sys.stderr)


# ═════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    args = parser.parse_args()

    out = Path("results/step4_prior")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lines = ["# Step 4.5: Promotion & Audit Report\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds} | ")

    # Run all experiments
    exp_a_longer_horizon(args.n_seeds, lines)
    exp_b_feature_ablation(args.n_seeds, lines)
    exp_c_cgc_integration(args.n_seeds, lines)
    exp_d_theta_promotion(args.n_seeds, lines)
    exp_e_calibration(args.n_seeds, lines)

    elapsed = time.time() - t0
    lines[1] += f"**Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step4_5_promotion_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt}", file=sys.stderr)
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
