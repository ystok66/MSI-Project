"""T5 Exp-T5-1: Hidden Temptation Audit.

Tests GoalTemptationPosterior resilience when agent has private temptation
signal hidden from the robot.

Metrics:
1. Action NLL by true z_tempt
2. Posterior calibration (does q(z) recover true z?)
3. Posterior entropy reduction over time
4. Expected temptation accuracy
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.agent_belief_state import AgentBelief
from src.agents.world_state import WorldState
from src.teachers.action_predictor import ActionPredictor
from src.teachers.goal_temptation_posterior import GoalTemptationPosterior

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1)

TEMPT_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_STEPS = 20
N_SEEDS = 10


def run_hidden_temptation_episode(theta, seed, true_z_tempt):
    """Run episode with agent having hidden temptation z_tempt."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    ap = ActionPredictor(AP)
    gtp = GoalTemptationPosterior(
        goals=("true_goal",),  # single goal for temptation-only test
        action_predictor=ap)

    # Agent sees temptation-boosted branches
    base_branches = [
        BranchAttributes(safety_score=0.8, risk_penalty=0.15),
        BranchAttributes(safety_score=0.3, risk_penalty=0.4, temptation_score=0.2),
    ]

    nlls = []
    entropies = [gtp.entropy()]
    expected_tempts = [gtp.expected_tempt()]

    for step_i in range(N_STEPS):
        ws = WorldState(t=step_i, t_max=N_STEPS)
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)

        # Agent's TRUE branches: temptation boosted by z_tempt
        agent_branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.15),
            BranchAttributes(safety_score=0.3, risk_penalty=0.4,
                           temptation_score=0.2 + true_z_tempt),
        ]

        # Agent chooses (with hidden temptation)
        ac = sample_factored_choice(agent_branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])

        # Robot observes action, updates posterior using BASE branches
        # (robot doesn't know about z_tempt)
        nll = ap.nll(ws, ab, base_branches, ac)
        nlls.append(nll)

        gtp.update(ws, base_branches, ac, ab, risky_branch_idx=1)
        entropies.append(gtp.entropy())
        expected_tempts.append(gtp.expected_tempt())

        # Update learner state
        correct = (ac == 0)
        m.update_risk(0.4 if not correct else 0.05, 0.15)
        m.snapshot()

    # Final posterior
    mt = gtp.marginal_tempt()

    return {
        "mean_nll": np.mean(nlls),
        "final_entropy": entropies[-1],
        "entropy_reduction": entropies[0] - entropies[-1],
        "expected_tempt_final": expected_tempts[-1],
        "marginal_tempt": mt,
        "true_z": true_z_tempt,
    }


def main():
    print("═══ T5 Hidden Temptation Audit ═══\n", file=sys.stderr)
    L = ["# T5 Exp-T5-1: Hidden Temptation Audit\n\n"]

    # ═══ Table 1: NLL by z_tempt ═══
    L.append("## Action NLL by True Temptation Level\n\n")
    L.append("| θ | True z | Mean NLL | Entropy Δ | E[z]_final | MAP z |\n")
    L.append("|:-:|:------:|:--------:|:---------:|:----------:|:-----:|\n")

    for th in ["safe", "shiny"]:
        for z in TEMPT_LEVELS:
            nlls, ent_deltas, ez_finals, map_zs = [], [], [], []
            for sid in range(N_SEEDS):
                r = run_hidden_temptation_episode(th, sid, z)
                nlls.append(r["mean_nll"])
                ent_deltas.append(r["entropy_reduction"])
                ez_finals.append(r["expected_tempt_final"])
                # MAP z
                mt = r["marginal_tempt"]
                map_z = max(mt, key=mt.get)
                map_zs.append(map_z)

            most_common_map = max(set(map_zs), key=map_zs.count)
            L.append(f"| {th} | {z:.1f} | {np.mean(nlls):.3f} | "
                     f"{np.mean(ent_deltas):+.3f} | {np.mean(ez_finals):.3f} | "
                     f"{most_common_map:.1f} |\n")
        print(f"  {th} done", file=sys.stderr)

    # ═══ Table 2: Posterior calibration ═══
    L.append("\n## Posterior Calibration: P(MAP = true z)\n\n")
    L.append("| θ | True z | P(MAP=true) | P(MAP=±0.3) |\n")
    L.append("|:-:|:------:|:-----------:|:-----------:|\n")

    for th in ["safe", "shiny"]:
        for z in TEMPT_LEVELS:
            exact_match = 0
            near_match = 0
            for sid in range(N_SEEDS):
                r = run_hidden_temptation_episode(th, sid, z)
                mt = r["marginal_tempt"]
                map_z = max(mt, key=mt.get)
                if abs(map_z - z) < 0.01:
                    exact_match += 1
                if abs(map_z - z) <= 0.31:
                    near_match += 1
            L.append(f"| {th} | {z:.1f} | {exact_match/N_SEEDS:.2f} | "
                     f"{near_match/N_SEEDS:.2f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check: NLL should worsen smoothly with z_tempt
    nll_by_z_shiny = []
    for z in TEMPT_LEVELS:
        ns = [run_hidden_temptation_episode("shiny", s, z)["mean_nll"]
              for s in range(N_SEEDS)]
        nll_by_z_shiny.append(np.mean(ns))
    smooth = all(nll_by_z_shiny[i] <= nll_by_z_shiny[i+1] + 0.1
                 for i in range(len(nll_by_z_shiny)-1))
    L.append(f"> NLL worsens smoothly with z_tempt (shiny): {'✅' if smooth else '⚠️'}\n")

    # Check: entropy reduces
    for th in ["safe", "shiny"]:
        ent_deltas = [run_hidden_temptation_episode(th, 0, 0.6)["entropy_reduction"]]
        reduces = ent_deltas[0] > 0
        L.append(f"> Entropy reduces for {th} θ, z=0.6: {'✅' if reduces else '❌'}\n")

    rpt = out / "t5_hidden_temptation_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
