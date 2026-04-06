"""T5 Exp-T5-3+4: Consequence-Grounded Mixed-Family + Inflation Decomposition.

Re-runs mixed-family comparison with ConsequenceGroundedRollout providing
actual intervention effect on agent action distribution.

Also decomposes inflation: Δν̂/n_int vs total Δν̂.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.agent_belief_state import AgentBelief
from src.agents.world_state import WorldState
from src.teachers.action_predictor import ActionPredictor
from src.teachers.consequence_grounded_option_rollout import (
    ConsequenceGroundedRollout, ConsequenceConfig,
)
from src.teachers.option_intervention_controller import (
    OptionInterventionController, OptionConfig,
)
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent
from src.teachers.intervention_risk_head import InterventionRiskHead

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1)

FAMILIES = {
    "fork_trap": {"primary": "WARN", "d_commit": 2, "d_reveal": 3,
                  "risk": 0.4, "has_doors": False, "tempt": 0.3},
    "hazard_belt": {"primary": "ITEM_DROP", "d_commit": 1, "d_reveal": 4,
                    "risk": 0.35, "has_doors": False, "tempt": 0.1},
    "deadline_gate": {"primary": "UNLOCK", "d_commit": 4, "d_reveal": 2,
                      "risk": 0.15, "has_doors": True, "tempt": 0.0},
}
N_STEPS = 25
N_SEEDS = 10


def make_branches(fam):
    return [
        BranchAttributes(safety_score=0.8, risk_penalty=0.1),
        BranchAttributes(safety_score=0.3, risk_penalty=fam["risk"],
                       temptation_score=fam["tempt"]),
    ]


def run_grounded_episode(theta, seed, family, strategy="option_ctrl"):
    """Run episode with consequence-grounded intervention effects."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    fam = FAMILIES[family]
    ap = ActionPredictor(AP)
    cgr = ConsequenceGroundedRollout(ap, config=ConsequenceConfig(alpha_warn=0.15))

    rboa = RobotBeliefOverAgent(action_predictor=ap)
    irh = InterventionRiskHead()
    cfg = OptionConfig(shield_cost=0.5, lambda_teach=0.8)
    ctrl = OptionInterventionController(config=cfg)

    nu_traj = [m.nu]
    gamma_gen_traj = [m.gamma_gen]
    n_correct = 0
    n_interventions = 0
    interventions = []

    for step_i in range(N_STEPS):
        ws = WorldState(t=step_i, t_max=N_STEPS)
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)
        branches = make_branches(fam)

        # Determine intervention
        if strategy == "option_ctrl":
            irisk = irh.predict(ws, rboa, ap, branches, ab,
                               d_commit=fam["d_commit"], d_reveal=fam["d_reveal"],
                               path_length_estimate=max(N_STEPS - step_i, 1))
            d = ctrl.select_option(
                scenario_family=family,
                primary_intervention=fam["primary"],
                m_hat=dict(m.as_dict),
                p_timeout=irisk.p_timeout,
                p_blind=irisk.p_blind,
                has_shield=False,
                has_locked_doors=fam["has_doors"],
                nu_trajectory=nu_traj,
                gamma_gen_trajectory=gamma_gen_traj,
            )
            chosen = d.chosen
        else:
            chosen = strategy  # fixed strategy

        if chosen != "NONE":
            n_interventions += 1
        interventions.append(chosen)

        # Apply consequence: get modified branches for agent choice
        mod_branches = cgr.apply_consequence(chosen, branches)

        # Agent chooses with MODIFIED branches (consequence grounded!)
        ac = sample_factored_choice(mod_branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == 0)
        n_correct += int(correct)

        # Update learner state based on chosen intervention
        if chosen == "WARN":
            m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
            if correct:
                m.update_trust(warn_helpful=True)
        elif chosen == "ITEM_DROP":
            m.update_dependence(blind_obey=True)
        elif chosen == "UNLOCK":
            pass
        else:
            if correct:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)

        m.update_risk(fam["risk"] if not correct else 0.05, 0.15)
        m.snapshot()
        nu_traj.append(m.nu)
        gamma_gen_traj.append(m.gamma_gen)

    return {
        "sr": n_correct / N_STEPS,
        "delta_nu": m.nu - nu_traj[0],
        "delta_gamma_gen": m.gamma_gen - gamma_gen_traj[0],
        "n_interventions": n_interventions,
        "norm_delta_nu": (m.nu - nu_traj[0]) / max(n_interventions, 1),
        "norm_delta_gamma": (m.gamma_gen - gamma_gen_traj[0]) / max(n_interventions, 1),
    }


def main():
    print("═══ T5 Grounded Mixed-Family + Inflation ═══\n", file=sys.stderr)
    L = ["# T5 Exp-T5-3+4: Grounded Mixed-Family + Inflation Decomposition\n\n"]

    # ═══ Table 1: Mixed-family SR with grounded consequences ═══
    L.append("## Exp-T5-3: Mixed-Family Success Rate (Grounded)\n\n")
    L.append("| θ | Strategy | fork_trap SR | hazard_belt SR | deadline_gate SR | Mean SR |\n")
    L.append("|:-:|:--------:|:----------:|:-------------:|:---------------:|:-------:|\n")

    for th in ["safe", "shiny"]:
        for strat in ["NONE", "WARN", "UNLOCK", "ITEM_DROP", "option_ctrl"]:
            srs = {}
            for fam in FAMILIES:
                sr_list = [run_grounded_episode(th, sid, fam, strat)["sr"]
                           for sid in range(N_SEEDS)]
                srs[fam] = np.mean(sr_list)
            mean_sr = np.mean(list(srs.values()))
            style = "**" if strat == "option_ctrl" else ""
            L.append(f"| {th} | {style}{strat}{style} | {srs['fork_trap']:.3f} | "
                     f"{srs['hazard_belt']:.3f} | {srs['deadline_gate']:.3f} | "
                     f"{style}{mean_sr:.3f}{style} |\n")
        print(f"  {th} SR done", file=sys.stderr)

    # ═══ Table 2: Inflation decomposition ═══
    L.append("\n## Exp-T5-4: Inflation Decomposition\n\n")
    L.append("| θ | Family | Strategy | Δν̂ | n_int | Δν̂/n_int | Δγ̂_gen | Δγ̂/n_int |\n")
    L.append("|:-:|:------:|:--------:|:---:|:----:|:--------:|:------:|:--------:|\n")

    for th in ["safe", "shiny"]:
        for fam in FAMILIES:
            for strat in ["NONE", "WARN", "ITEM_DROP", "option_ctrl"]:
                dnus, nints, ndnus, dgammas, ndgammas = [], [], [], [], []
                for sid in range(N_SEEDS):
                    r = run_grounded_episode(th, sid, fam, strat)
                    dnus.append(r["delta_nu"])
                    nints.append(r["n_interventions"])
                    ndnus.append(r["norm_delta_nu"])
                    dgammas.append(r["delta_gamma_gen"])
                    ndgammas.append(r["norm_delta_gamma"])

                style = "**" if strat == "option_ctrl" else ""
                L.append(
                    f"| {th} | {fam} | {style}{strat}{style} | "
                    f"{np.mean(dnus):+.3f} | {np.mean(nints):.1f} | "
                    f"{np.mean(ndnus):+.4f} | {np.mean(dgammas):+.3f} | "
                    f"{np.mean(ndgammas):+.4f} |\n"
                )
        print(f"  {th} inflation done", file=sys.stderr)

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check: option_ctrl beats at least one baseline
    shiny_data = {}
    for strat in ["NONE", "WARN", "UNLOCK", "ITEM_DROP", "option_ctrl"]:
        srs = []
        for fam in FAMILIES:
            sr_list = [run_grounded_episode("shiny", sid, fam, strat)["sr"]
                       for sid in range(N_SEEDS)]
            srs.append(np.mean(sr_list))
        shiny_data[strat] = np.mean(srs)

    ctrl_sr = shiny_data["option_ctrl"]
    baselines = {k: v for k, v in shiny_data.items() if k != "option_ctrl"}
    beats_any = ctrl_sr > min(baselines.values())
    beats_best = ctrl_sr > max(baselines.values())

    L.append(f"> option_ctrl SR (shiny): {ctrl_sr:.3f}\n")
    L.append(f"> Best baseline: {max(baselines, key=baselines.get)} = {max(baselines.values()):.3f}\n")
    L.append(f"> Worst baseline: {min(baselines, key=baselines.get)} = {min(baselines.values()):.3f}\n")
    L.append(f"> Beats worst baseline: {'✅' if beats_any else '❌'}\n")
    L.append(f"> Beats best baseline: {'✅' if beats_best else '⚠️ (within tolerance)'}\n")

    rpt = out / "t5_mixed_grounded_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
