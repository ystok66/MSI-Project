"""T4 Exp-T4-3: Inflation / Transfer Audit.

Monitors Δν̂, Δγ̂_gen across intervention types.
Checks that intervention gains don't come at dependence/suppression cost.
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
from src.teachers.option_intervention_controller import (
    OptionInterventionController, OptionConfig,
)
from src.teachers.action_predictor import ActionPredictor
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent
from src.teachers.intervention_risk_head import InterventionRiskHead

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

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


def run_episode(theta, seed, family, strategy):
    """Run one episode with fixed strategy, track ν̂ and γ̂_gen trajectories."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    fam = FAMILIES[family]
    nu_traj = [m.nu]
    gamma_gen_traj = [m.gamma_gen]
    n_correct = 0

    for step_i in range(N_STEPS):
        branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=fam["risk"],
                           temptation_score=fam["tempt"]),
        ]
        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == 0)
        n_correct += int(correct)

        if strategy == "WARN":
            m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
            if correct:
                m.update_trust(warn_helpful=True)
        elif strategy == "ITEM_DROP":
            m.update_dependence(blind_obey=True)
        elif strategy == "UNLOCK":
            pass
        else:  # NONE
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
        "final_nu": m.nu,
        "final_gamma_gen": m.gamma_gen,
    }


def run_option_episode(theta, seed, family):
    """Run one episode with option controller."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    fam = FAMILIES[family]
    ap = ActionPredictor(AP)
    rboa = RobotBeliefOverAgent(action_predictor=ap)
    irh = InterventionRiskHead()
    cfg = OptionConfig(shield_cost=0.5, lambda_teach=0.8)
    ctrl = OptionInterventionController(config=cfg)

    nu_traj = [m.nu]
    gamma_gen_traj = [m.gamma_gen]
    n_correct = 0
    interventions = []

    for step_i in range(N_STEPS):
        ws = WorldState(t=step_i, t_max=N_STEPS)
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)
        branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=fam["risk"],
                           temptation_score=fam["tempt"]),
        ]
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
        interventions.append(d.chosen)

        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == 0)
        n_correct += int(correct)

        if d.chosen == "WARN":
            m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
            if correct:
                m.update_trust(warn_helpful=True)
        elif d.chosen == "ITEM_DROP":
            m.update_dependence(blind_obey=True)
        elif d.chosen == "UNLOCK":
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
        "final_nu": m.nu,
        "final_gamma_gen": m.gamma_gen,
        "interventions": interventions,
    }


def main():
    print("═══ T4 Inflation / Transfer Audit ═══\n", file=sys.stderr)
    L = ["# T4 Exp-T4-3: Inflation / Transfer Audit\n\n"]

    # ═══ Table 1: Δν̂ and Δγ̂_gen by strategy ═══
    L.append("## Δν̂ and Δγ̂_gen by Strategy\n\n")
    L.append("| θ | Family | Strategy | SR | Δν̂ | Δγ̂_gen | Final ν̂ | Final γ̂_gen |\n")
    L.append("|:-:|:------:|:--------:|:--:|:---:|:------:|:------:|:----------:|\n")

    for th in ["safe", "shiny"]:
        for fam in FAMILIES:
            for strat in ["NONE", "WARN", "UNLOCK", "ITEM_DROP", "option_ctrl"]:
                srs, dnus, dgammas, fnus, fgammas = [], [], [], [], []
                for sid in range(N_SEEDS):
                    if strat == "option_ctrl":
                        r = run_option_episode(th, sid, fam)
                    else:
                        r = run_episode(th, sid, fam, strat)
                    srs.append(r["sr"])
                    dnus.append(r["delta_nu"])
                    dgammas.append(r["delta_gamma_gen"])
                    fnus.append(r["final_nu"])
                    fgammas.append(r["final_gamma_gen"])

                style = "**" if strat == "option_ctrl" else ""
                L.append(
                    f"| {th} | {fam} | {style}{strat}{style} | "
                    f"{np.mean(srs):.3f} | {np.mean(dnus):+.4f} | "
                    f"{np.mean(dgammas):+.4f} | {np.mean(fnus):.3f} | "
                    f"{np.mean(fgammas):.3f} |\n"
                )
        print(f"  {th} done", file=sys.stderr)

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check: option_ctrl Δν̂ < 0.15
    nu_ok = True
    gamma_ok = True
    for th in ["safe", "shiny"]:
        for fam in FAMILIES:
            dnus = []
            dgammas = []
            for sid in range(N_SEEDS):
                r = run_option_episode(th, sid, fam)
                dnus.append(r["delta_nu"])
                dgammas.append(r["delta_gamma_gen"])
            if np.mean(dnus) > 0.15:
                nu_ok = False
            if np.mean(dgammas) > 0.10:
                gamma_ok = False

    L.append(f"> Δν̂ < 0.15: {'✅' if nu_ok else '❌'}\n")
    L.append(f"> Δγ̂_gen < 0.10: {'✅' if gamma_ok else '❌'}\n")

    if nu_ok and gamma_ok:
        L.append("> **✅ Inflation controlled — interventions don't create excessive dependence**\n")
    else:
        L.append("> **⚠️ Inflation check failed — needs weight tuning**\n")

    rpt = out / "t4_inflation_transfer_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
