"""T4 Exp-T4-1/2: Family-Selective + Mixed-Family Audit.

Exp-T4-1: Verify option controller selects correct lever per family
Exp-T4-2: Compare option controller vs single-strategy baselines in mixed
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
from src.teachers.intervention_risk_head import InterventionRiskHead
from src.teachers.action_predictor import ActionPredictor
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent
from src.teachers.warning_utterance_policy import WarningUtterancePolicy
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Family configs with expected primary lever
FAMILIES = {
    "fork_trap": {"primary": "WARN", "d_commit": 2, "d_reveal": 3,
                  "risk": 0.4, "has_doors": False, "tempt": 0.3},
    "hazard_belt": {"primary": "ITEM_DROP", "d_commit": 1, "d_reveal": 4,
                    "risk": 0.35, "has_doors": False, "tempt": 0.1},
    "deadline_gate": {"primary": "UNLOCK", "d_commit": 4, "d_reveal": 2,
                      "risk": 0.15, "has_doors": True, "tempt": 0.0},
}
N_STEPS = 20
N_SEEDS = 10


def run_family_audit(theta, seed, family, cfg_override=None):
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    fam = FAMILIES[family]

    cfg = cfg_override or OptionConfig(shield_cost=0.5, lambda_teach=0.8)
    ctrl = OptionInterventionController(config=cfg)
    ap = ActionPredictor(AP)
    rboa = RobotBeliefOverAgent(action_predictor=ap)
    irh = InterventionRiskHead()
    wup = WarningUtterancePolicy()

    results = []
    nu_traj = []
    gamma_gen_traj = []

    for step_i in range(N_STEPS):
        ws = WorldState(t=step_i, t_max=N_STEPS)
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)

        # Compute timing risks
        path_est = max(N_STEPS - step_i, 1)
        irisk = irh.predict(ws, rboa, ap, [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=fam["risk"]),
        ], ab, d_commit=fam["d_commit"], d_reveal=fam["d_reveal"],
            path_length_estimate=path_est)

        # Option controller decision
        nu_traj.append(m.nu)
        gamma_gen_traj.append(m.gamma_gen)

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

        # Warning subtype (if WARN chosen)
        warn_subtype = None
        if d.chosen == "WARN":
            p_self = estimate_self_discovery_prob(fam["d_commit"], fam["d_reveal"])
            ud = wup.select_subtype(
                p_self=p_self, p_blind=irisk.p_blind,
                p_timeout=irisk.p_timeout,
                time_remaining=(N_STEPS - step_i) / N_STEPS,
                m_hat=dict(m.as_dict),
                scenario_family=family)
            warn_subtype = ud.subtype

        # Simulate learner response
        branches = [
            BranchAttributes(safety_score=0.8, risk_penalty=0.1),
            BranchAttributes(safety_score=0.3, risk_penalty=fam["risk"],
                           temptation_score=fam["tempt"]),
        ]
        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == 0)

        # Simulate intervention effect
        if d.chosen == "WARN":
            if correct:
                m.update_trust(warn_helpful=True)
            m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        elif d.chosen == "ITEM_DROP":
            m.update_dependence(blind_obey=True)
        elif d.chosen == "UNLOCK":
            pass  # minimal state effect
        else:
            if correct:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        m.update_risk(fam["risk"] if not correct else 0.05, 0.15)
        m.snapshot()

        results.append({
            "step": step_i, "family": family, "chosen": d.chosen,
            "correct": correct, "warn_subtype": warn_subtype,
        })

    return results, ctrl, wup


def run_single_strategy(theta, seed, family, strategy):
    """Baseline: always use one strategy."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    fam = FAMILIES[family]
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
        elif strategy == "ITEM_DROP":
            m.update_dependence(blind_obey=True)
        m.update_risk(fam["risk"] if not correct else 0.05, 0.15)
        m.snapshot()
    return n_correct / N_STEPS


def main():
    print("═══ T4 Family-Selective + Mixed Audit ═══\n", file=sys.stderr)
    L = ["# T4 Exp-T4-1/2: Family Selectivity + Mixed Audit\n\n"]

    # ═══ Exp-T4-1: Family-selective lever ═══
    L.append("## Exp-T4-1: Family-Selective Lever\n\n")
    L.append("| θ | Family | Primary | P(primary) | P(WARN) | P(UNLOCK) | P(ITEM) | SelGap |\n")
    L.append("|:-:|:------:|:-------:|:----------:|:-------:|:---------:|:-------:|:------:|\n")

    for th in ["safe", "shiny"]:
        for fam in FAMILIES:
            all_choices = []
            for sid in range(N_SEEDS):
                results, _, _ = run_family_audit(th, sid, fam)
                all_choices.extend([r["chosen"] for r in results])

            n = len(all_choices)
            primary = FAMILIES[fam]["primary"]
            p_primary = sum(1 for c in all_choices if c == primary) / n
            p_warn = sum(1 for c in all_choices if c == "WARN") / n
            p_unlock = sum(1 for c in all_choices if c == "UNLOCK") / n
            p_item = sum(1 for c in all_choices if c == "ITEM_DROP") / n

            # SelGap = P(primary | this family) - mean P(primary | other families)
            other_rates = []
            for other_fam in FAMILIES:
                if other_fam == fam: continue
                oc = []
                for sid in range(N_SEEDS):
                    r2, _, _ = run_family_audit(th, sid, other_fam)
                    oc.extend([r["chosen"] for r in r2])
                other_rates.append(sum(1 for c in oc if c == primary) / len(oc))
            selgap = p_primary - np.mean(other_rates)

            L.append(f"| {th} | {fam} | {primary} | {p_primary:.3f} | "
                     f"{p_warn:.3f} | {p_unlock:.3f} | {p_item:.3f} | "
                     f"{selgap:+.3f} |\n")
        print(f"  {th} done", file=sys.stderr)

    # ═══ Exp-T4-2: Mixed-family comparison ═══
    L.append("\n## Exp-T4-2: Mixed-Family Comparison\n\n")
    L.append("| θ | Strategy | fork_trap SR | hazard_belt SR | deadline_gate SR | Mean SR |\n")
    L.append("|:-:|:--------:|:----------:|:-------------:|:---------------:|:-------:|\n")

    for th in ["safe", "shiny"]:
        # Single-strategy baselines
        for strat in ["WARN", "UNLOCK", "ITEM_DROP", "NONE"]:
            srs = {}
            for fam in FAMILIES:
                sr_list = [run_single_strategy(th, sid, fam, strat)
                           for sid in range(N_SEEDS)]
                srs[fam] = np.mean(sr_list)
            mean_sr = np.mean(list(srs.values()))
            L.append(f"| {th} | always_{strat} | {srs['fork_trap']:.3f} | "
                     f"{srs['hazard_belt']:.3f} | {srs['deadline_gate']:.3f} | "
                     f"{mean_sr:.3f} |\n")

        # Option controller
        srs_opt = {}
        for fam in FAMILIES:
            sr_list = []
            for sid in range(N_SEEDS):
                results, _, _ = run_family_audit(th, sid, fam)
                sr_list.append(
                    sum(1 for r in results if r["correct"]) / len(results))
            srs_opt[fam] = np.mean(sr_list)
        mean_sr_opt = np.mean(list(srs_opt.values()))
        L.append(f"| {th} | **option_ctrl** | {srs_opt['fork_trap']:.3f} | "
                 f"{srs_opt['hazard_belt']:.3f} | {srs_opt['deadline_gate']:.3f} | "
                 f"**{mean_sr_opt:.3f}** |\n")

    # ═══ Warning subtype distribution ═══
    L.append("\n## Warning Subtype Distribution (shadow)\n\n")
    L.append("| θ | Family | hint | alert | explain | directive |\n")
    L.append("|:-:|:------:|:----:|:-----:|:-------:|:---------:|\n")

    for th in ["safe", "shiny"]:
        for fam in FAMILIES:
            all_subtypes = []
            for sid in range(N_SEEDS):
                results, _, wup = run_family_audit(th, sid, fam)
                for r in results:
                    if r["warn_subtype"]:
                        all_subtypes.append(r["warn_subtype"])
            n = max(len(all_subtypes), 1)
            freqs = {s: sum(1 for x in all_subtypes if x == s) / n
                     for s in ["hint", "alert", "explain", "directive"]}
            L.append(f"| {th} | {fam} | {freqs['hint']:.3f} | {freqs['alert']:.3f} | "
                     f"{freqs['explain']:.3f} | {freqs['directive']:.3f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    rpt = out / "t4_family_selective_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
