"""BC-ICT-v4 Main Experiment on TIC-v4.

8 strategies × 2θ × 8 seeds, 5-phase dual transfer.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.teaching_internalization_corridor_v4 import (
    generate_tic_v4_session, generate_tic_v4_scenario,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal, neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams,
)
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.behavior_probes import all_probes
from src.agents.behavior_bridge import predict_all_probes, bridge_zone_hit
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.teachers.internalization_control_tutor_v2 import ICTv2
from src.teachers.internalization_control_tutor_v3 import BIICTv3
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2

out = Path("results")
out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)


def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def make_tutor(name):
    if name == "cajt_v3": return CAJTv3(agent_params=AP), "legacy"
    if name == "ict_v2": return ICTv2(agent_params=AP), "ict"
    if name == "bi_ict_v3": return BIICTv3(agent_params=AP), "ict"
    if name == "bc_v4_no_cal":
        return BCICTv4(agent_params=AP, use_calibrated_zones=False), "bc"
    if name == "bc_v4_no_dose":
        return BCICTv4(agent_params=AP, use_dose=False), "bc"
    if name == "bc_v4":
        return BCICTv4(agent_params=AP), "bc"
    return None, "none"


def run_session(strategy, theta, seed=0):
    sess = generate_tic_v4_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor, ttype = make_tutor(strategy)
    m = FactoredInternalizationState()
    m.snapshot()

    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    n_warns, n_soft = 0, 0

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_v4_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        branches = [bas, bar]
        phase = ep.phase
        subtype = ep.subtype

        dose = 0.0
        give_advice = False
        advice_correct = True

        if phase == "A":
            if strategy == "no_tutor":
                dose = 0.0
            elif strategy == "oracle_teach_beh":
                if subtype in ("self_discovery_needed", "beneficial_novelty"):
                    dose = 0.0
                elif m.nu > 0.20 or m.gamma_gen > 0.12:
                    dose = 0.0
                elif subtype == "warn_rescue":
                    dose = 1.0
                elif ep.d_commit <= ep.d_reveal:
                    dose = 0.5 if (m.nu > 0.15 or m.gamma_gen > 0.08) else 1.0
                else:
                    dose = 0.0
            elif ttype == "bc":
                action, dose, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            elif ttype == "ict":
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
                dose = 1.0 if action == "WARN" else 0.0
            elif ttype == "legacy":
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                dose = 1.0 if action == "WARN" else 0.0
        elif phase == "C":
            if rng.random() < 0.5:
                give_advice = True; advice_correct = True
        elif phase == "D":
            if rng.random() < 0.5:
                give_advice = True; advice_correct = False

        # Warning bonuses scaled by dose
        if phase == "A":
            wb = [0.3 * dose, -0.3 * dose]
            if dose > 0:
                n_warns += 1
                if dose < 1.0:
                    n_soft += 1
        elif give_advice:
            if advice_correct:
                wb = [0.3, -0.3] if sc.oracle_safe_branch_id == 0 else [-0.3, 0.3]
            else:
                wb = [-0.3, 0.3] if sc.oracle_safe_branch_id == 0 else [0.3, -0.3]
        else:
            wb = [0.0, 0.0]

        novel_flags = [False, False]
        if subtype == "beneficial_novelty":
            novel_flags = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]

        ac = sample_factored_choice(branches, theta, m, AP, rng, wb, novel_flags)

        # State updates (dose-scaled)
        chose_risky = (ac != sc.oracle_safe_branch_id)
        real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05
        m.update_risk(real_risk, 0.15)

        has_self_ev = (ep.d_commit > ep.d_reveal + 1)
        oracle_warn = (ep.d_commit <= ep.d_reveal)

        if phase == "A" and dose > 0:
            if oracle_warn:
                m.update_trust(warn_helpful=True)
            else:
                m.update_trust(warn_bad=True)
            if not has_self_ev:
                # Dose-scaled dependence push
                old_nu = m.nu
                m.update_dependence(blind_obey=True)
                m.nu = old_nu + dose * (m.nu - old_nu)
            old_gg = m.gamma_gen
            m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
        elif phase == "A" and dose == 0:
            if not chose_risky and has_self_ev:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        elif phase in ("C", "D") and give_advice:
            if phase == "C" and not chose_risky:
                m.update_trust(warn_helpful=True)
            if phase == "D" and chose_risky:
                m.update_dependence(blind_obey=True)
            elif phase == "D" and not chose_risky and has_self_ev:
                m.update_dependence(self_discovery=True)

        tempt_high = branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
        if chose_risky and tempt_high:
            m.update_gamma_spec(tempt_error=True)
        if subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
            m.update_gamma_spec(false_suppression=True)

        m.snapshot()

        # Correctness
        if subtype in ("false_suppression_cost", "beneficial_novelty"):
            correct = chose_risky
        else:
            correct = (ac == sc.oracle_safe_branch_id)

        traces[phase].append({
            "correct": correct,
            "agent_safe": (ac == sc.oracle_safe_branch_id),
            "subtype": subtype, "dose": dose,
            "gave_advice": give_advice,
            "advice_correct": advice_correct if give_advice else None,
            "followed_advice": (give_advice and
                                ((ac == sc.oracle_safe_branch_id) == advice_correct)),
        })

    def correct_rate(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["correct"]) / max(len(t), 1) if t else None

    def follow_rate(ph):
        t = [x for x in traces.get(ph, []) if x["gave_advice"]]
        return sum(1 for x in t if x["followed_advice"]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    bridge = predict_all_probes(m, 0.3, 0.3, 0.0, 0.5)
    otr = overteach_rate_v2(m)
    wr = n_warns / max(len(traces.get("A", [])), 1)
    sr = n_soft / max(n_warns, 1) if n_warns > 0 else 0

    return {
        "correct_a": correct_rate("A"),
        "correct_b": correct_rate("B"),
        "correct_c": correct_rate("C"),
        "correct_d": correct_rate("D"),
        "correct_e": correct_rate("E"),
        "follow_c": follow_rate("C"),
        "follow_d": follow_rate("D"),
        "wr": round(wr, 3), "soft_ratio": round(sr, 3),
        "RC": probes["RC"], "TR": probes["TR"], "EP": probes["EP"],
        "VA": probes["VA"], "IA": probes["IA"],
        "b_EP": bridge["EP"], "b_VA": bridge["VA"], "b_IA": bridge["IA"],
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gs": round(m.gamma_spec, 3), "gg": round(m.gamma_gen, 3),
        "otr": otr["total"],
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ BC-ICT-v4 ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "cajt_v3", "ict_v2", "bi_ict_v3",
                  "bc_v4_no_cal", "bc_v4_no_dose", "bc_v4", "oracle_teach_beh"]
    lines = ["# BC-ICT-v4: Behavior-Calibrated ICT\n\n"]
    lines.append("## 5-Phase Transfer\n\n")
    lines.append("| θ | Strategy | WR | Soft% | **B** | **C** | **D** | **E** | FollC | FollD |\n")
    lines.append("|---|----------|----|----|----|----|----|----|------|------|\n")

    all_r = []
    for theta in ["safe", "shiny"]:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in [
                "correct_a", "correct_b", "correct_c", "correct_d", "correct_e",
                "follow_c", "follow_d", "wr", "soft_ratio",
                "RC", "TR", "EP", "VA", "IA",
                "b_EP", "b_VA", "b_IA",
                "tau", "nu", "gs", "gg", "otr"]}
            a["theta"] = theta; a["strategy"] = s
            all_r.append(a)
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | **{}** | **{}** | {} | {} |\n".format(
                theta, s, sf(a["wr"]), sf(a["soft_ratio"]),
                sf(a["correct_b"]), sf(a["correct_c"]),
                sf(a["correct_d"]), sf(a["correct_e"]),
                sf(a["follow_c"]), sf(a["follow_d"])))
            print(f"  {theta}×{s}: B={sf(a['correct_b'])} C={sf(a['correct_c'])} "
                  f"D={sf(a['correct_d'])} E={sf(a['correct_e'])} "
                  f"ν={sf(a['nu'],'{:.2f}')} γg={sf(a['gg'],'{:.2f}')}",
                  file=sys.stderr)

    # Probes + bridge
    lines.append("\n## Probes & Bridge\n\n")
    lines.append("| θ | Strategy | EP | VA | IA | b_EP | b_VA | b_IA |\n")
    lines.append("|---|----------|----|----|----|----|------|------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                    theta, s,
                    sf(a["EP"], "{:.3f}"), sf(a["VA"], "{:.3f}"), sf(a["IA"], "{:.3f}"),
                    sf(a["b_EP"], "{:.3f}"), sf(a["b_VA"], "{:.3f}"), sf(a["b_IA"], "{:.3f}")))

    # Trust-dependence + dose
    lines.append("\n## State + Dose\n\n")
    lines.append("| θ | Strategy | τ | ν | **τ-ν** | γg | OTR | Dose |\n")
    lines.append("|---|----------|---|---|---------|----|----|------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                gap = round(a["tau"] - a["nu"], 3) if a["tau"] and a["nu"] else None
                dose_str = f"{sf(a['wr'])}({sf(a['soft_ratio'])}soft)"
                lines.append("| {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                    theta, s,
                    sf(a["tau"], "{:.3f}"), sf(a["nu"], "{:.3f}"),
                    sf(gap, "{:+.3f}"),
                    sf(a["gg"], "{:.3f}"), sf(a["otr"], "{:.3f}"), dose_str))

    with open(out / "bc_ict_v4_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/bc_ict_v4_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
