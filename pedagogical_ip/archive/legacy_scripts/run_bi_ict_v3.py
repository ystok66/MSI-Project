"""BI-ICT-v3 Main Experiment on TIC-v3.

7 strategies × 2θ × 8 seeds, 4-phase dual transfer.
Phase A: Tutor (10 ep), Phase B: Autonomy (4 ep),
Phase C: Sparse valid advice (4 ep), Phase D: Sparse invalid advice (4 ep).
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.teaching_internalization_corridor_v3 import (
    generate_tic_v3_session, generate_tic_v3_scenario,
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
from src.agents.behavior_probes import all_probes, behavior_zone_hit_rate
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.teachers.internalization_control_tutor_v2 import ICTv2
from src.teachers.internalization_control_tutor_v3 import BIICTv3
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
    if name == "cajt_v3": return CAJTv3(agent_params=AP)
    if name == "ict_v2": return ICTv2(agent_params=AP)
    if name == "bi_ict_v3": return BIICTv3(agent_params=AP)
    if name == "bi_ict_v3_beh_only":
        return BIICTv3(agent_params=AP, lambda_sd=0.0, lambda_dep=0.0)
    if name == "bi_ict_v3_path_only":
        return BIICTv3(agent_params=AP, lambda_teach=0.0, lambda_over=0.0)
    return None


def run_session(strategy, theta, seed=0):
    sess = generate_tic_v3_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    m = FactoredInternalizationState()
    m.snapshot()

    traces = {"A": [], "B": [], "C": [], "D": []}

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_v3_scenario(ep)
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
        do_warn = False
        advice_correct = True
        give_advice = False

        if phase == "A":
            # Tutor phase
            if strategy == "no_tutor":
                do_warn = False
            elif strategy == "oracle_teach_beh":
                if subtype in ("self_discovery_needed", "beneficial_novelty"):
                    do_warn = False
                elif m.nu > 0.20 or m.gamma_gen > 0.12:
                    do_warn = False
                elif subtype == "warn_rescue":
                    do_warn = True
                else:
                    do_warn = (ep.d_commit <= ep.d_reveal)
            elif isinstance(tutor, (BIICTv3, ICTv2)):
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
                do_warn = (action == "WARN")
            elif tutor is not None:
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                do_warn = (action == "WARN")
        elif phase == "C":
            # Sparse valid advice: 50% chance of a correct hint
            if rng.random() < 0.5:
                give_advice = True
                advice_correct = True
        elif phase == "D":
            # Sparse invalid advice: 50% chance of a wrong hint
            if rng.random() < 0.5:
                give_advice = True
                advice_correct = False

        # Compute warn bonuses
        if phase == "A":
            wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        elif give_advice:
            if advice_correct:
                # Correct advice points to safe branch
                wb = [0.3, -0.3] if sc.oracle_safe_branch_id == 0 else [-0.3, 0.3]
            else:
                # Wrong advice points to risky branch
                wb = [-0.3, 0.3] if sc.oracle_safe_branch_id == 0 else [0.3, -0.3]
        else:
            wb = [0.0, 0.0]

        novel_flags = [False, False]
        if subtype == "beneficial_novelty":
            novel_flags = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]

        ac = sample_factored_choice(branches, theta, m, AP, rng, wb, novel_flags)

        # Internalization updates
        chose_risky = (ac != sc.oracle_safe_branch_id)
        real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05
        m.update_risk(real_risk, 0.15)

        oracle_warn = (ep.d_commit <= ep.d_reveal)
        has_self_ev = (ep.d_commit > ep.d_reveal + 1)

        if phase == "A":
            if do_warn and oracle_warn:
                m.update_trust(warn_helpful=True)
            elif do_warn and not oracle_warn:
                m.update_trust(warn_bad=True)
            if do_warn and not has_self_ev:
                m.update_dependence(blind_obey=True)
            elif not do_warn and not chose_risky and has_self_ev:
                m.update_dependence(self_discovery=True)
            if do_warn:
                m.update_gamma_gen(sustained_pressure=True)
            elif not chose_risky and has_self_ev:
                m.update_gamma_gen(successful_exploration=True)
        elif phase == "C" and give_advice:
            if not chose_risky:
                m.update_trust(warn_helpful=True)
            if not has_self_ev:
                m.update_dependence(blind_obey=True)
        elif phase == "D" and give_advice:
            if chose_risky:  # followed bad advice
                m.update_dependence(blind_obey=True)
            elif not chose_risky and has_self_ev:
                m.update_dependence(self_discovery=True)

        tempt_high = branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
        if chose_risky and tempt_high:
            m.update_gamma_spec(tempt_error=True)
        elif subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
            m.update_gamma_spec(false_suppression=True)

        m.snapshot()

        # Determine "correct" choice depending on subtype
        if subtype in ("false_suppression_cost", "beneficial_novelty"):
            agent_correct = chose_risky  # risky branch is actually good
        else:
            agent_correct = (ac == sc.oracle_safe_branch_id)

        traces[phase].append({
            "correct": agent_correct,
            "agent_safe": (ac == sc.oracle_safe_branch_id),
            "subtype": subtype,
            "gave_advice": give_advice,
            "advice_correct": advice_correct if give_advice else None,
            "followed_advice": (give_advice and
                                ((ac == sc.oracle_safe_branch_id) == advice_correct)),
        })

    def sbcr(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["agent_safe"]) / max(len(t), 1)

    def correct_rate(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["correct"]) / max(len(t), 1)

    def advice_follow_rate(ph):
        t = [x for x in traces.get(ph, []) if x["gave_advice"]]
        return sum(1 for x in t if x["followed_advice"]) / max(len(t), 1) if t else None

    wr_a = sum(1 for t in traces.get("A", []) if t.get("gave_advice") or t.get("correct"))
    wr_a_real = sum(1 for t in traces.get("A", [])
                    if (strategy != "no_tutor" and t.get("agent_safe") != t.get("correct")))

    probes = all_probes(m, AP, theta)
    beh_zhr = behavior_zone_hit_rate(m, AP, theta)
    otr = overteach_rate_v2(m)

    return {
        "sbcr_a": round(sbcr("A"), 3),
        "correct_b": round(correct_rate("B"), 3),
        "correct_c": round(correct_rate("C"), 3),
        "correct_d": round(correct_rate("D"), 3),
        "follow_c": advice_follow_rate("C"),
        "follow_d": advice_follow_rate("D"),
        "RC": probes["RC"], "TR": probes["TR"], "EP": probes["EP"],
        "VA": probes["VA"], "IA": probes["IA"],
        "beh_zhr": round(beh_zhr, 3),
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gs": round(m.gamma_spec, 3), "gg": round(m.gamma_gen, 3),
        "otr": otr["total"],
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ BI-ICT-v3: Behavioral Identification ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "cajt_v3", "ict_v2",
                   "bi_ict_v3_beh_only", "bi_ict_v3_path_only",
                   "bi_ict_v3", "oracle_teach_beh"]
    thetas = ["safe", "shiny"]
    lines = ["# BI-ICT-v3: Behaviorally-Identified ICT\n\n"]

    # Main results
    lines.append("## 4-Phase Transfer\n\n")
    lines.append("| θ | Strategy | A(safe) | **B(correct)** | **C(correct)** | **D(correct)** | FollowC | FollowD |\n")
    lines.append("|---|----------|---------|-------------|-------------|-------------|---------|----------|\n")

    all_r = []
    for theta in thetas:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in [
                "sbcr_a", "correct_b", "correct_c", "correct_d",
                "follow_c", "follow_d",
                "RC", "TR", "EP", "VA", "IA", "beh_zhr",
                "tau", "nu", "gs", "gg", "otr"]}
            a["theta"] = theta; a["strategy"] = s
            all_r.append(a)
            lines.append("| {} | {} | {} | **{}** | **{}** | **{}** | {} | {} |\n".format(
                theta, s, sf(a["sbcr_a"]),
                sf(a["correct_b"]), sf(a["correct_c"]),
                sf(a["correct_d"]),
                sf(a["follow_c"]), sf(a["follow_d"])))
            print(f"  {theta}×{s}: B={sf(a['correct_b'])} C={sf(a['correct_c'])} "
                  f"D={sf(a['correct_d'])} IA={sf(a['IA'],'{:.3f}')}",
                  file=sys.stderr)

    # Behavior probes
    lines.append("\n## Behavior Probes (Final State)\n\n")
    lines.append("| θ | Strategy | RC | TR | **EP** | **VA** | **IA** | BehZHR |\n")
    lines.append("|---|----------|----|----|--------|--------|--------|--------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | **{}** | **{}** | **{}** | {} |\n".format(
                    theta, s,
                    sf(a["RC"], "{:.3f}"), sf(a["TR"], "{:.3f}"),
                    sf(a["EP"], "{:.3f}"), sf(a["VA"], "{:.3f}"),
                    sf(a["IA"], "{:.3f}"), sf(a["beh_zhr"], "{:.3f}")))

    # Trust vs Dependence
    lines.append("\n## Trust vs Dependence\n\n")
    lines.append("| θ | Strategy | τ | ν | **τ-ν** | γ_gen | OTR |\n")
    lines.append("|---|----------|---|---|---------|-------|-----|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                gap = round(a["tau"] - a["nu"], 3) if a["tau"] and a["nu"] else None
                lines.append("| {} | {} | {} | {} | **{}** | {} | {} |\n".format(
                    theta, s,
                    sf(a["tau"], "{:.3f}"), sf(a["nu"], "{:.3f}"),
                    sf(gap, "{:+.3f}"),
                    sf(a["gg"], "{:.3f}"), sf(a["otr"], "{:.3f}")))

    # Key comparisons
    lines.append("\n## Key Δ: Phase C (valid advice) and D (invalid advice)\n\n")
    lines.append("| θ | Strategy | FollowC(valid) | FollowD(invalid) | Δ(C-D) |\n")
    lines.append("|---|----------|----------------|------------------|--------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                fc = a["follow_c"]; fd = a["follow_d"]
                delta = round(fc - fd, 3) if fc is not None and fd is not None else None
                lines.append("| {} | {} | {} | {} | {} |\n".format(
                    theta, s, sf(fc), sf(fd), sf(delta, "{:+.3f}")))

    with open(out / "bi_ict_v3_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/bi_ict_v3_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
