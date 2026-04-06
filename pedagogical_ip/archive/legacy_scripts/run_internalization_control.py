"""ICT-v1 Main Experiment: Teaching vs Helping with Internalization Control.

6 strategies:
  1. no_tutor
  2. cajt_v3
  3. cajt_v3 + teach_only (V_teach, no R_over)
  4. cajt_v3 + over_only (R_over, no V_teach)
  5. ICT_v1_full
  6. oracle_teaching_v2

All use InternalizationStateV2 + TIC 3-phase.
Report: Phase B/C SBCR, LG, ZoneHitRate, OverTeachRate, PE.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.teaching_internalization_corridor import (
    generate_tic_session, generate_tic_scenario, TIC_SUBTYPES,
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
from src.agents.internalization_dynamics_v2 import (
    InternalizationStateV2, sample_pia_v2_choice,
)
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.teachers.internalization_control_tutor_v1 import ICTv1
from src.metrics.teaching_zone import teaching_loss, zone_hit_rate
from src.metrics.overteaching import overteach_rate, overteach_decomposed
from src.metrics.pedagogical_metrics import learning_gain

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


def run_tic_ict(strategy, theta, seed=0):
    sess = generate_tic_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)

    m = InternalizationStateV2()
    m_pre = m.copy()
    m.snapshot()

    # Make tutor based on strategy
    if strategy == "cajt_v3":
        tutor = CAJTv3(agent_params=AP)
    elif strategy == "ict_v1":
        tutor = ICTv1(agent_params=AP)
    elif strategy == "teach_only":
        tutor = ICTv1(agent_params=AP, lambda_over=0.0)
    elif strategy == "over_only":
        tutor = ICTv1(agent_params=AP, lambda_teach=0.0)
    else:
        tutor = None

    n_warns = 0
    traces = {"A": [], "B": [], "C": []}

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
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

        do_warn = False
        phase = ep.phase

        if phase != "A":
            do_warn = False
        elif strategy == "no_tutor":
            do_warn = False
        elif strategy == "oracle_teach_v2":
            # Oracle that avoids overteaching
            if ep.subtype == "self_discovery_teach" and ep.d_commit > ep.d_reveal:
                do_warn = False
            elif ep.subtype == "warn_rescue":
                do_warn = True
            elif m.kappa > 2.0 or m.gamma > 0.5:
                do_warn = False  # avoid overteaching
            else:
                do_warn = (ep.d_commit <= ep.d_reveal)
        elif isinstance(tutor, ICTv1):
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            do_warn = (action == "WARN")
        elif tutor is not None:
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")

        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_pia_v2_choice(branches, theta, m, AP, rng, wb)

        if do_warn:
            n_warns += 1

        # v2 internalization updates
        chose_risky = (ac != sc.oracle_safe_branch_id)
        real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05

        m.update_risk(real_risk, 0.15)

        # Warning quality assessment
        oracle_would_warn = (ep.d_commit <= ep.d_reveal)
        if do_warn and oracle_would_warn:
            m.update_trust(warn_helpful=True)
        elif do_warn and not oracle_would_warn:
            m.update_trust(warn_unnecessary=True)
        elif not do_warn and oracle_would_warn:
            m.update_trust(warn_missed=True)

        tempt_high = branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
        # False suppression: chose safe but missed a good opportunity
        false_supp = (not chose_risky and m.gamma > 0.3
                      and sc.risk_level if hasattr(sc, 'risk_level') else 0.3 < 0.2)
        m.update_suppression(
            temptation_error=(chose_risky and tempt_high),
            false_suppression=false_supp)
        m.snapshot()

        traces[phase].append({
            "warned": do_warn, "agent_safe": (ac == sc.oracle_safe_branch_id),
            "subtype": ep.subtype,
        })

    def phase_sbcr(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["agent_safe"]) / max(len(t), 1)

    m_post = m.copy()
    lg = learning_gain(m_pre, m_post)
    zhr = zone_hit_rate(m.kappa_history, m.eta_history, m.gamma_history, theta)
    otr = overteach_rate(m.kappa_history, m.eta_history, m.gamma_history)
    otd = overteach_decomposed(m.kappa_history, m.eta_history, m.gamma_history)

    a_traces = traces.get("A", [])
    wr_a = sum(1 for t in a_traces if t["warned"]) / max(len(a_traces), 1)

    return {
        "sbcr_a": round(phase_sbcr("A"), 3),
        "sbcr_b": round(phase_sbcr("B"), 3),
        "sbcr_c": round(phase_sbcr("C"), 3),
        "wr_a": round(wr_a, 3), "n_warns": n_warns,
        "kappa_f": round(m.kappa, 3), "eta_f": round(m.eta, 3),
        "gamma_f": round(m.gamma, 3),
        "lg_total": lg["lg_total"],
        "zone_hr": round(zhr, 3),
        "over_rate": round(otr, 3),
        "k_over": round(otd["kappa_over"], 3),
        "g_over": round(otd["gamma_over"], 3),
        "e_under": round(otd["eta_under"], 3),
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ ICT-v1: Internalization Control ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "cajt_v3", "teach_only", "over_only",
                   "ict_v1", "oracle_teach_v2"]
    thetas = ["safe", "shiny"]
    lines = ["# ICT-v1: Internalization-Control Tutor\n\n"]

    lines.append("## Main Results\n\n")
    lines.append("| θ | Strategy | SBCR(A) | WR(A) | **SBCR(B)** | **SBCR(C)** | κ_f | η_f | γ_f | ZoneHR | OverRate |\n")
    lines.append("|---|----------|---------|-------|------------|------------|-----|-----|-----|--------|----------|\n")

    all_results = []
    for theta in thetas:
        for s in strategies:
            rs = [run_tic_ict(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["sbcr_a", "sbcr_b", "sbcr_c", "wr_a",
                 "kappa_f", "eta_f", "gamma_f", "lg_total",
                 "zone_hr", "over_rate", "k_over", "g_over", "e_under"]}
            a["theta"] = theta; a["strategy"] = s
            all_results.append(a)
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr_a"]), sf(a["wr_a"]),
                sf(a["sbcr_b"]), sf(a["sbcr_c"]),
                sf(a["kappa_f"], "{:.3f}"), sf(a["eta_f"], "{:.3f}"),
                sf(a["gamma_f"], "{:.3f}"),
                sf(a["zone_hr"], "{:.3f}"), sf(a["over_rate"], "{:.3f}")))
            print(f"  {theta} × {s}: B={sf(a['sbcr_b'])} C={sf(a['sbcr_c'])} "
                  f"ZHR={sf(a['zone_hr'], '{:.3f}')} OR={sf(a['over_rate'], '{:.3f}')}",
                  file=sys.stderr)

    # Transfer comparison
    lines.append("\n## Transfer: SBCR by Phase\n\n")
    lines.append("| θ | Strategy | Phase A | **Phase B** | **Phase C** | Δ(B-base) | Δ(C-base) |\n")
    lines.append("|---|----------|---------|------------|------------|-----------|----------|\n")
    for theta in thetas:
        base = [x for x in all_results if x["theta"] == theta and x["strategy"] == "no_tutor"]
        bb = base[0]["sbcr_b"] if base else 0
        bc = base[0]["sbcr_c"] if base else 0
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                db = round(a["sbcr_b"] - bb, 3) if a["sbcr_b"] is not None else None
                dc = round(a["sbcr_c"] - bc, 3) if a["sbcr_c"] is not None else None
                lines.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                    theta, s, sf(a["sbcr_a"]),
                    sf(a["sbcr_b"]), sf(a["sbcr_c"]),
                    sf(db, "{:+.3f}"), sf(dc, "{:+.3f}")))

    # Overteaching decomposition
    lines.append("\n## Overteaching Decomposition\n\n")
    lines.append("| θ | Strategy | κ_over | γ_over | η_under | **Total** |\n")
    lines.append("|---|----------|--------|--------|---------|----------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | {} | **{}** |\n".format(
                    theta, s,
                    sf(a["k_over"], "{:.3f}"), sf(a["g_over"], "{:.3f}"),
                    sf(a["e_under"], "{:.3f}"), sf(a["over_rate"], "{:.3f}")))

    with open(out / "ict_v1_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/ict_v1_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
