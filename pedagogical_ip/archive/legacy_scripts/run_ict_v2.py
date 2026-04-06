"""ICT-v2 Main Experiment on TIC-v2.

8 strategies:
  1. no_tutor           4. ict_v1
  2. cajt_v3            5. ict_v2_zone_only (V_teach, no path)
  3. always_warn        6. ict_v2_path_only (path-sensitive, no zone)
                        7. ict_v2_full
                        8. oracle_teach_v3
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.teaching_internalization_corridor_v2 import (
    generate_tic_v2_session, generate_tic_v2_scenario, TIC_V2_SUBTYPES,
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
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.teachers.internalization_control_tutor_v1 import ICTv1
from src.teachers.internalization_control_tutor_v2 import ICTv2
from src.metrics.teaching_zone_v2 import zone_hit_rate_v2, overteach_rate_v2

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
    if name == "ict_v1":
        # Wrap ICTv1 to accept factored state
        return ICTv1(agent_params=AP)
    if name == "ict_v2_zone":
        return ICTv2(agent_params=AP, lambda_sd=0.0, lambda_dep=0.0)
    if name == "ict_v2_path":
        return ICTv2(agent_params=AP, lambda_teach=0.0, lambda_over=0.0)
    if name == "ict_v2":
        return ICTv2(agent_params=AP)
    return None


def run_session(strategy, theta, seed=0):
    sess = generate_tic_v2_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    m = FactoredInternalizationState()
    m.snapshot()

    n_warns = 0
    traces = {"A": [], "B": [], "C": []}

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_v2_scenario(ep)
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

        do_warn = False
        phase = ep.phase
        subtype = ep.subtype

        if phase != "A":
            do_warn = False
        elif strategy == "no_tutor":
            do_warn = False
        elif strategy == "always_warn":
            do_warn = True
        elif strategy == "oracle_teach_v3":
            # Avoid overteaching: check ν and γ_gen
            if subtype == "self_discovery_needed" and ep.d_commit > ep.d_reveal:
                do_warn = False
            elif subtype == "false_suppression_cost":
                do_warn = False  # let agent explore
            elif m.nu > 0.25 or m.gamma_gen > 0.15:
                do_warn = False  # back off
            elif subtype == "warn_rescue":
                do_warn = True
            else:
                do_warn = (ep.d_commit <= ep.d_reveal)
        elif isinstance(tutor, ICTv2):
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            do_warn = (action == "WARN")
        elif isinstance(tutor, ICTv1):
            # ICTv1 uses v2 internalization state — approximate
            from src.agents.internalization_dynamics_v2 import InternalizationStateV2
            m_approx = InternalizationStateV2(
                kappa=m.kappa, eta=m.tau, gamma=(m.gamma_spec + m.gamma_gen))
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m_approx)
            do_warn = (action == "WARN")
        elif tutor is not None:
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")

        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_factored_choice(branches, theta, m, AP, rng, wb)

        if do_warn:
            n_warns += 1

        # Factored internalization updates
        chose_risky = (ac != sc.oracle_safe_branch_id)
        real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05
        m.update_risk(real_risk, 0.15)

        oracle_warn = (ep.d_commit <= ep.d_reveal)
        has_self_ev = (ep.d_commit > ep.d_reveal + 1)

        # Trust: quality-driven
        if do_warn and oracle_warn:
            m.update_trust(warn_helpful=True)
        elif do_warn and not oracle_warn:
            m.update_trust(warn_bad=True)

        # Dependence vs self-discovery
        if do_warn and not has_self_ev:
            m.update_dependence(blind_obey=True)
        elif not do_warn and not chose_risky and has_self_ev:
            m.update_dependence(self_discovery=True)

        # γ_spec: temptation-specific
        tempt_high = branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
        if chose_risky and tempt_high:
            m.update_gamma_spec(tempt_error=True)
        elif not chose_risky and subtype == "false_suppression_cost":
            m.update_gamma_spec(false_suppression=True)

        # γ_gen: general suppression
        if do_warn:
            m.update_gamma_gen(sustained_pressure=True)
        elif not chose_risky and has_self_ev:
            m.update_gamma_gen(successful_exploration=True)

        m.snapshot()

        traces[phase].append({
            "warned": do_warn, "agent_safe": (ac == sc.oracle_safe_branch_id),
            "subtype": subtype,
        })

    def sbcr(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["agent_safe"]) / max(len(t), 1)

    wr_a = sum(1 for t in traces.get("A", []) if t["warned"]) / max(len(traces.get("A", [])), 1)
    zhr = zone_hit_rate_v2(m)
    otr = overteach_rate_v2(m)

    # Per-subtype traces in Phase A
    sub_safe = {}
    for st in TIC_V2_SUBTYPES:
        eps = [t for t in traces.get("A", []) if t["subtype"] == st]
        sub_safe[st] = sum(1 for t in eps if t["agent_safe"]) / max(len(eps), 1) if eps else None

    return {
        "sbcr_a": round(sbcr("A"), 3), "sbcr_b": round(sbcr("B"), 3),
        "sbcr_c": round(sbcr("C"), 3), "wr_a": round(wr_a, 3),
        "kappa": round(m.kappa, 3), "tau": round(m.tau, 3),
        "nu": round(m.nu, 3), "gs": round(m.gamma_spec, 3),
        "gg": round(m.gamma_gen, 3),
        "zhr": round(zhr, 3),
        "nu_over": otr["nu_over"], "gs_over": otr["gs_over"],
        "gg_over": otr["gg_over"], "otr_total": otr["total"],
        "sub_safe": sub_safe,
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ ICT-v2: State-Factored ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "always_warn", "cajt_v3", "ict_v1",
                   "ict_v2_zone", "ict_v2_path", "ict_v2", "oracle_teach_v3"]
    thetas = ["safe", "shiny"]
    lines = ["# ICT-v2: State-Factored Internalization Control\n\n"]

    lines.append("## Main Results\n\n")
    lines.append("| θ | Strategy | A | WR | **B** | **C** | κ | τ | ν | γs | γg | ZHR | OTR |\n")
    lines.append("|---|----------|---|----|----|----|----|---|---|----|----|-----|-----|\n")

    all_r = []
    for theta in thetas:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["sbcr_a", "sbcr_b", "sbcr_c", "wr_a",
                 "kappa", "tau", "nu", "gs", "gg", "zhr",
                 "nu_over", "gs_over", "gg_over", "otr_total"]}
            a["theta"] = theta; a["strategy"] = s
            all_r.append(a)
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr_a"]), sf(a["wr_a"]),
                sf(a["sbcr_b"]), sf(a["sbcr_c"]),
                sf(a["kappa"], "{:.2f}"), sf(a["tau"], "{:.2f}"),
                sf(a["nu"], "{:.2f}"), sf(a["gs"], "{:.2f}"),
                sf(a["gg"], "{:.2f}"), sf(a["zhr"], "{:.3f}"),
                sf(a["otr_total"], "{:.3f}")))
            print(f"  {theta}×{s}: B={sf(a['sbcr_b'])} C={sf(a['sbcr_c'])} "
                  f"ν={sf(a['nu'],'{:.2f}')} γg={sf(a['gg'],'{:.2f}')} "
                  f"ZHR={sf(a['zhr'],'{:.3f}')}", file=sys.stderr)

    # Transfer Δ
    lines.append("\n## Transfer Δ vs no_tutor\n\n")
    lines.append("| θ | Strategy | Δ(B) | Δ(C) |\n")
    lines.append("|---|----------|------|------|\n")
    for theta in thetas:
        base = [x for x in all_r if x["theta"] == theta and x["strategy"] == "no_tutor"]
        bb = base[0]["sbcr_b"] if base else 0
        bc = base[0]["sbcr_c"] if base else 0
        for s in strategies:
            if s == "no_tutor": continue
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                db = round(a["sbcr_b"] - bb, 3) if a["sbcr_b"] is not None else None
                dc = round(a["sbcr_c"] - bc, 3) if a["sbcr_c"] is not None else None
                lines.append("| {} | {} | {} | {} |\n".format(
                    theta, s, sf(db, "{:+.3f}"), sf(dc, "{:+.3f}")))

    # Overteaching decomposition
    lines.append("\n## Overteaching Decomposition\n\n")
    lines.append("| θ | Strategy | ν_over | γs_over | **γg_over** | Total |\n")
    lines.append("|---|----------|--------|---------|------------|-------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | **{}** | {} |\n".format(
                    theta, s,
                    sf(a["nu_over"], "{:.3f}"), sf(a["gs_over"], "{:.3f}"),
                    sf(a["gg_over"], "{:.3f}"), sf(a["otr_total"], "{:.3f}")))

    # Mechanism comparison: τ vs ν
    lines.append("\n## Trust vs Dependence\n\n")
    lines.append("| θ | Strategy | τ (trust) | ν (dependence) | τ-ν gap |\n")
    lines.append("|---|----------|-----------|----------------|----------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                gap = round(a["tau"] - a["nu"], 3) if a["tau"] and a["nu"] else None
                lines.append("| {} | {} | {} | {} | {} |\n".format(
                    theta, s, sf(a["tau"], "{:.3f}"), sf(a["nu"], "{:.3f}"),
                    sf(gap, "{:+.3f}")))

    with open(out / "ict_v2_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/ict_v2_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
