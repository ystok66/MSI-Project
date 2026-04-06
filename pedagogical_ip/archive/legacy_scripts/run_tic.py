"""TIC Experiment: Teaching vs Helping.

Transfer is the headline metric.

6 strategies:
  1. no_tutor — baseline, PIA agent without guidance
  2. always_warn — always intervene
  3. v1_1 — persistent tutor
  4. cajt_v3 — calibrated adaptive joint tutor
  5. oracle_current — oracle for current episode
  6. oracle_teaching — oracle that maximizes κ/η/γ gain

Reports: SBCR by phase (A/B/C), LG(κ,η,γ), TI_same, TI_shift, PE
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
from src.agents.internalization_agent import (
    InternalizationState, sample_pia_choice, compute_expected_m_change,
)
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.metrics.pedagogical_metrics import (
    learning_gain, transfer_improvement, pedagogical_efficiency,
)

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
    if name == "v1_1": return PersistentTutorV1_1(agent_params=AP)
    if name == "cajt_v3": return CAJTv3(agent_params=AP)
    return None


def run_tic_session(strategy, theta, seed=0):
    sess = generate_tic_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    m = InternalizationState()
    m_pre = m.copy()
    m.snapshot()

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
        lib.update("safe_branch", ss)
        lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        bas = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        branches = [bas, bar]

        # Decide tutor action
        do_warn = False
        diag = {}
        phase = ep.phase

        if phase != "A":
            # No tutor in phases B/C
            do_warn = False
        elif strategy == "no_tutor":
            do_warn = False
        elif strategy == "always_warn":
            do_warn = True
        elif strategy == "oracle_current":
            do_warn = (ep.d_commit <= ep.d_reveal)
        elif strategy == "oracle_teaching":
            # Oracle that prefers WAIT when self-discovery possible, WARN when rescue needed
            if ep.subtype == "self_discovery_teach" and ep.d_commit > ep.d_reveal:
                do_warn = False  # let self-discover for stronger κ update
            elif ep.subtype == "warn_rescue":
                do_warn = True
            else:
                do_warn = (ep.d_commit <= ep.d_reveal)
        elif tutor is not None:
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")

        # Compute warn bonus for PIA
        warn_bonus_safe = 0.3 if do_warn else 0.0
        warn_bonus_risky = -0.3 if do_warn else 0.0
        warn_bonuses = [warn_bonus_safe, warn_bonus_risky]

        # Agent chooses using PIA
        agent_choice = sample_pia_choice(branches, theta, m, AP, rng, warn_bonuses)

        if do_warn:
            n_warns += 1

        # Update tutor posterior
        if tutor is not None and phase == "A" and hasattr(tutor, 'observe_agent_choice'):
            tutor.observe_agent_choice(agent_choice, branches)

        # Compute outcome for internalization updates
        chose_risky = (agent_choice != sc.oracle_safe_branch_id)
        real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05
        expected_risk = 0.15  # agent's naive expectation

        # Warn accuracy check
        warn_matched = False
        if do_warn and chose_risky:
            warn_matched = True  # warn was right
        elif do_warn and not chose_risky:
            warn_matched = True  # agent listened → counts as consistent

        tempt_high = (branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
                      if sc.oracle_safe_branch_id < len(branches) else False)
        tempt_error = chose_risky and tempt_high

        # Update m_t
        m.update_risk(real_risk, expected_risk)
        m.update_trust(warn_matched)
        m.update_suppression(tempt_error)
        m.snapshot()

        traces[phase].append({
            "subtype": ep.subtype, "warned": do_warn,
            "agent_safe": (agent_choice == sc.oracle_safe_branch_id),
            "kappa": m.kappa, "eta": m.eta, "gamma": m.gamma,
        })

    # Aggregate
    def phase_sbcr(ph):
        t = traces.get(ph, [])
        return sum(1 for x in t if x["agent_safe"]) / max(len(t), 1)

    sbcr_a = phase_sbcr("A")
    sbcr_b = phase_sbcr("B")
    sbcr_c = phase_sbcr("C")

    # No-tutor baseline comparison
    m_post = m.copy()
    lg = learning_gain(m_pre, m_post)

    # Phase A WR
    a_traces = traces.get("A", [])
    wr_a = sum(1 for t in a_traces if t["warned"]) / max(len(a_traces), 1)

    # Per-subtype WR
    sub_wr = {}
    for st in TIC_SUBTYPES:
        eps = [t for t in a_traces if t["subtype"] == st]
        sub_wr[st] = round(sum(1 for t in eps if t["warned"]) / max(len(eps), 1), 3) if eps else None

    return {
        "sbcr_a": round(sbcr_a, 3), "sbcr_b": round(sbcr_b, 3),
        "sbcr_c": round(sbcr_c, 3), "wr_a": round(wr_a, 3),
        "n_warns": n_warns,
        "kappa_final": round(m.kappa, 3), "eta_final": round(m.eta, 3),
        "gamma_final": round(m.gamma, 3),
        "lg_kappa": lg["lg_kappa"], "lg_eta": lg["lg_eta"],
        "lg_gamma": lg["lg_gamma"], "lg_total": lg["lg_total"],
        "sub_wr": sub_wr,
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ TIC: Teaching vs Helping ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "always_warn", "v1_1", "cajt_v3",
                   "oracle_current", "oracle_teaching"]
    thetas = ["safe", "shiny"]
    lines = ["# TIC: Teaching-Internalization Corridor\n\n"]
    lines.append("**Transfer is the headline metric.**\n\n")

    # ── Main results ──
    lines.append("## Main Results\n\n")
    lines.append("| θ | Strategy | SBCR(A) | WR(A) | SBCR(B) | SBCR(C) | κ_f | η_f | γ_f | LG_total |\n")
    lines.append("|---|----------|---------|-------|---------|---------|-----|-----|-----|----------|\n")

    all_results = []
    for theta in thetas:
        for s in strategies:
            rs = [run_tic_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["sbcr_a", "sbcr_b", "sbcr_c", "wr_a", "n_warns",
                                           "kappa_final", "eta_final", "gamma_final",
                                           "lg_kappa", "lg_eta", "lg_gamma", "lg_total"]}
            a["theta"] = theta; a["strategy"] = s
            all_results.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr_a"]), sf(a["wr_a"]),
                sf(a["sbcr_b"]), sf(a["sbcr_c"]),
                sf(a["kappa_final"], "{:.3f}"), sf(a["eta_final"], "{:.3f}"),
                sf(a["gamma_final"], "{:.3f}"), sf(a["lg_total"], "{:.4f}")))
            print(f"  {theta} × {s}: A={sf(a['sbcr_a'])} B={sf(a['sbcr_b'])} C={sf(a['sbcr_c'])} "
                  f"LG={sf(a['lg_total'], '{:.4f}')}", file=sys.stderr)

    # ── Transfer comparison ──
    lines.append("\n## Transfer: SBCR by Phase\n\n")
    lines.append("| θ | Strategy | Phase A | **Phase B (same)** | **Phase C (shift)** | Δ(B-baseline) | Δ(C-baseline) |\n")
    lines.append("|---|----------|---------|-------------------|---------------------|----------------|----------------|\n")
    for theta in thetas:
        baseline = [x for x in all_results if x["theta"] == theta and x["strategy"] == "no_tutor"]
        b_base = baseline[0]["sbcr_b"] if baseline else 0
        c_base = baseline[0]["sbcr_c"] if baseline else 0
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                db = round(a["sbcr_b"] - b_base, 3) if a["sbcr_b"] is not None else None
                dc = round(a["sbcr_c"] - c_base, 3) if a["sbcr_c"] is not None else None
                lines.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                    theta, s, sf(a["sbcr_a"]),
                    sf(a["sbcr_b"]), sf(a["sbcr_c"]),
                    sf(db, "{:+.3f}"), sf(dc, "{:+.3f}")))

    # ── Learning Gain decomposed ──
    lines.append("\n## Learning Gain (LG) Decomposition\n\n")
    lines.append("| θ | Strategy | LG_κ | LG_η | LG_γ | **LG_total** |\n")
    lines.append("|---|----------|------|------|------|--------------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | {} | **{}** |\n".format(
                    theta, s,
                    sf(a["lg_kappa"], "{:+.4f}"), sf(a["lg_eta"], "{:+.4f}"),
                    sf(a["lg_gamma"], "{:+.4f}"), sf(a["lg_total"], "{:.4f}")))

    # ── Pedagogical Efficiency ──
    lines.append("\n## Pedagogical Efficiency (PE = TI / #warnings)\n\n")
    lines.append("| θ | Strategy | TI_same | TI_shift | #warns | **PE_same** | **PE_shift** |\n")
    lines.append("|---|----------|---------|----------|--------|------------|-------------|\n")
    for theta in thetas:
        baseline = [x for x in all_results if x["theta"] == theta and x["strategy"] == "no_tutor"]
        b_base = baseline[0]["sbcr_b"] if baseline else 0
        c_base = baseline[0]["sbcr_c"] if baseline else 0
        for s in strategies:
            if s == "no_tutor":
                continue
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                ti_same = round(a["sbcr_b"] - b_base, 4) if a["sbcr_b"] is not None else None
                ti_shift = round(a["sbcr_c"] - c_base, 4) if a["sbcr_c"] is not None else None
                nw = a["n_warns"] or 1
                pe_same = round(ti_same / max(nw, 0.1), 4) if ti_same is not None else None
                pe_shift = round(ti_shift / max(nw, 0.1), 4) if ti_shift is not None else None
                lines.append("| {} | {} | {} | {} | {} | **{}** | **{}** |\n".format(
                    theta, s,
                    sf(ti_same, "{:+.3f}"), sf(ti_shift, "{:+.3f}"),
                    sf(nw, "{:.0f}"),
                    sf(pe_same, "{:+.4f}"), sf(pe_shift, "{:+.4f}")))

    # ── Internalization state ──
    lines.append("\n## Final Internalization State\n\n")
    lines.append("| θ | Strategy | κ_final | η_final | γ_final |\n")
    lines.append("|---|----------|---------|---------|--------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | {} |\n".format(
                    theta, s,
                    sf(a["kappa_final"], "{:.3f}"),
                    sf(a["eta_final"], "{:.3f}"),
                    sf(a["gamma_final"], "{:.3f}")))

    with open(out / "tic_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/tic_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
