"""CAJT-v3 Experiment: Calibrated Adaptive Joint Tutor.

6 conditions × 3 families + stress tests.

Conditions:
  1. v1_1_persistent
  2. joint_v2_coupled
  3. cajt_v3_no_cal    (adaptive only)
  4. cajt_v3_no_adapt  (calibration only)
  5. cajt_v3_full
  6. oracle

Stress:
  A. Session-order shuffle (cajt_v3 vs v1.1 vs joint_v2)
  B. Mild latent drift (θ switches mid-session)
  C. Wrong-memory recovery speed
  D. Calibration gap comparison
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario, EPISODE_SUBTYPES,
)
from src.envs.compositional_goal_corridor import (
    generate_cgc_session, generate_cgc_scenario, CGC_SUBTYPES,
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
    BranchAttributes, AgentPolicyParams, sample_branch_choice, PREFERENCE_TYPES,
)
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.joint_tutor_v2 import JointTutorV2
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3

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
    if name == "joint_v2": return JointTutorV2(agent_params=AP)
    if name == "cajt_v3_no_cal":
        return CAJTv3(agent_params=AP, enable_calibration=False)
    if name == "cajt_v3_no_adapt":
        return CAJTv3(agent_params=AP, enable_adaptive=False)
    if name == "cajt_v3_full": return CAJTv3(agent_params=AP)
    return None


def run_ppmrb(tutor_name, theta, n_eps=12, seed=0, drift_at=None):
    """Run PP-MRB. If drift_at, switch theta mid-session."""
    session = generate_session(seed * 1000 + abs(hash(theta)) % 1000,
                               n_eps, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(tutor_name)
    traces = []

    for ep_i, ep in enumerate(session.episodes):
        # Apply drift
        active_theta = theta
        if drift_at is not None and ep_i >= drift_at:
            active_theta = "neutral" if theta != "neutral" else "shortcut"

        gm, cfg, meta, sc = generate_episode_scenario(ep, theta)
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

        ba_s = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        ba_r = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=0.4)
        branches = [ba_s, ba_r]
        agent_choice = sample_branch_choice(branches, active_theta, AP, rng)

        do_warn = False
        diag = {}
        if tutor_name == "oracle":
            do_warn = (ep.d_commit < ep.d_reveal)
        elif tutor is not None:
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
            if hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(agent_choice, branches)

        traces.append({
            "subtype": ep.episode_subtype, "warned": do_warn,
            "agent_safe": (agent_choice == 0),
            "C_t": diag.get("C_t"), "D_t": diag.get("D_t"),
            "adapt_val": diag.get("adapt_val"), "calib_top1": diag.get("calib_top1"),
        })

    n = len(traces)
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n
    sw = {}
    for st in EPISODE_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
    sg = (sw.get("warn_trap", 0) or 0) - (sw.get("wait_clean", 0) or 0) if sw.get("warn_trap") is not None and sw.get("wait_clean") is not None else None

    ct_vals = [t["C_t"] for t in traces if t["C_t"] is not None]
    ct_avg = round(np.mean(ct_vals), 4) if ct_vals else None
    cal_vals = [t["calib_top1"] for t in traces if t["calib_top1"] is not None]
    cal_avg = round(np.mean(cal_vals), 4) if cal_vals else None
    dt_vals = [t["D_t"] for t in traces if t["D_t"] is not None]
    dt_avg = round(np.mean(dt_vals), 4) if dt_vals else None

    return {"sbcr": round(sbcr, 3), "wr": round(wr, 3), "sg": round(sg, 3) if sg is not None else None,
            "wr_wc": round(sw.get("wait_clean", 0), 3) if sw.get("wait_clean") is not None else None,
            "wr_wt": round(sw.get("warn_trap", 0), 3) if sw.get("warn_trap") is not None else None,
            "C_t": ct_avg, "cal_top1": cal_avg, "D_t": dt_avg}


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ CAJT-v3 Experiment ═══\n", file=sys.stderr)
    strategies = ["v1_1", "joint_v2", "cajt_v3_no_cal", "cajt_v3_no_adapt",
                   "cajt_v3_full", "oracle"]
    thetas = ["safe", "shiny"]
    lines = ["# CAJT-v3: Calibrated Adaptive Joint Tutor\n\n"]

    # ── Main experiment: PP-MRB ──
    print("A. PP-MRB main...", file=sys.stderr)
    lines.append("## A. PP-MRB Main Results\n\n")
    lines.append("| θ | Strategy | SBCR | WR | WR(wc) | WR(wt) | **SelGap** | C_t | CalTop1 | D_t |\n")
    lines.append("|---|----------|------|----|--------|--------|-----------|-----|---------|-----|\n")
    pp_results = []
    for theta in thetas:
        for s in strategies:
            rs = [run_ppmrb(s, theta, seed=sid) for sid in range(6)]
            a = {k: avg(rs, k) for k in ["sbcr", "wr", "sg", "wr_wc", "wr_wt", "C_t", "cal_top1", "D_t"]}
            a["theta"] = theta; a["strategy"] = s
            pp_results.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr"]), sf(a["wr"]),
                sf(a["wr_wc"]), sf(a["wr_wt"]),
                sf(a["sg"], "{:.3f}"), sf(a["C_t"], "{:.4f}"),
                sf(a["cal_top1"], "{:.4f}"), sf(a["D_t"], "{:.4f}")))
            print(f"  {theta} × {s}: SelGap={sf(a['sg'], '{:.3f}')}", file=sys.stderr)

    # SelGap summary
    lines.append("\n### SelGap Comparison\n\n")
    lines.append("| θ | v1.1 | joint_v2 | v3_no_cal | v3_no_adapt | **v3_full** | oracle |\n")
    lines.append("|---|------|----------|-----------|-------------|------------|--------|\n")
    for theta in thetas:
        vals = {}
        for s in strategies:
            r = [x for x in pp_results if x["theta"] == theta and x["strategy"] == s]
            vals[s] = r[0]["sg"] if r else None
        lines.append("| {} | {} | {} | {} | {} | **{}** | {} |\n".format(
            theta,
            sf(vals["v1_1"], "{:.3f}"), sf(vals["joint_v2"], "{:.3f}"),
            sf(vals["cajt_v3_no_cal"], "{:.3f}"), sf(vals["cajt_v3_no_adapt"], "{:.3f}"),
            sf(vals["cajt_v3_full"], "{:.3f}"), sf(vals["oracle"], "{:.3f}")))

    # ── Stress B: Mild latent drift ──
    print("\nB. Mild drift...", file=sys.stderr)
    lines.append("\n## B. Mild Latent Drift (θ switches at episode 8)\n\n")
    lines.append("| θ→ | Strategy | SelGap(stable) | SelGap(drift) | |Δ| |\n")
    lines.append("|-----|----------|----------------|---------------|---------|\n")
    for theta in ["safe"]:
        for s in ["v1_1", "joint_v2", "cajt_v3_full"]:
            rs_stable = [run_ppmrb(s, theta, seed=sid) for sid in range(6)]
            rs_drift = [run_ppmrb(s, theta, seed=sid, drift_at=8) for sid in range(6)]
            sg_s = avg(rs_stable, "sg")
            sg_d = avg(rs_drift, "sg")
            delta = abs(sg_s - sg_d) if sg_s is not None and sg_d is not None else None
            lines.append("| {}→neutral | {} | {} | {} | {} |\n".format(
                theta, s, sf(sg_s, "{:.3f}"), sf(sg_d, "{:.3f}"), sf(delta, "{:.3f}")))

    # ── Stress C: Wrong-memory recovery ──
    print("\nC. Wrong-memory recovery...", file=sys.stderr)
    lines.append("\n## C. Wrong-Memory Recovery\n\n")
    lines.append("| θ | Strategy | SG(correct) | SG(adversarial) | Recovery |\n")
    lines.append("|---|----------|-------------|-----------------|----------|\n")
    for theta in ["safe", "shiny"]:
        for s in ["v1_1", "cajt_v3_full"]:
            rs_c = [run_ppmrb(s, theta, seed=sid) for sid in range(6)]
            # Adversarial: manually set wrong prior
            rs_w = []
            for sid in range(6):
                session = generate_session(sid * 1000 + abs(hash(theta)) % 1000, 12, theta)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
                tutor = make_tutor(s)
                wrong = "shiny" if theta != "shiny" else "safe"
                if hasattr(tutor, 'pref_posterior'):
                    wi = PREFERENCE_TYPES.index(wrong)
                    tutor.pref_posterior.log_probs[wi] = 5.0
                    tutor.pref_posterior.log_probs -= np.mean(tutor.pref_posterior.log_probs)
                elif hasattr(tutor, 'joint_posterior'):
                    from src.agents.goal_posterior_v1 import GOAL_TYPES as GT
                    from src.agents.joint_posterior_v2 import N_GOALS as NG, N_PREF as NP
                    wi = PREFERENCE_TYPES.index(wrong)
                    tutor.joint_posterior.log_table[:, wi] += 5.0
                    tutor.joint_posterior.log_table -= np.mean(tutor.joint_posterior.log_table)

                traces = []
                for ep in session.episodes:
                    gm, cfg, meta, sc = generate_episode_scenario(ep, theta)
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
                    ba_s = BranchAttributes(safety_score=float(ss[0]),
                        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
                    ba_r = BranchAttributes(safety_score=float(sr[0]),
                        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a, risk_penalty=0.4)
                    branches = [ba_s, ba_r]
                    ac = sample_branch_choice(branches, theta, AP, rng)
                    action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                    if hasattr(tutor, 'observe_agent_choice'):
                        tutor.observe_agent_choice(ac, branches)
                    traces.append({"subtype": ep.episode_subtype, "warned": (action == "WARN")})
                n = len(traces)
                sw = {}
                for st in EPISODE_SUBTYPES:
                    eps = [t for t in traces if t["subtype"] == st]
                    sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
                sg = (sw.get("warn_trap", 0) or 0) - (sw.get("wait_clean", 0) or 0) if sw.get("warn_trap") is not None and sw.get("wait_clean") is not None else None
                rs_w.append({"sg": sg})

            sg_c = avg(rs_c, "sg")
            sg_w = avg(rs_w, "sg")
            rec = "✅" if sg_w is not None and sg_c is not None and sg_w >= sg_c * 0.8 else "⚠️"
            lines.append("| {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(sg_c, "{:.3f}"), sf(sg_w, "{:.3f}"), rec))

    # ── Stress D: Calibration gap ──
    print("\nD. Calibration gap...", file=sys.stderr)
    lines.append("\n## D. Calibration Gap (PredTop1 vs ActualCorrect)\n\n")
    lines.append("| θ | Strategy | PredTop1 | ActualCorr | |Gap| |\n")
    lines.append("|---|----------|----------|------------|--------|\n")
    for theta in ["safe", "shiny"]:
        for s in ["v1_1", "joint_v2", "cajt_v3_full"]:
            confs, corrs = [], []
            for sid in range(6):
                session = generate_session(sid * 1000 + abs(hash(theta)) % 1000, 12, theta)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
                tutor = make_tutor(s)
                for ep in session.episodes:
                    gm, cfg, meta, sc = generate_episode_scenario(ep, theta)
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
                    ba_s = BranchAttributes(safety_score=float(ss[0]),
                        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
                    ba_r = BranchAttributes(safety_score=float(sr[0]),
                        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a, risk_penalty=0.4)
                    branches = [ba_s, ba_r]
                    ac = sample_branch_choice(branches, theta, AP, rng)
                    _, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                    if hasattr(tutor, 'observe_agent_choice'):
                        tutor.observe_agent_choice(ac, branches)
                # Final prediction
                if hasattr(tutor, 'pref_posterior'):
                    confs.append(tutor.pref_posterior.predicted_prob)
                    corrs.append(1.0 if tutor.pref_posterior.predicted_type == theta else 0.0)
                elif hasattr(tutor, 'joint_posterior'):
                    _, pp = tutor.joint_posterior.predicted_joint
                    confs.append(tutor.joint_posterior.joint_confidence)
                    corrs.append(1.0 if pp == theta else 0.0)
                    if hasattr(tutor, '_calibrated_table'):
                        ct = tutor._calibrated_table()
                        confs[-1] = float(np.max(ct))
            ac = np.mean(confs) if confs else 0
            ar = np.mean(corrs) if corrs else 0
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |\n".format(
                theta, s, ac, ar, abs(ac - ar)))

    with open(out / "cajt_v3_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/cajt_v3_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
