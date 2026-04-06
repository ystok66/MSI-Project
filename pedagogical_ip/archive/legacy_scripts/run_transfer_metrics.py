"""Unified Decision-Aware Metrics + Transfer Experiment.

Structure:
  Phase 1: Tutor-guided sessions (PP-MRB, 12 episodes)
            → collect OAA, TimingRegret, ECE, SelGap
  Phase 2: Remove tutor, run 6 more episodes (same distribution)
            → collect TransferSBCR, AutonomyGain
  Phase 3: Remove tutor, run 6 more episodes (shifted distribution)
            → collect TransferShiftSBCR

5 conditions: no_tutor, v1_1, cajt_v3, factor_cajt, oracle
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario, EPISODE_SUBTYPES,
)
from src.envs.compositional_goal_corridor_v2 import (
    generate_cgc2_session, generate_cgc2_scenario, CGC2_SUBTYPES,
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
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.metrics.decision_aware_metrics import (
    compute_oaa, compute_timing_regret, compute_ece,
    compute_transfer_sbcr, compute_autonomy_gain,
    compute_intervention_efficiency,
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


def run_episode(ep, theta, tutor, tutor_name, lp, lib, scorer):
    """Run a single PP-MRB episode. Returns trace dict."""
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

    bas = BranchAttributes(
        safety_score=float(ss[0]),
        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
        risk_penalty=0.1)
    bar = BranchAttributes(
        safety_score=float(sr[0]),
        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
        risk_penalty=0.4)
    branches = [bas, bar]
    agent_choice = sample_branch_choice(branches, theta, AP, rng)

    do_warn = False
    diag = {}
    if tutor_name == "oracle":
        do_warn = (ep.d_commit < ep.d_reveal)
    elif tutor_name == "no_tutor":
        do_warn = False
    elif tutor is not None:
        action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
        do_warn = (action == "WARN")
        if hasattr(tutor, 'observe_agent_choice'):
            tutor.observe_agent_choice(agent_choice, branches)

    if do_warn:
        for r, c in sc.risky_cells:
            z = fb[r, c]
            lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)

    # Posterior confidence for ECE
    pred_conf = 0.5
    pred_correct = 0.0
    if tutor is not None:
        if hasattr(tutor, 'pref_posterior'):
            pred_conf = tutor.pref_posterior.predicted_prob
            pred_correct = 1.0 if tutor.pref_posterior.predicted_type == theta else 0.0
        elif hasattr(tutor, 'joint_posterior'):
            _, pp = tutor.joint_posterior.predicted_joint
            pred_conf = tutor.joint_posterior.joint_confidence
            pred_correct = 1.0 if pp == theta else 0.0
            if hasattr(tutor, '_calibrated_table'):
                ct = tutor._calibrated_table()
                pred_conf = float(np.max(ct))

    return {
        "subtype": ep.episode_subtype,
        "warned": do_warn,
        "agent_safe": (agent_choice == 0),
        "d_commit": ep.d_commit,
        "d_reveal": ep.d_reveal,
        "pred_conf": pred_conf,
        "pred_correct": pred_correct,
    }


def run_full_transfer(tutor_name, theta, seed=0):
    """Phase 1: tutor-guided (12 ep), Phase 2: no-tutor same (6 ep), Phase 3: no-tutor shift (6 ep)."""
    # Phase 1: Tutor-guided
    session = generate_session(seed * 1000 + abs(hash(theta)) % 1000, 12, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(tutor_name)

    guided_traces = []
    for ep in session.episodes:
        t = run_episode(ep, theta, tutor, tutor_name, lp, lib, scorer)
        guided_traces.append(t)

    # Phase 2: Remove tutor, same distribution
    session_p2 = generate_session(seed * 1000 + abs(hash(theta)) % 1000 + 200, 6, theta)
    transfer_same = []
    for ep in session_p2.episodes:
        t = run_episode(ep, theta, None, "no_tutor", lp, lib, scorer)
        transfer_same.append(t)

    # Phase 3: Remove tutor, shifted distribution (different theta mix)
    shift_theta = "neutral" if theta != "neutral" else "shortcut"
    session_p3 = generate_session(seed * 1000 + abs(hash(theta)) % 1000 + 400, 6, theta)
    transfer_shift = []
    for ep in session_p3.episodes:
        rng_shift = np.random.default_rng(ep.cue_layout_seed + 7777)
        active_theta = shift_theta if rng_shift.random() < 0.3 else theta
        t = run_episode(ep, active_theta, None, "no_tutor", lp, lib, scorer)
        transfer_shift.append(t)

    # Compute metrics
    if not guided_traces:
        return {}

    # OAA
    oaa = compute_oaa(guided_traces)

    # Timing Regret
    tr = compute_timing_regret(guided_traces)

    # ECE
    confs = [t["pred_conf"] for t in guided_traces]
    corrs = [t["pred_correct"] for t in guided_traces]
    ece = compute_ece(confs, corrs)

    # SelGap
    n = len(guided_traces)
    sbcr_g = sum(1 for t in guided_traces if t["agent_safe"]) / n
    wr_g = sum(1 for t in guided_traces if t["warned"]) / n
    sw = {}
    for st in EPISODE_SUBTYPES:
        eps = [t for t in guided_traces if t["subtype"] == st]
        sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
    sg = None
    if sw.get("warn_trap") is not None and sw.get("wait_clean") is not None:
        sg = sw["warn_trap"] - sw["wait_clean"]

    ie = compute_intervention_efficiency(sbcr_g, wr_g)

    # Transfer
    sbcr_same = compute_transfer_sbcr(transfer_same)
    sbcr_shift = compute_transfer_sbcr(transfer_shift)

    # Pre-tutor baseline (first 3 guided episodes as proxy)
    pre_sbcr = sum(1 for t in guided_traces[:3] if t["agent_safe"]) / max(len(guided_traces[:3]), 1)
    auto_gain = compute_autonomy_gain(pre_sbcr, sbcr_same)

    return {
        "strategy": tutor_name,
        "oaa": round(oaa, 3),
        "tr": round(tr, 3),
        "ece": round(ece, 3),
        "sg": round(sg, 3) if sg is not None else None,
        "sbcr_g": round(sbcr_g, 3),
        "wr_g": round(wr_g, 3),
        "ie": round(ie, 3),
        "sbcr_same": round(sbcr_same, 3),
        "sbcr_shift": round(sbcr_shift, 3),
        "auto_gain": round(auto_gain, 3),
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ Decision-Aware Metrics + Transfer ═══\n", file=sys.stderr)
    strategies = ["no_tutor", "v1_1", "cajt_v3", "oracle"]
    thetas = ["safe", "shiny"]
    lines = ["# Decision-Aware Metrics + Transfer Report\n\n"]

    # ── Main experiment ──
    print("Phase 1-3: Guided → Transfer (same) → Transfer (shift)...", file=sys.stderr)
    lines.append("## Unified Metrics (PP-MRB)\n\n")
    lines.append("| θ | Strategy | OAA | TR | ECE | SelGap | SBCR(g) | WR | IE | SBCR(same) | SBCR(shift) | AutoGain |\n")
    lines.append("|---|----------|-----|-----|-----|--------|---------|-----|-----|------------|-------------|----------|\n")

    all_results = []
    for theta in thetas:
        for s in strategies:
            rs = [run_full_transfer(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["oaa", "tr", "ece", "sg", "sbcr_g", "wr_g",
                                           "ie", "sbcr_same", "sbcr_shift", "auto_gain"]}
            a["theta"] = theta; a["strategy"] = s
            all_results.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, s,
                sf(a["oaa"], "{:.3f}"), sf(a["tr"], "{:.3f}"),
                sf(a["ece"], "{:.3f}"), sf(a["sg"], "{:.3f}"),
                sf(a["sbcr_g"]), sf(a["wr_g"]),
                sf(a["ie"], "{:.2f}"),
                sf(a["sbcr_same"]), sf(a["sbcr_shift"]),
                sf(a["auto_gain"], "{:.3f}")))
            print(f"  {theta} × {s}: OAA={sf(a['oaa'], '{:.3f}')} TR={sf(a['tr'], '{:.3f}')} "
                  f"TransSame={sf(a['sbcr_same'])} AutoGain={sf(a['auto_gain'], '{:.3f}')}",
                  file=sys.stderr)

    # ── Key comparisons ──
    lines.append("\n## Key Comparisons\n\n")

    # OAA comparison
    lines.append("### Oracle Action Agreement\n\n")
    lines.append("| θ | no_tutor | v1.1 | **cajt_v3** | oracle |\n")
    lines.append("|---|---------|------|------------|--------|\n")
    for theta in thetas:
        vals = {}
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            vals[s] = r[0]["oaa"] if r else None
        lines.append("| {} | {} | {} | **{}** | {} |\n".format(
            theta, sf(vals["no_tutor"], "{:.3f}"),
            sf(vals["v1_1"], "{:.3f}"), sf(vals["cajt_v3"], "{:.3f}"),
            sf(vals["oracle"], "{:.3f}")))

    # Transfer comparison
    lines.append("\n### Transfer: Tutor→No-Tutor\n\n")
    lines.append("| θ | Strategy | SBCR(guided) | SBCR(post,same) | SBCR(post,shift) | **AutoGain** |\n")
    lines.append("|---|----------|-------------|-----------------|------------------|--------------|\n")
    for theta in thetas:
        for s in strategies:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            if r:
                a = r[0]
                lines.append("| {} | {} | {} | {} | {} | **{}** |\n".format(
                    theta, s,
                    sf(a["sbcr_g"]), sf(a["sbcr_same"]),
                    sf(a["sbcr_shift"]), sf(a["auto_gain"], "{:.3f}")))

    # ECE comparison
    lines.append("\n### Posterior Calibration (ECE)\n\n")
    lines.append("| θ | v1.1 | **cajt_v3** |\n")
    lines.append("|---|------|------------|\n")
    for theta in thetas:
        vals = {}
        for s in ["v1_1", "cajt_v3"]:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            vals[s] = r[0]["ece"] if r else None
        lines.append("| {} | {} | **{}** |\n".format(
            theta, sf(vals["v1_1"], "{:.3f}"), sf(vals["cajt_v3"], "{:.3f}")))

    # Intervention efficiency
    lines.append("\n### Intervention Efficiency (IE = SBCR / WarnRate)\n\n")
    lines.append("| θ | v1.1 IE | **cajt_v3 IE** | oracle IE |\n")
    lines.append("|---|---------|---------------|----------|\n")
    for theta in thetas:
        vals = {}
        for s in ["v1_1", "cajt_v3", "oracle"]:
            r = [x for x in all_results if x["theta"] == theta and x["strategy"] == s]
            vals[s] = r[0]["ie"] if r else None
        lines.append("| {} | {} | **{}** | {} |\n".format(
            theta, sf(vals["v1_1"], "{:.2f}"),
            sf(vals["cajt_v3"], "{:.2f}"), sf(vals["oracle"], "{:.2f}")))

    with open(out / "transfer_metrics_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/transfer_metrics_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
