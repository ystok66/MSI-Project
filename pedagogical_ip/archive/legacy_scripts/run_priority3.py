"""Priority 3: DCE + Strong Drift + Cross-Family Teaching Transfer.

Part A: Decision Calibration Error (DCE)
  DCE = E[|C_t - 1[action is oracle-compatible]|]
  Compare v1.1, cajt_v3, hier_cajt across PP-MRB and TIC

Part B: Strong Drift Robustness (3 types)
  1. Abrupt: θ switches at episode 6 (safe→shiny)
  2. Gradual: θ mix shifts linearly over 16 episodes
  3. Intra-episode: θ flip mid-session with reset marker

Part C: Cross-Family Teaching Transfer
  Train tutor on PP-MRB (8 ep) → test on TIC (8 ep no-tutor)
  vs train on TIC → test on TIC
  Measure: TransferSBCR, AutonomyGain
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario, EPISODE_SUBTYPES,
)
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
    BranchAttributes, AgentPolicyParams, sample_branch_choice, PREFERENCE_TYPES,
)
from src.agents.internalization_agent import (
    InternalizationState, sample_pia_choice,
)
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.metrics.decision_aware_metrics import compute_oaa, compute_ece
from src.metrics.pedagogical_metrics import (
    learning_gain, transfer_improvement, pedagogical_efficiency,
    compute_decision_calibration_error,
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


def run_ppmrb_episode(ep, theta, tutor, tutor_name, lp, lib, scorer):
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

    bas = BranchAttributes(safety_score=float(ss[0]),
        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
        risk_penalty=0.1)
    bar = BranchAttributes(safety_score=float(sr[0]),
        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
        risk_penalty=0.4)
    branches = [bas, bar]
    ac = sample_branch_choice(branches, theta, AP, rng)

    do_warn = False
    C_t = 0.5
    if tutor_name == "oracle":
        do_warn = (ep.d_commit <= ep.d_reveal)
    elif tutor_name == "no_tutor":
        do_warn = False
    elif tutor is not None:
        action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
        do_warn = (action == "WARN")
        C_t = diag.get("C_t", 0.5) if isinstance(diag, dict) else 0.5
        if hasattr(tutor, 'observe_agent_choice'):
            tutor.observe_agent_choice(ac, branches)

    oracle_act = "WARN" if ep.d_commit <= ep.d_reveal else "WAIT"
    tutor_act = "WARN" if do_warn else "WAIT"
    oracle_compat = (tutor_act == oracle_act)

    return {
        "warned": do_warn, "agent_safe": (ac == 0),
        "d_commit": ep.d_commit, "d_reveal": ep.d_reveal,
        "C_t": C_t, "oracle_compat": oracle_compat,
        "subtype": ep.episode_subtype,
    }


def run_tic_episode(ep, theta, tutor, tutor_name, lp, lib, scorer, m):
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
    if ep.phase != "A":
        do_warn = False
    elif tutor_name == "no_tutor":
        do_warn = False
    elif tutor_name == "oracle":
        do_warn = (ep.d_commit <= ep.d_reveal)
    elif tutor is not None:
        action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
        do_warn = (action == "WARN")
        if hasattr(tutor, 'observe_agent_choice'):
            wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
            ac = sample_pia_choice(branches, theta, m, AP, rng, wb)
            tutor.observe_agent_choice(ac, branches)

    if tutor is None or tutor_name in ("no_tutor", "oracle"):
        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_pia_choice(branches, theta, m, AP, rng, wb)
    elif not (ep.phase != "A"):
        pass  # ac already set above
    else:
        wb = [0.0, 0.0]
        ac = sample_pia_choice(branches, theta, m, AP, rng, wb)

    if do_warn:
        for r, c in sc.risky_cells:
            z = fb[r, c]
            lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)

    chose_risky = (ac != sc.oracle_safe_branch_id)
    real_risk = sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05
    m.update_risk(real_risk, 0.15)
    m.update_trust(do_warn)
    tempt_high = branches[1 - sc.oracle_safe_branch_id].temptation_score > 0.5
    m.update_suppression(chose_risky and tempt_high)
    m.snapshot()

    return {
        "warned": do_warn, "agent_safe": (ac == sc.oracle_safe_branch_id),
        "phase": ep.phase, "subtype": ep.subtype,
    }


# ═══════════════════════════════════
# Part A: Decision Calibration Error
# ═══════════════════════════════════

def run_dce(strategy, theta, seed=0):
    session = generate_session(seed * 1000 + abs(hash(theta)) % 1000, 12, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    traces = []
    for ep in session.episodes:
        t = run_ppmrb_episode(ep, theta, tutor, strategy, lp, lib, scorer)
        traces.append(t)
    confs = [t["C_t"] for t in traces]
    oracle_compat = [t["oracle_compat"] for t in traces]
    dce = compute_decision_calibration_error(confs, oracle_compat)
    oaa = sum(1 for t in traces if t["oracle_compat"]) / max(len(traces), 1)
    return {"dce": round(dce, 3), "oaa": round(oaa, 3)}


# ═══════════════════════════════════
# Part B: Strong Drift
# ═══════════════════════════════════

def run_drift(strategy, drift_type, seed=0):
    n_ep = 16
    if drift_type == "abrupt":
        thetas = ["safe"] * 8 + ["shiny"] * 8
    elif drift_type == "gradual":
        rng_d = np.random.default_rng(seed + 77)
        thetas = []
        for i in range(n_ep):
            p_shiny = i / (n_ep - 1)
            thetas.append("shiny" if rng_d.random() < p_shiny else "safe")
    else:  # intra
        thetas = []
        rng_d = np.random.default_rng(seed + 88)
        for i in range(n_ep):
            if i % 4 == 3:
                thetas.append("shiny" if rng_d.random() < 0.7 else "safe")
            else:
                thetas.append("safe")

    session = generate_session(seed * 1000, n_ep, "safe")
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)

    traces_pre, traces_post = [], []
    for i, ep in enumerate(session.episodes):
        theta_i = thetas[i]
        t = run_ppmrb_episode(ep, theta_i, tutor, strategy, lp, lib, scorer)
        if i < 8:
            traces_pre.append(t)
        else:
            traces_post.append(t)

    def sg(traces):
        sw = {}
        for st in EPISODE_SUBTYPES:
            eps = [t for t in traces if t["subtype"] == st]
            sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
        if sw.get("warn_trap") is not None and sw.get("wait_clean") is not None:
            return round(sw["warn_trap"] - sw["wait_clean"], 3)
        return None

    sg_pre = sg(traces_pre)
    sg_post = sg(traces_post)
    recovery = None
    if sg_pre is not None and sg_post is not None:
        recovery = round(sg_post - sg_pre, 3)

    return {
        "sg_pre": sg_pre, "sg_post": sg_post, "recovery": recovery,
        "sbcr_pre": round(sum(1 for t in traces_pre if t["agent_safe"]) / max(len(traces_pre), 1), 3),
        "sbcr_post": round(sum(1 for t in traces_post if t["agent_safe"]) / max(len(traces_post), 1), 3),
    }


# ═══════════════════════════════════
# Part C: Cross-Family Teaching Transfer
# ═══════════════════════════════════

def run_cross_family(train_family, strategy, theta, seed=0):
    """Train tutor on one family, test PIA transfer on TIC."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    m = InternalizationState()
    m_pre = m.copy()

    # Phase 1: Train on source family
    if train_family == "ppmrb":
        session = generate_session(seed * 1000 + abs(hash(theta)) % 1000, 8, theta)
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

            bas = BranchAttributes(safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=0.4)
            branches = [bas, bar]

            do_warn = False
            if strategy != "no_tutor" and tutor is not None:
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                do_warn = (action == "WARN")

            wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
            ac = sample_pia_choice(branches, theta, m, AP, rng, wb)
            if tutor is not None and hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(ac, branches)

            chose_risky = (ac != sc.oracle_safe_branch_id)
            real_risk = 0.35 if chose_risky else 0.05
            m.update_risk(real_risk, 0.15)
            m.update_trust(do_warn)
            m.update_suppression(chose_risky and bar.temptation_score > 0.5)
            m.snapshot()

    elif train_family == "tic":
        tic_sess = generate_tic_session(seed * 1000 + abs(hash(theta)) % 1000, theta, 8, 0, 0)
        for ep in tic_sess.episodes:
            run_tic_episode(ep, theta, tutor, strategy, lp, lib, scorer, m)

    # Phase 2: Test on TIC (no tutor, 8 ep)
    tic_test = generate_tic_session(seed * 1000 + abs(hash(theta)) % 1000 + 500, theta, 0, 8, 0)
    test_traces = []
    for ep in tic_test.episodes:
        t = run_tic_episode(ep, theta, None, "no_tutor", lp, lib, scorer, m)
        test_traces.append(t)

    m_post = m.copy()
    lg = learning_gain(m_pre, m_post)
    test_sbcr = sum(1 for t in test_traces if t["agent_safe"]) / max(len(test_traces), 1)

    return {
        "test_sbcr": round(test_sbcr, 3),
        "lg_total": lg["lg_total"],
        "kappa_f": round(m.kappa, 3),
        "eta_f": round(m.eta, 3),
        "gamma_f": round(m.gamma, 3),
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ Priority 3: DCE + Drift + Cross-Family ═══\n", file=sys.stderr)
    lines = ["# Priority 3: DCE + Strong Drift + Cross-Family Transfer\n\n"]

    # ── Part A: DCE ──
    print("Part A: Decision Calibration Error...", file=sys.stderr)
    lines.append("## A. Decision Calibration Error (DCE)\n\n")
    lines.append("DCE = E[|C_t − 1[oracle-compatible]|]\n\n")
    lines.append("| θ | Strategy | DCE | OAA |\n")
    lines.append("|---|----------|-----|-----|\n")
    for theta in ["safe", "shiny"]:
        for s in ["v1_1", "cajt_v3"]:
            rs = [run_dce(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["dce", "oaa"]}
            lines.append("| {} | {} | {} | {} |\n".format(
                theta, s, sf(a["dce"], "{:.3f}"), sf(a["oaa"], "{:.3f}")))
            print(f"  {theta} × {s}: DCE={sf(a['dce'], '{:.3f}')} OAA={sf(a['oaa'], '{:.3f}')}",
                  file=sys.stderr)

    # ── Part B: Strong Drift ──
    print("\nPart B: Strong Drift...", file=sys.stderr)
    lines.append("\n## B. Strong Drift Robustness\n\n")
    for drift_type in ["abrupt", "gradual", "intra"]:
        lines.append(f"### {drift_type.title()} Drift\n\n")
        lines.append("| Strategy | SG(pre) | SG(post) | Δ(recovery) | SBCR(pre) | SBCR(post) |\n")
        lines.append("|----------|---------|----------|-------------|-----------|------------|\n")
        for s in ["v1_1", "cajt_v3"]:
            rs = [run_drift(s, drift_type, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["sg_pre", "sg_post", "recovery", "sbcr_pre", "sbcr_post"]}
            lines.append("| {} | {} | {} | {} | {} | {} |\n".format(
                s, sf(a["sg_pre"], "{:.3f}"), sf(a["sg_post"], "{:.3f}"),
                sf(a["recovery"], "{:+.3f}"),
                sf(a["sbcr_pre"]), sf(a["sbcr_post"])))
            print(f"  {drift_type} × {s}: pre={sf(a['sg_pre'], '{:.3f}')} "
                  f"post={sf(a['sg_post'], '{:.3f}')} Δ={sf(a['recovery'], '{:+.3f}')}",
                  file=sys.stderr)

    # ── Part C: Cross-Family Teaching Transfer ──
    print("\nPart C: Cross-Family Transfer...", file=sys.stderr)
    lines.append("\n## C. Cross-Family Teaching Transfer\n\n")
    lines.append("Train on source family (8 ep) → test on TIC (8 ep, no tutor)\n\n")
    lines.append("| θ | Train Family | Strategy | Test SBCR | LG_total | κ_f | η_f | γ_f |\n")
    lines.append("|---|-------------|----------|-----------|----------|-----|-----|-----|\n")
    for theta in ["safe", "shiny"]:
        for train_fam in ["ppmrb", "tic"]:
            for s in ["no_tutor", "v1_1", "cajt_v3"]:
                rs = [run_cross_family(train_fam, s, theta, sid) for sid in range(8)]
                a = {k: avg(rs, k) for k in ["test_sbcr", "lg_total", "kappa_f", "eta_f", "gamma_f"]}
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                    theta, train_fam, s,
                    sf(a["test_sbcr"]), sf(a["lg_total"], "{:.4f}"),
                    sf(a["kappa_f"], "{:.3f}"), sf(a["eta_f"], "{:.3f}"),
                    sf(a["gamma_f"], "{:.3f}")))
                print(f"  {theta} {train_fam}→TIC × {s}: SBCR={sf(a['test_sbcr'])} "
                      f"LG={sf(a['lg_total'], '{:.4f}')}", file=sys.stderr)

    # ── Cross-family Δ comparison ──
    lines.append("\n### Transfer Improvement: PP-MRB→TIC vs TIC→TIC\n\n")
    lines.append("| θ | Strategy | SBCR(ppmrb→tic) | SBCR(tic→tic) | Δ |\n")
    lines.append("|---|----------|-----------------|---------------|---|\n")
    # Re-extract from above
    for theta in ["safe", "shiny"]:
        for s in ["v1_1", "cajt_v3"]:
            pm = [run_cross_family("ppmrb", s, theta, sid) for sid in range(4)]
            tc = [run_cross_family("tic", s, theta, sid) for sid in range(4)]
            sbcr_pm = avg(pm, "test_sbcr")
            sbcr_tc = avg(tc, "test_sbcr")
            delta = round(sbcr_pm - sbcr_tc, 3) if sbcr_pm is not None and sbcr_tc is not None else None
            lines.append("| {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(sbcr_pm), sf(sbcr_tc), sf(delta, "{:+.3f}")))

    with open(out / "priority3_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/priority3_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
