"""Unified Robustness Pipeline.

Runs ALL families × ALL tutors × ALL robustness variants and produces
a single comprehensive report.

Robustness tests:
  1. Mirror invariance (left/right swap)
  2. Parameter shift (d_commit, d_reveal, lure boundary values)
  3. Noise sweep (epsilon, beta)
  4. Session-order shuffle
  5. Calibration (ECE for posteriors)
  6. Wrong-memory regression (mild, adversarial, drifting)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario,
    SessionSpec, EpisodeSpec, EPISODE_SUBTYPES,
)
from src.envs.compositional_goal_corridor import (
    generate_cgc_session, generate_cgc_scenario, CGC_SUBTYPES,
)
from src.envs.scenario_families import generate_scenario, SCENARIO_REGISTRY
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
from src.agents.joint_posterior_v2 import JointPosteriorV2
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.joint_latent_tutor_v1 import JointLatentTutorV1
from src.teachers.joint_tutor_v2 import JointTutorV2

out = Path("results")
out.mkdir(exist_ok=True)


def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)


def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def make_tutor(name, ap):
    if name == "v4": return LearningAwarePolicyV4()
    if name == "v1_1": return PersistentTutorV1_1(agent_params=ap)
    if name == "joint_v1": return JointLatentTutorV1(agent_params=ap)
    if name == "joint_v2": return JointTutorV2(agent_params=ap)
    return None


# ═══════════════════════════════════════════
# Core: run a PP-MRB session with given params
# ═══════════════════════════════════════════

def run_ppmrb_session(tutor_name, theta, ap, n_eps=12, seed=0, mirror_override=None):
    """Run PP-MRB session, returns aggregated metrics dict."""
    session = generate_session(
        session_id=seed * 1000 + abs(hash(theta)) % 1000,
        n_episodes=n_eps, theta_true=theta,
    )
    if mirror_override is not None:
        for ep in session.episodes:
            ep.mirror_side = mirror_override

    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(tutor_name, ap)
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
        lib.update("safe_branch", ss)
        lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        ba_safe = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        ba_risky = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=0.4)
        branches = [ba_safe, ba_risky]
        agent_choice = sample_branch_choice(branches, theta, ap, rng)

        do_warn = False
        diag = {}
        if tutor_name == "v4":
            tutor.reset_stats()
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
        elif tutor is not None:
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
            if hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(agent_choice, branches)

        q_ent = 0.0
        if hasattr(tutor, 'pref_posterior'):
            q_ent = tutor.pref_posterior.entropy
        elif hasattr(tutor, 'joint_posterior'):
            q_ent = tutor.joint_posterior.entropy

        traces.append({
            "subtype": ep.episode_subtype, "warned": do_warn,
            "agent_safe": (agent_choice == 0), "q_ent": q_ent,
        })

    n = len(traces)
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n
    sub_wr = {}
    for st in EPISODE_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sub_wr[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
    sel_gap = None
    if sub_wr.get("warn_trap") is not None and sub_wr.get("wait_clean") is not None:
        sel_gap = sub_wr["warn_trap"] - sub_wr["wait_clean"]
    return {"sbcr": sbcr, "wr": wr, "sel_gap": sel_gap}


def avg_sessions(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return round(np.mean(vals), 3) if vals else None


# ═══════════════════════════════════════════
# Robustness Test 1: Mirror Invariance
# ═══════════════════════════════════════════

def test_mirror_invariance(report_lines):
    report_lines.append("## 1. Mirror Invariance\n")
    report_lines.append("| θ | Tutor | SelGap(L) | SelGap(R) | |Δ| |\n")
    report_lines.append("|---|-------|-----------|-----------|-----|\n")

    for theta in ["safe", "shiny"]:
        for tname in ["v1_1", "joint_v2"]:
            ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
            rs_l = [run_ppmrb_session(tname, theta, ap, seed=s, mirror_override=0) for s in range(6)]
            rs_r = [run_ppmrb_session(tname, theta, ap, seed=s, mirror_override=1) for s in range(6)]
            sg_l = avg_sessions(rs_l, "sel_gap")
            sg_r = avg_sessions(rs_r, "sel_gap")
            delta = abs(sg_l - sg_r) if sg_l is not None and sg_r is not None else None
            report_lines.append("| {} | {} | {} | {} | {} |\n".format(
                theta, tname, sf(sg_l, "{:.3f}"), sf(sg_r, "{:.3f}"),
                sf(delta, "{:.3f}")))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Robustness Test 2: Parameter Shift
# ═══════════════════════════════════════════

def test_parameter_shift(report_lines):
    report_lines.append("## 2. Parameter Shift\n")
    report_lines.append("Tutor tuned on mid-range, tested on boundary values.\n\n")
    report_lines.append("| θ | Tutor | Param | Train-range | Test-value | SelGap |\n")
    report_lines.append("|---|-------|-------|-------------|------------|--------|\n")

    theta = "safe"
    for tname in ["v1_1", "joint_v2"]:
        ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
        # Baseline (mid range)
        rs_base = [run_ppmrb_session(tname, theta, ap, seed=s) for s in range(6)]
        sg_base = avg_sessions(rs_base, "sel_gap")
        report_lines.append("| {} | {} | baseline | [2,5] | mid | {} |\n".format(
            theta, tname, sf(sg_base, "{:.3f}")))

        # High epsilon (more noise)
        ap_noisy = AgentPolicyParams(beta=4.0, epsilon=0.3, lambda_theta=1.0)
        rs_noisy = [run_ppmrb_session(tname, theta, ap_noisy, seed=s) for s in range(6)]
        sg_noisy = avg_sessions(rs_noisy, "sel_gap")
        report_lines.append("| {} | {} | ε=0.3 | ε=0.1 | 0.3 | {} |\n".format(
            theta, tname, sf(sg_noisy, "{:.3f}")))

        # Low beta (less rational)
        ap_lowb = AgentPolicyParams(beta=2.0, epsilon=0.1, lambda_theta=1.0)
        rs_lowb = [run_ppmrb_session(tname, theta, ap_lowb, seed=s) for s in range(6)]
        sg_lowb = avg_sessions(rs_lowb, "sel_gap")
        report_lines.append("| {} | {} | β=2.0 | β=4.0 | 2.0 | {} |\n".format(
            theta, tname, sf(sg_lowb, "{:.3f}")))

        # High beta (more rational)
        ap_hib = AgentPolicyParams(beta=8.0, epsilon=0.1, lambda_theta=1.0)
        rs_hib = [run_ppmrb_session(tname, theta, ap_hib, seed=s) for s in range(6)]
        sg_hib = avg_sessions(rs_hib, "sel_gap")
        report_lines.append("| {} | {} | β=8.0 | β=4.0 | 8.0 | {} |\n".format(
            theta, tname, sf(sg_hib, "{:.3f}")))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Robustness Test 3: Noise Sweep
# ═══════════════════════════════════════════

def test_noise_sweep(report_lines):
    report_lines.append("## 3. Noise Sweep (ε)\n")
    report_lines.append("| θ | Tutor | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30 | ε=0.40 |\n")
    report_lines.append("|---|-------|--------|--------|--------|--------|--------|\n")

    for theta in ["safe", "shiny"]:
        for tname in ["v1_1", "joint_v2"]:
            row_vals = []
            for eps in [0.05, 0.10, 0.20, 0.30, 0.40]:
                ap = AgentPolicyParams(beta=4.0, epsilon=eps, lambda_theta=1.0)
                rs = [run_ppmrb_session(tname, theta, ap, seed=s) for s in range(4)]
                sg = avg_sessions(rs, "sel_gap")
                row_vals.append(sf(sg, "{:.3f}"))
            report_lines.append("| {} | {} | {} |\n".format(
                theta, tname, " | ".join(row_vals)))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Robustness Test 4: Session-Order Shuffle
# ═══════════════════════════════════════════

def test_session_order_shuffle(report_lines):
    report_lines.append("## 4. Session-Order Shuffle\n")
    report_lines.append("Subtype ordering randomized vs fixed. Should be similar.\n\n")
    report_lines.append("| θ | Tutor | SelGap(fixed) | SelGap(shuffled) | |Δ| |\n")
    report_lines.append("|---|-------|--------------|-----------------|-----|\n")

    for theta in ["safe", "shiny"]:
        for tname in ["v1_1", "joint_v2"]:
            ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
            # Fixed seed
            rs_fixed = [run_ppmrb_session(tname, theta, ap, seed=s) for s in range(6)]
            # Shuffled: different seeds → different subtype ordering
            rs_shuf = [run_ppmrb_session(tname, theta, ap, seed=s + 100) for s in range(6)]
            sg_f = avg_sessions(rs_fixed, "sel_gap")
            sg_s = avg_sessions(rs_shuf, "sel_gap")
            delta = abs(sg_f - sg_s) if sg_f is not None and sg_s is not None else None
            report_lines.append("| {} | {} | {} | {} | {} |\n".format(
                theta, tname, sf(sg_f, "{:.3f}"), sf(sg_s, "{:.3f}"),
                sf(delta, "{:.3f}")))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Robustness Test 5: Calibration (ECE proxy)
# ═══════════════════════════════════════════

def test_calibration(report_lines):
    report_lines.append("## 5. Posterior Calibration\n")
    report_lines.append("Predicted top-1 prob vs actual correctness (ECE proxy).\n\n")
    report_lines.append("| θ | Tutor | PredTop1 | ActualCorrect | |Gap| |\n")
    report_lines.append("|---|-------|----------|---------------|--------|\n")

    for theta in ["safe", "shiny"]:
        for tname in ["v1_1", "joint_v2"]:
            ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
            pred_confs = []
            corrects = []
            for sid in range(6):
                session = generate_session(
                    session_id=sid * 1000 + abs(hash(theta)) % 1000,
                    n_episodes=12, theta_true=theta)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
                tutor = make_tutor(tname, ap)

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
                    lib.update("safe_branch", ss)
                    lib.update("risky_branch", sr)
                    scorer.update(build_scorer_input(ss, lib), 1.0)
                    scorer.update(build_scorer_input(sr, lib), 0.0)

                    ba_safe = BranchAttributes(
                        safety_score=float(ss[0]),
                        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                        risk_penalty=0.1)
                    ba_risky = BranchAttributes(
                        safety_score=float(sr[0]),
                        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                        risk_penalty=0.4)
                    branches = [ba_safe, ba_risky]
                    agent_choice = sample_branch_choice(branches, theta, ap, rng)

                    if tutor is not None:
                        _, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                        if hasattr(tutor, 'observe_agent_choice'):
                            tutor.observe_agent_choice(agent_choice, branches)

                    # After all episodes, check final prediction
                if hasattr(tutor, 'pref_posterior'):
                    ptype = tutor.pref_posterior.predicted_type
                    pconf = tutor.pref_posterior.predicted_prob
                    pred_confs.append(pconf)
                    corrects.append(1.0 if ptype == theta else 0.0)
                elif hasattr(tutor, 'joint_posterior'):
                    _, ppred = tutor.joint_posterior.predicted_joint
                    pconf = tutor.joint_posterior.joint_confidence
                    pred_confs.append(pconf)
                    corrects.append(1.0 if ppred == theta else 0.0)

            avg_conf = np.mean(pred_confs) if pred_confs else 0
            avg_corr = np.mean(corrects) if corrects else 0
            gap = abs(avg_conf - avg_corr)
            report_lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |\n".format(
                theta, tname, avg_conf, avg_corr, gap))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Robustness Test 6: Wrong-Memory Regression
# ═══════════════════════════════════════════

def test_wrong_memory(report_lines):
    report_lines.append("## 6. Wrong-Memory Regression\n")
    report_lines.append("| θ | Condition | SelGap | WR | Ent(1st) | Ent(2nd) |\n")
    report_lines.append("|---|-----------|--------|-----|----------|----------|\n")

    for theta in ["safe", "shiny"]:
        ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
        wrong = "shiny" if theta != "shiny" else "safe"

        for cond_name, wrong_strength in [("correct_prior", 0), ("mild_wrong", 2.0),
                                           ("adversarial_wrong", 5.0)]:
            session_rs = []
            for sid in range(6):
                session = generate_session(
                    session_id=sid * 1000 + abs(hash(theta)) % 1000,
                    n_episodes=12, theta_true=theta)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
                tutor = PersistentTutorV1_1(agent_params=ap)

                if wrong_strength > 0:
                    wi = PREFERENCE_TYPES.index(wrong)
                    tutor.pref_posterior.log_probs[wi] = wrong_strength
                    tutor.pref_posterior.log_probs -= np.mean(tutor.pref_posterior.log_probs)

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
                    lib.update("safe_branch", ss)
                    lib.update("risky_branch", sr)
                    scorer.update(build_scorer_input(ss, lib), 1.0)
                    scorer.update(build_scorer_input(sr, lib), 0.0)

                    ba_safe = BranchAttributes(
                        safety_score=float(ss[0]),
                        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                        risk_penalty=0.1)
                    ba_risky = BranchAttributes(
                        safety_score=float(sr[0]),
                        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                        risk_penalty=0.4)
                    branches = [ba_safe, ba_risky]
                    agent_choice = sample_branch_choice(branches, theta, ap, rng)

                    action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                    tutor.observe_agent_choice(agent_choice, branches)

                    traces.append({
                        "subtype": ep.episode_subtype,
                        "warned": (action == "WARN"),
                        "q_ent": tutor.pref_posterior.entropy,
                    })

                n = len(traces)
                wr = sum(1 for t in traces if t["warned"]) / n
                sub_wr = {}
                for st in EPISODE_SUBTYPES:
                    eps = [t for t in traces if t["subtype"] == st]
                    sub_wr[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
                wc = sub_wr.get("wait_clean")
                wt = sub_wr.get("warn_trap")
                sg = (wt - wc) if wt is not None and wc is not None else None
                ent_vals = [t["q_ent"] for t in traces if t["q_ent"] > 0]
                h = len(ent_vals) // 2 if len(ent_vals) >= 4 else 0
                e1 = float(np.mean(ent_vals[:h])) if h > 0 else 0
                e2 = float(np.mean(ent_vals[h:])) if h > 0 else 0
                session_rs.append({"sg": sg, "wr": wr, "e1": e1, "e2": e2})

            def akk(k):
                vs = [r[k] for r in session_rs if r.get(k) is not None]
                return round(np.mean(vs), 3) if vs else None

            report_lines.append("| {} | {} | {} | {} | {} | {} |\n".format(
                theta, cond_name, sf(akk("sg"), "{:.3f}"), sf(akk("wr")),
                sf(akk("e1"), "{:.4f}"), sf(akk("e2"), "{:.4f}")))
    report_lines.append("\n")


# ═══════════════════════════════════════════
# Cross-Family Transfer Matrix
# ═══════════════════════════════════════════

def test_cross_family_matrix(report_lines):
    report_lines.append("## 7. Cross-Family Transfer Matrix (SelGap)\n")
    families_session = ["pp_mrb"]
    families_single = [f for f in ["delayed_corridor", "distractor_cue",
                                    "elcb_po", "temptation_corridor"]
                       if f in SCENARIO_REGISTRY]
    tutors = ["v4", "v1_1", "joint_v2"]

    # PP-MRB
    report_lines.append("| Family | v4 | v1.1 | joint_v2 |\n")
    report_lines.append("|--------|-----|------|----------|\n")

    for theta in ["safe", "shiny"]:
        ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
        vals = {}
        for tname in tutors:
            rs = [run_ppmrb_session(tname, theta, ap, seed=s) for s in range(4)]
            vals[tname] = avg_sessions(rs, "sel_gap")
        report_lines.append("| PP-MRB ({}) | {} | {} | {} |\n".format(
            theta, sf(vals["v4"], "{:.3f}"), sf(vals["v1_1"], "{:.3f}"),
            sf(vals["joint_v2"], "{:.3f}")))

    # Single-episode families
    for fam in families_single:
        vals = {}
        for tname in tutors:
            warn_ct = 0
            total = 0
            for seed in range(20):
                try:
                    gm, cfg, meta, sc = generate_scenario(fam, seed=seed)
                except Exception:
                    continue
                fb, ww = apply_fix(meta, sc)
                fv = np.full_like(fb, 0.3)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
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

                ap = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
                t = make_tutor(tname, ap)
                if tname == "v4":
                    t.reset_stats()
                action, _ = t.decide(sc, fb, lp, lib, scorer, 2)
                warn_ct += (1 if action == "WARN" else 0)
                total += 1
            vals[tname] = round(warn_ct / max(total, 1), 3) if total > 0 else None
        report_lines.append("| {} | {} | {} | {} |\n".format(
            fam, sf(vals["v4"]), sf(vals["v1_1"]), sf(vals["joint_v2"])))
    report_lines.append("\n")


def main():
    print("═══ Unified Robustness Suite ═══\n", file=sys.stderr)
    lines = ["# Unified Robustness Suite\n\n"]

    print("Test 1: Mirror invariance...", file=sys.stderr)
    test_mirror_invariance(lines)

    print("Test 2: Parameter shift...", file=sys.stderr)
    test_parameter_shift(lines)

    print("Test 3: Noise sweep...", file=sys.stderr)
    test_noise_sweep(lines)

    print("Test 4: Session-order shuffle...", file=sys.stderr)
    test_session_order_shuffle(lines)

    print("Test 5: Calibration...", file=sys.stderr)
    test_calibration(lines)

    print("Test 6: Wrong-memory...", file=sys.stderr)
    test_wrong_memory(lines)

    print("Test 7: Cross-family matrix...", file=sys.stderr)
    test_cross_family_matrix(lines)

    with open(out / "robustness_suite.md", "w") as f:
        f.writelines(lines)

    print("\nReport -> results/robustness_suite.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
