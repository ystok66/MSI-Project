"""Curriculum-to-Transfer Matrix (CTM).

4 source curricula × 2 target families, all with v2 PIA + ICT-v1.
Sources: PP-MRB-reveal, TIC-rescue-heavy, mixed-balanced, CGC-v2
Targets: TIC-same, TIC-shift
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario,
)
from src.envs.teaching_internalization_corridor import (
    generate_tic_session, generate_tic_scenario,
)
from src.envs.compositional_goal_corridor_v2 import (
    generate_cgc2_session, generate_cgc2_scenario,
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
from src.teachers.internalization_control_tutor_v1 import ICTv1
from src.metrics.teaching_zone import zone_hit_rate
from src.metrics.overteaching import overteach_rate

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


def train_on_ppmrb(tutor, m, theta, seed, n_ep=8):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    session = generate_session(seed, n_ep, theta)
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_episode_scenario(ep, theta)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(3):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a, risk_penalty=0.4)
        branches = [bas, bar]
        if tutor is not None:
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            do_warn = (action == "WARN")
        else:
            do_warn = False
        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_pia_v2_choice(branches, theta, m, AP, rng, wb)
        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(0.35 if chose_risky else 0.05, 0.15)
        oracle_warn = (ep.d_commit <= ep.d_reveal)
        if do_warn and oracle_warn: m.update_trust(warn_helpful=True)
        elif do_warn and not oracle_warn: m.update_trust(warn_unnecessary=True)
        elif not do_warn and oracle_warn: m.update_trust(warn_missed=True)
        m.update_suppression(temptation_error=(chose_risky and bar.temptation_score > 0.5))
        m.snapshot()
    return lp, lib, scorer


def train_on_tic(tutor, m, theta, seed, n_ep=8, rescue_heavy=False):
    if rescue_heavy:
        sess = generate_tic_session(seed, theta, n_ep, 0, 0)
    else:
        sess = generate_tic_session(seed, theta, n_ep, 0, 0)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(3):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        branches = [bas, bar]
        if tutor is not None and ep.phase == "A":
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            do_warn = (action == "WARN")
        else:
            do_warn = False
        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_pia_v2_choice(branches, theta, m, AP, rng, wb)
        chose_risky = (ac != sc.oracle_safe_branch_id)
        rl = sc.risk_level if hasattr(sc, 'risk_level') else 0.3
        m.update_risk(rl if chose_risky else 0.05, 0.15)
        if do_warn: m.update_trust(warn_helpful=True)
        m.update_suppression(temptation_error=(chose_risky and bar.temptation_score > 0.5))
        m.snapshot()
    return lp, lib, scorer


def test_on_tic(m, theta, seed, phase="same"):
    n_same = 8 if phase == "same" else 0
    n_shift = 8 if phase == "shift" else 0
    sess = generate_tic_session(seed + 500, theta, 0, n_same, n_shift)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    safe_count = 0
    total = 0
    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(3):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        ac = sample_pia_v2_choice([bas, bar], theta, m, AP, rng)
        if ac == sc.oracle_safe_branch_id:
            safe_count += 1
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and ac != sc.oracle_safe_branch_id else 0.05, 0.15)
        m.update_suppression(temptation_error=(ac != sc.oracle_safe_branch_id and bar.temptation_score > 0.5))
        m.snapshot()
        total += 1
    return safe_count / max(total, 1)


def run_ctm(source, theta, seed):
    tutor = ICTv1(agent_params=AP)
    m = InternalizationStateV2()
    m.snapshot()

    sid = seed * 1000 + abs(hash(theta)) % 1000
    if source == "ppmrb":
        train_on_ppmrb(tutor, m, theta, sid, 8)
    elif source == "tic_rescue":
        train_on_tic(tutor, m, theta, sid, 8, rescue_heavy=True)
    elif source == "tic_mixed":
        train_on_tic(tutor, m, theta, sid, 8, rescue_heavy=False)
    elif source == "none":
        pass

    sbcr_same = test_on_tic(m, theta, sid, "same")
    m2 = m.copy()
    m2.kappa_history = list(m.kappa_history)
    m2.eta_history = list(m.eta_history)
    m2.gamma_history = list(m.gamma_history)
    sbcr_shift = test_on_tic(m2, theta, sid, "shift")

    return {
        "sbcr_same": round(sbcr_same, 3),
        "sbcr_shift": round(sbcr_shift, 3),
        "kappa_f": round(m.kappa, 3),
        "gamma_f": round(m.gamma, 3),
        "zhr": round(zone_hit_rate(m.kappa_history, m.eta_history, m.gamma_history, theta), 3),
        "otr": round(overteach_rate(m.kappa_history, m.eta_history, m.gamma_history), 3),
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ Curriculum-to-Transfer Matrix ═══\n", file=sys.stderr)
    sources = ["none", "ppmrb", "tic_rescue", "tic_mixed"]
    lines = ["# Curriculum-to-Transfer Matrix (CTM)\n\n"]
    lines.append("Train on source (8 ep, ICT-v1) → test on TIC (8 ep, no tutor)\n\n")
    lines.append("| θ | Source | SBCR(same) | SBCR(shift) | κ_f | γ_f | ZHR | OTR |\n")
    lines.append("|---|--------|-----------|------------|-----|-----|-----|-----|\n")

    for theta in ["safe", "shiny"]:
        for src in sources:
            rs = [run_ctm(src, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["sbcr_same", "sbcr_shift", "kappa_f", "gamma_f", "zhr", "otr"]}
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, src, sf(a["sbcr_same"]), sf(a["sbcr_shift"]),
                sf(a["kappa_f"], "{:.3f}"), sf(a["gamma_f"], "{:.3f}"),
                sf(a["zhr"], "{:.3f}"), sf(a["otr"], "{:.3f}")))
            print(f"  {theta} × {src}: same={sf(a['sbcr_same'])} shift={sf(a['sbcr_shift'])} "
                  f"κ={sf(a['kappa_f'], '{:.3f}')} γ={sf(a['gamma_f'], '{:.3f}')}",
                  file=sys.stderr)

    # Δ comparison
    lines.append("\n## Transfer Improvement vs No-Source Baseline\n\n")
    lines.append("| θ | Source | Δ(same) | Δ(shift) |\n")
    lines.append("|---|--------|---------|----------|\n")
    for theta in ["safe", "shiny"]:
        base_rs = [run_ctm("none", theta, sid) for sid in range(4)]
        base_same = avg(base_rs, "sbcr_same")
        base_shift = avg(base_rs, "sbcr_shift")
        for src in ["ppmrb", "tic_rescue", "tic_mixed"]:
            rs = [run_ctm(src, theta, sid) for sid in range(4)]
            a_same = avg(rs, "sbcr_same")
            a_shift = avg(rs, "sbcr_shift")
            ds = round(a_same - base_same, 3) if a_same and base_same else None
            dd = round(a_shift - base_shift, 3) if a_shift and base_shift else None
            lines.append("| {} | {} | {} | {} |\n".format(
                theta, src, sf(ds, "{:+.3f}"), sf(dd, "{:+.3f}")))

    with open(out / "curriculum_matrix_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/curriculum_matrix_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
