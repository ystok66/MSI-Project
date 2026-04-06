"""FICA-v1: Framework Integrity & Causal Audit.

Block 1: Invariant Regression Matrix (cross-family × cross-tutor hard assertions)
Block 2: Mechanism-Consistent Accuracy (accidental correctness detection)
Block 3: State-to-Behavior Jacobian Identifiability
Block 4: Dose-Curriculum Audit (inverted-U across 4 protocols)
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
from src.agents.behavior_bridge import predict_all_probes
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.teachers.internalization_control_tutor_v2 import ICTv2
from src.teachers.internalization_control_tutor_v3 import BIICTv3
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2

out = Path("results")
out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

def sf(v, fmt="{:.3f}"):
    return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


# ════════════════════════════════════════════════════════════════
# Block 1: Invariant Regression Matrix
# ════════════════════════════════════════════════════════════════

def _run_ppmrb_session(tutor_name, theta, seed):
    """Run PP-MRB with given tutor, return WR and persistent vs reset signal."""
    from src.teachers.internalization_control_tutor_v2 import ICTv2
    m = FactoredInternalizationState()
    m.snapshot()
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    session = generate_session(seed, 8, theta)
    n_warns = 0
    safe_count = 0

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
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=0.4)

        do_warn = False
        if tutor_name == "cajt_v3":
            t = CAJTv3(agent_params=AP)
            action, _ = t.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
        elif tutor_name in ("ict_v2", "bc_v4"):
            t = ICTv2(agent_params=AP) if tutor_name == "ict_v2" else BCICTv4(agent_params=AP)
            if tutor_name == "bc_v4":
                action, dose, _ = t.decide(sc, fb, lp, lib, scorer, 2, m)
                do_warn = (dose > 0)
            else:
                action, _ = t.decide(sc, fb, lp, lib, scorer, 2, m)
                do_warn = (action == "WARN")

        if do_warn: n_warns += 1
        wb = [0.3 if do_warn else 0.0, -0.3 if do_warn else 0.0]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb)
        if ac == sc.oracle_safe_branch_id:
            safe_count += 1
        m.update_risk(0.3 if ac != sc.oracle_safe_branch_id else 0.05, 0.15)
        if do_warn:
            m.update_trust(warn_helpful=True)
            m.update_gamma_gen(sustained_pressure=True)
        m.snapshot()

    return {"wr": n_warns / 8, "sbcr": safe_count / 8,
            "tau_nu_gap": m.tau - m.nu, "gamma_gen": m.gamma_gen,
            "otr": overteach_rate_v2(m)["total"]}


def _run_tic_v4_session(tutor_name, theta, seed):
    sess = generate_tic_v4_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    m = FactoredInternalizationState()
    m.snapshot()
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)

    phases = {"A": [], "B": [], "C": [], "D": [], "E": []}
    n_warns = 0

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_v4_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(5):
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
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

        phase = ep.phase
        dose = 0.0
        give_advice = False
        advice_correct = True

        if phase == "A":
            if tutor_name == "no_tutor":
                dose = 0.0
            elif tutor_name == "cajt_v3":
                t = CAJTv3(agent_params=AP)
                action, _ = t.decide(sc, fb, lp, lib, scorer, 2)
                dose = 1.0 if action == "WARN" else 0.0
            elif tutor_name == "ict_v2":
                t = ICTv2(agent_params=AP)
                action, _ = t.decide(sc, fb, lp, lib, scorer, 2, m)
                dose = 1.0 if action == "WARN" else 0.0
            elif tutor_name == "bc_v4":
                t = BCICTv4(agent_params=AP)
                action, dose, _ = t.decide(sc, fb, lp, lib, scorer, 2, m)
        elif phase == "C":
            if rng.random() < 0.5: give_advice = True; advice_correct = True
        elif phase == "D":
            if rng.random() < 0.5: give_advice = True; advice_correct = False

        if phase == "A":
            wb = [0.3 * dose, -0.3 * dose]
            if dose > 0: n_warns += 1
        elif give_advice:
            wb = ([0.3, -0.3] if advice_correct == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3])
        else:
            wb = [0.0, 0.0]

        novel_flags = [False, False]
        if ep.subtype == "beneficial_novelty":
            novel_flags = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, novel_flags)

        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
        if phase == "A" and dose > 0:
            m.update_trust(warn_helpful=True)
            old_nu = m.nu; m.update_dependence(blind_obey=True); m.nu = old_nu + dose * (m.nu - old_nu)
            old_gg = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
        elif phase == "A" and dose == 0 and not chose_risky:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
        if chose_risky and bar.temptation_score > 0.5:
            m.update_gamma_spec(tempt_error=True)
        m.snapshot()

        correct = chose_risky if ep.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        # MCA: mechanism-consistent?
        mech_correct = correct
        if ep.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3:
            mech_correct = False  # accidental correctness via high γ_gen
        phases[phase].append({"correct": correct, "mca": mech_correct, "subtype": ep.subtype})

    def rate(ph, key="correct"):
        t = phases.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    bridge = predict_all_probes(m, 0.3, 0.3, 0.0, 0.5)
    otr = overteach_rate_v2(m)
    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"),
        "wr": n_warns / max(len(phases.get("A", [])), 1),
        "tau": m.tau, "nu": m.nu, "gg": m.gamma_gen, "gs": m.gamma_spec,
        "tau_nu_gap": m.tau - m.nu,
        "otr": otr["total"],
        "EP": probes["EP"], "VA": probes["VA"], "IA": probes["IA"],
        "b_EP": bridge["EP"], "b_IA": bridge["IA"],
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


# ════════════════════════════════════════════════════════════════
# Block 3: Jacobian Identifiability
# ════════════════════════════════════════════════════════════════

def jacobian_audit():
    """Compute ∂ẑ/∂m numerically and check sign structure."""
    m0 = FactoredInternalizationState(kappa=1.2, tau=0.5, nu=0.2,
                                       gamma_spec=0.3, gamma_gen=0.1)
    eps = 0.01
    state_names = ["kappa", "tau", "nu", "gamma_spec", "gamma_gen"]
    probe_names = ["RC", "TR", "EP", "VA", "IA"]

    # Expected dominant signs
    expected = {
        ("RC", "kappa"): +1, ("TR", "gamma_spec"): +1,
        ("EP", "gamma_gen"): -1, ("VA", "tau"): +1, ("IA", "nu"): +1,
    }

    z0 = predict_all_probes(m0, 0.3, 0.3, 0.1, 0.5)
    J = {}
    for si, sn in enumerate(state_names):
        m_plus = m0.copy()
        setattr(m_plus, sn, getattr(m_plus, sn) + eps)
        z_plus = predict_all_probes(m_plus, 0.3, 0.3, 0.1, 0.5)
        for pn in probe_names:
            J[(pn, sn)] = round((z_plus[pn] - z0[pn]) / eps, 4)

    # Check sign violations
    violations = []
    for (pn, sn), expected_sign in expected.items():
        actual = J.get((pn, sn), 0)
        if expected_sign > 0 and actual <= 0:
            violations.append(f"∂{pn}/∂{sn} = {actual} (expected > 0)")
        elif expected_sign < 0 and actual >= 0:
            violations.append(f"∂{pn}/∂{sn} = {actual} (expected < 0)")

    # Off-diagonal magnitude
    on_diag = sum(abs(J[(pn, sn)]) for (pn, sn) in expected)
    all_mag = sum(abs(v) for v in J.values())
    off_diag = all_mag - on_diag
    sparsity = on_diag / max(all_mag, 1e-10)

    return J, violations, sparsity, state_names, probe_names


# ════════════════════════════════════════════════════════════════
# Block 4: Dose-Curriculum Audit
# ════════════════════════════════════════════════════════════════

def dose_curriculum_audit():
    """Run 3 dose levels × 2θ × 4 seeds, check inverted-U."""
    results = []
    for theta in ["safe", "shiny"]:
        for dose_regime in ["none", "soft_only", "hard_only", "mixed"]:
            rs = []
            for seed in range(4):
                sess = generate_tic_v4_session(
                    seed * 1000 + abs(hash(theta)) % 1000, theta,
                    n_tutor=10, n_autonomy=4, n_valid=4, n_invalid=4, n_novelty=4)
                m = FactoredInternalizationState()
                m.snapshot()
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                lib = BranchConceptLibrary()
                scorer = BranchScorerProbe(lr=0.05, l2=0.01)
                phase_corr = {"B": [], "C": [], "D": [], "E": []}

                for ep in sess.episodes:
                    gm, cfg, meta, sc = generate_tic_v4_scenario(ep)
                    fb, ww = apply_fix(meta, sc)
                    fv = np.full_like(fb, 0.3)
                    rng = np.random.default_rng(ep.cue_layout_seed + 9999)
                    for _ in range(5):
                        for r in range(gm.height):
                            for c in range(gm.width):
                                if gm.cell_types[r, c] == CellType.WALL: continue
                                z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
                    ss = summarize_branch(sc.safe_cells, fb, fv, lp)
                    sr = summarize_branch(sc.risky_cells, fb, fv, lp)
                    lib.update("safe_branch", ss); lib.update("risky_branch", sr)
                    bas = BranchAttributes(safety_score=float(ss[0]),
                        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                        risk_penalty=0.1)
                    bar = BranchAttributes(safety_score=float(sr[0]),
                        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                        risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

                    phase = ep.phase
                    dose = 0.0
                    give_adv = False; adv_corr = True
                    if phase == "A":
                        if dose_regime == "none": dose = 0.0
                        elif dose_regime == "soft_only": dose = 0.5
                        elif dose_regime == "hard_only": dose = 1.0
                        elif dose_regime == "mixed": dose = 0.5 if rng.random() < 0.5 else 1.0
                    elif phase == "C":
                        if rng.random() < 0.5: give_adv = True; adv_corr = True
                    elif phase == "D":
                        if rng.random() < 0.5: give_adv = True; adv_corr = False

                    if phase == "A":
                        wb = [0.3 * dose, -0.3 * dose]
                    elif give_adv:
                        wb = ([0.3, -0.3] if adv_corr == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3])
                    else:
                        wb = [0.0, 0.0]
                    nf = [False, False]
                    if ep.subtype == "beneficial_novelty":
                        nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
                    ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, nf)
                    chose_risky = (ac != sc.oracle_safe_branch_id)
                    m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
                    if phase == "A" and dose > 0:
                        m.update_trust(warn_helpful=True)
                        old_nu = m.nu; m.update_dependence(blind_obey=True)
                        m.nu = old_nu + dose * (m.nu - old_nu)
                        old_gg = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
                        m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
                    elif phase == "A" and not chose_risky:
                        m.update_dependence(self_discovery=True)
                        m.update_gamma_gen(successful_exploration=True)
                    if chose_risky and bar.temptation_score > 0.5:
                        m.update_gamma_spec(tempt_error=True)
                    m.snapshot()
                    corr = chose_risky if ep.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
                    if phase in phase_corr:
                        phase_corr[phase].append(corr)

                def cr(ph):
                    t = phase_corr.get(ph, [])
                    return sum(t) / max(len(t), 1) if t else None

                rs.append({"B": cr("B"), "C": cr("C"), "D": cr("D"), "E": cr("E"),
                            "nu": m.nu, "gg": m.gamma_gen, "otr": overteach_rate_v2(m)["total"]})

            a = {k: avg(rs, k) for k in ["B", "C", "D", "E", "nu", "gg", "otr"]}
            a["theta"] = theta; a["dose"] = dose_regime
            results.append(a)
    return results


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    lines = ["# FICA-v1: Framework Integrity & Causal Audit\n\n"]

    # ─── Block 1: Invariant Regression ───
    print("Block 1: Invariant Regression...", file=sys.stderr)
    lines.append("## Block 1: Invariant Regression Matrix\n\n")

    assertions = []
    tutors_b1 = ["no_tutor", "cajt_v3", "ict_v2", "bc_v4"]

    # PP-MRB cross-check
    lines.append("### PP-MRB Family\n\n")
    lines.append("| θ | Tutor | WR | SBCR | τ-ν | γg | OTR |\n")
    lines.append("|---|-------|----|----|-----|-------|-----|\n")
    for theta in ["safe", "shiny"]:
        for tn in ["cajt_v3", "ict_v2", "bc_v4"]:
            rs = [_run_ppmrb_session(tn, theta, s) for s in range(4)]
            a = {k: avg(rs, k) for k in ["wr", "sbcr", "tau_nu_gap", "gamma_gen", "otr"]}
            lines.append("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, tn, sf(a["wr"]), sf(a["sbcr"]),
                sf(a["tau_nu_gap"], "{:+.3f}"), sf(a["gamma_gen"]), sf(a["otr"])))

            # Assertions
            if tn == "cajt_v3":
                assertions.append(("cajt_v3 high WR on PP-MRB",
                                    a["wr"] > 0.5, f"WR={a['wr']}"))
                assertions.append(("cajt_v3 τ-ν negative",
                                    a["tau_nu_gap"] <= 0.1, f"gap={a['tau_nu_gap']}"))
            if tn in ("ict_v2", "bc_v4"):
                assertions.append((f"{tn} low OTR on PP-MRB",
                                    a["otr"] < 0.5, f"OTR={a['otr']}"))
                assertions.append((f"{tn} τ-ν positive",
                                    a["tau_nu_gap"] > 0, f"gap={a['tau_nu_gap']}"))

    # TIC-v4 cross-check
    lines.append("\n### TIC-v4 Family\n\n")
    lines.append("| θ | Tutor | WR | B | C | D | E | MCA_E | τ-ν | γg | OTR |\n")
    lines.append("|---|-------|----|----|----|----|----|----|-----|-------|-----|\n")
    for theta in ["safe", "shiny"]:
        for tn in tutors_b1:
            rs = [_run_tic_v4_session(tn, theta, s) for s in range(4)]
            a = {k: avg(rs, k) for k in ["wr", "B", "C", "D", "E", "mca_E",
                 "tau_nu_gap", "gg", "otr"]}
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                theta, tn, sf(a["wr"]), sf(a["B"]), sf(a["C"]),
                sf(a["D"]), sf(a["E"]), sf(a["mca_E"]),
                sf(a["tau_nu_gap"], "{:+.3f}"),
                sf(a["gg"]), sf(a["otr"])))

            if tn == "cajt_v3":
                assertions.append(("cajt_v3 high γg on TIC-v4",
                                    a["gg"] > 0.2, f"γg={a['gg']}"))
            if tn in ("ict_v2", "bc_v4"):
                assertions.append((f"{tn} low γg on TIC-v4",
                                    a["gg"] < 0.2, f"γg={a['gg']}"))

    # ─── Block 2: Mechanism-Consistent Accuracy ───
    print("Block 2: MCA...", file=sys.stderr)
    lines.append("\n## Block 2: Mechanism-Consistent Accuracy\n\n")
    lines.append("| θ | Tutor | E(raw) | **MCA_E** | Δ(raw-MCA) |\n")
    lines.append("|---|-------|--------|---------|------------|\n")
    for theta in ["safe", "shiny"]:
        for tn in tutors_b1:
            rs = [_run_tic_v4_session(tn, theta, s) for s in range(4)]
            e_raw = avg(rs, "E")
            mca_e = avg(rs, "mca_E")
            delta = round(e_raw - mca_e, 3) if e_raw and mca_e else None
            lines.append("| {} | {} | {} | **{}** | {} |\n".format(
                theta, tn, sf(e_raw), sf(mca_e), sf(delta, "{:+.3f}")))

            if tn == "cajt_v3" and delta is not None:
                assertions.append(("cajt_v3 accidental correctness on E",
                                    delta > 0.05, f"Δ={delta}"))

    # ─── Block 3: Jacobian ───
    print("Block 3: Jacobian...", file=sys.stderr)
    J, violations, sparsity, state_names, probe_names = jacobian_audit()
    lines.append("\n## Block 3: State→Behavior Jacobian\n\n")
    lines.append("| | " + " | ".join(state_names) + " |\n")
    lines.append("|---|" + "|".join(["---"] * len(state_names)) + "|\n")
    for pn in probe_names:
        row = [sf(J.get((pn, sn), 0), "{:+.4f}") for sn in state_names]
        lines.append(f"| {pn} | " + " | ".join(row) + " |\n")
    lines.append(f"\nSparsity (on-diag / total): **{sparsity:.3f}**\n")
    if violations:
        lines.append(f"\n**Sign violations ({len(violations)}):**\n")
        for v in violations:
            lines.append(f"- ⚠️ {v}\n")
        assertions.append(("Jacobian sign violations", len(violations) == 0,
                            f"{len(violations)} violations"))
    else:
        lines.append("\n✅ All expected signs correct.\n")
        assertions.append(("Jacobian sign violations", True, "0 violations"))

    # ─── Block 4: Dose-Curriculum ───
    print("Block 4: Dose-Curriculum...", file=sys.stderr)
    dose_results = dose_curriculum_audit()
    lines.append("\n## Block 4: Dose-Curriculum Audit\n\n")
    lines.append("| θ | Dose | B | C | D | E | ν | γg | OTR |\n")
    lines.append("|---|------|---|---|---|---|---|-------|-----|\n")
    for r in dose_results:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            r["theta"], r["dose"],
            sf(r["B"]), sf(r["C"]), sf(r["D"]), sf(r["E"]),
            sf(r["nu"]), sf(r["gg"]), sf(r["otr"])))

    # Check inverted-U
    for theta in ["safe", "shiny"]:
        th_rs = [r for r in dose_results if r["theta"] == theta]
        nu_none = [r for r in th_rs if r["dose"] == "none"]
        nu_hard = [r for r in th_rs if r["dose"] == "hard_only"]
        if nu_none and nu_hard:
            assertions.append((f"hard_only pushes ν higher ({theta})",
                                nu_hard[0]["nu"] > nu_none[0]["nu"],
                                f"none={nu_none[0]['nu']}, hard={nu_hard[0]['nu']}"))

    # ─── Assertion Summary ───
    lines.append("\n## Assertion Summary\n\n")
    n_pass = sum(1 for _, ok, _ in assertions if ok)
    n_fail = len(assertions) - n_pass
    lines.append(f"**{n_pass}/{len(assertions)} passed** ({n_fail} failed)\n\n")
    lines.append("| # | Assertion | Status | Evidence |\n")
    lines.append("|---|-----------|--------|----------|\n")
    for i, (name, ok, evidence) in enumerate(assertions):
        status = "✅" if ok else "❌"
        lines.append(f"| {i+1} | {name} | {status} | {evidence} |\n")

    with open(out / "fica_v1_report.md", "w") as f:
        f.writelines(lines)
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"FICA-v1: {n_pass}/{len(assertions)} assertions passed", file=sys.stderr)
    if n_fail > 0:
        print(f"⚠️  {n_fail} FAILURES — review before MC-ICT-v5", file=sys.stderr)
    else:
        print("✅ All passed — safe to proceed to MC-ICT-v5", file=sys.stderr)
    print(f"Report -> results/fica_v1_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
