"""Ablation Enhancement Diagnostic Experiment.

Phase A: B2 integration verification (Usage_B2, ΔU)
Phase B: Fixed-dose diagnostic track (T_fix=6, no early STOP)
         + SDR_macro / SDR_micro divergence metrics
         + score decomposition per lesson per step
Phase C: Targeted subtype tests

Arms:
  canonical     (all flags OFF)
  eig           (B1 only)
  epi           (B2 only)
  zpd           (B3 only)
  eig+epi       (B1 + B2)
  all           (B1 + B2 + B3)
"""
import sys, os
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice, compute_factored_utility,
)
from src.agents.behavior_probes import all_probes
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13, ControllerV13Config
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.family_prior import FamilyPrior
from src.curriculum.dose_budget import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.teaching_zone_v2 import overteach_rate_v2

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
sf = lambda v, fmt="{:.0%}": "—" if v is None else fmt.format(v)
NS = 8

ARM_DEFS = {
    "canonical":     {"eig": False, "epi": False, "zpd": False},
    "eig":           {"eig": True,  "epi": False, "zpd": False},
    "epi":           {"eig": False, "epi": True,  "zpd": False},
    "zpd":           {"eig": False, "epi": False, "zpd": True},
    "eig+epi":       {"eig": True,  "epi": True,  "zpd": False},
    "all":           {"eig": True,  "epi": True,  "zpd": True},
}


def make_arm(theta, eig=False, epi=False, zpd=False, budget=4.0):
    cn = [l.name for l in LESSON_CATALOG_V2]
    cfg = ControllerV13Config(total_budget=budget, risk_budget_mode="theta")
    fp = FamilyPrior(enabled=True, use_saturation=True, use_rep_penalty=False)
    c = CurriculumControllerV13(
        cfg=cfg, theta=theta, family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn, theta=theta))
    c.use_eig_uncertainty = eig
    c.use_zpd_feature = zpd
    return c, epi


def apply_fix(meta, sc, rng=None):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_one(cct, th, seed, use_epi=False, mt=12, force_teach=0):
    """Run one session.

    Args:
        force_teach: If > 0, force this many TEACH actions before allowing STOP.
                     Used for fixed-dose diagnostic track.
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(th)) % 1000)
    mic = BCICTv4(agent_params=AP)
    m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); cct.reset_session(cct.cfg.total_budget)
    tr = {"A": [], "B": [], "C": [], "D": [], "E": []}
    nt = 0; ne = 0; idx = 0; fv = None
    fam_sel = {}
    score_log = []   # score decomposition per step
    b2_usage = []    # B2 utilization tracking

    for step in range(mt + 6):
        at, les, qv, inf = cct.select_action(m)
        # Fixed-dose track: override STOP if nt < force_teach
        if at == "STOP" and nt < force_teach:
            at = "TEACH"
            # Re-select lesson from the ranked list
            if inf and inf.get("ranked"):
                les = inf["ranked"][0] if inf["ranked"] else les
            elif les is None:
                # Fallback: pick a random lesson
                les = rng.choice(LESSON_CATALOG_V2)

        if at == "STOP":
            break
        if at == "EVAL":
            ne += 1; pr = all_probes(m, AP, th); cct.update_mastery(pr)
            continue
        if les is None:
            continue
        nt += 1
        if nt > mt:
            break

        fam_sel[les.family] = fam_sel.get(les.family, 0) + 1
        ub = cct.mastery.mastery()
        nub = m.nu; ggb = m.gamma_gen; ob = overteach_rate_v2(m)["total"]
        et = generate_episode_from_lesson_v2(les, idx + seed * 100, th, ub, rng)
        ep = et[0]
        cct.record_realization(ep)
        _, spec, gm, cfg_e, meta, sc = et
        fb, ww = apply_fix(meta, sc, rng)
        if fv is None:
            fv = np.full_like(fb, 0.3)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib = BranchConceptLibrary()
        scr = BranchScorerProbe(lr=0.05, l2=0.01)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)

        bas = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

        bt.reset(ep)
        _, rd, _ = mic.decide(sc, fb, lp, lib, scr, 2, m)
        fe = bt.feasible_doses()
        dose = rd if rd in fe else max(d for d in fe if d <= rd)
        bt.consume(dose); cct.consume_dose(dose)

        # ─── B2: Extract risk_unc from branch summaries ───
        # Branch summary is 8D: [safety, tempt, cost_mean, cost_unc, risk_mean,
        #                         risk_unc, texture_mean, novel]
        risk_unc_safe = float(ss[5]) if len(ss) > 5 else 0.0
        risk_unc_risky = float(sr[5]) if len(sr) > 5 else 0.0

        # B2 utilization tracking
        if use_epi:
            # Compute utility with and without B2
            u_base_safe = compute_factored_utility(bas, th, m, AP, 0.3 * dose, False)
            u_base_risky = compute_factored_utility(bar, th, m, AP, -0.3 * dose, False)
            u_b2_safe = compute_factored_utility(
                bas, th, m, AP, 0.3 * dose, False,
                risk_unc=risk_unc_safe, use_epistemic_risk=True)
            u_b2_risky = compute_factored_utility(
                bar, th, m, AP, -0.3 * dose, False,
                risk_unc=risk_unc_risky, use_epistemic_risk=True)
            b2_usage.append({
                "step": step, "lesson": les.name,
                "risk_unc_safe": round(risk_unc_safe, 4),
                "risk_unc_risky": round(risk_unc_risky, 4),
                "u_base_safe": round(u_base_safe, 4),
                "u_base_risky": round(u_base_risky, 4),
                "u_b2_safe": round(u_b2_safe, 4),
                "u_b2_risky": round(u_b2_risky, 4),
                "delta_safe": round(u_b2_safe - u_base_safe, 6),
                "delta_risky": round(u_b2_risky - u_base_risky, 6),
            })

        wb = [0.3 * dose, -0.3 * dose]
        nf = [False, False]
        if ep.subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]

        # Agent choice: pass B2 flags through
        ac = sample_factored_choice(
            [bas, bar], th, m, AP, rng, wb, nf,
            risk_uncs=[risk_unc_safe, risk_unc_risky],
            use_epistemic_risk=use_epi,
            use_epistemic_bonus=False,  # Phase-1: risk gate only
        )

        cr = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and cr else 0.05, 0.15)
        he = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not he:
                old = m.nu; m.update_dependence(blind_obey=True)
                m.nu = old + dose * (m.nu - old)
            old = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old + dose * (m.gamma_gen - old)
        elif not cr:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
        if cr and bar.temptation_score > 0.5:
            m.update_gamma_spec(tempt_error=True)
        if ep.subtype in ("false_suppression_cost", "beneficial_novelty") and not cr:
            m.update_gamma_spec(false_suppression=True)
        m.snapshot()
        pr = all_probes(m, AP, th); cct.update_mastery(pr)
        cct.bridge.update(m, pr, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                          bar.temptation_score, ep.novelty, 0.7 if he else 0.3)
        oa = overteach_rate_v2(m)["total"]
        try:
            cct.update_response(les.name, dict(ub), cct.mastery.mastery(),
                                nub, m.nu, ggb, m.gamma_gen, ob, oa)
        except TypeError:
            pass

        # Score decomposition logging
        if inf and isinstance(inf, dict):
            score_log.append({
                "step": step, "lesson": les.name,
                **{k: v for k, v in inf.items() if isinstance(v, (int, float, str, bool))}
            })

        correct = cr if ep.subtype in ("false_suppression_cost", "beneficial_novelty") \
            else (ac == sc.oracle_safe_branch_id)
        tr["A"].append({"correct": correct, "lesson": les.name}); idx += 1

    # Transfer phases
    for phase, nep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(nep):
            epp, spec, gm, cfg_e, meta, sc = generate_transfer_episode(
                phase, idx + seed * 100, th, rng)
            fb, ww = apply_fix(meta, sc, rng)
            if fv is None:
                fv = np.full_like(fb, 0.3)
            re = np.random.default_rng(spec.cue_layout_seed + 9999)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = fb[r, c]
                        lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(
                safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                risk_penalty=0.1)
            bar = BranchAttributes(
                safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
            ga = False; aok = True
            if phase == "C" and re.random() < 0.5:
                ga = True
            elif phase == "D" and re.random() < 0.5:
                ga = True; aok = False
            wb = ([0.3, -0.3] if aok == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3]) if ga else [0.0, 0.0]
            nff = [False, False]
            if epp.subtype == "beneficial_novelty":
                nff = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
            # Transfer: no B2 in transfer (no tutor)
            ac = sample_factored_choice([bas, bar], th, m, AP, re, wb, nff)
            cr = (ac != sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and cr else 0.05, 0.15)
            if phase in ("C", "D") and ga:
                hs = (spec.d_commit > spec.d_reveal + 1)
                if phase == "C" and not cr:
                    m.update_trust(warn_helpful=True)
                if phase == "D" and cr:
                    m.update_dependence(blind_obey=True)
                elif phase == "D" and not cr and hs:
                    m.update_dependence(self_discovery=True)
            if cr and bar.temptation_score > 0.5:
                m.update_gamma_spec(tempt_error=True)
            m.snapshot()
            correct = cr if epp.subtype in ("false_suppression_cost", "beneficial_novelty") \
                else (ac == sc.oracle_safe_branch_id)
            tr[phase].append({"correct": correct}); idx += 1

    rate = lambda ph: sum(1 for x in tr.get(ph, []) if x["correct"]) / max(len(tr.get(ph, [])), 1) \
        if tr.get(ph) else None
    otr = overteach_rate_v2(m)
    total_fam = sum(fam_sel.values()) if fam_sel else 1
    fam_probs = [c / total_fam for c in fam_sel.values()] if fam_sel else [1.0]
    fam_ent = -sum(p * np.log(p + 1e-10) for p in fam_probs)

    # B2 utilization stats
    b2_active = sum(1 for b in b2_usage if b["delta_risky"] != 0 or b["delta_safe"] != 0)
    b2_total = len(b2_usage) if b2_usage else 1

    return {
        "C": rate("B"), "E": rate("E"), "n_teach": nt, "n_eval": ne,
        "otr": otr["total"], "fam_ent": round(fam_ent, 3),
        "fam_usage": fam_sel,
        "lessons_taught": [x.get("lesson", "?") for x in tr["A"]],
        "b2_usage_rate": round(b2_active / b2_total, 3) if b2_usage else 0.0,
        "b2_log": b2_usage[:3],  # first 3 for inspection
        "score_log": score_log[:3],
    }


avg = lambda rs, k: round(np.mean([r[k] for r in rs if r.get(k) is not None]), 3) \
    if any(r.get(k) is not None for r in rs) else None
avg_int = lambda rs, k: round(np.mean([r[k] for r in rs if r.get(k) is not None]), 1) \
    if any(r.get(k) is not None for r in rs) else None


def compute_sdr(results_on, results_off):
    """Compute Selection Divergence Rate between two arms."""
    divergent = 0
    total = 0
    for r_on, r_off in zip(results_on, results_off):
        l_on = r_on.get("lessons_taught", [])
        l_off = r_off.get("lessons_taught", [])
        T = min(len(l_on), len(l_off))
        for t in range(T):
            total += 1
            if l_on[t] != l_off[t]:
                divergent += 1
    return round(divergent / max(total, 1), 3)


def main():
    print("═══ Ablation Enhancement Diagnostic Experiment ═══\n", file=sys.stderr)
    L = ["# Ablation Enhancement Diagnostic Experiment\n\n"]
    L.append(f"> Seeds: {NS} | Fixed-dose track: T_fix=6\n\n")

    # ─── Section 1: Phase A — B2 Integration Check ─────────────────
    L.append("## Phase A: B2 Integration Verification\n\n")
    print("Phase A: B2 integration check...", file=sys.stderr)

    cct_epi, _ = make_arm("safe", epi=True)
    r_epi = run_one(cct_epi, "safe", 42, use_epi=True)
    b2_active = r_epi["b2_usage_rate"]
    L.append(f"- **B2 Usage Rate**: {b2_active:.1%}\n")
    L.append(f"- **B2 Active Episodes**: {sum(1 for b in r_epi.get('b2_log',[]) if b.get('delta_risky',0) != 0)}/{len(r_epi.get('b2_log',[]))}\n")
    if r_epi.get("b2_log"):
        L.append("\nB2 sample trace:\n\n")
        L.append("| Step | Lesson | risk_unc_risky | ΔU_risky | ΔU_safe |\n")
        L.append("|------|--------|:-:|:-:|:-:|\n")
        for b in r_epi["b2_log"]:
            L.append("| {} | {} | {:.4f} | {:.6f} | {:.6f} |\n".format(
                b["step"], b["lesson"], b["risk_unc_risky"],
                b["delta_risky"], b["delta_safe"]))
    print(f"  B2 usage rate: {b2_active:.1%}", file=sys.stderr)

    # ─── Section 2: Natural STOP track ─────────────────────────────
    L.append("\n## Natural STOP Track (canonical regime)\n\n")
    L.append("| θ | Arm | #T | **C** | **E** | OTR | H_fam |\n")
    L.append("|---|-----|---|---|---|---|---|\n")
    print("\nNatural STOP track...", file=sys.stderr)

    all_results = {}  # (arm, theta) -> [results]
    for th in ["safe", "shiny"]:
        for arm_name, flags in ARM_DEFS.items():
            all_rs = []
            for sid in range(NS):
                cct, use_epi = make_arm(th, **flags)
                r = run_one(cct, th, sid, use_epi=use_epi)
                all_rs.append(r)
            all_results[(arm_name, th)] = all_rs
            a = {k: avg(all_rs, k) for k in ["C", "E", "otr", "fam_ent"]}
            a["nt"] = avg_int(all_rs, "n_teach")
            L.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th, arm_name, sf(a["nt"], "{:.0f}"),
                sf(a["C"]), sf(a["E"]),
                sf(a["otr"], "{:.3f}"), sf(a["fam_ent"], "{:.3f}")))
            print(f"  {th}×{arm_name}: C={sf(a['C'])} #T={sf(a['nt'],'{:.0f}')}",
                  file=sys.stderr)

    # ─── Section 3: SDR metrics ────────────────────────────────────
    L.append("\n## Selection Divergence Rate (SDR)\n\n")
    L.append("| θ | Arm vs Canonical | SDR_macro |\n|---|---|:-:|\n")
    for th in ["safe", "shiny"]:
        base_rs = all_results[("canonical", th)]
        for arm_name in ["eig", "epi", "zpd", "eig+epi", "all"]:
            arm_rs = all_results[(arm_name, th)]
            sdr = compute_sdr(arm_rs, base_rs)
            L.append(f"| {th} | {arm_name} | {sdr:.3f} |\n")
            print(f"  SDR {th}×{arm_name}: {sdr:.3f}", file=sys.stderr)

    # ─── Section 4: Fixed-dose diagnostic track ───────────────────
    L.append("\n## Fixed-Dose Diagnostic Track (T_fix=6)\n\n")
    L.append("| θ | Arm | #T | **C** | **E** | OTR | H_fam |\n")
    L.append("|---|-----|---|---|---|---|---|\n")
    print("\nFixed-dose track (T_fix=6)...", file=sys.stderr)

    fd_results = {}
    for th in ["safe", "shiny"]:
        for arm_name, flags in ARM_DEFS.items():
            all_rs = []
            for sid in range(NS):
                cct, use_epi = make_arm(th, **flags, budget=8.0)
                r = run_one(cct, th, sid, use_epi=use_epi, force_teach=6)
                all_rs.append(r)
            fd_results[(arm_name, th)] = all_rs
            a = {k: avg(all_rs, k) for k in ["C", "E", "otr", "fam_ent"]}
            a["nt"] = avg_int(all_rs, "n_teach")
            L.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th, arm_name, sf(a["nt"], "{:.0f}"),
                sf(a["C"]), sf(a["E"]),
                sf(a["otr"], "{:.3f}"), sf(a["fam_ent"], "{:.3f}")))
            print(f"  {th}×{arm_name} (fd): C={sf(a['C'])} #T={sf(a['nt'],'{:.0f}')}",
                  file=sys.stderr)

    # SDR for fixed-dose
    L.append("\n### Fixed-Dose SDR\n\n")
    L.append("| θ | Arm vs Canonical | SDR_macro |\n|---|---|:-:|\n")
    for th in ["safe", "shiny"]:
        base_rs = fd_results[("canonical", th)]
        for arm_name in ["eig", "epi", "zpd", "eig+epi", "all"]:
            arm_rs = fd_results[(arm_name, th)]
            sdr = compute_sdr(arm_rs, base_rs)
            L.append(f"| {th} | {arm_name} | {sdr:.3f} |\n")

    rpt = out / "ablation_diagnostic_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
