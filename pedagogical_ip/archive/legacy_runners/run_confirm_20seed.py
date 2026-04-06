"""Phase 6.9 — 20-Seed Confirmation + Pareto + STOP Audit.

Exp A: 20 seeds × 4 arms × 2θ × {natural, fixed-dose}
Exp B: Pareto S_λ = C - λ·OTR_teach for λ ∈ {0.25, 0.5, 1.0}
Exp E: STOP counterfactual regret

Arms: canonical, B1 (eig), B3 (zpd), all (eig+epi+zpd)
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
NS = 20

# Only 4 shortlisted arms
ARM_DEFS = {
    "canonical": {"eig": False, "epi": False, "zpd": False},
    "B1":        {"eig": True,  "epi": False, "zpd": False},
    "B3":        {"eig": False, "epi": False, "zpd": True},
    "all":       {"eig": True,  "epi": True,  "zpd": True},
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


def run_one(cct, th, seed, use_epi=False, mt=12, force_teach=0, ood="none"):
    rng = np.random.default_rng(seed * 1000 + abs(hash(th)) % 1000)
    mic = BCICTv4(agent_params=AP)
    m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); cct.reset_session(cct.cfg.total_budget)
    tr = {"A": [], "B": [], "C": [], "D": [], "E": []}
    nt = 0; ne = 0; idx = 0; fv = None
    fam_sel = {}; lessons_taught = []; stop_regrets = []

    for step in range(mt + 6):
        at, les, qv, inf = cct.select_action(m)
        # STOP audit: record regret
        if at == "STOP":
            if inf and isinstance(inf, dict) and "margin" in inf:
                stop_regrets.append(inf.get("margin", 0))
            if nt < force_teach:
                at = "TEACH"
                if les is None:
                    les = rng.choice(LESSON_CATALOG_V2)
            else:
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
        lessons_taught.append(les.name)
        ub = cct.mastery.mastery()
        nub = m.nu; ggb = m.gamma_gen; ob = overteach_rate_v2(m)["total"]
        et = generate_episode_from_lesson_v2(les, idx + seed * 100, th, ub, rng)
        ep = et[0]; cct.record_realization(ep)
        _, spec, gm, cfg_e, meta, sc = et
        fb, ww = apply_fix(meta, sc, rng)
        if ood == "sign_flip":
            fb = np.where(rng.random(fb.shape) < 0.3, -fb, fb)
        elif ood == "noise_heavy":
            fb = fb + rng.normal(0, 0.5, fb.shape)
        if fv is None:
            fv = np.full_like(fb, 0.3)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scr.update(build_scorer_input(ss, lib), 1.0); scr.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        bt.reset(ep); _, rd, _ = mic.decide(sc, fb, lp, lib, scr, 2, m)
        fe = bt.feasible_doses(); dose = rd if rd in fe else max(d for d in fe if d <= rd)
        bt.consume(dose); cct.consume_dose(dose)
        risk_unc_s = float(ss[5]) if len(ss) > 5 else 0.0
        risk_unc_r = float(sr[5]) if len(sr) > 5 else 0.0
        wb = [0.3 * dose, -0.3 * dose]; nf = [False, False]
        if ep.subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], th, m, AP, rng, wb, nf,
            risk_uncs=[risk_unc_s, risk_unc_r], use_epistemic_risk=use_epi)
        cr = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and cr else 0.05, 0.15)
        he = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not he: old = m.nu; m.update_dependence(blind_obey=True); m.nu = old + dose * (m.nu - old)
            old = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True); m.gamma_gen = old + dose * (m.gamma_gen - old)
        elif not cr:
            m.update_dependence(self_discovery=True); m.update_gamma_gen(successful_exploration=True)
        if cr and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
        if ep.subtype in ("false_suppression_cost", "beneficial_novelty") and not cr: m.update_gamma_spec(false_suppression=True)
        m.snapshot(); pr = all_probes(m, AP, th); cct.update_mastery(pr)
        cct.bridge.update(m, pr, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                          bar.temptation_score, ep.novelty, 0.7 if he else 0.3)
        oa = overteach_rate_v2(m)["total"]
        try: cct.update_response(les.name, dict(ub), cct.mastery.mastery(), nub, m.nu, ggb, m.gamma_gen, ob, oa)
        except TypeError: pass
        correct = cr if ep.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        tr["A"].append({"correct": correct}); idx += 1

    for phase, nep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(nep):
            epp, spec, gm, cfg_e, meta, sc = generate_transfer_episode(phase, idx + seed * 100, th, rng)
            fb, ww = apply_fix(meta, sc, rng)
            if ood == "sign_flip": fb = np.where(rng.random(fb.shape) < 0.3, -fb, fb)
            elif ood == "noise_heavy": fb = fb + rng.normal(0, 0.5, fb.shape)
            if fv is None: fv = np.full_like(fb, 0.3)
            re = np.random.default_rng(spec.cue_layout_seed + 9999)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL: continue
                        z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp); sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
            ga = False; aok = True
            if phase == "C" and re.random() < 0.5: ga = True
            elif phase == "D" and re.random() < 0.5: ga = True; aok = False
            wb = ([0.3, -0.3] if aok == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3]) if ga else [0.0, 0.0]
            nff = [False, False]
            if epp.subtype == "beneficial_novelty": nff = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
            ac = sample_factored_choice([bas, bar], th, m, AP, re, wb, nff)
            cr = (ac != sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and cr else 0.05, 0.15)
            if phase in ("C", "D") and ga:
                hs = (spec.d_commit > spec.d_reveal + 1)
                if phase == "C" and not cr: m.update_trust(warn_helpful=True)
                if phase == "D" and cr: m.update_dependence(blind_obey=True)
                elif phase == "D" and not cr and hs: m.update_dependence(self_discovery=True)
            if cr and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
            m.snapshot()
            correct = cr if epp.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
            tr[phase].append({"correct": correct}); idx += 1

    rate = lambda ph: sum(1 for x in tr.get(ph, []) if x["correct"]) / max(len(tr.get(ph, [])), 1) if tr.get(ph) else None
    otr = overteach_rate_v2(m)
    total_fam = sum(fam_sel.values()) if fam_sel else 1
    fam_probs = [c / total_fam for c in fam_sel.values()] if fam_sel else [1.0]
    fam_ent = -sum(p * np.log(p + 1e-10) for p in fam_probs)
    return {
        "C": rate("B"), "E": rate("E"), "n_teach": nt, "n_eval": ne,
        "otr": otr["total"], "fam_ent": round(fam_ent, 3),
        "lessons": lessons_taught,
        "stop_regret_mean": round(np.mean(stop_regrets), 4) if stop_regrets else None,
        "stop_regret_max": round(max(stop_regrets), 4) if stop_regrets else None,
    }


def compute_sdr(results_on, results_off):
    divergent = total = 0
    for r_on, r_off in zip(results_on, results_off):
        l_on, l_off = r_on.get("lessons", []), r_off.get("lessons", [])
        for t in range(min(len(l_on), len(l_off))):
            total += 1
            if l_on[t] != l_off[t]: divergent += 1
    return round(divergent / max(total, 1), 3)


def stats(rs, k):
    vals = [r[k] for r in rs if r.get(k) is not None]
    if not vals: return None, None, None
    m, s = np.mean(vals), np.std(vals)
    ci = 1.96 * s / max(np.sqrt(len(vals)), 1)
    return round(m, 3), round(s, 3), round(ci, 3)


def main():
    print("═══ Phase 6.9: 20-Seed Confirmation ═══\n", file=sys.stderr)
    L = ["# Phase 6.9 — 20-Seed Confirmation\n\n"]
    L.append(f"> Seeds: {NS} | Arms: canonical / B1 / B3 / all\n\n")

    # ─── Exp A: Natural STOP ─────────────────────────────
    L.append("## Exp A: Natural STOP Track\n\n")
    L.append("| θ | Arm | #T | C (mean±std) | C 95%CI | E | OTR | H_fam |\n")
    L.append("|---|-----|---|---|---|---|---|---|\n")
    print("Exp A: Natural STOP...", file=sys.stderr)

    all_results_nat = {}
    for th in ["safe", "shiny"]:
        for arm, flags in ARM_DEFS.items():
            rs = []
            for sid in range(NS):
                cct, use_epi = make_arm(th, **flags)
                rs.append(run_one(cct, th, sid, use_epi=use_epi))
            all_results_nat[(arm, th)] = rs
            c_m, c_s, c_ci = stats(rs, "C")
            e_m, _, _ = stats(rs, "E")
            otr_m, _, _ = stats(rs, "otr")
            nt_m, _, _ = stats(rs, "n_teach")
            fe_m, _, _ = stats(rs, "fam_ent")
            L.append(f"| {th} | {arm} | {nt_m} | {sf(c_m)}±{sf(c_s,'{:.3f}')} | ±{sf(c_ci,'{:.3f}')} | {sf(e_m)} | {sf(otr_m,'{:.3f}')} | {sf(fe_m,'{:.3f}')} |\n")
            print(f"  {th}×{arm}: C={sf(c_m)}±{sf(c_s,'{:.3f}')} #T={nt_m}", file=sys.stderr)

    # SDR natural
    L.append("\n### SDR (Natural)\n\n| θ | Arm | SDR |\n|---|---|:-:|\n")
    for th in ["safe", "shiny"]:
        base = all_results_nat[("canonical", th)]
        for arm in ["B1", "B3", "all"]:
            sdr = compute_sdr(all_results_nat[(arm, th)], base)
            L.append(f"| {th} | {arm} | {sdr} |\n")

    # ─── Exp A2: Fixed-Dose ──────────────────────────────
    L.append("\n## Exp A: Fixed-Dose Track (T_fix=6)\n\n")
    L.append("| θ | Arm | #T | C (mean±std) | C 95%CI | E | OTR | H_fam |\n")
    L.append("|---|-----|---|---|---|---|---|---|\n")
    print("\nExp A: Fixed-dose...", file=sys.stderr)

    all_results_fd = {}
    for th in ["safe", "shiny"]:
        for arm, flags in ARM_DEFS.items():
            rs = []
            for sid in range(NS):
                cct, use_epi = make_arm(th, **flags, budget=8.0)
                rs.append(run_one(cct, th, sid, use_epi=use_epi, force_teach=6))
            all_results_fd[(arm, th)] = rs
            c_m, c_s, c_ci = stats(rs, "C")
            e_m, _, _ = stats(rs, "E")
            otr_m, _, _ = stats(rs, "otr")
            nt_m, _, _ = stats(rs, "n_teach")
            fe_m, _, _ = stats(rs, "fam_ent")
            L.append(f"| {th} | {arm} | {nt_m} | {sf(c_m)}±{sf(c_s,'{:.3f}')} | ±{sf(c_ci,'{:.3f}')} | {sf(e_m)} | {sf(otr_m,'{:.3f}')} | {sf(fe_m,'{:.3f}')} |\n")
            print(f"  {th}×{arm} (fd): C={sf(c_m)}±{sf(c_s,'{:.3f}')} OTR={sf(otr_m,'{:.3f}')}", file=sys.stderr)

    # SDR fixed-dose
    L.append("\n### SDR (Fixed-Dose)\n\n| θ | Arm | SDR |\n|---|---|:-:|\n")
    for th in ["safe", "shiny"]:
        base = all_results_fd[("canonical", th)]
        for arm in ["B1", "B3", "all"]:
            sdr = compute_sdr(all_results_fd[(arm, th)], base)
            L.append(f"| {th} | {arm} | {sdr} |\n")

    # ─── Exp B: Pareto ───────────────────────────────────
    L.append("\n## Exp B: Pareto Analysis\n\n")
    L.append("| θ | Arm | C | OTR | ΔC | ΔOTR | ΔC/ΔOTR | S_0.25 | S_0.5 | S_1.0 |\n")
    L.append("|---|-----|---|-----|---|------|---------|--------|-------|-------|\n")
    for th in ["safe", "shiny"]:
        c_base, _, _ = stats(all_results_fd[("canonical", th)], "C")
        otr_base, _, _ = stats(all_results_fd[("canonical", th)], "otr")
        for arm in ARM_DEFS:
            c_m, _, _ = stats(all_results_fd[(arm, th)], "C")
            otr_m, _, _ = stats(all_results_fd[(arm, th)], "otr")
            if c_m is None or otr_m is None: continue
            dc = round(c_m - (c_base or 0), 3)
            dotr = round(otr_m - (otr_base or 0), 3)
            ratio = round(dc / (abs(dotr) + 0.001), 2)
            s25 = round(c_m - 0.25 * otr_m, 3)
            s50 = round(c_m - 0.5 * otr_m, 3)
            s10 = round(c_m - 1.0 * otr_m, 3)
            L.append(f"| {th} | {arm} | {sf(c_m)} | {sf(otr_m,'{:.3f}')} | {dc:+.3f} | {dotr:+.3f} | {ratio} | {s25:.3f} | {s50:.3f} | {s10:.3f} |\n")

    # ─── Exp C: OOD ──────────────────────────────────────
    L.append("\n## Exp C: OOD Robustness (Fixed-Dose)\n\n")
    L.append("| θ | Arm | OOD | C | E |\n|---|-----|-----|---|---|\n")
    print("\nExp C: OOD...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        for arm, flags in [("canonical", ARM_DEFS["canonical"]), ("all", ARM_DEFS["all"])]:
            for ood in ["none", "sign_flip", "noise_heavy"]:
                rs = []
                for sid in range(min(NS, 8)):  # 8 seeds for OOD
                    cct, use_epi = make_arm(th, **flags, budget=8.0)
                    rs.append(run_one(cct, th, sid, use_epi=use_epi, force_teach=6, ood=ood))
                c_m, _, _ = stats(rs, "C"); e_m, _, _ = stats(rs, "E")
                L.append(f"| {th} | {arm} | {ood} | {sf(c_m)} | {sf(e_m)} |\n")
                print(f"  {th}×{arm}×{ood}: C={sf(c_m)}", file=sys.stderr)

    # ─── Exp E: STOP Audit ───────────────────────────────
    L.append("\n## Exp E: STOP Counterfactual Audit\n\n")
    L.append("| θ | Arm | Mean Regret | Max Regret | #STOP events |\n")
    L.append("|---|-----|:-:|:-:|:-:|\n")
    for th in ["safe", "shiny"]:
        for arm in ["canonical", "all"]:
            rs = all_results_nat[(arm, th)]
            regrets = [r["stop_regret_mean"] for r in rs if r.get("stop_regret_mean") is not None]
            max_regs = [r["stop_regret_max"] for r in rs if r.get("stop_regret_max") is not None]
            mr = round(np.mean(regrets), 4) if regrets else "—"
            mx = round(np.mean(max_regs), 4) if max_regs else "—"
            n_ev = len(regrets)
            L.append(f"| {th} | {arm} | {mr} | {mx} | {n_ev} |\n")

    rpt = out / "enhancement_confirm_20seed.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
