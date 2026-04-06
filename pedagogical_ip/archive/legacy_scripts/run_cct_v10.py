"""CCT-v10: Hybrid Dueling Constrained — Full Experiment.

Exp A: v10 vs v8 vs v9 vs fixed (4 strats × 2θ × 8 seeds, 4 sessions)
Exp B: Actionability audit 2.0 (sub-component AM/PCR)
Exp C: Hybrid ablation (full vs no_duel vs no_res vs no_constraint)
Exp D: Budget sweep (2,4,8)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import FactoredInternalizationState, sample_factored_choice
from src.agents.behavior_probes import all_probes
from src.agents.trainable_bridge import TrainableBridge
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, LESSON_V2_BY_NAME
from src.curriculum.curriculum_controller_v8 import CurriculumControllerV8
from src.curriculum.curriculum_controller_v9 import CurriculumControllerV9
from src.curriculum.curriculum_controller_v10 import CurriculumControllerV10
from src.curriculum.lesson_response_model_v3 import LessonResponseModelV3
from src.curriculum.curriculum_controller_v2 import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.curriculum_metrics import stop_efficiency

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
def sf(v, fmt="{:.0%}"): return "—" if v is None else fmt.format(v)
def apply_fix(meta, sc):
    rng = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    return neutralize_identity_features(meta.cell_features, allb, 0.5), ww
def fixed_lesson(strategy, rng):
    ls = {"ppmrb_only": ["ppmrb_standard","ppmrb_self_discovery"],
          "tic_heavy": ["tic_rescue_heavy","tic_temptation"]}
    name = rng.choice(ls.get(strategy, [l.name for l in LESSON_CATALOG_V2]))
    return LESSON_V2_BY_NAME.get(name, LESSON_CATALOG_V2[0])

def run_one_session(cct, strategy, theta, seed, max_teach=12, budget=4.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    micro = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker()
    if cct and hasattr(cct, 'reset_session'): cct.reset_session(budget)
    bridge = cct.bridge if cct else TrainableBridge()
    traces = {"A":[],"B":[],"C":[],"D":[],"E":[]}; lf = []; n_teach = 0; n_eval = 0
    stopped_at = None; idx = 0; fv = None
    for step in range(max_teach + 4):
        if cct: at, les, qv, inf = cct.select_action(m)
        else: at, les, qv = "TEACH", fixed_lesson(strategy, rng), 0
        if at == "STOP": stopped_at = step; break
        if at == "EVAL": n_eval += 1; probes = all_probes(m, AP, theta); cct.update_mastery(probes); continue
        n_teach += 1
        if n_teach > max_teach: break
        ub = cct.mastery.mastery() if cct else {p: 0.5 for p in ["RC","TR","EP","VA","IA"]}
        nub = m.nu; ggb = m.gamma_gen; ob = overteach_rate_v2(m)["total"]
        et = generate_episode_from_lesson_v2(les, idx + seed * 100, theta, ub, rng)
        ep = et[0]
        if cct and hasattr(cct, 'record_realization'): cct.record_realization(ep)
        lf.append(ep.fidelity_to(type('L',(),{'subtype':les.subtype,'severity':les.severity,'dose_profile':les.dose_profile,'family':les.family})()))
        _, spec, gm, cfg, meta, sc = et
        fb, ww = apply_fix(meta, sc)
        if fv is None: fv = np.full_like(fb, 0.3)
        re = np.random.default_rng(spec.cue_layout_seed + 9999)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r,c] == CellType.WALL: continue
                    z = fb[r,c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp); sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scr.update(build_scorer_input(ss, lib), 1.0); scr.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]), temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]), temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a, risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
        bt.reset(ep)
        _, rd, _ = micro.decide(sc, fb, lp, lib, scr, 2, m)
        fe = bt.feasible_doses(); dose = rd if rd in fe else max(d for d in fe if d <= rd); bt.consume(dose)
        if cct and hasattr(cct, 'consume_dose'): cct.consume_dose(dose)
        wb = [0.3*dose, -0.3*dose]; nf = [False, False]
        if ep.subtype == "beneficial_novelty": nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, re, wb, nf)
        cr = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and cr else 0.05, 0.15)
        he = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not he: old = m.nu; m.update_dependence(blind_obey=True); m.nu = old + dose*(m.nu-old)
            old = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True); m.gamma_gen = old + dose*(m.gamma_gen-old)
        elif not cr: m.update_dependence(self_discovery=True); m.update_gamma_gen(successful_exploration=True)
        if cr and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
        if ep.subtype in ("false_suppression_cost","beneficial_novelty") and not cr: m.update_gamma_spec(false_suppression=True)
        m.snapshot()
        probes = all_probes(m, AP, theta)
        bridge.update(m, probes, sc.risk_level if hasattr(sc, 'risk_level') else 0.3, bar.temptation_score, ep.novelty, 0.7 if he else 0.3)
        if cct and hasattr(cct, 'update_response'):
            oa = overteach_rate_v2(m)["total"]
            try: cct.update_response(les.name, dict(ub), cct.mastery.mastery(), nub, m.nu, ggb, m.gamma_gen, ob, oa)
            except TypeError: cct.update_response(les.name, dict(ub), cct.mastery.mastery())
        correct = cr if ep.subtype in ("false_suppression_cost","beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        mca = correct
        if ep.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
        traces["A"].append({"correct":correct,"mca":mca}); idx += 1
    for phase, ne in [("B",4),("C",4),("D",4),("E",4)]:
        for _ in range(ne):
            epp, spec, gm, cfg, meta, sc = generate_transfer_episode(phase, idx+seed*100, theta, rng)
            fb, ww = apply_fix(meta, sc)
            if fv is None: fv = np.full_like(fb, 0.3)
            re = np.random.default_rng(spec.cue_layout_seed+9999)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r,c] == CellType.WALL: continue
                        z = fb[r,c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp); sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(safety_score=float(ss[0]), temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b, risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]), temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a, risk_penalty=sc.risk_level if hasattr(sc,'risk_level') else 0.4)
            ga = False; aok = True
            if phase == "C" and re.random() < 0.5: ga = True
            elif phase == "D" and re.random() < 0.5: ga = True; aok = False
            wb = ([0.3,-0.3] if aok==(sc.oracle_safe_branch_id==0) else [-0.3,0.3]) if ga else [0.0,0.0]
            nff = [False,False]
            if epp.subtype == "beneficial_novelty": nff = [False,True] if sc.oracle_safe_branch_id==1 else [True,False]
            ac = sample_factored_choice([bas, bar], theta, m, AP, re, wb, nff)
            cr = (ac != sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc,'risk_level') and cr else 0.05, 0.15)
            if phase in ("C","D") and ga:
                hs = (spec.d_commit > spec.d_reveal + 1)
                if phase == "C" and not cr: m.update_trust(warn_helpful=True)
                if phase == "D" and cr: m.update_dependence(blind_obey=True)
                elif phase == "D" and not cr and hs: m.update_dependence(self_discovery=True)
            if cr and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
            m.snapshot()
            correct = cr if epp.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
            mca = correct
            if epp.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
            traces[phase].append({"correct":correct,"mca":mca}); idx += 1
    def rate(ph, key="correct"):
        t = traces.get(ph, []); return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None
    otr = overteach_rate_v2(m)
    se = stop_efficiency(sum(rate(p) or 0 for p in ["B","C","D","E"])/4, n_teach) if n_teach > 0 else 0.0
    pu = cct.posterior_stats().get("n_updated",0) if cct and hasattr(cct,'posterior_stats') else 0
    audit = cct.actionability_audit() if cct and hasattr(cct,'actionability_audit') else {}
    return {"B":rate("B"),"C":rate("C"),"D":rate("D"),"E":rate("E"),
            "mca_E":rate("E","mca"),"n_teach":n_teach,"n_eval":n_eval,
            "stopped":stopped_at is not None,
            "n_blocked":cct.budget_blocked_count if cct and hasattr(cct,'budget_blocked_count') else 0,
            "tau":round(m.tau,3),"nu":round(m.nu,3),"gg":round(m.gamma_gen,3),
            "otr":otr["total"],"se":round(se,4),"post_up":pu,"audit":audit}

def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs), 3) if vs else None
def avg_int(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs), 1) if vs else None

def make_cct(strategy, theta, budget, ablation=None):
    if strategy == "cct_v10" or (ablation and ablation not in ("v8","v9")):
        cct = CurriculumControllerV10(theta=theta, total_budget=budget)
        if ablation == "no_duel": cct.use_dueling = False; cct.response.duel_gain._n = -1  # disable
        if ablation == "no_res": cct.use_residual = False
        if ablation == "no_constraint": cct.use_constraint = False
        if ablation == "no_unc": cct.use_uncertainty = False
        if ablation == "no_stop": cct.use_stop = False
        return cct
    elif strategy == "cct_v9":
        from src.curriculum.curriculum_controller_v9 import CurriculumControllerV9
        return CurriculumControllerV9(theta=theta, total_budget=budget)
    elif strategy == "cct_v8":
        return CurriculumControllerV8(theta=theta, total_budget=budget, response=LessonResponseModelV3())
    return None

def run_multi(strategy, theta, seed, n_sessions=4, budget=4.0, ablation=None):
    cct = make_cct(strategy, theta, budget, ablation)
    results = []
    for sess in range(n_sessions):
        r = run_one_session(cct, strategy, theta, seed*10+sess, budget=budget); results.append(r)
    last = results[-1]; first = results[0]
    last["dC"] = round((last["C"] or 0) - (first["C"] or 0), 3)
    last["dE"] = round((last["E"] or 0) - (first["E"] or 0), 3)
    return last

def main():
    print("═══ CCT-v10: Hybrid Dueling Constrained ═══\n", file=sys.stderr)
    lines = ["# CCT-v10: Hybrid Dueling Constrained Bayesian Planner\n\n"]
    # ─── Exp A ───
    strats = ["ppmrb_only", "cct_v8", "cct_v9", "cct_v10"]
    lines.append("## Exp A: CCT-v10 vs v9 vs v8 vs Fixed (4 sessions)\n\n")
    lines.append("| θ | Strat | #T | Stop | **C** | **E** | SE | OTR | ν | PostUp |\n")
    lines.append("|---|------|---|---|---|---|---|---|---|---|\n")
    for theta in ["safe", "shiny"]:
        for s in strats:
            rs = [run_multi(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","se","otr","nu","gg"]}
            a["n_teach"] = avg_int(rs, "n_teach"); a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"])/len(rs), 2)
            a["post_up"] = avg_int(rs, "post_up")
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["n_teach"],"{:.0f}"), sf(a["stopped_frac"]),
                sf(a["C"]), sf(a["E"]), sf(a["se"],"{:.4f}"), sf(a["otr"],"{:.3f}"),
                sf(a["nu"],"{:.3f}"), sf(a["post_up"],"{:.0f}")))
            print(f"  {theta}×{s}: C={sf(a['C'])} E={sf(a['E'])} SE={sf(a['se'],'{:.3f}')} OTR={sf(a['otr'],'{:.2f}')} ν={sf(a['nu'],'{:.2f}')} PostUp={sf(a['post_up'],'{:.0f}')}", file=sys.stderr)
    # ─── Exp B: Actionability ───
    print("\nExp B: Actionability...", file=sys.stderr)
    lines.append("\n## Exp B: Actionability Audit 2.0\n\n")
    lines.append("| θ | Term | AM | PCR |\n|---|------|-----|----|\n")
    for theta in ["safe", "shiny"]:
        rs = [run_multi("cct_v10", theta, sid) for sid in range(8)]
        all_am = {}; all_pcr = {}
        for r in rs:
            a = r.get("audit", {})
            for tn, v in a.get("AM", {}).items(): all_am.setdefault(tn, []).append(v)
            for tn, v in a.get("PCR", {}).items(): all_pcr.setdefault(tn, []).append(v)
        for tn in ["G","G_hier","G_res","G_duel","U","H","H_hier","H_res","H_duel"]:
            am_v = round(np.mean(all_am.get(tn, [0])), 6)
            pcr_v = round(np.mean(all_pcr.get(tn, [0])), 3)
            lines.append(f"| {theta} | {tn} | {am_v:.6f} | {pcr_v:.1%} |\n")
            print(f"  {theta}×{tn}: AM={am_v:.6f} PCR={pcr_v:.1%}", file=sys.stderr)
    # ─── Exp C: Hybrid ablation ───
    print("\nExp C: Hybrid ablation...", file=sys.stderr)
    lines.append("\n## Exp C: Hybrid Ablation\n\n")
    lines.append("| θ | Condition | **C** | **E** | SE | OTR | ν |\n|---|----------|---|---|---|---|---|\n")
    for theta in ["safe", "shiny"]:
        for abl in [None, "no_constraint", "no_unc", "no_stop"]:
            label = abl if abl else "full"
            rs = [run_multi("cct_v10", theta, sid, ablation=abl) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","se","otr","nu"]}
            lines.append("| {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                theta, label, sf(a["C"]), sf(a["E"]), sf(a["se"],"{:.4f}"), sf(a["otr"],"{:.3f}"), sf(a["nu"],"{:.3f}")))
            print(f"  {theta}×{label}: C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')}", file=sys.stderr)
    # ─── Exp D: Budget ───
    print("\nExp D: Budget...", file=sys.stderr)
    lines.append("\n## Exp D: Budget Sweep\n\n")
    lines.append("| θ | Budget | #T | **C** | **E** | ν | OTR | BdgBlk | SE |\n|---|-------|---|---|---|---|---|----|---|\n")
    for theta in ["safe", "shiny"]:
        for bud in [2.0, 4.0, 8.0]:
            rs = [run_multi("cct_v10", theta, sid, budget=bud) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","nu","otr","se"]}
            a["n_teach"] = avg_int(rs, "n_teach"); a["n_blocked"] = avg_int(rs, "n_blocked")
            lines.append("| {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                theta, bud, sf(a["n_teach"],"{:.0f}"), sf(a["C"]), sf(a["E"]),
                sf(a["nu"],"{:.3f}"), sf(a["otr"],"{:.3f}"), sf(a["n_blocked"],"{:.0f}"), sf(a["se"],"{:.4f}")))
            print(f"  {theta}×bud={bud}: C={sf(a['C'])} E={sf(a['E'])} BdgBlk={sf(a['n_blocked'],'{:.0f}')}", file=sys.stderr)
    with open(out / "cct_v10_report.md", "w") as f: f.writelines(lines)
    print(f"\nReport -> results/cct_v10_report.md", file=sys.stderr); print("Done.", file=sys.stderr)

if __name__ == "__main__": main()
