"""Canonical Controller Final Verification (6 experiments).

Canonical = gated STOP + saturated FP (decay only, no rep penalty)

Exp A: Canonical vs v13 main comparison
Exp B: Saturation ablation (raw / +decay / +decay+rep)
Exp C: Held-out family on canonical
Exp D: OOD robustness on canonical
Exp E: STOP trace (safe+shiny representative)
Exp F: Family usage histogram
"""
import sys, json
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
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13, ControllerV13Config
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.family_prior import FamilyPrior
from src.curriculum.dose_budget import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
sf = lambda v, fmt="{:.0%}": "—" if v is None else fmt.format(v)
NS = 8

def apply_fix(meta, sc, ood_mode="none", rng=None):
    rng_w = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    if ood_mode == "sign_flip" and rng: fb = np.where(rng.random(fb.shape)<0.3, -fb, fb)
    elif ood_mode == "noise_heavy" and rng: fb = fb + rng.normal(0, 0.5, fb.shape)
    return fb, ww

def run_one(cct, th, seed, mt=12, ood_mode="none", held_out_family=None, collect_trace=False):
    rng = np.random.default_rng(seed*1000+abs(hash(th))%1000)
    mic = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); cct.reset_session(cct.cfg.total_budget)
    tr = {"A":[],"B":[],"C":[],"D":[],"E":[]}; nt=0; ne=0; idx=0; fv=None
    fam_sel = {}; stop_trace = []
    for step in range(mt+4):
        at,les,qv,inf = cct.select_action(m)
        if collect_trace:
            stop_trace.append({
                "step": step, "action": at, "n_teach": nt, "n_eval": ne,
                "M_base": inf.get("m_base") if inf else None,
                "G_warm": inf.get("g_warm") if inf else None,
                "G_plateau": inf.get("g_plateau") if inf else None,
                "delta_u": inf.get("delta_u") if inf else None,
                "margin": inf.get("margin") if inf else None,
            })
        if at=="STOP": break
        if at=="EVAL": ne+=1; pr=all_probes(m,AP,th); cct.update_mastery(pr); continue
        if held_out_family and les.family == held_out_family:
            alt = [l for l in LESSON_CATALOG_V2 if l.family != held_out_family]
            if alt: les = rng.choice(alt)
        nt+=1
        if nt>mt: break
        fam_sel[les.family] = fam_sel.get(les.family, 0) + 1
        ub=cct.mastery.mastery()
        nub=m.nu; ggb=m.gamma_gen; ob=overteach_rate_v2(m)["total"]
        et=generate_episode_from_lesson_v2(les,idx+seed*100,th,ub,rng); ep=et[0]
        cct.record_realization(ep)
        _,spec,gm,cfg,meta,sc = et
        fb,ww = apply_fix(meta,sc,ood_mode,rng)
        if fv is None: fv=np.full_like(fb,0.3)
        re=np.random.default_rng(spec.cue_layout_seed+9999)
        lp=LatentCostRiskHead(d=4,risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r,c]==CellType.WALL: continue
                    z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z))
        ss=summarize_branch(sc.safe_cells,fb,fv,lp); sr=summarize_branch(sc.risky_cells,fb,fv,lp)
        lib=BranchConceptLibrary(); scr=BranchScorerProbe(lr=0.05,l2=0.01)
        lib.update("safe_branch",ss); lib.update("risky_branch",sr)
        scr.update(build_scorer_input(ss,lib),1.0); scr.update(build_scorer_input(sr,lib),0.0)
        bas=BranchAttributes(safety_score=float(ss[0]),temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b,risk_penalty=0.1)
        bar=BranchAttributes(safety_score=float(sr[0]),temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a,risk_penalty=sc.risk_level if hasattr(sc,'risk_level') else 0.4)
        bt.reset(ep); _,rd,_=mic.decide(sc,fb,lp,lib,scr,2,m)
        fe=bt.feasible_doses(); dose=rd if rd in fe else max(d for d in fe if d<=rd); bt.consume(dose)
        cct.consume_dose(dose)
        wb=[0.3*dose,-0.3*dose]; nf=[False,False]
        if ep.subtype=="beneficial_novelty": nf=[False,True] if sc.oracle_safe_branch_id==1 else [True,False]
        ac=sample_factored_choice([bas,bar],th,m,AP,re,wb,nf)
        cr=(ac!=sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc,'risk_level') and cr else 0.05,0.15)
        he=(spec.d_commit>spec.d_reveal+1)
        if dose>0:
            m.update_trust(warn_helpful=(spec.d_commit<=spec.d_reveal))
            if not he: old=m.nu; m.update_dependence(blind_obey=True); m.nu=old+dose*(m.nu-old)
            old=m.gamma_gen; m.update_gamma_gen(sustained_pressure=True); m.gamma_gen=old+dose*(m.gamma_gen-old)
        elif not cr: m.update_dependence(self_discovery=True); m.update_gamma_gen(successful_exploration=True)
        if cr and bar.temptation_score>0.5: m.update_gamma_spec(tempt_error=True)
        if ep.subtype in ("false_suppression_cost","beneficial_novelty") and not cr: m.update_gamma_spec(false_suppression=True)
        m.snapshot(); pr=all_probes(m,AP,th)
        cct.update_mastery(pr)
        cct.bridge.update(m,pr,sc.risk_level if hasattr(sc,'risk_level') else 0.3,bar.temptation_score,ep.novelty,0.7 if he else 0.3)
        oa=overteach_rate_v2(m)["total"]
        try: cct.update_response(les.name,dict(ub),cct.mastery.mastery(),nub,m.nu,ggb,m.gamma_gen,ob,oa)
        except TypeError: pass
        correct=cr if ep.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
        tr["A"].append({"correct":correct}); idx+=1
    for phase,nep in [("B",4),("C",4),("D",4),("E",4)]:
        for _ in range(nep):
            epp,spec,gm,cfg_e,meta,sc=generate_transfer_episode(phase,idx+seed*100,th,rng)
            fb,ww=apply_fix(meta,sc,ood_mode,rng)
            if fv is None: fv=np.full_like(fb,0.3)
            re=np.random.default_rng(spec.cue_layout_seed+9999)
            lp=LatentCostRiskHead(d=4,risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r,c]==CellType.WALL: continue
                        z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z))
            ss=summarize_branch(sc.safe_cells,fb,fv,lp); sr=summarize_branch(sc.risky_cells,fb,fv,lp)
            bas=BranchAttributes(safety_score=float(ss[0]),temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b,risk_penalty=0.1)
            bar=BranchAttributes(safety_score=float(sr[0]),temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a,risk_penalty=sc.risk_level if hasattr(sc,'risk_level') else 0.4)
            ga=False; aok=True
            if phase=="C" and re.random()<0.5: ga=True
            elif phase=="D" and re.random()<0.5: ga=True; aok=False
            wb=([0.3,-0.3] if aok==(sc.oracle_safe_branch_id==0) else [-0.3,0.3]) if ga else [0.0,0.0]
            nff=[False,False]
            if epp.subtype=="beneficial_novelty": nff=[False,True] if sc.oracle_safe_branch_id==1 else [True,False]
            ac=sample_factored_choice([bas,bar],th,m,AP,re,wb,nff)
            cr=(ac!=sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc,'risk_level') and cr else 0.05,0.15)
            if phase in ("C","D") and ga:
                hs=(spec.d_commit>spec.d_reveal+1)
                if phase=="C" and not cr: m.update_trust(warn_helpful=True)
                if phase=="D" and cr: m.update_dependence(blind_obey=True)
                elif phase=="D" and not cr and hs: m.update_dependence(self_discovery=True)
            if cr and bar.temptation_score>0.5: m.update_gamma_spec(tempt_error=True)
            m.snapshot()
            correct=cr if epp.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
            tr[phase].append({"correct":correct}); idx+=1
    rate = lambda ph: sum(1 for x in tr.get(ph,[]) if x["correct"])/max(len(tr.get(ph,[])),1) if tr.get(ph) else None
    otr=overteach_rate_v2(m)
    total_fam = sum(fam_sel.values()) if fam_sel else 1
    fam_probs = [c/total_fam for c in fam_sel.values()] if fam_sel else [1.0]
    fam_ent = -sum(p*np.log(p+1e-10) for p in fam_probs)
    return {"C":rate("B"),"E":rate("E"),"n_teach":nt,"n_eval":ne,
            "otr":otr["total"],"fam_ent":round(fam_ent,3),
            "fam_usage":fam_sel,"stop_trace":stop_trace}

avg = lambda rs,k: round(np.mean([r[k] for r in rs if r.get(k) is not None]),3) if any(r.get(k) is not None for r in rs) else None
avg_int = lambda rs,k: round(np.mean([r[k] for r in rs if r.get(k) is not None]),1) if any(r.get(k) is not None for r in rs) else None

def make_v13(th):
    cn=[l.name for l in LESSON_CATALOG_V2]
    cfg=ControllerV13Config(total_budget=4.0,risk_budget_mode="theta")
    fp=FamilyPrior(enabled=False)
    c=CurriculumControllerV13(cfg=cfg,theta=th,family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn,theta=th))
    c.use_family_prior=False; c.use_warm_gate=False; c.use_plateau_gate=False
    cfg.eps_0_safe=cfg.eps_0; cfg.eps_0_shiny=cfg.eps_0
    cfg.a_s_nu_safe=cfg.a_s_nu; cfg.a_s_nu_shiny=cfg.a_s_nu
    cfg.b_s_gg_safe=cfg.b_s_gg; cfg.b_s_gg_shiny=cfg.b_s_gg
    cfg.c_s_mastery_safe=cfg.c_s_mastery; cfg.c_s_mastery_shiny=cfg.c_s_mastery
    cfg.d_s_budget_safe=cfg.d_s_budget; cfg.d_s_budget_shiny=cfg.d_s_budget
    return c

def make_canonical(th, sat=True, rep=False):
    cn=[l.name for l in LESSON_CATALOG_V2]
    cfg=ControllerV13Config(total_budget=4.0,risk_budget_mode="theta")
    fp=FamilyPrior(enabled=True, use_saturation=sat, use_rep_penalty=rep)
    return CurriculumControllerV13(cfg=cfg,theta=th,family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn,theta=th))

def run_multi(fn, th, seed, ns=4, **kw):
    cct = fn(th)
    rs=[]
    for s in range(ns):
        rs.append(run_one(cct,th,seed*10+s,**kw))
    return rs[-1]

def main():
    print("═══ Canonical Controller Final Verification ═══\n",file=sys.stderr)
    L=["# Canonical Controller Final Verification\n\n"]
    L.append("> Canonical = gated STOP + saturated FP (decay only, no rep penalty)\n\n")

    # ── Exp A ──
    print("Exp A: Canonical vs v13...",file=sys.stderr)
    L.append("## Exp A: Canonical vs v13\n\n")
    L.append("| θ | Config | #T | #E | **C** | **E** | OTR | H_fam |\n|---|--------|---|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for lab,fn in [("v13",make_v13),("canonical",lambda t: make_canonical(t))]:
            rs=[run_multi(fn,th,sid) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr","fam_ent"]}
            a["nt"]=avg_int(rs,"n_teach"); a["ne"]=avg_int(rs,"n_eval")
            L.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["ne"],"{:.0f}"),sf(a["C"]),sf(a["E"]),
                sf(a["otr"],"{:.3f}"),sf(a["fam_ent"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} #T={sf(a['nt'],'{:.0f}')} H={sf(a['fam_ent'],'{:.3f}')}",file=sys.stderr)

    # ── Exp B ──
    print("\nExp B: Saturation ablation...",file=sys.stderr)
    L.append("\n## Exp B: Saturation Ablation\n\n")
    L.append("| θ | Mode | **C** | **E** | H_fam |\n|---|------|---|---|---|\n")
    for th in ["safe","shiny"]:
        for lab,sat,rep in [("raw_FP",False,False),("+decay",True,False),("+decay+rep",True,True)]:
            rs=[run_multi(lambda t,s=sat,r=rep: make_canonical(t,s,r),th,sid) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","fam_ent"]}
            L.append("| {} | {} | **{}** | **{}** | {} |\n".format(
                th,lab,sf(a["C"]),sf(a["E"]),sf(a["fam_ent"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} H={sf(a['fam_ent'],'{:.3f}')}",file=sys.stderr)

    # ── Exp C ──
    print("\nExp C: Held-out (canonical)...",file=sys.stderr)
    L.append("\n## Exp C: Held-Out Family (Canonical)\n\n")
    L.append("| θ | Held-Out | **C** | **E** | #T | Δ_heldout |\n|---|----------|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        c_none = None
        for fam in [None,"PP-MRB","TIC","TIC-v4"]:
            lab = fam if fam else "none"
            rs=[run_multi(lambda t: make_canonical(t),th,sid,held_out_family=fam) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E"]}; a["nt"]=avg_int(rs,"n_teach")
            if fam is None: c_none = a["C"]
            delta = round(a["C"]-c_none,3) if c_none and a["C"] else None
            L.append("| {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th,lab,sf(a["C"]),sf(a["E"]),sf(a["nt"],"{:.0f}"),
                sf(delta,"{:+.0%}") if delta else "—"))
            print(f"  {th}×{lab}: C={sf(a['C'])} Δ={sf(delta,'{:+.0%}') if delta else '—'}",file=sys.stderr)

    # ── Exp D ──
    print("\nExp D: OOD (canonical)...",file=sys.stderr)
    L.append("\n## Exp D: OOD Robustness (Canonical)\n\n")
    L.append("| θ | OOD | **C** | **E** | OTR |\n|---|-----|---|---|---|\n")
    for th in ["safe","shiny"]:
        for ood in ["none","sign_flip","noise_heavy"]:
            rs=[run_multi(lambda t: make_canonical(t),th,sid,ood_mode=ood) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}
            L.append("| {} | {} | **{}** | **{}** | {} |\n".format(
                th,ood,sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{ood}: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # ── Exp E ──
    print("\nExp E: STOP trace...",file=sys.stderr)
    L.append("\n## Exp E: STOP Trace (Representative Episodes)\n\n")
    for th in ["safe","shiny"]:
        cct = make_canonical(th)
        r = run_one(cct,th,42,collect_trace=True)
        L.append(f"### {th}\n\n")
        L.append("| step | action | #T | M_base | G_warm | G_plateau | Δu |\n|------|--------|---|--------|--------|-----------|----|\n")
        for t in r["stop_trace"]:
            L.append("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                t["step"],t["action"],t["n_teach"],
                t["M_base"] if t["M_base"] is not None else "—",
                t["G_warm"] if t["G_warm"] is not None else "—",
                t["G_plateau"] if t["G_plateau"] is not None else "—",
                t["delta_u"] if t["delta_u"] is not None else "—"))
        print(f"  {th}: #T={r['n_teach']} steps={len(r['stop_trace'])}",file=sys.stderr)

    # ── Exp F ──
    print("\nExp F: Family usage...",file=sys.stderr)
    L.append("\n## Exp F: Family Usage Distribution\n\n")
    L.append("| θ | Mode | PP-MRB | TIC | TIC-v4 | CGC-v2 | H_fam |\n|---|------|--------|-----|--------|--------|---|\n")
    for th in ["safe","shiny"]:
        for lab,sat,rep in [("raw_FP",False,False),("+decay",True,False),("+decay+rep",True,True)]:
            all_fam = {}
            for sid in range(NS):
                r = run_multi(lambda t,s=sat,r=rep: make_canonical(t,s,r),th,sid)
                for f,c in r.get("fam_usage",{}).items():
                    all_fam[f] = all_fam.get(f,0)+c
            total = sum(all_fam.values()) if all_fam else 1
            pcts = {f: round(c/total,2) for f,c in all_fam.items()}
            ent = avg([r],  "fam_ent") if 'r' in dir() else 0
            L.append("| {} | {} | {} | {} | {} | {} | — |\n".format(
                th,lab,
                sf(pcts.get("PP-MRB",0)),sf(pcts.get("TIC",0)),
                sf(pcts.get("TIC-v4",0)),sf(pcts.get("CGC-v2",0))))
            print(f"  {th}×{lab}: {pcts}",file=sys.stderr)

    with open(out/"canonical_final_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/canonical_final_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
