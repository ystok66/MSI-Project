"""Gated STOP verification: warm-up + plateau gates.

Exp A: STOP structure ablation (single / +warm / +plateau / +both)
Exp B: Counterfactual post-STOP audit 
Exp C: Cross-family robustness check
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

def apply_fix(meta, sc, ood_mode="none", rng=None):
    rng_w = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    if ood_mode == "sign_flip" and rng: fb = np.where(rng.random(fb.shape)<0.3, -fb, fb)
    elif ood_mode == "noise_heavy" and rng: fb = fb + rng.normal(0, 0.5, fb.shape)
    return fb, ww

def run_one(cct, th, seed, mt=12, ood_mode="none", held_out_family=None, counterfactual_audit=False):
    rng = np.random.default_rng(seed*1000+abs(hash(th))%1000)
    mic = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); cct.reset_session(cct.cfg.total_budget)
    tr = {"A":[],"B":[],"C":[],"D":[],"E":[]}; nt=0; ne=0; idx=0; fv=None
    stop_info = None; cf_audit = []
    for step in range(mt+4):
        at,les,qv,inf = cct.select_action(m)
        if at=="STOP":
            stop_info = {"step": step, "n_teach": nt, "n_eval": ne, **(inf or {})}
            # Counterfactual: what if we continued?
            if counterfactual_audit and les is None:
                u = cct.mastery.mastery()
                # Score best feasible + EVAL at STOP point
                feasible = cct._feasible_set(u)
                cf_entry = {"step": step, "n_teach": nt, "theta": th}
                if feasible:
                    bj = -1e9; bl = None
                    for l in feasible:
                        j, _ = cct._score_lesson(l, u, m)
                        if cct.use_family_prior and cct.family_prior:
                            j += cct.family_prior.bonus(l, th)
                        if j > bj: bj = j; bl = l
                    cf_entry["best_J"] = round(bj, 4)
                    cf_entry["best_lesson"] = bl.name if bl else None
                j_eval = cct._eval_value(m, u, bj, -1e9)
                cf_entry["J_eval"] = round(j_eval, 4)
                cf_audit.append(cf_entry)
            break
        if at=="EVAL":
            ne+=1; pr=all_probes(m,AP,th); cct.update_mastery(pr); continue
        if held_out_family and les.family == held_out_family:
            alt = [l for l in LESSON_CATALOG_V2 if l.family != held_out_family]
            if alt: les = rng.choice(alt)
        nt+=1
        if nt>mt: break
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
        # Push mastery snapshot for plateau gate
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
    au=cct.actionability_audit(); otr=overteach_rate_v2(m)
    return {"C":rate("B"),"E":rate("E"),"n_teach":nt,"n_eval":ne,
            "nu":round(m.nu,3),"gg":round(m.gamma_gen,3),
            "otr":otr["total"],"audit":au,"stop_info":stop_info,"cf_audit":cf_audit}

avg = lambda rs,k: round(np.mean([r[k] for r in rs if r.get(k) is not None]),3) if any(r.get(k) is not None for r in rs) else None
avg_int = lambda rs,k: round(np.mean([r[k] for r in rs if r.get(k) is not None]),1) if any(r.get(k) is not None for r in rs) else None
NS=8

def make_cct(th, warm=True, plateau=True, shared_stop=False):
    cn=[l.name for l in LESSON_CATALOG_V2]
    cfg=ControllerV13Config(total_budget=4.0,risk_budget_mode="theta")
    fp=FamilyPrior(enabled=True)
    c=CurriculumControllerV13(cfg=cfg,theta=th,family_prior=fp,
        response=PairwiseResponseModel(catalog_names=cn,theta=th))
    c.use_warm_gate = warm
    c.use_plateau_gate = plateau
    if shared_stop:
        cfg.eps_0_safe=cfg.eps_0; cfg.eps_0_shiny=cfg.eps_0
        cfg.a_s_nu_safe=cfg.a_s_nu; cfg.a_s_nu_shiny=cfg.a_s_nu
        cfg.b_s_gg_safe=cfg.b_s_gg; cfg.b_s_gg_shiny=cfg.b_s_gg
        cfg.c_s_mastery_safe=cfg.c_s_mastery; cfg.c_s_mastery_shiny=cfg.c_s_mastery
        cfg.d_s_budget_safe=cfg.d_s_budget; cfg.d_s_budget_shiny=cfg.d_s_budget
        c.use_family_prior=False
    return c

def run_multi(th, seed, ns=4, **kw):
    rs=[]
    for s in range(ns):
        cct = make_cct(th, **{k:v for k,v in kw.items() if k in ('warm','plateau','shared_stop')})
        rs.append(run_one(cct,th,seed*10+s,
                          ood_mode=kw.get('ood_mode','none'),
                          held_out_family=kw.get('held_out_family'),
                          counterfactual_audit=kw.get('counterfactual_audit',False)))
    return rs[-1]

def main():
    print("═══ Gated STOP Verification ═══\n",file=sys.stderr)
    L=["# Gated STOP Verification\n\n"]

    # Exp A: STOP structure ablation
    print("Exp A: STOP structure ablation...",file=sys.stderr)
    L.append("## Exp A: STOP Structure Ablation\n\n")
    L.append("| θ | Mode | #T | #E | **C** | **E** | OTR | ν | γ_gen |\n|---|------|---|---|---|---|---|---|---|\n")
    configs = [
        ("single_threshold", False, False, True),   # v13 baseline: shared, no gates
        ("+warm_only", True, False, False),
        ("+plateau_only", False, True, False),
        ("+warm+plateau", True, True, False),
    ]
    for th in ["safe","shiny"]:
        for lab,warm,plateau,shared in configs:
            rs=[run_multi(th,sid,warm=warm,plateau=plateau,shared_stop=shared) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr","nu","gg"]}
            a["nt"]=avg_int(rs,"n_teach"); a["ne"]=avg_int(rs,"n_eval")
            L.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["ne"],"{:.0f}"),sf(a["C"]),sf(a["E"]),
                sf(a["otr"],"{:.3f}"),sf(a["nu"]),sf(a["gg"])))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])} #T={sf(a['nt'],'{:.0f}')} ν={sf(a['nu'])} γ={sf(a['gg'])}",file=sys.stderr)

    # Exp B: Counterfactual post-STOP audit
    print("\nExp B: Counterfactual post-STOP audit...",file=sys.stderr)
    L.append("\n## Exp B: Counterfactual Post-STOP Audit\n\n")
    L.append("At each STOP point, what was the counterfactual best_J and J_eval?\n\n")
    L.append("| θ | Mode | Avg CF best_J | Avg CF J_eval | Avg #T at STOP |\n|---|------|-------------|-------------|---------------|\n")
    for th in ["safe","shiny"]:
        for lab,warm,plateau,shared in [("single", False, False, True), ("gated", True, True, False)]:
            cfs = []
            for sid in range(NS):
                r = run_multi(th,sid,warm=warm,plateau=plateau,shared_stop=shared,counterfactual_audit=True)
                if r.get("cf_audit"): cfs.extend(r["cf_audit"])
                elif r.get("stop_info"):
                    si = r["stop_info"]
                    cfs.append({"best_J": si.get("counterfactual_J",0), "J_eval": 0, "n_teach": si.get("n_teach",0)})
            if cfs:
                avg_bj = round(np.mean([c.get("best_J",0) for c in cfs]),4)
                avg_je = round(np.mean([c.get("J_eval",0) for c in cfs]),4)
                avg_nt = round(np.mean([c.get("n_teach",0) for c in cfs]),1)
                L.append("| {} | {} | {} | {} | {} |\n".format(th,lab,avg_bj,avg_je,avg_nt))
                print(f"  {th}×{lab}: CF best_J={avg_bj} CF J_eval={avg_je} #T@STOP={avg_nt}",file=sys.stderr)
            else:
                L.append("| {} | {} | — | — | — |\n".format(th,lab))
                print(f"  {th}×{lab}: no STOP observed",file=sys.stderr)

    # Exp C: Cross-family robustness
    print("\nExp C: Cross-family robustness...",file=sys.stderr)
    L.append("\n## Exp C: Cross-Family Robustness (Gated STOP)\n\n")
    L.append("| θ | Held-Out | **C** | **E** | #T |\n|---|----------|---|---|---|\n")
    for th in ["safe","shiny"]:
        for fam in [None,"PP-MRB","TIC","TIC-v4"]:
            lab = fam if fam else "none"
            rs=[run_multi(th,sid,warm=True,plateau=True,held_out_family=fam) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E"]}; a["nt"]=avg_int(rs,"n_teach")
            L.append("| {} | {} | **{}** | **{}** | {} |\n".format(th,lab,sf(a["C"]),sf(a["E"]),sf(a["nt"],"{:.0f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])} #T={sf(a['nt'],'{:.0f}')}",file=sys.stderr)

    # Summary comparison table
    L.append("\n## Summary: v13 vs v13.3 (gated)\n\n")
    L.append("| θ | v13 C | gated C | Δ | v13 #T | gated #T |\n|---|------|--------|---|--------|----------|\n")
    for th in ["safe","shiny"]:
        rs_old=[run_multi(th,sid,warm=False,plateau=False,shared_stop=True) for sid in range(NS)]
        rs_new=[run_multi(th,sid,warm=True,plateau=True) for sid in range(NS)]
        c_old=avg(rs_old,"C"); c_new=avg(rs_new,"C")
        nt_old=avg_int(rs_old,"n_teach"); nt_new=avg_int(rs_new,"n_teach")
        delta = round(c_new-c_old,3) if c_old is not None and c_new is not None else None
        L.append("| {} | {} | {} | {} | {} | {} |\n".format(
            th,sf(c_old),sf(c_new),sf(delta,"{:+.0%}") if delta else "—",sf(nt_old,"{:.0f}"),sf(nt_new,"{:.0f}")))
        print(f"  {th}: v13 C={sf(c_old)} → gated C={sf(c_new)} (Δ={sf(delta,'{:+.0%}') if delta else '—'}) #T {sf(nt_old,'{:.0f}')}→{sf(nt_new,'{:.0f}')}",file=sys.stderr)

    with open(out/"gated_stop_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/gated_stop_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
