"""Stage-4 — Canonical Controller v13 Experiment Suite.

Exp A: v13 vs v12 vs v11 vs v8 (main comparison)
Exp B: STOP ablation (learned vs no_stop, same-#T fair)
Exp C: EVAL ablation (full vs no_eval)
Exp D: Actionability audit 4.1 (v13 G_pw PCR preserved?)
Exp E: Constraint ablation (constraint vs none)
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
from src.curriculum.curriculum_controller_v11 import CurriculumControllerV11
from src.curriculum.curriculum_controller_v12 import CurriculumControllerV12
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13
from src.curriculum.lesson_response_model_v3 import LessonResponseModelV3
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.dose_budget import DoseBudgetTracker
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
def fixed_lesson(st, rng):
    ls = {"ppmrb_only":["ppmrb_standard","ppmrb_self_discovery"]}
    name = rng.choice(ls.get(st, [l.name for l in LESSON_CATALOG_V2]))
    return LESSON_V2_BY_NAME.get(name, LESSON_CATALOG_V2[0])

def run_one(cct, st, th, seed, mt=12, bud=4.0):
    rng = np.random.default_rng(seed*1000+abs(hash(th))%1000)
    mic = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker()
    if cct and hasattr(cct,'reset_session'): cct.reset_session(bud)
    br = cct.bridge if cct else TrainableBridge()
    tr = {"A":[],"B":[],"C":[],"D":[],"E":[]}; nt=0; ne=0; sa=None; idx=0; fv=None
    for step in range(mt+4):
        if cct: at,les,qv,inf = cct.select_action(m)
        else: at,les,qv = "TEACH",fixed_lesson(st,rng),0
        if at=="STOP": sa=step; break
        if at=="EVAL": ne+=1; pr=all_probes(m,AP,th); cct.update_mastery(pr); continue
        nt+=1
        if nt>mt: break
        ub=cct.mastery.mastery() if cct else {p:0.5 for p in ["RC","TR","EP","VA","IA"]}
        nub=m.nu; ggb=m.gamma_gen; ob=overteach_rate_v2(m)["total"]
        et=generate_episode_from_lesson_v2(les,idx+seed*100,th,ub,rng); ep=et[0]
        if cct and hasattr(cct,'record_realization'): cct.record_realization(ep)
        _,spec,gm,cfg,meta,sc = et
        fb,ww = apply_fix(meta,sc)
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
        if cct and hasattr(cct,'consume_dose'): cct.consume_dose(dose)
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
        br.update(m,pr,sc.risk_level if hasattr(sc,'risk_level') else 0.3,bar.temptation_score,ep.novelty,0.7 if he else 0.3)
        if cct and hasattr(cct,'update_response'):
            oa=overteach_rate_v2(m)["total"]
            try: cct.update_response(les.name,dict(ub),cct.mastery.mastery(),nub,m.nu,ggb,m.gamma_gen,ob,oa)
            except TypeError: cct.update_response(les.name,dict(ub),cct.mastery.mastery())
        correct=cr if ep.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
        mca=correct
        if ep.subtype=="beneficial_novelty" and correct and m.gamma_gen>0.3: mca=False
        tr["A"].append({"correct":correct,"mca":mca}); idx+=1
    for phase,nep in [("B",4),("C",4),("D",4),("E",4)]:
        for _ in range(nep):
            epp,spec,gm,cfg,meta,sc=generate_transfer_episode(phase,idx+seed*100,th,rng)
            fb,ww=apply_fix(meta,sc)
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
            mca=correct
            if epp.subtype=="beneficial_novelty" and correct and m.gamma_gen>0.3: mca=False
            tr[phase].append({"correct":correct,"mca":mca}); idx+=1
    def rate(ph,key="correct"):
        t=tr.get(ph,[]); return sum(1 for x in t if x[key])/max(len(t),1) if t else None
    otr=overteach_rate_v2(m)
    se=stop_efficiency(sum(rate(p) or 0 for p in ["B","C","D","E"])/4,nt) if nt>0 else 0.0
    pu=cct.posterior_stats() if cct and hasattr(cct,'posterior_stats') else {}
    au=cct.actionability_audit() if cct and hasattr(cct,'actionability_audit') else {}
    cs=cct.controller_summary() if cct and hasattr(cct,'controller_summary') else {}
    return {"B":rate("B"),"C":rate("C"),"D":rate("D"),"E":rate("E"),
            "n_teach":nt,"n_eval":ne,"stopped":sa is not None,
            "n_blocked":cct.budget_blocked_count if cct and hasattr(cct,'budget_blocked_count') else 0,
            "nu":round(m.nu,3),"gg":round(m.gamma_gen,3),"otr":otr["total"],"se":round(se,4),
            "post_up":pu.get("n_updated",0),"n_pw":pu.get("n_pw_gain",0),"audit":au,
            "stop_margin":cs.get("stop_margin"),
            "stop_reason":cs.get("stop_reason")}

def avg(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),3) if vs else None
def avg_int(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),1) if vs else None

def make_cct(st,th,bud,abl=None):
    cn=[l.name for l in LESSON_CATALOG_V2]
    if st in ("cct_v13",) or (abl and st not in ("cct_v12","cct_v11","cct_v8")):
        c=CurriculumControllerV13(theta=th)
        c.cfg.total_budget=bud
        if abl=="no_stop": c.use_stop=False
        if abl=="no_eval": c.use_eval=False
        if abl=="no_constraint": c.use_constraint=False
        if abl=="no_prereq": c.use_prerequisite=False
        if abl=="no_replay": c.use_replay=False
        return c
    elif st=="cct_v12":
        return CurriculumControllerV12(theta=th,total_budget=bud,
            response=PairwiseResponseModel(catalog_names=cn,theta=th))
    elif st=="cct_v11":
        return CurriculumControllerV11(theta=th,total_budget=bud)
    elif st=="cct_v8":
        return CurriculumControllerV8(theta=th,total_budget=bud,response=LessonResponseModelV3())
    return None

def run_multi(st,th,seed,ns=4,bud=4.0,abl=None):
    cct=make_cct(st,th,bud,abl); rs=[]
    for s in range(ns): rs.append(run_one(cct,st,th,seed*10+s,bud=bud))
    return rs[-1]

def main():
    print("═══ Stage-4: Canonical Controller v13 ═══\n",file=sys.stderr)
    L=["# Stage-4: Canonical Controller v13\n\n"]

    # Exp A: Main comparison
    strats=["ppmrb_only","cct_v8","cct_v11","cct_v12","cct_v13"]
    L.append("## Exp A: v13 vs v12 vs v11 vs v8 vs Fixed\n\n")
    L.append("| θ | Strat | #T | #Eval | Stop | **C** | **E** | SE | OTR | ν | #PW |\n|---|------|---|-----|---|---|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for s in strats:
            rs=[run_multi(s,th,sid) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","se","otr","nu","gg"]}
            a["nt"]=avg_int(rs,"n_teach"); a["ne"]=avg_int(rs,"n_eval")
            a["sf"]=round(sum(1 for r in rs if r["stopped"])/len(rs),2)
            a["npw"]=avg_int(rs,"n_pw")
            L.append("| {} | {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                th,s,sf(a["nt"],"{:.0f}"),sf(a["ne"],"{:.0f}"),sf(a["sf"]),sf(a["C"]),sf(a["E"]),
                sf(a["se"],"{:.4f}"),sf(a["otr"],"{:.3f}"),sf(a["nu"],"{:.3f}"),sf(a["npw"],"{:.0f}")))
            print(f"  {th}×{s}: #T={sf(a['nt'],'{:.0f}')} #E={sf(a['ne'],'{:.0f}')} C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')} #PW={sf(a['npw'],'{:.0f}')}",file=sys.stderr)

    # Exp B: STOP ablation
    print("\nExp B: STOP ablation...",file=sys.stderr)
    L.append("\n## Exp B: STOP Ablation\n\n| θ | Cond | #T | **C** | **E** | OTR | ν | γ_gen |\n|---|------|---|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for abl,lab in [(None,"learned_stop"),("no_stop","no_stop")]:
            rs=[run_multi("cct_v13",th,sid,abl=abl) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","otr","nu","gg"]}; a["nt"]=avg_int(rs,"n_teach")
            L.append("| {} | {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}"),sf(a["nu"],"{:.3f}"),sf(a["gg"],"{:.3f}")))
            print(f"  {th}×{lab}: #T={sf(a['nt'],'{:.0f}')} C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')} ν={sf(a['nu'],'{:.2f}')} γg={sf(a['gg'],'{:.2f}')}",file=sys.stderr)

    # Exp C: EVAL ablation
    print("\nExp C: EVAL ablation...",file=sys.stderr)
    L.append("\n## Exp C: EVAL Ablation\n\n| θ | Cond | #T | #Eval | **C** | **E** | OTR |\n|---|------|---|-----|---|---|---|\n")
    for th in ["safe","shiny"]:
        for abl,lab in [(None,"full_eval"),("no_eval","no_eval")]:
            rs=[run_multi("cct_v13",th,sid,abl=abl) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach"); a["ne"]=avg_int(rs,"n_eval")
            L.append("| {} | {} | {} | {} | **{}** | **{}** | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["ne"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{lab}: #T={sf(a['nt'],'{:.0f}')} #E={sf(a['ne'],'{:.0f}')} C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp D: Actionability audit
    print("\nExp D: Actionability audit 4.1...",file=sys.stderr)
    L.append("\n## Exp D: Actionability Audit 4.1\n\n| θ | Term | AM | PCR |\n|---|------|----|-----|\n")
    for th in ["safe","shiny"]:
        rs=[run_multi("cct_v13",th,sid) for sid in range(8)]
        aa={}; ap={}
        for r in rs:
            a=r.get("audit",{})
            for tn,v in a.get("AM",{}).items(): aa.setdefault(tn,[]).append(v)
            for tn,v in a.get("PCR",{}).items(): ap.setdefault(tn,[]).append(v)
        for tn in ["G","G_hier","G_res","G_pw","U","H"]:
            amv=round(np.mean(aa.get(tn,[0])),6); pcrv=round(np.mean(ap.get(tn,[0])),3)
            L.append(f"| {th} | {tn} | {amv:.6f} | {pcrv:.1%} |\n")
            if tn in ("G_pw","U","G"):
                print(f"  {th}×{tn}: AM={amv:.6f} PCR={pcrv:.1%}",file=sys.stderr)

    # Exp E: Constraint ablation
    print("\nExp E: Constraint ablation...",file=sys.stderr)
    L.append("\n## Exp E: Constraint Ablation\n\n| θ | Cond | **C** | **E** | OTR | ν |\n|---|------|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for abl,lab in [(None,"constraint"),("no_constraint","none")]:
            rs=[run_multi("cct_v13",th,sid,abl=abl) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","otr","nu"]}
            L.append("| {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th,lab,sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}"),sf(a["nu"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')}",file=sys.stderr)

    with open(out/"stage4_v13_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/stage4_v13_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
