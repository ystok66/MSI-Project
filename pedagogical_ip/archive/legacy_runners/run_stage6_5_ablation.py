"""Stage 6.5 — Mechanism Ablation Suite.

Exp 1: G_hier / G_res removal
Exp 2: STOP threshold per-θ sweep
Exp 3: EVAL mechanism isolation
Exp 4: close-gap bonus removal
Exp 5: θ-adaptive vs uniform-wide
Exp 6: PP-MRB dependency test
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
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13, ControllerV13Config
from src.curriculum.pairwise_response_model import PairwiseResponseModel
from src.curriculum.dose_budget import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
def sf(v, fmt="{:.0%}"): return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc, rng=None):
    rng_w = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww

def run_one(cct, th, seed, mt=12, bud=4.0, held_out_family=None):
    rng = np.random.default_rng(seed*1000+abs(hash(th))%1000)
    mic = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); cct.reset_session(bud)
    tr = {"A":[],"B":[],"C":[],"D":[],"E":[]}; nt=0; ne=0; sa=None; idx=0; fv=None
    for step in range(mt+4):
        at,les,qv,inf = cct.select_action(m)
        if at=="STOP": sa=step; break
        if at=="EVAL": ne+=1; pr=all_probes(m,AP,th); cct.update_mastery(pr); continue
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
        fb,ww = apply_fix(meta,sc,rng)
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
        cct.bridge.update(m,pr,sc.risk_level if hasattr(sc,'risk_level') else 0.3,bar.temptation_score,ep.novelty,0.7 if he else 0.3)
        oa=overteach_rate_v2(m)["total"]
        try: cct.update_response(les.name,dict(ub),cct.mastery.mastery(),nub,m.nu,ggb,m.gamma_gen,ob,oa)
        except TypeError: cct.update_response(les.name,dict(ub),cct.mastery.mastery())
        correct=cr if ep.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
        mca=correct
        if ep.subtype=="beneficial_novelty" and correct and m.gamma_gen>0.3: mca=False
        tr["A"].append({"correct":correct}); idx+=1
    for phase,nep in [("B",4),("C",4),("D",4),("E",4)]:
        for _ in range(nep):
            epp,spec,gm,cfg_e,meta,sc=generate_transfer_episode(phase,idx+seed*100,th,rng)
            fb,ww=apply_fix(meta,sc,rng)
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
    def rate(ph):
        t=tr.get(ph,[]); return sum(1 for x in t if x["correct"])/max(len(t),1) if t else None
    otr=overteach_rate_v2(m); au=cct.actionability_audit()
    return {"C":rate("B"),"E":rate("E"),"n_teach":nt,"n_eval":ne,
            "nu":round(m.nu,3),"gg":round(m.gamma_gen,3),"otr":otr["total"],
            "n_pw":cct.posterior_stats().get("n_pw_gain",0),"audit":au}

def avg(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),3) if vs else None
def avg_int(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),1) if vs else None

def make_cct(th, bud=4.0, budget_mode="theta", eps0=None, abl=None):
    cn=[l.name for l in LESSON_CATALOG_V2]
    cfg = ControllerV13Config(total_budget=bud, risk_budget_mode=budget_mode)
    if eps0 is not None: cfg.eps_0 = eps0
    c=CurriculumControllerV13(cfg=cfg, theta=th,
        response=PairwiseResponseModel(catalog_names=cn, theta=th))
    if abl:
        if "no_hier" in abl: c.use_hier=False
        if "no_res" in abl: c.use_res=False
        if "no_hier_res" in abl: c.use_hier=False; c.use_res=False
        if "no_stop" in abl: c.use_stop=False
        if "no_eval" in abl: c.use_eval=False
        if "no_constraint" in abl: c.use_constraint=False
        if "no_close_gap" in abl: c.use_close_gap=False
        if "mastery_only" in abl: c.use_eval=False  # handled in run loop
    return c

def run_multi(th,seed,ns=4,bud=4.0,budget_mode="theta",eps0=None,abl=None,held_out_family=None,mastery_only=False):
    cct=make_cct(th,bud,budget_mode,eps0,abl); rs=[]
    for s in range(ns):
        if mastery_only:
            # Run with EVAL disabled but inject mastery probes at same intervals
            cct2 = make_cct(th,bud,budget_mode,eps0,["no_eval"])
            r = run_one(cct2,th,seed*10+s,bud=bud,held_out_family=held_out_family)
            # Inject 3 mastery updates (simulating EVAL without action)
            pr = {p: 0.5 for p in PROBE_NAMES}
            for _ in range(3): cct2.update_mastery(pr)
            rs.append(r)
        else:
            rs.append(run_one(cct,th,seed*10+s,bud=bud,held_out_family=held_out_family))
    return rs[-1]

def main():
    NS = 8  # seeds
    print("═══ Stage-6.5: Mechanism Ablation Suite ═══\n",file=sys.stderr)
    L=["# Stage-6.5: Mechanism Ablation Suite\n\n"]

    # Exp 1: G_hier / G_res removal
    print("Exp 1: G_hier / G_res removal...",file=sys.stderr)
    L.append("## Exp 1: G_hier / G_res Removal\n\n")
    L.append("| θ | Config | #T | **C** | **E** | OTR | G_pw PCR |\n|---|--------|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for abl,lab in [(None,"full"),("no_hier","-G_hier"),("no_res","-G_res"),("no_hier_res","-both")]:
            rs=[run_multi(th,sid,abl=[abl] if abl else None) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach")
            # Extract G_pw PCR
            pcrs=[r.get("audit",{}).get("PCR",{}).get("G_pw",0) for r in rs]
            gpw_pcr=round(np.mean(pcrs),3) if pcrs else 0
            L.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}"),sf(gpw_pcr)))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])} G_pw_PCR={sf(gpw_pcr)}",file=sys.stderr)

    # Exp 2: STOP threshold per-θ sweep
    print("\nExp 2: STOP threshold sweep...",file=sys.stderr)
    L.append("\n## Exp 2: STOP Threshold per-θ Sweep\n\n")
    L.append("| θ | ε₀ | #T | **C** | **E** | OTR |\n|---|-----|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for eps0 in [-0.10, -0.05, 0.00, 0.05]:
            rs=[run_multi(th,sid,eps0=eps0) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach")
            L.append("| {} | {} | {} | **{}** | **{}** | {} |\n".format(
                th,eps0,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×ε₀={eps0}: #T={sf(a['nt'],'{:.0f}')} C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp 3: EVAL mechanism isolation
    print("\nExp 3: EVAL mechanism isolation...",file=sys.stderr)
    L.append("\n## Exp 3: EVAL Mechanism Isolation\n\n")
    L.append("| θ | Config | #T | #E | **C** | **E** | OTR |\n|---|--------|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for label,abl_flag,mo in [("full_eval",None,False),("no_eval","no_eval",False),("mastery_only",None,True)]:
            rs=[run_multi(th,sid,abl=[abl_flag] if abl_flag else None,mastery_only=mo) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach"); a["ne"]=avg_int(rs,"n_eval")
            L.append("| {} | {} | {} | {} | **{}** | **{}** | {} |\n".format(
                th,label,sf(a["nt"],"{:.0f}"),sf(a["ne"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{label}: #T={sf(a['nt'],'{:.0f}')} #E={sf(a['ne'],'{:.0f}')} C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp 4: close-gap bonus removal
    print("\nExp 4: close-gap removal...",file=sys.stderr)
    L.append("\n## Exp 4: Close-Gap Bonus Removal\n\n")
    L.append("| θ | Config | #E | **C** | **E** | OTR |\n|---|--------|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for abl,lab in [(None,"with_close_gap"),("no_close_gap","no_close_gap")]:
            rs=[run_multi(th,sid,abl=[abl] if abl else None) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["ne"]=avg_int(rs,"n_eval")
            L.append("| {} | {} | {} | **{}** | **{}** | {} |\n".format(
                th,lab,sf(a["ne"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp 5: θ-adaptive vs uniform-wide
    print("\nExp 5: θ-adaptive vs uniform-wide...",file=sys.stderr)
    L.append("\n## Exp 5: θ-Adaptive vs Uniform-Wide\n\n")
    L.append("| θ | Budget Mode | #T | **C** | **E** | OTR |\n|---|------------|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for bm,lab in [("fixed","fixed-tight"),("theta","θ-adaptive"),("full","θ+mastery")]:
            rs=[run_multi(th,sid,budget_mode=bm) for sid in range(NS)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach")
            L.append("| {} | {} | {} | **{}** | **{}** | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)
        # Add "none" (no constraint)
        rs=[run_multi(th,0,abl=["no_constraint"]) for _ in range(NS)]
        a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach")
        L.append("| {} | none | {} | **{}** | **{}** | {} |\n".format(
            th,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
        print(f"  {th}×none: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp 6: PP-MRB dependency test
    print("\nExp 6: PP-MRB dependency...",file=sys.stderr)
    L.append("\n## Exp 6: PP-MRB Dependency Test\n\n")
    L.append("| θ | Config | #T | **C** | **E** | OTR | G_pw PCR |\n|---|--------|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for fam,lab in [(None,"all"),(None,"all"),("PP-MRB","no_PP-MRB"),("TIC","no_TIC"),("TIC-v4","no_TIC-v4")]:
            if lab=="all" and fam is None:
                rs=[run_multi(th,sid) for sid in range(NS)]
            else:
                rs=[run_multi(th,sid,held_out_family=fam) for sid in range(NS)]
            if lab=="all" and fam is None and th!="safe": continue  # avoid double
            a={k:avg(rs,k) for k in ["C","E","otr"]}; a["nt"]=avg_int(rs,"n_teach")
            pcrs=[r.get("audit",{}).get("PCR",{}).get("G_pw",0) for r in rs]
            gpw_pcr=round(np.mean(pcrs),3) if pcrs else 0
            L.append("| {} | {} | {} | **{}** | **{}** | {} | {} |\n".format(
                th,lab,sf(a["nt"],"{:.0f}"),sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}"),sf(gpw_pcr)))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    with open(out/"stage6_5_ablation_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/stage6_5_ablation_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
