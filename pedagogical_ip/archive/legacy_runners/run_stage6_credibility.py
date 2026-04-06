"""Stage-6 — Credibility Closure.

Exp A: Adversarial OOD (sign-flip, prior shift, held-out family)
Exp B: STOP calibration audit (margin vs counterfactual gain)
Exp C: EVAL calibration audit (trigger reason, rank-change rate)
Exp D: OTR decomposition (teach vs eval overhead)
Exp E: Actionability regression (G_pw PCR preserved?)
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
from src.curriculum.pedagogical_framework import PedagogicalFramework, FrameworkConfig
from src.curriculum.dose_budget import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.curriculum_metrics import stop_efficiency
out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
def sf(v, fmt="{:.0%}"): return "—" if v is None else fmt.format(v)
def sf4(v): return "—" if v is None else f"{v:.4f}"

def apply_fix(meta, sc, ood_mode="none", rng=None):
    rng_w = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    if ood_mode == "sign_flip" and rng is not None:
        mask = rng.random(fb.shape) < 0.3
        fb = np.where(mask, -fb, fb)
    elif ood_mode == "noise_heavy" and rng is not None:
        fb = fb + rng.normal(0, 0.5, fb.shape)
    elif ood_mode == "scale_shift" and rng is not None:
        fb = fb * rng.uniform(0.3, 1.7, fb.shape)
    return fb, ww

def run_one(fw, th, seed, mt=12, bud=4.0, ood_mode="none", held_out_family=None):
    rng = np.random.default_rng(seed*1000+abs(hash(th))%1000)
    mic = BCICTv4(agent_params=AP); m = FactoredInternalizationState(); m.snapshot()
    bt = DoseBudgetTracker(); fw.reset_session()
    tr = {"A":[],"B":[],"C":[],"D":[],"E":[]}; nt=0; ne=0; sa=None; idx=0; fv=None
    for step in range(mt+4):
        action, les, info = fw.macro_step(m)
        if action=="STOP": sa=step; break
        if action=="EVAL":
            ne+=1; fw.run_eval(m); continue
        # TEACH
        if held_out_family and les.family == held_out_family:
            # Skip held-out family lesson, pick another
            alt = [l for l in LESSON_CATALOG_V2 if l.family != held_out_family]
            if alt: les = rng.choice(alt)
        nt+=1
        if nt>mt: break
        ub=fw.controller.mastery.mastery()
        nub=m.nu; ggb=m.gamma_gen; ob=overteach_rate_v2(m)["total"]
        et=generate_episode_from_lesson_v2(les,idx+seed*100,th,ub,rng); ep=et[0]
        fw.controller.record_realization(ep)
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
        fw.controller.consume_dose(dose)
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
        fw.controller.bridge.update(m,pr,sc.risk_level if hasattr(sc,'risk_level') else 0.3,bar.temptation_score,ep.novelty,0.7 if he else 0.3)
        oa=overteach_rate_v2(m)["total"]
        try: fw.update_after_teach(les.name,dict(ub),fw.controller.mastery.mastery(),nub,m.nu,ggb,m.gamma_gen,ob,oa)
        except TypeError: pass
        correct=cr if ep.subtype in ("false_suppression_cost","beneficial_novelty") else (ac==sc.oracle_safe_branch_id)
        mca=correct
        if ep.subtype=="beneficial_novelty" and correct and m.gamma_gen>0.3: mca=False
        tr["A"].append({"correct":correct,"mca":mca}); idx+=1
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
            mca=correct
            if epp.subtype=="beneficial_novelty" and correct and m.gamma_gen>0.3: mca=False
            tr[phase].append({"correct":correct,"mca":mca}); idx+=1
    def rate(ph,key="correct"):
        t=tr.get(ph,[]); return sum(1 for x in t if x[key])/max(len(t),1) if t else None
    fin = fw.finalize_session(m)
    return {"B":rate("B"),"C":rate("C"),"D":rate("D"),"E":rate("E"),
            "n_teach":nt,"n_eval":ne,"stopped":sa is not None,
            "nu":round(m.nu,3),"gg":round(m.gamma_gen,3),
            "otr":fin["otr_decomp"]["otr_total"],
            "otr_teach":fin["otr_decomp"]["otr_teach"],
            "otr_eval":fin["otr_decomp"]["otr_eval_overhead"],
            "fam_rep":fin["otr_decomp"]["family_repeats"],
            "n_pw":fin["posterior"].get("n_pw_gain",0),
            "audit":fin["audit"],
            "cal":fin["calibration"]}

def avg(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),3) if vs else None
def avg_int(rs,k):
    vs=[r[k] for r in rs if r.get(k) is not None]; return round(np.mean(vs),1) if vs else None

def make_fw(th, bud=4.0, budget_mode="theta"):
    return PedagogicalFramework(FrameworkConfig(theta=th, total_budget=bud, risk_budget_mode=budget_mode))

def run_multi(th, seed, ns=4, bud=4.0, ood_mode="none", held_out_family=None):
    fw=make_fw(th,bud); rs=[]
    for s in range(ns): rs.append(run_one(fw,th,seed*10+s,bud=bud,ood_mode=ood_mode,held_out_family=held_out_family))
    return rs[-1]

def main():
    print("═══ Stage-6: Credibility Closure ═══\n",file=sys.stderr)
    L=["# Stage-6: Credibility Closure\n\n"]

    # Exp A: Adversarial OOD
    print("Exp A: Adversarial OOD...",file=sys.stderr)
    L.append("## Exp A: Adversarial OOD\n\n")
    L.append("| θ | OOD Mode | **C** | **E** | OTR | ν | #PW |\n|---|----------|---|---|---|---|---|\n")
    for th in ["safe","shiny"]:
        for ood in ["none","sign_flip","noise_heavy","scale_shift"]:
            rs=[run_multi(th,sid,ood_mode=ood) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","otr","nu"]}; a["npw"]=avg_int(rs,"n_pw")
            L.append("| {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                th,ood,sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}"),sf(a["nu"],"{:.3f}"),sf(a["npw"],"{:.0f}")))
            print(f"  {th}×{ood}: C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')}",file=sys.stderr)

    # Exp A2: Held-out family
    print("\nExp A2: Held-out family...",file=sys.stderr)
    L.append("\n### Held-Out Family\n\n| θ | Held-Out | **C** | **E** | OTR |\n|---|----------|---|---|---|\n")
    for th in ["safe","shiny"]:
        for fam in [None, "PP-MRB", "TIC", "TIC-v4"]:
            lab = fam if fam else "none"
            rs=[run_multi(th,sid,held_out_family=fam) for sid in range(8)]
            a={k:avg(rs,k) for k in ["C","E","otr"]}
            L.append("| {} | {} | **{}** | **{}** | {} |\n".format(
                th,lab,sf(a["C"]),sf(a["E"]),sf(a["otr"],"{:.3f}")))
            print(f"  {th}×{lab}: C={sf(a['C'])} E={sf(a['E'])}",file=sys.stderr)

    # Exp B: STOP calibration
    print("\nExp B: STOP calibration audit...",file=sys.stderr)
    L.append("\n## Exp B: STOP Calibration Audit\n\n| θ | Avg Margin | Monotonicity | #Stops |\n|---|-----------|-------------|-------|\n")
    for th in ["safe","shiny"]:
        rs=[run_multi(th,sid) for sid in range(8)]
        margins=[]; monos=[]; nstops=[]
        for r in rs:
            c=r.get("cal",{})
            m=c.get("avg_stop_margin"); margins.append(m if m else 0)
            mo=c.get("stop_monotonicity"); monos.append(mo if mo else 0)
            nstops.append(c.get("n_stop_recorded",0))
        L.append("| {} | {} | {} | {} |\n".format(
            th,sf4(np.mean(margins)),sf4(np.mean(monos)),f"{np.mean(nstops):.1f}"))
        print(f"  {th}: margin={np.mean(margins):.4f} mono={np.mean(monos):.4f}",file=sys.stderr)

    # Exp C: EVAL calibration
    print("\nExp C: EVAL calibration audit...",file=sys.stderr)
    L.append("\n## Exp C: EVAL Calibration Audit\n\n| θ | Rank-Change Rate | #Evals | Trigger |\n|---|-----------------|--------|--------|\n")
    for th in ["safe","shiny"]:
        rs=[run_multi(th,sid) for sid in range(8)]
        rcrs=[]; nevals=[]; triggers=[]
        for r in rs:
            c=r.get("cal",{})
            rcrs.append(c.get("eval_rank_change_rate",0))
            nevals.append(c.get("n_eval_recorded",0))
            tc=c.get("eval_trigger_counts",{})
            triggers.append(tc)
        # Aggregate triggers
        all_trig = {}
        for tc in triggers:
            for k,v in tc.items(): all_trig[k] = all_trig.get(k,0)+v
        trig_str = ", ".join(f"{k}:{v}" for k,v in all_trig.items()) if all_trig else "—"
        L.append("| {} | {} | {} | {} |\n".format(
            th,sf(np.mean(rcrs),"{:.1%}"),f"{np.mean(nevals):.1f}",trig_str))
        print(f"  {th}: rank_change={np.mean(rcrs):.1%} #eval={np.mean(nevals):.1f} triggers={trig_str}",file=sys.stderr)

    # Exp D: OTR decomposition
    print("\nExp D: OTR decomposition...",file=sys.stderr)
    L.append("\n## Exp D: OTR Decomposition\n\n| θ | OTR Total | OTR Teach | OTR Eval | Fam Repeats |\n|---|----------|----------|---------|------------|\n")
    for th in ["safe","shiny"]:
        rs=[run_multi(th,sid) for sid in range(8)]
        a={k:avg(rs,k) for k in ["otr","otr_teach","otr_eval"]}; a["fr"]=avg_int(rs,"fam_rep")
        L.append("| {} | {} | {} | {} | {} |\n".format(
            th,sf(a["otr"],"{:.3f}"),sf(a["otr_teach"],"{:.3f}"),sf(a["otr_eval"],"{:.3f}"),sf(a["fr"],"{:.0f}")))
        print(f"  {th}: OTR={sf(a['otr'],'{:.3f}')} teach={sf(a['otr_teach'],'{:.3f}')} eval={sf(a['otr_eval'],'{:.3f}')} fam_rep={sf(a['fr'],'{:.0f}')}",file=sys.stderr)

    # Exp E: Actionability regression
    print("\nExp E: Actionability regression...",file=sys.stderr)
    L.append("\n## Exp E: Actionability Regression (θ-adaptive)\n\n| θ | Term | AM | PCR |\n|---|------|----|-----|\n")
    for th in ["safe","shiny"]:
        rs=[run_multi(th,sid) for sid in range(8)]
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

    with open(out/"stage6_credibility_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/stage6_credibility_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
