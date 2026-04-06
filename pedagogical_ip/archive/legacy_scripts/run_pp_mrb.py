"""PP-MRB v2: Persistent-Profile Selective Fading Experiment.

Exp A: Ablation (original / +autonomy / +gated_tempt / +both / +both+gated_mw / reset)
Exp B: Time-series WarnRate by episode-index bins per subtype
Exp C: Actionability audit (PCR per new term)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario,
    SessionSpec, EpisodeSpec, PREF_TYPES_PP, EPISODE_SUBTYPES,
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
    BranchAttributes, AgentPolicyParams, sample_branch_choice, PREFERENCE_TYPES,
)
from src.agents.preference_posterior_v2 import PreferencePosteriorV2
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2, PrefV2Config

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
OBS_RADIUS = 2

N_EPISODES = 30
N_SEEDS = 20
THETAS = ["shiny", "safe"]
SUBTYPE_MIX = {"wait_clean": 0.2, "wait_lure": 0.2, "warn_trap": 0.4, "boundary_obs": 0.2}

def apply_fix(meta, sc):
    rng = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww

def make_tutor(strategy, wrong_theta=None):
    if strategy == "v4_reset": return LearningAwarePolicyV4(), "v4"
    if strategy in ("persistent_v2.1", "reset_v2.1"): return PreferenceAwarePolicyV2(agent_params=AP), "pref"
    if strategy == "persistent_original":
        cfg = PrefV2Config(lambda_a=0.3, lambda_m=1.5, lambda_t=2.0, kappa_m=0.0, eps_nec=0.0, rho_unc=1.0)
        return PreferenceAwarePolicyV2(config=cfg, agent_params=AP), "pref"
    if strategy == "autonomy_only":
        cfg = PrefV2Config(lambda_a=2.5, lambda_m=1.5, lambda_t=2.0, kappa_m=0.0, eps_nec=0.0, rho_unc=1.0)
        return PreferenceAwarePolicyV2(config=cfg, agent_params=AP), "pref"
    if strategy == "gated_tempt_only":
        cfg = PrefV2Config(lambda_a=0.3, lambda_m=1.5, lambda_t=1.5, kappa_m=0.0, eps_nec=0.15, rho_unc=0.05)
        return PreferenceAwarePolicyV2(config=cfg, agent_params=AP), "pref"
    if strategy == "autonomy+gated_tempt":
        cfg = PrefV2Config(lambda_a=2.5, lambda_m=1.5, lambda_t=1.5, kappa_m=0.0, eps_nec=0.15, rho_unc=0.05)
        return PreferenceAwarePolicyV2(config=cfg, agent_params=AP), "pref"
    if strategy == "wrong_memory":
        t = PreferenceAwarePolicyV2(agent_params=AP)
        if wrong_theta:
            idx = PREFERENCE_TYPES.index(wrong_theta)
            t.pref_posterior.log_probs[idx] = 5.0
            t.pref_posterior.log_probs -= np.mean(t.pref_posterior.log_probs)
        return t, "pref"
    return None, None

def run_session(session, strategy, wrong_theta=None):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary(); scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor, ttype = make_tutor(strategy, wrong_theta)
    theta_true = session.theta_true; traces = []
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_episode_scenario(ep, theta_true)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r,c]==CellType.WALL: continue
                    z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z))
        ss=summarize_branch(sc.safe_cells,fb,fv,lp); sr=summarize_branch(sc.risky_cells,fb,fv,lp)
        lib.update("safe_branch",ss); lib.update("risky_branch",sr)
        scorer.update(build_scorer_input(ss,lib),1.0); scorer.update(build_scorer_input(sr,lib),0.0)
        ba_safe=BranchAttributes(safety_score=float(ss[0]),temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b, risk_penalty=0.1)
        ba_risky=BranchAttributes(safety_score=float(sr[0]),temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a, risk_penalty=0.4)
        agent_choice=sample_branch_choice([ba_safe,ba_risky],theta_true,AP,rng)
        agent_safe=(agent_choice==0)
        do_warn=False; diag={}
        if strategy=="always_warn": do_warn=True
        elif strategy=="always_wait": do_warn=False
        elif strategy=="oracle_theta": do_warn=(ep.d_commit<ep.d_reveal)
        elif ttype=="v4":
            if strategy=="v4_reset": tutor.reset_stats()
            action,diag=tutor.decide(sc,fb,lp,lib,scorer,OBS_RADIUS)
            do_warn=(action=="WARN")
        elif ttype=="pref":
            if strategy=="reset_v2.1": tutor.pref_posterior=PreferencePosteriorV2()
            action,diag=tutor.decide(sc,fb,lp,lib,scorer,OBS_RADIUS)
            do_warn=(action=="WARN")
            tutor.observe_agent_choice(agent_choice,[ba_safe,ba_risky])
        if do_warn:
            for r,c in sc.risky_cells:
                z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z),weight=1.0)
        tr={"ep_idx":ep.episode_idx,"subtype":ep.episode_subtype,"delta":ep.d_commit-ep.d_reveal,
            "warned":do_warn,"agent_safe":agent_safe,"d_commit":ep.d_commit,"d_reveal":ep.d_reveal,
            "lure":ep.lure_strength}
        tr.update({k:diag.get(k) for k in ["Q_warn","Q_wait","p_self","confidence_c_t","autonomy_bonus",
                                             "tempt_risk_raw","tempt_risk_gated","missed_window_raw",
                                             "missed_window_gated","pref_entropy","pref_uncertainty"]})
        traces.append(tr)
    return traces

def sf(v,fmt="{:.0%}"): return "—" if v is None else fmt.format(v)

def wr_by_subtype(traces,st):
    ts=[t for t in traces if t["subtype"]==st]
    return sum(1 for t in ts if t["warned"])/max(len(ts),1) if ts else None

def wr_by_bin(traces,st,lo,hi):
    ts=[t for t in traces if t["subtype"]==st and lo<=t["ep_idx"]<hi]
    return sum(1 for t in ts if t["warned"])/max(len(ts),1) if ts else None

def main():
    print("PP-MRB v2: Selective Fading Experiment",file=sys.stderr)
    strats=["always_wait","always_warn","oracle_theta","persistent_original",
            "autonomy_only","gated_tempt_only","autonomy+gated_tempt","persistent_v2.1",
            "reset_v2.1","v4_reset"]
    L=["# PP-MRB v2: Persistent-Profile Selective Fading\n\n"]
    L.append(f"**Config**: {N_SEEDS} seeds × {N_EPISODES} episodes × {len(THETAS)} θ\n\n")

    # Exp A: Main results
    L.append("## Exp A: Main Results\n\n")
    L.append("| θ | Strategy | SBCR | WarnRate | WR(wait_fav) | WR(warn_nec) | SelGap | Ent(1st) | Ent(2nd) |\n")
    L.append("|---|----------|------|---------|:------------:|:------------:|:------:|:--------:|:--------:|\n")
    all_traces={}
    for th in THETAS:
        for st in strats:
            seeds_traces=[]
            for sid in range(N_SEEDS):
                ses=generate_session(sid*1000+abs(hash(th))%1000,N_EPISODES,th,SUBTYPE_MIX)
                wrong="shiny" if th!="shiny" else "safe"
                tr=run_session(ses,st,wrong); seeds_traces.append(tr)
            all_traces[(th,st)]=seeds_traces
            # Averages
            sbcrs=[]; wrs=[]; wf=[]; wn=[]; ent1=[]; ent2=[]
            for tr in seeds_traces:
                n=len(tr); sbcrs.append(sum(1 for t in tr if t["agent_safe"])/n)
                wrs.append(sum(1 for t in tr if t["warned"])/n)
                wf.append(wr_by_subtype(tr,"wait_clean")); wn.append(wr_by_subtype(tr,"warn_trap"))
                ev=[t.get("pref_entropy") for t in tr if t.get("pref_entropy") is not None and t["pref_entropy"]>0]
                if len(ev)>=4: h=len(ev)//2; ent1.append(np.mean(ev[:h])); ent2.append(np.mean(ev[h:]))
            # Also include wait_lure as wait_fav
            wf2=[]
            for tr in seeds_traces:
                wfl=wr_by_subtype(tr,"wait_lure"); wfc=wr_by_subtype(tr,"wait_clean")
                if wfl is not None and wfc is not None: wf2.append((wfl+wfc)/2)
                elif wfc is not None: wf2.append(wfc)
            a_sbcr=round(np.mean(sbcrs),3); a_wr=round(np.mean(wrs),3)
            a_wf=round(np.mean([v for v in wf2 if v is not None]),3) if wf2 else None
            a_wn=round(np.mean([v for v in wn if v is not None]),3) if any(v is not None for v in wn) else None
            sg=round(a_wn-a_wf,3) if a_wn is not None and a_wf is not None else None
            a_e1=round(np.mean(ent1),4) if ent1 else 0; a_e2=round(np.mean(ent2),4) if ent2 else 0
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                th,st,sf(a_sbcr),sf(a_wr),sf(a_wf),sf(a_wn),sf(sg,"{:.3f}"),sf(a_e1,"{:.4f}"),sf(a_e2,"{:.4f}")))
            print(f"  {th}×{st}: WR={sf(a_wr)} WF={sf(a_wf)} WN={sf(a_wn)} SG={sf(sg,'{:.3f}')}",file=sys.stderr)

    # Exp B: Time-series
    print("\nExp B: Time-series...",file=sys.stderr)
    L.append("\n## Exp B: Time-series WarnRate by Episode Bins\n\n")
    bins=[(0,10,"1-10"),(10,20,"11-20"),(20,30,"21-30")]
    focus_strats=["persistent_original","persistent_v2.1","reset_v2.1","oracle_theta"]
    for th in THETAS:
        L.append(f"\n### θ = {th}\n\n")
        for stype in ["wait_clean","wait_lure","warn_trap"]:
            L.append(f"\n**{stype}**\n\n| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |\n|----------|:-------:|:--------:|:--------:|:--------------:|\n")
            for st in focus_strats:
                trs=all_traces.get((th,st),[])
                bvals=[]
                for lo,hi,_ in bins:
                    vs=[wr_by_bin(tr,stype,lo,hi) for tr in trs]
                    vs=[v for v in vs if v is not None]
                    bvals.append(round(np.mean(vs),3) if vs else None)
                delta=round(bvals[0]-bvals[2],3) if bvals[0] is not None and bvals[2] is not None else None
                L.append("| {} | {} | {} | {} | {} |\n".format(
                    st,sf(bvals[0]),sf(bvals[1]),sf(bvals[2]),sf(delta,"{:.3f}")))
                if stype=="wait_clean":
                    print(f"  {th}×{st}×{stype}: {sf(bvals[0])}→{sf(bvals[2])} Δ={sf(delta,'{:.3f}')}",file=sys.stderr)

    # Exp C: PCR audit
    print("\nExp C: PCR audit...",file=sys.stderr)
    L.append("\n## Exp C: Actionability Audit\n\n")
    L.append("| θ | Subtype | Term | PCR |\n|---|---------|------|-----|\n")
    terms=["autonomy_bonus","tempt_risk_gated","missed_window_gated"]
    for th in THETAS:
        trs=all_traces.get((th,"persistent_v2.1"),[])
        for stype in ["wait_clean","warn_trap"]:
            for term in terms:
                flips=0; total=0
                for tr in trs:
                    for t in tr:
                        if t["subtype"]!=stype: continue
                        qw=t.get("Q_warn"); qwt=t.get("Q_wait")
                        tv=t.get(term)
                        if qw is None or qwt is None or tv is None: continue
                        total+=1
                        action_full="WARN" if qw>qwt else "WAIT"
                        if term=="autonomy_bonus":
                            qwt_without=qwt-tv; action_without="WARN" if qw>qwt_without else "WAIT"
                        else:
                            qw_without=qw-tv; action_without="WARN" if qw_without>qwt else "WAIT"
                        if action_full!=action_without: flips+=1
                pcr=round(flips/max(total,1),3)
                L.append(f"| {th} | {stype} | {term} | {pcr:.1%} |\n")
                print(f"  {th}×{stype}×{term}: PCR={pcr:.1%}",file=sys.stderr)

    with open(out/"pp_mrb_v2_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/pp_mrb_v2_report.md",file=sys.stderr); print("Done.",file=sys.stderr)

if __name__=="__main__": main()
