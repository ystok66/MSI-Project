"""PP-MRB v2.1 Robustness Suite.

Go/no-go gate for Stage 2.

Exp A: Mirror swap (left=safe vs right=safe)
Exp B: Cue-noise sweep (low / med / high)
Exp C: Δ sweep (systematic d_commit - d_reveal)
Exp D: Learner-type sweep (safe / shiny / shortcut / risky)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from copy import deepcopy

from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario,
    SessionSpec, EpisodeSpec, PREF_TYPES_PP, EPISODE_SUBTYPES, SUBTYPE_PARAMS,
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
from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2, PrefV2Config

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
OBS_RADIUS = 2
N_SEEDS = 16

def sf(v, fmt="{:.0%}"): return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww

def run_session_core(session, persistent=True, noise_scale=0.0):
    """Run v2.1 persistent (or reset) under optional feature noise."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary(); scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = PreferenceAwarePolicyV2(agent_params=AP)
    theta_true = session.theta_true; traces = []
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_episode_scenario(ep, theta_true)
        fb, ww = apply_fix(meta, sc)
        # Add cue noise
        if noise_scale > 0:
            rng_n = np.random.default_rng(ep.cue_layout_seed + 7777)
            fb = fb + rng_n.normal(0, noise_scale, fb.shape)
            fb = np.clip(fb, 0.01, 0.99)
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
        ba_safe=BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b, risk_penalty=0.1)
        ba_risky=BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a, risk_penalty=0.4)
        agent_choice=sample_branch_choice([ba_safe,ba_risky],theta_true,AP,rng)
        agent_safe=(agent_choice==0)
        if not persistent: tutor.pref_posterior = PreferencePosteriorV2()
        action,diag = tutor.decide(sc,fb,lp,lib,scorer,OBS_RADIUS)
        do_warn = (action == "WARN")
        tutor.observe_agent_choice(agent_choice,[ba_safe,ba_risky])
        if do_warn:
            for r,c in sc.risky_cells:
                z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z),weight=1.0)
        traces.append({"ep_idx":ep.episode_idx,"subtype":ep.episode_subtype,
                       "delta":ep.d_commit-ep.d_reveal,"warned":do_warn,"agent_safe":agent_safe,
                       "mirror":ep.mirror_side, "c_t": diag.get("confidence_c_t",0)})
    return traces

def compute_metrics(all_traces):
    """Compute SelGap, WR(wait_fav), WR(warn_nec) from list-of-sessions."""
    wf_all=[]; wn_all=[]; sg_all=[]
    for tr in all_traces:
        wf_eps=[t for t in tr if t["subtype"] in ("wait_clean","wait_lure")]
        wn_eps=[t for t in tr if t["subtype"]=="warn_trap"]
        wf=sum(1 for t in wf_eps if t["warned"])/max(len(wf_eps),1) if wf_eps else None
        wn=sum(1 for t in wn_eps if t["warned"])/max(len(wn_eps),1) if wn_eps else None
        if wf is not None: wf_all.append(wf)
        if wn is not None: wn_all.append(wn)
        if wf is not None and wn is not None: sg_all.append(wn-wf)
    return (round(np.mean(wf_all),3) if wf_all else None,
            round(np.mean(wn_all),3) if wn_all else None,
            round(np.mean(sg_all),3) if sg_all else None)

def main():
    print("PP-MRB v2.1 Robustness Suite", file=sys.stderr)
    L = ["# PP-MRB v2.1 Robustness Suite\n\n"]

    # ═══ Exp A: Mirror Swap ═══
    print("\nExp A: Mirror swap...", file=sys.stderr)
    L.append("## Exp A: Mirror Swap\n\n")
    L.append("| θ | Mirror | WR(wait_fav) | WR(warn_nec) | SelGap |\n")
    L.append("|---|--------|:------------:|:------------:|:------:|\n")
    for th in ["shiny", "safe"]:
        for forced_mirror in [0, 1, "mixed"]:
            trs = []
            for sid in range(N_SEEDS):
                ses = generate_session(sid*1000+abs(hash(th))%1000, 30, th,
                    {"wait_clean":0.2,"wait_lure":0.2,"warn_trap":0.4,"boundary_obs":0.2})
                if forced_mirror != "mixed":
                    for ep in ses.episodes: ep.mirror_side = forced_mirror
                trs.append(run_session_core(ses, persistent=True))
            wf,wn,sg = compute_metrics(trs)
            mlbl = "left=safe" if forced_mirror==0 else ("right=safe" if forced_mirror==1 else "mixed")
            L.append(f"| {th} | {mlbl} | {sf(wf)} | {sf(wn)} | {sf(sg,'{:.3f}')} |\n")
            print(f"  {th}×{mlbl}: WF={sf(wf)} WN={sf(wn)} SG={sf(sg,'{:.3f}')}", file=sys.stderr)

    # ═══ Exp B: Cue-Noise Sweep ═══
    print("\nExp B: Cue-noise sweep...", file=sys.stderr)
    L.append("\n## Exp B: Cue-Noise Sweep\n\n")
    L.append("| θ | Noise | WR(wait_fav) | WR(warn_nec) | SelGap |\n")
    L.append("|---|-------|:------------:|:------------:|:------:|\n")
    for th in ["shiny","safe"]:
        for noise in [0.0, 0.05, 0.10, 0.20]:
            trs = []
            for sid in range(N_SEEDS):
                ses = generate_session(sid*1000+abs(hash(th))%1000, 30, th,
                    {"wait_clean":0.2,"wait_lure":0.2,"warn_trap":0.4,"boundary_obs":0.2})
                trs.append(run_session_core(ses, persistent=True, noise_scale=noise))
            wf,wn,sg = compute_metrics(trs)
            nlbl = f"σ={noise:.2f}"
            L.append(f"| {th} | {nlbl} | {sf(wf)} | {sf(wn)} | {sf(sg,'{:.3f}')} |\n")
            print(f"  {th}×{nlbl}: WF={sf(wf)} WN={sf(wn)} SG={sf(sg,'{:.3f}')}", file=sys.stderr)

    # ═══ Exp C: Δ Sweep ═══
    print("\nExp C: Δ sweep...", file=sys.stderr)
    L.append("\n## Exp C: Δ = d_commit − d_reveal Sweep\n\n")
    L.append("| θ | Δ | d_commit | d_reveal | WarnRate | c_t(final) |\n")
    L.append("|---|---|----------|----------|---------|:-----------:|\n")
    for th in ["shiny","safe"]:
        for dc, dr in [(1,5),(2,4),(3,3),(4,2),(5,1),(6,1)]:
            delta = dc - dr
            trs = []
            for sid in range(N_SEEDS):
                rng = np.random.default_rng(sid*1000+abs(hash(th))%1000)
                episodes = []
                for ei in range(30):
                    episodes.append(EpisodeSpec(
                        episode_idx=ei, goal_type="goal_direct",
                        episode_subtype="wait_clean" if delta>0 else ("warn_trap" if delta<0 else "boundary_obs"),
                        mirror_side=int(rng.integers(0,2)),
                        d_commit=dc, d_reveal=dr,
                        lure_strength=0.5, risk_gap=0.2,
                        cue_layout_seed=int(rng.integers(0,100000)), branch_len=10))
                ses = SessionSpec(session_id=sid, theta_true=th, episodes=episodes)
                trs.append(run_session_core(ses, persistent=True))
            wr_all = []
            ct_final = []
            for tr in trs:
                wr_all.append(sum(1 for t in tr if t["warned"])/len(tr))
                ct_final.append(tr[-1]["c_t"] if tr else 0)
            a_wr = round(np.mean(wr_all),3)
            a_ct = round(np.mean(ct_final),3)
            L.append(f"| {th} | {delta:+d} | {dc} | {dr} | {sf(a_wr)} | {a_ct:.3f} |\n")
            print(f"  {th}×Δ={delta:+d}: WR={sf(a_wr)} c_t={a_ct:.3f}", file=sys.stderr)

    # ═══ Exp D: Learner-Type Sweep ═══
    print("\nExp D: Learner-type sweep...", file=sys.stderr)
    L.append("\n## Exp D: Learner-Type Sweep\n\n")
    L.append("| θ | Cond | WR(wait_fav) | WR(warn_nec) | SelGap | SBCR |\n")
    L.append("|---|------|:------------:|:------------:|:------:|:----:|\n")
    for th in ["safe","shiny","shortcut","risky","neutral"]:
        for cond in ["persistent","reset"]:
            trs = []
            for sid in range(N_SEEDS):
                ses = generate_session(sid*1000+abs(hash(th))%1000, 30, th,
                    {"wait_clean":0.2,"wait_lure":0.2,"warn_trap":0.4,"boundary_obs":0.2})
                trs.append(run_session_core(ses, persistent=(cond=="persistent")))
            wf,wn,sg = compute_metrics(trs)
            sbcr = round(np.mean([sum(1 for t in tr if t["agent_safe"])/len(tr) for tr in trs]),3)
            L.append(f"| {th} | {cond} | {sf(wf)} | {sf(wn)} | {sf(sg,'{:.3f}')} | {sf(sbcr)} |\n")
            print(f"  {th}×{cond}: WF={sf(wf)} WN={sf(wn)} SG={sf(sg,'{:.3f}')} SBCR={sf(sbcr)}", file=sys.stderr)

    # ═══ Summary ═══
    L.append("\n## Go/No-Go Assessment\n\n")
    L.append("_To be filled after reviewing results._\n")

    with open(out/"pp_mrb_v2_robustness.md", "w") as f: f.writelines(L)
    print(f"\nReport -> results/pp_mrb_v2_robustness.md", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__ == "__main__": main()
