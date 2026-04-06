"""Stage-2 Experiment: Joint-Latent Tutor on CGC-v2.

Exp A: factorized vs coupled_v1 vs coupled_v2 vs oracle vs always_warn
Exp B: Aligned vs Conflict episode breakdown
Exp C: Time-series c_t^joint convergence
Exp D: PCR audit for V_obs and autonomy_bonus
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.cgc_v2_family import (
    generate_cgc_session, generate_cgc_episode_scenario,
    CGCSessionSpec, CGCEpisodeSpec, CGC_EPISODE_SUBTYPES,
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
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.agents.joint_posterior_v2 import JointPosteriorV2
from src.agents.preference_posterior_v2 import PreferencePosteriorV2
from src.agents.goal_posterior_v1 import GoalPosteriorV1
from src.teachers.joint_latent_tutor_v1 import JointLatentTutorV1
from src.teachers.joint_latent_tutor_v2 import JointLatentTutorV2
from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
OBS_RADIUS = 2
N_SEEDS = 16
N_EPISODES = 30

def sf(v, fmt="{:.0%}"): return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42); ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww

def make_tutor(strategy):
    if strategy == "coupled_v2": return JointLatentTutorV2(agent_params=AP), "joint"
    if strategy == "coupled_v1": return JointLatentTutorV1(agent_params=AP), "joint"
    if strategy == "pref_only": return PreferenceAwarePolicyV2(agent_params=AP), "pref"
    return None, None

def run_cgc_session(session, strategy):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary(); scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor, ttype = make_tutor(strategy)
    theta_true = session.theta_true; traces = []
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_cgc_episode_scenario(ep, theta_true)
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
        ba_safe=BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id==0 else sc.tempt_score_b,
            risk_penalty=0.1)
        ba_risky=BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id==0 else sc.tempt_score_a,
            risk_penalty=0.4)
        agent_choice=sample_branch_choice([ba_safe,ba_risky],theta_true,AP,rng)
        agent_safe=(agent_choice==0)
        do_warn=False; diag={}
        if strategy=="always_warn": do_warn=True
        elif strategy=="always_wait": do_warn=False
        elif strategy=="oracle":
            do_warn = (ep.d_commit < ep.d_reveal)
        elif ttype in ("joint","pref"):
            action,diag=tutor.decide(sc,fb,lp,lib,scorer,OBS_RADIUS)
            do_warn=(action=="WARN")
            tutor.observe_agent_choice(agent_choice,[ba_safe,ba_risky])
        if do_warn:
            for r,c in sc.risky_cells:
                z=fb[r,c]; lp.update_from_outcome(z,ww.true_cost(z),ww.true_risk(z),weight=1.0)
        tr={"ep_idx":ep.episode_idx,"subtype":ep.episode_subtype,"delta":ep.d_commit-ep.d_reveal,
            "warned":do_warn,"agent_safe":agent_safe,"goal":ep.goal_label,
            "is_composite":ep.is_composite, "diag_steps":ep.diagnostic_steps}
        for k in ["Q_warn","Q_wait","c_t_joint","autonomy_bonus","v_obs_gated",
                   "tempt_risk_gated","missed_window_gated","joint_entropy",
                   "predicted_goal","predicted_pref","joint_conf",
                   "confidence_c_t","pref_entropy"]:
            tr[k]=diag.get(k)
        traces.append(tr)
    return traces

def wr_by_subtype(traces, st):
    ts=[t for t in traces if t["subtype"]==st]
    return sum(1 for t in ts if t["warned"])/max(len(ts),1) if ts else None

def main():
    print("Stage-2: Joint Tutor on CGC-v2", file=sys.stderr)
    strats = ["always_wait", "always_warn", "oracle", "pref_only", "coupled_v1", "coupled_v2"]

    # Test conditions: aligned + conflict
    conditions = [
        ("shiny", "collect_red", None,        "shiny-aligned"),
        ("safe",  "collect_red", None,        "safe-conflict"),
        ("shiny", "collect_red", "avoid_blue","shiny-composite"),
        ("safe",  "collect_red", "use_safe",  "safe-aligned-comp"),
    ]

    L = ["# Stage-2: Joint Tutor on CGC-v2\n\n"]
    L.append(f"**Config**: {N_SEEDS} seeds × {N_EPISODES} episodes × {len(conditions)} conditions\n\n")

    # Exp A: Main results
    L.append("## Exp A: Main Results\n\n")
    L.append("| Condition | Strategy | WarnRate | WR(aligned) | WR(conflict) | SelGap | SBCR |\n")
    L.append("|-----------|----------|---------|:-----------:|:------------:|:------:|:----:|\n")
    all_traces = {}
    for th, gobj, gcon, label in conditions:
        for st in strats:
            seed_traces = []
            for sid in range(N_SEEDS):
                ses = generate_cgc_session(sid*1000+abs(hash(label))%1000, N_EPISODES, th, gobj, gcon)
                tr = run_cgc_session(ses, st)
                seed_traces.append(tr)
            all_traces[(label, st)] = seed_traces
            wrs=[sum(1 for t in tr if t["warned"])/len(tr) for tr in seed_traces]
            wa_all=[wr_by_subtype(tr,"goal_aligned") for tr in seed_traces]
            wc_all=[wr_by_subtype(tr,"goal_conflict") for tr in seed_traces]
            sbcrs=[sum(1 for t in tr if t["agent_safe"])/len(tr) for tr in seed_traces]
            a_wr=round(np.mean(wrs),3)
            a_wa=round(np.mean([v for v in wa_all if v is not None]),3) if any(v is not None for v in wa_all) else None
            a_wc=round(np.mean([v for v in wc_all if v is not None]),3) if any(v is not None for v in wc_all) else None
            sg=round(a_wc-a_wa,3) if a_wa is not None and a_wc is not None else None
            a_sbcr=round(np.mean(sbcrs),3)
            L.append(f"| {label} | {st} | {sf(a_wr)} | {sf(a_wa)} | {sf(a_wc)} | {sf(sg,'{:.3f}')} | {sf(a_sbcr)} |\n")
            print(f"  {label}×{st}: WR={sf(a_wr)} WA={sf(a_wa)} WC={sf(a_wc)} SG={sf(sg,'{:.3f}')} SBCR={sf(a_sbcr)}", file=sys.stderr)

    # Exp B: Time-series c_t^joint
    print("\nExp B: Time-series...", file=sys.stderr)
    L.append("\n## Exp B: Time-series Joint Confidence\n\n")
    bins = [(0,10,"1-10"),(10,20,"11-20"),(20,30,"21-30")]
    focus = ["coupled_v1","coupled_v2","pref_only"]
    for label_cond in [c[3] for c in conditions[:2]]:
        L.append(f"\n### {label_cond}\n\n| Strategy | c_t(1-10) | c_t(11-20) | c_t(21-30) | WR(1-10) | WR(21-30) | ΔWR |\n|----------|:---------:|:----------:|:----------:|:--------:|:---------:|:---:|\n")
        for st in focus:
            trs = all_traces.get((label_cond, st), [])
            ct_bins = []; wr_bins = []
            for lo,hi,_ in bins:
                cts=[]; wrs=[]
                for tr in trs:
                    eps_bin=[t for t in tr if lo<=t["ep_idx"]<hi]
                    cv=[t.get("c_t_joint") or t.get("confidence_c_t") or 0 for t in eps_bin]
                    if cv: cts.append(np.mean(cv))
                    if eps_bin: wrs.append(sum(1 for t in eps_bin if t["warned"])/len(eps_bin))
                ct_bins.append(round(np.mean(cts),3) if cts else 0)
                wr_bins.append(round(np.mean(wrs),3) if wrs else 0)
            dwr = round(wr_bins[0]-wr_bins[2],3)
            L.append(f"| {st} | {ct_bins[0]:.3f} | {ct_bins[1]:.3f} | {ct_bins[2]:.3f} | {sf(wr_bins[0])} | {sf(wr_bins[2])} | {sf(dwr,'{:.3f}')} |\n")
            print(f"  {label_cond}×{st}: c_t {ct_bins[0]:.3f}→{ct_bins[2]:.3f} WR {sf(wr_bins[0])}→{sf(wr_bins[2])}", file=sys.stderr)

    # Exp C: PCR audit
    print("\nExp C: PCR audit...", file=sys.stderr)
    L.append("\n## Exp C: Actionability Audit (coupled_v2 only)\n\n")
    L.append("| Condition | Subtype | Term | PCR |\n|-----------|---------|------|-----|\n")
    terms = ["autonomy_bonus","v_obs_gated","tempt_risk_gated","missed_window_gated"]
    for label_cond in [c[3] for c in conditions]:
        trs = all_traces.get((label_cond, "coupled_v2"), [])
        for stype in ["goal_aligned","goal_conflict"]:
            for term in terms:
                flips=0; total=0
                for tr in trs:
                    for t in tr:
                        if t["subtype"]!=stype: continue
                        qw=t.get("Q_warn"); qwt=t.get("Q_wait"); tv=t.get(term)
                        if qw is None or qwt is None or tv is None: continue
                        total+=1
                        action_full="WARN" if qw>qwt else "WAIT"
                        if term in ("autonomy_bonus","v_obs_gated"):
                            action_without="WARN" if qw>(qwt-tv) else "WAIT"
                        else:
                            action_without="WARN" if (qw-tv)>qwt else "WAIT"
                        if action_full!=action_without: flips+=1
                pcr=round(flips/max(total,1),3) if total>0 else 0
                L.append(f"| {label_cond} | {stype} | {term} | {pcr:.1%} |\n")
            print(f"  {label_cond}×{stype}: auto={sf(None)}", file=sys.stderr, end="")
        print("", file=sys.stderr)

    # Exp D: v1 vs v2 differential
    print("\nExp D: v1 vs v2 comparison...", file=sys.stderr)
    L.append("\n## Exp D: Coupled v1 vs v2 Differential\n\n")
    L.append("| Condition | Metric | coupled_v1 | coupled_v2 | Δ |\n|-----------|--------|:----------:|:----------:|:--:|\n")
    for label_cond in [c[3] for c in conditions]:
        for metric_name, metric_fn in [
            ("SelGap", lambda trs: np.mean([
                (wr_by_subtype(tr,"goal_conflict") or 0)-(wr_by_subtype(tr,"goal_aligned") or 0) for tr in trs])),
            ("WR(aligned)", lambda trs: np.mean([wr_by_subtype(tr,"goal_aligned") or 0 for tr in trs])),
            ("WR(conflict)", lambda trs: np.mean([wr_by_subtype(tr,"goal_conflict") or 0 for tr in trs])),
        ]:
            v1 = round(metric_fn(all_traces.get((label_cond, "coupled_v1"), [])),3)
            v2 = round(metric_fn(all_traces.get((label_cond, "coupled_v2"), [])),3)
            d = round(v2-v1, 3)
            L.append(f"| {label_cond} | {metric_name} | {v1:.3f} | {v2:.3f} | {d:+.3f} |\n")

    with open(out/"stage2_joint_report.md","w") as f: f.writelines(L)
    print(f"\nReport -> results/stage2_joint_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__=="__main__": main()
