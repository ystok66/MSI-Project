"""CGC Experiment: Compositional-Goal Corridor.

8 strategies on train compositions + held-out composition generalization.

Key metrics:
  - SBCR, WarnRate, SelGap
  - GoalFactorAcc: per-factor hit rate
  - ConflictResolutionRate: correct branch in conflict episodes
  - HeldOutAcc: performance on unseen compositions
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.compositional_goal_corridor import (
    generate_cgc_session, generate_cgc_scenario,
    CGCSessionSpec, CGCEpisodeSpec,
    COMP_GOALS, TRAIN_GOALS, HELDOUT_GOALS, CGC_SUBTYPES,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal,
    neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
    PREFERENCE_TYPES,
)
from src.agents.joint_posterior_v2 import JointPosteriorV2
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.joint_latent_tutor_v1 import JointLatentTutorV1
from src.teachers.joint_tutor_v2 import JointTutorV2

out = Path("results")
out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
OBS = 2


def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)


def run_cgc_session(session, strategy):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    theta = session.theta_true

    # Create tutor
    if strategy == "v4_reset":
        tutor = LearningAwarePolicyV4()
    elif strategy == "v1_1_persistent":
        tutor = PersistentTutorV1_1(agent_params=AP)
    elif strategy == "joint_v1":
        tutor = JointLatentTutorV1(agent_params=AP)
    elif strategy == "joint_v2_coupled":
        tutor = JointTutorV2(agent_params=AP)
    elif strategy == "joint_v2_fact_abl":
        tutor = JointTutorV2(agent_params=AP)
    else:
        tutor = None

    traces = []
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_cgc_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_seed + 9999)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss)
        lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        ba_safe = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        ba_risky = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=0.4)
        branches = [ba_safe, ba_risky]

        agent_choice = sample_branch_choice(branches, theta, AP, rng)

        # Tutor decision
        do_warn = False
        diag = {}
        if strategy == "always_warn":
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "oracle":
            do_warn = (ep.d_commit < ep.d_reveal_primary)
        elif strategy == "v4_reset":
            tutor.reset_stats()
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        elif strategy == "joint_v2_fact_abl":
            tutor.joint_posterior = JointPosteriorV2()
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
            tutor.observe_agent_choice(agent_choice, branches)
        else:
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
            if hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(agent_choice, branches)

        if do_warn:
            for r, c in sc.risky_cells:
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)

        traces.append({
            "goal_name": ep.goal_name,
            "subtype": ep.episode_subtype,
            "delta": ep.d_commit - ep.d_reveal_primary,
            "warned": do_warn,
            "agent_safe": (agent_choice == 0),
            "obs_value": diag.get("obs_value_1step"),
            "R_conflict": diag.get("R_conflict"),
            "C_q": diag.get("C_q"),
        })

    # Aggregate
    n = len(traces)
    if n == 0:
        return {}
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n

    sub_wr = {}
    for st in CGC_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sub_wr[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None

    # Conflict resolution rate
    conflict_eps = [t for t in traces if t["subtype"] == "compositional_conflict"]
    conflict_res = (sum(1 for t in conflict_eps if t["agent_safe"]) / len(conflict_eps)
                    if conflict_eps else None)

    # Per-subtype SBCR
    sub_sbcr = {}
    for st in CGC_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sub_sbcr[st] = sum(1 for t in eps if t["agent_safe"]) / len(eps) if eps else None

    sel_gap = None
    wc = sub_wr.get("compositional_aligned")
    wt = sub_wr.get("compositional_conflict")
    if wc is not None and wt is not None:
        sel_gap = wt - wc

    obs_vals = [t["obs_value"] for t in traces if t["obs_value"] is not None]
    conf_vals = [t["R_conflict"] for t in traces if t["R_conflict"] is not None]

    return {
        "strategy": strategy, "sbcr": round(sbcr, 3), "wr": round(wr, 3),
        "wr_aligned": round(sub_wr.get("compositional_aligned", 0), 3) if sub_wr.get("compositional_aligned") is not None else None,
        "wr_conflict": round(sub_wr.get("compositional_conflict", 0), 3) if sub_wr.get("compositional_conflict") is not None else None,
        "wr_boundary": round(sub_wr.get("compositional_boundary_obs", 0), 3) if sub_wr.get("compositional_boundary_obs") is not None else None,
        "wr_decoy": round(sub_wr.get("compositional_decoy", 0), 3) if sub_wr.get("compositional_decoy") is not None else None,
        "sel_gap": round(sel_gap, 3) if sel_gap is not None else None,
        "conflict_res": round(conflict_res, 3) if conflict_res is not None else None,
        "avg_obs": round(np.mean(obs_vals), 4) if obs_vals else None,
        "avg_conflict": round(np.mean(conf_vals), 4) if conf_vals else None,
    }


def main():
    print("CGC Experiment: Compositional-Goal Corridor", file=sys.stderr)
    strategies = [
        "always_wait", "always_warn", "v4_reset",
        "v1_1_persistent", "joint_v1",
        "joint_v2_coupled", "joint_v2_fact_abl", "oracle",
    ]
    thetas = ["safe", "shiny"]
    n_sessions = 6

    # ── Train compositions ──
    print("\n═══ Train Compositions ═══", file=sys.stderr)
    train_results = []
    for theta in thetas:
        for strat in strategies:
            session_rs = []
            for sid in range(n_sessions):
                sess = generate_cgc_session(
                    session_id=sid * 1000 + abs(hash(theta)) % 1000,
                    n_episodes=12, theta_true=theta, use_heldout=False)
                r = run_cgc_session(sess, strat)
                session_rs.append(r)

            def ak(key):
                vs = [r[key] for r in session_rs if r.get(key) is not None]
                return round(np.mean(vs), 3) if vs else None

            avg = {
                "theta": theta, "strategy": strat,
                "sbcr": ak("sbcr"), "wr": ak("wr"),
                "wr_aligned": ak("wr_aligned"), "wr_conflict": ak("wr_conflict"),
                "wr_boundary": ak("wr_boundary"), "wr_decoy": ak("wr_decoy"),
                "sel_gap": ak("sel_gap"), "conflict_res": ak("conflict_res"),
                "avg_obs": ak("avg_obs"), "avg_conflict": ak("avg_conflict"),
            }
            train_results.append(avg)
            print(f"  {theta:5s} × {strat:20s}: SBCR={sf(avg['sbcr'])} WR={sf(avg['wr'])} "
                  f"SelGap={sf(avg['sel_gap'], '{:.3f}')} ConflRes={sf(avg['conflict_res'])}",
                  file=sys.stderr)

    # ── Held-out compositions ──
    print("\n═══ Held-Out Compositions ═══", file=sys.stderr)
    heldout_results = []
    for theta in thetas:
        for strat in ["joint_v2_coupled", "v1_1_persistent", "oracle"]:
            session_rs = []
            for sid in range(n_sessions):
                sess = generate_cgc_session(
                    session_id=sid * 1000 + abs(hash(theta)) % 1000 + 500,
                    n_episodes=12, theta_true=theta, use_heldout=True)
                r = run_cgc_session(sess, strat)
                session_rs.append(r)

            def ak2(key):
                vs = [r[key] for r in session_rs if r.get(key) is not None]
                return round(np.mean(vs), 3) if vs else None

            avg = {"theta": theta, "strat": strat,
                   "sbcr": ak2("sbcr"), "wr": ak2("wr"),
                   "sel_gap": ak2("sel_gap"), "conflict_res": ak2("conflict_res")}
            heldout_results.append(avg)
            print(f"  HELDOUT {theta:5s} × {strat:20s}: SBCR={sf(avg['sbcr'])} "
                  f"SelGap={sf(avg['sel_gap'], '{:.3f}')}",
                  file=sys.stderr)

    # ── Write report ──
    with open(out / "cgc_report.md", "w") as f:
        f.write("# CGC: Compositional-Goal Corridor\n\n")
        f.write(f"**Config**: {n_sessions} sessions × 12 episodes × {len(thetas)}θ × {len(strategies)} strategies\n\n")

        f.write("## Train Compositions\n\n")
        f.write("| θ | Strategy | SBCR | WR | WR(aln) | WR(cnf) | WR(bnd) | WR(dcy) | **SelGap** | ConflRes |\n")
        f.write("|---|----------|------|----|---------|---------|---------|---------|-----------|----------|\n")
        for r in train_results:
            f.write("| {} | {} | {} | {} | {} | {} | {} | {} | **{}** | {} |\n".format(
                r["theta"], r["strategy"],
                sf(r["sbcr"]), sf(r["wr"]),
                sf(r["wr_aligned"]), sf(r["wr_conflict"]),
                sf(r["wr_boundary"]), sf(r["wr_decoy"]),
                sf(r["sel_gap"], "{:.3f}"),
                sf(r["conflict_res"]),
            ))

        f.write("\n## SelGap Comparison (train compositions)\n\n")
        f.write("| θ | v4 | v1.1 | joint_v1 | **joint_v2** | fact_abl | oracle |\n")
        f.write("|---|-----|------|----------|-------------|----------|--------|\n")
        for theta in thetas:
            vals = {}
            for s in strategies:
                row = [r for r in train_results if r["theta"] == theta and r["strategy"] == s]
                vals[s] = row[0]["sel_gap"] if row else None
            f.write("| {} | {} | {} | {} | **{}** | {} | {} |\n".format(
                theta,
                sf(vals.get("v4_reset"), "{:.3f}"),
                sf(vals.get("v1_1_persistent"), "{:.3f}"),
                sf(vals.get("joint_v1"), "{:.3f}"),
                sf(vals.get("joint_v2_coupled"), "{:.3f}"),
                sf(vals.get("joint_v2_fact_abl"), "{:.3f}"),
                sf(vals.get("oracle"), "{:.3f}"),
            ))

        f.write("\n## Held-Out Composition Generalization\n\n")
        f.write("| θ | Strategy | SBCR | WR | SelGap | ConflRes |\n")
        f.write("|---|----------|------|----|--------|----------|\n")
        for r in heldout_results:
            f.write("| {} | {} | {} | {} | {} | {} |\n".format(
                r["theta"], r["strat"],
                sf(r["sbcr"]), sf(r["wr"]),
                sf(r["sel_gap"], "{:.3f}"),
                sf(r["conflict_res"]),
            ))

    print("\nReport -> results/cgc_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
