"""Joint Tutor v2 Experiment + Old-Family Regression.

Phase 1: Regression on delayed_corridor, distractor_cue, elcb_po, temptation_corridor
  - Verify v1.1 and joint_v2 don't break selectivity law
Phase 2: Joint conflict corridor experiment
  - 8 conditions comparing v4, v1.1, joint_v1, joint_v2 (coupled vs factorized)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.scenario_families import generate_scenario, SCENARIO_REGISTRY
from src.envs.persistent_profile_mixed_reveal import (
    generate_session, generate_episode_scenario,
    PREF_TYPES_PP, EPISODE_SUBTYPES,
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


def sf(val, fmt="{:.0%}"):
    return "—" if val is None else fmt.format(val)


# ═══════════════════════════════════════════
# Phase 1: Old-family regression
# ═══════════════════════════════════════════

def run_regression_family(family_name, tutor_name, n_seeds=50):
    """Run a single family × tutor combination."""
    safe_choices = 0
    warnings = 0
    total = 0

    for seed in range(n_seeds):
        try:
            gm, cfg, meta, sc = generate_scenario(family_name, seed=seed)
        except Exception:
            continue

        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)

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

        if tutor_name == "always_wait":
            do_warn = False
        elif tutor_name == "always_warn":
            do_warn = True
        elif tutor_name == "v4":
            t = LearningAwarePolicyV4()
            action, _ = t.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        elif tutor_name == "v1_1":
            t = PersistentTutorV1_1(agent_params=AP)
            action, _ = t.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        elif tutor_name == "joint_v1":
            t = JointLatentTutorV1(agent_params=AP)
            action, _ = t.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        elif tutor_name == "joint_v2":
            t = JointTutorV2(agent_params=AP)
            action, _ = t.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        else:
            do_warn = False

        if do_warn:
            for r, c in sc.risky_cells:
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
            ss2 = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr2 = summarize_branch(sc.risky_cells, fb, fv, lp)
            lib.update("safe_branch", ss2)
            lib.update("risky_branch", sr2)

        # Check agent choice (deterministic for regression)
        rng = np.random.default_rng(seed + 7777)
        ba_safe = BranchAttributes(
            safety_score=float(ss[0]),
            temptation_score=getattr(sc, 'tempt_score_a', 0.1),
            risk_penalty=0.1)
        ba_risky = BranchAttributes(
            safety_score=float(sr[0]),
            temptation_score=getattr(sc, 'tempt_score_b', 0.3),
            risk_penalty=0.4)
        branches = [ba_safe, ba_risky]
        choice = sample_branch_choice(branches, "neutral", AP, rng)
        safe_choices += (1 if choice == 0 else 0)
        warnings += (1 if do_warn else 0)
        total += 1

    if total == 0:
        return None
    return {
        "sbcr": round(safe_choices / total, 3),
        "warn_rate": round(warnings / total, 3),
        "n": total,
    }


# ═══════════════════════════════════════════
# Phase 2: Joint conflict corridor
# ═══════════════════════════════════════════

def run_joint_conflict(strategy, n_sessions=8, n_episodes=12):
    """Run joint_conflict_corridor sessions."""
    thetas = ["safe", "shiny"]
    goals = ["goal_safe_long", "goal_collect"]
    all_traces = []

    for theta in thetas:
        for sid in range(n_sessions):
            session = generate_session(
                session_id=sid * 1000 + abs(hash(theta)) % 1000,
                n_episodes=n_episodes,
                theta_true=theta,
            )

            if strategy == "v4_reset":
                tutor = LearningAwarePolicyV4()
            elif strategy == "v1_1_persistent":
                tutor = PersistentTutorV1_1(agent_params=AP)
            elif strategy == "joint_v1":
                tutor = JointLatentTutorV1(agent_params=AP)
            elif strategy == "joint_v2_coupled":
                tutor = JointTutorV2(agent_params=AP, use_coupled=True)
            elif strategy == "joint_v2_factorized_abl":
                tutor = JointTutorV2(agent_params=AP, use_coupled=True)
                # Use same v2 but with fresh posterior each episode (ablation)
            else:
                tutor = None

            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            lib = BranchConceptLibrary()
            scorer = BranchScorerProbe(lr=0.05, l2=0.01)

            for ep in session.episodes:
                gm, cfg, meta, sc = generate_episode_scenario(ep, theta)
                fb, ww = apply_fix(meta, sc)
                fv = np.full_like(fb, 0.3)
                rng = np.random.default_rng(ep.cue_layout_seed + 9999)

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
                elif strategy == "oracle_joint":
                    do_warn = (ep.d_commit < ep.d_reveal)
                elif strategy == "v4_reset":
                    tutor.reset_stats()
                    action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
                    do_warn = (action == "WARN")
                elif strategy == "joint_v2_factorized_abl":
                    # Reset q(g,θ) each episode = factorized-like
                    tutor.joint_posterior = JointPosteriorV2()
                    action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
                    do_warn = (action == "WARN")
                    tutor.observe_agent_choice(agent_choice, branches)
                else:
                    action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
                    do_warn = (action == "WARN")
                    if hasattr(tutor, 'observe_agent_choice'):
                        tutor.observe_agent_choice(agent_choice, branches)

                all_traces.append({
                    "theta": theta, "strategy": strategy,
                    "subtype": ep.episode_subtype,
                    "delta": ep.d_commit - ep.d_reveal,
                    "warned": do_warn,
                    "agent_safe": (agent_choice == 0),
                    "obs_value": diag.get("obs_value_1step"),
                    "br_current": diag.get("br_current"),
                    "R_conflict": diag.get("R_conflict"),
                    "C_q": diag.get("C_q"),
                    "joint_entropy": diag.get("joint_entropy"),
                })

    # Aggregate
    n = len(all_traces)
    if n == 0:
        return {}
    sbcr = sum(1 for t in all_traces if t["agent_safe"]) / n
    wr = sum(1 for t in all_traces if t["warned"]) / n

    sub_wr = {}
    for st in EPISODE_SUBTYPES:
        eps = [t for t in all_traces if t["subtype"] == st]
        sub_wr[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None

    sel_gap = None
    if sub_wr.get("warn_trap") is not None and sub_wr.get("wait_clean") is not None:
        sel_gap = sub_wr["warn_trap"] - sub_wr["wait_clean"]

    obs_vals = [t["obs_value"] for t in all_traces if t["obs_value"] is not None]
    br_vals = [t["br_current"] for t in all_traces if t["br_current"] is not None]
    conf_vals = [t["R_conflict"] for t in all_traces if t["R_conflict"] is not None]

    return {
        "strategy": strategy,
        "sbcr": round(sbcr, 3), "wr": round(wr, 3),
        "wr_wc": round(sub_wr.get("wait_clean", 0), 3) if sub_wr.get("wait_clean") is not None else None,
        "wr_wt": round(sub_wr.get("warn_trap", 0), 3) if sub_wr.get("warn_trap") is not None else None,
        "sel_gap": round(sel_gap, 3) if sel_gap is not None else None,
        "avg_obs_value": round(np.mean(obs_vals), 4) if obs_vals else None,
        "avg_br": round(np.mean(br_vals), 4) if br_vals else None,
        "avg_conflict": round(np.mean(conf_vals), 4) if conf_vals else None,
    }


def main():
    # ═══════════════════════════════════════
    # Phase 1: Regression
    # ═══════════════════════════════════════
    print("Phase 1: Old-family regression", file=sys.stderr)
    families = ["delayed_corridor", "distractor_cue", "elcb_po", "temptation_corridor"]
    tutors_reg = ["always_wait", "always_warn", "v4", "v1_1", "joint_v2"]

    reg_results = []
    for fam in families:
        if fam not in SCENARIO_REGISTRY:
            print(f"  SKIP {fam} (not in registry)", file=sys.stderr)
            continue
        for tname in tutors_reg:
            r = run_regression_family(fam, tname, n_seeds=30)
            if r is None:
                continue
            r["family"] = fam
            r["tutor"] = tname
            reg_results.append(r)
            print(f"  {fam:22s} × {tname:12s}: SBCR={sf(r['sbcr'])} WR={sf(r['warn_rate'])}",
                  file=sys.stderr)

    # ═══════════════════════════════════════
    # Phase 2: Joint conflict experiments
    # ═══════════════════════════════════════
    print("\nPhase 2: Joint conflict corridor", file=sys.stderr)
    strategies = [
        "always_wait", "always_warn", "v4_reset",
        "v1_1_persistent", "joint_v1", "joint_v2_coupled",
        "joint_v2_factorized_abl", "oracle_joint",
    ]

    joint_results = []
    for strat in strategies:
        r = run_joint_conflict(strat, n_sessions=6, n_episodes=12)
        r["strategy"] = strat
        joint_results.append(r)
        print(f"  {strat:25s}: SBCR={sf(r.get('sbcr'))} WR={sf(r.get('wr'))} "
              f"SelGap={sf(r.get('sel_gap'), '{:.3f}')} "
              f"ObsVal={sf(r.get('avg_obs_value'), '{:.4f}')}",
              file=sys.stderr)

    # ═══════════════════════════════════════
    # Write report
    # ═══════════════════════════════════════
    with open(out / "joint_tutor_v2_report.md", "w") as f:
        f.write("# Joint Tutor v2 + Regression Report\n\n")

        # Phase 1
        f.write("## Phase 1: Old-Family Regression\n\n")
        f.write("Verifying v1.1 and joint_v2 don't break selectivity on established families.\n\n")
        f.write("| Family | Tutor | SBCR | WarnRate |\n")
        f.write("|--------|-------|------|----------|\n")
        for r in reg_results:
            f.write("| {} | {} | {} | {} |\n".format(
                r["family"], r["tutor"], sf(r["sbcr"]), sf(r["warn_rate"])))

        # Regression delta table
        f.write("\n### Regression: v4 vs v1.1 vs joint_v2 WarnRate\n\n")
        f.write("| Family | v4 | v1.1 | joint_v2 | Δ(v1.1-v4) | Δ(jv2-v4) |\n")
        f.write("|--------|-----|------|----------|------------|----------|\n")
        for fam in families:
            vals = {}
            for tname in ["v4", "v1_1", "joint_v2"]:
                row = [r for r in reg_results if r["family"] == fam and r["tutor"] == tname]
                vals[tname] = row[0]["warn_rate"] if row else None
            if vals["v4"] is not None:
                d11 = (vals["v1_1"] - vals["v4"]) if vals["v1_1"] is not None else None
                djv2 = (vals["joint_v2"] - vals["v4"]) if vals["joint_v2"] is not None else None
                f.write("| {} | {} | {} | {} | {} | {} |\n".format(
                    fam,
                    sf(vals["v4"]), sf(vals["v1_1"]), sf(vals["joint_v2"]),
                    sf(d11, "{:+.3f}") if d11 is not None else "—",
                    sf(djv2, "{:+.3f}") if djv2 is not None else "—",
                ))

        # Phase 2
        f.write("\n## Phase 2: Joint Conflict Corridor\n\n")
        f.write("| Strategy | SBCR | WarnRate | WR(wc) | WR(wt) | **SelGap** | AvgObsVal | AvgBR | AvgConflict |\n")
        f.write("|----------|------|---------|--------|--------|-----------|-----------|-------|-------------|\n")
        for r in joint_results:
            f.write("| {} | {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                r["strategy"],
                sf(r.get("sbcr")), sf(r.get("wr")),
                sf(r.get("wr_wc")), sf(r.get("wr_wt")),
                sf(r.get("sel_gap"), "{:.3f}"),
                sf(r.get("avg_obs_value"), "{:.4f}"),
                sf(r.get("avg_br"), "{:.4f}"),
                sf(r.get("avg_conflict"), "{:.4f}"),
            ))

        # Key comparison
        f.write("\n### Key: SelGap Comparison\n\n")
        f.write("| v4_reset | v1.1 | joint_v1 | **joint_v2** | fact_abl | oracle |\n")
        f.write("|----------|------|----------|-------------|----------|--------|\n")
        vals = {}
        for s in strategies:
            row = [r for r in joint_results if r["strategy"] == s]
            vals[s] = row[0].get("sel_gap") if row else None
        f.write("| {} | {} | {} | **{}** | {} | {} |\n".format(
            sf(vals.get("v4_reset"), "{:.3f}"),
            sf(vals.get("v1_1_persistent"), "{:.3f}"),
            sf(vals.get("joint_v1"), "{:.3f}"),
            sf(vals.get("joint_v2_coupled"), "{:.3f}"),
            sf(vals.get("joint_v2_factorized_abl"), "{:.3f}"),
            sf(vals.get("oracle_joint"), "{:.3f}"),
        ))

    print("\nReport -> results/joint_tutor_v2_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
