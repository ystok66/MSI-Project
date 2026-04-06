"""Decision-Boundary Repair: Ablation Experiment.

7 conditions:
  1. v4_reset          — timing-only baseline (no posterior)
  2. pref_v2_current   — current pref-aware persistent (broken selectivity)
  3. v1_1_conf_only    — + confidence gate only
  4. v1_1_opp_only     — + opportunity gate only
  5. v1_1_full         — all 4 gates (confidence+susceptibility+opportunity+slack)
  6. oracle_theta      — upper bound
  7. wrong_memory      — control: persistent with wrong θ prior
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

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
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1

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


def make_tutor(strategy, wrong_theta=None):
    """Factory for tutor instances."""
    if strategy == "v4_reset":
        return LearningAwarePolicyV4()
    elif strategy == "pref_v2_current":
        return PreferenceAwarePolicyV2(agent_params=AP)
    elif strategy == "v1_1_conf_only":
        return PersistentTutorV1_1(agent_params=AP,
            enable_confidence=True, enable_susceptibility=False,
            enable_opportunity=False, enable_obs_slack=False)
    elif strategy == "v1_1_opp_only":
        return PersistentTutorV1_1(agent_params=AP,
            enable_confidence=False, enable_susceptibility=False,
            enable_opportunity=True, enable_obs_slack=False)
    elif strategy == "v1_1_full":
        return PersistentTutorV1_1(agent_params=AP)
    elif strategy == "wrong_memory":
        t = PersistentTutorV1_1(agent_params=AP)
        if wrong_theta:
            wi = PREFERENCE_TYPES.index(wrong_theta)
            t.pref_posterior.log_probs[wi] = 5.0
            t.pref_posterior.log_probs -= np.mean(t.pref_posterior.log_probs)
        return t
    return None


def run_session(session, strategy, wrong_theta=None):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy, wrong_theta)
    theta_true = session.theta_true
    traces = []

    for ep in session.episodes:
        gm, cfg, meta, sc = generate_episode_scenario(ep, theta_true)
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

        agent_choice = sample_branch_choice(branches, theta_true, AP, rng)
        agent_safe = (agent_choice == 0)

        # Tutor decision
        do_warn = False
        diag = {}
        if strategy == "always_warn":
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "oracle_theta":
            do_warn = (ep.d_commit < ep.d_reveal)
        elif strategy == "v4_reset":
            tutor.reset_stats()
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
        elif strategy in ("pref_v2_current", "v1_1_conf_only", "v1_1_opp_only",
                          "v1_1_full", "wrong_memory"):
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, OBS)
            do_warn = (action == "WARN")
            tutor.observe_agent_choice(agent_choice, branches)

        if do_warn:
            for r, c in sc.risky_cells:
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)

        # Posterior state
        q_ent = 0.0
        q_top1 = "—"
        if hasattr(tutor, 'pref_posterior'):
            q_ent = tutor.pref_posterior.entropy
            q_top1 = tutor.pref_posterior.predicted_type

        traces.append({
            "ep_idx": ep.episode_idx, "subtype": ep.episode_subtype,
            "delta": ep.d_commit - ep.d_reveal, "lure": ep.lure_strength,
            "warned": do_warn, "agent_safe": agent_safe,
            "q_ent": round(q_ent, 4), "q_top1": q_top1,
            "C_q": diag.get("C_q"), "O_wait": diag.get("O_wait"),
            "R_tempt": diag.get("R_tempt"), "conf_bonus": diag.get("conf_bonus"),
            "tempt_bonus": diag.get("tempt_bonus"),
        })

    # Aggregate
    n = len(traces)
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n

    sub_wr = {}
    for st in EPISODE_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sub_wr[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None

    sel_gap = None
    if sub_wr.get("warn_trap") is not None and sub_wr.get("wait_clean") is not None:
        sel_gap = sub_wr["warn_trap"] - sub_wr["wait_clean"]

    # Entropy by episode half
    ent_vals = [t["q_ent"] for t in traces if t["q_ent"] > 0]
    if len(ent_vals) >= 4:
        h = len(ent_vals) // 2
        ent_1, ent_2 = float(np.mean(ent_vals[:h])), float(np.mean(ent_vals[h:]))
    else:
        ent_1 = ent_2 = 0.0

    # C_q by episode half
    cq_vals = [t["C_q"] for t in traces if t["C_q"] is not None]
    if len(cq_vals) >= 4:
        h = len(cq_vals) // 2
        cq_1, cq_2 = float(np.mean(cq_vals[:h])), float(np.mean(cq_vals[h:]))
    else:
        cq_1 = cq_2 = 0.0

    return {
        "strategy": strategy, "theta": theta_true,
        "sbcr": round(sbcr, 3), "warn_rate": round(wr, 3),
        "wr_wc": round(sub_wr.get("wait_clean", 0), 3) if sub_wr.get("wait_clean") is not None else None,
        "wr_wl": round(sub_wr.get("wait_lure", 0), 3) if sub_wr.get("wait_lure") is not None else None,
        "wr_bo": round(sub_wr.get("boundary_obs", 0), 3) if sub_wr.get("boundary_obs") is not None else None,
        "wr_wt": round(sub_wr.get("warn_trap", 0), 3) if sub_wr.get("warn_trap") is not None else None,
        "sel_gap": round(sel_gap, 3) if sel_gap is not None else None,
        "ent_1": round(ent_1, 4), "ent_2": round(ent_2, 4),
        "cq_1": round(cq_1, 4), "cq_2": round(cq_2, 4),
    }


def sf(val, fmt="{:.0%}"):
    return "—" if val is None else fmt.format(val)


def main():
    print("Decision-Boundary Repair: Ablation", file=sys.stderr)
    strategies = [
        "v4_reset", "pref_v2_current",
        "v1_1_conf_only", "v1_1_opp_only", "v1_1_full",
        "oracle_theta", "wrong_memory",
    ]
    thetas = ["safe", "shiny", "shortcut", "neutral"]
    n_sessions = 8

    all_results = []
    for theta in thetas:
        for strat in strategies:
            session_results = []
            for sid in range(n_sessions):
                session = generate_session(
                    session_id=sid * 1000 + abs(hash(theta)) % 1000,
                    n_episodes=12, theta_true=theta)
                wrong = "shiny" if theta != "shiny" else "safe"
                r = run_session(session, strat, wrong_theta=wrong)
                session_results.append(r)

            def ak(key):
                vs = [r[key] for r in session_results if r[key] is not None]
                return round(np.mean(vs), 3) if vs else None

            avg = {
                "theta": theta, "strategy": strat,
                "sbcr": ak("sbcr"), "wr": ak("warn_rate"),
                "wr_wc": ak("wr_wc"), "wr_wl": ak("wr_wl"),
                "wr_wt": ak("wr_wt"), "sel_gap": ak("sel_gap"),
                "ent_1": ak("ent_1"), "ent_2": ak("ent_2"),
                "cq_1": ak("cq_1"), "cq_2": ak("cq_2"),
            }
            all_results.append(avg)
            print(f"  {theta:8s} × {strat:20s}: SBCR={sf(avg['sbcr'])} "
                  f"WR={sf(avg['wr'])} SelGap={sf(avg['sel_gap'], '{:.3f}')} "
                  f"WR_wc={sf(avg['wr_wc'])} WR_wt={sf(avg['wr_wt'])}",
                  file=sys.stderr)

    with open(out / "decision_boundary_repair.md", "w") as f:
        f.write("# Decision-Boundary Repair: Ablation Results\n\n")
        f.write(f"**Config**: {n_sessions} sessions × 12 episodes × 4θ × 7 strategies\n\n")

        # Main table
        f.write("## Full Results\n\n")
        f.write("| θ | Strategy | SBCR | WarnRate | WR(wc) | WR(wt) | **SelGap** | Ent(1st) | Ent(2nd) | C_q(1st) | C_q(2nd) |\n")
        f.write("|---|----------|------|---------|--------|--------|-----------|----------|----------|----------|----------|\n")
        for r in all_results:
            f.write("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} |\n".format(
                r["theta"], r["strategy"],
                sf(r["sbcr"]), sf(r["wr"]),
                sf(r["wr_wc"]), sf(r["wr_wt"]),
                sf(r["sel_gap"], "{:.3f}"),
                sf(r["ent_1"], "{:.4f}"), sf(r["ent_2"], "{:.4f}"),
                sf(r["cq_1"], "{:.4f}"), sf(r["cq_2"], "{:.4f}"),
            ))

        # Focused comparison
        f.write("\n## Key Comparison: SelGap\n\n")
        f.write("| θ | v4_reset | current | conf_only | opp_only | **v1.1_full** | oracle | wrong_mem |\n")
        f.write("|---|---------|---------|-----------|----------|-------------|--------|----------|\n")
        for theta in thetas:
            vals = {}
            for s in strategies:
                row = [r for r in all_results if r["theta"] == theta and r["strategy"] == s]
                vals[s] = row[0]["sel_gap"] if row else None
            f.write("| {} | {} | {} | {} | {} | **{}** | {} | {} |\n".format(
                theta,
                sf(vals["v4_reset"], "{:.3f}"),
                sf(vals["pref_v2_current"], "{:.3f}"),
                sf(vals["v1_1_conf_only"], "{:.3f}"),
                sf(vals["v1_1_opp_only"], "{:.3f}"),
                sf(vals["v1_1_full"], "{:.3f}"),
                sf(vals["oracle_theta"], "{:.3f}"),
                sf(vals["wrong_memory"], "{:.3f}"),
            ))

        # WarnRate on wait_clean
        f.write("\n## WarnRate on wait_clean Episodes\n\n")
        f.write("| θ | v4_reset | current | conf_only | opp_only | **v1.1_full** | oracle | wrong_mem |\n")
        f.write("|---|---------|---------|-----------|----------|-------------|--------|----------|\n")
        for theta in thetas:
            vals = {}
            for s in strategies:
                row = [r for r in all_results if r["theta"] == theta and r["strategy"] == s]
                vals[s] = row[0]["wr_wc"] if row else None
            f.write("| {} | {} | {} | {} | {} | **{}** | {} | {} |\n".format(
                theta,
                sf(vals["v4_reset"]), sf(vals["pref_v2_current"]),
                sf(vals["v1_1_conf_only"]), sf(vals["v1_1_opp_only"]),
                sf(vals["v1_1_full"]),
                sf(vals["oracle_theta"]), sf(vals["wrong_memory"]),
            ))

    print("Report -> results/decision_boundary_repair.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
