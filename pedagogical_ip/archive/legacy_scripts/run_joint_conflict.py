"""L1+L2+L3+L4: Joint Conflict + Persistent Profile.

L1: Coupled joint posterior v2 vs factorized — aligned vs conflict
L2: Joint-Latent Conflict Corridor experiments
L3: Nonmyopic joint tutor vs factorized tutors
L4: Persistent agent profile across episodes
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal,
    neutralize_identity_features,
)
from src.envs.observation_mask import make_observation_mask
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
    PREFERENCE_TYPES,
)
from src.agents.joint_posterior_v2 import (
    JointPosteriorV2, compute_joint_likelihood,
    GOAL_TYPES, N_GOALS, N_PREF,
)
from src.agents.joint_latent_belief import JointLatentBelief
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.joint_latent_tutor_v1 import JointLatentTutorV1

out = Path("results")
out.mkdir(exist_ok=True)

DIFF = "medium"
N_BOOT = 200


def apply_fix(gm, meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def vis_candidates(sc, obs_r):
    fk, mg = sc.fork_cell, sc.merge_cell
    ma = make_observation_mask(sc.branch_a_cells, fk, obs_r)
    mb = make_observation_mask(sc.branch_b_cells, fk, obs_r)
    va = [c for c, m in zip(sc.branch_a_cells, ma) if m > 0.5]
    vb = [c for c, m in zip(sc.branch_b_cells, mb) if m > 0.5]
    return [
        BranchCandidate(0, va, len(va), fk, mg, (1, fk[1]), (1, mg[1])),
        BranchCandidate(1, vb, len(vb), fk, mg, (3, fk[1]), (3, mg[1])),
    ]


# ══════════════════════════════════════════════════════════════
# L1: Coupled vs Factorized posterior
# ══════════════════════════════════════════════════════════════
def l1_coupled_vs_factorized():
    print("L1: Coupled vs Factorized", file=sys.stderr)
    params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
    results = []

    # Test aligned vs conflict conditions
    conditions = [
        ("aligned",   "shiny",  "goal_collect"),     # both → temptation
        ("aligned",   "safe",   "goal_safe_long"),    # both → safe
        ("conflict",  "shiny",  "goal_safe_short"),   # pref→tempt, goal→safe
        ("conflict",  "safe",   "goal_collect"),       # pref→safe, goal→tempt
    ]

    for cond_name, true_theta, true_goal in conditions:
        # Coupled
        coupled_pref_ok, coupled_goal_ok, coupled_joint_ok = 0, 0, 0
        # Factorized
        fact_pref_ok, fact_goal_ok = 0, 0
        n_trials = 50

        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 4000)
            coupled = JointPosteriorV2()
            factorized = JointLatentBelief()

            for obs_i in range(25):
                safe_br = BranchAttributes(
                    safety_score=0.7 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.1 + rng.uniform(0, 0.1),
                    shortcut_bonus=rng.uniform(0, 0.4),
                    texture_novelty=rng.uniform(0, 0.3),
                    risk_penalty=0.1)
                tempt_br = BranchAttributes(
                    safety_score=0.3 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.8 + rng.uniform(-0.1, 0.1),
                    shortcut_bonus=rng.uniform(0, 0.2),
                    texture_novelty=rng.uniform(0.2, 0.5),
                    risk_penalty=0.4)
                branches = [safe_br, tempt_br]

                # Joint-driven choice
                lik0 = compute_joint_likelihood(0, branches, true_goal, true_theta, params)
                lik1 = compute_joint_likelihood(1, branches, true_goal, true_theta, params)
                p0 = lik0 / (lik0 + lik1 + 1e-10)
                chosen = int(rng.random() > p0)
                coupled.update_from_choice(chosen, branches, params)
                factorized.update_from_choice(chosen, branches, params)

            # Evaluate
            cg, cp = coupled.predicted_joint
            if cp == true_theta:
                coupled_pref_ok += 1
            if cg == true_goal:
                coupled_goal_ok += 1
            if cp == true_theta and cg == true_goal:
                coupled_joint_ok += 1
            if factorized.pref_posterior.predicted_type == true_theta:
                fact_pref_ok += 1
            if factorized.goal_posterior.predicted_type == true_goal:
                fact_goal_ok += 1

        results.append({
            "condition": cond_name,
            "theta": true_theta, "goal": true_goal,
            "coupled_pref": round(coupled_pref_ok / n_trials, 3),
            "coupled_goal": round(coupled_goal_ok / n_trials, 3),
            "coupled_joint": round(coupled_joint_ok / n_trials, 3),
            "factorized_pref": round(fact_pref_ok / n_trials, 3),
            "factorized_goal": round(fact_goal_ok / n_trials, 3),
        })
    return results


# ══════════════════════════════════════════════════════════════
# L2+L3: Joint Conflict Corridor with joint tutor
# ══════════════════════════════════════════════════════════════
def l2_l3_conflict_corridor():
    print("L2+L3: Conflict Corridor", file=sys.stderr)
    agent_params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
    train_seeds = list(range(50))
    probe_seeds = list(range(100, 160))
    strategies = ["always_wait", "always_warn", "v4", "joint_v1", "oracle"]
    conflict_levels = {"low": 0.3, "med": 0.6, "high": 0.9}
    results = []

    for cname, cval in conflict_levels.items():
        for strat in strategies:
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            lib = BranchConceptLibrary()
            scorer = BranchScorerProbe(lr=0.05, l2=0.01)
            tutor_v4 = LearningAwarePolicyV4() if strat == "v4" else None
            tutor_joint = JointLatentTutorV1(agent_params=agent_params) if strat == "joint_v1" else None
            warns, waits = 0, 0
            pref_ok, goal_ok, joint_ok, total_latent = 0, 0, 0, 0

            for seed in train_seeds:
                gm, _, meta, sc = generate_scenario(
                    "joint_conflict_corridor", seed, DIFF,
                    latent_mode=True, conflict_strength=cval)
                fb, ww = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                rng = np.random.default_rng(seed + 7000)

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

                # Agent choice
                ba_safe = BranchAttributes(
                    safety_score=float(ss[0]), temptation_score=0.1,
                    risk_penalty=0.1)
                ba_risky = BranchAttributes(
                    safety_score=float(sr[0]),
                    temptation_score=cval * 0.8,
                    risk_penalty=0.4)
                branches = [ba_safe, ba_risky]
                true_theta = getattr(sc, 'latent_preference', 'neutral')
                agent_choice = sample_branch_choice(branches, true_theta, agent_params, rng)

                # Tutor
                do_warn = False
                if strat in ("always_warn", "oracle"):
                    do_warn = True
                elif strat == "always_wait":
                    do_warn = False
                elif strat == "v4":
                    action, _ = tutor_v4.decide(sc, fb, lp, lib, scorer, 2)
                    do_warn = (action == "WARN")
                elif strat == "joint_v1":
                    action, _ = tutor_joint.decide(sc, fb, lp, lib, scorer, 2)
                    do_warn = (action == "WARN")
                    tutor_joint.observe_agent_choice(agent_choice, branches)

                if do_warn:
                    warns += 1
                    for r, c in sc.risky_cells:
                        z = fb[r, c]
                        lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
                    ss2 = summarize_branch(sc.safe_cells, fb, fv, lp)
                    sr2 = summarize_branch(sc.risky_cells, fb, fv, lp)
                    lib.update("safe_branch", ss2)
                    lib.update("risky_branch", sr2)
                    scorer.update(build_scorer_input(ss2, lib), 1.0)
                    scorer.update(build_scorer_input(sr2, lib), 0.0)
                else:
                    waits += 1

                # Joint latent accuracy
                if strat == "joint_v1" and tutor_joint is not None:
                    total_latent += 1
                    gp, pp = tutor_joint.joint_posterior.predicted_joint
                    if pp == true_theta:
                        pref_ok += 1
                    true_g = getattr(sc, 'latent_goal', 'goal_safe_short')
                    if gp == true_g:
                        goal_ok += 1
                    if pp == true_theta and gp == true_g:
                        joint_ok += 1

            # Probe
            per_seed = []
            for ps in probe_seeds:
                gm, _, meta, sc = generate_scenario(
                    "joint_conflict_corridor", ps, DIFF,
                    latent_mode=True, conflict_strength=cval)
                fb, _ = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
                trng = np.random.default_rng(ps + 777)
                cands = vis_candidates(sc, 2)
                best, _ = choose_branch(
                    cands, fb, fv, lp, passable, lib, scorer,
                    lambda_b=1.0, score_mode="hybrid", tie_rng=trng)
                per_seed.append(int(best.branch_id == sc.oracle_safe_branch_id))

            per_seed = np.array(per_seed)
            sbcr = float(np.mean(per_seed))
            brng = np.random.default_rng(42)
            bm = [float(np.mean(per_seed[brng.integers(0, len(per_seed), len(per_seed))]))
                   for _ in range(N_BOOT)]
            total = warns + waits

            results.append({
                "conflict": cname, "strategy": strat,
                "SBCR": round(sbcr, 3),
                "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
                "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
                "warn_rate": round(warns / max(total, 1), 3),
                "pref_acc": round(pref_ok / max(total_latent, 1), 3) if total_latent > 0 else None,
                "goal_acc": round(goal_ok / max(total_latent, 1), 3) if total_latent > 0 else None,
                "joint_acc": round(joint_ok / max(total_latent, 1), 3) if total_latent > 0 else None,
            })
    return results


# ══════════════════════════════════════════════════════════════
# L4: Persistent Agent Profile
# ══════════════════════════════════════════════════════════════
def l4_persistent_profile():
    print("L4: Persistent Profile", file=sys.stderr)
    agent_params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
    n_episodes = 8
    episode_seeds = list(range(n_episodes))
    probe_seeds = list(range(200, 230))
    true_theta = "shiny"  # Fixed across episodes

    # Compare: fresh posterior each episode vs persistent
    results = []
    for mode in ["fresh", "persistent"]:
        tutor = JointLatentTutorV1(agent_params=agent_params)
        ep_warns = []
        ep_pref_acc = []
        ep_sbcr = []

        for ep_idx, ep_seed in enumerate(episode_seeds):
            if mode == "fresh":
                tutor = JointLatentTutorV1(agent_params=agent_params)
            # else: keep tutor's joint_posterior across episodes

            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            lib = BranchConceptLibrary()
            scorer = BranchScorerProbe(lr=0.05, l2=0.01)
            warns, waits = 0, 0

            for seed in range(ep_seed * 5, ep_seed * 5 + 5):
                gm, _, meta, sc = generate_scenario(
                    "temptation_corridor", seed, DIFF,
                    latent_mode=True, temptation_strength=0.8)
                fb, ww = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                rng = np.random.default_rng(seed + 9000)

                # Force consistent theta
                sc.latent_preference = true_theta

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

                ba_safe = BranchAttributes(safety_score=float(ss[0]),
                    temptation_score=0.1, risk_penalty=0.1)
                ba_risky = BranchAttributes(safety_score=float(sr[0]),
                    temptation_score=0.7, risk_penalty=0.4)
                branches = [ba_safe, ba_risky]
                agent_choice = sample_branch_choice(
                    branches, true_theta, agent_params, rng)

                action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
                tutor.observe_agent_choice(agent_choice, branches)

                if action == "WARN":
                    warns += 1
                    for r, c in sc.risky_cells:
                        z = fb[r, c]
                        lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
                else:
                    waits += 1

            total = warns + waits
            _, pp = tutor.joint_posterior.predicted_joint
            ep_warns.append(warns / max(total, 1))
            ep_pref_acc.append(1.0 if pp == true_theta else 0.0)

            # Probe SBCR
            ep_correct = []
            for ps in probe_seeds:
                gm, _, meta, sc = generate_scenario(
                    "temptation_corridor", ps, DIFF,
                    latent_mode=True, temptation_strength=0.8)
                fb, _ = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
                trng = np.random.default_rng(ps + 777)
                cands = vis_candidates(sc, 2)
                best, _ = choose_branch(
                    cands, fb, fv, lp, passable, lib, scorer,
                    lambda_b=1.0, score_mode="hybrid", tie_rng=trng)
                ep_correct.append(int(best.branch_id == sc.oracle_safe_branch_id))
            ep_sbcr.append(float(np.mean(ep_correct)))

        for i in range(n_episodes):
            results.append({
                "mode": mode, "episode": i,
                "warn_rate": round(ep_warns[i], 3),
                "pref_acc": round(ep_pref_acc[i], 3),
                "SBCR": round(ep_sbcr[i], 3),
            })
    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    l1 = l1_coupled_vs_factorized()
    l2l3 = l2_l3_conflict_corridor()
    l4 = l4_persistent_profile()

    with open(out / "joint_conflict_report.md", "w") as f:
        f.write("# Joint Latent Conflict Report\n\n")

        # L1
        f.write("## L1: Coupled Joint vs Factorized (25 obs, 50 trials)\n\n")
        f.write("| Condition | θ | Goal | Coupled Pref | Coupled Goal | Coupled Joint | Fact Pref | Fact Goal |\n")
        f.write("|-----------|---|------|-------------|-------------|--------------|----------|----------|\n")
        for r in l1:
            f.write("| {} | {} | {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |\n".format(
                r["condition"], r["theta"], r["goal"],
                r["coupled_pref"], r["coupled_goal"], r["coupled_joint"],
                r["factorized_pref"], r["factorized_goal"]))

        # L2+L3
        f.write("\n## L2+L3: Joint Conflict Corridor\n\n")
        f.write("| Conflict | Strategy | SBCR | CI | WarnRate | PrefAcc | GoalAcc | JointAcc |\n")
        f.write("|----------|----------|------|----|---------|---------|---------|---------|\n")
        for r in l2l3:
            pa = "{:.0%}".format(r["pref_acc"]) if r["pref_acc"] is not None else "—"
            ga = "{:.0%}".format(r["goal_acc"]) if r["goal_acc"] is not None else "—"
            ja = "{:.0%}".format(r["joint_acc"]) if r["joint_acc"] is not None else "—"
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} | {} | {} | {} |\n".format(
                r["conflict"], r["strategy"], r["SBCR"],
                r["CI_lo"], r["CI_hi"], r["warn_rate"], pa, ga, ja))

        # L4
        f.write("\n## L4: Persistent Agent Profile (θ=shiny, 8 episodes)\n\n")
        f.write("| Mode | Episode | WarnRate | PrefAcc | SBCR |\n")
        f.write("|------|---------|---------|---------|------|\n")
        for r in l4:
            f.write("| {} | {} | {:.0%} | {:.0%} | {:.0%} |\n".format(
                r["mode"], r["episode"], r["warn_rate"], r["pref_acc"], r["SBCR"]))

        # L4 summary
        f.write("\n### Persistent vs Fresh Summary\n\n")
        f.write("| Metric | Fresh (avg) | Persistent (avg) |\n")
        f.write("|--------|------------|------------------|\n")
        fresh = [r for r in l4 if r["mode"] == "fresh"]
        persist = [r for r in l4 if r["mode"] == "persistent"]
        f.write("| WarnRate | {:.0%} | {:.0%} |\n".format(
            np.mean([r["warn_rate"] for r in fresh]),
            np.mean([r["warn_rate"] for r in persist])))
        f.write("| PrefAcc | {:.0%} | {:.0%} |\n".format(
            np.mean([r["pref_acc"] for r in fresh]),
            np.mean([r["pref_acc"] for r in persist])))
        f.write("| SBCR | {:.0%} | {:.0%} |\n".format(
            np.mean([r["SBCR"] for r in fresh]),
            np.mean([r["SBCR"] for r in persist])))

    print("Report -> results/joint_conflict_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
