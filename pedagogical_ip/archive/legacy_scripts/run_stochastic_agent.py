"""J1+J2+J3: Stochastic Agent + Preference Inference + Tutor v2.

Validates:
  J1: Agent behavior varies by θ (preference sensitivity, temperature)
  J2: q(θ) converges to true θ with PrefAcc > chance
  J3: pref_v2 balances safety vs observation value

Key experiment: temptation corridor with stochastic agent,
comparing v4 / pref_v1 / pref_v2 / oracle.
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
    compute_choice_probs, PREFERENCE_TYPES,
)
from src.agents.preference_posterior_v2 import PreferencePosteriorV2
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2

out = Path("results")
out.mkdir(exist_ok=True)

DIFF = "medium"
TRAIN_SEEDS = list(range(50))
PROBE_SEEDS = list(range(100, 160))
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
# J1: Validate stochastic agent behavior
# ══════════════════════════════════════════════════════════════
def j1_agent_validation():
    print("J1: Stochastic Agent Validation", file=sys.stderr)
    results = []

    # Test across preference types and betas
    for beta in [2.0, 4.0, 8.0]:
        for theta in PREFERENCE_TYPES:
            # Create test branches
            safe_br = BranchAttributes(
                safety_score=0.8, temptation_score=0.1,
                risk_penalty=0.1)
            tempt_br = BranchAttributes(
                safety_score=0.3, temptation_score=0.9,
                risk_penalty=0.5)
            branches = [safe_br, tempt_br]
            params = AgentPolicyParams(beta=beta, epsilon=0.1, lambda_theta=1.0)
            probs = compute_choice_probs(branches, theta, params)
            results.append({
                "beta": beta, "theta": theta,
                "P_safe": round(float(probs[0]), 3),
                "P_tempt": round(float(probs[1]), 3),
            })
    return results


# ══════════════════════════════════════════════════════════════
# J2: Preference posterior convergence
# ══════════════════════════════════════════════════════════════
def j2_posterior_convergence():
    print("J2: Posterior Convergence", file=sys.stderr)
    results = []
    params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

    for true_theta in PREFERENCE_TYPES:
        pref_correct = 0
        n_trials = 50

        for trial in range(n_trials):
            rng = np.random.default_rng(trial + 1000)
            posterior = PreferencePosteriorV2()

            # Simulate 20 observations
            for obs_i in range(20):
                # Randomize branch attributes slightly per observation
                safe_br = BranchAttributes(
                    safety_score=0.7 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.1 + rng.uniform(0, 0.1),
                    risk_penalty=0.1)
                tempt_br = BranchAttributes(
                    safety_score=0.3 + rng.uniform(-0.1, 0.1),
                    temptation_score=0.8 + rng.uniform(-0.1, 0.1),
                    risk_penalty=0.4)
                branches = [safe_br, tempt_br]

                chosen = sample_branch_choice(branches, true_theta, params, rng)
                posterior.update_from_choice(chosen, branches, params)

            if posterior.predicted_type == true_theta:
                pref_correct += 1

        acc = pref_correct / n_trials
        results.append({
            "true_theta": true_theta,
            "PrefAcc": round(acc, 3),
            "n_trials": n_trials,
        })

    return results


# ══════════════════════════════════════════════════════════════
# J3: Full temptation experiment with stochastic agent + pref_v2
# ══════════════════════════════════════════════════════════════
def j3_temptation_full():
    print("J3: Full Temptation Experiment", file=sys.stderr)
    agent_params = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
    strengths = {"low": 0.3, "med": 0.6, "high": 0.9}
    strategies = ["always_wait", "always_warn", "v4", "pref_v2", "oracle"]
    results = []

    for sname, sval in strengths.items():
        for strat in strategies:
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            lib = BranchConceptLibrary()
            scorer = BranchScorerProbe(lr=0.05, l2=0.01)
            tutor_v4 = LearningAwarePolicyV4() if strat == "v4" else None
            tutor_v2 = PreferenceAwarePolicyV2(agent_params=agent_params) if strat == "pref_v2" else None
            warns, waits = 0, 0
            pref_correct, pref_total = 0, 0
            agent_chose_safe_count, agent_total = 0, 0

            for seed in TRAIN_SEEDS:
                gm, _, meta, sc = generate_scenario(
                    "temptation_corridor", seed, DIFF,
                    latent_mode=True, temptation_strength=sval)
                fb, ww = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                rng = np.random.default_rng(seed + 5000)

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

                # Build branch attrs for stochastic agent
                tempt_str = getattr(sc, 'temptation_strength', sval)
                ba_safe = BranchAttributes(
                    safety_score=float(ss[0]),
                    temptation_score=getattr(sc, 'tempt_score_a', 0.1)
                        if sc.oracle_safe_branch_id == 0
                        else getattr(sc, 'tempt_score_b', tempt_str * 0.8),
                    risk_penalty=0.1)
                ba_risky = BranchAttributes(
                    safety_score=float(sr[0]),
                    temptation_score=getattr(sc, 'tempt_score_b', tempt_str * 0.8)
                        if sc.oracle_safe_branch_id == 0
                        else getattr(sc, 'tempt_score_a', 0.1),
                    risk_penalty=0.4)
                branches = [ba_safe, ba_risky]

                # Stochastic agent chooses
                true_theta = getattr(sc, 'latent_preference', 'neutral')
                agent_choice = sample_branch_choice(
                    branches, true_theta, agent_params, rng)
                agent_total += 1
                if agent_choice == 0:  # chose safe
                    agent_chose_safe_count += 1

                # Tutor decides
                do_warn = False
                if strat in ("always_warn", "oracle"):
                    do_warn = True
                elif strat == "always_wait":
                    do_warn = False
                elif strat == "v4":
                    action, _ = tutor_v4.decide(sc, fb, lp, lib, scorer, 2)
                    do_warn = (action == "WARN")
                elif strat == "pref_v2":
                    action, diag = tutor_v2.decide(sc, fb, lp, lib, scorer, 2)
                    do_warn = (action == "WARN")
                    # Robot observes agent's stochastic choice
                    tutor_v2.observe_agent_choice(agent_choice, branches)

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

                # Track preference inference
                if strat == "pref_v2" and tutor_v2 is not None:
                    pref_total += 1
                    if tutor_v2.pref_posterior.predicted_type == true_theta:
                        pref_correct += 1

            # Probe
            per_seed = []
            for ps in PROBE_SEEDS:
                gm, _, meta, sc = generate_scenario(
                    "temptation_corridor", ps, DIFF,
                    latent_mode=True, temptation_strength=sval)
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
            pref_acc = pref_correct / max(pref_total, 1) if pref_total > 0 else None
            agent_safe_rate = agent_chose_safe_count / max(agent_total, 1)

            results.append({
                "tempt": sname, "strategy": strat,
                "SBCR": round(sbcr, 3),
                "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
                "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
                "warn_rate": round(warns / max(total, 1), 3),
                "pref_acc": round(pref_acc, 3) if pref_acc is not None else None,
                "agent_safe_rate": round(agent_safe_rate, 3),
            })

    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    j1 = j1_agent_validation()
    j2 = j2_posterior_convergence()
    j3 = j3_temptation_full()

    with open(out / "stochastic_agent_report.md", "w") as f:
        f.write("# Stochastic Agent + Preference Inference Report\n\n")

        # J1: Agent behavior
        f.write("## J1: Stochastic Agent Behavior Validation\n\n")
        f.write("| β | θ | P(safe) | P(tempt) |\n")
        f.write("|---|---|---------|----------|\n")
        for r in j1:
            f.write("| {} | {} | {} | {} |\n".format(
                r["beta"], r["theta"], r["P_safe"], r["P_tempt"]))

        # J2: Posterior convergence
        f.write("\n## J2: Preference Posterior Convergence (20 obs, 50 trials)\n\n")
        f.write("| True θ | PrefAcc |\n")
        f.write("|--------|--------|\n")
        for r in j2:
            f.write("| {} | {:.0%} |\n".format(r["true_theta"], r["PrefAcc"]))
        mean_acc = np.mean([r["PrefAcc"] for r in j2])
        f.write("\n**Mean PrefAcc: {:.1%}** (chance = {:.1%})\n".format(mean_acc, 1/5))

        # J3: Full experiment
        f.write("\n## J3: Temptation Corridor with Stochastic Agent\n\n")
        f.write("| Tempt | Strategy | SBCR | CI | WarnRate | PrefAcc | AgentSafe% |\n")
        f.write("|-------|----------|------|----|---------|---------|-----------|\n")
        for r in j3:
            pa = "{:.0%}".format(r["pref_acc"]) if r["pref_acc"] is not None else "—"
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} | {} | {:.0%} |\n".format(
                r["tempt"], r["strategy"], r["SBCR"],
                r["CI_lo"], r["CI_hi"], r["warn_rate"], pa,
                r["agent_safe_rate"]))

        # pref_v2 comparison
        f.write("\n### pref_v2 vs v4 vs oracle\n\n")
        f.write("| Tempt | v4 WR | v4 SBCR | pref_v2 WR | pref_v2 SBCR | pref_v2 Acc | Oracle |\n")
        f.write("|-------|-------|---------|-----------|-------------|------------|--------|\n")
        for t in ["low", "med", "high"]:
            v4 = [r for r in j3 if r["tempt"] == t and r["strategy"] == "v4"][0]
            pv = [r for r in j3 if r["tempt"] == t and r["strategy"] == "pref_v2"][0]
            orc = [r for r in j3 if r["tempt"] == t and r["strategy"] == "oracle"][0]
            pa = "{:.0%}".format(pv["pref_acc"]) if pv["pref_acc"] is not None else "—"
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {} | {:.0%} |\n".format(
                t, v4["warn_rate"], v4["SBCR"],
                pv["warn_rate"], pv["SBCR"], pa, orc["SBCR"]))

    print("Report -> results/stochastic_agent_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
