"""I1+I2+I4: Framework Integration + Temptation + Robustness.

I1: Runtime adapter parity test (state_types integration)
I2: Temptation corridor experiments with preference-aware tutor
I4: Cross-family robustness with all 4 families
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
from src.agents.preference_posterior import PreferencePosterior
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
from src.teachers.preference_aware_policy_v1 import PreferenceAwarePolicyV1
from src.core.adapters import (
    from_scenario_to_world, from_world_to_observation,
    update_agent_belief, compute_branch_posterior, infer_robot_belief,
)
from src.core.state_types import AgentBelief

out = Path("results")
out.mkdir(exist_ok=True)

DIFF = "medium"
N_BOOT = 200
TRAIN_SEEDS = list(range(40))
PROBE_SEEDS = list(range(100, 150))


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


def run_condition(family, strategy, train_seeds, probe_seeds,
                   obs_radius=2, kw=None):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor_v4 = LearningAwarePolicyV4() if strategy == "v4" else None
    tutor_pref = PreferenceAwarePolicyV1() if strategy == "pref_v1" else None
    warns, waits = 0, 0
    pref_correct = 0
    pref_total = 0
    kw = kw or {}

    for seed in train_seeds:
        gm, _, meta, sc = generate_scenario(family, seed, DIFF, latent_mode=True, **kw)
        fb, ww = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)

        # I1: Build WorldState + AgentBelief via adapters
        ws = from_scenario_to_world(gm, meta, sc, fb, ww)
        obs = from_world_to_observation(ws, sc, obs_radius)
        belief = AgentBelief(risk_head_params=lp, concept_library=lib, scorer_probe=scorer)
        belief = update_agent_belief(belief, obs, fb, lp)

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

        # I1: Compute BranchPosterior + RobotBelief via adapters
        belief.branch_summaries[0] = ss
        belief.branch_summaries[1] = sr
        bp = compute_branch_posterior(belief)
        rb = infer_robot_belief(ws, obs, belief)

        do_warn = False
        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        if strategy in ("always_warn", "oracle"):
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "v4":
            action, _ = tutor_v4.decide(sc, fb, lp, lib, scorer, obs_radius)
            do_warn = (action == "WARN")
        elif strategy == "pref_v1":
            action, diag = tutor_pref.decide(
                sc, fb, lp, lib, scorer, obs_radius,
                temptation_strength=tempt_str)
            do_warn = (action == "WARN")
            # Observe agent's simulated "choice" for preference update
            branch_a = np.array([ss[0], 1 - ss[0], tempt_str * 0.1, 0.0])
            branch_b = np.array([sr[0], 1 - sr[0], tempt_str * 0.8, 0.0])
            # Agent "chooses" based on oracle safety
            tutor_pref.observe_agent_choice(branch_a, True)
            tutor_pref.observe_agent_choice(branch_b, False)

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

        # Track preference inference accuracy
        if hasattr(sc, 'latent_preference') and tutor_pref is not None:
            pref_total += 1
            if tutor_pref.pref_posterior.predicted_type == sc.latent_preference:
                pref_correct += 1

    # Probe
    per_seed = []
    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(family, ps, DIFF, latent_mode=True, **kw)
        fb, _ = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)
        passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
        trng = np.random.default_rng(ps + 777)
        cands = vis_candidates(sc, obs_radius)
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
    pref_acc = pref_correct / max(pref_total, 1)
    return {
        "SBCR": round(sbcr, 3),
        "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
        "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
        "warn_rate": round(warns / max(total, 1), 3),
        "pref_acc": round(pref_acc, 3) if pref_total > 0 else None,
    }


# ══════════════════════════════════════════════════════════════
# I2: Temptation experiments
# ══════════════════════════════════════════════════════════════
def i2_temptation():
    print("I2: Temptation Corridor", file=sys.stderr)
    strengths = {"low": 0.3, "med": 0.6, "high": 0.9}
    strategies = ["always_wait", "always_warn", "v4", "pref_v1", "oracle"]
    results = []
    for sname, sval in strengths.items():
        for strat in strategies:
            r = run_condition(
                "temptation_corridor", strat, TRAIN_SEEDS, PROBE_SEEDS,
                obs_radius=2, kw={"temptation_strength": sval})
            r["tempt"] = sname
            r["strategy"] = strat
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════
# I4: Cross-family robustness
# ══════════════════════════════════════════════════════════════
def i4_robustness():
    print("I4: Cross-Family Robustness", file=sys.stderr)
    configs = {
        "elcb_po":        {"fam": "elcb_po",           "kw": {},             "obs": 2},
        "delayed_Δ=-2":   {"fam": "delayed_corridor",  "kw": {"commit_depth": 1, "reveal_depth": 3}, "obs": 1},
        "delayed_Δ=2":    {"fam": "delayed_corridor",  "kw": {"commit_depth": 5, "reveal_depth": 3}, "obs": 5},
        "distractor_high":{"fam": "distractor_cue",    "kw": {"distractor_salience": 0.95}, "obs": 2},
        "tempt_high":     {"fam": "temptation_corridor","kw": {"temptation_strength": 0.9}, "obs": 2},
    }
    strategies = ["always_wait", "v4", "pref_v1", "oracle"]
    results = []
    for cname, cfg in configs.items():
        for strat in strategies:
            r = run_condition(
                cfg["fam"], strat, TRAIN_SEEDS, PROBE_SEEDS,
                obs_radius=cfg["obs"], kw=cfg["kw"])
            r["family"] = cname
            r["strategy"] = strat
            results.append(r)
    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    i2 = i2_temptation()
    i4 = i4_robustness()

    with open(out / "framework_integration_report.md", "w") as f:
        f.write("# Framework Integration Report\n\n")

        # I2
        f.write("## I2: Temptation Corridor — Preference-Aware Tutor\n\n")
        f.write("| Temptation | Strategy | SBCR | CI | WarnRate | PrefAcc |\n")
        f.write("|------------|----------|------|----|---------|---------|\n")
        for r in i2:
            pa = "{:.0%}".format(r["pref_acc"]) if r["pref_acc"] is not None else "—"
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} | {} |\n".format(
                r["tempt"], r["strategy"], r["SBCR"],
                r["CI_lo"], r["CI_hi"], r["warn_rate"], pa))

        # pref_v1 vs v4 comparison
        f.write("\n### pref_v1 vs v4 Comparison\n\n")
        f.write("| Tempt | v4 WR | v4 SBCR | pref WR | pref SBCR | pref Acc | Oracle |\n")
        f.write("|-------|-------|---------|---------|-----------|---------|--------|\n")
        for t in ["low", "med", "high"]:
            v4 = [r for r in i2 if r["tempt"] == t and r["strategy"] == "v4"][0]
            pv = [r for r in i2 if r["tempt"] == t and r["strategy"] == "pref_v1"][0]
            orc = [r for r in i2 if r["tempt"] == t and r["strategy"] == "oracle"][0]
            pa = "{:.0%}".format(pv["pref_acc"]) if pv["pref_acc"] is not None else "—"
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {} | {:.0%} |\n".format(
                t, v4["warn_rate"], v4["SBCR"],
                pv["warn_rate"], pv["SBCR"], pa, orc["SBCR"]))

        # I4
        f.write("\n## I4: Cross-Family Robustness (4 families)\n\n")
        f.write("| Family | Strategy | SBCR | CI | WarnRate | PrefAcc |\n")
        f.write("|--------|----------|------|----|---------|---------|\n")
        for r in i4:
            pa = "{:.0%}".format(r["pref_acc"]) if r["pref_acc"] is not None else "—"
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} | {} |\n".format(
                r["family"], r["strategy"], r["SBCR"],
                r["CI_lo"], r["CI_hi"], r["warn_rate"], pa))

        # Overall v4 vs pref summary
        f.write("\n### Framework Summary: v4 vs pref_v1\n\n")
        f.write("| Family | v4 SBCR | pref SBCR | Oracle | v4=Oracle? | pref=Oracle? |\n")
        f.write("|--------|---------|-----------|--------|-----------|-------------|\n")
        for fname in ["elcb_po", "delayed_Δ=-2", "delayed_Δ=2", "distractor_high", "tempt_high"]:
            v4 = [r for r in i4 if r["family"] == fname and r["strategy"] == "v4"][0]
            pv = [r for r in i4 if r["family"] == fname and r["strategy"] == "pref_v1"][0]
            orc = [r for r in i4 if r["family"] == fname and r["strategy"] == "oracle"][0]
            m4 = "✅" if abs(v4["SBCR"] - orc["SBCR"]) < 0.15 else "⚠️"
            mp = "✅" if abs(pv["SBCR"] - orc["SBCR"]) < 0.15 else "⚠️"
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {} | {} |\n".format(
                fname, v4["SBCR"], pv["SBCR"], orc["SBCR"], m4, mp))

    print("Report -> results/framework_integration_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
