"""G4+G5+G6: Continuous Phase Diagram + Distractor Robustness + CI.

G4: Delayed Commitment Δ ∈ {-3,-2,-1,0,1,2,3} with v3 vs v4
G5: Distractor Cue × salience sweep
G6: 50 seeds bootstrap CI on all
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
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch
from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4, TutorV4Config

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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def tutor_v3_decide(sc, fb, lp, lib, scorer, obs_radius):
    """v3: binary V_self."""
    fv = np.full_like(fb, 0.3)
    cands = vis_candidates(sc, obs_radius)
    s_a = summarize_branch(cands[0].cells, fb, fv, lp)
    s_b = summarize_branch(cands[1].cells, fb, fv, lp)
    margin_pre = abs(s_a[0] - s_b[0])
    s_a_f = summarize_branch(sc.branch_a_cells, fb, fv, lp)
    s_b_f = summarize_branch(sc.branch_b_cells, fb, fv, lp)
    margin_post = abs(s_a_f[0] - s_b_f[0])
    delta_m = max(margin_post - margin_pre, 0)
    d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
    d_reveal = getattr(sc, 'reveal_depth', 3)
    urgency = float(_sigmoid((d_reveal - d_commit) / 2.0))
    v_self = 1.0 if d_commit >= d_reveal else 0.0
    p_miss = 1.0 if d_commit < d_reveal else 0.0
    Q_warn = 1.0 * delta_m + 2.0 * delta_m + 1.5 * urgency - 0.05
    Q_wait = 1.5 * v_self - 1.0 * p_miss
    return "WARN" if Q_warn > Q_wait else "WAIT"


def run_condition(family, strategy, train_seeds, probe_seeds, obs_radius=2, kw=None):
    """Train + probe a single condition. Returns results dict."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor_v4 = LearningAwarePolicyV4() if strategy == "v4" else None
    warns, waits = 0, 0
    kw = kw or {}

    for seed in train_seeds:
        gm, _, meta, sc = generate_scenario(family, seed, DIFF, latent_mode=True, **kw)
        fb, ww = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)

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

        do_warn = False
        if strategy in ("always_warn", "oracle"):
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "v3":
            do_warn = (tutor_v3_decide(sc, fb, lp, lib, scorer, obs_radius) == "WARN")
        elif strategy == "v4":
            action, _ = tutor_v4.decide(sc, fb, lp, lib, scorer, obs_radius)
            do_warn = (action == "WARN")

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
    return {
        "SBCR": round(sbcr, 3),
        "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
        "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
        "warn_rate": round(warns / max(total, 1), 3),
        "warns": warns, "waits": waits,
    }


# ══════════════════════════════════════════════════════════════
# G4: Continuous Δ sweep
# ══════════════════════════════════════════════════════════════
def g4_continuous_phase():
    print("G4: Continuous Phase Diagram", file=sys.stderr)
    deltas = [-3, -2, -1, 0, 1, 2, 3]
    base_reveal = 3
    strategies = ["always_wait", "always_warn", "v3", "v4", "oracle"]
    results = []

    for delta in deltas:
        commit = base_reveal + delta
        for strat in strategies:
            r = run_condition(
                "delayed_corridor", strat, TRAIN_SEEDS, PROBE_SEEDS,
                obs_radius=max(commit, 1),
                kw={"commit_depth": commit, "reveal_depth": base_reveal})
            r["delta"] = delta
            r["strategy"] = strat
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# G5: Distractor robustness
# ══════════════════════════════════════════════════════════════
def g5_distractor():
    print("G5: Distractor Robustness", file=sys.stderr)
    saliences = {"low": 0.3, "med": 0.6, "high": 0.95}
    strategies = ["always_wait", "always_warn", "v4", "oracle"]
    results = []

    for sname, sval in saliences.items():
        for strat in strategies:
            r = run_condition(
                "distractor_cue", strat, TRAIN_SEEDS, PROBE_SEEDS,
                obs_radius=2,
                kw={"distractor_salience": sval})
            r["salience"] = sname
            r["strategy"] = strat
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# G6: Cross-family ELCB-PO validation
# ══════════════════════════════════════════════════════════════
def g6_elcb_po():
    print("G6: ELCB-PO validation", file=sys.stderr)
    strategies = ["always_wait", "always_warn", "v4", "oracle"]
    results = []
    for strat in strategies:
        r = run_condition("elcb_po", strat, TRAIN_SEEDS, PROBE_SEEDS, obs_radius=2)
        r["strategy"] = strat
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    g4 = g4_continuous_phase()
    g5 = g5_distractor()
    g6 = g6_elcb_po()

    with open(out / "selectivity_law_report.md", "w") as f:
        f.write("# Selectivity Law Report: Tutor v4\n\n")

        # G4
        f.write("## G4: Delayed Commitment — Continuous Selectivity (Δ sweep)\n\n")
        f.write("| Δ | Strategy | SBCR | CI | WarnRate |\n")
        f.write("|---|----------|------|----|---------|\n")
        for r in g4:
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} |\n".format(
                r["delta"], r["strategy"], r["SBCR"], r["CI_lo"], r["CI_hi"], r["warn_rate"]))

        # v3 vs v4 comparison
        f.write("\n### v3 vs v4 Selectivity\n\n")
        f.write("| Δ | v3 WR | v3 SBCR | v4 WR | v4 SBCR | Oracle SBCR |\n")
        f.write("|---|-------|---------|-------|---------|-------------|\n")
        for delta in [-3, -2, -1, 0, 1, 2, 3]:
            v3 = [r for r in g4 if r["delta"] == delta and r["strategy"] == "v3"][0]
            v4 = [r for r in g4 if r["delta"] == delta and r["strategy"] == "v4"][0]
            orc = [r for r in g4 if r["delta"] == delta and r["strategy"] == "oracle"][0]
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |\n".format(
                delta, v3["warn_rate"], v3["SBCR"], v4["warn_rate"], v4["SBCR"], orc["SBCR"]))

        # G5
        f.write("\n## G5: Distractor Cue Robustness\n\n")
        f.write("| Salience | Strategy | SBCR | CI | WarnRate |\n")
        f.write("|----------|----------|------|----|---------|\n")
        for r in g5:
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} |\n".format(
                r["salience"], r["strategy"], r["SBCR"], r["CI_lo"], r["CI_hi"], r["warn_rate"]))

        # G6
        f.write("\n## G6: ELCB-PO Cross-Validation\n\n")
        f.write("| Strategy | SBCR | CI | WarnRate |\n")
        f.write("|----------|------|----|---------|\n")
        for r in g6:
            f.write("| {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} |\n".format(
                r["strategy"], r["SBCR"], r["CI_lo"], r["CI_hi"], r["warn_rate"]))

    print("Report -> results/selectivity_law_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
