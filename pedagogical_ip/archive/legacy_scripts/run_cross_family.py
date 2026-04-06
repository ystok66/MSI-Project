"""Cross-Family Phase Diagram: D1+D2+E1 unified experiment.

D1: Delayed Commitment sweep — Δ ∈ {-2,-1,0,1,2}
D2: Distractor Cue — salience ∈ {low, med, high}
E1: Tutor v3 with self-discovery term

Outputs phase diagram tables + CI.
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

out = Path("results")
out.mkdir(exist_ok=True)

DIFF = "medium"
N_BOOT = 200
TRAIN_SEEDS = list(range(40))
PROBE_SEEDS = list(range(100, 150))

# ══════════════════════════════════════════════════════════════
# Helpers (shared infrastructure)
# ══════════════════════════════════════════════════════════════

def apply_fix(gm, meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def vis_candidates(sc, obs_radius):
    """Build BranchCandidates from visible cells."""
    fk = sc.fork_cell
    mg = sc.merge_cell
    ma = make_observation_mask(sc.branch_a_cells, fk, obs_radius)
    mb = make_observation_mask(sc.branch_b_cells, fk, obs_radius)
    va = [c for c, m in zip(sc.branch_a_cells, ma) if m > 0.5]
    vb = [c for c, m in zip(sc.branch_b_cells, mb) if m > 0.5]
    return [
        BranchCandidate(0, va, len(va), fk, mg, (1, fk[1]), (1, mg[1])),
        BranchCandidate(1, vb, len(vb), fk, mg, (3, fk[1]), (3, mg[1])),
    ]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def tutor_v3_decide(sc, fb, lp, lib, scorer, obs_radius,
                     lambda_s=1.0, lambda_d=2.0, lambda_u=1.5,
                     lambda_v=1.5, lambda_m=1.0, lambda_c=0.05,
                     lambda_r=0.3, tau_u=2.0):
    """Tutor v3: urgency + self-discovery + margin gain.

    Returns (action, diag_dict).
    """
    fv = np.full_like(fb, 0.3)

    # Pre-warning (visible only)
    cands = vis_candidates(sc, obs_radius)
    s_a_pre = summarize_branch(cands[0].cells, fb, fv, lp)
    s_b_pre = summarize_branch(cands[1].cells, fb, fv, lp)
    margin_pre = abs(s_a_pre[0] - s_b_pre[0])

    # Post-warning (full branch)
    s_a_post = summarize_branch(sc.branch_a_cells, fb, fv, lp)
    s_b_post = summarize_branch(sc.branch_b_cells, fb, fv, lp)
    margin_post = abs(s_a_post[0] - s_b_post[0])

    delta_m = max(margin_post - margin_pre, 0)
    delta_s = delta_m

    # Timing: commit_depth vs reveal_depth
    d_commit = getattr(sc, 'commit_depth', obs_radius + 1)
    d_reveal = getattr(sc, 'reveal_depth', 3)
    urgency = float(_sigmoid((d_reveal - d_commit) / tau_u))

    # Self-discovery: can agent see strong cues before committing?
    v_self = 1.0 if d_commit >= d_reveal else 0.0

    # Missed window: if WAIT and can't self-discover
    p_miss = 1.0 if d_commit < d_reveal else 0.0

    # Redundancy: if scorer already confident
    inp_a = build_scorer_input(s_a_pre, lib)
    inp_b = build_scorer_input(s_b_pre, lib)
    score_a = scorer.score(inp_a)
    score_b = scorer.score(inp_b)
    confidence = abs(score_a - score_b)
    redundancy = max(confidence - 0.7, 0)

    Q_warn = (lambda_s * delta_s + lambda_d * delta_m + lambda_u * urgency
              - lambda_c * 1.0 - lambda_r * redundancy)
    Q_wait = lambda_v * v_self - lambda_m * p_miss

    action = "WARN" if Q_warn > Q_wait else "WAIT"
    return action, {
        "Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
        "urgency": round(urgency, 4), "v_self": v_self, "p_miss": p_miss,
        "delta_m": round(delta_m, 4), "d_commit": d_commit, "d_reveal": d_reveal,
    }


def train_and_probe(family, train_seeds, probe_seeds, strategy="oracle",
                     obs_radius=2, extra_kw=None):
    """Train + probe for a given family/strategy. Returns per-seed results."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    warn_count = 0
    wait_count = 0
    kw = extra_kw or {}

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
        if strategy == "always_warn" or strategy == "oracle":
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "v3":
            action, _ = tutor_v3_decide(sc, fb, lp, lib, scorer, obs_radius)
            do_warn = (action == "WARN")

        if do_warn:
            warn_count += 1
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
            wait_count += 1

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
    ci_lo = float(np.percentile(bm, 2.5))
    ci_hi = float(np.percentile(bm, 97.5))

    total = warn_count + wait_count
    wr = warn_count / max(total, 1)
    return {
        "SBCR": round(sbcr, 3), "CI_lo": round(ci_lo, 3), "CI_hi": round(ci_hi, 3),
        "warn_rate": round(wr, 3), "warn_count": warn_count, "wait_count": wait_count,
    }


# ══════════════════════════════════════════════════════════════
# D1: Delayed Commitment Phase Diagram
# ══════════════════════════════════════════════════════════════
def d1_phase_diagram():
    print("D1: Delayed Commitment Phase Diagram", file=sys.stderr)
    deltas = [-2, -1, 0, 1, 2]
    base_reveal = 3
    strategies = ["always_wait", "always_warn", "v3", "oracle"]
    results = []

    for delta in deltas:
        commit = base_reveal + delta
        for strat in strategies:
            r = train_and_probe(
                "delayed_corridor", TRAIN_SEEDS, PROBE_SEEDS,
                strategy=strat, obs_radius=commit,
                extra_kw={"commit_depth": commit, "reveal_depth": base_reveal})
            r["delta"] = delta
            r["commit_depth"] = commit
            r["reveal_depth"] = base_reveal
            r["strategy"] = strat
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# D2: Distractor Cue
# ══════════════════════════════════════════════════════════════
def d2_distractor():
    print("D2: Distractor Cue", file=sys.stderr)
    saliences = {"low": 0.3, "med": 0.6, "high": 0.95}
    strategies = ["always_wait", "always_warn", "v3", "oracle"]
    results = []

    for sal_name, sal_val in saliences.items():
        for strat in strategies:
            r = train_and_probe(
                "distractor_cue", TRAIN_SEEDS, PROBE_SEEDS,
                strategy=strat, obs_radius=2,
                extra_kw={"distractor_salience": sal_val})
            r["salience"] = sal_name
            r["salience_val"] = sal_val
            r["strategy"] = strat
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    d1 = d1_phase_diagram()
    d2 = d2_distractor()

    with open(out / "cross_family_report.md", "w") as f:
        f.write("# Cross-Family Phase Diagram Report\n\n")

        # D1
        f.write("## D1: Delayed Commitment — Selectivity Phase Diagram\n\n")
        f.write("Δ = commit_depth - reveal_depth (base reveal=3)\n\n")
        f.write("| Δ | Strategy | SBCR | CI | WarnRate |\n")
        f.write("|---|----------|------|----|---------|\n")
        for r in d1:
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} |\n".format(
                r["delta"], r["strategy"], r["SBCR"], r["CI_lo"], r["CI_hi"], r["warn_rate"]))

        # Selectivity summary
        f.write("\n### v3 Selectivity Summary\n\n")
        f.write("| Δ | v3 WarnRate | v3 SBCR | Oracle SBCR | Match? |\n")
        f.write("|---|-----------|---------|-------------|--------|\n")
        for delta in [-2, -1, 0, 1, 2]:
            v3 = [r for r in d1 if r["delta"] == delta and r["strategy"] == "v3"][0]
            orc = [r for r in d1 if r["delta"] == delta and r["strategy"] == "oracle"][0]
            match = "≈" if abs(v3["SBCR"] - orc["SBCR"]) < 0.15 else "≠"
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {} |\n".format(
                delta, v3["warn_rate"], v3["SBCR"], orc["SBCR"], match))

        # D2
        f.write("\n## D2: Distractor Cue — Robustness\n\n")
        f.write("| Salience | Strategy | SBCR | CI | WarnRate |\n")
        f.write("|----------|----------|------|----|---------|\n")
        for r in d2:
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} |\n".format(
                r["salience"], r["strategy"], r["SBCR"], r["CI_lo"], r["CI_hi"], r["warn_rate"]))

    print("Report -> results/cross_family_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
