"""H2+H3+H4: Framework Robustness Suite.

H2: p_self calibration + WarnRate(Δ) curve
H3: DVOI consolidation
H4: Cross-family 50-seed CI robustness

Unified experiment across ELCB-PO, Delayed Commitment, Distractor Cue.
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
from src.metrics.self_discovery import estimate_self_discovery_prob
from src.metrics.calibration import (
    expected_calibration_error, compute_empirical_self_discovery, compute_warm_rate_curve,
)
from src.core.state_types import TutorDecisionTrace

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


def run_full_condition(family, strategy, train_seeds, probe_seeds,
                        obs_radius=2, kw=None):
    """Train + probe with full diagnostics including DVOI and traces."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = LearningAwarePolicyV4() if strategy == "v4" else None
    warns, waits = 0, 0
    traces = []
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
        elif strategy == "v4":
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, obs_radius)
            do_warn = (action == "WARN")
            t = TutorDecisionTrace(
                selected_action=action, Q_warn=diag["Q_warn"], Q_wait=diag["Q_wait"],
                dvoi=diag["dvoi"], p_self=diag["p_self"],
                margin_pre=diag["margin_pre"], margin_post=diag["margin_post"],
                delta_margin=diag["delta_s"],
                d_commit=diag["d_commit"], d_reveal=diag["d_reveal"],
                delta=diag.get("delta", diag["d_commit"] - diag["d_reveal"]),
            )
            traces.append(t)

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

    # Probe with DVOI measurement
    per_seed = []
    dvoi_vals = []
    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(family, ps, DIFF, latent_mode=True, **kw)
        fb, _ = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)
        passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
        trng = np.random.default_rng(ps + 777)
        cands = vis_candidates(sc, obs_radius)

        # Pre-warning margin
        s_a_pre = summarize_branch(cands[0].cells, fb, fv, lp)
        s_b_pre = summarize_branch(cands[1].cells, fb, fv, lp)
        margin_pre = abs(s_a_pre[0] - s_b_pre[0])
        u_pre = float(_sigmoid(margin_pre))

        # Post-warning margin (hypothetical)
        s_a_post = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        margin_post = abs(s_a_post[0] - s_b_post[0])
        u_post = float(_sigmoid(margin_post))
        dvoi = max(u_post - u_pre, 0)
        dvoi_vals.append(dvoi)

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
        "mean_dvoi": round(float(np.mean(dvoi_vals)), 4),
        "traces": traces,
    }


# ══════════════════════════════════════════════════════════════
# H2: Calibration — continuous Δ sweep with p_self
# ══════════════════════════════════════════════════════════════
def h2_calibration():
    print("H2: Calibration", file=sys.stderr)
    deltas = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    base_reveal = 3

    predicted_p_selfs = []
    empirical_selfs = []
    warn_rates = []
    sbcrs = []

    for delta in deltas:
        commit = base_reveal + delta
        obs_r = max(commit, 1)

        p_self = estimate_self_discovery_prob(commit, base_reveal, tau_v=1.0)
        emp_self = compute_empirical_self_discovery(obs_r, base_reveal, 10)
        predicted_p_selfs.append(p_self)
        empirical_selfs.append(emp_self)

        r = run_full_condition(
            "delayed_corridor", "v4", TRAIN_SEEDS, PROBE_SEEDS,
            obs_radius=obs_r,
            kw={"commit_depth": commit, "reveal_depth": base_reveal})
        warn_rates.append(r["warn_rate"])
        sbcrs.append(r["SBCR"])

    ece, bins = expected_calibration_error(
        np.array(predicted_p_selfs), np.array(empirical_selfs))

    curve = compute_warm_rate_curve(deltas, warn_rates)

    return {
        "deltas": deltas, "p_self": predicted_p_selfs,
        "emp_self": empirical_selfs, "warn_rates": warn_rates,
        "sbcrs": sbcrs, "ece": ece, "bins": bins, "curve": curve,
    }


# ══════════════════════════════════════════════════════════════
# H4: Cross-family robustness
# ══════════════════════════════════════════════════════════════
def h4_robustness():
    print("H4: Cross-Family Robustness", file=sys.stderr)
    families = {
        "elcb_po": {"kw": {}, "obs_r": 2},
        "delayed_Δ=-2": {"kw": {"commit_depth": 1, "reveal_depth": 3}, "obs_r": 1},
        "delayed_Δ=0":  {"kw": {"commit_depth": 3, "reveal_depth": 3}, "obs_r": 3},
        "delayed_Δ=2":  {"kw": {"commit_depth": 5, "reveal_depth": 3}, "obs_r": 5},
        "distractor_low": {"kw": {"distractor_salience": 0.3}, "obs_r": 2},
        "distractor_high": {"kw": {"distractor_salience": 0.95}, "obs_r": 2},
    }
    strategies = ["always_wait", "v4", "oracle"]
    results = []

    for fname, cfg in families.items():
        fam_key = "elcb_po" if fname == "elcb_po" else (
            "delayed_corridor" if "delayed" in fname else "distractor_cue")
        for strat in strategies:
            r = run_full_condition(
                fam_key, strat, TRAIN_SEEDS, PROBE_SEEDS,
                obs_radius=cfg["obs_r"], kw=cfg["kw"])
            r["family"] = fname
            r["strategy"] = strat
            results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cal = h2_calibration()
    rob = h4_robustness()

    with open(out / "framework_report.md", "w") as f:
        f.write("# Pedagogical Decision Framework Report\n\n")

        # H2 Calibration
        f.write("## H2: Selectivity Law Calibration\n\n")
        f.write("### p_self vs Empirical Self-Discovery\n\n")
        f.write("| Δ | p_self | emp_self | WarnRate | SBCR |\n")
        f.write("|---|-------|---------|---------|------|\n")
        for i, d in enumerate(cal["deltas"]):
            f.write("| {} | {:.2f} | {:.2f} | {:.0%} | {:.0%} |\n".format(
                d, cal["p_self"][i], cal["emp_self"][i],
                cal["warn_rates"][i], cal["sbcrs"][i]))

        f.write("\n### Calibration Metrics\n\n")
        f.write("- **ECE(p_self)**: {:.4f}\n".format(cal["ece"]))
        f.write("- **WarnRate curve monotonic**: {}\n".format(cal["curve"]["monotonic"]))
        f.write("- **Transition zone**: {}\n".format(cal["curve"]["transition_zone"]))
        f.write("- **Slope at Δ=0**: {}\n".format(cal["curve"]["slope_at_zero"]))

        # H4 Robustness
        f.write("\n## H4: Cross-Family Robustness\n\n")
        f.write("| Family | Strategy | SBCR | CI | WarnRate | DVOI |\n")
        f.write("|--------|----------|------|----|---------|----- |\n")
        for r in rob:
            f.write("| {} | {} | {:.0%} | [{:.0%},{:.0%}] | {:.0%} | {:.4f} |\n".format(
                r["family"], r["strategy"], r["SBCR"],
                r["CI_lo"], r["CI_hi"], r["warn_rate"], r["mean_dvoi"]))

        # v4 summary
        f.write("\n### v4 Selectivity Summary\n\n")
        f.write("| Family | v4 WR | v4 SBCR | Oracle SBCR | Match |\n")
        f.write("|--------|-------|---------|-------------|-------|\n")
        for fname in ["elcb_po", "delayed_Δ=-2", "delayed_Δ=0", "delayed_Δ=2",
                       "distractor_low", "distractor_high"]:
            v4 = [r for r in rob if r["family"] == fname and r["strategy"] == "v4"][0]
            orc = [r for r in rob if r["family"] == fname and r["strategy"] == "oracle"][0]
            m = "✅" if abs(v4["SBCR"] - orc["SBCR"]) < 0.15 else "⚠️"
            f.write("| {} | {:.0%} | {:.0%} | {:.0%} | {} |\n".format(
                fname, v4["warn_rate"], v4["SBCR"], orc["SBCR"], m))

    print("Report -> results/framework_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
