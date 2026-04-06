"""B1: Learning-Aware WAIT vs WARN Tutor.

Implements four tutor strategies:
  1. always_wait:  never warn
  2. always_warn:  always warn (oracle)
  3. learning_aware: Q_teach-driven WAIT vs WARN
  4. oracle_warn:  oracle full warn (upper bound)

Q_teach(a) = λ_S ΔS(a) + λ_I IG(a) + λ_A ΔAutonomy(a) - λ_C C(a) - λ_R R(a)
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
from src.teachers.rsa_warning_v2 import RSAWarningV2

FAM = "elcb_po"
DIFF = "medium"
out = Path("results")
out.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Common
# ══════════════════════════════════════════════════════════════

def apply_fix(gm, meta, sc):
    rng = np.random.default_rng(42)
    ww_new = generate_world_weights_orthogonal(rng, d=4)
    all_branch = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb_mod = neutralize_identity_features(meta.cell_features, all_branch, 0.5)
    return fb_mod, ww_new


def make_candidates(sc, cells_a, cells_b):
    fork = sc.fork_cell
    merge = sc.merge_cell
    return [
        BranchCandidate(0, list(cells_a), len(cells_a),
                        fork, merge, (1, fork[1]), (1, merge[1])),
        BranchCandidate(1, list(cells_b), len(cells_b),
                        fork, merge, (3, fork[1]), (3, merge[1])),
    ]


def make_visible_candidates(sc, obs_radius=2):
    fork_pos = sc.fork_cell
    mask_a = make_observation_mask(sc.branch_a_cells, fork_pos, obs_radius)
    mask_b = make_observation_mask(sc.branch_b_cells, fork_pos, obs_radius)
    vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
    vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]
    return make_candidates(sc, vis_a, vis_b)


def compute_branch_margin(sc, fb_mod, lp):
    """Mean risk difference between branches (positive = branch_a riskier)."""
    risk_a = float(np.mean([lp.predict_risk(fb_mod[r, c]) for r, c in sc.branch_a_cells]))
    risk_b = float(np.mean([lp.predict_risk(fb_mod[r, c]) for r, c in sc.branch_b_cells]))
    return risk_a - risk_b


def compute_branch_entropy(summary, lib):
    scores = lib.score_all(summary, obs_var=0.01, tau=1.0)
    if not scores or len(scores) < 2:
        return 0.693
    vals = np.array(list(scores.values()))
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return float(-np.sum(probs * np.log(probs + 1e-15)))


# ══════════════════════════════════════════════════════════════
# Learning-Aware Tutor
# ══════════════════════════════════════════════════════════════

class LearningAwareTutor:
    """Q_teach-driven WAIT vs WARN decision.

    Q_teach(a) = λ_S ΔS + λ_I IG + λ_A ΔM_warn - λ_C C - λ_R R

    Where:
      ΔS = predicted success improvement from warning
      IG = predicted entropy reduction
      ΔM_warn = predicted branch margin increase (autonomy proxy)
      C = intervention cost (fixed for warn)
      R = redundancy penalty (high if already confident)
    """

    def __init__(self, lambda_s=1.0, lambda_i=0.5, lambda_a=0.3,
                 lambda_c=0.1, lambda_r=0.5,
                 confidence_threshold=0.7, rsa=None):
        self.lambda_s = lambda_s
        self.lambda_i = lambda_i
        self.lambda_a = lambda_a
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.confidence_threshold = confidence_threshold
        self.rsa = rsa or RSAWarningV2()
        self.warn_history = []  # track past warns for redundancy

    def decide(self, sc, fb_mod, lp, lib, scorer):
        """Decide WAIT or WARN for current episode.

        Returns: (action, Q_warn, Q_wait, diagnostics)
        """
        fv = np.full_like(fb_mod, 0.3)
        passable = np.ones((fb_mod.shape[0], fb_mod.shape[1]), dtype=bool)

        # Current branch summaries from visible cells
        candidates = make_visible_candidates(sc, obs_radius=2)
        s_a = summarize_branch(candidates[0].cells, fb_mod, fv, lp)
        s_b = summarize_branch(candidates[1].cells, fb_mod, fv, lp)

        # Pre-warning entropy
        h_a_pre = compute_branch_entropy(s_a, lib)
        h_b_pre = compute_branch_entropy(s_b, lib)
        U_pre = h_a_pre + h_b_pre

        # Pre-warning branch margin (from visible cells only)
        r_a = s_a[0]  # mean risk from summary
        r_b = s_b[0]
        margin_pre = abs(r_a - r_b)

        # Scorer confidence
        score_a = scorer.score(build_scorer_input(s_a, lib))
        score_b = scorer.score(build_scorer_input(s_b, lib))
        confidence = abs(score_a - score_b)

        # Estimate post-warning state (simulate warning effect)
        # Warning reveals full branch → use full-branch summaries
        s_a_full = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
        s_b_full = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
        h_a_post = compute_branch_entropy(s_a_full, lib)
        h_b_post = compute_branch_entropy(s_b_full, lib)
        U_post = h_a_post + h_b_post

        margin_post = abs(s_a_full[0] - s_b_full[0])

        # Components
        delta_s = max(margin_post - margin_pre, 0)  # success proxy
        ig = max(U_pre - U_post, 0)
        delta_m = max(margin_post - margin_pre, 0)  # autonomy proxy
        cost_warn = 1.0  # fixed intervention cost
        redundancy = max(confidence - self.confidence_threshold, 0)

        # Q values
        Q_warn = (self.lambda_s * delta_s
                  + self.lambda_i * ig
                  + self.lambda_a * delta_m
                  - self.lambda_c * cost_warn
                  - self.lambda_r * redundancy)
        Q_wait = 0.0  # WAIT has no immediate cost or benefit

        action = "WARN" if Q_warn > Q_wait else "WAIT"

        diag = {
            "U_pre": round(U_pre, 4),
            "U_post": round(U_post, 4),
            "IG": round(ig, 4),
            "margin_pre": round(margin_pre, 4),
            "margin_post": round(margin_post, 4),
            "delta_m": round(delta_m, 4),
            "confidence": round(confidence, 4),
            "redundancy": round(redundancy, 4),
            "Q_warn": round(Q_warn, 4),
            "Q_wait": round(Q_wait, 4),
        }

        return action, Q_warn, Q_wait, diag


# ══════════════════════════════════════════════════════════════
# Training with strategy
# ══════════════════════════════════════════════════════════════

def train_with_strategy(seeds, strategy="always_warn", lp=None, lib=None,
                         scorer=None, tutor=None):
    """Train with specified intervention strategy."""
    if lp is None:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    if lib is None:
        lib = BranchConceptLibrary()
    if scorer is None:
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    if tutor is None:
        tutor = LearningAwareTutor()

    warn_count = 0
    wait_count = 0
    total = 0

    for seed in seeds:
        gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)
        fb_mod, ww = apply_fix(gm, meta, sc)
        fv = np.full_like(fb_mod, 0.3)

        # Cell-level learning
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb_mod[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        # Branch concept/scorer learning (always)
        s_s = summarize_branch(sc.safe_cells, fb_mod, fv, lp)
        s_r = summarize_branch(sc.risky_cells, fb_mod, fv, lp)
        lib.update("safe_branch", s_s)
        lib.update("risky_branch", s_r)
        scorer.update(build_scorer_input(s_s, lib), 1.0)
        scorer.update(build_scorer_input(s_r, lib), 0.0)

        # Decide intervention
        total += 1
        if strategy == "always_wait":
            do_warn = False
        elif strategy == "always_warn":
            do_warn = True
        elif strategy == "learning_aware":
            action, _, _, _ = tutor.decide(sc, fb_mod, lp, lib, scorer)
            do_warn = (action == "WARN")
        elif strategy == "oracle":
            do_warn = True
        else:
            do_warn = False

        if do_warn:
            warn_count += 1
            for r, c in sc.risky_cells:
                z = fb_mod[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
            s_s2 = summarize_branch(sc.safe_cells, fb_mod, fv, lp)
            s_r2 = summarize_branch(sc.risky_cells, fb_mod, fv, lp)
            lib.update("safe_branch", s_s2)
            lib.update("risky_branch", s_r2)
            scorer.update(build_scorer_input(s_s2, lib), 1.0)
            scorer.update(build_scorer_input(s_r2, lib), 0.0)
        else:
            wait_count += 1

    stats = {
        "warn_rate": round(warn_count / max(total, 1), 3),
        "warn_count": warn_count,
        "wait_count": wait_count,
    }
    return lp, lib, scorer, stats


def probe_sbcr(probe_seeds, lp, lib, scorer, obs_radius=2):
    safe = 0
    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
        fb_mod, _ = apply_fix(gm, meta, sc)
        fv = np.full_like(fb_mod, 0.3)
        passable = np.ones((fb_mod.shape[0], fb_mod.shape[1]), dtype=bool)
        tie_rng = np.random.default_rng(ps + 777)
        candidates = make_visible_candidates(sc, obs_radius)
        best, _ = choose_branch(
            candidates, fb_mod, fv, lp, passable, lib, scorer,
            lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
        safe += int(best.branch_id == sc.oracle_safe_branch_id)
    return round(safe / len(probe_seeds), 3)


# ══════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 150))
    N_BOOTSTRAP = 200

    strategies = {
        "always_wait": "always_wait",
        "always_warn": "always_warn",
        "learning_aware": "learning_aware",
        "oracle": "oracle",
    }

    all_results = []
    all_stats = {}

    for strat_name, strat in strategies.items():
        print("Strategy: {}".format(strat_name), file=sys.stderr)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        tutor = LearningAwareTutor() if strat == "learning_aware" else None
        k_done = 0
        cumulative_stats = {"warn_count": 0, "wait_count": 0}

        for k_target in K_VALUES:
            batch = list(range(k_done, k_target))
            if batch:
                lp, lib, scorer, stats = train_with_strategy(
                    batch, strategy=strat, lp=lp, lib=lib, scorer=scorer,
                    tutor=tutor)
                cumulative_stats["warn_count"] += stats["warn_count"]
                cumulative_stats["wait_count"] += stats["wait_count"]
            k_done = k_target

            # Per-seed probe for bootstrap
            per_seed = []
            for ps in PROBE_SEEDS:
                gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
                fb_mod, _ = apply_fix(gm, meta, sc)
                fv = np.full_like(fb_mod, 0.3)
                passable = np.ones((fb_mod.shape[0], fb_mod.shape[1]), dtype=bool)
                tie_rng = np.random.default_rng(ps + 777)
                candidates = make_visible_candidates(sc, 2)
                best, _ = choose_branch(
                    candidates, fb_mod, fv, lp, passable, lib, scorer,
                    lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
                per_seed.append(int(best.branch_id == sc.oracle_safe_branch_id))

            per_seed = np.array(per_seed)
            sbcr = float(np.mean(per_seed))

            boot_rng = np.random.default_rng(k_target * 100 + 7777)
            boot_m = [float(np.mean(per_seed[boot_rng.integers(0, len(per_seed), len(per_seed))]))
                       for _ in range(N_BOOTSTRAP)]
            ci_lo = float(np.percentile(boot_m, 2.5))
            ci_hi = float(np.percentile(boot_m, 97.5))

            all_results.append({
                "strategy": strat_name,
                "k": k_target,
                "SBCR": round(sbcr, 3),
                "CI_lo": round(ci_lo, 3),
                "CI_hi": round(ci_hi, 3),
            })

        total_ep = cumulative_stats["warn_count"] + cumulative_stats["wait_count"]
        all_stats[strat_name] = {
            "warn_count": cumulative_stats["warn_count"],
            "wait_count": cumulative_stats["wait_count"],
            "warn_rate": round(cumulative_stats["warn_count"] / max(total_ep, 1), 3),
        }

    # Write report
    with open(out / "learning_aware_report.md", "w") as f:
        f.write("# B1: Learning-Aware WAIT vs WARN Report\n\n")

        f.write("## Intervention Statistics\n\n")
        f.write("| Strategy | Warn Count | Wait Count | Warn Rate |\n")
        f.write("|----------|-----------|------------|----------|\n")
        for sn, st in all_stats.items():
            f.write("| {} | {} | {} | {:.0%} |\n".format(
                sn, st["warn_count"], st["wait_count"], st["warn_rate"]))

        f.write("\n## Training-Probe Results (50 seeds, 95% CI)\n\n")
        f.write("| Strategy | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|----------|-----|-----|-----|------|------|\n")
        for sn in strategies:
            rows = [r for r in all_results if r["strategy"] == sn]
            cells = ["{:.0%} [{:.0%},{:.0%}]".format(r["SBCR"], r["CI_lo"], r["CI_hi"])
                     for r in rows]
            f.write("| {} | {} |\n".format(sn, " | ".join(cells)))

        f.write("\n### Learning Gain LG(k) = SBCR(k) - SBCR(0)\n\n")
        for sn in strategies:
            rows = [r for r in all_results if r["strategy"] == sn]
            k0 = rows[0]["SBCR"]
            parts = ["k={}:{:+.0%}".format(r["k"], r["SBCR"] - k0) for r in rows[1:]]
            f.write("- **{}**: {}\n".format(sn, ", ".join(parts)))

        f.write("\n### Pedagogical Efficiency PE(k) = LG(k) / warn_rate\n\n")
        for sn in ["always_warn", "learning_aware", "oracle"]:
            rows = [r for r in all_results if r["strategy"] == sn]
            k0 = rows[0]["SBCR"]
            wr = all_stats[sn]["warn_rate"]
            if wr > 0:
                parts = ["k={}:{:.2f}".format(r["k"], (r["SBCR"] - k0) / wr) for r in rows[1:]]
                f.write("- **{}** (WR={:.0%}): {}\n".format(sn, wr, ", ".join(parts)))

    print("Report -> results/learning_aware_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
