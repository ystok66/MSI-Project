"""Tutor Value Audit: A1 Timing Curve + A2 Value Decomposition + A3 CI.

A1: Warning timing sweep (5 timing points)
A2: Tutor value decomposition (OHG, IG, LG, PE, WFR)
A3: Training-probe loop with 50 seeds and bootstrap CI
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
# Common infrastructure
# ══════════════════════════════════════════════════════════════

def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def apply_fix(gm, meta, sc):
    """Apply full three-layer fix: lane-neutral + orthogonal weights."""
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
    """Candidates using only fork-visible cells."""
    fork_pos = sc.fork_cell
    mask_a = make_observation_mask(sc.branch_a_cells, fork_pos, obs_radius)
    mask_b = make_observation_mask(sc.branch_b_cells, fork_pos, obs_radius)
    vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
    vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]
    return make_candidates(sc, vis_a, vis_b)


def compute_branch_entropy(summary, lib):
    """H(π) = -Σ P(k|π) log P(k|π) via concept scores."""
    scores = lib.score_all(summary, obs_var=0.01, tau=1.0)
    if not scores or len(scores) < 2:
        return 0.693  # log(2) = max entropy for 2 classes
    vals = np.array(list(scores.values()))
    vals = vals - np.max(vals)
    probs = np.exp(vals) / np.sum(np.exp(vals))
    return float(-np.sum(probs * np.log(probs + 1e-15)))


def train_models(seeds, use_rsa=False, oracle_warn=False,
                  lp=None, lib=None, scorer=None,
                  warn_depth=None):
    """Train V5 models with full fix. warn_depth controls how many
    branch cells the warning reveals (None = all risky cells)."""
    if lp is None:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    if lib is None:
        lib = BranchConceptLibrary()
    if scorer is None:
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    rsa = RSAWarningV2()

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

        # Full branch concept/scorer
        s_s = summarize_branch(sc.safe_cells, fb_mod, fv, lp)
        s_r = summarize_branch(sc.risky_cells, fb_mod, fv, lp)
        lib.update("safe_branch", s_s)
        lib.update("risky_branch", s_r)
        scorer.update(build_scorer_input(s_s, lib), 1.0)
        scorer.update(build_scorer_input(s_r, lib), 0.0)

        if oracle_warn:
            # Oracle reveals ALL risky cells or up to warn_depth
            reveal_cells = sc.risky_cells
            if warn_depth is not None:
                reveal_cells = sc.risky_cells[:warn_depth]
            for r, c in reveal_cells:
                z = fb_mod[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
            s_s2 = summarize_branch(sc.safe_cells, fb_mod, fv, lp)
            s_r2 = summarize_branch(sc.risky_cells, fb_mod, fv, lp)
            lib.update("safe_branch", s_s2)
            lib.update("risky_branch", s_r2)
            scorer.update(build_scorer_input(s_s2, lib), 1.0)
            scorer.update(build_scorer_input(s_r2, lib), 0.0)
        elif use_rsa:
            s_a = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp)
            s_b = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp)
            z_state = rsa.classify_risk_state(s_a[0], s_b[0], "left", gap_threshold=0.01)
            _, u_name = rsa.choose_utterance(z_state)
            if u_name != "silence":
                reveal_cells = sc.risky_cells
                if warn_depth is not None:
                    reveal_cells = sc.risky_cells[:warn_depth]
                for r, c in reveal_cells:
                    z = fb_mod[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=0.5)

    return lp, lib, scorer


def probe_choice(sc, fb_mod, lp, lib, scorer, use_branch=True,
                 obs_radius=2, tie_rng=None):
    """Single probe: return (chose_safe, entropy_a, entropy_b)."""
    fv = np.full_like(fb_mod, 0.3)
    passable = np.ones((fb_mod.shape[0], fb_mod.shape[1]), dtype=bool)

    candidates = make_visible_candidates(sc, obs_radius)

    # Compute branch entropies from visible summaries
    s_a = summarize_branch(candidates[0].cells, fb_mod, fv, lp)
    s_b = summarize_branch(candidates[1].cells, fb_mod, fv, lp)
    h_a = compute_branch_entropy(s_a, lib)
    h_b = compute_branch_entropy(s_b, lib)

    if use_branch:
        best, _ = choose_branch(
            candidates, fb_mod, fv, lp, passable, lib, scorer,
            lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
        chose_safe = int(best.branch_id == sc.oracle_safe_branch_id)
    else:
        c_a = sum(lp.predict_cost(fb_mod[r, c]) for r, c in candidates[0].cells)
        c_b = sum(lp.predict_cost(fb_mod[r, c]) for r, c in candidates[1].cells)
        margin = c_b - c_a
        if abs(margin) < 1e-4:
            chosen = int(tie_rng.integers(0, 2))
        else:
            chosen = 0 if c_a < c_b else 1
        chose_safe = int(chosen == sc.oracle_safe_branch_id)

    return chose_safe, h_a, h_b


# ══════════════════════════════════════════════════════════════
# A1: Timing Curve
# ══════════════════════════════════════════════════════════════
def a1_timing_curve():
    """Sweep warning timing: how much of branch does warning reveal."""
    print("A1: Timing Curve", file=sys.stderr)
    train_seeds = list(range(40))
    probe_seeds = list(range(100, 150))

    # Timing points: warn_depth = how many risky cells the warning reveals
    # For medium: branch_len=10, reveal_depth=3
    # depth=0: no warning info (baseline)
    # depth=1: just first risky cell (very early, weak cue only)
    # depth=3: up to reveal_depth (still weak)
    # depth=5: past reveal_depth (some strong cues)
    # depth=10: all cells (oracle)
    timing_points = {
        "no_warn":       0,
        "1_cell_early":  1,
        "weak_only":     3,   # = reveal_depth
        "partial_strong": 5,
        "full_oracle":   10,
    }

    results = {}
    for timing_name, depth in timing_points.items():
        use_warn = depth > 0
        lp, lib, scorer = train_models(
            train_seeds, oracle_warn=use_warn,
            warn_depth=depth if depth > 0 else None)

        safe_count = 0
        entropies = []
        for ps in probe_seeds:
            gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
            fb_mod, _ = apply_fix(gm, meta, sc)
            tie_rng = np.random.default_rng(ps + 777)
            chose_safe, h_a, h_b = probe_choice(
                sc, fb_mod, lp, lib, scorer, use_branch=True, tie_rng=tie_rng)
            safe_count += chose_safe
            entropies.append(h_a + h_b)

        n = len(probe_seeds)
        results[timing_name] = {
            "depth": depth,
            "SBCR": round(safe_count / n, 3),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "n": n,
        }

    return results


# ══════════════════════════════════════════════════════════════
# A2: Tutor Value Decomposition
# ══════════════════════════════════════════════════════════════
def a2_value_decomposition():
    """Full tutor value decomposition with entropy and IG."""
    print("A2: Value Decomposition", file=sys.stderr)
    train_seeds = list(range(40))
    probe_seeds = list(range(100, 150))

    configs = {
        "no_tutor+old":       (False, False, False, False),
        "no_tutor+branch":    (True,  False, False, True),
        "rsa_warn+branch":    (True,  True,  False, True),
        "oracle_warn+branch": (True,  False, True,  True),
    }

    results = {}
    for name, (use_br, use_rsa, oracle, use_mask) in configs.items():
        lp, lib, scorer = train_models(
            train_seeds, use_rsa=use_rsa, oracle_warn=oracle)

        safe_count = 0
        entropies_pre = []
        n = len(probe_seeds)

        for ps in probe_seeds:
            gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
            fb_mod, _ = apply_fix(gm, meta, sc)
            tie_rng = np.random.default_rng(ps + 777)
            obs_r = 2 if use_mask else 99
            chose_safe, h_a, h_b = probe_choice(
                sc, fb_mod, lp, lib, scorer,
                use_branch=use_br, obs_radius=obs_r, tie_rng=tie_rng)
            safe_count += chose_safe
            entropies_pre.append(h_a + h_b)

        results[name] = {
            "SBCR": round(safe_count / n, 3),
            "mean_entropy": round(float(np.mean(entropies_pre)), 4),
            "n": n,
        }

    # Compute decomposition
    sbcr_no = results["no_tutor+branch"]["SBCR"]
    sbcr_rsa = results["rsa_warn+branch"]["SBCR"]
    sbcr_oracle = results["oracle_warn+branch"]["SBCR"]
    sbcr_old = results["no_tutor+old"]["SBCR"]

    ent_no = results["no_tutor+branch"]["mean_entropy"]
    ent_rsa = results["rsa_warn+branch"]["mean_entropy"]
    ent_oracle = results["oracle_warn+branch"]["mean_entropy"]

    decomp = {
        "OHG_rsa": round(sbcr_rsa - sbcr_no, 3),
        "OHG_oracle": round(sbcr_oracle - sbcr_no, 3),
        "IG_rsa": round(ent_no - ent_rsa, 4),
        "IG_oracle": round(ent_no - ent_oracle, 4),
        "branch_vs_old": round(sbcr_no - sbcr_old, 3),
    }

    return results, decomp


# ══════════════════════════════════════════════════════════════
# A3: Training-Probe with Bootstrap CI
# ══════════════════════════════════════════════════════════════
def a3_training_probe_ci():
    """50-seed training-probe with bootstrap 95% CI."""
    print("A3: Training-Probe CI", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 150))  # 50 seeds
    N_BOOTSTRAP = 200

    configs = {
        "no_tutor+branch":    (True,  False, False),
        "rsa_warn+branch":    (True,  True,  False),
        "oracle_warn+branch": (True,  False, True),
    }

    results = []
    for cond, (use_br, use_rsa, oracle) in configs.items():
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        k_done = 0

        for k_target in K_VALUES:
            batch = list(range(k_done, k_target))
            if batch:
                lp, lib, scorer = train_models(
                    batch, use_rsa=use_rsa, oracle_warn=oracle,
                    lp=lp, lib=lib, scorer=scorer)
            k_done = k_target

            # Collect per-seed results for bootstrap
            per_seed = []
            for ps in PROBE_SEEDS:
                gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
                fb_mod, _ = apply_fix(gm, meta, sc)
                tie_rng = np.random.default_rng(ps + 777)
                chose_safe, _, _ = probe_choice(
                    sc, fb_mod, lp, lib, scorer,
                    use_branch=use_br, obs_radius=2, tie_rng=tie_rng)
                per_seed.append(chose_safe)

            per_seed = np.array(per_seed)
            sbcr = float(np.mean(per_seed))

            # Bootstrap 95% CI
            boot_means = []
            boot_rng = np.random.default_rng(k_target * 100 + 999)
            for _ in range(N_BOOTSTRAP):
                idx = boot_rng.integers(0, len(per_seed), size=len(per_seed))
                boot_means.append(float(np.mean(per_seed[idx])))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            results.append({
                "condition": cond,
                "k": k_target,
                "SBCR": round(sbcr, 3),
                "CI_lo": round(ci_lo, 3),
                "CI_hi": round(ci_hi, 3),
                "n": len(PROBE_SEEDS),
            })

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    timing = a1_timing_curve()
    audit, decomp = a2_value_decomposition()
    ci = a3_training_probe_ci()

    with open(out / "tutor_value_report.md", "w") as f:
        f.write("# Tutor Value Audit Report\n\n")

        # A1
        f.write("## A1: Timing Curve\n\n")
        f.write("| Timing | Depth | SBCR | Entropy |\n")
        f.write("|--------|-------|------|---------|\n")
        for name, d in timing.items():
            f.write("| {} | {} | {:.0%} | {:.4f} |\n".format(
                name, d["depth"], d["SBCR"], d["mean_entropy"]))

        # A2
        f.write("\n## A2: Tutor Value Decomposition\n\n")
        f.write("| Condition | SBCR | Entropy |\n")
        f.write("|-----------|------|---------|\n")
        for name, d in audit.items():
            f.write("| {} | {:.0%} | {:.4f} |\n".format(name, d["SBCR"], d["mean_entropy"]))

        f.write("\n### Decomposition\n")
        f.write("- **OHG (RSA)**: {:+.0%} (SBCR lift from RSA warning)\n".format(decomp["OHG_rsa"]))
        f.write("- **OHG (Oracle)**: {:+.0%} (SBCR lift from oracle)\n".format(decomp["OHG_oracle"]))
        f.write("- **IG (RSA)**: {:.4f} (entropy reduction)\n".format(decomp["IG_rsa"]))
        f.write("- **IG (Oracle)**: {:.4f} (entropy reduction)\n".format(decomp["IG_oracle"]))
        f.write("- **Branch vs Old**: {:+.0%}\n".format(decomp["branch_vs_old"]))

        # A3
        f.write("\n## A3: Training-Probe Loop (50 seeds, bootstrap 95% CI)\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["no_tutor+branch", "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in ci if r["condition"] == cond]
            cells = []
            for r in rows:
                cells.append("{:.0%} [{:.0%},{:.0%}]".format(r["SBCR"], r["CI_lo"], r["CI_hi"]))
            f.write("| {} | {} |\n".format(cond, " | ".join(cells)))

        # Learning gain
        f.write("\n### Learning Gain LG(k) = SBCR(k) - SBCR(0)\n\n")
        for cond in ["no_tutor+branch", "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in ci if r["condition"] == cond]
            k0 = rows[0]["SBCR"]
            parts = []
            for r in rows[1:]:
                parts.append("k={}:{:+.0%}".format(r["k"], r["SBCR"] - k0))
            f.write("- **{}**: {}\n".format(cond, ", ".join(parts)))

    print("Report -> results/tutor_value_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
