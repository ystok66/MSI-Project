"""PO Leakage Audit + Tutor Sensitivity — Three-layer fix evaluation.

Layer A: Lane-ID neutralization (identity dims → 0.5)
Layer B: Semantic subspace (w_risk ⊥ identity dims)
Layer C: Observation mask (visible-only branch summary at fork)

Experiments:
  Exp A: Leakage audit (identity probe accuracy + risk gap ratio)
  Exp B: Tutor sensitivity (5 conditions × SBCR)
  Exp C: Training-probe loop (learning gain)
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
    identity_leakage_probe,
    IDENTITY_DIMS,
)
from src.envs.observation_mask import (
    make_observation_mask,
    summarize_branch_masked,
    branch_entropy,
)
from src.agents.cost_risk_model import LatentCostRiskHead, generate_world_weights
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


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def make_candidates(sc, cells_a, cells_b):
    fork = sc.fork_cell
    merge = sc.merge_cell
    return [
        BranchCandidate(0, list(cells_a), len(cells_a),
                        fork, merge, (1, fork[1]), (1, merge[1])),
        BranchCandidate(1, list(cells_b), len(cells_b),
                        fork, merge, (3, fork[1]), (3, merge[1])),
    ]


def apply_layer(layer, gm, meta, sc):
    """Apply leakage fix layer.

    Returns: (features, ww) — potentially modified.
    Does NOT modify the gridmap (it's read-only for evaluation).
    """
    fb = meta.cell_features
    ww = meta.world_weights
    rng = np.random.default_rng(42)

    if layer == "original":
        return fb.copy(), ww

    elif layer == "lane_neutral":
        # Layer A: set identity dims to 0.5 on branch cells
        all_branch = list(sc.branch_a_cells) + list(sc.branch_b_cells)
        fb_mod = neutralize_identity_features(fb, all_branch, 0.5)
        return fb_mod, ww

    elif layer == "semantic_subspace":
        # Layer B: regenerate world_weights with w_risk ⊥ identity
        ww_new = generate_world_weights_orthogonal(rng, d=4)
        all_branch = list(sc.branch_a_cells) + list(sc.branch_b_cells)
        fb_mod = neutralize_identity_features(fb, all_branch, 0.5)
        return fb_mod, ww_new

    elif layer == "full_fix":
        # Layer A+B+C: all three layers
        ww_new = generate_world_weights_orthogonal(rng, d=4)
        all_branch = list(sc.branch_a_cells) + list(sc.branch_b_cells)
        fb_mod = neutralize_identity_features(fb, all_branch, 0.5)
        return fb_mod, ww_new


def train_models(seeds, layer="full_fix", use_rsa=False, oracle_warn=False,
                  lp=None, lib=None, scorer=None):
    """Train V5 models with specified fix layer. Full-branch learning."""
    use_ortho = layer in ("semantic_subspace", "full_fix")
    rng_ww = np.random.default_rng(42)

    if lp is None:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    if lib is None:
        lib = BranchConceptLibrary()
    if scorer is None:
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    rsa = RSAWarningV2()

    for seed in seeds:
        gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)

        if use_ortho:
            ww = generate_world_weights_orthogonal(rng_ww, d=4)
        else:
            ww = meta.world_weights

        fb_mod, ww_used = apply_layer(layer, gm, meta, sc)

        fv = np.full_like(fb_mod, 0.3)

        # Cell-level supervised learning
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb_mod[r, c]
                    lp.update_from_outcome(z, ww_used.true_cost(z), ww_used.true_risk(z))

        # Full branch concept/scorer learning
        s_s = summarize_branch(sc.safe_cells, fb_mod, fv, lp)
        s_r = summarize_branch(sc.risky_cells, fb_mod, fv, lp)
        lib.update("safe_branch", s_s)
        lib.update("risky_branch", s_r)
        scorer.update(build_scorer_input(s_s, lib), 1.0)
        scorer.update(build_scorer_input(s_r, lib), 0.0)

        if oracle_warn:
            for r, c in sc.risky_cells:
                z = fb_mod[r, c]
                lp.update_from_outcome(z, ww_used.true_cost(z), ww_used.true_risk(z), weight=1.0)
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
                for r, c in sc.risky_cells:
                    z = fb_mod[r, c]
                    lp.update_from_outcome(z, ww_used.true_cost(z), ww_used.true_risk(z), weight=0.5)

    return lp, lib, scorer


def probe_sbcr(probe_seeds, lp, lib, scorer, layer="full_fix",
               use_branch=True, use_obs_mask=False, obs_radius=2):
    """Probe SBCR with specified layer. Optionally use observation mask."""
    safe = 0
    rng_ww = np.random.default_rng(42)

    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
        fb_mod, ww_used = apply_layer(layer, gm, meta, sc)
        fv = np.full_like(fb_mod, 0.3)
        passable = make_passable(gm)
        tie_rng = np.random.default_rng(ps + 777)

        if use_obs_mask:
            # Layer C: observation mask — only visible cells from fork
            fork_pos = sc.fork_cell
            mask_a = make_observation_mask(sc.branch_a_cells, fork_pos, obs_radius)
            mask_b = make_observation_mask(sc.branch_b_cells, fork_pos, obs_radius)
            vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
            vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]
            candidates = make_candidates(sc, vis_a, vis_b)
        else:
            candidates = make_candidates(sc, sc.branch_a_cells, sc.branch_b_cells)

        if use_branch:
            best, _ = choose_branch(
                candidates, fb_mod, fv, lp, passable, lib, scorer,
                lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
            safe += int(best.branch_id == sc.oracle_safe_branch_id)
        else:
            c_a = sum(lp.predict_cost(fb_mod[r, c]) for r, c in candidates[0].cells)
            c_b = sum(lp.predict_cost(fb_mod[r, c]) for r, c in candidates[1].cells)
            margin = c_b - c_a
            if abs(margin) < 1e-4:
                chosen = int(tie_rng.integers(0, 2))
            else:
                chosen = 0 if c_a < c_b else 1
            safe += int(chosen == sc.oracle_safe_branch_id)

    return round(safe / len(probe_seeds), 3)


# ══════════════════════════════════════════════════════════════════
# Exp A: Leakage Audit
# ══════════════════════════════════════════════════════════════════
def exp_a_leakage():
    print("Exp A: Leakage Audit", file=sys.stderr)
    layers = ["original", "lane_neutral", "semantic_subspace", "full_fix"]
    results = {}

    for layer in layers:
        leak_accs = []
        early_gaps = []
        full_gaps = []
        rng_ww = np.random.default_rng(42)

        for seed in range(20):
            gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)
            fb_mod, ww = apply_layer(layer, gm, meta, sc)

            # Identity leakage probe
            leak = identity_leakage_probe(fb_mod, sc.safe_cells, sc.risky_cells)
            leak_accs.append(leak)

            # Train temp model
            lp_temp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = fb_mod[r, c]
                        lp_temp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

            fv = np.full_like(fb_mod, 0.3)
            rd = sc.reveal_depth

            # Early cells summary
            s_vis_a = summarize_branch(sc.branch_a_cells[:rd], fb_mod, fv, lp_temp)
            s_vis_b = summarize_branch(sc.branch_b_cells[:rd], fb_mod, fv, lp_temp)

            # Full cells summary
            s_full_a = summarize_branch(sc.branch_a_cells, fb_mod, fv, lp_temp)
            s_full_b = summarize_branch(sc.branch_b_cells, fb_mod, fv, lp_temp)

            early_gaps.append(abs(s_vis_a[0] - s_vis_b[0]))
            full_gaps.append(abs(s_full_a[0] - s_full_b[0]))

        mean_eg = float(np.mean(early_gaps))
        mean_fg = float(np.mean(full_gaps))
        results[layer] = {
            "leak_acc": round(float(np.mean(leak_accs)), 3),
            "early_gap": round(mean_eg, 4),
            "full_gap": round(mean_fg, 4),
            "ratio": round(mean_fg / max(mean_eg, 1e-6), 1),
        }

    return results


# ══════════════════════════════════════════════════════════════════
# Exp B: Tutor Sensitivity
# ══════════════════════════════════════════════════════════════════
def exp_b_tutor():
    print("Exp B: Tutor Sensitivity", file=sys.stderr)
    train_seeds = list(range(40))
    probe_seeds = list(range(100, 130))
    layer = "full_fix"

    configs = {
        "no_tutor+old":       (False, False, False, False),
        "no_tutor+branch":    (True,  False, False, True),
        "rsa_warn+branch":    (True,  True,  False, True),
        "oracle_warn+branch": (True,  False, True,  True),
    }

    results = {}
    for name, (use_br, use_rsa, oracle, use_mask) in configs.items():
        lp, lib, scorer = train_models(
            train_seeds, layer=layer, use_rsa=use_rsa, oracle_warn=oracle)
        sbcr = probe_sbcr(
            probe_seeds, lp, lib, scorer, layer=layer,
            use_branch=use_br, use_obs_mask=use_mask, obs_radius=2)
        results[name] = sbcr

    return results


# ══════════════════════════════════════════════════════════════════
# Exp C: Training-Probe Loop
# ══════════════════════════════════════════════════════════════════
def exp_c_training_probe():
    print("Exp C: Training-Probe Loop", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))
    layer = "full_fix"

    configs = {
        "no_tutor+old":       (False, False, False, False),
        "no_tutor+branch":    (True,  False, False, True),
        "rsa_warn+branch":    (True,  True,  False, True),
        "oracle_warn+branch": (True,  False, True,  True),
    }

    results = []
    for cond, (use_br, use_rsa, oracle, use_mask) in configs.items():
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        k_done = 0

        for k_target in K_VALUES:
            batch = list(range(k_done, k_target))
            if batch:
                lp, lib, scorer = train_models(
                    batch, layer=layer, use_rsa=use_rsa, oracle_warn=oracle,
                    lp=lp, lib=lib, scorer=scorer)
            k_done = k_target

            sbcr = probe_sbcr(
                PROBE_SEEDS, lp, lib, scorer, layer=layer,
                use_branch=use_br, use_obs_mask=use_mask, obs_radius=2)
            results.append({"condition": cond, "k": k_target, "SBCR": sbcr})

    return results


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    a = exp_a_leakage()
    b = exp_b_tutor()
    c = exp_c_training_probe()

    with open(out / "po_leakage_report.md", "w") as f:
        f.write("# PO Leakage Audit + Tutor Sensitivity Report\n\n")

        # Exp A
        f.write("## Exp A: Leakage Audit\n\n")
        f.write("| Layer | ID Probe Acc | Early Gap | Full Gap | Ratio |\n")
        f.write("|-------|-------------|-----------|----------|-------|\n")
        for layer, d in a.items():
            f.write("| {} | {:.0%} | {:.4f} | {:.4f} | {}× |\n".format(
                layer, d["leak_acc"], d["early_gap"], d["full_gap"], d["ratio"]))

        # Exp B
        f.write("\n## Exp B: Tutor Sensitivity (full_fix layer)\n\n")
        f.write("| Condition | SBCR |\n|-----------|------|\n")
        for name, sbcr in b.items():
            f.write("| {} | {:.0%} |\n".format(name, sbcr))

        oracle_lift = b.get("oracle_warn+branch", 0) - b.get("no_tutor+branch", 0)
        rsa_lift = b.get("rsa_warn+branch", 0) - b.get("no_tutor+branch", 0)
        f.write("\n### Key Comparisons\n")
        f.write("- Oracle lift: {:+.0%}\n".format(oracle_lift))
        f.write("- RSA lift: {:+.0%}\n".format(rsa_lift))

        # Exp C
        f.write("\n## Exp C: Training-Probe Loop\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["no_tutor+old", "no_tutor+branch", "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in c if r["condition"] == cond]
            vals = " | ".join("{:.0%}".format(r["SBCR"]) for r in rows)
            f.write("| {} | {} |\n".format(cond, vals))

        f.write("\n### Learning Gain LG(k)\n\n")
        for cond in ["no_tutor+branch", "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in c if r["condition"] == cond]
            k0 = rows[0]["SBCR"]
            parts = []
            for r in rows[1:]:
                parts.append("k={}:{:+.0%}".format(r["k"], r["SBCR"] - k0))
            f.write("- **{}**: {}\n".format(cond, ", ".join(parts)))

    print("Report → results/po_leakage_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
