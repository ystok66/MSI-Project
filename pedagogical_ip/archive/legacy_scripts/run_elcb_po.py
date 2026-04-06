"""ELCB-PO Experiments v2: With Proper Partial Observability.

Key fix: at decision time, branch planner only sees early cells (depth < reveal_depth).
Tutor/oracle can see full branch and provide informative warnings.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch, branch_cell_cost
from src.teachers.rsa_warning_v2 import RSAWarningV2

FAM = "elcb_po"
DIFF = "medium"
out = Path("results")
out.mkdir(exist_ok=True)


def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def get_visible_cells(sc, full=False):
    """Return branch cells visible at decision time.

    If full=False (agent perspective): only early cells (depth < reveal_depth)
    If full=True (oracle/tutor perspective): all cells
    """
    rd = sc.reveal_depth if not full else sc.branch_len
    a_vis = sc.branch_a_cells[:rd]
    b_vis = sc.branch_b_cells[:rd]
    return a_vis, b_vis


def make_candidates_po(sc, visible_a, visible_b):
    """Build BranchCandidates from partially visible cells."""
    fork = sc.fork_cell
    merge = sc.merge_cell
    return [
        BranchCandidate(0, list(visible_a), len(visible_a),
                        fork, merge, (1, fork[1]), (1, merge[1])),
        BranchCandidate(1, list(visible_b), len(visible_b),
                        fork, merge, (3, fork[1]), (3, merge[1])),
    ]


def train_on_seeds(seeds, lp=None, lib=None, scorer=None,
                    use_rsa=False, oracle_warn=False, full_visibility=False):
    """Train V5 models. Training always sees full branch (supervised)."""
    if lp is None:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    if lib is None:
        lib = BranchConceptLibrary()
    if scorer is None:
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    rsa = RSAWarningV2()

    for seed in seeds:
        gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)
        ww = meta.world_weights
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)

        # Supervised cell-level training (agent visits and observes)
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        # Concept/scorer training uses FULL branch (post-episode learning)
        s_s = summarize_branch(sc.safe_cells, fb, fv, lp)
        s_r = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", s_s)
        lib.update("risky_branch", s_r)
        scorer.update(build_scorer_input(s_s, lib), 1.0)
        scorer.update(build_scorer_input(s_r, lib), 0.0)

        # Oracle warning: extra supervised signal on risky cells
        if oracle_warn:
            for r, c in sc.risky_cells:
                z2 = fb[r, c]
                lp.update_from_outcome(z2, ww.true_cost(z2), ww.true_risk(z2), weight=1.0)
            # Also update concept library with oracle-informed full summaries
            s_s2 = summarize_branch(sc.safe_cells, fb, fv, lp)
            s_r2 = summarize_branch(sc.risky_cells, fb, fv, lp)
            lib.update("safe_branch", s_s2)
            lib.update("risky_branch", s_r2)
            scorer.update(build_scorer_input(s_s2, lib), 1.0)
            scorer.update(build_scorer_input(s_r2, lib), 0.0)
        elif use_rsa:
            # RSA: speaker tells agent about risk based on FULL branch view
            s_a_full = summarize_branch(sc.branch_a_cells, fb, fv, lp)
            s_b_full = summarize_branch(sc.branch_b_cells, fb, fv, lp)
            risk_a_full = s_a_full[0]  # mean_risk from full branch
            risk_b_full = s_b_full[0]
            z_state = rsa.classify_risk_state(
                risk_a_full, risk_b_full, branch_a_side="left",
                gap_threshold=0.01)  # lower threshold for PO
            u_idx, u_name = rsa.choose_utterance(z_state)
            if u_name != "silence":
                # Warning causes extra learning on risky cells
                for r, c in sc.risky_cells:
                    z2 = fb[r, c]
                    lp.update_from_outcome(z2, ww.true_cost(z2),
                                           ww.true_risk(z2), weight=0.5)

    return lp, lib, scorer


def probe_sbcr(probe_seeds, lp, lib, scorer, use_branch=True):
    """Probe: planner only sees VISIBLE cells (partial observability)."""
    safe = 0
    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)
        passable = make_passable(gm)
        tie_rng = np.random.default_rng(ps + 777)

        # Agent only sees early cells at fork decision
        vis_a, vis_b = get_visible_cells(sc, full=False)
        candidates = make_candidates_po(sc, vis_a, vis_b)

        if use_branch:
            best, _ = choose_branch(
                candidates, fb, fv, lp, passable, lib, scorer,
                lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
            safe += int(best.branch_id == sc.oracle_safe_branch_id)
        else:
            # Old planner: only uses cell costs from visible cells
            ca = sum(lp.predict_cost(fb[r, c]) for r, c in vis_a)
            cb = sum(lp.predict_cost(fb[r, c]) for r, c in vis_b)
            margin = cb - ca
            if abs(margin) < 1e-4:
                chosen = int(tie_rng.integers(0, 2))
            else:
                chosen = 0 if ca < cb else 1
            safe += int(chosen == sc.oracle_safe_branch_id)

    return round(safe / len(probe_seeds), 3)


# ═══════════════════════════════════════════════════════════════
# Step 2: Sanity
# ═══════════════════════════════════════════════════════════════
def step2():
    print("Step 2: Sanity", file=sys.stderr)
    weak_gaps, strong_gaps = [], []
    for seed in range(20):
        gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)
        fb = meta.cell_features
        # Check visibility gap
        vis_a, vis_b = get_visible_cells(sc, full=False)
        full_a, full_b = get_visible_cells(sc, full=True)

        # Visible-only summary gap
        fv = np.full_like(fb, 0.3)
        lp_temp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        ww = meta.world_weights
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = fb[r, c]
                lp_temp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        s_vis_a = summarize_branch(list(vis_a), fb, fv, lp_temp)
        s_vis_b = summarize_branch(list(vis_b), fb, fv, lp_temp)
        s_full_a = summarize_branch(list(full_a), fb, fv, lp_temp)
        s_full_b = summarize_branch(list(full_b), fb, fv, lp_temp)

        # Mean risk gap
        vis_risk_gap = abs(s_vis_a[0] - s_vis_b[0])
        full_risk_gap = abs(s_full_a[0] - s_full_b[0])
        weak_gaps.append(vis_risk_gap)
        strong_gaps.append(full_risk_gap)

    return {
        "mean_vis_gap": round(float(np.mean(weak_gaps)), 4),
        "mean_full_gap": round(float(np.mean(strong_gaps)), 4),
        "ratio": round(float(np.mean(strong_gaps)) / max(float(np.mean(weak_gaps)), 1e-6), 1),
    }


# ═══════════════════════════════════════════════════════════════
# Step 3: Tutor Value Audit
# ═══════════════════════════════════════════════════════════════
def step3():
    print("Step 3: Tutor Audit", file=sys.stderr)
    train_seeds = list(range(40))
    probe_seeds = list(range(100, 130))

    configs = {
        "no_tutor+old":        (False, False, False),
        "rsa_warn+old":        (False, True,  False),
        "no_tutor+branch":     (True,  False, False),
        "rsa_warn+branch":     (True,  True,  False),
        "oracle_warn+branch":  (True,  False, True),
    }

    results = {}
    for name, (use_branch, use_rsa, oracle_warn) in configs.items():
        lp, lib, scorer = train_on_seeds(
            train_seeds, use_rsa=use_rsa, oracle_warn=oracle_warn)
        sbcr = probe_sbcr(probe_seeds, lp, lib, scorer, use_branch=use_branch)
        results[name] = sbcr
    return results


# ═══════════════════════════════════════════════════════════════
# Step 4: Training-Probe Loop
# ═══════════════════════════════════════════════════════════════
def step4():
    print("Step 4: Train-Probe", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))

    configs = {
        "no_tutor+old":       (False, False, False),
        "rsa_warn+old":       (False, True,  False),
        "no_tutor+branch":    (True,  False, False),
        "rsa_warn+branch":    (True,  True,  False),
        "oracle_warn+branch": (True,  False, True),
    }

    results = []
    for cond, (use_branch, use_rsa, oracle_warn) in configs.items():
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        k_done = 0

        for k_target in K_VALUES:
            batch = list(range(k_done, k_target))
            if batch:
                lp, lib, scorer = train_on_seeds(
                    batch, lp=lp, lib=lib, scorer=scorer,
                    use_rsa=use_rsa, oracle_warn=oracle_warn)
            k_done = k_target

            sbcr = probe_sbcr(PROBE_SEEDS, lp, lib, scorer, use_branch=use_branch)
            results.append({"condition": cond, "k": k_target, "SBCR": sbcr})

    return results


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    s2 = step2()
    s3 = step3()
    s4 = step4()

    with open(out / "elcb_po_report.md", "w") as f:
        f.write("# ELCB-PO Results (v2 — proper partial observability)\n\n")

        f.write("## Step 2: Sanity\n\n")
        f.write("- **Visible-only mean risk gap**: {}\n".format(s2["mean_vis_gap"]))
        f.write("- **Full-branch mean risk gap**: {}\n".format(s2["mean_full_gap"]))
        f.write("- **Full/Visible ratio**: {}×\n\n".format(s2["ratio"]))

        f.write("## Step 3: Tutor Value Audit (40 train, 30 probe)\n\n")
        f.write("| Condition | SBCR |\n|-----------|------|\n")
        for name, sbcr in s3.items():
            f.write("| {} | {:.0%} |\n".format(name, sbcr))

        ohg_old = s3.get("rsa_warn+old", 0) - s3.get("no_tutor+old", 0)
        ohg_br = s3.get("rsa_warn+branch", 0) - s3.get("no_tutor+branch", 0)
        orc = s3.get("oracle_warn+branch", 0) - s3.get("no_tutor+branch", 0)
        f.write("\n### Decomposition\n")
        f.write("- OHG (old planner): {:+.0%}\n".format(ohg_old))
        f.write("- OHG (branch planner): {:+.0%}\n".format(ohg_br))
        f.write("- Oracle lift: {:+.0%}\n".format(orc))

        f.write("\n## Step 4: Training-Probe Loop\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["no_tutor+old", "rsa_warn+old", "no_tutor+branch",
                      "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in s4 if r["condition"] == cond]
            vals = " | ".join("{:.0%}".format(r["SBCR"]) for r in rows)
            f.write("| {} | {} |\n".format(cond, vals))

        f.write("\n### Learning Gain LG(k)\n\n")
        for cond in ["no_tutor+branch", "rsa_warn+branch", "oracle_warn+branch"]:
            rows = [r for r in s4 if r["condition"] == cond]
            k0 = rows[0]["SBCR"]
            parts = []
            for r in rows[1:]:
                parts.append("k={}:{:+.0%}".format(r["k"], r["SBCR"] - k0))
            f.write("- **{}**: {}\n".format(cond, ", ".join(parts)))

    print("Report → results/elcb_po_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
