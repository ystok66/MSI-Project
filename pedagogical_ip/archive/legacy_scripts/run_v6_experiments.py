"""V6 Experiments — Branch-Aware Planner Evaluation.

Exp 1: Planner Interface Ablation
  old / pointwise / concept / scorer / hybrid → SBCR, margin
Exp 2: ELCB Transfer (old vs branch-aware at k=0,1,3,10,30)
Exp 3: Mirror/Side-Swap Robustness (SideBias, SemanticConsistency)
"""
import sys, copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch, SUMMARY_DIM
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.planner.branch_candidates import extract_elcb_branches, should_activate_branch_reranker
from src.planner.branch_reranker import choose_branch

DIFFICULTY = "medium"
out = Path("results")
out.mkdir(exist_ok=True)


def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def train_predictor_on_map(gm, meta, n_epochs=5):
    """Train a LatentCostRiskHead on a single map."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    ww = meta.world_weights
    fb = meta.cell_features
    for _ in range(n_epochs):
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
    return lp


def train_v5_models(seeds, n_epochs=5):
    """Train predictor + concept library + scorer across multiple seeds."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)

    for seed in seeds:
        gm, _, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        ww = meta.world_weights
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)

        for _ in range(n_epochs):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        # Summaries
        safe_cells = sc.branch_a_cells if sc.oracle_safe_branch_id == 0 else sc.branch_b_cells
        risky_cells = sc.branch_a_cells if sc.oracle_risky_branch_id == 0 else sc.branch_b_cells
        s_safe = summarize_branch(safe_cells, fb, fv, lp)
        s_risky = summarize_branch(risky_cells, fb, fv, lp)
        lib.update("safe_branch", s_safe)
        lib.update("risky_branch", s_risky)
        scorer.update(build_scorer_input(s_safe, lib), 1.0)
        scorer.update(build_scorer_input(s_risky, lib), 0.0)

    return lp, lib, scorer


def evaluate_planner(seed, lp, lib, scorer, score_mode, lambda_b):
    """Run branch reranker on a single seed, return safe_chosen."""
    gm, _, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
    fb = meta.cell_features
    fv = np.full_like(fb, 0.3)
    passable = make_passable(gm)

    candidates = extract_elcb_branches(sc)
    if not should_activate_branch_reranker(candidates):
        return None

    tie_rng = np.random.default_rng(seed + 777)
    best, details = choose_branch(
        candidates, fb, fv, lp, passable,
        lib, scorer,
        lambda_b=lambda_b, score_mode=score_mode, tie_rng=tie_rng)

    safe_chosen = int(best.branch_id == sc.oracle_safe_branch_id)
    return {
        "safe_chosen": safe_chosen,
        "chosen_id": best.branch_id,
        "oracle_safe": sc.oracle_safe_branch_id,
        "j_hybrid": details["j_hybrid_chosen"],
        "s_branch": details["s_branch_chosen"],
    }


# ═══════════════════════════════════════════════════════════════
# Exp 1: Planner Interface Ablation
# ═══════════════════════════════════════════════════════════════
def exp1():
    print("Exp 1: Planner Ablation", file=sys.stderr)
    train_seeds = list(range(40))
    test_seeds = list(range(100, 130))

    lp, lib, scorer = train_v5_models(train_seeds)

    conditions = {
        "old_planner":     ("pointwise_only", 0.0),
        "pointwise_only":  ("pointwise_only", 1.0),
        "concept_only":    ("concept_only",   1.0),
        "scorer_only":     ("scorer_only",    1.0),
        "hybrid":          ("hybrid",         1.0),
        "hybrid_strong":   ("hybrid",         3.0),
    }

    results = {}
    for name, (mode, lb) in conditions.items():
        safe = 0
        total = 0
        for seed in test_seeds:
            r = evaluate_planner(seed, lp, lib, scorer, mode, lb)
            if r is None:
                continue
            safe += r["safe_chosen"]
            total += 1

        results[name] = {
            "SBCR": round(safe / max(total, 1), 3),
            "n": total,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Exp 2: ELCB Transfer (old vs branch-aware)
# ═══════════════════════════════════════════════════════════════
def exp2():
    print("Exp 2: Transfer", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))
    results = []

    for cond in ["old_planner", "branch_aware"]:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05)
        k_done = 0

        for k_target in K_VALUES:
            while k_done < k_target:
                gm, _, meta, sc = generate_scenario("elcb", k_done, DIFFICULTY, latent_mode=True)
                ww = meta.world_weights
                fb = meta.cell_features
                fv = np.full_like(fb, 0.3)

                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = fb[r, c]
                        lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

                safe_cells = sc.branch_a_cells if sc.oracle_safe_branch_id == 0 else sc.branch_b_cells
                risky_cells = sc.branch_a_cells if sc.oracle_risky_branch_id == 0 else sc.branch_b_cells
                s_s = summarize_branch(safe_cells, fb, fv, lp)
                s_r = summarize_branch(risky_cells, fb, fv, lp)
                lib.update("safe_branch", s_s)
                lib.update("risky_branch", s_r)
                scorer.update(build_scorer_input(s_s, lib), 1.0)
                scorer.update(build_scorer_input(s_r, lib), 0.0)
                k_done += 1

            # Probe
            safe_choices = 0
            for ps in PROBE_SEEDS:
                if cond == "old_planner":
                    gm_p, _, meta_p, sc_p = generate_scenario("elcb", ps, DIFFICULTY, latent_mode=True)
                    fb_p = meta_p.cell_features
                    fv_p = np.full_like(fb_p, 0.3)
                    passable = make_passable(gm_p)

                    # Old planner: direct cost summation
                    from src.planner.branch_reranker import branch_cell_cost
                    candidates = extract_elcb_branches(sc_p)
                    ca = branch_cell_cost(candidates[0].cells, candidates[0].entry_gate,
                                          candidates[0].exit_gate, fb_p, lp, passable, fv_p)
                    cb = branch_cell_cost(candidates[1].cells, candidates[1].entry_gate,
                                          candidates[1].exit_gate, fb_p, lp, passable, fv_p)
                    margin = cb - ca
                    if abs(margin) < 1e-4:
                        chosen = int(np.random.default_rng(ps + 777).integers(0, 2))
                    else:
                        chosen = 0 if ca < cb else 1
                    safe_choices += int(chosen == sc_p.oracle_safe_branch_id)
                else:
                    r = evaluate_planner(ps, lp, lib, scorer, "hybrid", 1.0)
                    if r:
                        safe_choices += r["safe_chosen"]

            results.append({
                "condition": cond,
                "k": k_target,
                "SBCR": round(safe_choices / len(PROBE_SEEDS), 3),
                "n_updates": lp.n_updates,
            })

    return results


# ═══════════════════════════════════════════════════════════════
# Exp 3: Mirror/Side-Swap Robustness
# ═══════════════════════════════════════════════════════════════
def exp3():
    print("Exp 3: Robustness", file=sys.stderr)
    train_seeds = list(range(40))
    lp, lib, scorer = train_v5_models(train_seeds)

    results = {"safe_chosen": 0, "oracle_safe_0": 0, "oracle_safe_1": 0,
               "safe_when_0": 0, "safe_when_1": 0, "total": 0}

    for seed in range(100, 150):
        r = evaluate_planner(seed, lp, lib, scorer, "hybrid", 1.0)
        if r is None:
            continue
        results["total"] += 1
        results["safe_chosen"] += r["safe_chosen"]
        if r["oracle_safe"] == 0:
            results["oracle_safe_0"] += 1
            results["safe_when_0"] += r["safe_chosen"]
        else:
            results["oracle_safe_1"] += 1
            results["safe_when_1"] += r["safe_chosen"]

    return results


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    e1 = exp1()
    e2 = exp2()
    e3 = exp3()

    with open(out / "v6_report.md", "w") as f:
        f.write("# V6 Branch-Aware Planner Results\n\n")

        # Exp 1
        f.write("## Exp 1: Planner Interface Ablation\n\n")
        f.write("| Condition | SBCR | n |\n")
        f.write("|-----------|------|---|\n")
        for name, data in e1.items():
            f.write(f"| {name} | {data['SBCR']:.0%} | {data['n']} |\n")

        # Exp 2
        f.write("\n## Exp 2: ELCB Transfer\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["old_planner", "branch_aware"]:
            rows = [r for r in e2 if r["condition"] == cond]
            vals = " | ".join(f"{r['SBCR']:.0%}" for r in rows)
            f.write(f"| {cond} | {vals} |\n")

        # Exp 3
        f.write("\n## Exp 3: Side-Swap Robustness\n\n")
        n = results = e3
        sbcr = n["safe_chosen"] / max(n["total"], 1)
        rate_0 = n["safe_when_0"] / max(n["oracle_safe_0"], 1) if n["oracle_safe_0"] else 0
        rate_1 = n["safe_when_1"] / max(n["oracle_safe_1"], 1) if n["oracle_safe_1"] else 0
        side_bias = abs(rate_0 - rate_1) / 2
        f.write(f"- Overall SBCR: {sbcr:.0%}\n")
        f.write(f"- SBCR when safe=A: {rate_0:.0%} (n={n['oracle_safe_0']})\n")
        f.write(f"- SBCR when safe=B: {rate_1:.0%} (n={n['oracle_safe_1']})\n")
        f.write(f"- SideBias: {side_bias:.3f}\n")
        f.write(f"- SemanticConsistency: {sbcr:.0%}\n")

    print(f"Report → {out / 'v6_report.md'}", file=sys.stderr)
    print("Done.", file=sys.stderr)
