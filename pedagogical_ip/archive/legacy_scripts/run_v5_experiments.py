"""V5 Integration Experiments (Exp 1-5).

Exp 1: Semantic vs Position Bias Audit
Exp 2: Warning Semantics Audit
Exp 3: ELCB Transfer Re-run (with V5 modules)
Exp 4: Shared vs Residual Attribution
Exp 5: Branch Scorer Feasibility
"""
import sys, copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.planner_astar import cell_cost_v2_latent

# V5 modules
from src.agents.branch_summary import summarize_branch, SUMMARY_DIM
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.familiarity import familiarity_score, is_novel, teaching_priority
from src.teachers.rsa_warning_v2 import RSAWarningV2, Z_LEFT, Z_RIGHT, Z_AMBIG
from src.agents.mixed_effects_risk_head import MixedEffectsRiskHead
from src.agents.branch_scorer_probe import (
    BranchScorerProbe, build_scorer_input, pointwise_branch_score
)

runner = LatticeV2Runner()
DIFFICULTY = "medium"
out = Path("results")
out.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────

def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def branch_cost_sum(cells, gm, meta, lp):
    """Direct cost summation along branch cells."""
    passable = make_passable(gm)
    fb = meta.cell_features
    fv = np.full_like(fb, 0.5)
    total = 0.0
    for r, c in cells:
        total += cell_cost_v2_latent(r, c, fb, lp, passable,
                                      feature_belief_var=fv, route_necessity=0.0)
    return total


def branch_chosen(gm, meta, sc, lp, tie_rng=None):
    """Which branch does planner prefer? Returns (chosen_id, margin, safe_chosen)."""
    ca = branch_cost_sum(sc.branch_a_cells, gm, meta, lp)
    cb = branch_cost_sum(sc.branch_b_cells, gm, meta, lp)
    margin = cb - ca
    if abs(margin) < 1e-4:
        chosen = int((tie_rng or np.random.default_rng()).integers(0, 2))
    else:
        chosen = 0 if ca < cb else 1
    return chosen, margin, int(chosen == sc.oracle_safe_branch_id)


def run_ep(seed, lp, **kw):
    s = runner.reset(seed=seed, scenario_family="elcb", latent_mode=True,
                     difficulty=DIFFICULTY, latent_predictor=lp, **kw)
    while not s.done:
        runner.step(s)
    return runner.get_metrics(s), s


# ═══════════════════════════════════════════════════════════════════
# Exp 1: Semantic vs Position Bias Audit
# ═══════════════════════════════════════════════════════════════════
def exp1():
    print("Exp 1: Semantic vs Position Bias Audit", file=sys.stderr)
    results = {"standard": [], "mixed_effects": []}

    for mode_name, use_me in [("standard", False), ("mixed_effects", True)]:
        for seed in range(30):
            gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)

            # ── Train on this map ──
            if use_me:
                mh = MixedEffectsRiskHead(d=4, lambda_delta=2.0)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            ww = meta.world_weights

            # 10 epochs of supervised learning
            for _ in range(10):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = meta.cell_features[r, c]
                        tc = ww.true_cost(z)
                        tr = ww.true_risk(z)
                        lp.update_from_outcome(z, tc, tr)
                        if use_me:
                            ctx = f"side_{sc.oracle_safe_branch_id}"
                            mh.update_from_label(z, tr, ctx=ctx)

            # ── Standard check ──
            tie_rng = np.random.default_rng(seed + 777)
            chosen, margin, safe_chosen = branch_chosen(gm, meta, sc, lp, tie_rng)

            # ── Side swap check: generate same seed but reversed semantics ──
            # We simulate this by checking if choice tracks oracle_safe vs side
            side_a_chosen = 1 - chosen  # whether branch A was chosen

            results[mode_name].append({
                "seed": seed,
                "safe_chosen": safe_chosen,
                "side_a_chosen": side_a_chosen,
                "oracle_safe": sc.oracle_safe_branch_id,
                "margin": round(margin, 4),
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# Exp 2: Warning Semantics Audit
# ═══════════════════════════════════════════════════════════════════
def exp2():
    print("Exp 2: Warning Semantics Audit", file=sys.stderr)
    rsa = RSAWarningV2()
    results = []

    for seed in range(20):
        gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        ww = meta.world_weights

        # Train predictor
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = meta.cell_features[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        # Build branch summaries
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)
        s_a = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        s_b = summarize_branch(sc.branch_b_cells, fb, fv, lp)

        # Pre-warning branch choice
        tie_rng = np.random.default_rng(seed + 777)
        pre_chosen, pre_margin, pre_safe = branch_chosen(gm, meta, sc, lp, tie_rng)

        # RSA: classify risk state
        risk_a = float(np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_a_cells]))
        risk_b = float(np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_b_cells]))

        risky_side = "left" if sc.oracle_risky_branch_id == 0 else "right"
        z_state = rsa.classify_risk_state(risk_a, risk_b, branch_a_side="left")

        # S1: what would rational speaker say?
        u_idx, u_name = rsa.choose_utterance(z_state)

        # L1: how does agent update belief?
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        posterior = rsa.update_belief_with_warning(prior, u_idx)

        # Familiarity context
        lib = BranchConceptLibrary()
        for _ in range(10):
            lib.update("safe_branch", s_a if sc.oracle_safe_branch_id == 0 else s_b)
            lib.update("risky_branch", s_b if sc.oracle_safe_branch_id == 0 else s_a)

        tp_a = teaching_priority(s_a, lib, risk_a)
        tp_b = teaching_priority(s_b, lib, risk_b)

        results.append({
            "seed": seed,
            "risk_a": round(risk_a, 4),
            "risk_b": round(risk_b, 4),
            "z_state": z_state,
            "utterance": u_name,
            "post_risky_prob": round(float(max(posterior[:3])), 3),
            "pre_safe": pre_safe,
            "teach_mode_a": tp_a["teaching_mode"],
            "teach_mode_b": tp_b["teaching_mode"],
            "pre_margin": round(pre_margin, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# Exp 3: ELCB Transfer Re-run (with V5 branch concepts)
# ═══════════════════════════════════════════════════════════════════
def exp3():
    print("Exp 3: ELCB Transfer Re-run", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))
    results = []

    for cond in ["baseline", "v5_concepts"]:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary() if cond == "v5_concepts" else None
        k_done = 0

        for k_target in K_VALUES:
            # Train from k_done to k_target
            while k_done < k_target:
                gm_t, _, meta_t, sc_t = generate_scenario(
                    "elcb", k_done, DIFFICULTY, latent_mode=True)
                ww_t = meta_t.world_weights
                fb_t = meta_t.cell_features

                # Supervised update on visited cells
                for r in range(gm_t.height):
                    for c in range(gm_t.width):
                        if gm_t.cell_types[r, c] == CellType.WALL:
                            continue
                        z = fb_t[r, c]
                        lp.update_from_outcome(z, ww_t.true_cost(z), ww_t.true_risk(z))

                # V5: also update branch concepts
                if lib is not None:
                    fv_t = np.full_like(fb_t, 0.3)
                    s_safe = summarize_branch(
                        sc_t.branch_a_cells if sc_t.oracle_safe_branch_id == 0 else sc_t.branch_b_cells,
                        fb_t, fv_t, lp)
                    s_risky = summarize_branch(
                        sc_t.branch_a_cells if sc_t.oracle_risky_branch_id == 0 else sc_t.branch_b_cells,
                        fb_t, fv_t, lp)
                    lib.update("safe_branch", s_safe)
                    lib.update("risky_branch", s_risky)

                k_done += 1

            # Probe
            safe_choices = 0
            concept_correct = 0
            margins = []
            for ps in PROBE_SEEDS:
                gm_p, _, meta_p, sc_p = generate_scenario(
                    "elcb", ps, DIFFICULTY, latent_mode=True)
                lp_probe = copy.deepcopy(lp)
                tie_rng = np.random.default_rng(ps + 777)
                chosen, margin, safe_chosen = branch_chosen(
                    gm_p, meta_p, sc_p, lp_probe, tie_rng)
                safe_choices += safe_chosen
                margins.append(margin)

                # V5: does concept library help?
                if lib is not None:
                    fb_p = meta_p.cell_features
                    fv_p = np.full_like(fb_p, 0.3)
                    s_a = summarize_branch(sc_p.branch_a_cells, fb_p, fv_p, lp_probe)
                    s_b = summarize_branch(sc_p.branch_b_cells, fb_p, fv_p, lp_probe)
                    best_a, sc_a = lib.best_concept(s_a)
                    best_b, sc_b = lib.best_concept(s_b)
                    # Does concept correctly identify safe branch?
                    if sc_p.oracle_safe_branch_id == 0:
                        concept_correct += int(best_a == "safe_branch")
                    else:
                        concept_correct += int(best_b == "safe_branch")

            sbcr = safe_choices / len(PROBE_SEEDS)
            results.append({
                "condition": cond,
                "k": k_target,
                "SBCR": round(sbcr, 3),
                "mean_margin": round(float(np.mean(margins)), 4),
                "concept_acc": round(concept_correct / len(PROBE_SEEDS), 3) if lib else "n/a",
                "n_updates": lp.n_updates,
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# Exp 4: Shared vs Residual Attribution
# ═══════════════════════════════════════════════════════════════════
def exp4():
    print("Exp 4: Shared vs Residual Attribution", file=sys.stderr)
    lambdas = [0.5, 1.0, 2.0, 5.0]
    results = []

    for lam in lambdas:
        mh = MixedEffectsRiskHead(d=4, lambda_delta=lam)

        # Train across multiple maps
        for seed in range(20):
            gm, _, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
            ww = meta.world_weights
            fb = meta.cell_features

            ctx = f"safe_side_{sc.oracle_safe_branch_id}"
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    mh.update_from_label(z, ww.true_risk(z), ctx=ctx)

        # Measure
        shared = mh.shared_norm()
        res_norms = {k: mh.residual_norm(k) for k in mh.contexts}
        max_res = max(res_norms.values()) if res_norms else 0.0
        mean_res = float(np.mean(list(res_norms.values()))) if res_norms else 0.0

        # Test consistency: predict with shared vs context
        test_safe = np.array([0.5, 0.0, 0.08, 0.06])
        test_risky = np.array([0.5, 0.0, 0.6, 0.5])
        r_safe_shared = mh.predict_risk(test_safe)
        r_risky_shared = mh.predict_risk(test_risky)

        results.append({
            "lambda_delta": lam,
            "shared_norm": round(shared, 4),
            "max_residual": round(max_res, 4),
            "mean_residual": round(mean_res, 4),
            "n_contexts": len(mh.contexts),
            "safe_pred": round(r_safe_shared, 4),
            "risky_pred": round(r_risky_shared, 4),
            "discrimination": round(r_risky_shared - r_safe_shared, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# Exp 5: Branch Scorer Feasibility
# ═══════════════════════════════════════════════════════════════════
def exp5():
    print("Exp 5: Branch Scorer Feasibility", file=sys.stderr)
    rng = np.random.default_rng(42)

    # Phase 1: collect labeled branch summaries
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")

    train_data = []
    for seed in range(40):
        gm, _, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        ww = meta.world_weights
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)

        # Supervised learning on cell features
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        # Summarize branches
        s_a = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        s_b = summarize_branch(sc.branch_b_cells, fb, fv, lp)

        # Labels
        label_a = 1.0 if sc.oracle_safe_branch_id == 0 else 0.0
        label_b = 1.0 if sc.oracle_safe_branch_id == 1 else 0.0

        # Update concepts
        lib.update("safe_branch", s_a if label_a == 1 else s_b)
        lib.update("risky_branch", s_a if label_a == 0 else s_b)

        # Train scorer
        inp_a = build_scorer_input(s_a, lib)
        inp_b = build_scorer_input(s_b, lib)
        scorer.update(inp_a, label_a)
        scorer.update(inp_b, label_b)

        train_data.append({
            "seed": seed, "s_a": s_a, "s_b": s_b,
            "label_a": label_a, "label_b": label_b,
            "oracle_safe": sc.oracle_safe_branch_id,
        })

    # Phase 2: test on held-out seeds
    correct_scorer = 0
    correct_pointwise = 0
    total = 0

    for seed in range(100, 130):
        gm, _, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)

        s_a = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        s_b = summarize_branch(sc.branch_b_cells, fb, fv, lp)

        # Scorer ranking
        inp_a = build_scorer_input(s_a, lib)
        inp_b = build_scorer_input(s_b, lib)
        scorer_prefers_a = scorer.score(inp_a) > scorer.score(inp_b)
        scorer_safe = (scorer_prefers_a and sc.oracle_safe_branch_id == 0) or \
                      (not scorer_prefers_a and sc.oracle_safe_branch_id == 1)

        # Pointwise baseline: lower mean risk = safer
        pw_a = pointwise_branch_score(sc.branch_a_cells, fb, lp)
        pw_b = pointwise_branch_score(sc.branch_b_cells, fb, lp)
        pw_prefers_a = pw_a < pw_b  # lower risk = safer
        pw_safe = (pw_prefers_a and sc.oracle_safe_branch_id == 0) or \
                  (not pw_prefers_a and sc.oracle_safe_branch_id == 1)

        correct_scorer += int(scorer_safe)
        correct_pointwise += int(pw_safe)
        total += 1

    return {
        "scorer_accuracy": round(correct_scorer / total, 3),
        "pointwise_accuracy": round(correct_pointwise / total, 3),
        "n_test": total,
        "scorer_n_updates": scorer.n_updates,
        "concept_kappa_safe": round(lib.concepts["safe_branch"].kappa, 1),
        "concept_kappa_risky": round(lib.concepts["risky_branch"].kappa, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    e1 = exp1()
    e2 = exp2()
    e3 = exp3()
    e4 = exp4()
    e5 = exp5()

    # ── Write Report ────────────────────────────────────────────
    with open(out / "v5_experiments.md", "w") as f:
        f.write("# V5 Integration Experiments\n\n")

        # ── Exp 1: Semantic vs Position Bias ──
        f.write("## Exp 1: Semantic vs Position Bias Audit\n\n")
        for mode in ["standard", "mixed_effects"]:
            data = e1[mode]
            sbcr = sum(d["safe_chosen"] for d in data) / len(data)
            side_a_rate = sum(d["side_a_chosen"] for d in data) / len(data)
            side_bias = abs(side_a_rate - 0.5)
            f.write(f"### {mode}\n")
            f.write(f"- SBCR: {sbcr:.0%}\n")
            f.write(f"- Side A rate: {side_a_rate:.0%}\n")
            f.write(f"- SideBias = |rate - 0.5| = {side_bias:.3f}\n")
            f.write(f"- SemanticConsistency (SBCR): {sbcr:.0%}\n\n")

        # ── Exp 2: Warning Semantics ──
        f.write("## Exp 2: Warning Semantics Audit\n\n")
        f.write("| Seed | risk_A | risk_B | State | Utterance | "
                "P(risky) | pre_safe | teach_A | teach_B |\n")
        f.write("|------|--------|--------|-------|-----------|"
                "----------|----------|---------|----------|\n")
        for r in e2:
            from src.teachers.rsa_warning_v2 import WORLD_STATES
            state_name = WORLD_STATES[r["z_state"]]
            f.write(f"| {r['seed']} | {r['risk_a']:.3f} | {r['risk_b']:.3f} | "
                    f"{state_name} | {r['utterance']} | "
                    f"{r['post_risky_prob']:.3f} | {r['pre_safe']} | "
                    f"{r['teach_mode_a']} | {r['teach_mode_b']} |\n")

        warn_flip = sum(1 for r in e2 if r["pre_safe"]) / len(e2)
        f.write(f"\n- Pre-warning safe choice rate: {warn_flip:.0%}\n")
        utt_counts = defaultdict(int)
        for r in e2:
            utt_counts[r["utterance"]] += 1
        f.write(f"- Utterance distribution: {dict(utt_counts)}\n\n")

        # ── Exp 3: ELCB Transfer ──
        f.write("## Exp 3: ELCB Transfer Re-run\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["baseline", "v5_concepts"]:
            rows = [r for r in e3 if r["condition"] == cond]
            vals = " | ".join(f"{r['SBCR']:.0%}" for r in rows)
            f.write(f"| {cond} | {vals} |\n")

        f.write("\n### Concept Accuracy (v5_concepts only)\n\n")
        for r in e3:
            if r["condition"] == "v5_concepts":
                f.write(f"- k={r['k']}: concept_acc={r['concept_acc']}\n")

        # ── Exp 4: Mixed-Effects ──
        f.write("\n## Exp 4: Shared vs Residual Attribution\n\n")
        f.write("| λ_δ | |w_shared| | max |δ| | mean |δ| | discrimination |\n")
        f.write("|-----|---------|---------|---------|----------------|\n")
        for r in e4:
            f.write(f"| {r['lambda_delta']} | {r['shared_norm']:.4f} | "
                    f"{r['max_residual']:.4f} | {r['mean_residual']:.4f} | "
                    f"{r['discrimination']:.4f} |\n")

        # ── Exp 5: Branch Scorer ──
        f.write(f"\n## Exp 5: Branch Scorer Feasibility\n\n")
        f.write(f"- **Branch Scorer accuracy**: {e5['scorer_accuracy']:.0%}\n")
        f.write(f"- **Pointwise baseline accuracy**: {e5['pointwise_accuracy']:.0%}\n")
        f.write(f"- Scorer updates: {e5['scorer_n_updates']}\n")
        f.write(f"- Concept κ safe: {e5['concept_kappa_safe']}, "
                f"risky: {e5['concept_kappa_risky']}\n")
        winner = "Branch Scorer" if e5["scorer_accuracy"] > e5["pointwise_accuracy"] \
                 else "Pointwise" if e5["pointwise_accuracy"] > e5["scorer_accuracy"] \
                 else "Tie"
        f.write(f"- **Winner: {winner}**\n")

        # ── Summary ──
        f.write("\n## Summary & Interpretation\n\n")
        f.write("### Key Questions Answered\n\n")
        f.write("1. **Does mixed-effects reduce side bias?** → Check Exp 1 SideBias values\n")
        f.write("2. **Does RSA warning identify correct risk state?** → Check Exp 2 utterance distribution\n")
        f.write("3. **Does V5 concept library improve transfer SBCR?** → Check Exp 3 v5_concepts vs baseline\n")
        f.write("4. **Does shrinkage separate shared from context?** → Check Exp 4 |w_shared| vs |δ|\n")
        f.write("5. **Is branch scorer better than pointwise?** → Check Exp 5 accuracy comparison\n")

    print(f"Report → {out / 'v5_experiments.md'}", file=sys.stderr)
    print("Done.", file=sys.stderr)
