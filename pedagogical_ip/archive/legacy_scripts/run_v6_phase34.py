"""V6 Phase 3–4: Canonical Compatibility + Tutor Re-Integration.

P3.1: Branch gating audit (gate rates per family)
P3.2: Canonical sweep (SR/DR/TR per family × difficulty)
P3.3: ELCB regression lock
P4.1: RSA Warning + Branch Planner (4 conditions on ELCB)
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
from src.agents.branch_summary import summarize_branch, SUMMARY_DIM
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.planner.branch_candidates import (
    extract_elcb_branches, BranchCandidate, should_activate_branch_reranker
)
from src.planner.branch_reranker import choose_branch, branch_cell_cost
from src.teachers.rsa_warning_v2 import RSAWarningV2

runner = LatticeV2Runner()
out = Path("results")
out.mkdir(exist_ok=True)

CANONICAL_FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTIES = ["easy", "medium", "hard"]
N_SEEDS = 20


def make_passable(gm):
    return np.array(gm.cell_types) != CellType.WALL


def train_v5_elcb(train_seeds, difficulty="medium"):
    """Train V5 models on ELCB seeds."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    for seed in train_seeds:
        gm, _, meta, sc = generate_scenario("elcb", seed, difficulty, latent_mode=True)
        ww = meta.world_weights
        fb = meta.cell_features
        fv = np.full_like(fb, 0.3)
        for _ in range(5):
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
    return lp, lib, scorer


# ═══════════════════════════════════════════════════════════════
# P3.1: Branch Gating Audit
# ═══════════════════════════════════════════════════════════════
def p3_1_gating_audit():
    """Check which families trigger branch gating and how often."""
    print("P3.1: Gating Audit", file=sys.stderr)
    results = {}

    for fam in CANONICAL_FAMILIES + ["elcb"]:
        trigger_count = 0
        total_maps = 0

        for seed in range(N_SEEDS):
            try:
                gm, cfg, meta, sc = generate_scenario(
                    fam, seed, "medium", latent_mode=True)
            except Exception:
                continue

            total_maps += 1

            # For ELCB: use known branch structure
            if fam == "elcb" and hasattr(sc, 'branch_a_cells'):
                candidates = extract_elcb_branches(sc)
                if should_activate_branch_reranker(candidates):
                    trigger_count += 1
                continue

            # For canonical families: detect branches from topology
            # A branch competition exists if there are multiple distinct
            # paths with similar length from start to goal
            # For simplicity: check if fork_trap has fork structure
            if hasattr(sc, 'branch_a_cells') and hasattr(sc, 'branch_b_cells'):
                candidates = [
                    BranchCandidate(0, list(sc.branch_a_cells), len(sc.branch_a_cells),
                                    (0, 0), (0, 0), (0, 0), (0, 0)),
                    BranchCandidate(1, list(sc.branch_b_cells), len(sc.branch_b_cells),
                                    (0, 0), (0, 0), (0, 0), (0, 0)),
                ]
                if should_activate_branch_reranker(candidates):
                    trigger_count += 1
            # hazard_belt / deadline_gate typically don't have branch_a/b
            # so they won't trigger → correct behavior

        results[fam] = {
            "total": total_maps,
            "triggered": trigger_count,
            "rate": round(trigger_count / max(total_maps, 1), 3),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# P3.2: Canonical Compatibility Sweep
# ═══════════════════════════════════════════════════════════════
def p3_2_canonical_sweep():
    """Run canonical families to verify branch planner doesn't break them."""
    print("P3.2: Canonical Sweep", file=sys.stderr)

    CANONICAL_CONFIGS = {
        "fork_trap": {
            "no_tutor": dict(tutor_mode="none", warning_mode="none"),
            "robot_belief": dict(
                tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5),
        },
        "hazard_belt": {
            "no_tutor": dict(tutor_mode="none", warning_mode="none"),
            "robot_belief": dict(
                tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5),
        },
        "deadline_gate": {
            "no_tutor": dict(tutor_mode="none", warning_mode="none"),
            "robot_belief": dict(
                tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5),
        },
    }

    results = []
    for fam, teachers in CANONICAL_CONFIGS.items():
        for t_name, t_kw in teachers.items():
            for diff in ["medium"]:
                n_s = n_d = n_t = 0
                steps_list = []
                for seed in range(N_SEEDS):
                    s = runner.reset(
                        seed=seed, scenario_family=fam,
                        latent_mode=True, difficulty=diff, **t_kw)
                    while not s.done:
                        runner.step(s)
                    m = runner.get_metrics(s)
                    if m["reached_goal"] and m["survived"]:
                        n_s += 1
                    elif not m["survived"]:
                        n_d += 1
                    else:
                        n_t += 1
                    steps_list.append(m["steps"])

                n = N_SEEDS
                results.append({
                    "family": fam,
                    "condition": t_name,
                    "difficulty": diff,
                    "SR": round(n_s / n, 3),
                    "DR": round(n_d / n, 3),
                    "TR": round(n_t / n, 3),
                    "mean_steps": round(float(np.mean(steps_list)), 1),
                })

    return results


# ═══════════════════════════════════════════════════════════════
# P3.3: ELCB Regression Lock
# ═══════════════════════════════════════════════════════════════
def p3_3_elcb_regression():
    """Verify V6 ELCB results still hold."""
    print("P3.3: ELCB Regression", file=sys.stderr)
    lp, lib, scorer = train_v5_elcb(list(range(40)))
    test_seeds = list(range(100, 130))

    conditions = {
        "old_planner":   ("pointwise_only", 0.0),
        "concept_only":  ("concept_only",   1.0),
        "scorer_only":   ("scorer_only",    1.0),
        "hybrid":        ("hybrid",         1.0),
    }

    results = {}
    for name, (mode, lb) in conditions.items():
        safe = 0
        side_a = 0
        for seed in test_seeds:
            gm, _, meta, sc = generate_scenario("elcb", seed, "medium", latent_mode=True)
            fb = meta.cell_features
            fv = np.full_like(fb, 0.3)
            passable = make_passable(gm)
            candidates = extract_elcb_branches(sc)
            tie_rng = np.random.default_rng(seed + 777)
            best, _ = choose_branch(
                candidates, fb, fv, lp, passable, lib, scorer,
                lambda_b=lb, score_mode=mode, tie_rng=tie_rng)
            safe += int(best.branch_id == sc.oracle_safe_branch_id)
            side_a += int(best.branch_id == 0)

        n = len(test_seeds)
        results[name] = {
            "SBCR": round(safe / n, 3),
            "SideBias": round(abs(side_a / n - 0.5), 3),
            "n": n,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# P4.1: RSA Warning + Branch Planner
# ═══════════════════════════════════════════════════════════════
def p4_1_tutor_planner():
    """Compare warning effect with old vs branch planner."""
    print("P4.1: Tutor+Planner", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))
    rsa = RSAWarningV2()

    results = []

    for cond in ["no_tutor+old", "rsa_warn+old", "no_tutor+branch", "rsa_warn+branch"]:
        use_rsa = "rsa_warn" in cond
        use_branch = "branch" in cond

        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05)
        k_done = 0

        for k_target in K_VALUES:
            # Train up to k_target
            while k_done < k_target:
                gm, _, meta, sc = generate_scenario("elcb", k_done, "medium", latent_mode=True)
                ww = meta.world_weights
                fb = meta.cell_features
                fv = np.full_like(fb, 0.3)

                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = fb[r, c]
                        lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

                # Update V5 models
                safe_cells = sc.branch_a_cells if sc.oracle_safe_branch_id == 0 else sc.branch_b_cells
                risky_cells = sc.branch_a_cells if sc.oracle_risky_branch_id == 0 else sc.branch_b_cells
                s_s = summarize_branch(safe_cells, fb, fv, lp)
                s_r = summarize_branch(risky_cells, fb, fv, lp)
                lib.update("safe_branch", s_s)
                lib.update("risky_branch", s_r)
                scorer.update(build_scorer_input(s_s, lib), 1.0)
                scorer.update(build_scorer_input(s_r, lib), 0.0)

                # RSA: if warning enabled, simulate warning and additional update
                if use_rsa:
                    risk_a = float(np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_a_cells]))
                    risk_b = float(np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_b_cells]))
                    z_state = rsa.classify_risk_state(risk_a, risk_b, branch_a_side="left")
                    u_idx, u_name = rsa.choose_utterance(z_state)
                    # Warning acts as additional evidence → re-weight risky branch learning
                    if u_name != "silence":
                        for r2, c2 in risky_cells:
                            z2 = fb[r2, c2]
                            lp.update_from_outcome(z2, ww.true_cost(z2), ww.true_risk(z2), weight=0.5)

                k_done += 1

            # Probe (no tutor, autonomous choice)
            safe_choices = 0
            for ps in PROBE_SEEDS:
                gm_p, _, meta_p, sc_p = generate_scenario("elcb", ps, "medium", latent_mode=True)
                fb_p = meta_p.cell_features
                fv_p = np.full_like(fb_p, 0.3)
                passable = make_passable(gm_p)
                candidates = extract_elcb_branches(sc_p)
                tie_rng = np.random.default_rng(ps + 777)

                if use_branch:
                    best, _ = choose_branch(
                        candidates, fb_p, fv_p, lp, passable, lib, scorer,
                        lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
                    safe_choices += int(best.branch_id == sc_p.oracle_safe_branch_id)
                else:
                    ca = branch_cell_cost(
                        candidates[0].cells, candidates[0].entry_gate,
                        candidates[0].exit_gate, fb_p, lp, passable, fv_p)
                    cb = branch_cell_cost(
                        candidates[1].cells, candidates[1].entry_gate,
                        candidates[1].exit_gate, fb_p, lp, passable, fv_p)
                    margin = cb - ca
                    if abs(margin) < 1e-4:
                        chosen = int(tie_rng.integers(0, 2))
                    else:
                        chosen = 0 if ca < cb else 1
                    safe_choices += int(chosen == sc_p.oracle_safe_branch_id)

            results.append({
                "condition": cond,
                "k": k_target,
                "SBCR": round(safe_choices / len(PROBE_SEEDS), 3),
            })

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    g = p3_1_gating_audit()
    sweep = p3_2_canonical_sweep()
    reg = p3_3_elcb_regression()
    tutor = p4_1_tutor_planner()

    with open(out / "v6_phase34_report.md", "w") as f:
        f.write("# V6 Phase 3–4 Report\n\n")

        # P3.1
        f.write("## P3.1: Branch Gating Audit\n\n")
        f.write("| Family | Maps | Triggered | Rate |\n")
        f.write("|--------|------|-----------|------|\n")
        for fam, data in g.items():
            f.write(f"| {fam} | {data['total']} | {data['triggered']} | "
                    f"{data['rate']:.0%} |\n")

        # P3.2
        f.write("\n## P3.2: Canonical Compatibility Sweep\n\n")
        f.write("| Family | Condition | SR | DR | TR | Steps |\n")
        f.write("|--------|-----------|----|----|-------|-------|\n")
        for r in sweep:
            f.write(f"| {r['family']} | {r['condition']} | {r['SR']:.0%} | "
                    f"{r['DR']:.0%} | {r['TR']:.0%} | {r['mean_steps']} |\n")

        # P3.3
        f.write("\n## P3.3: ELCB Regression Lock\n\n")
        f.write("| Condition | SBCR | SideBias |\n")
        f.write("|-----------|------|----------|\n")
        for name, data in reg.items():
            f.write(f"| {name} | {data['SBCR']:.0%} | {data['SideBias']:.3f} |\n")

        # P4.1
        f.write("\n## P4.1: RSA Warning + Branch Planner\n\n")
        f.write("| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|-----------|-----|-----|-----|------|------|\n")
        for cond in ["no_tutor+old", "rsa_warn+old", "no_tutor+branch", "rsa_warn+branch"]:
            rows = [r for r in tutor if r["condition"] == cond]
            vals = " | ".join(f"{r['SBCR']:.0%}" for r in rows)
            f.write(f"| {cond} | {vals} |\n")

        # Summary
        f.write("\n## Summary\n\n")
        f.write("### Canonical Compatibility\n")
        f.write("Check that SR values are not degraded vs existing baselines.\n\n")
        f.write("### ELCB Regression\n")
        f.write("Check that concept/scorer/hybrid still ≈100%.\n\n")
        f.write("### Tutor Effect\n")
        f.write("Compare `rsa_warn+branch` vs `no_tutor+branch` at each k.\n")

    print(f"Report → {out / 'v6_phase34_report.md'}", file=sys.stderr)
    print("Done.", file=sys.stderr)
