"""ELCB Unified Experiment Suite.

Sanity Checks:
  SC1: Oracle semantic flip curve — sweep Δr, find flip threshold
  SC2: Length-neutrality test — equal semantics → no systematic bias
  SC3: Prediction-planning coupling rate (PPCR)
  SC4: Passability neutrality audit (zero walls in either branch)

Online Sweep:
  no_tutor / warn_only / robot_belief_post (WAIT+WARN)
  Metrics: SBCR, RFR, branch margin, PPCR

Transfer:
  k ∈ {0,1,3,10,30}, probe with SBCR/RFR/margin

Outputs:
  results/elcb_sanity_checks.md
  results/elcb_sweep.csv
  results/elcb_transfer.csv
  results/elcb_report.md
"""
import sys, csv, copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.planner_astar import cell_cost_v2_latent, _astar_core
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()

DIFFICULTY = "medium"
SEEDS = list(range(20))

# ── Monkeypatch ──────────────────────────────────────────────────
_orig_score = ip.score_interventions
_current_flags = {}

def _patched_score(*args, **kwargs):
    cfg = kwargs.get("config")
    if cfg is not None and _current_flags:
        for k, v in _current_flags.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return _orig_score(*args, **kwargs)

ip.score_interventions = _patched_score
runner_mod.score_interventions = _patched_score

# ── Core helpers ─────────────────────────────────────────────────

def make_passable(gm):
    """Derive passable mask from cell_types."""
    return np.array(gm.cell_types) != CellType.WALL


def branch_chosen_by_astar(gm, meta, sc, lp, fb_var=None, necessity=0.0,
                           tie_rng=None):
    """Compute branch preference by directly summing costs along known paths.
    
    Does NOT use A* (which has move-order bias). Instead, sums cell costs
    along each branch's known cell list for a clean comparison.
    
    Returns: (branch_id: 0=a, 1=b, margin, safe_chosen, cost_a, cost_b)
    """
    passable = make_passable(gm)
    fb_mean = meta.cell_features.copy()
    if fb_var is None:
        fb_var = np.full_like(fb_mean, 0.5)

    def cost_fn(r, c):
        return cell_cost_v2_latent(r, c, fb_mean, lp, passable,
                                    feature_belief_var=fb_var,
                                    route_necessity=necessity)

    # Direct summation along known branch cells (no A* bias)
    fork_col = sc.fork_cell[1]
    merge_col = sc.merge_cell[1]
    cost_a = cost_fn(1, fork_col)  # entry gate
    for r, c in sc.branch_a_cells:
        cost_a += cost_fn(r, c)
    cost_a += cost_fn(1, merge_col)  # exit gate

    cost_b = cost_fn(3, fork_col)  # entry gate
    for r, c in sc.branch_b_cells:
        cost_b += cost_fn(r, c)
    cost_b += cost_fn(3, merge_col)  # exit gate

    margin = cost_b - cost_a  # positive = A cheaper
    # Tie-breaking: when costs are within epsilon, an uninformed agent
    # has no preference. Use seeded random for unbiased tie-break.
    if abs(margin) < 1e-4:
        if tie_rng is not None:
            chosen = int(tie_rng.integers(0, 2))
        else:
            chosen = int(np.random.randint(0, 2))
    else:
        chosen = 0 if cost_a < cost_b else 1
    safe_chosen = int(chosen == sc.oracle_safe_branch_id)

    return chosen, margin, safe_chosen, cost_a, cost_b


def run_episode(family, seed, kw, lp, difficulty=DIFFICULTY):
    s = runner.reset(seed=seed, scenario_family=family,
                     latent_mode=True, difficulty=difficulty,
                     latent_predictor=lp, **kw)
    while not s.done:
        runner.step(s)
    m = runner.get_metrics(s)
    return m, s


# ═══════════════════════════════════════════════════════════════════
# SC1: Oracle Semantic Flip Curve
# ═══════════════════════════════════════════════════════════════════
def run_sc1():
    """Sweep oracle risk on risky branch, find where planner flips to safe."""
    print("  SC1: Oracle Flip Curve", file=sys.stderr)
    results = []
    risk_offsets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]

    for seed in [0, 1, 2, 3, 4]:
        gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        safe_cells = sc.safe_cells
        risky_cells = sc.risky_cells

        for offset in risk_offsets:
            # Create oracle predictor with known risk difference
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            ww = meta.world_weights

            if ww is not None:
                # Train on all cells with true labels
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL:
                            continue
                        z = meta.cell_features[r, c]
                        true_c = ww.true_cost(z)
                        # For risky cells, add offset to risk
                        if (r, c) in set(risky_cells):
                            true_r = min(ww.true_risk(z) + offset, 0.95)
                        else:
                            true_r = ww.true_risk(z)
                        lp.update_from_outcome(z, true_c, true_r)

            chosen, margin, safe_chosen, ca, cb = branch_chosen_by_astar(
                gm, meta, sc, lp)

            results.append({
                "seed": seed,
                "risk_offset": offset,
                "chosen_branch": chosen,
                "safe_chosen": safe_chosen,
                "margin": round(margin, 4),
                "cost_a": round(ca, 4),
                "cost_b": round(cb, 4),
                "oracle_safe": sc.oracle_safe_branch_id,
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# SC2: Length-Neutrality Test
# ═══════════════════════════════════════════════════════════════════
def run_sc2():
    """With equal semantics, check for systematic branch bias."""
    print("  SC2: Length-Neutrality", file=sys.stderr)
    a_chosen = 0
    total = 0

    for seed in range(50):
        gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
        # Equal predictor: fresh (uniform prior)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        tie_rng = np.random.default_rng(seed + 999)  # separate from scenario rng
        chosen, margin, _, _, _ = branch_chosen_by_astar(gm, meta, sc, lp, tie_rng=tie_rng)
        a_chosen += (1 - chosen)  # chosen==0 means A
        total += 1

    return {"a_rate": a_chosen / total, "b_rate": 1 - a_chosen / total,
            "total": total, "bias": abs(a_chosen / total - 0.5)}


# ═══════════════════════════════════════════════════════════════════
# SC3: Prediction-Planning Coupling Rate (PPCR)
# ═══════════════════════════════════════════════════════════════════
def run_sc3():
    """After training, does predicted branch preference agree with planner?"""
    print("  SC3: PPCR", file=sys.stderr)
    results = []

    for seed in SEEDS[:10]:
        gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)

        # Train for 10 episodes
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        global _current_flags
        _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True)
        for k in range(10):
            run_episode("elcb", k, dict(tutor_mode="none", robot_belief_mode=True,
                        intervention_family_mode=True, item_drop_enabled=True,
                        prefix_horizon=5), lp)
        _current_flags = {}

        # Prediction preference: which branch has lower predicted risk?
        fb = meta.cell_features
        pred_risk_a = np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_a_cells])
        pred_risk_b = np.mean([lp.predict_risk(fb[r, c]) for r, c in sc.branch_b_cells])
        pred_prefers = 0 if pred_risk_a < pred_risk_b else 1

        # Planner preference
        chosen, margin, safe_chosen, _, _ = branch_chosen_by_astar(gm, meta, sc, lp)

        coupled = int(pred_prefers == chosen)

        results.append({
            "seed": seed,
            "pred_risk_a": round(pred_risk_a, 4),
            "pred_risk_b": round(pred_risk_b, 4),
            "pred_prefers": pred_prefers,
            "planner_chosen": chosen,
            "coupled": coupled,
            "margin": round(margin, 4),
            "safe_chosen": safe_chosen,
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# SC4: Passability Audit
# ═══════════════════════════════════════════════════════════════════
def run_sc4():
    """Hard constraint: zero walls in either branch across all seeds/difficulties."""
    print("  SC4: Passability Audit", file=sys.stderr)
    violations = 0
    total = 0
    for diff in ["easy", "medium", "hard"]:
        for seed in range(50):
            gm, cfg, meta, sc = generate_scenario("elcb", seed, diff, latent_mode=True)
            for cell in sc.branch_a_cells + sc.branch_b_cells:
                total += 1
                if gm.cell_types[cell[0], cell[1]] == CellType.WALL:
                    violations += 1
            # Also check cost is finite
            for cell in sc.branch_a_cells + sc.branch_b_cells:
                if gm.true_cost[cell[0], cell[1]] == np.inf:
                    violations += 1
    return {"violations": violations, "total_cells": total}


# ═══════════════════════════════════════════════════════════════════
# Online Sweep
# ═══════════════════════════════════════════════════════════════════
def run_online_sweep():
    """no_tutor / warn_only / robot_belief_post — SBCR, RFR, margin."""
    print("  Online Sweep", file=sys.stderr)
    global _current_flags
    results = []

    conditions = {
        "no_tutor": dict(tutor_mode="none", warning_mode="none"),
        "warn_only": dict(tutor_mode="none", warning_mode="risk_above_threshold"),
        "robot_belief_post": dict(tutor_mode="none", robot_belief_mode=True,
                                   intervention_family_mode=True,
                                   item_drop_enabled=False,
                                   prefix_horizon=5),
    }

    for cond_name, kw in conditions.items():
        if cond_name == "robot_belief_post":
            _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                                  use_unlock_memory=True, use_perceptual_access=True)
        else:
            _current_flags = {}

        safe_choices = 0
        successes = 0
        total = 0

        for seed in SEEDS:
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            m, s = run_episode("elcb", seed, kw, lp)

            gm, cfg, meta, sc = generate_scenario("elcb", seed, DIFFICULTY, latent_mode=True)
            chosen, margin, safe_chosen, _, _ = branch_chosen_by_astar(gm, meta, sc, lp)

            safe_choices += safe_chosen
            successes += int(m["reached_goal"] and m["survived"])
            total += 1

        results.append({
            "condition": cond_name,
            "SBCR": round(safe_choices / total, 3),
            "SR": round(successes / total, 3),
            "n": total,
        })

    _current_flags = {}
    return results


# ═══════════════════════════════════════════════════════════════════
# Transfer Experiment
# ═══════════════════════════════════════════════════════════════════
def run_transfer():
    """k ∈ {0,1,3,10,30}: train with TPM, probe SBCR on new seeds."""
    print("  Transfer", file=sys.stderr)
    global _current_flags
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 120))
    results = []

    train_kw = dict(tutor_mode="none", robot_belief_mode=True,
                    intervention_family_mode=True, item_drop_enabled=False,
                    prefix_horizon=5)

    for cond in ["no_tutor", "robot_belief_post"]:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        k_done = 0

        for k_target in K_VALUES:
            if cond == "robot_belief_post":
                _current_flags = dict(use_bottleneck_matching=True,
                                      use_warn_damping=True,
                                      use_unlock_memory=True,
                                      use_perceptual_access=True)
            else:
                _current_flags = {}

            # Train from k_done to k_target
            while k_done < k_target:
                kw = train_kw if cond == "robot_belief_post" else \
                     dict(tutor_mode="none", warning_mode="none")
                run_episode("elcb", k_done, kw, lp)
                k_done += 1

            # Probe on new seeds
            _current_flags = {}
            safe_choices = 0
            margins = []
            for ps in PROBE_SEEDS:
                gm, cfg, meta, sc = generate_scenario("elcb", ps, DIFFICULTY, latent_mode=True)
                lp_probe = copy.deepcopy(lp)
                chosen, margin, safe_chosen, _, _ = branch_chosen_by_astar(
                    gm, meta, sc, lp_probe)
                safe_choices += safe_chosen
                margins.append(margin)

            sbcr = safe_choices / len(PROBE_SEEDS)
            results.append({
                "condition": cond,
                "k": k_target,
                "SBCR": round(sbcr, 3),
                "mean_margin": round(np.mean(margins), 4),
                "n_updates": lp.n_updates,
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Sanity checks
    print("=== SC1: Oracle Flip Curve ===", file=sys.stderr)
    sc1 = run_sc1()
    print("=== SC2: Length-Neutrality ===", file=sys.stderr)
    sc2 = run_sc2()
    print("=== SC3: PPCR ===", file=sys.stderr)
    sc3 = run_sc3()
    print("=== SC4: Passability ===", file=sys.stderr)
    sc4 = run_sc4()

    # Online sweep
    print("=== Online Sweep ===", file=sys.stderr)
    sweep = run_online_sweep()

    # Transfer
    print("=== Transfer ===", file=sys.stderr)
    transfer = run_transfer()

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── Write Report ────────────────────────────────────────────────
    md = out_dir / "elcb_report.md"
    with open(md, "w") as f:
        f.write("# ELCB Experiment Report\n\n")

        # SC1
        f.write("## SC1: Oracle Semantic Flip Curve\n\n")
        f.write("| Risk Offset | Safe Chosen Rate | Mean Margin |\n")
        f.write("|-------------|-----------------|-------------|\n")
        sc1_by_offset = defaultdict(lambda: {"safe": 0, "n": 0, "margins": []})
        for r in sc1:
            a = sc1_by_offset[r["risk_offset"]]
            a["safe"] += r["safe_chosen"]
            a["n"] += 1
            a["margins"].append(r["margin"])
        for offset in sorted(sc1_by_offset.keys()):
            a = sc1_by_offset[offset]
            f.write(f"| {offset:.2f} | {a['safe']/a['n']:.0%} "
                    f"| {np.mean(a['margins']):.3f} |\n")

        # SC2
        f.write(f"\n## SC2: Length-Neutrality\n\n")
        f.write(f"- Branch A chosen: {sc2['a_rate']:.0%}\n")
        f.write(f"- Branch B chosen: {sc2['b_rate']:.0%}\n")
        f.write(f"- Bias (|rate - 0.5|): {sc2['bias']:.3f}\n")
        f.write(f"- ✅ PASS" if sc2["bias"] < 0.15 else f"- ⚠️ BIAS > 0.15")
        f.write("\n")

        # SC3
        f.write(f"\n## SC3: Prediction-Planning Coupling (PPCR)\n\n")
        coupled = sum(r["coupled"] for r in sc3)
        f.write(f"- PPCR: {coupled}/{len(sc3)} ({coupled/len(sc3):.0%})\n")
        f.write("| Seed | pred_r_A | pred_r_B | pred_pref | planner | coupled | safe |\n")
        f.write("|------|----------|----------|-----------|---------|---------|------|\n")
        for r in sc3:
            f.write(f"| {r['seed']} | {r['pred_risk_a']:.4f} | {r['pred_risk_b']:.4f} "
                    f"| {'A' if r['pred_prefers']==0 else 'B'} "
                    f"| {'A' if r['planner_chosen']==0 else 'B'} "
                    f"| {r['coupled']} | {r['safe_chosen']} |\n")

        # SC4
        f.write(f"\n## SC4: Passability Audit\n\n")
        f.write(f"- Violations: {sc4['violations']} / {sc4['total_cells']}\n")
        f.write(f"- {'✅ PASS (zero violations)' if sc4['violations']==0 else '❌ FAIL'}\n")

        # Online Sweep
        f.write(f"\n## Online Sweep\n\n")
        f.write(f"| Condition | SBCR | SR |\n")
        f.write(f"|-----------|------|----|\n")
        for r in sweep:
            f.write(f"| {r['condition']} | {r['SBCR']:.0%} | {r['SR']:.0%} |\n")

        # Transfer
        f.write(f"\n## Transfer (SBCR after k training episodes)\n\n")
        f.write(f"| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write(f"|-----------|-----|-----|-----|------|------|\n")
        for cond in ["no_tutor", "robot_belief_post"]:
            row = [r for r in transfer if r["condition"] == cond]
            vals = " | ".join(f"{r['SBCR']:.0%}" for r in row)
            f.write(f"| {cond} | {vals} |\n")

        # Interpretation
        f.write(f"\n## Interpretation\n\n")
        f.write("### SC1: Does planner flip with oracle risk?\n")
        f.write("If SBCR rises with risk_offset → planner IS risk-sensitive.\n")
        f.write("The flip threshold tells us how much prediction difference is needed.\n\n")
        f.write("### SC2: Is topology neutral?\n")
        f.write("If bias < 0.15 → no systematic branch preference from geometry.\n\n")
        f.write("### SC3: Do predictions control planning?\n")
        f.write("If PPCR > 80% → predictions successfully drive route choice.\n")
        f.write("If PPCR < 50% → other terms (uncertainty, cost) dominate.\n\n")
        f.write("### Transfer: Does SBCR improve with training?\n")
        f.write("If SBCR rises with k → learner can transfer semantic knowledge.\n")
        f.write("If flat → same bottleneck as original families.\n")

    print(f"Report -> {md}", file=sys.stderr)

    # CSVs
    with open(out_dir / "elcb_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep[0].keys())
        w.writeheader()
        w.writerows(sweep)

    with open(out_dir / "elcb_transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=transfer[0].keys())
        w.writeheader()
        w.writerows(transfer)

    print("Done.", file=sys.stderr)
