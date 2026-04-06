"""Planning Sensitivity Audit v2 (A1).

Direct measurement: for each scenario, compute cell-level planner costs
on risky vs safe branch cells using fresh vs trained LatentCostRiskHead.

Key metrics:
  - Cell cost gap: mean_risky_cost - mean_safe_cost (pre/post training)
  - Gap shift: does training widen the risky-safe cost gap?
  - Route flip: does A* from the fork entry switch path after training?
  - Per-cell cost decomposition: which J term dominates the gap?

Outputs:
  results/planning_sensitivity_v2.csv
  results/planning_sensitivity_v2_report.md
"""
import sys, csv, copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_scenario
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.planner_astar import (
    cell_cost_v2_latent, _astar_core
)
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()

FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTY = "medium"
SEEDS = list(range(10))

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


def cost_decomposition(r, c, fb_mean, lp, passable, fb_var=None, n_val=0.0):
    """Decompose cell cost into components."""
    if not passable[r, c]:
        return {"total": np.inf, "c_hat": np.inf, "r_hat": 0, "risk_penalty": 0,
                "risk_adjusted": 0, "uc": 0, "ur": 0, "learning_factor": 0}

    x = fb_mean[r, c]
    c_hat = lp.predict_cost(x)
    r_hat = lp.predict_risk(x)

    if fb_var is not None:
        x_var = fb_var[r, c]
        uc = lp.predict_cost_uncertainty_from_var(x_var)
        ur = lp.predict_risk_uncertainty_from_var(x_var)
    else:
        uc = lp.predict_cost_uncertainty(x)
        ur = lp.predict_risk_uncertainty(x)

    uc = max(0.0, float(np.nan_to_num(uc, nan=0.0)))
    ur = max(0.0, float(np.nan_to_num(ur, nan=0.0)))

    eps = 1e-4
    mu_rho = float(np.clip(r_hat, eps, 1.0 - eps))
    risk_penalty = 5.0 * (-np.log(1.0 - mu_rho))

    # learning factor
    n_upd = lp.n_updates if hasattr(lp, 'n_updates') else 0
    lf = min(1.0, n_upd / 10.0)
    nd = 1.0 - n_val
    risk_adj = risk_penalty * (lf + (1 - lf) * nd)

    total = (1.0 * c_hat + risk_adj + 0.1 * nd * uc + 0.1 * nd * ur)

    return {
        "total": round(float(total), 4),
        "c_hat": round(float(c_hat), 4),
        "r_hat": round(float(r_hat), 4),
        "risk_penalty": round(float(risk_penalty), 4),
        "risk_adjusted": round(float(risk_adj), 4),
        "uc": round(float(uc), 4),
        "ur": round(float(ur), 4),
        "learning_factor": round(lf, 4),
    }


def run_sensitivity(family, seed):
    """Compute cost decomposition on risky vs safe cells, pre/post training."""
    global _current_flags

    # Generate scenario to get meta
    gm, cfg, meta, sc = generate_scenario(family, seed, DIFFICULTY, latent_mode=True)
    if not meta.segments:
        return None

    seg = meta.segments[0]  # first fork segment
    risky_cells = seg.risky_cells[:8]
    safe_cells = seg.safe_cells[:8]
    if not risky_cells or not safe_cells:
        return None

    # Feature belief: use true features as observation (agent sees them)
    fb_mean = meta.cell_features.copy()
    fb_var = np.full_like(fb_mean, 0.5)  # prior variance
    passable = np.array(gm.cell_types) > 0  # non-wall

    # ═══ Fresh learner (pre-training) ═══
    lp_fresh = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    risky_decomp_pre = [cost_decomposition(r, c, fb_mean, lp_fresh, passable, fb_var)
                        for r, c in risky_cells]
    safe_decomp_pre = [cost_decomposition(r, c, fb_mean, lp_fresh, passable, fb_var)
                       for r, c in safe_cells]

    # A* from fork entry with fresh learner
    fork_entry = (2, seg.col_start)
    goal = (2, gm.width - 2)

    def cost_fn_fresh(r, c):
        return cell_cost_v2_latent(r, c, fb_mean, lp_fresh, passable,
                                    feature_belief_var=fb_var, route_necessity=0.0)
    path_pre = _astar_core(fork_entry, goal, cost_fn_fresh,
                            gm.height, gm.width, 40, passable)

    # ═══ Trained learner (10 eps with TPM) ═══
    lp_trained = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                          use_unlock_memory=True, use_perceptual_access=True)

    train_kw = dict(tutor_mode="none", robot_belief_mode=True,
                    intervention_family_mode=True, item_drop_enabled=True,
                    prefix_horizon=5)

    for k in range(10):
        s = runner.reset(seed=k, scenario_family=family,
                         latent_mode=True, difficulty=DIFFICULTY,
                         latent_predictor=lp_trained, **train_kw)
        while not s.done:
            runner.step(s)

    _current_flags = {}

    risky_decomp_post = [cost_decomposition(r, c, fb_mean, lp_trained, passable, fb_var)
                         for r, c in risky_cells]
    safe_decomp_post = [cost_decomposition(r, c, fb_mean, lp_trained, passable, fb_var)
                        for r, c in safe_cells]

    # A* from fork entry with trained learner
    def cost_fn_trained(r, c):
        return cell_cost_v2_latent(r, c, fb_mean, lp_trained, passable,
                                    feature_belief_var=fb_var, route_necessity=0.0)
    path_post = _astar_core(fork_entry, goal, cost_fn_trained,
                             gm.height, gm.width, 40, passable)

    # Route flip: does the first diverging step differ?
    def first_branch_step(path, risky_set, safe_set):
        for pos in path:
            if pos in risky_set:
                return "risky"
            if pos in safe_set:
                return "safe"
        return "unknown"

    risky_set = set(risky_cells)
    safe_set = set(safe_cells)
    branch_pre = first_branch_step(path_pre, risky_set, safe_set)
    branch_post = first_branch_step(path_post, risky_set, safe_set)
    route_flip = int(branch_pre != branch_post)

    # Aggregate
    mean_risky_pre = np.mean([d["total"] for d in risky_decomp_pre])
    mean_safe_pre = np.mean([d["total"] for d in safe_decomp_pre])
    mean_risky_post = np.mean([d["total"] for d in risky_decomp_post])
    mean_safe_post = np.mean([d["total"] for d in safe_decomp_post])

    gap_pre = mean_risky_pre - mean_safe_pre
    gap_post = mean_risky_post - mean_safe_post

    # Component breakdown
    risky_risk_pre = np.mean([d["risk_adjusted"] for d in risky_decomp_pre])
    safe_risk_pre = np.mean([d["risk_adjusted"] for d in safe_decomp_pre])
    risky_risk_post = np.mean([d["risk_adjusted"] for d in risky_decomp_post])
    safe_risk_post = np.mean([d["risk_adjusted"] for d in safe_decomp_post])

    risky_uc_pre = np.mean([d["uc"] for d in risky_decomp_pre])
    safe_uc_pre = np.mean([d["uc"] for d in safe_decomp_pre])
    risky_uc_post = np.mean([d["uc"] for d in risky_decomp_post])
    safe_uc_post = np.mean([d["uc"] for d in safe_decomp_post])

    # Weight change
    theta0 = np.concatenate([lp_fresh.cost_head.w, [lp_fresh.cost_head.b],
                              lp_fresh.risk_head.w, [lp_fresh.risk_head.b]])
    theta1 = np.concatenate([lp_trained.cost_head.w, [lp_trained.cost_head.b],
                              lp_trained.risk_head.w, [lp_trained.risk_head.b]])
    delta_theta = float(np.linalg.norm(theta1 - theta0))

    return {
        "family": family, "seed": seed,
        "mean_risky_pre": round(mean_risky_pre, 4),
        "mean_safe_pre": round(mean_safe_pre, 4),
        "gap_pre": round(gap_pre, 4),
        "mean_risky_post": round(mean_risky_post, 4),
        "mean_safe_post": round(mean_safe_post, 4),
        "gap_post": round(gap_post, 4),
        "delta_gap": round(gap_post - gap_pre, 4),
        "risk_gap_pre": round(risky_risk_pre - safe_risk_pre, 4),
        "risk_gap_post": round(risky_risk_post - safe_risk_post, 4),
        "uc_gap_pre": round(risky_uc_pre - safe_uc_pre, 4),
        "uc_gap_post": round(risky_uc_post - safe_uc_post, 4),
        "branch_pre": branch_pre,
        "branch_post": branch_post,
        "route_flip": route_flip,
        "delta_theta": round(delta_theta, 6),
        "n_updates": lp_trained.n_updates,
        # Include example cell predictions
        "ex_risky_r_pre": risky_decomp_pre[0]["r_hat"],
        "ex_risky_r_post": risky_decomp_post[0]["r_hat"],
        "ex_safe_r_pre": safe_decomp_pre[0]["r_hat"],
        "ex_safe_r_post": safe_decomp_post[0]["r_hat"],
    }


if __name__ == "__main__":
    all_results = []
    total = len(FAMILIES) * len(SEEDS)
    i = 0

    for fam in FAMILIES:
        for seed in SEEDS:
            i += 1
            print(f"  [{i}/{total}] {fam}/s{seed}", file=sys.stderr)
            r = run_sensitivity(fam, seed)
            if r:
                all_results.append(r)

    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── CSV ────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "planning_sensitivity_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    # ── Report ────────────────────────────────────────────────────
    md_path = out_dir / "planning_sensitivity_v2_report.md"
    agg = defaultdict(lambda: {"n": 0, "flips": 0, "rows": []})
    for r in all_results:
        a = agg[r["family"]]
        a["n"] += 1
        a["flips"] += r["route_flip"]
        a["rows"].append(r)

    with open(md_path, "w") as f:
        f.write("# Planning Sensitivity Audit v2\n\n")

        # Summary per family
        f.write("## Route Flip & Cost Gap\n\n")
        f.write("| Family | Flip Rate | Δθ | Gap_pre | Gap_post | Δ(gap) | "
                "risk_gap Δ | uc_gap Δ |\n")
        f.write("|--------|-----------|-----|---------|----------|--------|"
                "-----------|----------|\n")
        for fam in FAMILIES:
            a = agg[fam]
            if a["n"] == 0:
                continue
            rows = a["rows"]
            fr = a["flips"] / a["n"]
            dt = np.mean([r["delta_theta"] for r in rows])
            gp = np.mean([r["gap_pre"] for r in rows])
            gpo = np.mean([r["gap_post"] for r in rows])
            dg = np.mean([r["delta_gap"] for r in rows])
            rgp = np.mean([r["risk_gap_pre"] for r in rows])
            rgpo = np.mean([r["risk_gap_post"] for r in rows])
            ugp = np.mean([r["uc_gap_pre"] for r in rows])
            ugpo = np.mean([r["uc_gap_post"] for r in rows])
            f.write(f"| {fam} | {fr:.0%} | {dt:.3f} | {gp:.3f} | {gpo:.3f} "
                    f"| {dg:+.3f} | {rgpo - rgp:+.3f} | {ugpo - ugp:+.3f} |\n")

        # Example predictions
        f.write("\n## Example Risk Predictions (first risky vs safe cell)\n\n")
        f.write("| Family | risky_r̂_pre | risky_r̂_post | safe_r̂_pre | safe_r̂_post |\n")
        f.write("|--------|-----------|------------|----------|----------|\n")
        for fam in FAMILIES:
            a = agg[fam]
            if not a["rows"]:
                continue
            r = a["rows"][0]  # first seed example
            f.write(f"| {fam} | {r['ex_risky_r_pre']:.4f} | {r['ex_risky_r_post']:.4f} "
                    f"| {r['ex_safe_r_pre']:.4f} | {r['ex_safe_r_post']:.4f} |\n")

        # A* branch choices
        f.write("\n## A* Branch Selection (from fork entry)\n\n")
        f.write("| Family | seed | branch_pre | branch_post | flip |\n")
        f.write("|--------|------|-----------|------------|------|\n")
        for fam in FAMILIES:
            for r in agg[fam]["rows"][:5]:
                f.write(f"| {fam} | {r['seed']} | {r['branch_pre']} "
                        f"| {r['branch_post']} | {r['route_flip']} |\n")

        # Analysis
        f.write("\n## Diagnosis\n\n")
        f.write("### If gap_pre ≈ gap_post ≈ 0:\n")
        f.write("→ Risky and safe cells have IDENTICAL planner costs.\n")
        f.write("The latent predictor produces the SAME cost/risk for both because:\n")
        f.write("- Prior predicts uniform risk (sigmoid(0)=0.5) for unvisited cells.\n")
        f.write("- Features differ but the learned weights don't discriminate.\n\n")
        f.write("### If gap_post > gap_pre but no flip:\n")
        f.write("→ Training widens the gap but not enough to overcome the "
                "topology/uncertainty terms.\n\n")
        f.write("### If gap_post > gap_pre AND flips occur:\n")
        f.write("→ Training successfully influences route selection! "
                "Transfer should follow.\n")

    print(f"Report -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
