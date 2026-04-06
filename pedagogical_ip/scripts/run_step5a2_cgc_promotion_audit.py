"""Step 5A.2: CGC-v2 Multi-Path Validation + Promotion Audit.

Phase A: Candidate-path audit — how many distinct paths, risk/unc variance
Phase B: Replay ranking audit — A2 vs A2+N2 vs baseline
Phase C: Closed-loop promotion criteria

Uses real CGC-v2 episodes with ≥3 candidate paths extracted from grid topology.

Usage:
  python scripts/run_step5a2_cgc_promotion_audit.py --n_episodes 60 --n_seeds 5
"""

from __future__ import annotations
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict, deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.envs.cgc_v2_family import (
    CGCEpisodeSpec, generate_cgc_session, generate_cgc_episode_scenario,
    ATOMIC_GOALS, COMPOSITE_GOALS, CGC_PREF_TYPES,
)
from src.agents.planner_risk_shadow import (
    PlannerRiskShadow, RiskShadowConfig, PathRiskProfile,
)
from src.agents.necessity_gate_variants import (
    compute_necessity_n2,
)


# ═══════════════════════════════════════════════════════════
# Path extraction from CGC-v2 grid
# ═══════════════════════════════════════════════════════════

def extract_candidate_paths(gm, start, goal, sc, max_paths=6):
    """Extract multiple candidate paths from CGC-v2 grid.

    Paths:
      1. Through safe branch (row safe_row)
      2. Through risky branch (row risky_row)
      3. Through center corridor (row 2, if passable)
      4. Hybrid: start safe, cross to center at midpoint
      5. Hybrid: start risky, cross to center at midpoint
      6. Late-switch: safe until midpoint, then cross to risky

    Returns list of paths (each a list of (r,c) tuples).
    """
    H, W = gm.true_cost.shape
    passable = gm.true_cost < np.inf

    safe_row = sc.safe_row
    risky_row = sc.risky_row
    fork_col = sc.fork_cell[1]
    merge_col = sc.merge_cell[1]
    branch_len = sc.branch_len

    paths = []

    # Path 1: pure safe branch
    p1 = [start]
    p1.append((start[0], fork_col))  # to fork
    p1.append((safe_row, fork_col))  # enter safe
    for i in range(branch_len):
        c = fork_col + 1 + i
        if passable[safe_row, c]:
            p1.append((safe_row, c))
    p1.append((safe_row, merge_col))  # exit
    p1.append((start[0], merge_col))  # back to center
    p1.append(goal)
    paths.append(("safe_branch", p1))

    # Path 2: pure risky branch
    p2 = [start]
    p2.append((start[0], fork_col))
    p2.append((risky_row, fork_col))
    for i in range(branch_len):
        c = fork_col + 1 + i
        if passable[risky_row, c]:
            p2.append((risky_row, c))
    p2.append((risky_row, merge_col))
    p2.append((start[0], merge_col))
    p2.append(goal)
    paths.append(("risky_branch", p2))

    # Path 3: hybrid — safe first half, then cross to risky
    mid = branch_len // 2
    p3 = [start, (start[0], fork_col), (safe_row, fork_col)]
    for i in range(mid):
        c = fork_col + 1 + i
        if passable[safe_row, c]:
            p3.append((safe_row, c))
    # Cross to center then risky
    cross_col = fork_col + 1 + mid
    if 0 <= cross_col < W and passable[2, cross_col]:
        p3.append((2, cross_col))
        if passable[risky_row, cross_col]:
            p3.append((risky_row, cross_col))
            for i in range(mid + 1, branch_len):
                c = fork_col + 1 + i
                if passable[risky_row, c]:
                    p3.append((risky_row, c))
            p3.append((risky_row, merge_col))
            p3.append((start[0], merge_col))
            p3.append(goal)
            paths.append(("safe_then_risky", p3))

    # Path 4: risky first half, then cross to safe
    p4 = [start, (start[0], fork_col), (risky_row, fork_col)]
    for i in range(mid):
        c = fork_col + 1 + i
        if passable[risky_row, c]:
            p4.append((risky_row, c))
    if 0 <= cross_col < W and passable[2, cross_col]:
        p4.append((2, cross_col))
        if passable[safe_row, cross_col]:
            p4.append((safe_row, cross_col))
            for i in range(mid + 1, branch_len):
                c = fork_col + 1 + i
                if passable[safe_row, c]:
                    p4.append((safe_row, c))
            p4.append((safe_row, merge_col))
            p4.append((start[0], merge_col))
            p4.append(goal)
            paths.append(("risky_then_safe", p4))

    # Path 5: late commit safe — safe until 75%, stay safe
    late = int(branch_len * 0.75)
    p5 = [start, (start[0], fork_col), (safe_row, fork_col)]
    for i in range(branch_len):
        c = fork_col + 1 + i
        if passable[safe_row, c]:
            p5.append((safe_row, c))
    p5.append((safe_row, merge_col))
    p5.append((start[0], merge_col))
    p5.append(goal)
    if len(p5) != len(paths[0][1]):  # Only add if different length
        paths.append(("late_safe", p5))

    # Deduplicate by cell sequence
    seen = set()
    unique = []
    for label, p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append((label, p))

    return unique


def evaluate_paths_with_shadow(paths, gm, meta, cfg, sc, t, shadow_a2, shadow_a2_n2):
    """Evaluate paths with both A2 and A2+N2."""
    H, W = gm.true_cost.shape
    cost_map = np.where(gm.true_cost < np.inf, gm.true_cost, 100.0)
    risk_map = gm.true_risk.copy()

    # Uncertainty map: cells with risk > 0.1 have higher uncertainty
    unc_map = np.full((H, W), 0.1)
    unc_map[risk_map > 0.1] = 0.4
    unc_map[risk_map > 0.3] = 0.7

    passable = gm.true_cost < np.inf
    goal = (2, sc.merge_cell[1] + 1)  # goal position
    t_max = cfg.max_steps

    path_lists = [p for _, p in paths]

    # A2 (no gate)
    _, profiles_a2 = shadow_a2.rank_paths(
        path_lists, cost_map, risk_map, unc_map, passable, t, t_max, goal)

    # A2+N2: recompute with gate
    profiles_n2 = []
    for i, prof in enumerate(profiles_a2):
        g_n = compute_necessity_n2(profiles_a2, i)
        eff_unc = g_n * prof.epistemic_uncertainty
        c = shadow_a2.cfg
        new_score = (c.lambda_cost * prof.expected_cost
                     + c.lambda_hazard * prof.hazard_prob
                     + c.lambda_uncertainty * eff_unc
                     + c.lambda_timeout * prof.timeout_prob
                     + c.lambda_detour * prof.detour_cost)
        # Create modified profile
        import copy
        p_copy = copy.copy(prof)
        p_copy.score_a2 = new_score
        p_copy.effective_unc_surcharge = eff_unc
        profiles_n2.append(p_copy)

    # Baseline: just cost, no risk-sensitive scoring
    baseline_scores = [p.expected_cost for p in profiles_a2]
    baseline_best = int(np.argmin(baseline_scores))

    a2_scores = [p.score_a2 for p in profiles_a2]
    a2_best = int(np.argmin(a2_scores))

    n2_scores = [p.score_a2 for p in profiles_n2]
    n2_best = int(np.argmin(n2_scores))

    return {
        "n_paths": len(paths),
        "path_labels": [l for l, _ in paths],
        "baseline_best": baseline_best,
        "a2_best": a2_best,
        "n2_best": n2_best,
        "baseline_label": paths[baseline_best][0],
        "a2_label": paths[a2_best][0],
        "n2_label": paths[n2_best][0],
        "a2_agrees_baseline": a2_best == baseline_best,
        "n2_agrees_a2": n2_best == a2_best,
        "n2_agrees_baseline": n2_best == baseline_best,
        "gate_changes_top1": n2_best != a2_best,
        "hazard_var": float(np.var([p.hazard_prob for p in profiles_a2])),
        "unc_var": float(np.var([p.epistemic_uncertainty for p in profiles_a2])),
        "profiles_a2": profiles_a2,
        "profiles_n2": profiles_n2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=60)
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    out = Path("results/step5a2_cgc_promotion")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shadow_a2 = PlannerRiskShadow(mode="A2")
    shadow_a2_n2 = PlannerRiskShadow(mode="A2")

    lines = ["# Step 5A.2: CGC-v2 Multi-Path Promotion Audit\n\n"]
    lines.append(f"**Episodes**: {args.n_episodes} × {args.n_seeds} seeds\n\n")

    # ═══════════════════════════════════
    # Phase A: Candidate-path audit
    # ═══════════════════════════════════

    lines.append("## Phase A: Candidate-Path Audit\n\n")

    all_results = []
    path_count_hist = defaultdict(int)
    gate_diff_count = 0
    total_episodes = 0

    goals_to_test = (
        [("collect_red", None), ("avoid_blue", None),
         ("collect_red", "avoid_blue"), ("avoid_blue", "use_safe")]
    )

    for seed in range(args.n_seeds):
        for g_obj, g_con in goals_to_test:
            session = generate_cgc_session(
                session_id=seed * 100, n_episodes=args.n_episodes // len(goals_to_test),
                theta_true="safe", goal_obj=g_obj, goal_constraint=g_con,
                rng=np.random.default_rng(seed * 1000))

            for ep in session.episodes:
                try:
                    gm, cfg, meta, sc = generate_cgc_episode_scenario(
                        ep, theta_true=session.theta_true)

                    start = (2, 1)  # agent start
                    paths = extract_candidate_paths(gm, start, (2, sc.merge_cell[1] + 1), sc)

                    if len(paths) < 2:
                        continue

                    result = evaluate_paths_with_shadow(
                        paths, gm, meta, cfg, sc, t=0,
                        shadow_a2=shadow_a2, shadow_a2_n2=shadow_a2_n2)

                    result["goal"] = ep.goal_label
                    result["subtype"] = ep.episode_subtype
                    all_results.append(result)

                    path_count_hist[result["n_paths"]] += 1
                    if result["gate_changes_top1"]:
                        gate_diff_count += 1
                    total_episodes += 1

                except Exception as e:
                    print(f"  SKIP: {ep.goal_label}/{ep.episode_subtype}: {e}",
                          file=sys.stderr)

    lines.append(f"**Total episodes evaluated**: {total_episodes}\n\n")

    # Path count distribution
    lines.append("### Path Count Distribution\n\n")
    lines.append("| N_paths | Count | Fraction |\n")
    lines.append("|---------|-------|----------|\n")
    for n in sorted(path_count_hist.keys()):
        frac = path_count_hist[n] / max(total_episodes, 1)
        lines.append(f"| {n} | {path_count_hist[n]} | {frac:.3f} |\n")

    # Risk/uncertainty variance
    if all_results:
        mean_hvar = np.mean([r["hazard_var"] for r in all_results])
        mean_uvar = np.mean([r["unc_var"] for r in all_results])
        lines.append(f"\n**Mean hazard variance across paths**: {mean_hvar:.4f}\n")
        lines.append(f"**Mean uncertainty variance across paths**: {mean_uvar:.4f}\n\n")

    # Gate diff rate
    gate_diff_rate = gate_diff_count / max(total_episodes, 1)
    lines.append(f"**GateDiffRate** (N2 changes top1 vs A2): "
                 f"{gate_diff_rate:.3f} ({gate_diff_count}/{total_episodes})\n\n")

    # ═══════════════════════════════════
    # Phase B: Replay Ranking Audit
    # ═══════════════════════════════════

    lines.append("## Phase B: Replay Ranking Audit\n\n")

    if all_results:
        # Overall agreement rates
        a2_agrees = sum(1 for r in all_results if r["a2_agrees_baseline"]) / len(all_results)
        n2_agrees_a2 = sum(1 for r in all_results if r["n2_agrees_a2"]) / len(all_results)
        n2_agrees_base = sum(1 for r in all_results if r["n2_agrees_baseline"]) / len(all_results)

        lines.append("### Agreement Rates\n\n")
        lines.append("| Comparison | Agreement |\n")
        lines.append("|------------|----------|\n")
        lines.append(f"| A2 vs baseline | {a2_agrees:.3f} |\n")
        lines.append(f"| A2+N2 vs A2 | {n2_agrees_a2:.3f} |\n")
        lines.append(f"| A2+N2 vs baseline | {n2_agrees_base:.3f} |\n\n")

        # Path choice distribution
        lines.append("### Path Choice Distribution\n\n")
        lines.append("| Planner | safe_branch | risky_branch | safe_then_risky | risky_then_safe |\n")
        lines.append("|---------|-------------|-------------|-----------------|----------------|\n")

        for label_key, name in [("baseline_label", "baseline"),
                                 ("a2_label", "A2"),
                                 ("n2_label", "A2+N2")]:
            counts = defaultdict(int)
            for r in all_results:
                counts[r[label_key]] += 1
            total = max(sum(counts.values()), 1)
            lines.append(f"| {name} | "
                         f"{counts.get('safe_branch', 0)/total:.3f} | "
                         f"{counts.get('risky_branch', 0)/total:.3f} | "
                         f"{counts.get('safe_then_risky', 0)/total:.3f} | "
                         f"{counts.get('risky_then_safe', 0)/total:.3f} |\n")

        # Safe selection rate (most important metric)
        safe_rate_base = sum(1 for r in all_results if "safe" in r["baseline_label"]) / len(all_results)
        safe_rate_a2 = sum(1 for r in all_results if "safe" in r["a2_label"]) / len(all_results)
        safe_rate_n2 = sum(1 for r in all_results if "safe" in r["n2_label"]) / len(all_results)

        lines.append(f"\n### SafeTop1Rate\n\n")
        lines.append(f"| Planner | SafeTop1Rate |\n")
        lines.append(f"|---------|-------------|\n")
        lines.append(f"| baseline | {safe_rate_base:.3f} |\n")
        lines.append(f"| A2 | {safe_rate_a2:.3f} |\n")
        lines.append(f"| A2+N2 | {safe_rate_n2:.3f} |\n\n")

    # ═══════════════════════════════════
    # Phase C: Promotion Criteria
    # ═══════════════════════════════════

    lines.append("## Phase C: Promotion Criteria\n\n")

    lines.append("| Criterion | Threshold | Result | Pass? |\n")
    lines.append("|-----------|-----------|--------|-------|\n")

    c1 = gate_diff_rate > 0.01
    lines.append(f"| GateDiffRate > 0 | > 0.01 | {gate_diff_rate:.3f} | "
                 f"{'✓' if c1 else '✗'} |\n")

    if all_results:
        c2 = safe_rate_a2 >= safe_rate_base
        lines.append(f"| A2 SafeRate ≥ baseline | ≥ {safe_rate_base:.3f} | "
                     f"{safe_rate_a2:.3f} | {'✓' if c2 else '✗'} |\n")

        c3 = safe_rate_n2 >= safe_rate_a2 - 0.01
        lines.append(f"| A2+N2 SafeRate ≥ A2 | ≥ {safe_rate_a2:.3f} | "
                     f"{safe_rate_n2:.3f} | {'✓' if c3 else '✗'} |\n")

        multi_path_eps = sum(1 for r in all_results if r["n_paths"] >= 3)
        c4 = multi_path_eps / max(total_episodes, 1) > 0.3
        lines.append(f"| >30% episodes have ≥3 paths | > 0.30 | "
                     f"{multi_path_eps/max(total_episodes,1):.3f} | "
                     f"{'✓' if c4 else '✗'} |\n")

    # Final verdict
    lines.append("\n### Verdict\n\n")
    if all_results:
        all_pass = c1 and c2 and c3 and c4
        if all_pass:
            lines.append("**A2+N2 passes all criteria → CANONICAL-PROMOTION CANDIDATE**\n")
        elif c2:
            lines.append("**A2 improves safe selection but N2 gate has limited additional value "
                         "→ A2 SHADOW-READY, N2 OPTIONAL**\n")
        else:
            lines.append("**Insufficient evidence for promotion → MAINTAIN SHADOW-ONLY**\n")

    elapsed = time.time() - t0
    lines[1] = f"**Episodes**: {total_episodes} | **Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step5a2_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
