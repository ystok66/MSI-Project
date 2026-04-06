"""Step 5A.1: Necessity Gate Hardening Audit.

Phase A: Compare N1/N2/N3 necessity gates on 6 scenarios
Phase B: Add M1/M2 monotonicity fixes on best gate
Phase C: Summary promotion audit

Usage:
  python scripts/run_step5a1_necessity_gate_audit.py --n_seeds 50
"""

from __future__ import annotations
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.agents.planner_risk_shadow import (
    PlannerRiskShadow, RiskShadowConfig, PathRiskProfile,
)
from src.agents.necessity_gate_variants import (
    compute_necessity_n1, compute_necessity_n2, compute_necessity_n3,
    apply_monotonicity_m1, apply_monotonicity_m2,
)


# ═══════════════════════════════════════════════════════════
# Same grid scenarios from Step 5A
# ═══════════════════════════════════════════════════════════

def make_grid(H=8, W=10, hazard_cells=None, wall_cells=None):
    cost_map = np.ones((H, W))
    risk_map = np.zeros((H, W))
    unc_map = np.full((H, W), 0.1)
    passable = np.ones((H, W), dtype=bool)
    if hazard_cells:
        for r, c, risk, unc in hazard_cells:
            risk_map[r, c] = risk
            unc_map[r, c] = unc
    if wall_cells:
        for r, c in wall_cells:
            passable[r, c] = False
    return cost_map, risk_map, unc_map, passable


def generate_paths_fork(H=8, W=10, start=(3, 0), goal=(3, 9)):
    path_a = [(start[0], c) for c in range(start[1], goal[1] + 1)]
    path_b = [(start[0], start[1])]
    path_b += [(r, start[1]) for r in range(start[0] - 1, -1, -1)]
    path_b += [(0, c) for c in range(start[1] + 1, goal[1] + 1)]
    path_b += [(r, goal[1]) for r in range(1, goal[0] + 1)]
    return path_a, path_b


SCENARIOS = {}

def _build():
    H, W = 8, 10
    start, goal = (3, 0), (3, 9)
    pa, pb = generate_paths_fork(H, W, start, goal)

    # hazard_belt
    hz = [(3, c, 0.6, 0.2) for c in range(4, 6)]
    SCENARIOS["hazard_belt"] = {
        "maps": make_grid(H, W, hz), "paths": [pa, pb],
        "start": start, "goal": goal, "t_max": 40}

    # hazard_belt_high
    hh = [(3, c, 0.9, 0.3) for c in range(4, 6)]
    SCENARIOS["hazard_belt_high"] = {
        "maps": make_grid(H, W, hh), "paths": [pa, pb],
        "start": start, "goal": goal, "t_max": 40}

    # deadline_gate
    SCENARIOS["deadline_gate"] = {
        "maps": make_grid(H, W), "paths": [pa, pb],
        "start": start, "goal": goal, "t_max": 12}

    # fork_trap (high unc, low risk)
    uc = [(3, c, 0.1, 0.8) for c in range(4, 6)]
    SCENARIOS["fork_trap"] = {
        "maps": make_grid(H, W, uc), "paths": [pa, pb],
        "start": start, "goal": goal, "t_max": 40}

    # fork_trap_necessary (same unc but detour blocked)
    walls = [(0, c) for c in range(3, 8)]
    SCENARIOS["fork_trap_necessary"] = {
        "maps": make_grid(H, W, uc, walls), "paths": [pa, pa],  # only one viable path
        "start": start, "goal": goal, "t_max": 40}

    # elcb_po
    mx = [(3, c, 0.3, 0.5) for c in range(3, 7)]
    SCENARIOS["elcb_po"] = {
        "maps": make_grid(H, W, mx), "paths": [pa, pb],
        "start": start, "goal": goal, "t_max": 35}

_build()


# ═══════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════

def evaluate_with_gate(scenario_name, gate_mode, mono_mode, seed, rng):
    """Run one episode with specified gate and monotonicity mode."""
    sc = SCENARIOS[scenario_name]
    cost, risk, unc, passable = sc["maps"]
    paths = sc["paths"]
    start, goal = sc["start"], sc["goal"]
    t_max = sc["t_max"]

    # Use base A2 shadow to get profiles
    shadow = PlannerRiskShadow(mode="A2")
    t = int(rng.integers(0, max(t_max // 2, 1)))

    # Evaluate all paths with base shadow
    profiles = []
    for p in paths:
        prof = shadow.evaluate_path(p, cost, risk, unc, passable, t, t_max, goal)
        profiles.append(prof)

    # Recompute necessity gate
    for i, prof in enumerate(profiles):
        if gate_mode == "original":
            g_n = 1.0 - prof.necessity  # original: g_N = 1 - necessity
        elif gate_mode == "N1":
            g_n = compute_necessity_n1(profiles, i)
        elif gate_mode == "N2":
            g_n = compute_necessity_n2(profiles, i)
        elif gate_mode == "N3":
            g_n = compute_necessity_n3(
                prof.path, passable, risk, t, t_max, goal)
        else:
            g_n = 1.0  # baseline: always full surcharge

        epi_unc = prof.epistemic_uncertainty

        # Apply monotonicity fix
        if mono_mode == "M1":
            all_uncs = [p.epistemic_uncertainty for p in profiles]
            epi_unc = apply_monotonicity_m1(epi_unc, all_uncs)
        elif mono_mode == "M2":
            epi_unc = apply_monotonicity_m2(epi_unc, prof.hazard_prob)

        # Recompute A2 score with new gate
        eff_unc = g_n * epi_unc
        cfg = shadow.cfg
        new_a2 = (cfg.lambda_cost * prof.expected_cost
                  + cfg.lambda_hazard * prof.hazard_prob
                  + cfg.lambda_uncertainty * eff_unc
                  + cfg.lambda_timeout * prof.timeout_prob
                  + cfg.lambda_detour * prof.detour_cost)

        prof.score_a2 = new_a2
        prof.effective_unc_surcharge = eff_unc
        prof.necessity = 1.0 - g_n  # for reporting

    # Choose best
    scores = [p.score_a2 for p in profiles]
    best_idx = int(np.argmin(scores))
    chosen = profiles[best_idx]

    # Simulate outcome
    crashed = float(rng.random() < chosen.hazard_prob)
    timed_out = float(chosen.path_length > (t_max - t))
    tbsr = float(not crashed and not timed_out)

    # Monotonicity check
    mono = shadow.check_risk_monotonicity(profiles)

    return {
        "tbsr": tbsr,
        "crashed": crashed,
        "timed_out": timed_out,
        "chosen_idx": best_idx,
        "gate_value": profiles[best_idx].effective_unc_surcharge,
        "monotonic": mono["monotonic"],
        "mono_violations": mono["violations"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=50)
    args = parser.parse_args()

    out = Path("results/step5a1_necessity_gate")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    t0 = time.time()

    scenarios = list(SCENARIOS.keys())

    # ═══════════════════════════════════════
    # Phase A: Necessity gate comparison
    # ═══════════════════════════════════════

    gates = ["baseline", "original", "N1", "N2", "N3"]
    results_a = defaultdict(list)

    lines = ["# Step 5A.1: Necessity Gate Hardening Audit\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds}\n\n")
    lines.append("## Phase A: Necessity Gate Comparison\n\n")
    lines.append("| Gate | Scenario | TBSR↑ | Crash↓ | Timeout↓ | Mono |\n")
    lines.append("|------|----------|-------|--------|----------|------|\n")

    for gate in gates:
        for scenario in scenarios:
            rs = [evaluate_with_gate(scenario, gate, "none", s, np.random.default_rng(s))
                  for s in range(args.n_seeds)]
            results_a[(gate, scenario)] = rs

            tbsr = np.mean([r["tbsr"] for r in rs])
            crash = np.mean([r["crashed"] for r in rs])
            timeout = np.mean([r["timed_out"] for r in rs])
            mono_ok = all([r["monotonic"] for r in rs])

            lines.append(f"| {gate} | {scenario} | {tbsr:.3f} | {crash:.3f} | "
                         f"{timeout:.3f} | {'✓' if mono_ok else '✗'} |\n")
            print(f"  {gate}/{scenario}: TBSR={tbsr:.3f} Crash={crash:.3f}",
                  file=sys.stderr)

    # ═══════════════════════════════════════
    # Phase A summary: necessity regression gap
    # ═══════════════════════════════════════

    lines.append("\n### Necessity Regression Gap\n\n")
    lines.append("| Gate | fork_trap TBSR | fork_trap_nec TBSR | Δ_nec | Passes? |\n")
    lines.append("|------|---------------|-------------------|------|--------|\n")

    for gate in gates:
        ft = results_a.get((gate, "fork_trap"), [])
        fn = results_a.get((gate, "fork_trap_necessary"), [])
        if ft and fn:
            ft_t = np.mean([r["tbsr"] for r in ft])
            fn_t = np.mean([r["tbsr"] for r in fn])
            delta = fn_t - np.mean([r["tbsr"] for r in results_a.get(("baseline", "fork_trap_necessary"), results_a.get(("original", "fork_trap_necessary"), []))])
            passes = fn_t >= 0.75  # Must not regress below ~baseline
            lines.append(f"| {gate} | {ft_t:.3f} | {fn_t:.3f} | {delta:+.3f} | "
                         f"{'✓' if passes else '✗'} |\n")

    # ═══════════════════════════════════════
    # Phase B: Monotonicity fixes on best gate
    # ═══════════════════════════════════════

    lines.append("\n## Phase B: Monotonicity Fixes (on N2 gate)\n\n")
    lines.append("| Mono | Scenario | TBSR↑ | Crash↓ | Mono_OK | Violations |\n")
    lines.append("|------|----------|-------|--------|---------|------------|\n")

    mono_modes = ["none", "M1", "M2"]
    for mono in mono_modes:
        for scenario in scenarios:
            rs = [evaluate_with_gate(scenario, "N2", mono, s, np.random.default_rng(s))
                  for s in range(args.n_seeds)]

            tbsr = np.mean([r["tbsr"] for r in rs])
            crash = np.mean([r["crashed"] for r in rs])
            mono_ok = all([r["monotonic"] for r in rs])
            viol = sum([r["mono_violations"] for r in rs])

            lines.append(f"| {mono} | {scenario} | {tbsr:.3f} | {crash:.3f} | "
                         f"{'✓' if mono_ok else '✗'} | {viol} |\n")
            print(f"  N2+{mono}/{scenario}: TBSR={tbsr:.3f}", file=sys.stderr)

    # ═══════════════════════════════════════
    # Phase C: Promotion summary
    # ═══════════════════════════════════════

    lines.append("\n## Phase C: Promotion Summary\n\n")

    # Find best gate for each criterion
    # Criterion 1: fork_trap TBSR preserved
    # Criterion 2: fork_trap_necessary not regressed
    # Criterion 3: monotonicity violations minimized

    lines.append("### Criteria Assessment\n\n")
    lines.append("| Gate | fork_trap ↑ | fork_trap_nec ≥ baseline | elcb_po ↑ | deadline ≥ parity | Mono |\n")
    lines.append("|------|------------|-------------------------|----------|-----------------|------|\n")

    base_fn = np.mean([r["tbsr"] for r in results_a.get(("baseline", "fork_trap_necessary"), [])])
    base_dl = np.mean([r["tbsr"] for r in results_a.get(("baseline", "deadline_gate"), [])])

    for gate in gates:
        ft = np.mean([r["tbsr"] for r in results_a.get((gate, "fork_trap"), [])])
        fn = np.mean([r["tbsr"] for r in results_a.get((gate, "fork_trap_necessary"), [])])
        ep = np.mean([r["tbsr"] for r in results_a.get((gate, "elcb_po"), [])])
        dl = np.mean([r["tbsr"] for r in results_a.get((gate, "deadline_gate"), [])])
        mono_all = all([r["monotonic"]
                        for sc in scenarios
                        for r in results_a.get((gate, sc), [])])

        c1 = "✓" if ft >= 0.85 else "✗"
        c2 = "✓" if fn >= base_fn - 0.03 else "✗"
        c3 = "✓" if ep >= 0.25 else "✗"
        c4 = "✓" if dl >= base_dl - 0.1 else "✗"
        c5 = "✓" if mono_all else "✗"

        lines.append(f"| {gate} | {c1} ({ft:.3f}) | {c2} ({fn:.3f}) | "
                     f"{c3} ({ep:.3f}) | {c4} ({dl:.3f}) | {c5} |\n")

    elapsed = time.time() - t0
    lines[1] = f"**Seeds**: {args.n_seeds} | **Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step5a1_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
