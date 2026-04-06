"""Step 5A: Risk-Sensitive Planner Shadow Experiment.

Tests three modes (A1, A2, A3) against baseline linear planner on
synthetic scenarios covering: hazard_belt, deadline_gate, fork_trap, ELCB-PO.

Diagnostics:
  - TBSR, crash rate, timeout rate, detour regret
  - Risk monotonicity validation
  - Necessity gate sanity check
  - Path agreement with baseline

Usage:
  python scripts/run_step5a_planner_shadow.py
  python scripts/run_step5a_planner_shadow.py --n_seeds 50
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


# ═══════════════════════════════════════════════════════════════
# Synthetic Grid Scenarios
# ═══════════════════════════════════════════════════════════════

def make_grid(H=8, W=10, hazard_cells=None, wall_cells=None):
    """Create synthetic cost/risk/uncertainty/passable maps."""
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
    """Fork trap: two paths, one risky short, one safe long."""
    # Path A: straight through (short, but passes through hazard belt)
    path_a = [(start[0], c) for c in range(start[1], goal[1] + 1)]
    # Path B: detour via top row (long, safe)
    path_b = [(start[0], start[1])]
    path_b += [(r, start[1]) for r in range(start[0] - 1, -1, -1)]
    path_b += [(0, c) for c in range(start[1] + 1, goal[1] + 1)]
    path_b += [(r, goal[1]) for r in range(1, goal[0] + 1)]
    return path_a, path_b


SCENARIOS = {}


def _build_scenarios():
    H, W = 8, 10
    start = (3, 0)
    goal = (3, 9)

    # ── hazard_belt: column 4-5 is hazardous ──
    hazards_belt = [(3, c, 0.6, 0.2) for c in range(4, 6)]
    cost, risk, unc, passable = make_grid(H, W, hazards_belt)
    pa, pb = generate_paths_fork(H, W, start, goal)
    SCENARIOS["hazard_belt"] = {
        "maps": (cost, risk, unc, passable),
        "paths": [pa, pb],
        "start": start, "goal": goal,
        "t_max": 40, "description": "hazard belt across short path",
    }

    # ── hazard_belt_high: same but risk=0.9 ──
    hazards_high = [(3, c, 0.9, 0.3) for c in range(4, 6)]
    cost2, risk2, unc2, pass2 = make_grid(H, W, hazards_high)
    SCENARIOS["hazard_belt_high"] = {
        "maps": (cost2, risk2, unc2, pass2),
        "paths": [pa, pb],
        "start": start, "goal": goal,
        "t_max": 40, "description": "high-risk hazard belt (risk=0.9)",
    }

    # ── deadline_gate: short path near deadline ──
    cost3, risk3, unc3, pass3 = make_grid(H, W)
    SCENARIOS["deadline_gate"] = {
        "maps": (cost3, risk3, unc3, pass3),
        "paths": [pa, pb],
        "start": start, "goal": goal,
        "t_max": 12,  # tight deadline favors short path
        "description": "tight deadline, short path needed",
    }

    # ── fork_trap: high uncertainty on short path (unknown cells) ──
    unc_cells = [(3, c, 0.1, 0.8) for c in range(4, 6)]
    cost4, risk4, unc4, pass4 = make_grid(H, W, unc_cells)
    SCENARIOS["fork_trap"] = {
        "maps": (cost4, risk4, unc4, pass4),
        "paths": [pa, pb],
        "start": start, "goal": goal,
        "t_max": 40, "description": "unknown cells (high unc, low risk) on short path",
    }

    # ── fork_trap_necessary: same unc but detour blocked ──
    walls = [(0, c) for c in range(3, 8)]  # block top row
    cost5, risk5, unc5, pass5 = make_grid(H, W, unc_cells, walls)
    pb_blocked = [(start[0], c) for c in range(start[1], goal[1] + 1)]
    SCENARIOS["fork_trap_necessary"] = {
        "maps": (cost5, risk5, unc5, pass5),
        "paths": [pa, pb_blocked],
        "start": start, "goal": goal,
        "t_max": 40, "description": "unknown cells, NO safe alternative (necessity=1)",
    }

    # ── elcb_po: partial observability, mixed risk+uncertainty ──
    mixed = [(3, c, 0.3, 0.5) for c in range(3, 7)]
    cost6, risk6, unc6, pass6 = make_grid(H, W, mixed)
    SCENARIOS["elcb_po"] = {
        "maps": (cost6, risk6, unc6, pass6),
        "paths": [pa, pb],
        "start": start, "goal": goal,
        "t_max": 35, "description": "partial observability (moderate risk + high unc)",
    }


_build_scenarios()


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_scenario(scenario_name, mode, n_seeds, rng):
    """Evaluate a scoring mode on a scenario."""
    sc = SCENARIOS[scenario_name]
    cost, risk, unc, passable = sc["maps"]
    paths = sc["paths"]
    start, goal = sc["start"], sc["goal"]
    t_max = sc["t_max"]

    shadow = PlannerRiskShadow(mode=mode)

    # Evaluate at different timesteps
    results = []
    for seed in range(n_seeds):
        t = int(rng.integers(0, max(t_max // 2, 1)))

        _, profiles = shadow.rank_paths(
            paths, cost, risk, unc, passable, t, t_max, goal)

        best_idx = int(np.argmin([shadow.score(p) for p in profiles]))

        # Simulate: did we crash? timeout?
        chosen = profiles[best_idx]
        crashed = float(rng.random() < chosen.hazard_prob)
        timed_out = float(chosen.path_length > (t_max - t))

        # TBSR: reached goal safely without timeout
        tbsr = float(not crashed and not timed_out)

        # Detour regret
        shortest = min(p.path_length for p in profiles)
        detour_regret = chosen.path_length - shortest

        # Risk monotonicity
        mono = shadow.check_risk_monotonicity(profiles)

        # Necessity sanity
        nec_check = shadow.check_necessity_sanity(profiles)

        results.append({
            "tbsr": tbsr,
            "crashed": crashed,
            "timed_out": timed_out,
            "detour_regret": detour_regret,
            "chosen_idx": best_idx,
            "necessity": chosen.necessity,
            "monotonic": mono["monotonic"],
            "nec_gate_ok": nec_check.get("gate_working"),
            "score": shadow.score(chosen),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    args = parser.parse_args()

    out = Path("results/step5a_planner")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    modes = ["A1", "A2", "A3"]
    scenarios = list(SCENARIOS.keys())

    t0 = time.time()
    lines = ["# Step 5A: Risk-Sensitive Planner Shadow Results\n\n"]
    lines.append(f"**Seeds**: {args.n_seeds}\n\n")

    # ── Baseline: use A1 with lambda_uncertainty=0 as "linear" baseline ──
    modes_all = ["baseline"] + modes

    lines.append("## Headline Metrics\n\n")
    lines.append("| Mode | Scenario | TBSR↑ | Crash↓ | Timeout↓ | Detour↓ | Mono | NecGate |\n")
    lines.append("|------|----------|-------|--------|----------|---------|------|--------|\n")

    all_data = {}
    for mode in modes_all:
        for scenario in scenarios:
            if mode == "baseline":
                shadow = PlannerRiskShadow(
                    mode="A1",
                    config=RiskShadowConfig(lambda_uncertainty=0.0))
                sc = SCENARIOS[scenario]
                cost, risk, unc, passable = sc["maps"]
                paths = sc["paths"]

                results = []
                for seed in range(args.n_seeds):
                    t = int(rng.integers(0, max(sc["t_max"] // 2, 1)))
                    _, profiles = shadow.rank_paths(
                        paths, cost, risk, unc, passable,
                        t, sc["t_max"], sc["goal"])
                    best_idx = int(np.argmin([shadow.score(p) for p in profiles]))
                    chosen = profiles[best_idx]
                    crashed = float(rng.random() < chosen.hazard_prob)
                    timed_out = float(chosen.path_length > (sc["t_max"] - t))
                    tbsr = float(not crashed and not timed_out)
                    shortest = min(p.path_length for p in profiles)
                    results.append({
                        "tbsr": tbsr, "crashed": crashed,
                        "timed_out": timed_out,
                        "detour_regret": chosen.path_length - shortest,
                        "monotonic": True, "nec_gate_ok": None,
                    })
            else:
                results = evaluate_scenario(scenario, mode, args.n_seeds, rng)

            all_data[(mode, scenario)] = results

            # Aggregate
            tbsr = np.mean([r["tbsr"] for r in results])
            crash = np.mean([r["crashed"] for r in results])
            timeout = np.mean([r["timed_out"] for r in results])
            detour = np.mean([r["detour_regret"] for r in results])
            mono = all([r["monotonic"] for r in results])
            nec_vals = [r.get("nec_gate_ok") for r in results if r.get("nec_gate_ok") is not None]
            nec_ok = all(nec_vals) if nec_vals else "N/A"

            lines.append(f"| {mode} | {scenario} | {tbsr:.3f} | {crash:.3f} | "
                         f"{timeout:.3f} | {detour:.1f} | "
                         f"{'✓' if mono else '✗'} | "
                         f"{'✓' if nec_ok == True else ('✗' if nec_ok == False else 'N/A')} |\n")

            print(f"  {mode}/{scenario}: TBSR={tbsr:.3f} Crash={crash:.3f} "
                  f"Timeout={timeout:.3f} Detour={detour:.1f}",
                  file=sys.stderr)

    # ── Promotion analysis ──
    lines.append("\n## Promotion Analysis\n\n")
    for scenario in scenarios:
        base = all_data.get(("baseline", scenario), [])
        a2 = all_data.get(("A2", scenario), [])
        if base and a2:
            bt = np.mean([r["tbsr"] for r in base])
            at = np.mean([r["tbsr"] for r in a2])
            bc = np.mean([r["crashed"] for r in base])
            ac = np.mean([r["crashed"] for r in a2])
            lines.append(f"### {scenario}\n")
            lines.append(f"- TBSR: {at:.3f} vs {bt:.3f} "
                         f"{'**BETTER**' if at > bt + 0.01 else ('WORSE' if at < bt - 0.01 else 'PARITY')}\n")
            lines.append(f"- Crash: {ac:.3f} vs {bc:.3f} "
                         f"{'**BETTER**' if ac < bc - 0.01 else ('WORSE' if ac > bc + 0.01 else 'PARITY')}\n\n")

    elapsed = time.time() - t0
    lines[1] = f"**Seeds**: {args.n_seeds} | **Elapsed**: {elapsed:.1f}s\n\n"

    rpt = out / "step5a_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt} ({elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
