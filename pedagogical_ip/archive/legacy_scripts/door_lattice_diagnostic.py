"""
DoorLatticeSanityCheck — 50-seed diagnostic.

Tests whether UNLOCK actually changes bounded-planner discoverability.
Reports: TrapValid, DoorExpandRate, ΔPlanPrefix, UnlockUseful.

Usage:  python scripts/door_lattice_diagnostic.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.envs.benchmark_generator import generate_benchmark_map
from src.agents.planner_astar import bounded_astar
from src.envs.map_generator import CellType

N_SEEDS = 50
PLAN_PREFIX_K = 4  # compare first k steps


def run_diagnostic(seed: int, difficulty: str = "medium"):
    gm, cfg = generate_benchmark_map("door_lattice_sanity", seed, difficulty)
    H, W = gm.height, gm.width
    budget = cfg.search_budget  # same budget for both — no cheating

    # Build passable masks
    passable_wait = (gm.cell_types != CellType.WALL).copy()
    for dr, dc in gm.door_positions:
        passable_wait[dr, dc] = False  # locked doors impassable

    passable_unlock = passable_wait.copy()
    for dr, dc in gm.door_positions:
        passable_unlock[dr, dc] = True  # all doors opened

    # Learner's belief
    rm = np.full((H, W), cfg.prior_risk_mean)
    cm_wait = np.where(passable_wait, 1.0, np.inf)
    cm_unlock = np.where(passable_unlock, 1.0, np.inf)
    cv = np.full((H, W), cfg.prior_risk_var)

    goal = gm.object_spawn  # first objective

    # ── Plan WITHOUT unlock ──
    wait_plan = bounded_astar(
        gm.agent_start, goal, cm_wait, rm, cv,
        budget=budget, lambda_risk=3.0, passable_mask=passable_wait,
    ) or []
    wait_reaches = bool(wait_plan and wait_plan[-1] == goal)

    # ── Plan WITH unlock (same budget!) ──
    unlock_plan = bounded_astar(
        gm.agent_start, goal, cm_unlock, rm, cv,
        budget=budget, lambda_risk=3.0, passable_mask=passable_unlock,
    ) or []
    unlock_reaches = bool(unlock_plan and unlock_plan[-1] == goal)

    # ── TrapValid: goal not reachable without door, reachable with door ──
    trap_valid = (not wait_reaches) and unlock_reaches

    # ── DoorExpandRate: what fraction of door cells appear in the plan? ──
    door_set = set(gm.door_positions)
    wait_door_hits = sum(1 for p in wait_plan if p in door_set) if wait_plan else 0
    unlock_door_hits = sum(1 for p in unlock_plan if p in door_set) if unlock_plan else 0
    n_doors = max(len(door_set), 1)
    door_expand_wait = wait_door_hits / n_doors
    door_expand_unlock = unlock_door_hits / n_doors
    delta_door_expand = door_expand_unlock - door_expand_wait

    # ── ΔPlanPrefix: does unlock change the first k steps? ──
    wp_prefix = wait_plan[:PLAN_PREFIX_K] if wait_plan else []
    up_prefix = unlock_plan[:PLAN_PREFIX_K] if unlock_plan else []
    plan_prefix_changed = int(wp_prefix != up_prefix)

    # ── UnlockUseful: is unlock plan shorter or lower-risk? ──
    wait_len = len(wait_plan) - 1 if wait_plan else 999
    unlock_len = len(unlock_plan) - 1 if unlock_plan else 999
    wait_risk_sum = sum(gm.true_risk[r, c] for r, c in wait_plan) if wait_plan else 999
    unlock_risk_sum = sum(gm.true_risk[r, c] for r, c in unlock_plan) if unlock_plan else 999
    unlock_useful = int(
        unlock_reaches and (unlock_len < wait_len or unlock_risk_sum < wait_risk_sum)
    )

    return {
        "seed": seed,
        "n_doors": len(door_set),
        "wait_reaches": wait_reaches,
        "unlock_reaches": unlock_reaches,
        "wait_len": wait_len if wait_reaches else None,
        "unlock_len": unlock_len if unlock_reaches else None,
        "trap_valid": trap_valid,
        "delta_door_expand": delta_door_expand,
        "plan_prefix_changed": plan_prefix_changed,
        "unlock_useful": unlock_useful,
    }


if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        results = [run_diagnostic(seed, diff) for seed in range(N_SEEDS)]

        n = len(results)
        trap_valid_rate = sum(r["trap_valid"] for r in results) / n
        prefix_changed_rate = sum(r["plan_prefix_changed"] for r in results) / n
        unlock_useful_rate = sum(r["unlock_useful"] for r in results) / n
        avg_delta_expand = np.mean([r["delta_door_expand"] for r in results])
        avg_n_doors = np.mean([r["n_doors"] for r in results])
        wait_reach_rate = sum(r["wait_reaches"] for r in results) / n
        unlock_reach_rate = sum(r["unlock_reaches"] for r in results) / n

        wait_lens = [r["wait_len"] for r in results if r["wait_len"] is not None]
        unlock_lens = [r["unlock_len"] for r in results if r["unlock_len"] is not None]

        print(f"\n{'='*65}")
        print(f"  DoorLatticeSanityCheck — {diff}  ({n} seeds, budget={results[0].get('budget', 30)})")
        print(f"{'='*65}")
        print(f"  Avg doors per map:     {avg_n_doors:.1f}")
        print(f"  Wait reaches goal:     {wait_reach_rate*100:.0f}%")
        print(f"  Unlock reaches goal:   {unlock_reach_rate*100:.0f}%")
        print(f"  TrapValid rate:        {trap_valid_rate*100:.0f}%")
        print(f"  D_DoorExpandRate:      {avg_delta_expand:.3f}")
        print(f"  D_PlanPrefix changed:  {prefix_changed_rate*100:.0f}%")
        print(f"  UnlockUseful:          {unlock_useful_rate*100:.0f}%")
        if wait_lens:
            print(f"  Wait plan length:      {np.mean(wait_lens):.1f} (mean)")
        if unlock_lens:
            print(f"  Unlock plan length:    {np.mean(unlock_lens):.1f} (mean)")

        # Verdict
        if prefix_changed_rate > 0.3 and unlock_useful_rate > 0.3:
            print(f"\n  → SITUATION A: UNLOCK redirects planner (prefix Δ={prefix_changed_rate*100:.0f}%)")
        elif prefix_changed_rate < 0.1:
            print(f"\n  → SITUATION B: UNLOCK does NOT redirect planner (prefix Δ={prefix_changed_rate*100:.0f}%)")
        else:
            print(f"\n  → MIXED: partial redirection (prefix Δ={prefix_changed_rate*100:.0f}%)")
