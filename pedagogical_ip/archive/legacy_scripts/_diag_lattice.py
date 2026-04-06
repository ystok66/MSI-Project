"""
Lattice Trap Diagnostic: No-teacher survival test.

Test environment:
  1TTTT1
  100001
  1TTTT1
  100001
  1TTTT1

5x6 grid. 3 T-rows (rows 0,2,4), 2 free corridors (rows 1,3).
2 of 3 T-rows have exactly one trap cell each.
S at (1,1), G at (3,4).

Agent must cross T-rows to go from row 1 to row 3 (or use T-row shortcut).
Shortest safe path avoids trap cells.

Sweep T_max = {1.2, 1.4, 1.6, 1.8} * shortest_safe_path_length.
200 seeds per condition. No teacher (WAIT only).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import deque
from src.envs.map_generator import CellType, GridMap
from src.envs.map_families import (
    _empty_grid, _set_wall, _set_risky, _build_gridmap, FamilyConfig,
)
from src.envs.pedagogical_grid import PedagogicalGridEnv

N_SEEDS = 200
TRAP_RISK = 1.0  # deterministic death


def generate_lattice_test(
    seed: int = 0,
    n_trap_rows: int = 2,
    t_max_override: int = None,
):
    """
    Generate the 5x6 lattice test environment.

    1TTTT1     Row 0: T-row (wall borders, T cells in cols 1-4)
    S00001     Row 1: free corridor, S at (1,0)
    1TTTT1     Row 2: T-row
    10000G     Row 3: free corridor, G at (3,5)
    1TTTT1     Row 4: T-row

    n_trap_rows of 3 T-rows (rows 0,2,4) get exactly one trap cell each.
    Trap position within each chosen row is random (cols 1-4).
    """
    rng = np.random.default_rng(seed)
    H, W = 5, 6
    ct, cost, risk = _empty_grid(H, W)

    # ── Set walls: borders at col 0 and col 5 ──
    wall_cells = []
    for r in range(H):
        wall_cells.append((r, 0))
        wall_cells.append((r, W - 1))
    _set_wall(ct, cost, wall_cells)

    # ── Free corridors: rows 1, 3 ──
    for r in [1, 3]:
        for c in range(1, W - 1):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0

    # ── T-rows: rows 0, 2, 4 — all cells passable, some risky ──
    t_rows = [0, 2, 4]
    for r in t_rows:
        for c in range(1, W - 1):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0

    # ── Place traps: choose n_trap_rows of the 3 T-rows ──
    chosen_rows = rng.choice(t_rows, size=n_trap_rows, replace=False)
    trap_cells = []
    for r in chosen_rows:
        trap_col = rng.integers(1, W - 1)  # cols 1-4
        trap_cells.append((int(r), int(trap_col)))
    _set_risky(ct, cost, risk, trap_cells, TRAP_RISK)

    # ── Start and Goal ──
    # S at left side of free corridor, G at right side of other corridor
    agent_start = (1, 0)  # overlaps wall — make it passable
    ct[1, 0] = CellType.NORMAL
    cost[1, 0] = 1.0

    target_pos = (3, W - 1)  # overlaps wall — make it passable
    ct[3, W - 1] = CellType.NORMAL
    cost[3, W - 1] = 1.0

    object_spawn = target_pos  # single-goal mode

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    # ── Compute shortest safe path (BFS avoiding trap cells) ──
    shortest_safe = _bfs_shortest(gm, agent_start, target_pos, trap_cells)
    shortest_any = _bfs_shortest(gm, agent_start, target_pos, [])

    if t_max_override is not None:
        t_max = t_max_override
    else:
        t_max = int(1.4 * shortest_safe) if shortest_safe < 999 else 20

    cfg = FamilyConfig(
        max_steps=t_max,
        risk_budget=1.0,
        prior_risk_mean=0.02,
        prior_risk_var=0.20,
        search_budget=30,
        budget_class=8,
    )
    return gm, cfg, {
        "trap_cells": trap_cells,
        "shortest_safe": shortest_safe,
        "shortest_any": shortest_any,
    }


def _bfs_shortest(gm, start, goal, avoid_cells):
    """BFS shortest path, avoiding specified cells."""
    H, W = gm.height, gm.width
    avoid_set = set(avoid_cells)
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)

    while queue:
        pos, dist = queue.popleft()
        if pos == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            if gm.cell_types[nr, nc] == CellType.WALL:
                continue
            if (nr, nc) in avoid_set:
                continue
            visited.add((nr, nc))
            queue.append(((nr, nc), dist + 1))
    return 999  # unreachable


def run_episode(seed, t_max_override=None, epsilon=0.05):
    """Run one episode, no teacher."""
    gm, cfg, meta = generate_lattice_test(seed=seed, n_trap_rows=2,
                                           t_max_override=t_max_override)
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0, prior_cost_var=0.1,
        prior_risk_mean=cfg.prior_risk_mean, prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget, lambda_risk=0.8, lambda_uncertainty=0.02,
        seed=seed,
    )
    env.reset()
    env.agent.epsilon_greedy = epsilon

    hit_trap = False
    success = False
    fail_cause = "timeout"

    for t in range(cfg.max_steps):
        obs, reward, terminated, truncated, info = env.step(0)
        pos = tuple(env.agent.pos)

        if pos in set(meta["trap_cells"]):
            hit_trap = True

        if terminated:
            if env.risk_budget_left <= 0:
                fail_cause = "fatal"
            elif env.object_delivered or pos == cfg.__dict__.get("target_pos",
                                                                  gm.target_pos):
                fail_cause = "success"
                success = True
            break
        if truncated:
            fail_cause = "timeout"
            break

    return {
        "seed": seed,
        "success": success,
        "hit_trap": hit_trap,
        "fail_cause": fail_cause,
        "steps": t + 1,
        "t_max": cfg.max_steps,
        "shortest_safe": meta["shortest_safe"],
        "shortest_any": meta["shortest_any"],
        "trap_cells": meta["trap_cells"],
    }


if __name__ == "__main__":
    print("Lattice Trap Diagnostic: No-Teacher Survival")
    print(f"Grid: 5x6, 2/3 T-rows have traps, rho*={TRAP_RISK}")
    print(f"Seeds: {N_SEEDS}\n")

    # First, compute typical shortest safe path
    safe_lengths = []
    any_lengths = []
    for s in range(N_SEEDS):
        _, _, meta = generate_lattice_test(seed=s, n_trap_rows=2)
        safe_lengths.append(meta["shortest_safe"])
        any_lengths.append(meta["shortest_any"])
    avg_safe = np.mean(safe_lengths)
    avg_any = np.mean(any_lengths)
    print(f"Avg shortest_any:  {avg_any:.1f}")
    print(f"Avg shortest_safe: {avg_safe:.1f}")
    print(f"Safe path reachable: {sum(1 for x in safe_lengths if x < 999)}/{N_SEEDS}")
    print()

    # Sweep T_max ratios
    ratios = [1.2, 1.4, 1.6, 1.8]
    print(f"{'Ratio':>6s} {'T_max':>6s} {'CSR':>6s} {'Fatal':>6s} {'Timeout':>8s} "
          f"{'HitTrap':>8s} {'AvgStep':>8s}")
    print("-" * 55)

    for ratio in ratios:
        results = []
        for s in range(N_SEEDS):
            _, _, meta = generate_lattice_test(seed=s, n_trap_rows=2)
            t_max = max(int(ratio * meta["shortest_safe"]), meta["shortest_safe"] + 1) \
                    if meta["shortest_safe"] < 999 else 20
            r = run_episode(s, t_max_override=t_max)
            results.append(r)

        n_succ = sum(1 for r in results if r["success"])
        n_fatal = sum(1 for r in results if r["fail_cause"] == "fatal")
        n_timeout = sum(1 for r in results if r["fail_cause"] == "timeout")
        n_trap = sum(1 for r in results if r["hit_trap"])
        avg_steps = np.mean([r["steps"] for r in results])
        avg_tmax = np.mean([r["t_max"] for r in results])

        print(f"{ratio:6.1f} {avg_tmax:6.0f} {100*n_succ/N_SEEDS:5.0f}% "
              f"{n_fatal:5d} {n_timeout:7d} {n_trap:7d} {avg_steps:8.1f}")

    # Also show some example seeds
    print(f"\nExample layouts (first 5 seeds):")
    for s in range(5):
        _, _, meta = generate_lattice_test(seed=s, n_trap_rows=2)
        print(f"  seed={s}: traps={meta['trap_cells']}  "
              f"shortest_safe={meta['shortest_safe']}  shortest_any={meta['shortest_any']}")
