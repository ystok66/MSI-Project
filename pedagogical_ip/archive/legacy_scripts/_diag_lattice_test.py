"""
Lattice Test Baseline: No tutor, no time limit.

Test environment:
  1TTTT1
  S00001
  1TTTT1
  10000G
  1TTTT1

5x6, 2/3 T-rows have 1 trap each.
T_max = 100 (effectively infinite).
No teacher (WAIT only).

Also tests with S and G requiring T-row crossing:
  - Config A: S(1,0), G(3,5) — must cross 1 T-row
  - Config B: S(1,0), G(1,5) — same row, no crossing needed
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
TRAP_RISK = 1.0
T_MAX = 100  # effectively unlimited


def generate_lattice_test(seed, n_trap_rows=2, sg_cross=True):
    """
    5x6 lattice test grid.

    sg_cross=True:  S(1,0) G(3,5) — agent must cross T-row
    sg_cross=False: S(1,0) G(1,5) — agent stays on same row
    """
    rng = np.random.default_rng(seed)
    H, W = 5, 6
    ct, cost, risk = _empty_grid(H, W)

    # Walls: top/bottom edges of T-rows at col 0 and col 5
    wall_cells = [(r, 0) for r in range(H)] + [(r, W-1) for r in range(H)]
    _set_wall(ct, cost, wall_cells)

    # All interior cells passable
    for r in range(H):
        for c in range(1, W-1):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0

    # Place traps: choose n_trap_rows of T-rows {0,2,4}
    t_rows = [0, 2, 4]
    chosen = rng.choice(t_rows, size=n_trap_rows, replace=False)
    trap_cells = []
    for r in chosen:
        tc = rng.integers(1, W-1)  # cols 1-4
        trap_cells.append((int(r), int(tc)))
    _set_risky(ct, cost, risk, trap_cells, TRAP_RISK)

    # S and G
    agent_start = (1, 0)
    ct[1, 0] = CellType.NORMAL; cost[1, 0] = 1.0

    if sg_cross:
        target_pos = (3, W-1)
        ct[3, W-1] = CellType.NORMAL; cost[3, W-1] = 1.0
    else:
        target_pos = (1, W-1)
        ct[1, W-1] = CellType.NORMAL; cost[1, W-1] = 1.0

    object_spawn = target_pos

    gm = _build_gridmap(H, W, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    # BFS shortest avoiding traps
    shortest_safe = _bfs(gm, agent_start, target_pos, set(trap_cells))
    shortest_any = _bfs(gm, agent_start, target_pos, set())

    cfg = FamilyConfig(max_steps=T_MAX, risk_budget=1.0,
                       prior_risk_mean=0.02, prior_risk_var=0.20,
                       search_budget=30, budget_class=8)
    return gm, cfg, {"trap_cells": trap_cells,
                      "shortest_safe": shortest_safe,
                      "shortest_any": shortest_any}


def _bfs(gm, start, goal, avoid):
    H, W = gm.height, gm.width
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, d = queue.popleft()
        if pos == goal: return d
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            if (nr,nc) in visited: continue
            if gm.cell_types[nr,nc] == CellType.WALL: continue
            if (nr,nc) in avoid: continue
            visited.add((nr,nc))
            queue.append(((nr,nc), d+1))
    return 999


def run_episode(seed, sg_cross=True, epsilon=0.05):
    gm, cfg, meta = generate_lattice_test(seed, n_trap_rows=2, sg_cross=sg_cross)
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=T_MAX,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0, prior_cost_var=0.1,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, lambda_risk=0.8, lambda_uncertainty=0.02,
        seed=seed)
    env.reset()
    env.agent.epsilon_greedy = epsilon

    hit_trap = False
    success = False
    trap_set = set(meta["trap_cells"])

    for t in range(T_MAX):
        _, _, term, trunc, _ = env.step(0)
        pos = tuple(env.agent.pos)
        if pos in trap_set: hit_trap = True
        if term:
            if env.object_delivered or pos == gm.target_pos:
                success = True
            break
        if trunc: break

    return {"seed": seed, "success": success, "hit_trap": hit_trap,
            "steps": t+1, **meta}


if __name__ == "__main__":
    print("Lattice Test Baseline: No tutor, T_max=100")
    print(f"Seeds: {N_SEEDS}\n")

    for cross_label, sg_cross in [("Cross (S row1 -> G row3)", True),
                                   ("Same row (S row1 -> G row1)", False)]:
        for eps_val in [0.05, 0.0]:
            results = [run_episode(s, sg_cross=sg_cross, epsilon=eps_val)
                       for s in range(N_SEEDS)]
            n_succ = sum(1 for r in results if r["success"])
            n_trap = sum(1 for r in results if r["hit_trap"])
            avg_steps = np.mean([r["steps"] for r in results])
            avg_safe = np.mean([r["shortest_safe"] for r in results])
            avg_any = np.mean([r["shortest_any"] for r in results])

            print(f"--- {cross_label}, eps={eps_val} ---")
            print(f"  CSR:           {n_succ:3d}/{N_SEEDS} = {100*n_succ/N_SEEDS:.0f}%")
            print(f"  Hit trap:      {n_trap:3d}/{N_SEEDS} = {100*n_trap/N_SEEDS:.0f}%")
            print(f"  Avg steps:     {avg_steps:.1f}")
            print(f"  Avg safe path: {avg_safe:.1f}")
            print(f"  Avg any path:  {avg_any:.1f}")
            print()
