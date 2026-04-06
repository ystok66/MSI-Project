"""
Training Lattice Environment: 5 rows x 16 cols, doors, traps, time limits.

Layout (user specification):
  1DTTTTDTTTTDTTT1     Row 0: T-row with doors
  10000D0000D00001     Row 1: free corridor with doors
  1DTTTTDTTTTDTTT1     Row 2: T-row with doors
  10000D0000D00001     Row 3: free corridor with doors
  1DTTTTDTTTTDTTT1     Row 4: T-row with doors

The grid has 3 segments separated by door columns.
S at left side, G at right side (on free corridors).

Difficulties:
  Easy:   each segment 60% chance of 1 T-row having trap, max 1 per segment
  Medium: each T-row 60% chance of trap, at least 1 T-row safe per segment
  Hard:   each segment has exactly 2 T-rows with traps

Doors start OPEN. Tutor can CLOSE doors (block them) to prevent agent from
entering dangerous passages. Tutor must balance:
  - Letting agent learn about risk (exploration value)
  - Preventing agent from dying (safety)
  - Time constraints (blocking forces detours)

Tutor actions: WAIT (do nothing) or CLOSE_DOOR (block a door cell temporarily)

Diagnostic: compare no-tutor vs oracle-tutor at different time ratios.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import deque
from src.envs.map_generator import CellType, GridMap
from src.envs.map_families import (
    _empty_grid, _set_wall, _set_risky, _set_door, _build_gridmap, FamilyConfig,
)
from src.envs.pedagogical_grid import PedagogicalGridEnv

N_SEEDS = 200
TRAP_RISK = 1.0

# ── Full layout constants ──
#   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
# 0 W  D  T  T  T  T  D  T  T  T  T  D  T  T  T  W
# 1 W  .  .  .  .  D  .  .  .  .  D  .  .  .  .  W
# 2 W  D  T  T  T  T  D  T  T  T  T  D  T  T  T  W
# 3 W  .  .  .  .  D  .  .  .  .  D  .  .  .  .  W
# 4 W  D  T  T  T  T  D  T  T  T  T  D  T  T  T  W

H, W_FULL = 5, 16
T_ROWS = [0, 2, 4]
FREE_ROWS = [1, 3]

# Door column positions (where segments connect)
# In T-rows: cols 1, 6, 11 are doors
# In free-rows: cols 5, 10 are doors
T_ROW_DOOR_COLS = [1, 6, 11]
FREE_ROW_DOOR_COLS = [5, 10]

# Segments of T-cells (cols after each T-row door)
# Segment 0: cols 2-5
# Segment 1: cols 7-10
# Segment 2: cols 12-14
SEGMENTS = [
    list(range(2, 6)),    # cols 2,3,4,5
    list(range(7, 11)),   # cols 7,8,9,10
    list(range(12, 15)),  # cols 12,13,14
]


def generate_training_lattice(seed, difficulty="medium"):
    """
    Generate 5x16 training lattice.

    Returns: (GridMap, FamilyConfig, metadata dict)
    """
    rng = np.random.default_rng(seed)
    ct, cost, risk = _empty_grid(H, W_FULL)

    # ── Walls: col 0 and col 15 ──
    wall_cells = [(r, 0) for r in range(H)] + [(r, W_FULL-1) for r in range(H)]
    _set_wall(ct, cost, wall_cells)

    # ── All interior cells are passable ──
    for r in range(H):
        for c in range(1, W_FULL-1):
            ct[r, c] = CellType.NORMAL
            cost[r, c] = 1.0

    # ── Doors (start as OPEN / passable — tutor can close them) ──
    # We mark door positions but DON'T lock them.
    # Instead we track them as metadata so tutor knows where doors are.
    door_positions_all = []
    for r in T_ROWS:
        for c in T_ROW_DOOR_COLS:
            door_positions_all.append((r, c))
    for r in FREE_ROWS:
        for c in FREE_ROW_DOOR_COLS:
            door_positions_all.append((r, c))

    # ── Place traps based on difficulty ──
    trap_cells = []
    for seg_idx, seg_cols in enumerate(SEGMENTS):
        n_t_cols = len(seg_cols)  # 4 or 3 cells per T-row in this segment

        if difficulty == "easy":
            # 60% chance one T-row in this segment has a trap, max 1
            if rng.random() < 0.6:
                chosen_row = rng.choice(T_ROWS)
                trap_col = rng.choice(seg_cols)
                trap_cells.append((int(chosen_row), int(trap_col)))

        elif difficulty == "medium":
            # Each T-row has 60% chance of trap, but ensure at least 1 safe
            rows_with_trap = [r for r in T_ROWS if rng.random() < 0.6]
            # Ensure at least one row is safe
            if len(rows_with_trap) == len(T_ROWS):
                rows_with_trap.remove(rng.choice(rows_with_trap))
            for r in rows_with_trap:
                trap_col = rng.choice(seg_cols)
                trap_cells.append((int(r), int(trap_col)))

        elif difficulty == "hard":
            # Exactly 2 T-rows have traps per segment
            chosen_rows = rng.choice(T_ROWS, size=2, replace=False)
            for r in chosen_rows:
                trap_col = rng.choice(seg_cols)
                trap_cells.append((int(r), int(trap_col)))

    _set_risky(ct, cost, risk, trap_cells, TRAP_RISK)

    # ── S and G ──
    agent_start = (1, 0)
    ct[1, 0] = CellType.NORMAL; cost[1, 0] = 1.0
    target_pos = (3, W_FULL-1)
    ct[3, W_FULL-1] = CellType.NORMAL; cost[3, W_FULL-1] = 1.0
    object_spawn = target_pos  # single-goal

    # Don't register doors as locked — they start open
    gm = _build_gridmap(H, W_FULL, ct, cost, risk,
                         agent_start, object_spawn, target_pos, [])

    # ── Shortest paths ──
    shortest_safe = _bfs(gm, agent_start, target_pos, set(trap_cells))
    shortest_any = _bfs(gm, agent_start, target_pos, set())

    cfg = FamilyConfig(max_steps=100, risk_budget=1.0,
                       prior_risk_mean=0.02, prior_risk_var=0.20,
                       search_budget=30, budget_class=8)

    return gm, cfg, {
        "trap_cells": trap_cells,
        "door_positions": door_positions_all,
        "shortest_safe": shortest_safe,
        "shortest_any": shortest_any,
        "difficulty": difficulty,
    }


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


def run_episode(seed, difficulty="medium", t_max=None, tutor_mode="none",
                epsilon=0.05):
    """
    Run one episode.

    tutor_mode:
      "none":         WAIT every step
      "oracle_close":  Close doors leading to trapped T-rows at the right time
    """
    gm, cfg, meta = generate_training_lattice(seed, difficulty)
    actual_tmax = t_max if t_max else cfg.max_steps
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=actual_tmax,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0, prior_cost_var=0.1,
        prior_risk_mean=0.02, prior_risk_var=0.20,
        search_budget=30, lambda_risk=0.8, lambda_uncertainty=0.02,
        seed=seed)
    env.reset()
    env.agent.epsilon_greedy = epsilon

    trap_set = set(meta["trap_cells"])
    door_set = set(meta["door_positions"])

    # For oracle tutor: identify which doors lead to trapped T-rows
    # A T-row door at (r, c) is dangerous if row r has any trap in any segment
    dangerous_doors = set()
    if tutor_mode == "oracle_close":
        trapped_rows = set(r for r, c in meta["trap_cells"])
        for r, c in meta["door_positions"]:
            if r in trapped_rows and r in T_ROWS:
                dangerous_doors.add((r, c))

    hit_trap = False
    success = False
    doors_closed = 0

    for t in range(actual_tmax):
        agent_pos = tuple(env.agent.pos)

        # Oracle tutor: close dangerous doors when agent is near
        action = 0  # WAIT
        if tutor_mode == "oracle_close":
            # Check if agent is about to enter a dangerous door
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                adj = (agent_pos[0]+dr, agent_pos[1]+dc)
                if adj in dangerous_doors and adj not in env.blocked_cells:
                    # Block this door
                    env.block_target = adj
                    action = 4  # BLOCK_PATH
                    doors_closed += 1
                    dangerous_doors.discard(adj)
                    break

        obs, reward, term, trunc, info = env.step(action)
        pos = tuple(env.agent.pos)
        if pos in trap_set: hit_trap = True
        if term:
            if env.object_delivered or pos == gm.target_pos:
                success = True
            break
        if trunc: break

    return {"seed": seed, "success": success, "hit_trap": hit_trap,
            "steps": t+1, "doors_closed": doors_closed,
            "t_max": actual_tmax, **meta}


if __name__ == "__main__":
    print("Training Lattice Diagnostic")
    print(f"Grid: 5x16, 3 segments, doors start open")
    print(f"Seeds: {N_SEEDS}\n")

    for diff in ["easy", "medium", "hard"]:
        # Compute avg paths
        safes, anys = [], []
        for s in range(N_SEEDS):
            _, _, m = generate_training_lattice(s, diff)
            safes.append(m["shortest_safe"])
            anys.append(m["shortest_any"])
        avg_safe = np.mean(safes)
        avg_any = np.mean(anys)
        n_traps_avg = np.mean([len(m["trap_cells"])
                                for s in range(N_SEEDS)
                                for _, _, m in [generate_training_lattice(s, diff)]])

        print(f"=== Difficulty: {diff} ===")
        print(f"  Avg traps: {n_traps_avg:.1f}  "
              f"Avg shortest_any: {avg_any:.1f}  Avg shortest_safe: {avg_safe:.1f}")

        for ratio in [1.2, 1.4, 1.6, 1.8]:
            for tutor in ["none", "oracle_close"]:
                results = []
                for s in range(N_SEEDS):
                    _, _, m = generate_training_lattice(s, diff)
                    t_max = max(int(ratio * m["shortest_safe"]),
                                m["shortest_any"] + 1) \
                            if m["shortest_safe"] < 999 else 30
                    r = run_episode(s, diff, t_max=t_max,
                                    tutor_mode=tutor, epsilon=0.05)
                    results.append(r)

                n_succ = sum(1 for r in results if r["success"])
                n_trap = sum(1 for r in results if r["hit_trap"])
                n_to = sum(1 for r in results if not r["success"] and not r["hit_trap"])
                avg_dc = np.mean([r["doors_closed"] for r in results])

                print(f"  ratio={ratio:.1f} tutor={tutor:13s}  "
                      f"CSR={100*n_succ/N_SEEDS:4.0f}%  "
                      f"trap={n_trap:3d}  timeout={n_to:3d}  "
                      f"doors_closed={avg_dc:.1f}")
        print()
