"""Batch B Investigation: 100-seed reachability & latent-mode semantic audit.

This is a DIAGNOSTIC script — no production code changes.
It generates scenarios across families and checks:
1. Goal reachability (BFS)
2. Safe path existence
3. Metadata consistency (shortest_any/shortest_safe vs BFS)
4. Latent-mode risk mismatch (WorldWeights risk vs executed risk)
5. fork_trap safe_row==1 frequency and impact
"""

import sys
import numpy as np
from collections import defaultdict

# Ensure project on path
sys.path.insert(0, '.')

from src.envs.scenario_families import generate_scenario, SCENARIO_REGISTRY
from src.envs.lattice_v2 import _bfs_len
from src.envs.map_generator import CellType


def bfs_reachable(passable, start, goal):
    """BFS: is goal reachable from start?"""
    H, W = passable.shape
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and passable[nr, nc]:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def bfs_shortest(passable, start, goal):
    """BFS shortest path length (or -1 if unreachable)."""
    H, W = passable.shape
    visited = set()
    queue = [(start, 0)]
    visited.add(start)
    while queue:
        (r, c), d = queue.pop(0)
        if (r, c) == goal:
            return d
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and passable[nr, nc]:
                visited.add((nr, nc))
                queue.append(((nr, nc), d+1))
    return -1


def bfs_shortest_safe(passable, cell_types, start, goal):
    """BFS avoiding RISKY cells (or -1 if no safe path)."""
    H, W = passable.shape
    visited = set()
    queue = [(start, 0)]
    visited.add(start)
    while queue:
        (r, c), d = queue.pop(0)
        if (r, c) == goal:
            return d
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and passable[nr, nc]:
                if cell_types[nr, nc] != CellType.RISKY or (nr, nc) == goal:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), d+1))
    return -1


FAMILIES_TO_TEST = [
    "baseline_v2",
    "fork_trap",
    "hazard_belt",
    "deadline_gate",
    "delayed_corridor",
    "distractor_cue",
    "harder_baseline_v2",
]

N_SEEDS = 100


def run_reachability_audit():
    """B1/B3: 100-seed reachability + metadata consistency."""
    print("=" * 70)
    print("BATCH B INVESTIGATION: 100-seed reachability audit")
    print("=" * 70)

    results = {}
    for family in FAMILIES_TO_TEST:
        stats = {
            "goal_reachable": 0,
            "safe_path_exists": 0,
            "metadata_shortest_match": 0,
            "metadata_safe_match": 0,
            "total": 0,
            "fork_safe_row_1": 0,  # fork_trap specific
            "fork_safe_row_1_broken": 0,
        }

        for seed in range(N_SEEDS):
            try:
                gm, cfg, meta, sc = generate_scenario(family, seed=seed, latent_mode=True)
            except Exception as e:
                print(f"  [{family}] seed={seed} GENERATION FAILED: {e}")
                continue

            stats["total"] += 1
            H, W = gm.height, gm.width
            passable = np.ones((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    if gm.cell_types[r, c] == CellType.WALL:
                        passable[r, c] = False

            start = getattr(gm, 'agent_start', (2, 1)) or (2, 1)
            goal = getattr(gm, 'target_pos', (2, W - 2)) or (2, W - 2)

            # 1. Goal reachable
            if bfs_reachable(passable, start, goal):
                stats["goal_reachable"] += 1

            # 2. Safe path exists
            safe_len = bfs_shortest_safe(passable, gm.cell_types, start, goal)
            if safe_len > 0:
                stats["safe_path_exists"] += 1

            # 3. Metadata consistency
            bfs_any = bfs_shortest(passable, start, goal)
            meta_any = getattr(meta, 'shortest_any', None)
            if meta_any is not None and bfs_any == meta_any:
                stats["metadata_shortest_match"] += 1

            meta_safe = getattr(meta, 'shortest_safe', None)
            if meta_safe is not None and safe_len == meta_safe:
                stats["metadata_safe_match"] += 1
            elif meta_safe is None:
                stats["metadata_safe_match"] += 1  # no metadata to check

            # 4. fork_trap specific: safe_row check
            if family == "fork_trap":
                # Check which row is safe by looking at segments
                if hasattr(meta, 'segments') and meta.segments:
                    seg = meta.segments[0]
                    if hasattr(seg, 'risky_lane') and seg.risky_lane is not None:
                        safe_r = 3 if seg.risky_lane == 1 else 1
                    else:
                        safe_r = -1
                    if safe_r == 1:
                        stats["fork_safe_row_1"] += 1
                        # Check if detour is actually reachable from row 1
                        if safe_len < 0:
                            stats["fork_safe_row_1_broken"] += 1

        results[family] = stats

    # Print summary
    print()
    print(f"{'Family':<25} {'Goals':>6} {'Safe':>6} {'MetaOK':>7} {'SafeOK':>7} {'Total':>6}")
    print("-" * 70)
    for family in FAMILIES_TO_TEST:
        s = results[family]
        t = max(s["total"], 1)
        print(f"{family:<25} {s['goal_reachable']:>5}/{t:<1} {s['safe_path_exists']:>5}/{t:<1} "
              f"{s['metadata_shortest_match']:>6}/{t:<1} {s['metadata_safe_match']:>6}/{t:<1} {t:>6}")

    # Fork trap specific
    if "fork_trap" in results:
        s = results["fork_trap"]
        t = max(s["total"], 1)
        print()
        print(f"Fork trap: safe_row==1 frequency: {s['fork_safe_row_1']}/{t} "
              f"({100*s['fork_safe_row_1']/t:.0f}%)")
        print(f"Fork trap: safe_row==1 broken (no safe path): {s['fork_safe_row_1_broken']}/{s['fork_safe_row_1']}")

    return results


def run_latent_mode_audit():
    """B2: Latent-mode risk semantic audit."""
    print()
    print("=" * 70)
    print("BATCH B INVESTIGATION: latent_mode risk semantic audit")
    print("=" * 70)

    from src.agents.cost_risk_model import WorldWeights

    FAMILIES = ["baseline_v2", "fork_trap", "hazard_belt", "deadline_gate",
                "harder_baseline_v2"]
    N = 30

    for family in FAMILIES:
        mismatches = []
        overrides = 0
        total_risky = 0
        pure_latent_count = 0

        for seed in range(N):
            try:
                gm, cfg, meta, sc = generate_scenario(family, seed=seed, latent_mode=True)
            except Exception:
                continue

            H, W = gm.height, gm.width
            ww = getattr(gm, 'world_weights', None)
            if ww is None:
                continue

            for r in range(H):
                for c in range(W):
                    if gm.cell_types[r, c] in (CellType.RISKY, CellType.NORMAL):
                        z = gm.features[r, c]
                        ww_risk = ww.true_risk(z)
                        exec_risk = float(gm.true_risk[r, c])

                        if gm.cell_types[r, c] == CellType.RISKY:
                            total_risky += 1
                            diff = abs(exec_risk - ww_risk)
                            if diff > 0.01:
                                overrides += 1
                                mismatches.append(diff)
                            else:
                                pure_latent_count += 1

        if total_risky > 0:
            override_rate = 100 * overrides / total_risky
            mean_mismatch = np.mean(mismatches) if mismatches else 0.0
            print(f"\n{family}:")
            print(f"  Risky cells: {total_risky}, Override rate: {override_rate:.1f}%")
            print(f"  Pure latent: {pure_latent_count}, Post-hoc override: {overrides}")
            if mismatches:
                print(f"  Mismatch: mean={mean_mismatch:.3f}, max={max(mismatches):.3f}")
            print(f"  Classification: {'PURE_LATENT' if override_rate < 5 else 'CONTRACTED_OVERRIDE' if override_rate < 80 else 'FULL_OVERRIDE'}")
        else:
            print(f"\n{family}: No risky cells found in {N} seeds")


def run_fork_trap_detour_audit():
    """B1.2: Detailed fork_trap detour connectivity audit."""
    print()
    print("=" * 70)
    print("BATCH B INVESTIGATION: fork_trap detour connectivity")
    print("=" * 70)

    from src.envs.scenario_families import generate_fork_trap

    safe1_ok = 0
    safe1_broken = 0
    safe3_ok = 0
    safe3_broken = 0

    for seed in range(100):
        gm, cfg, meta, sc = generate_fork_trap(seed=seed, latent_mode=True)
        H, W = gm.height, gm.width
        passable = np.ones((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if gm.cell_types[r, c] == CellType.WALL:
                    passable[r, c] = False

        start = (2, 1)
        goal = (2, W - 2)
        safe_len = bfs_shortest_safe(passable, gm.cell_types, start, goal)

        # Determine safe row
        rng = np.random.default_rng(seed)
        risky_row = rng.choice([1, 3])
        safe_row = 3 if risky_row == 1 else 1

        if safe_row == 1:
            if safe_len > 0:
                safe1_ok += 1
            else:
                safe1_broken += 1
        else:
            if safe_len > 0:
                safe3_ok += 1
            else:
                safe3_broken += 1

    total = safe1_ok + safe1_broken + safe3_ok + safe3_broken
    print(f"  safe_row==1: {safe1_ok + safe1_broken} total, "
          f"{safe1_ok} OK, {safe1_broken} BROKEN")
    print(f"  safe_row==3: {safe3_ok + safe3_broken} total, "
          f"{safe3_ok} OK, {safe3_broken} BROKEN")
    print(f"  Overall broken rate: {100*(safe1_broken+safe3_broken)/total:.1f}%")


if __name__ == "__main__":
    run_reachability_audit()
    run_latent_mode_audit()
    run_fork_trap_detour_audit()
