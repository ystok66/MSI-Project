"""
P1 micro sweep: deadline_gate parameter exploration.
3 seeds × shortcut risk × t_max slack.
Goal: find regime where unlock_only > warning_only.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_deadline_gate
from src.envs.lattice_v2 import _bfs_len
from src.envs.map_generator import CellType

SEEDS = [0, 1, 2]

TEACHERS = {
    "no_tutor":      dict(tutor_mode="none", warning_mode="none"),
    "warning_only":  dict(tutor_mode="none", warning_mode="fixed", lambda_lane_warn=5.0),
    "unlock_only":   dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=False, prefix_horizon=5,
                         allowed_interventions=frozenset({"UNLOCK"})),
    "robot_belief":  dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5),
}

runner = LatticeV2Runner()

# First: analyze current params structure
print("=== Current deadline_gate structure ===")
for diff in ["easy", "medium", "hard"]:
    gm, cfg, meta, sc = generate_deadline_gate(seed=0, difficulty=diff, latent_mode=False)
    gate = meta.all_door_positions[0]
    gates = {gate}
    safe_d = _bfs_len(gm, gm.agent_start, gm.target_pos, gates)
    # Open gate for shortcut
    old_ct = gm.cell_types[gate[0], gate[1]]
    old_cost = gm.true_cost[gate[0], gate[1]]
    gm.cell_types[gate[0], gate[1]] = CellType.NORMAL
    gm.true_cost[gate[0], gate[1]] = 1.0
    short_d = _bfs_len(gm, gm.agent_start, gm.target_pos, set())
    gm.cell_types[gate[0], gate[1]] = old_ct
    gm.true_cost[gate[0], gate[1]] = old_cost
    # Shortcut risky cells
    shortcut_cells = [(r,c) for r,c in meta.segments[0].risky_cells]
    avg_risk = np.mean([gm.true_risk[r,c] for r,c in shortcut_cells]) if shortcut_cells else 0
    print(f"  {diff}: safe={safe_d}, short={short_d}, t_max={cfg.max_steps}, "
          f"slack={cfg.max_steps-safe_d}, advantage={safe_d-short_d}, "
          f"avg_shortcut_risk={avg_risk:.3f}, n_risky={len(shortcut_cells)}")

print("\n=== Sweep: 3 seeds × 4 teachers ===")
print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Unlk':>5} {'Steps':>6}")
print("-" * 55)

for diff in ["easy", "medium", "hard"]:
    for t_name, t_kw in TEACHERS.items():
        n_success = 0
        n_death = 0
        n_timeout = 0
        unlocks = []
        steps = []
        for seed in SEEDS:
            kw = dict(seed=seed, latent_mode=True, difficulty=diff,
                      scenario_family="deadline_gate", **t_kw)
            s = runner.reset(**kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            if m["reached_goal"] and m["survived"]:
                n_success += 1
            elif not m["survived"]:
                n_death += 1
            else:
                n_timeout += 1
            unlocks.append(m.get("unlock_count", 0))
            steps.append(m["steps"])
        n = len(SEEDS)
        print(f"{diff:<8} {t_name:<14} {n_success/n:4.0%} {n_death/n:4.0%} "
              f"{n_timeout/n:4.0%} {np.mean(unlocks):5.1f} {np.mean(steps):6.1f}")
    print()
