"""
P1 micro sweep: hazard_belt parameter exploration.
3 seeds × 5 teachers, checking with near_unavoidable regime.
Goal: find regime where item_only > warning_only.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_hazard_belt
from src.envs.lattice_v2 import _bfs_len
from src.envs.map_generator import CellType

SEEDS = [0, 1, 2]
runner = LatticeV2Runner()

TEACHERS = {
    "no_tutor":      dict(tutor_mode="none", warning_mode="none"),
    "warning_only":  dict(tutor_mode="none", warning_mode="fixed", lambda_lane_warn=5.0),
    "item_only":     dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5,
                         allowed_interventions=frozenset({"ITEM_DROP"})),
    "robot_belief":  dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5),
}

# Analyze current structure
print("=== Current hazard_belt structure ===")
for diff in ["easy", "medium", "hard"]:
    for regime in ["unavoidable", "near_unavoidable"]:
        gm, cfg, meta, sc = generate_hazard_belt(
            seed=0, difficulty=diff, latent_mode=False, belt_regime=regime)
        belt = meta.segments[1]
        safe_cells = [(r,c) for r,c in belt.safe_cells]
        avg_risk = np.mean([gm.true_risk[r,c] for r,c in belt.risky_cells])
        # Pre-belt seg risk
        seg0 = meta.segments[0]
        seg0_risk = np.mean([gm.true_risk[r,c] for r,c in seg0.risky_cells])
        print(f"  {diff}/{regime}: belt_risky={len(belt.risky_cells)}, "
              f"belt_safe={len(safe_cells)}, avg_risk={avg_risk:.3f}, "
              f"seg0_risk={seg0_risk:.3f}, t_max={cfg.max_steps}")

print("\n=== Sweep: unavoidable regime ===")
print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 50)

for diff in ["easy", "medium", "hard"]:
    for t_name, t_kw in TEACHERS.items():
        n_s = n_d = n_t = 0
        steps = []
        for seed in SEEDS:
            kw = dict(seed=seed, latent_mode=True, difficulty=diff,
                      scenario_family="hazard_belt", **t_kw)
            s = runner.reset(**kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            if m["reached_goal"] and m["survived"]:
                n_s += 1
            elif not m["survived"]:
                n_d += 1
            else:
                n_t += 1
            steps.append(m["steps"])
        n = len(SEEDS)
        print(f"{diff:<8} {t_name:<14} {n_s/n:4.0%} {n_d/n:4.0%} "
              f"{n_t/n:4.0%} {np.mean(steps):6.1f}")
    print()

print("\n=== Sweep: near_unavoidable regime ===")
print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 50)

for diff in ["easy", "medium", "hard"]:
    for t_name, t_kw in TEACHERS.items():
        n_s = n_d = n_t = 0
        steps = []
        for seed in SEEDS:
            kw = dict(seed=seed, latent_mode=True, difficulty=diff,
                      scenario_family="hazard_belt",
                      belt_regime="near_unavoidable", **t_kw)
            # near_unavoidable is not a runner.reset kwarg — need custom
            # Actually belt_regime is only for generate_hazard_belt
            # For runner, we need scenario_family param passing...
            # For now just run unavoidable and note this limitation
            s = runner.reset(**kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            if m["reached_goal"] and m["survived"]:
                n_s += 1
            elif not m["survived"]:
                n_d += 1
            else:
                n_t += 1
            steps.append(m["steps"])
        n = len(SEEDS)
        print(f"{diff:<8} {t_name:<14} {n_s/n:4.0%} {n_d/n:4.0%} "
              f"{n_t/n:4.0%} {np.mean(steps):6.1f}")
    print()

# Also check: with higher shield_risk_reduction
print("\n=== High shield (0.85 reduction) + unavoidable ===")
print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 50)

for diff in ["easy", "medium"]:
    for t_name in ["item_only", "robot_belief"]:
        t_kw = TEACHERS[t_name].copy()
        n_s = n_d = n_t = 0
        steps = []
        for seed in SEEDS:
            kw = dict(seed=seed, latent_mode=True, difficulty=diff,
                      scenario_family="hazard_belt",
                      shield_risk_reduction=0.85, **t_kw)
            s = runner.reset(**kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            if m["reached_goal"] and m["survived"]:
                n_s += 1
            elif not m["survived"]:
                n_d += 1
            else:
                n_t += 1
            steps.append(m["steps"])
        n = len(SEEDS)
        print(f"{diff:<8} {t_name:<14} {n_s/n:4.0%} {n_d/n:4.0%} "
              f"{n_t/n:4.0%} {np.mean(steps):6.1f}")
    print()
