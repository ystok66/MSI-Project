"""GTET-L Step 4 — Focused calibration after dual-belt fix.

Quick check: do dual belts create risk for the tutor-assisted runs?
Compare FULL vs G_ONLY vs no_tutor on updated hard.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "goal_preference_temptation_entanglement_lattice"
runner = LatticeV2Runner()
SEEDS = 30


def run_ep(seed, diff, factor_mode="FULL", no_tutor=False,
           allowed=None, ucfg=None):
    try:
        if no_tutor:
            s = runner.reset(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=False, intervention_family_mode=False,
                item_drop_enabled=False, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                user_cfg=ucfg)
        else:
            kw = dict(
                seed=seed, difficulty=diff, scenario_family=FAMILY,
                robot_belief_mode=True, intervention_family_mode=True,
                item_drop_enabled=True, belief_planning_mode=True,
                latent_mode=True, patch_radius=2, prefix_horizon=5,
                factor_mode=factor_mode, user_cfg=ucfg)
            if allowed:
                kw["allowed_interventions"] = allowed
            s = runner.reset(**kw)

        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        return bool(m["survived"]), bool(m["reached_goal"]), int(m.get("warnings", 0))
    except:
        return False, False, 0


def report(label, results):
    n = len(results)
    surv = np.mean([r[0] for r in results])
    goal = np.mean([r[1] for r in results])
    warn = np.mean([r[2] for r in results])
    print(f"  {label:20s}: surv={surv:.3f} goal={goal:.3f} warn={warn:.1f} n={n}")
    return surv


print("=" * 65)
print(f"GTET-L Dual-Belt Calibration — {SEEDS} seeds")
print("=" * 65)

policies = [
    ("FULL",     dict()),
    ("G_ONLY",   dict(factor_mode="G_ONLY")),
    ("G_THETA",  dict(factor_mode="G_THETA")),
    ("G_Z",      dict(factor_mode="G_Z")),
    ("THETA_Z",  dict(factor_mode="THETA_Z")),
    ("no_warn",  dict(allowed=frozenset({"WAIT", "UNLOCK", "ITEM_DROP"}))),
    ("no_tutor", dict(no_tutor=True)),
]

for diff in ["hard"]:
    print(f"\n--- difficulty: {diff} ---")
    survs = {}
    for label, cfg in policies:
        results = []
        for seed in range(SEEDS):
            r = run_ep(seed, diff, **cfg)
            results.append(r)
        survs[label] = report(label, results)

    # Δ_joint
    full = survs.get("FULL", 0)
    best_abl = max(survs.get("G_THETA", 0), survs.get("G_Z", 0),
                   survs.get("THETA_Z", 0))
    delta = full - best_abl
    print(f"\n  Δ_joint = {delta:+.3f}  (FULL={full:.3f} best_ablated={best_abl:.3f})")
    print(f"  no_tutor={survs.get('no_tutor', 0):.3f}")
    print(f"  no_warn={survs.get('no_warn', 0):.3f}")

print("\nDone.")
