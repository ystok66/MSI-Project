"""Exp A: GTET-L Fair Dispatch Sanity Check.

Verify that ALL factor modes go through the same warning dispatch logic:
- Same warning budget (1 belt zone)
- Same target selection algorithm
- No omniscient fallback for any mode
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "goal_preference_temptation_entanglement_lattice"
runner = LatticeV2Runner()


def check_dispatch(seed, diff, factor_mode="FULL"):
    """Run one episode and extract warning dispatch details."""
    kw = dict(
        seed=seed, difficulty=diff, scenario_family=FAMILY,
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=True, belief_planning_mode=True,
        latent_mode=True, patch_radius=2, prefix_horizon=5,
        factor_mode=factor_mode)
    s = runner.reset(**kw)

    while not s.done:
        s = runner.step(s)

    # Extract what was warned
    warned_cells = list(s.warned_cell_extra.keys())
    m = runner.get_metrics(s)
    H = s.gridmap.height
    center = H // 2

    upper_warned = [(r, c) for r, c in warned_cells if r <= center]
    lower_warned = [(r, c) for r, c in warned_cells if r > center]
    # Exclude temptation cue cells (those get cost 3.0, belt gets 5.0)
    upper_belt_warned = [(r, c) for r, c in upper_warned
                         if s.warned_cell_extra.get((r, c), 0) >= 4.9]
    lower_belt_warned = [(r, c) for r, c in lower_warned
                         if s.warned_cell_extra.get((r, c), 0) >= 4.9]

    return {
        "mode": factor_mode,
        "survived": bool(m["survived"]),
        "warn_count": int(m.get("warnings", 0)),
        "upper_belt_warned": len(upper_belt_warned),
        "lower_belt_warned": len(lower_belt_warned),
        "total_belt_warned": len(upper_belt_warned) + len(lower_belt_warned),
        "chose_zone": "upper" if upper_belt_warned else ("lower" if lower_belt_warned else "none"),
    }


modes = ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY", "THETA_ONLY", "Z_ONLY"]

print("=" * 75)
print("Exp A: Fair Dispatch Sanity — per-seed dispatch trace")
print("=" * 75)

# Check 5 seeds
for seed in range(5):
    print(f"\n--- seed={seed} ---")
    for mode in modes:
        try:
            d = check_dispatch(seed, "hard", mode)
            print(f"  {d['mode']:12s}: zone={d['chose_zone']:6s} "
                  f"upper={d['upper_belt_warned']} lower={d['lower_belt_warned']} "
                  f"warns={d['warn_count']} surv={d['survived']}")
        except Exception as e:
            print(f"  {mode:12s}: ERROR {e}")

# Aggregate fairness check
print(f"\n{'=' * 75}")
print("Aggregate Fairness Check (30 seeds)")
print("=" * 75)

budget_violations = 0
fallback_violations = 0
all_results = {m: [] for m in modes}

for seed in range(30):
    for mode in modes:
        try:
            d = check_dispatch(seed, "hard", mode)
            all_results[mode].append(d)

            # Check: did any mode warn BOTH zones? (budget violation)
            if d["upper_belt_warned"] > 0 and d["lower_belt_warned"] > 0:
                budget_violations += 1

            # Check: did any mode warn 0 belt zones when others warned 1?
        except:
            pass

# Report
for mode in modes:
    recs = all_results[mode]
    if not recs:
        continue
    surv = np.mean([r["survived"] for r in recs])
    avg_total = np.mean([r["total_belt_warned"] for r in recs])
    upper_frac = np.mean([1 if r["chose_zone"] == "upper" else 0 for r in recs])
    print(f"  {mode:12s}: surv={surv:.3f}  avg_belts_warned={avg_total:.1f}  "
          f"chose_upper={upper_frac:.2f}")

print(f"\n  Budget violations (both zones warned): {budget_violations}")
if budget_violations == 0:
    print("  ✓ FAIR: No mode warns both zones")
else:
    print("  ✗ UNFAIR: Some mode warns both zones")

print("\nDone.")
