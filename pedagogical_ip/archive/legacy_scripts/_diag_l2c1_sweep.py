"""
L2C.1 Experiment Sweep: planner-relevant warnings.

Changes from V1:
- Lane-level warning bias (warned_cell_extra_cost in planner)
- Action-gap utterance selection
- Warning-first loose mode
- Smaller detour delta (~4-6)
- Transfer evaluation (train→freeze→test)

Phase 2: Now uses LatticeV2Runner instead of embedded episode logic.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.risk_model import BayesianRiskHead


runner = LatticeV2Runner()


def run_episode(seed, risk_head=None, **kw):
    """Run a single V2 episode via the runner."""
    state = runner.reset(seed, risk_head=risk_head, **kw)
    while not state.done:
        state = runner.step(state)
    return runner.get_metrics(state)


def run_sweep(label, seeds, **kw):
    results = [run_episode(s, **kw) for s in seeds]
    n = len(results)
    surv = sum(r["survived"] for r in results) / n
    goal = sum(r["reached_goal"] for r in results) / n
    cls = np.mean([r["closures"] for r in results])
    wrn = np.mean([r["warnings"] for r in results])
    rsk = np.mean([r["risky"] for r in results])
    print(f"  {label:36s} surv={surv:5.0%} goal={goal:5.0%} "
          f"cls={cls:.1f} wrn={wrn:.1f} rsk={rsk:.1f}")
    return {"label": label, "surv": surv, "goal": goal, "cls": cls}


def main():
    N = 100
    seeds = list(range(N))

    print("=" * 90)
    print("L2C.1: Planner-Relevant Warning Experiments")
    print(f"Working point: trap_risk=[0.3,0.5], ratio=1.3, dt=1, N={N}")
    print("=" * 90)

    # ── Geometry check ──
    print("\n[0] Geometry check (dt=1)")
    deltas = []
    for s in range(50):
        _, _, m = generate_lattice_v2(seed=s)
        deltas.append(m.shortest_safe - m.shortest_any)
    print(f"  delta: min={min(deltas)} max={max(deltas)} mean={np.mean(deltas):.1f}")

    # ── Main 6-condition matrix ──
    print("\n[1] Main conditions")
    conds = [
        ("no_tutor",                 dict(tutor_mode="none", warning_mode="none")),
        ("door_2",                   dict(tutor_mode="time_aware", closure_budget=2)),
        ("warning_only_fixed",       dict(tutor_mode="none", warning_mode="fixed")),
        ("warning_only_selected",    dict(tutor_mode="none", warning_mode="selected")),
        ("door_2 + warn (old tutor)", dict(tutor_mode="time_aware", closure_budget=2, warning_mode="none")),
        ("warn_first (new tutor)",   dict(tutor_mode="warn_first", closure_budget=2)),
        ("door_3",                   dict(tutor_mode="time_aware", closure_budget=3)),
        ("always_close_3",          dict(tutor_mode="always_close", closure_budget=3)),
    ]
    for label, kw in conds:
        run_sweep(label, seeds, **kw)

    # ── Lambda lane_warn sweep ──
    print("\n[2] Lambda lane_warn sweep (warning_only_fixed)")
    for lw in [1.0, 3.0, 5.0, 7.0, 10.0]:
        run_sweep(f"warn_only lw={lw:.0f}",
                  seeds, tutor_mode="none", warning_mode="fixed", lambda_lane_warn=lw)

    # ── Transfer evaluation ──
    print("\n[3] Transfer evaluation: train 20ep, freeze, test 100 new seeds")
    train_seeds = list(range(20))
    test_seeds = list(range(500, 600))

    for cond_label, cond_kw in [
        ("no_tutor",   dict(tutor_mode="none")),
        ("warn_first", dict(tutor_mode="warn_first", closure_budget=2)),
    ]:
        for learn_mode in ["persistent", "reset"]:
            # Train phase
            rh = BayesianRiskHead(d=FEATURE_DIM)
            for s in train_seeds:
                for ep in range(20):
                    if learn_mode == "reset":
                        rh.reset()
                    run_episode(s, risk_head=rh, **cond_kw)

            # Test phase: freeze risk_head, no tutor
            surv_t, n_t = 0, 0
            for s in test_seeds:
                r = run_episode(s, tutor_mode="none", risk_head=rh)
                surv_t += r["survived"]
                n_t += 1
            print(f"  train={cond_label:12s} learn={learn_mode:12s}: "
                  f"test_surv={surv_t/n_t:.0%} (n={n_t})")

    print("\n" + "=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
