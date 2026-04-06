"""
L2B.5: Non-saturation experiments for Lattice V2.

Sweep matrix:
  A1: Time ratio sweep (1.1, 1.2, 1.3, 1.4, 1.6)
  B1: Closure budget sweep (max 1, 2, 3 gates)
  B2: Always-close vs time-aware vs no-tutor
  B3: Persistent vs reset risk_head
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2 import generate_lattice_v2, FEATURE_DIM
from src.envs.map_generator import CellType
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.risk_model import BayesianRiskHead
from src.agents.observation_model import observe_features
from src.agents.planner_astar import plan_next_action_v2
from src.teachers.time_aware_door_tutor import TimeAwareDoorTutor


def run_episode(
    seed,
    difficulty="medium",
    tutor_mode="none",       # "none", "time_aware", "always_close"
    time_ratio=None,         # if set, override T_max = ratio * shortest_safe
    closure_budget=None,     # max gates tutor can close (None = unlimited)
    risk_head=None,          # pass shared risk_head for persistent learning
):
    """Run one episode. Return metrics dict."""
    gm, cfg, meta = generate_lattice_v2(seed=seed, difficulty=difficulty)
    H, W = gm.height, gm.width
    rng = np.random.default_rng(seed * 1000 + 1)

    # Override T_max if time_ratio specified
    if time_ratio is not None:
        t_max = max(int(time_ratio * meta.shortest_safe), meta.shortest_safe + 1)
    else:
        t_max = cfg.max_steps

    # Agent
    fb = FeatureBeliefMap(H, W, FEATURE_DIM)
    if risk_head is None:
        risk_head_local = BayesianRiskHead(d=FEATURE_DIM)
    else:
        risk_head_local = risk_head  # shared across episodes

    belief_cost = np.ones((H, W), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            if gm.cell_types[r, c] == CellType.WALL:
                belief_cost[r, c] = 100.0

    passable = np.array([[gm.cell_types[r, c] != CellType.WALL
                          for c in range(W)] for r in range(H)])

    # Tutor
    tutor = None
    if tutor_mode == "time_aware":
        tutor = TimeAwareDoorTutor(gm, meta)
    # "always_close" handled manually below

    agent_pos = (2, 1)
    goal = (2, W - 2)

    survived = True
    reached_goal = False
    steps = 0
    traps_hit = 0
    risky_entered = 0
    closures = 0
    cue_cells_seen = 0  # risky cells where agent got observation

    for t in range(t_max):
        # Observe
        fobs = observe_features(
            agent_pos, meta.cell_features, gm.cell_types,
            self_noise_var=0.01, neighbor_noise_var=0.08, rng=rng)
        for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
            fb.update(pos[0], pos[1], f_obs, f_var)
            # Count cue cell observations
            if gm.cell_types[pos[0], pos[1]] == CellType.RISKY:
                cue_cells_seen += 1

        # Tutor
        if tutor_mode == "time_aware" and tutor:
            budget_ok = (closure_budget is None) or (closures < closure_budget)
            if budget_ok:
                t_left = t_max - t
                actions = tutor.step(agent_pos, t_left, t)
                for a in actions:
                    if a.action == "close_risky_gate":
                        if closure_budget is not None and closures >= closure_budget:
                            break
                        r_g, c_g = a.gate_cell
                        passable[r_g, c_g] = False
                        belief_cost[r_g, c_g] = 100.0
                        closures += 1

        elif tutor_mode == "always_close":
            # Close ALL risky gates immediately at t=0
            if t == 0:
                for seg in meta.segments:
                    gate = seg.risky_entry_gate
                    if closure_budget is not None and closures >= closure_budget:
                        break
                    r_g, c_g = gate
                    passable[r_g, c_g] = False
                    belief_cost[r_g, c_g] = 100.0
                    closures += 1

        # Plan
        _, next_pos = plan_next_action_v2(
            agent_pos, goal, belief_cost, fb.mean, risk_head_local,
            budget=30,
            passable_mask=passable)

        agent_pos = next_pos
        steps += 1

        # Outcome — probabilistic death
        r, c = agent_pos
        if gm.cell_types[r, c] == CellType.RISKY:
            risky_entered += 1
            tr = gm.true_risk[r, c]
            # Sample death from risk probability
            if rng.random() < tr:
                traps_hit += 1
                survived = False
                risk_head_local.update_from_label(fb.get_mean(r, c), 1.0, weight=4.0)
                break
            else:
                # Survived this risky cell — learn from near miss
                risk_head_local.update_from_label(fb.get_mean(r, c), tr, weight=1.5)
        else:
            risk_head_local.update_from_label(fb.get_mean(r, c), 0.0, weight=0.1)

        if agent_pos == goal:
            reached_goal = True
            break

    return {
        "survived": survived,
        "reached_goal": reached_goal,
        "steps": steps,
        "t_max": t_max,
        "traps": traps_hit,
        "risky": risky_entered,
        "closures": closures,
        "cue_seen": cue_cells_seen,
    }


def agg(results, key):
    return sum(r[key] for r in results) / len(results)


def run_sweep(label, seeds, **kwargs):
    results = [run_episode(s, **kwargs) for s in seeds]
    surv = agg(results, "survived")
    goal = agg(results, "reached_goal")
    cls = np.mean([r["closures"] for r in results])
    risky = np.mean([r["risky"] for r in results])
    steps = np.mean([r["steps"] for r in results])
    cue = np.mean([r["cue_seen"] for r in results])
    print(f"  {label:30s}  surv={surv:.0%}  goal={goal:.0%}  "
          f"cls={cls:.1f}  risky={risky:.1f}  steps={steps:.1f}  cue={cue:.1f}")
    return {"label": label, "surv": surv, "goal": goal, "closures": cls,
            "risky": risky, "steps": steps, "cue": cue}


def main():
    N = 50
    seeds = list(range(N))

    print("=" * 90)
    print("L2B.5: Non-Saturation Experiments")
    print("=" * 90)

    # ── A1: Time Ratio Sweep ──
    print("\n[A1] Time Ratio Sweep (no tutor vs time-aware tutor)")
    print(f"{'':34s}  {'surv':>5s}  {'goal':>5s}  {'cls':>4s}  {'risk':>5s}  {'step':>5s}  {'cue':>5s}")
    for ratio in [1.05, 1.1, 1.2, 1.3, 1.4, 1.6]:
        run_sweep(f"ratio={ratio:.2f} no_tutor",
                  seeds, tutor_mode="none", time_ratio=ratio)
        run_sweep(f"ratio={ratio:.2f} time_aware",
                  seeds, tutor_mode="time_aware", time_ratio=ratio)

    # ── B1: Closure Budget Sweep ──
    print("\n[B1] Closure Budget Sweep (time_aware, ratio=1.3)")
    for budget in [0, 1, 2, 3]:
        run_sweep(f"budget={budget}",
                  seeds, tutor_mode="time_aware", time_ratio=1.3, closure_budget=budget)

    # ── B2: Tutor Mode Comparison ──
    print("\n[B2] Tutor Mode Comparison (ratio=1.3)")
    for mode in ["none", "time_aware", "always_close"]:
        run_sweep(f"mode={mode}",
                  seeds, tutor_mode=mode, time_ratio=1.3)

    # ── B3: Persistent vs Reset Learning ──
    print("\n[B3] Persistent vs Reset Learning (no tutor, ratio=1.3, 10 episodes x 20 seeds)")
    for learn_mode in ["persistent", "reset"]:
        survived_total = 0
        goal_total = 0
        n_total = 0
        for s in range(20):
            risk_head = BayesianRiskHead(d=FEATURE_DIM)
            for ep in range(10):
                if learn_mode == "reset":
                    risk_head.reset()
                r = run_episode(s, tutor_mode="none", time_ratio=1.3,
                                risk_head=risk_head)
                survived_total += r["survived"]
                goal_total += r["reached_goal"]
                n_total += 1
        print(f"  {learn_mode:12s}: surv={survived_total/n_total:.0%}  "
              f"goal={goal_total/n_total:.0%} (n={n_total})")

    # Same with tutor
    print("\n[B3b] Persistent vs Reset Learning (time_aware tutor, ratio=1.3)")
    for learn_mode in ["persistent", "reset"]:
        survived_total = 0
        goal_total = 0
        n_total = 0
        for s in range(20):
            risk_head = BayesianRiskHead(d=FEATURE_DIM)
            for ep in range(10):
                if learn_mode == "reset":
                    risk_head.reset()
                r = run_episode(s, tutor_mode="time_aware", time_ratio=1.3,
                                risk_head=risk_head)
                survived_total += r["survived"]
                goal_total += r["reached_goal"]
                n_total += 1
        print(f"  {learn_mode:12s}: surv={survived_total/n_total:.0%}  "
              f"goal={goal_total/n_total:.0%} (n={n_total})")

    print("\n" + "=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
