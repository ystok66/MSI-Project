"""
L2C-minimal: Full experiment sweep.

6-condition matrix at working point:
  trap_risk=[0.3,0.5], ratio=1.3, budget=2

  1. no_tutor
  2. door_budget_2
  3. warning_only
  4. door_budget_2 + fixed_warning
  5. door_budget_2 + selected_warning
  6. door_budget_3

Plus: persistent vs reset learning ablation.
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
from src.agents.warning_update import (
    Utterance, apply_warning_to_risk_head, select_best_warning
)
from src.teachers.time_aware_door_tutor import TimeAwareDoorTutor


def run_episode(
    seed,
    difficulty="medium",
    tutor_mode="none",
    time_ratio=1.3,
    closure_budget=None,
    warning_mode="none",      # "none", "fixed", "selected"
    risk_head=None,
):
    """
    Run one lattice_v2 episode.

    tutor_mode: "none", "time_aware", "always_close"
    warning_mode: "none", "fixed", "selected"
    """
    gm, cfg, meta = generate_lattice_v2(seed=seed, difficulty=difficulty)
    H, W = gm.height, gm.width
    rng = np.random.default_rng(seed * 1000 + 1)

    t_max = max(int(time_ratio * meta.shortest_safe), meta.shortest_safe + 1)

    fb = FeatureBeliefMap(H, W, FEATURE_DIM)
    if risk_head is None:
        rh = BayesianRiskHead(d=FEATURE_DIM)
    else:
        rh = risk_head

    belief_cost = np.ones((H, W), dtype=np.float64)
    belief_cost[gm.cell_types == CellType.WALL] = 100.0
    passable = (gm.cell_types != CellType.WALL).copy()

    tutor = None
    if tutor_mode == "time_aware":
        tutor = TimeAwareDoorTutor(gm, meta)

    agent_pos = (2, 1)
    goal = (2, W - 2)

    survived = True
    reached_goal = False
    steps = 0
    traps_hit = 0
    risky_entered = 0
    closures = 0
    warnings_sent = 0
    warning_shifted_lane = 0
    cue_cells_seen = 0

    # Track which segments had warnings applied
    warned_segments = set()

    for t in range(t_max):
        # ── Observe ──
        fobs = observe_features(
            agent_pos, meta.cell_features, gm.cell_types,
            self_noise_var=0.01, neighbor_noise_var=0.08, rng=rng)
        for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
            fb.update(pos[0], pos[1], f_obs, f_var)
            if gm.cell_types[pos[0], pos[1]] == CellType.RISKY:
                cue_cells_seen += 1

        # ── Door tutor ──
        if tutor_mode == "time_aware" and tutor:
            budget_ok = (closure_budget is None) or (closures < closure_budget)
            if budget_ok:
                actions = tutor.step(agent_pos, t_max - t, t)
                for a in actions:
                    if a.action == "close_risky_gate":
                        if closure_budget is not None and closures >= closure_budget:
                            break
                        r_g, c_g = a.gate_cell
                        passable[r_g, c_g] = False
                        belief_cost[r_g, c_g] = 100.0
                        closures += 1

        elif tutor_mode == "always_close":
            if t == 0:
                for seg in meta.segments:
                    if closure_budget is not None and closures >= closure_budget:
                        break
                    r_g, c_g = seg.risky_entry_gate
                    passable[r_g, c_g] = False
                    belief_cost[r_g, c_g] = 100.0
                    closures += 1

        # ── Warning ──
        if warning_mode != "none" and agent_pos[0] == 2:
            # Trigger warning when agent is in corridor near a segment
            for seg in meta.segments:
                if seg.index in warned_segments:
                    continue
                gate = seg.risky_entry_gate
                # Don't warn if gate already closed
                if not passable[gate[0], gate[1]]:
                    continue
                # Trigger when within 2 cols of segment entry
                if abs(agent_pos[1] - seg.col_start) > 2:
                    continue

                # Get upcoming risky cells for this segment
                upcoming_risky = seg.risky_cells

                if warning_mode == "fixed":
                    # Always send RISKY_TEXTURE_AHEAD
                    utt = Utterance.RISKY_TEXTURE_AHEAD
                elif warning_mode == "selected":
                    # Select best utterance
                    utt = select_best_warning(upcoming_risky, fb, rh)
                    if utt is None:
                        continue
                else:
                    continue

                effect = apply_warning_to_risk_head(
                    utt, upcoming_risky, fb, rh, weight=5.0, tau=0.3)

                warned_segments.add(seg.index)
                warnings_sent += 1

                # Check if warning shifted lane choice
                # (compare risk prediction before/after for trap cell)
                if effect.risk_after and effect.risk_before:
                    if max(effect.risk_after) > max(effect.risk_before) + 0.05:
                        warning_shifted_lane += 1

        # ── Plan ──
        _, next_pos = plan_next_action_v2(
            agent_pos, goal, belief_cost, fb.mean, rh,
            budget=30, passable_mask=passable)

        agent_pos = next_pos
        steps += 1

        # ── Outcome ──
        r, c = agent_pos
        if gm.cell_types[r, c] == CellType.RISKY:
            risky_entered += 1
            tr = gm.true_risk[r, c]
            if rng.random() < tr:
                traps_hit += 1
                survived = False
                rh.update_from_label(fb.get_mean(r, c), 1.0, weight=4.0)
                break
            else:
                rh.update_from_label(fb.get_mean(r, c), tr, weight=1.5)
        else:
            rh.update_from_label(fb.get_mean(r, c), 0.0, weight=0.1)

        if agent_pos == goal:
            reached_goal = True
            break

    return {
        "survived": survived, "reached_goal": reached_goal,
        "steps": steps, "t_max": t_max,
        "traps": traps_hit, "risky": risky_entered,
        "closures": closures, "warnings": warnings_sent,
        "warn_shift": warning_shifted_lane,
        "cue_seen": cue_cells_seen,
    }


def run_sweep(label, seeds, **kw):
    results = [run_episode(s, **kw) for s in seeds]
    n = len(results)
    surv = sum(r["survived"] for r in results) / n
    goal = sum(r["reached_goal"] for r in results) / n
    cls = np.mean([r["closures"] for r in results])
    wrn = np.mean([r["warnings"] for r in results])
    rsk = np.mean([r["risky"] for r in results])
    cue = np.mean([r["cue_seen"] for r in results])
    stp = np.mean([r["steps"] for r in results])
    shf = np.mean([r["warn_shift"] for r in results])
    print(f"  {label:36s} surv={surv:5.0%} goal={goal:5.0%} "
          f"cls={cls:.1f} wrn={wrn:.1f} rsk={rsk:.1f} stp={stp:.1f} "
          f"cue={cue:.1f} shf={shf:.1f}")
    return {"label": label, "surv": surv, "goal": goal}


def main():
    N = 100
    seeds = list(range(N))

    print("=" * 95)
    print("L2C-minimal Experiment Sweep")
    print(f"Working point: trap_risk=[0.3,0.5], ratio=1.3, N={N}")
    print("=" * 95)

    # ── 6-condition matrix ──
    print("\n[1] Main 6-condition matrix")
    conds = [
        ("no_tutor",              dict(tutor_mode="none", warning_mode="none")),
        ("door_budget_2",         dict(tutor_mode="time_aware", closure_budget=2, warning_mode="none")),
        ("warning_only",          dict(tutor_mode="none", warning_mode="fixed")),
        ("door_2 + fixed_warn",   dict(tutor_mode="time_aware", closure_budget=2, warning_mode="fixed")),
        ("door_2 + select_warn",  dict(tutor_mode="time_aware", closure_budget=2, warning_mode="selected")),
        ("door_budget_3",         dict(tutor_mode="time_aware", closure_budget=3, warning_mode="none")),
    ]
    for label, kw in conds:
        run_sweep(label, seeds, **kw)

    # ── Persistent vs Reset ──
    print("\n[2] Persistent vs Reset Learning (10 ep x 30 seeds)")
    for cond_label, cond_kw in [
        ("no_tutor",             dict(tutor_mode="none", warning_mode="none")),
        ("door_2",               dict(tutor_mode="time_aware", closure_budget=2, warning_mode="none")),
        ("door_2 + fixed_warn",  dict(tutor_mode="time_aware", closure_budget=2, warning_mode="fixed")),
    ]:
        for learn_mode in ["persistent", "reset"]:
            surv_t, goal_t, n_t = 0, 0, 0
            for s in range(30):
                rh = BayesianRiskHead(d=FEATURE_DIM)
                for ep in range(10):
                    if learn_mode == "reset":
                        rh.reset()
                    r = run_episode(s, risk_head=rh, **cond_kw)
                    surv_t += r["survived"]
                    goal_t += r["reached_goal"]
                    n_t += 1
            print(f"  {cond_label:24s} {learn_mode:12s}: "
                  f"surv={surv_t/n_t:.0%}  goal={goal_t/n_t:.0%} (n={n_t})")

    # ── Always-close comparison ──
    print("\n[3] Additional comparisons")
    run_sweep("always_close_3",    seeds, tutor_mode="always_close", closure_budget=3, warning_mode="none")
    run_sweep("always_close_2",    seeds, tutor_mode="always_close", closure_budget=2, warning_mode="none")
    run_sweep("always_2 + warn",   seeds, tutor_mode="always_close", closure_budget=2, warning_mode="fixed")

    print("\n" + "=" * 95)
    print("Done.")


if __name__ == "__main__":
    main()
