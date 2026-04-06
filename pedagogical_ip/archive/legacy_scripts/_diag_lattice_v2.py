"""
Diagnostic: Lattice V2 integration test.

Tests the complete pipeline:
1. Generate lattice_v2 geometry
2. Run episodes with feature-based agent + Bayesian risk head
3. Compare: no tutor vs. time-aware tutor
4. Verify: agent learns risk from features, tutor improves survival
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2 import generate_lattice_v2, _bfs_len, FEATURE_DIM
from src.envs.map_generator import CellType
from src.agents.feature_belief import FeatureBeliefMap
from src.agents.risk_model import BayesianRiskHead
from src.agents.observation_model import observe_features
from src.agents.planner_astar import plan_next_action_v2
from src.teachers.time_aware_door_tutor import TimeAwareDoorTutor


def run_episode(seed, difficulty="medium", use_tutor=False, verbose=False):
    """Run one lattice_v2 episode. Return dict of metrics."""
    gm, cfg, meta = generate_lattice_v2(seed=seed, difficulty=difficulty)
    H, W = gm.height, gm.width
    rng = np.random.default_rng(seed * 1000 + 1)

    # Agent components
    fb = FeatureBeliefMap(H, W, FEATURE_DIM)
    risk_head = BayesianRiskHead(d=FEATURE_DIM)
    belief_cost = np.ones((H, W), dtype=np.float64)
    # Mark walls as impassable in belief
    for r in range(H):
        for c in range(W):
            if gm.cell_types[r, c] == CellType.WALL:
                belief_cost[r, c] = 100.0

    # Tutor
    tutor = None
    if use_tutor:
        tutor = TimeAwareDoorTutor(gm, meta)

    # Passable mask
    passable = np.array([[gm.cell_types[r, c] != CellType.WALL
                          for c in range(W)] for r in range(H)])

    # Episode variables
    agent_pos = (2, 1)  # start
    goal = (2, W - 2)
    t_max = cfg.max_steps
    
    survived = True
    reached_goal = False
    steps = 0
    traps_hit = 0
    risky_cells_entered = 0
    tutor_closures = 0

    for t in range(t_max):
        # ── Observe features ──
        fobs = observe_features(
            agent_pos, meta.cell_features, gm.cell_types,
            self_noise_var=0.01, neighbor_noise_var=0.08,
            rng=rng)

        # Update feature belief
        for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
            fb.update(pos[0], pos[1], f_obs, f_var)

        # ── Tutor step ──
        if tutor:
            t_left = t_max - t
            actions = tutor.step(agent_pos, t_left, t)
            for a in actions:
                if a.action == "close_risky_gate":
                    # Close gate: mark as impassable
                    r, c = a.gate_cell
                    passable[r, c] = False
                    belief_cost[r, c] = 100.0
                    tutor_closures += 1
                    if verbose:
                        print(f"  t={t}: Tutor closes {a.gate_cell} (mode={a.mode}, slack={a.slack:.2f})")

        # ── Plan ──
        action_name, next_pos = plan_next_action_v2(
            agent_pos, goal, belief_cost, fb.mean, risk_head,
            budget=30, lambda_risk=3.0, lambda_uncertainty=0.5,
            passable_mask=passable)

        if verbose:
            print(f"  t={t}: pos={agent_pos} -> {action_name} -> {next_pos}")

        # ── Move ──
        old_pos = agent_pos
        agent_pos = next_pos
        steps += 1

        # Check outcome
        r, c = agent_pos
        if gm.cell_types[r, c] == CellType.RISKY:
            risky_cells_entered += 1
            true_risk = gm.true_risk[r, c]

            if true_risk >= 0.99:
                # Fatal trap!
                traps_hit += 1
                survived = False
                # Update risk model from death
                obs_f = fb.get_mean(r, c)
                risk_head.update_from_label(obs_f, 1.0, weight=5.0)
                if verbose:
                    print(f"  *** TRAP HIT at {agent_pos}! ***")
                break
            else:
                # Mild risk — survive but update model
                obs_f = fb.get_mean(r, c)
                risk_head.update_from_label(obs_f, true_risk)
        else:
            # Safe cell — update model
            obs_f = fb.get_mean(r, c)
            risk_head.update_from_label(obs_f, 0.0, weight=0.1)

        if agent_pos == goal:
            reached_goal = True
            break

    return {
        "seed": seed,
        "survived": survived,
        "reached_goal": reached_goal,
        "steps": steps,
        "t_max": t_max,
        "traps_hit": traps_hit,
        "risky_cells": risky_cells_entered,
        "tutor_closures": tutor_closures,
        "shortest_any": meta.shortest_any,
        "shortest_safe": meta.shortest_safe,
    }


def main():
    N = 50
    
    print("=" * 60)
    print("Lattice V2 Integration Test")
    print("=" * 60)
    
    # Test 1: Geometry check
    print("\n[1] Geometry Check")
    for diff in ["easy", "medium", "hard"]:
        deltas = []
        for s in range(N):
            _, _, m = generate_lattice_v2(seed=s, difficulty=diff)
            deltas.append(m.shortest_safe - m.shortest_any)
        print(f"  {diff}: delta min={min(deltas)} max={max(deltas)} mean={np.mean(deltas):.1f}")

    # Test 2: No tutor baseline
    print("\n[2] No Tutor Baseline (medium, 50 seeds)")
    results_no = [run_episode(s, difficulty="medium", use_tutor=False) for s in range(N)]
    surv_no = sum(r["survived"] for r in results_no) / N
    goal_no = sum(r["reached_goal"] for r in results_no) / N
    print(f"  Survival:  {surv_no:.0%}")
    print(f"  Goal:      {goal_no:.0%}")
    print(f"  Avg steps: {np.mean([r['steps'] for r in results_no]):.1f}")
    print(f"  Avg risky: {np.mean([r['risky_cells'] for r in results_no]):.1f}")

    # Test 3: With tutor
    print("\n[3] Time-Aware Tutor (medium, 50 seeds)")
    results_tu = [run_episode(s, difficulty="medium", use_tutor=True) for s in range(N)]
    surv_tu = sum(r["survived"] for r in results_tu) / N
    goal_tu = sum(r["reached_goal"] for r in results_tu) / N
    closures = np.mean([r["tutor_closures"] for r in results_tu])
    print(f"  Survival:   {surv_tu:.0%}")
    print(f"  Goal:       {goal_tu:.0%}")
    print(f"  Avg steps:  {np.mean([r['steps'] for r in results_tu]):.1f}")
    print(f"  Avg risky:  {np.mean([r['risky_cells'] for r in results_tu]):.1f}")
    print(f"  Avg closes: {closures:.1f}")

    # Test 4: Cross-episode learning
    print("\n[4] Cross-Episode Learning (10 episodes on same seed)")
    risk_head = BayesianRiskHead(d=FEATURE_DIM)
    for ep in range(10):
        s = 42
        gm, cfg, meta = generate_lattice_v2(seed=s, difficulty="medium")
        H, W = gm.height, gm.width
        rng = np.random.default_rng(ep * 100)
        fb = FeatureBeliefMap(H, W, FEATURE_DIM)
        belief_cost = np.ones((H, W))
        for r in range(H):
            for c in range(W):
                if gm.cell_types[r, c] == CellType.WALL:
                    belief_cost[r, c] = 100.0
        passable = np.array([[gm.cell_types[r,c] != CellType.WALL
                              for c in range(W)] for r in range(H)])
        
        agent_pos = (2, 1)
        goal = (2, W - 2)
        hit_trap = False
        for t in range(cfg.max_steps):
            fobs = observe_features(agent_pos, meta.cell_features, gm.cell_types, rng=rng)
            for pos, f_obs, f_var in zip(fobs.positions, fobs.feature_obs, fobs.feature_var):
                fb.update(pos[0], pos[1], f_obs, f_var)
            _, next_pos = plan_next_action_v2(
                agent_pos, goal, belief_cost, fb.mean, risk_head,
                budget=30, passable_mask=passable)
            agent_pos = next_pos
            r, c = agent_pos
            if gm.cell_types[r,c] == CellType.RISKY:
                if gm.true_risk[r,c] >= 0.99:
                    risk_head.update_from_label(fb.get_mean(r,c), 1.0, weight=5.0)
                    hit_trap = True
                    break
                else:
                    risk_head.update_from_label(fb.get_mean(r,c), gm.true_risk[r,c])
            else:
                risk_head.update_from_label(fb.get_mean(r,c), 0.0, weight=0.1)
            if agent_pos == goal:
                break

        # Test risk predictions on trap features
        trap_preds = []
        safe_preds = []
        for seg in meta.segments:
            if seg.trap_cell:
                x = meta.cell_features[seg.trap_cell]
                trap_preds.append(risk_head.predict_risk(x))
            for sc in seg.safe_cells[:2]:
                x = meta.cell_features[sc]
                safe_preds.append(risk_head.predict_risk(x))

        trap_avg = np.mean(trap_preds) if trap_preds else 0
        safe_avg = np.mean(safe_preds) if safe_preds else 0
        status = "DIED" if hit_trap else "OK"
        print(f"  Ep {ep}: {status}  risk_head: trap={trap_avg:.3f} safe={safe_avg:.3f} w={risk_head.w}")

    # Test 5: Feature leak check
    print("\n[5] Feature Oracle Leak Test")
    gm, cfg, meta = generate_lattice_v2(seed=0, difficulty="medium")
    fb_fresh = FeatureBeliefMap(gm.height, gm.width, FEATURE_DIM)
    risk_head_fresh = BayesianRiskHead(d=FEATURE_DIM)
    # Before any observations, risk prediction should be ~0.5 (uninformative)
    x_prior = fb_fresh.get_mean(3, 3)
    p_prior = risk_head_fresh.predict_risk(x_prior)
    print(f"  Prior belief feature: {x_prior}")
    print(f"  Prior risk prediction: {p_prior:.3f} (should be ~0.5)")
    assert abs(p_prior - 0.5) < 0.1, f"LEAK: prior risk {p_prior:.3f} != 0.5"
    print("  [OK] No feature oracle leak")

    print("\n" + "=" * 60)
    print("Integration test complete!")


if __name__ == "__main__":
    main()
