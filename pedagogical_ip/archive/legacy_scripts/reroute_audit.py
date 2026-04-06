"""
Reroute audit: track planner path before/after UNLOCK in deadline_gate.
Diagnose why agent doesn't use shortcut after UNLOCK opens the gate.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.map_generator import CellType
from src.agents.planner_astar import plan_next_action_v2

runner = LatticeV2Runner()

for seed in [0, 1, 2]:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}: deadline_gate × unlock_only")
    print(f"{'='*60}")

    s = runner.reset(
        seed=seed, scenario_family="deadline_gate",
        latent_mode=True, difficulty="medium",
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=False, prefix_horizon=5,
        allowed_interventions=frozenset({"UNLOCK"}))

    gate = s.meta.all_door_positions[0]
    print(f"  Shortcut gate: {gate}")
    print(f"  Gate passable at start: {s.passable[gate[0], gate[1]]}")
    print(f"  t_max: {s.t_max}")

    unlock_step = None
    for step_i in range(60):
        if s.done:
            break

        # Check gate status
        gate_open = s.passable[gate[0], gate[1]]

        # Compute planner's preferred path BEFORE step
        extra = s.warned_cell_extra if s.warned_cell_extra else None
        _, next_pos_preview, path_preview = plan_next_action_v2(
            s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
            s.risk_head, budget=30, passable_mask=s.passable,
            warned_cell_extra_cost=extra,
            latent_predictor=s.latent_predictor)

        # Check if shortcut cells are in planned path
        uses_shortcut = any(r == 1 for r, c in path_preview) if path_preview else False
        path_len = len(path_preview) if path_preview else 0

        # Print key steps
        if step_i < 3 or gate_open or s.done or step_i == unlock_step:
            status = "OPEN" if gate_open else "CLOSED"
            shortcut_flag = " [USES SHORTCUT!]" if uses_shortcut else ""
            print(f"  t={step_i+1:3d} pos={s.agent_pos} gate={status} "
                  f"path_len={path_len} next={next_pos_preview}{shortcut_flag}")

            # If gate just opened, print the full path
            if gate_open and unlock_step is None:
                unlock_step = step_i
                print(f"    *** UNLOCK fired at step {step_i}!")
                print(f"    Full planned path: {path_preview[:10]}...")
                # Also compute: what would the shortcut path cost look like?
                # Check shortcut cells' believed cost
                for c in range(gate[1], gate[1]+5):
                    if c < s.gridmap.width:
                        bc = s.belief_cost[1, c]
                        pa = s.passable[1, c]
                        rsk = s.risk_head.predict_risk(s.feature_belief.get_mean(1, c)) if hasattr(s.risk_head, 'predict_risk') else 0
                        print(f"    shortcut (1,{c}): belief_cost={bc:.1f} "
                              f"passable={pa} predicted_risk={rsk:.3f}")

        runner.step(s)

    m = runner.get_metrics(s)
    result = "GOAL" if m["reached_goal"] else ("DEATH" if not m["survived"] else "TIMEOUT")
    print(f"  Result: {result} after {m['steps']} steps "
          f"(unlocks={m['unlock_count']}, warns={m.get('warn_count', 0)})")
