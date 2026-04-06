"""Full episode trace for 3 canonical families.
Logs: observation, tutor decision, chosen action, outcome per step.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.interventions import InterventionType

runner = LatticeV2Runner()

TRACES = [
    ("deadline_gate", dict(
        tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=False,
        prefix_horizon=5, allowed_interventions=frozenset({"UNLOCK"}))),
    ("hazard_belt", dict(
        tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=True,
        prefix_horizon=5)),
    ("fork_trap", dict(
        tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=True,
        prefix_horizon=5)),
]

for fam, kw in TRACES:
    print(f"\n{'='*72}")
    print(f"FAMILY: {fam}  |  seed=0  |  difficulty=medium")
    print(f"{'='*72}")
    s = runner.reset(seed=0, scenario_family=fam,
                     latent_mode=True, difficulty="medium", **kw)
    print(f"Grid: {s.gridmap.height}×{s.gridmap.width}  "
          f"t_max={s.t_max}  goal={s.goal}")
    if hasattr(s, 'meta') and s.meta and s.meta.all_door_positions:
        print(f"Door positions: {s.meta.all_door_positions}")
    if s.inventory:
        print(f"Inventory: shield={'YES' if s.inventory.has_shield() else 'NO'}")
    
    # Show risky cells
    from src.envs.map_generator import CellType
    risky = [(r,c) for r in range(s.gridmap.height)
             for c in range(s.gridmap.width)
             if s.gridmap.cell_types[r,c] == CellType.RISKY]
    print(f"Risky cells ({len(risky)}): {risky[:12]}{'...' if len(risky)>12 else ''}")
    
    print(f"\n{'Step':>4} {'Pos':<8} {'Obs_var':<8} {'Tutor':<12} "
          f"{'Plan_act':<9} {'r_hat':<6} {'n_upd':<5} {'Shield':<6} {'Event'}")
    print("-"*80)
    
    for step_i in range(min(60, s.t_max + 5)):
        if s.done:
            break
        pos_before = s.agent_pos
        
        # 1. Observe
        runner.observe(s)
        obs_var = float(np.mean(s.feature_belief.var[pos_before[0], pos_before[1]]))
        
        # 2. Tutor
        runner.apply_tutor(s)
        tutor_act = "WAIT"
        if s.last_intervention is not None:
            tutor_act = s.last_intervention.action
        
        # 3. Plan + move
        r_hat_here = float(s.latent_predictor.predict_risk(
            s.feature_belief.mean[pos_before[0], pos_before[1]]))
        n_upd = s.latent_predictor.n_updates
        shield = "YES" if (s.inventory and s.inventory.has_shield()) else "NO"
        
        runner.plan_and_move(s)
        pos_after = s.agent_pos
        
        # Determine event
        event = ""
        if s.gridmap.cell_types[pos_after[0], pos_after[1]] == CellType.RISKY:
            event = "RISKY_CELL"
        if not s.survived:
            event = "DEATH"
        elif s.reached_goal:
            event = "GOAL"
        elif s.t >= s.t_max:
            event = "TIMEOUT"
        
        print(f"{step_i+1:4d} {str(pos_before):<8} {obs_var:<8.4f} "
              f"{tutor_act:<12} {str(pos_before)+'→'+str(pos_after):<18} "
              f"{r_hat_here:<6.3f} {n_upd:<5d} {shield:<6} {event}")
    
    m = runner.get_metrics(s)
    result = "GOAL" if m["reached_goal"] else ("DEATH" if not m["survived"] else "TIMEOUT")
    print(f"\n>>> RESULT: {result} after {m['steps']} steps | "
          f"risky_entered={m.get('risky_entered',0)} | "
          f"traps_hit={m.get('traps_hit',0)} | "
          f"unlocks={m.get('unlock_count',0)} | "
          f"warns={m.get('warn_count',0)}")
