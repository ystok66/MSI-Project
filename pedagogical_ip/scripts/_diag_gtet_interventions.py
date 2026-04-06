"""Diagnose why interventions never fire on GTET-L."""
import sys
sys.path.insert(0, ".")

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.intervention_policy import score_interventions, InterventionConfig

FAMILY = "goal_preference_temptation_entanglement_lattice"

runner = LatticeV2Runner()
s = runner.reset(
    seed=0, difficulty="medium", scenario_family=FAMILY,
    robot_belief_mode=True, intervention_family_mode=True,
    item_drop_enabled=True, belief_planning_mode=True,
    latent_mode=True, patch_radius=2, prefix_horizon=5,
)

print(f"robot_belief_mode={s.robot_belief_mode}")
print(f"robot_belief is None: {s.robot_belief is None}")
print(f"latent_predictor is None: {s.latent_predictor is None}")
print(f"intervention_family_mode={s.intervention_family_mode}")
print()

# Run a few steps and check
for step_i in range(10):
    s = runner.step(s)
    if s.done:
        break
    if s.last_intervention is not None:
        d = s.last_intervention
        print(f"  t={s.t} intervention={d.action} scores={d.scores} reason={d.reason}")
    else:
        print(f"  t={s.t} no intervention computed")
        # Try manually calling score_interventions
        if s.robot_belief is not None and s.latent_predictor is not None:
            extra = s.warned_cell_extra if s.warned_cell_extra else None
            icfg = InterventionConfig(item_drop_enabled=True)
            try:
                decision = score_interventions(
                    s.robot_belief, s.agent_pos, s.goal,
                    s.belief_cost, s.passable, s.meta,
                    warned_cell_extra=extra,
                    warned_segments=s.warned_segments,
                    prefix_horizon=5,
                    t=s.t, t_max=s.t_max,
                    config=icfg,
                    inventory_state=s.inventory,
                    perceptual_access=s.perceptual_access,
                )
                print(f"    manual decision={decision.action} scores={decision.scores}")
            except Exception as e:
                print(f"    manual score_interventions error: {e}")
        else:
            print(f"    rb_is_none={s.robot_belief is None} lp_is_none={s.latent_predictor is None}")

print(f"\nFinal: survived={s.survived} goal={s.reached_goal} steps={s.steps}")
