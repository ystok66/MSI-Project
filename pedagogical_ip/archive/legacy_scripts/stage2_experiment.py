"""Stage 2: UNLOCK mechanism comparison experiment.

Conditions:
- no_tutor: baseline, no teaching
- unlock_only: UNLOCK intervention, no enhancement (tests legacy + necessity)

Note: necessity is always active now (baked into planner).
The comparison is: does UNLOCK via robot_belief scoring + apply_unlock_update
improve over simple no_tutor (safe path only)?
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
SEEDS = list(range(20))
DIFFS = ["easy", "medium", "hard"]

TEACHERS = {
    "no_tutor": dict(
        tutor_mode="none", warning_mode="none",
    ),
    "unlock_only": dict(
        tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=False,
        prefix_horizon=5,
        allowed_interventions=frozenset({"UNLOCK"}),
    ),
    "full_robot": dict(
        tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=True,
        prefix_horizon=5,
    ),
}

print(f"{'Diff':<8} {'Teacher':<14} {'SR':>5} {'DR':>5} {'TR':>5} "
      f"{'Steps':>6} {'Unlk':>4}")
print("-" * 56)

for diff in DIFFS:
    for t_name, t_kw in TEACHERS.items():
        n_s = n_d = n_t = 0
        steps_list = []
        unlock_list = []
        for seed in SEEDS:
            s = runner.reset(
                seed=seed, scenario_family="deadline_gate",
                latent_mode=True, difficulty=diff, **t_kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            if m["reached_goal"] and m["survived"]:
                n_s += 1
            elif not m["survived"]:
                n_d += 1
            else:
                n_t += 1
            steps_list.append(m["steps"])
            unlock_list.append(m.get("unlock_count", 0))
        n = len(SEEDS)
        print(f"{diff:<8} {t_name:<14} {n_s/n:5.0%} {n_d/n:5.0%} "
              f"{n_t/n:5.0%} {np.mean(steps_list):6.1f} "
              f"{np.mean(unlock_list):4.1f}")
    print()
