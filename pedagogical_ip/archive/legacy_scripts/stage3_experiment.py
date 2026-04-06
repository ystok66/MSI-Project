"""Stage 3: WARN semantics study on fork_trap + hazard_belt cross-family.

fork_trap conditions:
- no_tutor
- warning (heuristic lane warning)
- robot_belief (full tutor)

hazard_belt conditions:
- no_tutor
- warning (heuristic)
- item_only (shield)
- robot_belief (full)
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
SEEDS = list(range(20))

EXPERIMENTS = {
    "fork_trap": {
        "no_tutor": dict(
            tutor_mode="none", warning_mode="none"),
        "warning": dict(
            tutor_mode="none", warning_mode="lane",
            intervention_family_mode=True, item_drop_enabled=False),
        "robot_belief": dict(
            tutor_mode="none", robot_belief_mode=True,
            intervention_family_mode=True, item_drop_enabled=True,
            prefix_horizon=5),
    },
    "hazard_belt": {
        "no_tutor": dict(
            tutor_mode="none", warning_mode="none"),
        "warning": dict(
            tutor_mode="none", warning_mode="lane",
            intervention_family_mode=True, item_drop_enabled=False),
        "item_only": dict(
            tutor_mode="none", robot_belief_mode=True,
            intervention_family_mode=True, item_drop_enabled=True,
            prefix_horizon=5,
            allowed_interventions=frozenset({"ITEM_DROP"})),
        "robot_belief": dict(
            tutor_mode="none", robot_belief_mode=True,
            intervention_family_mode=True, item_drop_enabled=True,
            prefix_horizon=5),
    },
    "deadline_gate": {
        "no_tutor": dict(
            tutor_mode="none", warning_mode="none"),
        "unlock_only": dict(
            tutor_mode="none", robot_belief_mode=True,
            intervention_family_mode=True, item_drop_enabled=False,
            prefix_horizon=5,
            allowed_interventions=frozenset({"UNLOCK"})),
        "robot_belief": dict(
            tutor_mode="none", robot_belief_mode=True,
            intervention_family_mode=True, item_drop_enabled=True,
            prefix_horizon=5),
    },
}

print(f"{'Family':<14} {'Teacher':<14} {'SR':>5} {'DR':>5} {'TR':>5} "
      f"{'Steps':>6}")
print("-" * 56)

for fam, teachers in EXPERIMENTS.items():
    for t_name, t_kw in teachers.items():
        n_s = n_d = n_t = 0
        steps_list = []
        for seed in SEEDS:
            s = runner.reset(
                seed=seed, scenario_family=fam,
                latent_mode=True, difficulty="medium", **t_kw)
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
        n = len(SEEDS)
        print(f"{fam:<14} {t_name:<14} {n_s/n:5.0%} {n_d/n:5.0%} "
              f"{n_t/n:5.0%} {np.mean(steps_list):6.1f}")
    print()
