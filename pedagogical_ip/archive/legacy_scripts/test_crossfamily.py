"""Cross-family stability: hazard_belt and fork_trap should not regress."""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
SEEDS = list(range(10))

FAMILIES = {
    "hazard_belt": {
        "no_tutor": dict(tutor_mode="none", warning_mode="none"),
        "warning":  dict(tutor_mode="none", warning_mode="lane",
                        intervention_family_mode=True, item_drop_enabled=False),
        "item_only": dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True, item_drop_enabled=True,
                         prefix_horizon=5,
                         allowed_interventions=frozenset({"ITEM_DROP"})),
    },
    "fork_trap": {
        "no_tutor": dict(tutor_mode="none", warning_mode="none"),
        "warning":  dict(tutor_mode="none", warning_mode="lane",
                        intervention_family_mode=True, item_drop_enabled=False),
    },
}

print(f"{'Family':<14} {'Teacher':<12} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 52)

for fam, teachers in FAMILIES.items():
    for t_name, t_kw in teachers.items():
        n_s = n_d = n_t = 0
        steps = []
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
            steps.append(m["steps"])
        n = len(SEEDS)
        print(f"{fam:<14} {t_name:<12} {n_s/n:4.0%} {n_d/n:4.0%} "
              f"{n_t/n:4.0%} {np.mean(steps):6.1f}")
    print()
