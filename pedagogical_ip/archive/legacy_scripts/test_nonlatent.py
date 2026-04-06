"""Quick test: does deadline_gate work in non-latent mode?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()
TEACHERS = {
    "no_tutor":      dict(tutor_mode="none", warning_mode="none"),
    "warning_only":  dict(tutor_mode="none", warning_mode="fixed", lambda_lane_warn=5.0),
    "unlock_only":   dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=False, prefix_horizon=5,
                         allowed_interventions=frozenset({"UNLOCK"})),
    "robot_belief":  dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5),
}

SEEDS = [0, 1, 2]
print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 50)

for diff in ["easy", "medium", "hard"]:
    for t_name, t_kw in TEACHERS.items():
        n_s = n_d = n_t = 0
        steps = []
        for seed in SEEDS:
            kw = dict(seed=seed, scenario_family="deadline_gate",
                      latent_mode=False, difficulty=diff, **t_kw)
            s = runner.reset(**kw)
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
        print(f"{diff:<8} {t_name:<14} {n_s/n:4.0%} {n_d/n:4.0%} "
              f"{n_t/n:4.0%} {np.mean(steps):6.1f}")
    print()
