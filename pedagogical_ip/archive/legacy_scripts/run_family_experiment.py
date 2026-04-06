"""
Extended scenario family experiment: 4 families × 4 teachers × 20 seeds.
Outputs per-cell survival/death/timeout rates + intervention counts.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.metrics.phase9_metrics import compute_episode_summary, aggregate_summaries

FAMILIES = ["baseline_v2", "fork_trap", "hazard_belt", "deadline_gate"]
TEACHERS = {
    "no_tutor": dict(tutor_mode="none", warning_mode="none"),
    "warning_only": dict(tutor_mode="none", warning_mode="fixed",
                         lambda_lane_warn=5.0),
    "unlock_only": dict(tutor_mode="none", robot_belief_mode=True,
                        intervention_family_mode=True,
                        item_drop_enabled=False, prefix_horizon=5,
                        allowed_interventions=frozenset({"UNLOCK"})),
    "item_only": dict(tutor_mode="none", robot_belief_mode=True,
                      intervention_family_mode=True,
                      item_drop_enabled=True, prefix_horizon=5,
                      allowed_interventions=frozenset({"ITEM_DROP"})),
    "robot_belief": dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5),
}
N_SEEDS = 20

runner = LatticeV2Runner()

print(f"{'Family':<20} {'Teacher':<16} {'SR':>5} {'DR':>5} {'TR':>5} "
      f"{'Warn':>5} {'Unlock':>6} {'Steps':>6}")
print("-" * 80)

for family in FAMILIES:
    for t_name, t_kw in TEACHERS.items():
        summaries = []
        all_warn = []
        all_unlock = []
        all_steps = []
        for seed in range(N_SEEDS):
            kw = dict(
                seed=seed,
                latent_mode=True,
                difficulty="medium",
                **t_kw,
            )
            if family != "baseline_v2":
                kw["scenario_family"] = family
            s = runner.reset(**kw)
            while not s.done:
                runner.step(s)
            m = runner.get_metrics(s)
            summ = compute_episode_summary(
                s, seed=seed,
                agent_level="medium",
                teacher_condition=t_name,
                env_condition="medium",
            )
            summaries.append(summ)
            all_warn.append(m.get("warn_count", 0))
            all_unlock.append(m.get("unlock_count", 0))
            all_steps.append(m.get("steps", 0))

        agg = aggregate_summaries(summaries,
                                  agent_level="medium",
                                  teacher_condition=t_name,
                                  env_condition="medium")
        sr = agg.success_rate
        dr = agg.death_rate
        tr = agg.timeout_rate
        mean_warn = np.mean(all_warn)
        mean_unlock = np.mean(all_unlock)
        mean_steps = np.mean(all_steps)

        print(f"{family:<20} {t_name:<16} {sr:5.0%} {dr:5.0%} {tr:5.0%} "
              f"{mean_warn:5.1f} {mean_unlock:6.1f} {mean_steps:6.1f}")
    print()
