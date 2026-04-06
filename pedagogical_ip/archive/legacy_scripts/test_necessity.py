"""Phase A3: Does necessity alone fix the bouncing on deadline_gate?

Three conditions:
1. baseline_latent (no necessity, old behavior)
2. necessity_only (necessity active, no UNLOCK updates)
3. lower_lambda_r (lambda_r=1.0 for comparison)
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner

runner = LatticeV2Runner()

SEEDS = [0, 1, 2]
DIFFS = ["easy", "medium", "hard"]

TEACHERS = {
    "no_tutor":      dict(tutor_mode="none", warning_mode="none"),
    "unlock_only":   dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=False, prefix_horizon=5,
                         allowed_interventions=frozenset({"UNLOCK"})),
    "robot_belief":  dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True,
                         item_drop_enabled=True, prefix_horizon=5),
}

print(f"{'Diff':<8} {'Teacher':<14} {'SR':>4} {'DR':>4} {'TR':>4} {'Steps':>6}")
print("-" * 52)

for diff in DIFFS:
    for t_name, t_kw in TEACHERS.items():
        n_s = n_d = n_t = 0
        steps = []
        for seed in SEEDS:
            s = runner.reset(
                seed=seed, scenario_family="deadline_gate",
                latent_mode=True, difficulty=diff, **t_kw)
            
            # Step-by-step trace for first seed
            if seed == 0 and diff == "easy" and t_name == "unlock_only":
                print(f"\n=== Step trace: {diff}/{t_name}/seed=0 ===")
                print(f"  t_max={s.t_max}, goal={s.goal}")
                gate = s.meta.all_door_positions[0]
                for step_i in range(min(50, s.t_max + 2)):
                    if s.done:
                        break
                    print(f"  t={step_i+1:3d} pos={s.agent_pos} "
                          f"gate={'OPEN' if s.passable[gate[0],gate[1]] else 'CLOSED'}")
                    runner.step(s)
                m = runner.get_metrics(s)
                result = ("GOAL" if m["reached_goal"] else 
                         ("DEATH" if not m["survived"] else "TIMEOUT"))
                print(f"  Result: {result} after {m['steps']} steps\n")
            else:
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
