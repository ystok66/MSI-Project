"""
Phase 1 Diagnostic: Deceptive Fork — no teacher.

Measures whether the agent takes the bait path A or safe path B.
This is the fundamental pre-condition for all later experiments.

Metrics:
  - branch_A_rate: fraction choosing bait path
  - branch_B_rate: fraction choosing safe path (or stuck)
  - trap_entry_rate: fraction that actually hits the trap cell
  - CSR: baseline completion rate (should be low without teacher)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.envs.map_families import generate_deceptive_fork
from src.envs.pedagogical_grid import PedagogicalGridEnv

# Fork geometry constants
FORK_CELL = (2, 2)
PATH_A_CELLS = {(1, 2), (1, 3), (1, 4), (1, 5)}  # includes trap
PATH_B_CELLS = {(3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5)}
TRAP_CELL = (1, 4)
GOAL_CELL = (2, 5)

N_SEEDS = 100


def run_episode(seed: int, difficulty: str = "medium"):
    """Run one episode with no teacher (WAIT every step)."""
    gm, cfg = generate_deceptive_fork(seed=seed, difficulty=difficulty)

    env = PedagogicalGridEnv(
        grid_map=gm,
        max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0,
        prior_cost_var=0.1,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget,
        lambda_risk=0.8,
        lambda_uncertainty=0.02,
        seed=seed,
    )
    env.reset()

    branch = "none"
    hit_trap = False
    success = False
    step_count = 0
    positions = [env.agent.pos]

    for t in range(cfg.max_steps):
        # Teacher does nothing (WAIT = 0)
        obs, reward, terminated, truncated, info = env.step(0)
        step_count = t + 1
        pos = tuple(env.agent.pos)
        positions.append(pos)

        # Detect which branch agent entered
        if branch == "none":
            if pos in PATH_A_CELLS:
                branch = "A"
            elif pos in PATH_B_CELLS:
                branch = "B"

        if pos == TRAP_CELL:
            hit_trap = True

        if terminated:
            if env.object_delivered or pos == GOAL_CELL:
                success = True
            break
        if truncated:
            break

    return {
        "seed": seed,
        "branch": branch,
        "hit_trap": hit_trap,
        "success": success,
        "steps": step_count,
        "max_steps": cfg.max_steps,
        "final_pos": positions[-1],
    }


if __name__ == "__main__":
    print(f"Running {N_SEEDS} episodes on Deceptive Fork (no teacher)")
    print(f"Grid: 6x8, L_A=6, L_B=8, trap rho*=1.0, door on B")
    print()

    results = [run_episode(s) for s in range(N_SEEDS)]

    n_A = sum(1 for r in results if r["branch"] == "A")
    n_B = sum(1 for r in results if r["branch"] == "B")
    n_none = sum(1 for r in results if r["branch"] == "none")
    n_trap = sum(1 for r in results if r["hit_trap"])
    n_succ = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"  Branch A (bait):  {n_A:3d} / {N_SEEDS}  = {100*n_A/N_SEEDS:.0f}%")
    print(f"  Branch B (safe):  {n_B:3d} / {N_SEEDS}  = {100*n_B/N_SEEDS:.0f}%")
    print(f"  Neither:          {n_none:3d} / {N_SEEDS}  = {100*n_none/N_SEEDS:.0f}%")
    print(f"  Hit trap:         {n_trap:3d} / {N_SEEDS}  = {100*n_trap/N_SEEDS:.0f}%")
    print(f"  Success (CSR):    {n_succ:3d} / {N_SEEDS}  = {100*n_succ/N_SEEDS:.0f}%")
    print(f"  Avg steps:        {avg_steps:.1f}")
    print()

    # Show a few example trajectories
    print("Example trajectories (first 5):")
    for r in results[:5]:
        print(f"  seed={r['seed']}: branch={r['branch']}, "
              f"trap={r['hit_trap']}, success={r['success']}, "
              f"steps={r['steps']}, final={r['final_pos']}")
