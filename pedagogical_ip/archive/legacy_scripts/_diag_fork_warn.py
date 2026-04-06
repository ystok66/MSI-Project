"""
Phase 2A Diagnostic: Deterministic Oracle WARN on Env-A (no door).

Three baselines, 100 seeds each:
  1. no_teacher:           WAIT every step
  2. oracle_warn_trap_only: At fork, raise risk belief ONLY on trap cell
  3. oracle_warn_segment:   At fork, raise risk belief on entire A-path suffix
                            with graduated profile

Key metrics:
  - Branch choice at fork (A vs B)
  - Branch switch after warning
  - Replan triggered
  - CSR
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.envs.map_families import generate_deceptive_fork
from src.envs.pedagogical_grid import PedagogicalGridEnv

# ── Fork geometry constants ──
FORK_CELL = (2, 2)
TRAP_CELL = (1, 4)
GOAL_CELL = (2, 5)
PATH_A_CELLS = {(1, 2), (1, 3), (1, 4), (1, 5)}
PATH_B_CELLS = {(3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5)}

# Path A cells from fork onward (ordered by distance from fork)
PATH_A_SEGMENT = [(1, 2), (1, 3), (1, 4), (1, 5)]  # (1,4) is trap

# Graduated risk profile for segment warning (increasing toward trap)
SEGMENT_RISK_PROFILE = [0.20, 0.40, 0.85, 0.15]  # (1,2)=0.20, (1,3)=0.40, trap=0.85, (1,5)=0.15

N_SEEDS = 100


def apply_oracle_warning(agent, target_cells, risk_profile, obs_var=0.005):
    """Apply deterministic warning: strong pseudo-observation on target cells."""
    for (r, c), y in zip(target_cells, risk_profile):
        # Precision-weighted Kalman update (same API as normal observation)
        old_prec = 1.0 / max(agent.belief.risk_var[r, c], 1e-10)
        obs_prec = 1.0 / obs_var
        new_prec = old_prec + obs_prec
        new_var = 1.0 / new_prec
        new_mean = new_var * (
            agent.belief.risk_mean[r, c] * old_prec + y * obs_prec
        )
        agent.belief.risk_mean[r, c] = np.clip(new_mean, 0.0, 1.0)
        agent.belief.risk_var[r, c] = new_var
    # Invalidate plan so agent replans with new beliefs
    agent.plan_invalidated = True


def get_branch_from_plan(agent):
    """Determine which branch the agent's current plan heads toward."""
    if not agent.current_plan or len(agent.current_plan) < 2:
        return "none"
    for pos in agent.current_plan[1:]:  # skip current pos
        if pos in PATH_A_CELLS:
            return "A"
        if pos in PATH_B_CELLS:
            return "B"
    return "none"


def run_episode(seed, warn_mode="none"):
    """
    Run one episode on Env-A (no door).

    warn_mode:
      "none":       no teacher (WAIT only)
      "trap_only":  at fork, warn only trap cell (1,4)
      "segment":    at fork, warn entire A-path suffix with gradient
    """
    gm, cfg = generate_deceptive_fork(seed=seed, difficulty="medium", with_door=False)

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

    branch_chosen = "none"
    hit_trap = False
    success = False
    warned = False
    branch_before_warn = "none"
    branch_after_warn = "none"
    replan_after_warn = False

    for t in range(cfg.max_steps):
        pos = tuple(env.agent.pos)

        # ── Warning logic: intervene at fork ──
        if pos == FORK_CELL and not warned and warn_mode != "none":
            # Record branch BEFORE warning
            branch_before_warn = get_branch_from_plan(env.agent)

            if warn_mode == "trap_only":
                apply_oracle_warning(
                    env.agent,
                    [(1, 4)],          # only trap cell
                    [0.85],            # single high value
                )
            elif warn_mode == "segment":
                apply_oracle_warning(
                    env.agent,
                    PATH_A_SEGMENT,
                    SEGMENT_RISK_PROFILE,
                )

            warned = True

            # Force replan by stepping with WAIT (plan already invalidated by apply_oracle_warning)
            # The replan happens inside plan_and_act on next step

        # Teacher action = WAIT (0) every step
        obs, reward, terminated, truncated, info = env.step(0)
        pos_after = tuple(env.agent.pos)

        # Record branch AFTER warning (first step after warn)
        if warned and branch_after_warn == "none":
            branch_after_warn = get_branch_from_plan(env.agent)
            if branch_after_warn != branch_before_warn:
                replan_after_warn = True

        # Track branch choice
        if branch_chosen == "none":
            if pos_after in PATH_A_CELLS:
                branch_chosen = "A"
            elif pos_after in PATH_B_CELLS:
                branch_chosen = "B"

        if pos_after == TRAP_CELL:
            hit_trap = True
        if pos_after == GOAL_CELL:
            success = True

        if terminated or truncated:
            if env.object_delivered:
                success = True
            break

    return {
        "seed": seed,
        "warn_mode": warn_mode,
        "branch": branch_chosen,
        "hit_trap": hit_trap,
        "success": success,
        "warned": warned,
        "branch_before_warn": branch_before_warn,
        "branch_after_warn": branch_after_warn,
        "replan_switched": replan_after_warn,
        "steps": t + 1,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2A: Deterministic Oracle WARN on Env-A (no door)")
    print("=" * 70)
    print(f"Grid: 6x8, L_A=6, L_B=8 (no door), trap rho*=1.0")
    print(f"Seeds: {N_SEEDS}")
    print()

    modes = ["none", "trap_only", "segment"]
    all_results = {}

    for mode in modes:
        results = [run_episode(s, warn_mode=mode) for s in range(N_SEEDS)]
        all_results[mode] = results

        n_A = sum(1 for r in results if r["branch"] == "A")
        n_B = sum(1 for r in results if r["branch"] == "B")
        n_trap = sum(1 for r in results if r["hit_trap"])
        n_succ = sum(1 for r in results if r["success"])
        n_warned = sum(1 for r in results if r["warned"])
        n_switched = sum(1 for r in results if r["replan_switched"])
        avg_steps = np.mean([r["steps"] for r in results])

        print(f"--- {mode:20s} ---")
        print(f"  Branch A:     {n_A:3d}/{N_SEEDS} = {100*n_A/N_SEEDS:.0f}%")
        print(f"  Branch B:     {n_B:3d}/{N_SEEDS} = {100*n_B/N_SEEDS:.0f}%")
        print(f"  Hit trap:     {n_trap:3d}/{N_SEEDS} = {100*n_trap/N_SEEDS:.0f}%")
        print(f"  CSR:          {n_succ:3d}/{N_SEEDS} = {100*n_succ/N_SEEDS:.0f}%")
        if n_warned > 0:
            print(f"  Warned:       {n_warned:3d}/{N_SEEDS}")
            print(f"  Switched B:   {n_switched:3d}/{n_warned} = {100*n_switched/max(n_warned,1):.0f}%")
        print(f"  Avg steps:    {avg_steps:.1f}")
        print()

    # Summary comparison
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':20s} {'Branch A':>10s} {'Branch B':>10s} {'Trap':>8s} {'CSR':>8s} {'Switch':>8s}")
    for mode in modes:
        r = all_results[mode]
        nA = sum(1 for x in r if x["branch"] == "A")
        nB = sum(1 for x in r if x["branch"] == "B")
        nT = sum(1 for x in r if x["hit_trap"])
        nS = sum(1 for x in r if x["success"])
        nSw = sum(1 for x in r if x["replan_switched"])
        print(f"{mode:20s} {100*nA/N_SEEDS:>9.0f}% {100*nB/N_SEEDS:>9.0f}% "
              f"{100*nT/N_SEEDS:>7.0f}% {100*nS/N_SEEDS:>7.0f}% {100*nSw/N_SEEDS:>7.0f}%")
