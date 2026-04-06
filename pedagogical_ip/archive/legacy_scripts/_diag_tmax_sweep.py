"""
Phase 2A T_max Sweep: Find the right time budget for B-path execution.

Fixed: front-loaded profile + eps=0.
Sweep: T_max in {9, 10, 11, 12}.
200 seeds each.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.envs.map_families import generate_deceptive_fork
from src.envs.pedagogical_grid import PedagogicalGridEnv

FORK_CELL = (2, 2)
TRAP_CELL = (1, 4)
GOAL_CELL = (2, 5)
PATH_A_CELLS = {(1, 2), (1, 3), (1, 4), (1, 5)}
PATH_B_CELLS = {(3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5)}
PATH_A_SEGMENT = [(1, 2), (1, 3), (1, 4), (1, 5)]
FRONT_LOAD = [0.55, 0.75, 0.90, 0.30]

N_SEEDS = 200
L_B_IDEAL = 6  # fork -> goal via B shortest


def apply_oracle_warning(agent, target_cells, risk_profile, obs_var=0.005):
    for (r, c), y in zip(target_cells, risk_profile):
        old_prec = 1.0 / max(agent.belief.risk_var[r, c], 1e-10)
        obs_prec = 1.0 / obs_var
        new_prec = old_prec + obs_prec
        new_var = 1.0 / new_prec
        new_mean = new_var * (agent.belief.risk_mean[r, c] * old_prec + y * obs_prec)
        agent.belief.risk_mean[r, c] = np.clip(new_mean, 0.0, 1.0)
        agent.belief.risk_var[r, c] = new_var
    agent.plan_invalidated = True


def classify_pos(pos):
    if pos in PATH_A_CELLS: return "A"
    if pos in PATH_B_CELLS: return "B"
    return "O"


def run_episode(seed, t_max_override):
    gm, cfg = generate_deceptive_fork(seed=seed, difficulty="medium", with_door=False)
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=t_max_override,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0, prior_cost_var=0.1,
        prior_risk_mean=cfg.prior_risk_mean, prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget, lambda_risk=0.8, lambda_uncertainty=0.02,
        seed=seed,
    )
    env.reset()
    env.agent.epsilon_greedy = 0.0  # no randomness

    warned = False
    positions = [(2, 0)]
    warn_step = -1
    hit_trap = False
    success = False
    fail_cause = "timeout"

    for t in range(t_max_override):
        pos = tuple(env.agent.pos)
        if pos == FORK_CELL and not warned:
            apply_oracle_warning(env.agent, PATH_A_SEGMENT, FRONT_LOAD)
            warned = True
            warn_step = t

        obs, reward, terminated, truncated, info = env.step(0)
        pos_after = tuple(env.agent.pos)
        positions.append(pos_after)

        if pos_after == TRAP_CELL: hit_trap = True
        if pos_after == GOAL_CELL: success = True

        if terminated:
            if env.risk_budget_left <= 0: fail_cause = "fatal"
            elif env.object_delivered or pos_after == GOAL_CELL:
                fail_cause = "success"; success = True
            break
        if truncated:
            fail_cause = "timeout"; break

    # Post-hoc: classify trajectory after warning
    post_warn = []
    if warn_step >= 0:
        for p in positions[warn_step + 2:]:
            c = classify_pos(p)
            if c != "O": post_warn.append(c)

    first_branch = post_warn[0] if post_warn else "none"
    commit_B = (len(post_warn) >= 2 and post_warn[0] == "B" and post_warn[1] == "B")

    # B-path overhead: actual steps on B vs ideal
    b_steps = sum(1 for c in post_warn if c == "B")
    overhead = b_steps - L_B_IDEAL if first_branch == "B" else -1

    branch = "A" if sum(1 for c in post_warn if c == "A") > sum(1 for c in post_warn if c == "B") else \
             ("B" if any(c == "B" for c in post_warn) else "none")

    return {
        "seed": seed, "t_max": t_max_override,
        "first_branch": first_branch, "branch": branch,
        "commit_B": commit_B, "hit_trap": hit_trap, "success": success,
        "fail_cause": fail_cause, "steps": len(positions) - 1,
        "b_overhead": overhead, "post_classes": "".join(post_warn[:10]),
    }


if __name__ == "__main__":
    print("Phase 2A: T_max Sweep (front-loaded, eps=0)")
    print(f"Seeds: {N_SEEDS}")
    print()

    # Also run no_teacher baseline for each T_max
    print(f"{'T_max':>5s} {'firstB':>7s} {'commitB':>8s} {'CSR':>5s} "
          f"{'P(s|cB)':>8s} {'timeout':>8s} {'fatal':>6s} {'avgOH':>7s}")
    print("-" * 65)

    for tmax in [9, 10, 11, 12]:
        results = [run_episode(s, tmax) for s in range(N_SEEDS)]

        n_first_B = sum(1 for r in results if r["first_branch"] == "B")
        n_commit = sum(1 for r in results if r["commit_B"])
        n_succ = sum(1 for r in results if r["success"])
        n_timeout = sum(1 for r in results if r["fail_cause"] == "timeout")
        n_fatal = sum(1 for r in results if r["fail_cause"] == "fatal")

        commit_eps = [r for r in results if r["commit_B"]]
        p_succ_commit = sum(1 for r in commit_eps if r["success"]) / max(len(commit_eps), 1)

        overheads = [r["b_overhead"] for r in results if r["b_overhead"] >= 0]
        avg_oh = np.mean(overheads) if overheads else -1

        print(f"{tmax:5d} {100*n_first_B/N_SEEDS:6.0f}% {n_commit:7d} "
              f"{100*n_succ/N_SEEDS:4.0f}% {p_succ_commit:8.2f} "
              f"{n_timeout:7d} {n_fatal:5d} {avg_oh:7.1f}")

    # Also show no-teacher baseline at T_max=12 for reference
    print()
    print("--- No-teacher baseline at T_max=12 ---")
    gm0, cfg0 = generate_deceptive_fork(seed=0, difficulty="medium", with_door=False)
    from src.envs.pedagogical_grid import PedagogicalGridEnv as Env
    nt_succ = 0
    for s in range(N_SEEDS):
        gm, cfg = generate_deceptive_fork(seed=s, difficulty="medium", with_door=False)
        env = Env(grid_map=gm, max_steps=12, initial_risk_budget=cfg.risk_budget,
                  prior_cost_mean=1.0, prior_cost_var=0.1,
                  prior_risk_mean=0.02, prior_risk_var=0.20,
                  search_budget=30, lambda_risk=0.8, lambda_uncertainty=0.02, seed=s)
        env.reset()
        env.agent.epsilon_greedy = 0.0
        for t in range(12):
            _, _, term, trunc, _ = env.step(0)
            if term or trunc: break
        if env.object_delivered or tuple(env.agent.pos) == GOAL_CELL:
            nt_succ += 1
    print(f"  no_teacher CSR at T_max=12: {nt_succ}/{N_SEEDS} = {100*nt_succ/N_SEEDS:.0f}%")
