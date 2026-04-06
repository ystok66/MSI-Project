"""
Phase 2A Reversion Diagnosis v2: Fixed switch detection.

The warning is applied at fork, but the agent replans inside env.step().
So we detect branch switch by comparing post-fork trajectory positions,
not by checking the plan object before replan occurs.

Three experiments on Env-A (no door) with segment warning:
  Exp A: Original (epsilon=0.05)
  Exp B: epsilon=0
  Exp C: Front-loaded profile (0.55/0.75/0.90/0.30)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.envs.map_families import generate_deceptive_fork
from src.envs.pedagogical_grid import PedagogicalGridEnv

# ── Geometry ──
FORK_CELL = (2, 2)
TRAP_CELL = (1, 4)
GOAL_CELL = (2, 5)
PATH_A_CELLS = {(1, 2), (1, 3), (1, 4), (1, 5)}
PATH_B_CELLS = {(3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5)}
PATH_A_SEGMENT = [(1, 2), (1, 3), (1, 4), (1, 5)]

PROFILES = {
    "original":    [0.20, 0.40, 0.85, 0.15],
    "front_load":  [0.55, 0.75, 0.90, 0.30],
}

N_SEEDS = 200


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
    return "O"  # other (fork, start, goal)


def run_episode(seed, epsilon=0.05, profile_name="original"):
    gm, cfg = generate_deceptive_fork(seed=seed, difficulty="medium", with_door=False)
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_cost_mean=1.0, prior_cost_var=0.1,
        prior_risk_mean=cfg.prior_risk_mean, prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget, lambda_risk=0.8, lambda_uncertainty=0.02,
        seed=seed,
    )
    env.reset()
    env.agent.epsilon_greedy = epsilon

    profile = PROFILES[profile_name]
    warned = False
    positions = [(2, 0)]  # start
    hit_trap = False
    success = False
    fail_cause = "timeout"
    warn_step = -1

    for t in range(cfg.max_steps):
        pos = tuple(env.agent.pos)

        # Warning at fork (before env.step triggers replan)
        if pos == FORK_CELL and not warned:
            apply_oracle_warning(env.agent, PATH_A_SEGMENT, profile)
            warned = True
            warn_step = t

        obs, reward, terminated, truncated, info = env.step(0)
        pos_after = tuple(env.agent.pos)
        positions.append(pos_after)

        if pos_after == TRAP_CELL:
            hit_trap = True
        if pos_after == GOAL_CELL:
            success = True

        if terminated:
            if env.risk_budget_left <= 0:
                fail_cause = "fatal"
            elif env.object_delivered or pos_after == GOAL_CELL:
                fail_cause = "success"
                success = True
            break
        if truncated:
            fail_cause = "timeout"
            break

    # ── Post-hoc analysis of trajectory ──
    # After fork, which cells did agent visit first?
    post_warn_classes = []
    if warn_step >= 0:
        # positions after warn_step+1 (since env.step advances)
        for p in positions[warn_step + 2:]:
            c = classify_pos(p)
            if c != "O":
                post_warn_classes.append(c)

    # First branch entered after warning
    first_branch = post_warn_classes[0] if post_warn_classes else "none"

    # Commit B: first 2 non-O positions both B
    commit_B = (len(post_warn_classes) >= 2 and
                post_warn_classes[0] == "B" and post_warn_classes[1] == "B")

    # Revert: was on B then went to A
    revert_to_A = False
    was_on_B = False
    for c in post_warn_classes:
        if c == "B":
            was_on_B = True
        if was_on_B and c == "A":
            revert_to_A = True
            break

    # Overall branch: which branch did agent mainly use
    n_A = sum(1 for c in post_warn_classes if c == "A")
    n_B = sum(1 for c in post_warn_classes if c == "B")
    branch = "A" if n_A > n_B else ("B" if n_B > 0 else "none")

    return {
        "seed": seed,
        "warned": warned,
        "first_branch": first_branch,
        "branch": branch,
        "commit_B": commit_B,
        "revert_to_A": revert_to_A,
        "hit_trap": hit_trap,
        "success": success,
        "fail_cause": fail_cause,
        "steps": len(positions) - 1,
        "post_classes": "".join(post_warn_classes[:8]),
        "time_left_at_warn": cfg.max_steps - warn_step if warn_step >= 0 else -1,
    }


def analyze(results, label):
    N = len(results)
    n_warned = sum(1 for r in results if r["warned"])
    n_first_A = sum(1 for r in results if r["first_branch"] == "A")
    n_first_B = sum(1 for r in results if r["first_branch"] == "B")
    n_commit_B = sum(1 for r in results if r["commit_B"])
    n_revert = sum(1 for r in results if r["revert_to_A"])
    n_branch_A = sum(1 for r in results if r["branch"] == "A")
    n_branch_B = sum(1 for r in results if r["branch"] == "B")
    n_trap = sum(1 for r in results if r["hit_trap"])
    n_succ = sum(1 for r in results if r["success"])
    n_timeout = sum(1 for r in results if r["fail_cause"] == "timeout")
    n_fatal = sum(1 for r in results if r["fail_cause"] == "fatal")

    # Conditional on warned
    warned_eps = [r for r in results if r["warned"]]

    # P(first_B | warned)
    p_first_B = n_first_B / max(n_warned, 1)

    # P(commit_B | first_B)
    first_B_eps = [r for r in warned_eps if r["first_branch"] == "B"]
    p_commit_given_first = sum(1 for r in first_B_eps if r["commit_B"]) / max(len(first_B_eps), 1)

    # P(success | commit_B)
    commit_B_eps = [r for r in results if r["commit_B"]]
    p_succ_commit = sum(1 for r in commit_B_eps if r["success"]) / max(len(commit_B_eps), 1)

    # P(success | first_B)
    p_succ_first_B = sum(1 for r in first_B_eps if r["success"]) / max(len(first_B_eps), 1)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  N={N}  warned={n_warned}")
    print(f"  Overall:    A={n_branch_A}({100*n_branch_A/N:.0f}%)  "
          f"B={n_branch_B}({100*n_branch_B/N:.0f}%)  "
          f"trap={n_trap}({100*n_trap/N:.0f}%)  "
          f"CSR={n_succ}({100*n_succ/N:.0f}%)")
    print(f"  Fail:       fatal={n_fatal}  timeout={n_timeout}")
    print()
    print(f"  Post-warn first move:  A={n_first_A}  B={n_first_B}  "
          f"P(first_B|warn)={p_first_B:.2f}")
    print(f"  Commit B (2 steps):    {n_commit_B}  "
          f"P(commit|first_B)={p_commit_given_first:.2f}")
    print(f"  Revert A after B:      {n_revert}")
    print()
    print(f"  P(success | commit_B):   {p_succ_commit:.2f}  (n={len(commit_B_eps)})")
    print(f"  P(success | first_B):    {p_succ_first_B:.2f}  (n={len(first_B_eps)})")

    # Example trajectories
    interesting = sorted(warned_eps, key=lambda r: r["first_branch"])[:8]
    print(f"\n  Example post-warn trajectories:")
    for r in interesting:
        print(f"    seed={r['seed']:3d}: first={r['first_branch']}  "
              f"classes={r['post_classes']:8s}  commit_B={r['commit_B']}  "
              f"revert={r['revert_to_A']}  {r['fail_cause']:8s}  T_left={r['time_left_at_warn']}")


if __name__ == "__main__":
    print("Phase 2A Reversion Diagnosis v2")
    print(f"{N_SEEDS} seeds per experiment")

    # Exp A: Original profile, epsilon=0.05
    res_A = [run_episode(s, epsilon=0.05, profile_name="original") for s in range(N_SEEDS)]
    analyze(res_A, "Exp A: Original profile, eps=0.05")

    # Exp B: Original profile, epsilon=0
    res_B = [run_episode(s, epsilon=0.0, profile_name="original") for s in range(N_SEEDS)]
    analyze(res_B, "Exp B: Original profile, eps=0")

    # Exp C: Front-loaded profile, epsilon=0.05
    res_C = [run_episode(s, epsilon=0.05, profile_name="front_load") for s in range(N_SEEDS)]
    analyze(res_C, "Exp C: Front-loaded profile (0.55/0.75/0.90/0.30), eps=0.05")

    # Exp D: Front-loaded, epsilon=0
    res_D = [run_episode(s, epsilon=0.0, profile_name="front_load") for s in range(N_SEEDS)]
    analyze(res_D, "Exp D: Front-loaded profile, eps=0")
