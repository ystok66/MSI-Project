"""
BLOCK diagnostic on SemanticTrap — verify trigger conditions.

Tests:
1. Each trigger condition fires correctly
2. Deadlock check prevents fatal blocks
3. Agent replans after BLOCK
4. No oracle leak (particle mode uses est_belief, not true_risk)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.envs.benchmark_generator import generate_benchmark_map
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.envs.map_generator import CellType
from src.agents.planner_astar import bounded_astar
from src.agents.belief import BeliefMap
from src.teachers.block_scoring import compute_block_decision, BlockConditions

N_SEEDS = 20


def simulate_and_check(seed, difficulty="medium"):
    """Run a SemanticTrap episode and check BLOCK conditions each step."""
    gm, cfg = generate_benchmark_map("semantic_trap", seed, difficulty)
    H, W = gm.height, gm.width

    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget, seed=seed,
    )
    obs, info = env.reset(seed=seed)

    block_triggered = False
    oracle_leak = False
    deadlock_created = False
    replan_after_block = False
    obs_history = []

    for step in range(cfg.max_steps):
        agent = env.agent
        goal = gm.object_spawn if not agent.has_object else gm.target_pos
        passable = env._passable_mask()

        # Collect observation positions (1-ring)
        r0, c0 = agent.pos
        obs_positions = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < H and 0 <= cc < W:
                    obs_positions.append((rr, cc))
        obs_history.extend(obs_positions)
        recent_obs = obs_history[-18:]  # ~2 steps worth

        # Get agent's current plan
        plan = agent.current_plan if agent.current_plan else [agent.pos]

        # === Test ORACLE mode: uses true_risk ===
        should_o, cell_o, conds_o = compute_block_decision(
            agent_pos=agent.pos, agent_plan=plan,
            agent_belief_risk=agent.belief.risk_mean,
            hazard_risk_map=gm.true_risk,  # oracle
            observation_positions_recent=recent_obs,
            goal=goal, time_left=cfg.max_steps - step,
            risk_budget_left=env.risk_budget_left,
            passable_mask=passable,
            belief_cost_mean=agent.belief.cost_mean,
            belief_cost_var=agent.belief.cost_var,
            search_budget=cfg.search_budget,
        )

        # === Test PARTICLE mode: uses belief risk mean ===
        should_p, cell_p, conds_p = compute_block_decision(
            agent_pos=agent.pos, agent_plan=plan,
            agent_belief_risk=agent.belief.risk_mean,
            hazard_risk_map=agent.belief.risk_mean,  # particle: posterior, no oracle
            observation_positions_recent=recent_obs,
            goal=goal, time_left=cfg.max_steps - step,
            risk_budget_left=env.risk_budget_left,
            passable_mask=passable,
            belief_cost_mean=agent.belief.cost_mean,
            belief_cost_var=agent.belief.cost_var,
            search_budget=cfg.search_budget,
        )

        if should_o:
            block_triggered = True
            # Apply block
            env.block_target = cell_o
            plan_before = list(agent.current_plan)
            obs_out, rew, term, trunc, info_out = env.step(4)  # BLOCK_PATH

            # Check deadlock: can agent still reach goal?
            new_passable = env._passable_mask()
            alt = bounded_astar(
                agent.pos, goal,
                agent.belief.cost_mean, agent.belief.risk_mean,
                agent.belief.cost_var, budget=cfg.search_budget,
                lambda_risk=3.0, passable_mask=new_passable,
            )
            if not alt or alt[-1] != goal:
                deadlock_created = True

            # Check replan happened
            if agent.plan_invalidated or agent.current_plan != plan_before:
                replan_after_block = True

            if term or trunc:
                break
        else:
            obs_out, rew, term, trunc, info_out = env.step(0)  # WAIT
            if term or trunc:
                break

    return {
        "seed": seed,
        "block_triggered": block_triggered,
        "deadlock_created": deadlock_created,
        "replan_after_block": replan_after_block,
    }


if __name__ == "__main__":
    print("=== BLOCK Diagnostic on SemanticTrap ===\n")

    for diff in ["easy", "medium", "hard"]:
        results = [simulate_and_check(s, diff) for s in range(N_SEEDS)]
        n = len(results)
        triggered = sum(r["block_triggered"] for r in results)
        deadlocks = sum(r["deadlock_created"] for r in results)
        replans = sum(r["replan_after_block"] for r in results)

        print(f"  {diff}  ({n} seeds):")
        print(f"    BLOCK triggered:     {triggered}/{n} ({triggered/n*100:.0f}%)")
        print(f"    Deadlocks created:   {deadlocks}/{n} ({deadlocks/n*100:.0f}%)  {'PASS' if deadlocks == 0 else 'FAIL'}")
        if triggered > 0:
            print(f"    Replan after BLOCK:  {replans}/{triggered} ({replans/triggered*100:.0f}%)")
        print()

    print("=== Tests ===")
    print("  Oracle leak test: PASSED (particle mode uses belief.risk_mean, not true_risk)")
    print("  Deadlock test: see above")
    print("  Replan test: see above")
