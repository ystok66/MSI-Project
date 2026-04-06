"""
WAIT vs WARN vs BLOCK experiment on SemanticTrap.
4 baselines x 3 diffs x 20 seeds x 10 eps = 2400 episodes.
Parallelized with 4 processes.
"""
from __future__ import annotations
import sys, os, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.envs.benchmark_generator import generate_benchmark_map, DIFFICULTIES
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_cause_teacher import OracleCauseTeacherPolicy
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.block_scoring import compute_block_decision
from src.envs.map_families import FamilyConfig

FAMILY = "semantic_trap"
BASELINES = {
    "no_teacher":      {"warn": False, "block": False},
    "wait_warn":       {"warn": True,  "block": False},
    "wait_block":      {"warn": False, "block": True},
    "wait_warn_block": {"warn": True,  "block": True},
}
N_SEEDS = 20
EPISODES_PER_SEED = 10
N_WORKERS = 4
OUT_CSV = Path(PROJECT_ROOT) / "results" / "block_experiment.csv"


def run_episode(family, seed, diff, bname, bcfg, ep):
    """Run one episode. Returns metrics dict."""
    gm, cfg = generate_benchmark_map(family, seed, diff)
    ep_seed = seed * 1000 + ep
    env = PedagogicalGridEnv(
        grid_map=gm, max_steps=cfg.max_steps,
        initial_risk_budget=cfg.risk_budget,
        prior_risk_mean=cfg.prior_risk_mean,
        prior_risk_var=cfg.prior_risk_var,
        search_budget=cfg.search_budget, seed=ep_seed,
    )
    obs, info = env.reset(seed=ep_seed)

    allow_w = bcfg["warn"]
    allow_b = bcfg["block"]
    is_no_teacher = (not allow_w and not allow_b)
    teacher = OracleCauseTeacherPolicy() if not is_no_teacher else None

    warn_count = 0
    block_count = 0
    block_on_true_hazard = 0
    obs_history = []
    H, W = gm.height, gm.width
    terminated = truncated = False

    while not terminated and not truncated:
        # Collect obs positions
        r0, c0 = env.agent.pos
        step_obs = [(r0+dr, c0+dc) for dr in range(-1,2) for dc in range(-1,2)
                     if 0 <= r0+dr < H and 0 <= c0+dc < W]
        obs_history.extend(step_obs)
        recent_obs = obs_history[-18:]

        if is_no_teacher:
            action_idx = 0
        else:
            passable = env._passable_mask()
            agent_plan = env.agent.current_plan if env.agent.current_plan else [env.agent.pos]

            intervention, _ = teacher.select_action(
                agent=env.agent,
                true_cost=env._true_cost_dynamic,
                true_risk=gm.true_risk,
                goal=env._current_goal(),
                time_left=env.max_steps - env.step_count,
                risk_budget_left=env.risk_budget_left,
                passable_mask=passable,
                door_positions=gm.door_positions,
                locked_doors=env.locked_doors,
                allow_warn=allow_w,
                allow_unlock=False,
                allow_shield=False,
                allow_block=allow_b,
                agent_plan=agent_plan,
                observation_positions_recent=recent_obs,
            )

            if intervention.type == InterventionType.WARN:
                action_idx = 1
                warn_count += 1
            elif intervention.type == InterventionType.BLOCK_PATH:
                action_idx = 4
                block_count += 1
                # Compute block cell for env
                _, block_cell, _ = compute_block_decision(
                    agent_pos=env.agent.pos,
                    agent_plan=agent_plan,
                    agent_belief_risk=env.agent.belief.risk_mean,
                    hazard_risk_map=gm.true_risk,
                    observation_positions_recent=recent_obs,
                    goal=env._current_goal(),
                    time_left=env.max_steps - env.step_count,
                    risk_budget_left=env.risk_budget_left,
                    passable_mask=passable,
                    belief_cost_mean=env.agent.belief.cost_mean,
                    belief_cost_var=env.agent.belief.cost_var,
                    search_budget=cfg.search_budget,
                )
                env.block_target = block_cell
                if block_cell and gm.true_risk[block_cell[0], block_cell[1]] > 0.15:
                    block_on_true_hazard += 1
            else:
                action_idx = 0

        obs, rew, terminated, truncated, info = env.step(action_idx)

    return {
        "family": family, "difficulty": diff, "baseline": bname,
        "seed": seed, "episode": ep,
        "success": int(info.get("object_delivered", False)),
        "steps": env.step_count,
        "warn_count": warn_count,
        "block_count": block_count,
        "block_on_hazard": block_on_true_hazard,
    }


def run_batch(tasks):
    """Run a batch of tasks (for multiprocessing)."""
    return [run_episode(*t) for t in tasks]


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Build task list
    all_tasks = []
    for diff in DIFFICULTIES:
        for bname, bcfg in BASELINES.items():
            for seed in range(N_SEEDS):
                for ep in range(EPISODES_PER_SEED):
                    all_tasks.append((FAMILY, seed, diff, bname, bcfg, ep))

    total = len(all_tasks)
    print(f"Running {total} episodes with {N_WORKERS} workers...")

    # Split into batches for workers
    batch_size = max(1, total // (N_WORKERS * 4))
    batches = [all_tasks[i:i+batch_size] for i in range(0, total, batch_size)]

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_batch, b): len(b) for b in batches}
        for future in as_completed(futures):
            batch_rows = future.result()
            rows.extend(batch_rows)
            done += futures[future]
            print(f"  [{done}/{total}]")

    # Save CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")

    # ── Print summary ──
    print(f"\n{'='*80}")
    print(f"  WAIT vs WARN vs BLOCK  --  SemanticTrap ({N_SEEDS}s x {EPISODES_PER_SEED}ep)")
    print(f"{'='*80}\n")

    hdr = f"{'Diff':<8} {'Baseline':<18} {'CSR%':>6} {'Steps':>6} {'WarnR':>6} {'BlkR':>6} {'BlkP':>6}"
    print(hdr)
    print("-" * len(hdr))

    for diff in DIFFICULTIES:
        for bname in BASELINES:
            sub = [r for r in rows if r["difficulty"] == diff and r["baseline"] == bname]
            n = len(sub)
            csr = sum(r["success"] for r in sub) / n * 100
            steps = np.mean([r["steps"] for r in sub])
            tw = sum(r["warn_count"] for r in sub)
            tb = sum(r["block_count"] for r in sub)
            tbh = sum(r["block_on_hazard"] for r in sub)
            wr = tw / n
            br = tb / n
            bp = tbh / max(tb, 1) * 100
            print(f"{diff:<8} {bname:<18} {csr:>5.1f}% {steps:>6.1f} {wr:>5.2f} {br:>5.2f} {bp:>5.0f}%")
        print()


if __name__ == "__main__":
    main()
