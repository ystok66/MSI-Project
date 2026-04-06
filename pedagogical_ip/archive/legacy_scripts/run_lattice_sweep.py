"""
Door Lattice Sweep: door_prob x time_ratio x 4 baselines.

9x17 grid (5 wall rows x 5 wall cols),
risk behind each door with p=0.5.

Sweep:
  door_prob: [0.5, 0.7, 0.9]
  time_ratio: [1.2, 1.4, 1.6]
  baselines: no_teacher, wait_warn, wait_block, wait_warn_block
  10 seeds x 5 episodes = 50 episodes per cell
  Total: 3 x 3 x 4 x 50 = 1800 episodes

Parallelized with 4 processes.
"""
from __future__ import annotations
import sys, os, csv, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.envs.map_families import generate_door_lattice_sanity
from src.envs.pedagogical_grid import PedagogicalGridEnv
from src.teachers.oracle_cause_teacher import OracleCauseTeacherPolicy
from src.teachers.interventions import Intervention, InterventionType
from src.teachers.block_scoring import compute_block_decision

DOOR_PROBS = [0.5, 0.7, 0.9]
TIME_RATIOS = [1.2, 1.4, 1.6]
BASELINES = {
    "no_teacher":      {"warn": False, "block": False},
    "wait_warn":       {"warn": True,  "block": False},
    "wait_block":      {"warn": False, "block": True},
    "wait_warn_block": {"warn": True,  "block": True},
}
N_SEEDS = 10
EPS_PER_SEED = 5
N_WORKERS = 4
OUT_JSON = Path(PROJECT_ROOT) / "results" / "lattice_sweep.json"


def run_episode(seed, ep, door_prob, time_ratio, bname, bcfg):
    gm, cfg = generate_door_lattice_sanity(
        seed=seed, difficulty="medium",
        door_prob_override=door_prob,
        time_ratio_override=time_ratio,
    )
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
    block_on_hazard = 0
    obs_history = []
    H, W = gm.height, gm.width
    terminated = truncated = False

    while not terminated and not truncated:
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
                allow_warn=allow_w, allow_unlock=False,
                allow_shield=False, allow_block=allow_b,
                agent_plan=agent_plan,
                observation_positions_recent=recent_obs,
            )

            if intervention.type == InterventionType.WARN:
                action_idx = 1
                warn_count += 1
            elif intervention.type == InterventionType.BLOCK_PATH:
                action_idx = 4
                block_count += 1
                _, block_cell, _ = compute_block_decision(
                    agent_pos=env.agent.pos, agent_plan=agent_plan,
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
                    block_on_hazard += 1
            else:
                action_idx = 0

        obs, rew, terminated, truncated, info = env.step(action_idx)

    return {
        "door_prob": door_prob, "time_ratio": time_ratio,
        "baseline": bname, "seed": seed, "episode": ep,
        "success": int(info.get("object_delivered", False)),
        "steps": env.step_count, "max_steps": cfg.max_steps,
        "n_doors": len(gm.door_positions),
        "warn_count": warn_count, "block_count": block_count,
        "block_on_hazard": block_on_hazard,
    }


def run_batch(tasks):
    return [run_episode(*t) for t in tasks]


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    for dp in DOOR_PROBS:
        for tr in TIME_RATIOS:
            for bname, bcfg in BASELINES.items():
                for seed in range(N_SEEDS):
                    for ep in range(EPS_PER_SEED):
                        all_tasks.append((seed, ep, dp, tr, bname, bcfg))

    total = len(all_tasks)
    print(f"Running {total} episodes with {N_WORKERS} workers...")

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
            if done % 200 == 0 or done == total:
                print(f"  [{done}/{total}]")

    # Aggregate
    summary = {}
    for dp in DOOR_PROBS:
        for tr in TIME_RATIOS:
            key = f"dp={dp}_tr={tr}"
            summary[key] = {}
            for bname in BASELINES:
                sub = [r for r in rows
                       if r["door_prob"]==dp and r["time_ratio"]==tr and r["baseline"]==bname]
                n = len(sub)
                if n == 0:
                    continue
                csr = sum(r["success"] for r in sub)/n*100
                avg_steps = np.mean([r["steps"] for r in sub])
                avg_max = np.mean([r["max_steps"] for r in sub])
                avg_doors = np.mean([r["n_doors"] for r in sub])
                tw = sum(r["warn_count"] for r in sub)
                tb = sum(r["block_count"] for r in sub)
                tbh = sum(r["block_on_hazard"] for r in sub)
                summary[key][bname] = {
                    "CSR": round(csr,1), "Steps": round(float(avg_steps),1),
                    "MaxSteps": round(float(avg_max),1),
                    "Doors": round(float(avg_doors),1),
                    "WarnRate": round(tw/n,3), "BlockRate": round(tb/n,3),
                    "BlockPrec": round(tbh/max(tb,1)*100,0),
                }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*90}")
    print(f"  Door Lattice Sweep ({N_SEEDS}s x {EPS_PER_SEED}ep)")
    print(f"{'='*90}")
    hdr = f"{'DoorP':>5} {'TimeR':>5} {'Baseline':<18} {'CSR':>5} {'Steps':>6} {'Max':>5} {'Doors':>5} {'WR':>5} {'BR':>5}"
    print(hdr)
    print("-" * len(hdr))
    for dp in DOOR_PROBS:
        for tr in TIME_RATIOS:
            key = f"dp={dp}_tr={tr}"
            for bname in BASELINES:
                d = summary[key][bname]
                print(f"{dp:>5} {tr:>5} {bname:<18} {d['CSR']:>4.1f}% {d['Steps']:>6.1f} {d['MaxSteps']:>5.0f} {d['Doors']:>5.1f} {d['WarnRate']:>5.3f} {d['BlockRate']:>5.3f}")
            print()


if __name__ == "__main__":
    main()
