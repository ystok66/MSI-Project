"""
Phase 9 Experiment Matrix Runner.

Usage:
  python scripts/run_phase9_matrix.py               # full matrix
  python scripts/run_phase9_matrix.py --smoke        # smoke test (small subset)
  python scripts/run_phase9_matrix.py --filter agent=medium,teacher=robot_belief
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import os
from pathlib import Path

import yaml
import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.metrics.phase9_metrics import (
    compute_episode_summary, compute_transfer_summary,
    aggregate_summaries, aggregate_transfer_summaries,
    EpisodeSummary,
)
from src.metrics.transfer_eval import run_transfer_episodes


def load_config(path: str = "configs/phase9_eval.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_matrix(cfg: dict, smoke: bool = False, filters: dict = None):
    """Expand agent × teacher × env × scenario_family grid into job list."""
    if smoke:
        smoke_cfg = cfg.get("smoke", {})
        agent_keys = smoke_cfg.get("agent_levels", list(cfg["agent_levels"].keys()))
        teacher_keys = smoke_cfg.get("teacher_conditions", list(cfg["teacher_conditions"].keys()))
        env_keys = smoke_cfg.get("env_conditions", list(cfg["env_conditions"].keys()))
        family_keys = smoke_cfg.get("scenario_families",
                                     cfg.get("scenario_families", ["baseline_v2"]))
    else:
        agent_keys = list(cfg["agent_levels"].keys())
        teacher_keys = list(cfg["teacher_conditions"].keys())
        env_keys = list(cfg["env_conditions"].keys())
        family_keys = cfg.get("scenario_families", ["baseline_v2"])

    if filters:
        if "agent" in filters:
            agent_keys = [k for k in agent_keys if k in filters["agent"].split(",")]
        if "teacher" in filters:
            teacher_keys = [k for k in teacher_keys if k in filters["teacher"].split(",")]
        if "env" in filters:
            env_keys = [k for k in env_keys if k in filters["env"].split(",")]
        if "family" in filters:
            family_keys = [k for k in family_keys if k in filters["family"].split(",")]

    jobs = []
    for ak in agent_keys:
        for tk in teacher_keys:
            for ek in env_keys:
                for fk in family_keys:
                    jobs.append({
                        "agent_level": ak,
                        "agent_cfg": cfg["agent_levels"][ak],
                        "teacher_condition": tk,
                        "teacher_cfg": cfg["teacher_conditions"][tk],
                        "env_condition": ek,
                        "env_cfg": cfg["env_conditions"][ek],
                        "scenario_family": fk,
                    })
    return jobs


def run_job(runner, job, eval_cfg, smoke=False):
    """Run one cell of the matrix: train episodes + transfer episodes."""
    n_train = (eval_cfg.get("smoke", {}).get("n_train_episodes", 3)
               if smoke else eval_cfg["evaluation"]["n_train_episodes"])
    n_transfer = (eval_cfg.get("smoke", {}).get("n_transfer_episodes", 2)
                  if smoke else eval_cfg["evaluation"]["n_transfer_episodes"])
    seeds_start = eval_cfg["evaluation"].get("seeds_start", 0)
    latent_mode = eval_cfg["evaluation"].get("latent_mode", True)

    train_seeds = list(range(seeds_start, seeds_start + n_train))
    transfer_seeds = list(range(seeds_start + 1000, seeds_start + 1000 + n_transfer))

    agent_cfg = job["agent_cfg"]
    teacher_cfg = job["teacher_cfg"]
    env_cfg = job["env_cfg"]

    # Build runner kwargs from teacher + env config
    scenario_family = job.get("scenario_family", None)
    runner_kw = dict(
        latent_mode=latent_mode,
        difficulty=env_cfg.get("difficulty", "medium"),
    )
    if scenario_family and scenario_family != "baseline_v2":
        runner_kw["scenario_family"] = scenario_family
    # Teacher params
    for k in ("tutor_mode", "warning_mode", "lambda_lane_warn", "closure_budget",
              "robot_belief_mode", "intervention_family_mode", "item_drop_enabled",
              "prefix_horizon"):
        if k in teacher_cfg:
            runner_kw[k] = teacher_cfg[k]

    # Online training
    online_summaries = []
    last_state = None
    for seed in train_seeds:
        s = runner.reset(seed=seed, **runner_kw)
        while not s.done:
            runner.step(s)
        summary = compute_episode_summary(
            s, seed=seed,
            agent_level=job["agent_level"],
            teacher_condition=job["teacher_condition"],
            env_condition=job["env_condition"],
        )
        online_summaries.append(summary)
        last_state = s

    # Transfer evaluation
    transfer_summaries = []
    if last_state is not None:
        transfer_summaries = run_transfer_episodes(
            runner, last_state,
            n_episodes=n_transfer,
            seeds=transfer_seeds,
            agent_level=job["agent_level"],
            teacher_condition=job["teacher_condition"],
            env_condition=job["env_condition"],
            difficulty=env_cfg.get("difficulty", "medium"),
            latent_mode=latent_mode,
        )

    return online_summaries, transfer_summaries


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Experiment Matrix")
    parser.add_argument("--smoke", action="store_true", help="Run smoke subset")
    parser.add_argument("--filter", type=str, default="", help="e.g. agent=medium,teacher=robot_belief")
    parser.add_argument("--config", type=str, default="configs/phase9_eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    filters = {}
    if args.filter:
        for part in args.filter.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                filters[k] = v

    jobs = expand_matrix(cfg, smoke=args.smoke, filters=filters)
    runner = LatticeV2Runner()

    output_dir = Path(cfg["evaluation"].get("output_dir", "output/phase9"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_online = []
    all_transfer = []

    for i, job in enumerate(jobs):
        label = f"{job['agent_level']}/{job['teacher_condition']}/{job['env_condition']}/{job.get('scenario_family', 'baseline_v2')}"
        print(f"[{i+1}/{len(jobs)}] {label}")

        online, transfer = run_job(runner, job, cfg, smoke=args.smoke)

        # Aggregate
        online_agg = aggregate_summaries(
            online,
            agent_level=job["agent_level"],
            teacher_condition=job["teacher_condition"],
            env_condition=job["env_condition"],
        )
        transfer_agg = aggregate_transfer_summaries(
            transfer,
            agent_level=job["agent_level"],
            teacher_condition=job["teacher_condition"],
            env_condition=job["env_condition"],
        )

        print(f"  online:   n={online_agg.n} sr={online_agg.success_rate:.0%}")
        print(f"  transfer: n={transfer_agg.n} sr={transfer_agg.success_rate:.0%}")

        all_online.append(online_agg.to_dict())
        all_transfer.append(transfer_agg.to_dict())

    # Save results
    with open(output_dir / "online_results.json", "w") as f:
        json.dump(all_online, f, indent=2)
    with open(output_dir / "transfer_results.json", "w") as f:
        json.dump(all_transfer, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
