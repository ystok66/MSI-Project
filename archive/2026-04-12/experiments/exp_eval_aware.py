"""
exp_eval_aware.py -- Experiment A: Eval-aware tutor objective function sweep.

Conditions:
  A0: legacy scorer (lambda_probe=0, tutor_scorer_mode="legacy")
  A1: eval_aware, lambda_probe=0.5
  A2: eval_aware, lambda_probe=1.0
  A3: eval_aware, lambda_probe=2.0
  A4: probe-only, lambda_now=0, lambda_probe=1.0

All conditions: use_cls=True, reveal_learning_mode="cortex_em"

Sweep: n_sup x N_teach x seeds x grammars
Metrics: EVAL_SR, OBS_SR, TEACH_SR, TransferGap, action distribution,
         ΔProbe calibration (shadow prediction vs actual eval change)

Usage:
  python cls_option_tutor/exp_eval_aware.py --smoke --workers 4
  python cls_option_tutor/exp_eval_aware.py --workers 12
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.tutor_agent import TutorAgent


# ── Data directory ──
DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'BASIC', 'cls_learner', 'data')


# ── Conditions ──
CONDITIONS = {
    "A0_legacy": {"scorer_mode": "legacy", "lambda_now": 1.0, "lambda_probe": 0.0},
    "A1_lp05":   {"scorer_mode": "eval_aware", "lambda_now": 1.0, "lambda_probe": 0.5},
    "A2_lp10":   {"scorer_mode": "eval_aware", "lambda_now": 1.0, "lambda_probe": 1.0},
    "A3_lp20":   {"scorer_mode": "eval_aware", "lambda_now": 1.0, "lambda_probe": 2.0},
    "A4_probe_only": {"scorer_mode": "eval_aware", "lambda_now": 0.0, "lambda_probe": 1.0},
}


@dataclass
class JobResult:
    condition: str
    task_id: str
    seed: int
    n_sup: int
    n_teach: int
    # Phase metrics
    obs_correct: int = 0
    obs_total: int = 0
    teach_correct: int = 0
    teach_total: int = 0
    eval_correct: int = 0
    eval_total: int = 0
    # Aggregates
    total_damage: int = 0
    total_skips: int = 0
    # Action distribution
    action_counts: Dict[str, int] = None
    # Timing
    wall_time: float = 0.0
    # Calibration: shadow predictions vs actual
    shadow_delta_predictions: List[float] = None
    actual_eval_delta: float = 0.0
    error: str = ""

    def __post_init__(self):
        if self.action_counts is None:
            self.action_counts = {}
        if self.shadow_delta_predictions is None:
            self.shadow_delta_predictions = []


def run_one_job(
    condition: str,
    task_id: str,
    seed: int,
    n_sup: int,
    n_teach: int,
    data_dir: str,
) -> JobResult:
    """Run one experimental job."""
    result = JobResult(
        condition=condition, task_id=task_id, seed=seed,
        n_sup=n_sup, n_teach=n_teach)
    t0 = time.time()

    try:
        params = CONDITIONS[condition]

        cfg = FullConfig()
        cfg.seed = seed
        cfg.learner.use_cls = True
        cfg.learner.n_sup = n_sup
        cfg.learner.n_em = 2
        cfg.learner.use_hpc = True
        cfg.learner.reveal_learning_mode = "cortex_em"

        cfg.env.N_obs = 2
        cfg.env.N_teach = n_teach
        cfg.env.N_eval = 3
        cfg.env.M_queries = cfg.env.N_obs + cfg.env.N_teach + cfg.env.N_eval

        cfg.tutor.tutor_scorer_mode = params["scorer_mode"]
        cfg.tutor.lambda_now = params["lambda_now"]
        cfg.tutor.lambda_probe = params["lambda_probe"]
        cfg.tutor.n_probe = 12

        env = OptionEnv(cfg=cfg, data_dir=data_dir)
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, task_id, seed=seed)

        # Extract phase-level metrics
        obs_end = cfg.env.N_obs
        teach_end = obs_end + cfg.env.N_teach
        eval_end = teach_end + cfg.env.N_eval

        action_counts = {}
        for ts in block.tutor_trace:
            a = ts.action
            action_counts[a] = action_counts.get(a, 0) + 1

        for i, qs in enumerate(block.queries):
            if i < obs_end:
                result.obs_total += 1
                if qs.success:
                    result.obs_correct += 1
            elif i < teach_end:
                result.teach_total += 1
                if qs.success:
                    result.teach_correct += 1
            else:
                result.eval_total += 1
                if qs.success:
                    result.eval_correct += 1

        result.total_damage = block.total_damage
        result.total_skips = block.total_skips
        result.action_counts = action_counts

    except Exception as e:
        result.error = str(e)

    result.wall_time = time.time() - t0
    return result


def aggregate_results(results: List[JobResult]) -> Dict:
    """Aggregate results by condition."""
    groups = {}
    for r in results:
        if r.error:
            continue
        key = (r.condition, r.n_sup, r.n_teach)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    summary = {}
    for key, group in sorted(groups.items()):
        cond, nsup, nt = key
        n = len(group)
        obs_sr = np.mean([g.obs_correct / max(g.obs_total, 1) for g in group])
        teach_sr = np.mean([g.teach_correct / max(g.teach_total, 1) for g in group])
        eval_sr = np.mean([g.eval_correct / max(g.eval_total, 1) for g in group])
        eval_se = np.std([g.eval_correct / max(g.eval_total, 1) for g in group]) / max(np.sqrt(n), 1)
        transfer_gap = teach_sr - eval_sr
        avg_damage = np.mean([g.total_damage for g in group])
        avg_time = np.mean([g.wall_time for g in group])

        # Action distribution
        total_actions = {}
        for g in group:
            for a, c in g.action_counts.items():
                total_actions[a] = total_actions.get(a, 0) + c
        total_a = sum(total_actions.values()) or 1
        action_pct = {a: 100.0 * c / total_a for a, c in sorted(total_actions.items())}

        summary[f"{cond}|nsup={nsup}|nt={nt}"] = {
            "n": n,
            "OBS_SR": round(obs_sr, 4),
            "TEACH_SR": round(teach_sr, 4),
            "EVAL_SR": round(eval_sr, 4),
            "EVAL_SE": round(eval_se, 4),
            "TransferGap": round(transfer_gap, 4),
            "AvgDamage": round(avg_damage, 2),
            "AvgTime_s": round(avg_time, 2),
            "Actions": action_pct,
        }

    return summary


def print_summary(summary: Dict, output_file: str = None):
    """Print summary table."""
    lines = []
    header = (f"{'Condition':<35} {'N':>3} {'OBS_SR':>7} {'TEACH_SR':>9} "
              f"{'EVAL_SR':>8} {'SE':>6} {'Gap':>6} {'Dmg':>5} {'Time':>6}  Actions")
    lines.append(header)
    lines.append("=" * len(header) + "=" * 40)

    for key, vals in sorted(summary.items()):
        action_str = " ".join(f"{a}:{v:.0f}%" for a, v in vals["Actions"].items())
        line = (f"{key:<35} {vals['n']:>3} {vals['OBS_SR']:>7.3f} {vals['TEACH_SR']:>9.3f} "
                f"{vals['EVAL_SR']:>8.3f} {vals['EVAL_SE']:>6.3f} "
                f"{vals['TransferGap']:>6.3f} {vals['AvgDamage']:>5.1f} "
                f"{vals['AvgTime_s']:>6.1f}  {action_str}")
        lines.append(line)

    output = "\n".join(lines)
    print(output)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output + "\n\n")
            f.write("--- JSON ---\n")
            f.write(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Exp A: Eval-aware tutor sweep")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str,
                        default="cls_option_tutor/results/eval_aware_expA.txt")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Discover available tasks
    task_files = sorted([
        f.replace('.txt', '') for f in os.listdir(DATA_DIR)
        if f.endswith('.txt')
    ])

    if args.smoke:
        seeds = [42, 43, 44]
        grammars = task_files[:2]
        n_sups = [2, 6]
        n_teaches = [1]
        conditions = ["A0_legacy", "A1_lp05", "A2_lp10"]
    else:
        seeds = list(range(42, 62))  # 20 seeds
        grammars = task_files[:3]
        n_sups = [2, 4, 6, 8]
        n_teaches = [1, 2]
        conditions = list(CONDITIONS.keys())

    if args.conditions:
        conditions = [c for c in conditions if c in args.conditions]

    # Build job list
    jobs = []
    for cond in conditions:
        for task_id in grammars:
            for seed in seeds:
                for ns in n_sups:
                    for nt in n_teaches:
                        jobs.append((cond, task_id, seed, ns, nt))

    print(f"Exp A: {len(jobs)} jobs | {len(conditions)} conditions | "
          f"{len(grammars)} grammars | {len(seeds)} seeds | "
          f"workers={args.workers}")
    print(f"Conditions: {conditions}")
    print()

    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_job, *job, DATA_DIR): job
            for job in jobs
        }

        done = 0
        errors = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if r.error:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR [{r.condition}|{r.task_id}|s{r.seed}]: {r.error}")
            results.append(r)

            if done % max(len(jobs) // 10, 1) == 0 or done == len(jobs):
                elapsed = time.time() - t_start
                print(f"  Progress: {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) "
                      f"| {errors} errors | {elapsed:.0f}s elapsed")

    print(f"\nCompleted {len(results)} jobs in {time.time() - t_start:.0f}s "
          f"({errors} errors)\n")

    # Aggregate and print
    summary = aggregate_results(results)
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()
