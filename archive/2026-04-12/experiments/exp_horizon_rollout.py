"""
exp_horizon_rollout.py — Exp D: Short-horizon rollout (P2) sweep.

PURPOSE:
    Test whether H=2 short-horizon rollout improves eval-aware action ranking
    over single-step shadow (H=1 = P1 baseline).

MOTIVATION (from Exp B):
    Spearman ρ < 0.10 for all P1 conditions.
    Root cause: one-step shadow simulates only the current query reveal,
    while EVAL_SR reflects 3 subsequent queries. H=2 rollout extends the
    planning horizon to cover one additional teaching query, amplifying
    the signal that differentiates pedagogical actions.

CONDITIONS:
    D0: legacy (H=1, lambda_probe=0 — pure baseline, no eval-aware)
    D1: H=1, lambda_probe=1.0  (Exp A/C best, P1 baseline)
    D2: H=2, lambda_probe=0.5  (P2 pilot)
    D3: H=2, lambda_probe=1.0  (P2 main)
    D4: H=2, lambda_probe=2.0  (P2 aggressive)
    D5: H=2, lambda_probe=4.0  (P2 very aggressive)

CALIBRATION:
    Spearman ρ between shadow q_probe and actual EVAL_SR is reported
    (same as Exp B) to validate whether H=2 improves surrogate alignment.

Usage:
    python cls_option_tutor/exp_horizon_rollout.py --smoke --workers 12
    python cls_option_tutor/exp_horizon_rollout.py --workers 12
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.tutor_agent import TutorAgent


DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'BASIC', 'cls_learner', 'data')


CONDITIONS = {
    # Baseline: no eval-aware objective
    "D0_legacy":   {"horizon": 1, "lambda_now": 1.0, "lambda_probe": 0.0,
                    "scorer_mode": "legacy"},
    # H=1: existing P1 single-step shadow
    "D1_H1_lp1":  {"horizon": 1, "lambda_now": 1.0, "lambda_probe": 1.0,
                    "scorer_mode": "eval_aware"},
    # H=2: two-step rollout, lambda sweep
    "D2_H2_lp05": {"horizon": 2, "lambda_now": 1.0, "lambda_probe": 0.5,
                    "scorer_mode": "eval_aware"},
    "D3_H2_lp1":  {"horizon": 2, "lambda_now": 1.0, "lambda_probe": 1.0,
                    "scorer_mode": "eval_aware"},
    "D4_H2_lp2":  {"horizon": 2, "lambda_now": 1.0, "lambda_probe": 2.0,
                    "scorer_mode": "eval_aware"},
    "D5_H2_lp4":  {"horizon": 2, "lambda_now": 1.0, "lambda_probe": 4.0,
                    "scorer_mode": "eval_aware"},
}


@dataclass
class JobResult:
    condition: str
    task_id: str
    seed: int
    n_sup: int
    n_teach: int
    eval_sr: float = 0.0
    obs_sr: float = 0.0
    teach_sr: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    shadow_deltas: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    error: str = ""


def run_one_job(
    condition: str,
    task_id: str,
    seed: int,
    n_sup: int,
    n_teach: int,
    data_dir: str,
) -> JobResult:
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
        cfg.tutor.shadow_rollout_horizon = params["horizon"]
        cfg.tutor.shadow_rollout_gamma = 1.0   # no discount
        # Best probe config from Exp B: acc=True, n=12 (fastest with direction signal)
        cfg.tutor.n_probe = 12
        cfg.tutor.probe_use_accuracy = True
        cfg.tutor.probe_ood_ratio = 0.0        # ID-only (OOD didn't help in Exp B)

        env = OptionEnv(cfg=cfg, data_dir=data_dir)
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, task_id, seed=seed)

        # Phase metrics
        obs_end   = cfg.env.N_obs
        teach_end = obs_end + cfg.env.N_teach
        eval_end  = teach_end + cfg.env.N_eval

        obs_c = obs_t = teach_c = teach_t = eval_c = eval_t = 0
        for i, qs in enumerate(block.queries):
            if i < obs_end:
                obs_t += 1; obs_c += int(qs.success)
            elif i < teach_end:
                teach_t += 1; teach_c += int(qs.success)
            else:
                eval_t += 1; eval_c += int(qs.success)

        result.obs_sr   = obs_c   / max(obs_t,   1)
        result.teach_sr = teach_c / max(teach_t, 1)
        result.eval_sr  = eval_c  / max(eval_t,  1)

        # Action distribution
        for ts in block.tutor_trace:
            a = ts.action
            result.action_counts[a] = result.action_counts.get(a, 0) + 1
            if ts.q_scores and 'q_probe' in ts.q_scores:
                result.shadow_deltas.append(float(ts.q_scores['q_probe']))

    except Exception as e:
        import traceback
        result.error = f"{e}\n{traceback.format_exc()[:400]}"

    result.wall_time = time.time() - t0
    return result


def aggregate(results: List[JobResult]) -> Dict:
    from scipy.stats import spearmanr

    groups: Dict[str, List[JobResult]] = {}
    for r in results:
        if r.error:
            continue
        key = f"{r.condition}|nsup={r.n_sup}|nt={r.n_teach}"
        groups.setdefault(key, []).append(r)

    summary = {}
    for key, grp in sorted(groups.items()):
        n = len(grp)
        eval_srs  = [g.eval_sr for g in grp]
        avg_eval  = np.mean(eval_srs)
        eval_se   = np.std(eval_srs) / max(np.sqrt(n), 1)
        avg_obs   = np.mean([g.obs_sr   for g in grp])
        avg_teach = np.mean([g.teach_sr for g in grp])
        avg_time  = np.mean([g.wall_time for g in grp])

        # Action distribution
        total_a: Dict[str, int] = {}
        for g in grp:
            for a, c in g.action_counts.items():
                total_a[a] = total_a.get(a, 0) + c
        denom = sum(total_a.values()) or 1
        act_pct = {a: round(100 * c / denom, 1) for a, c in sorted(total_a.items())}

        # Calibration: Spearman ρ
        mean_deltas = [np.mean(g.shadow_deltas) if g.shadow_deltas else 0.0
                       for g in grp]
        if len(mean_deltas) >= 5 and np.std(mean_deltas) > 1e-8:
            corr, pval = spearmanr(mean_deltas, eval_srs)
            rho  = float(corr) if not np.isnan(corr) else 0.0
            pval = float(pval) if not np.isnan(pval) else 1.0
        else:
            rho, pval = 0.0, 1.0

        summary[key] = {
            "n": n,
            "OBS_SR":   round(avg_obs,   4),
            "TEACH_SR": round(avg_teach, 4),
            "EVAL_SR":  round(avg_eval,  4),
            "EVAL_SE":  round(eval_se,   4),
            "Spearman_rho": round(rho,  4),
            "Spearman_p":   round(pval, 4),
            "AvgTime_s": round(avg_time, 1),
            "Actions":   act_pct,
        }
    return summary


def print_summary(summary: Dict, output_file: Optional[str] = None):
    lines = []
    hdr = (f"{'Condition':<30} {'N':>3} {'OBS_SR':>7} {'TEACH_SR':>9} "
           f"{'EVAL_SR':>8} {'SE':>6} {'ρ':>7}  Actions")
    lines.append(hdr)
    lines.append("=" * (len(hdr) + 20))

    for key, v in sorted(summary.items()):
        act_str = " ".join(f"{a}:{p}%" for a, p in v["Actions"].items())
        line = (f"{key:<30} {v['n']:>3} {v['OBS_SR']:>7.3f} {v['TEACH_SR']:>9.3f} "
                f"{v['EVAL_SR']:>8.3f} {v['EVAL_SE']:>6.3f} "
                f"{v['Spearman_rho']:>7.4f}  {act_str}")
        lines.append(line)

    output = "\n".join(lines)
    print(output)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output + "\n\n--- JSON ---\n")
            f.write(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Exp D: H=2 Rollout")
    parser.add_argument("--smoke",   action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output",  type=str,
                        default="cls_option_tutor/results/exp_horizon_rollout.txt")
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: data dir not found: {DATA_DIR}")
        sys.exit(1)

    task_files = sorted([
        f.replace('.txt', '') for f in os.listdir(DATA_DIR) if f.endswith('.txt')])

    if args.smoke:
        seeds     = [42, 43, 44]
        grammars  = task_files[:2]
        n_sups    = [4, 6]
        n_teaches = [2]
        conditions = ["D0_legacy", "D1_H1_lp1", "D3_H2_lp1"]
    else:
        seeds     = list(range(42, 62))   # 20 seeds
        grammars  = task_files[:3]
        n_sups    = [4, 6, 8]
        n_teaches = [2]
        conditions = list(CONDITIONS.keys())

    if args.conditions:
        conditions = [c for c in args.conditions if c in CONDITIONS]

    jobs = [
        (cond, task, seed, ns, nt)
        for cond in conditions
        for task in grammars
        for seed in seeds
        for ns in n_sups
        for nt in n_teaches
    ]

    print(f"Exp D: {len(jobs)} jobs | {len(conditions)} conditions | "
          f"{len(grammars)} grammars | {len(seeds)} seeds | workers={args.workers}")
    print(f"Conditions: {conditions}")
    print()

    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_job, *job, DATA_DIR): job for job in jobs}
        done = errors = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if r.error:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR [{r.condition}|{r.task_id}|s{r.seed}]: "
                          f"{r.error[:120]}")
            results.append(r)

            if done % max(len(jobs) // 10, 1) == 0 or done == len(jobs):
                elapsed = time.time() - t_start
                print(f"  Progress: {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) "
                      f"| {errors} errors | {elapsed:.0f}s elapsed")

    print(f"\nCompleted {len(results)} jobs in {time.time()-t_start:.0f}s "
          f"({errors} errors)\n")

    summary = aggregate(results)
    print_summary(summary, args.output)

    # P2 Decision Gate
    print("\n--- P2 Decision Gate ---")
    d0_eval = {k: v["EVAL_SR"] for k, v in summary.items() if "D0_legacy" in k}
    h2_eval = {k: v["EVAL_SR"] for k, v in summary.items() if "D2_H" in k or "D3_H" in k or "D4_H" in k or "D5_H" in k}

    if d0_eval and h2_eval:
        avg_d0 = np.mean(list(d0_eval.values()))
        avg_h2 = np.mean(list(h2_eval.values()))
        delta = avg_h2 - avg_d0
        print(f"  D0 (legacy) avg EVAL_SR: {avg_d0:.4f}")
        print(f"  H=2 avg EVAL_SR:         {avg_h2:.4f}")
        print(f"  Delta:                   {delta:+.4f}")
        if delta >= 0.02:
            print("  ✓ H=2 improves eval by ≥2pp → proceed to full Exp D")
        elif delta >= 0.0:
            print("  ! H=2 neutral → marginal, analyse action dist before proceeding")
        else:
            print("  ✗ H=2 hurts → investigate lambda_probe range or n_teach")


if __name__ == "__main__":
    main()
