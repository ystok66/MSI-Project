"""
exp_probe_calibration.py — Exp B: Surrogate Calibration Study.

PURPOSE:
    Answer one diagnostic question:
    "Does the new accuracy-based probe surrogate (eval_score) correlate
     with true downstream EVAL_SR better than the legacy margin?"

CONDITIONS:
    B0: legacy margin probe, n=12 (baseline for comparison)
    B1: accuracy probe, n=12, ID-only
    B2: accuracy probe, n=30, ID-only
    B3: accuracy probe, n=50, ID-only
    B4: accuracy probe, n=30, ID+OOD (ood_ratio=0.5)
    B5: accuracy probe, n=50, ID+OOD (ood_ratio=0.5)

CALIBRATION METRICS (per condition):
    • Spearman ρ:     rank(shadow_delta_probe) vs rank(actual_eval_sr)
    • Sign agreement: sign(shadow_delta_probe) vs sign(actual_eval_delta)
    • TopK lift:      E[eval_sr | top-quartile shadow_delta] vs rest
    • Probe timing:   time per probe_accuracy() call

THRESHOLD (from Exp A report):
    ρ ≥ 0.3 → usable surrogate (proceed to Exp C)
    ρ < 0.1 → reject surrogate design

Usage:
    python cls_option_tutor/exp_probe_calibration.py --smoke --workers 12
    python cls_option_tutor/exp_probe_calibration.py --workers 12 --output ...
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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


# ── Exp B Conditions ──
CONDITIONS = {
    # Baseline: legacy margin (old P0 probe)
    "B0_margin":   {"use_accuracy": False, "n_probe": 12,  "ood_ratio": 0.0},
    # Accuracy-based, ID-only, varying n
    "B1_acc12":    {"use_accuracy": True,  "n_probe": 12,  "ood_ratio": 0.0},
    "B2_acc30":    {"use_accuracy": True,  "n_probe": 30,  "ood_ratio": 0.0},
    "B3_acc50":    {"use_accuracy": True,  "n_probe": 50,  "ood_ratio": 0.0},
    # Accuracy-based, ID+OOD (ood_ratio=0.5), varying n
    "B4_acc30ood": {"use_accuracy": True,  "n_probe": 30,  "ood_ratio": 0.5},
    "B5_acc50ood": {"use_accuracy": True,  "n_probe": 50,  "ood_ratio": 0.5},
}


@dataclass
class CalibrationRecord:
    """Per-teaching-step shadow prediction and actual outcome."""
    action: str
    shadow_delta_probe: float    # shadow's predicted ΔProbe for chosen action
    shadow_probe_before: float
    shadow_probe_after: float
    query_idx: int


@dataclass
class JobResult:
    condition: str
    task_id: str
    seed: int
    n_sup: int
    n_teach: int
    # Performance metrics
    eval_sr: float = 0.0
    obs_sr: float = 0.0
    teach_sr: float = 0.0
    # Calibration data
    calibration_records: List[CalibrationRecord] = field(default_factory=list)
    # Per-query shadow vs actual
    shadow_deltas: List[float] = field(default_factory=list)  # shadow ΔProbe each step
    actual_eval_sr: float = 0.0    # block-level actual EVAL_SR
    # Probe capacity
    n_id_probes: int = 0
    n_ood_probes: int = 0
    # Timing
    wall_time: float = 0.0
    probe_eval_time: float = 0.0  # time spent in probe evaluation
    error: str = ""

    def __post_init__(self):
        if self.calibration_records is None:
            self.calibration_records = []
        if self.shadow_deltas is None:
            self.shadow_deltas = []


def run_one_job(
    condition: str,
    task_id: str,
    seed: int,
    n_sup: int,
    n_teach: int,
    data_dir: str,
) -> JobResult:
    """Run one calibration job, capturing shadow predictions vs actuals."""
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

        cfg.tutor.tutor_scorer_mode = "eval_aware"
        cfg.tutor.lambda_now = 1.0
        cfg.tutor.lambda_probe = 1.0           # always 1.0 for calibration study
        cfg.tutor.n_probe = params["n_probe"]
        cfg.tutor.probe_ood_ratio = params["ood_ratio"]
        cfg.tutor.probe_use_accuracy = params["use_accuracy"]

        env = OptionEnv(cfg=cfg, data_dir=data_dir)
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, task_id, seed=seed)

        # Extract probe capacity from probe evaluator
        if tutor._probe_evaluator is not None:
            cap = tutor._probe_evaluator.capacity
            result.n_id_probes  = cap["n_id"]
            result.n_ood_probes = cap["n_ood"]

        # Extract phase metrics
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
        result.actual_eval_sr = result.eval_sr

        # Extract shadow calibration records from tutor trace
        # q_probe is stored in TutorStep.q_scores['q_probe'] by tutor_agent.act()
        if hasattr(block, 'tutor_trace'):
            for ts in block.tutor_trace:
                if ts.q_scores and 'q_probe' in ts.q_scores:
                    result.shadow_deltas.append(float(ts.q_scores['q_probe']))


    except Exception as e:
        import traceback
        result.error = f"{e}\n{traceback.format_exc()[:300]}"

    result.wall_time = time.time() - t0
    return result


def compute_calibration_metrics(results: List[JobResult]) -> Dict:
    """Compute Spearman ρ, sign agreement, and top-k lift per condition."""
    from scipy.stats import spearmanr

    groups: Dict[str, List[JobResult]] = {}
    for r in results:
        if r.error:
            continue
        key = f"{r.condition}|nsup={r.n_sup}|nt={r.n_teach}"
        groups.setdefault(key, []).append(r)

    summary = {}
    for key, group in sorted(groups.items()):
        n = len(group)
        eval_srs    = [g.actual_eval_sr for g in group]
        n_id        = int(np.mean([g.n_id_probes  for g in group]))
        n_ood       = int(np.mean([g.n_ood_probes for g in group]))
        avg_time    = np.mean([g.wall_time for g in group])
        avg_eval_sr = np.mean(eval_srs)
        eval_se     = np.std(eval_srs) / max(np.sqrt(n), 1)

        # Calibration metrics: use mean shadow_delta per job as signal
        mean_shadow_deltas = []
        for g in group:
            if g.shadow_deltas:
                mean_shadow_deltas.append(np.mean(g.shadow_deltas))
            else:
                mean_shadow_deltas.append(0.0)

        # Spearman rank correlation: shadow_delta vs actual_eval_sr
        if len(mean_shadow_deltas) >= 5:
            corr, pval = spearmanr(mean_shadow_deltas, eval_srs)
            spearman_rho = float(corr) if not np.isnan(corr) else 0.0
            spearman_p   = float(pval) if not np.isnan(pval) else 1.0
        else:
            spearman_rho, spearman_p = 0.0, 1.0

        # Sign agreement: fraction where sign(shadow_delta) == sign(eval_sr - mean)
        mean_eval = np.mean(eval_srs)
        sign_agrees = sum(
            1 for d, e in zip(mean_shadow_deltas, eval_srs)
            if np.sign(d) == np.sign(e - mean_eval)
        ) / max(n, 1)

        # Top-k lift: eval_sr for top 25% shadow_delta vs bottom 75%
        if n >= 8:
            k = max(1, n // 4)
            ranked = sorted(zip(mean_shadow_deltas, eval_srs),
                           key=lambda x: x[0], reverse=True)
            top_k_eval  = np.mean([e for _, e in ranked[:k]])
            rest_eval   = np.mean([e for _, e in ranked[k:]])
            topk_lift   = float(top_k_eval - rest_eval)
        else:
            topk_lift = float("nan")

        summary[key] = {
            "n": n,
            "EVAL_SR": round(avg_eval_sr, 4),
            "EVAL_SE": round(eval_se, 4),
            "Spearman_rho": round(spearman_rho, 4),
            "Spearman_p":   round(spearman_p, 4),
            "SignAgreement": round(sign_agrees, 3),
            "TopK_lift": round(topk_lift, 4) if not np.isnan(topk_lift) else "N/A",
            "n_id_probes":  n_id,
            "n_ood_probes": n_ood,
            "AvgTime_s": round(avg_time, 2),
            "ρ_usable": spearman_rho >= 0.30,  # threshold from Exp A report
        }

    return summary


def print_calibration_summary(summary: Dict, output_file: str = None):
    """Print calibration results table."""
    lines = []
    header = (f"{'Condition':<25} {'N':>3} {'EVAL_SR':>8} {'Spearman_ρ':>11} "
              f"{'p-val':>7} {'SignAgree':>10} {'TopK_lift':>10} "
              f"{'ID':>4} {'OOD':>4} {'Time':>6}  {'Usable?':>8}")
    lines.append(header)
    lines.append("=" * len(header))

    for key, v in sorted(summary.items()):
        topk = f"{v['TopK_lift']:.4f}" if v['TopK_lift'] != "N/A" else "  N/A  "
        usable = "✓ YES" if v["ρ_usable"] else "✗ NO "
        line = (f"{key:<25} {v['n']:>3} {v['EVAL_SR']:>8.3f} "
                f"{v['Spearman_rho']:>11.4f} {v['Spearman_p']:>7.4f} "
                f"{v['SignAgreement']:>10.3f} {topk:>10} "
                f"{v['n_id_probes']:>4} {v['n_ood_probes']:>4} "
                f"{v['AvgTime_s']:>6.1f}  {usable:>8}")
        lines.append(line)

    lines.append("")
    lines.append("Threshold: Spearman ρ ≥ 0.30 → usable (proceed to Exp C)")
    lines.append("           Spearman ρ < 0.10 → reject surrogate design")

    output = "\n".join(lines)
    print(output)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output + "\n\n")
            f.write("--- JSON ---\n")
            f.write(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {output_file}")


def print_grammar_capacity(data_dir: str, task_files: List[str]) -> None:
    """Print OOD synthesis capacity for each grammar."""
    from cls_option_tutor.grammar.task_adapter import TaskAdapter
    from cls_option_tutor.grammar.query_synthesizer import synthesize_queries

    adapter = TaskAdapter(data_dir)

    print("\n--- Grammar OOD Synthesis Capacity ---")
    print(f"{'Grammar':<14} {'ID(d3,L6)':>10} {'OOD(d5,L8)':>11} {'ood≥25?':>9} {'Total50?':>9}")
    print("-" * 58)

    all_sufficient = True
    for tid in task_files:
        support, queries, grammar = adapter.load_task(tid)
        existing = support + queries

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(100)

        id_probes  = synthesize_queries(grammar, n=200, max_depth=3, max_len=6,
                                        rng=rng1, existing=existing)
        ood_d5l8   = synthesize_queries(grammar, n=200, max_depth=5, max_len=8,
                                        rng=rng2, existing=existing + id_probes)

        id_ok  = len(id_probes)  >= 25
        ood_ok = len(ood_d5l8)   >= 25
        both   = id_ok and ood_ok
        if not both:
            all_sufficient = False

        print(f"{tid:<14} {len(id_probes):>10} {len(ood_d5l8):>11} "
              f"{'YES' if ood_ok else 'LOW':>9} {'OK' if both else 'LOW':>9}")

    print()
    if not all_sufficient:
        print("WARNING: Some grammars have low OOD capacity (<25 samples).")
        print("         B4/B5 (ood_ratio=0.5) will fall back to depth-4 probes.")
    else:
        print("All grammars have sufficient ID + OOD capacity for n_probe=50.")
    print()



def main():
    parser = argparse.ArgumentParser(description="Exp B: Probe Surrogate Calibration")
    parser.add_argument("--smoke",   action="store_true", help="Quick smoke test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output",  type=str,
                        default="cls_option_tutor/results/exp_probe_calibration.txt")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--skip-capacity", action="store_true",
                        help="Skip grammar OOD capacity scan")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    task_files = sorted([
        f.replace('.txt', '') for f in os.listdir(DATA_DIR)
        if f.endswith('.txt')
    ])

    if args.smoke:
        seeds     = [42, 43, 44]
        grammars  = task_files[:2]
        n_sups    = [4, 6]
        n_teaches = [2]
        conditions = ["B0_margin", "B1_acc12", "B2_acc30", "B4_acc30ood"]
    else:
        seeds     = list(range(42, 62))   # 20 seeds
        grammars  = task_files[:3]
        n_sups    = [4, 6, 8]
        n_teaches = [2]
        conditions = list(CONDITIONS.keys())

    if args.conditions:
        conditions = [c for c in conditions if c in CONDITIONS]

    # Step 0: Grammar capacity scan
    if not args.skip_capacity:
        print_grammar_capacity(DATA_DIR, grammars)

    # Build job list
    jobs = []
    for cond in conditions:
        for task_id in grammars:
            for seed in seeds:
                for ns in n_sups:
                    for nt in n_teaches:
                        jobs.append((cond, task_id, seed, ns, nt))

    print(f"Exp B: {len(jobs)} jobs | {len(conditions)} conditions | "
          f"{len(grammars)} grammars | {len(seeds)} seeds | workers={args.workers}")
    print(f"Conditions: {conditions}")
    print()

    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_job, *job, DATA_DIR): job
            for job in jobs
        }

        done = errors = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if r.error:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR [{r.condition}|{r.task_id}|s{r.seed}]: {r.error[:120]}")
            results.append(r)

            if done % max(len(jobs) // 10, 1) == 0 or done == len(jobs):
                elapsed = time.time() - t_start
                print(f"  Progress: {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) "
                      f"| {errors} errors | {elapsed:.0f}s elapsed")

    print(f"\nCompleted {len(results)} jobs in {time.time()-t_start:.0f}s "
          f"({errors} errors)\n")

    summary = compute_calibration_metrics(results)
    print_calibration_summary(summary, args.output)

    # Decision gate
    print("\n--- Decision Gate ---")
    best_rho = max((v["Spearman_rho"] for v in summary.values()), default=0.0)
    if best_rho >= 0.30:
        print(f"✓ Best Spearman ρ = {best_rho:.3f} ≥ 0.30. Proceed to Exp C.")
    elif best_rho >= 0.10:
        print(f"! Best Spearman ρ = {best_rho:.3f}. Marginal. Review before Exp C.")
    else:
        print(f"✗ Best Spearman ρ = {best_rho:.3f} < 0.10. Reject probe design.")


if __name__ == "__main__":
    main()
