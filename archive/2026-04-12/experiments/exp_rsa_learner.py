"""
exp_rsa_learner.py — Exp F: RSA L1 Learner evaluation.

CONDITIONS:
    F0: legacy learner, WAIT-only tutor             (baseline)
    F1: RSA learner (sem+risk), reveal=off, WAIT    (pure action learning)
    F2: RSA learner (sem+risk), reveal=on,  WAIT    (RSA + reveal combined)
    F3: RSA learner (sem-only), reveal=on,  WAIT    (semantic channel only)
    F4: RSA learner (risk-only), reveal=on, WAIT    (risk channel only)
    F5: RSA learner (sem+risk), reveal=on,  L0      (RSA + L0 tutor; single-thread)

KEY COMPARISONS:
    F0 vs F2: RSA + reveal vs legacy (clean 1-variable)
    F2 vs F5: Impact of intelligent L0 tutor
    F2 vs F3: Semantic RSA alone
    F2 vs F4: Risk RSA alone
    F1 vs F0: Pure RSA action learning (no reveal)

PARALLELISM:
    F0-F4: ProcessPoolExecutor (stateless workers, each constructs its own objects)
    F5:    Single-thread (L0 tutor holds reference to live LearnerAgent)

USAGE:
    python exp_rsa_learner.py --smoke             # quick sanity
    python exp_rsa_learner.py --workers 16        # full run
    python exp_rsa_learner.py --cond F0_legacy F2_rsa_reveal
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment conditions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Condition:
    name: str
    use_rsa: bool
    ban_teaches_risk: bool
    omega_hl: float
    omega_ban: float
    reveal_mode: str        # "cortex_em" | "off"
    use_l0_tutor: bool

CONDITIONS = [
    Condition("F0_legacy",
               use_rsa=False, ban_teaches_risk=False,
               omega_hl=0, omega_ban=0,
               reveal_mode="cortex_em", use_l0_tutor=False),
    Condition("F1_rsa_noreveal",
               use_rsa=True, ban_teaches_risk=True,
               omega_hl=2.0, omega_ban=3.0,
               reveal_mode="off", use_l0_tutor=False),
    Condition("F2_rsa_reveal",
               use_rsa=True, ban_teaches_risk=True,
               omega_hl=2.0, omega_ban=3.0,
               reveal_mode="cortex_em", use_l0_tutor=False),
    Condition("F3_sem_only",
               use_rsa=True, ban_teaches_risk=False,
               omega_hl=2.0, omega_ban=0.0,
               reveal_mode="cortex_em", use_l0_tutor=False),
    Condition("F4_risk_only",
               use_rsa=True, ban_teaches_risk=True,
               omega_hl=0.0, omega_ban=3.0,
               reveal_mode="cortex_em", use_l0_tutor=False),
    Condition("F5_rsa_l0tutor",
               use_rsa=True, ban_teaches_risk=True,
               omega_hl=2.0, omega_ban=3.0,
               reveal_mode="cortex_em", use_l0_tutor=True),
]

COND_BY_NAME = {c.name: c for c in CONDITIONS}


# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(cond: Condition, n_sup: int, grammar_id: str) -> FullConfig:
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.n_sup = n_sup
    cfg.learner.n_em = 2
    cfg.learner.use_hpc = True
    cfg.learner.reveal_learning_mode = cond.reveal_mode
    cfg.env.N_obs = 2
    cfg.env.N_teach = 2
    cfg.env.N_eval = 3
    cfg.env.M_queries = 7
    cfg.tutor.tutor_scorer_mode = "legacy"
    cfg.rsa.use_rsa = cond.use_rsa
    cfg.rsa.omega_hl = cond.omega_hl
    cfg.rsa.omega_ban = cond.omega_ban
    cfg.rsa.ban_teaches_risk = cond.ban_teaches_risk
    cfg.rsa.use_l0_tutor = cond.use_l0_tutor
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Phase SR helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_phase_sr(block) -> tuple:
    """Compute per-phase success rates from block queries.
    Returns (obs_sr, teach_sr, eval_sr) as floats.
    """
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries

    def _phase_sr(start, end):
        qs_list = block.queries[start:end]
        if not qs_list:
            return 0.0
        return sum(1 for q in qs_list if q.success) / len(qs_list)

    return (
        _phase_sr(0, obs_end),
        _phase_sr(obs_end, teach_end),
        _phase_sr(teach_end, len(block.queries)),
    )


def _tally_actions(block) -> Dict[str, float]:
    total = len(block.tutor_trace) or 1
    counts: Dict[str, int] = {}
    for ts in block.tutor_trace:
        counts[ts.action] = counts.get(ts.action, 0) + 1
    return {k: round(v * 100 / total, 1) for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Single-job runners
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_data_dir(file_anchor: str) -> str:
    d = os.path.join(os.path.dirname(file_anchor), '..', 'BASIC', 'cls_learner', 'data')
    if os.path.isdir(d):
        return d
    return os.path.join('BASIC', 'cls_learner', 'data')


def run_job(args):
    """Single job for F0-F4 (processpool-safe)."""
    cond_name, grammar_id, n_sup, seed = args
    cond = COND_BY_NAME[cond_name]
    assert not cond.use_l0_tutor, "F5 must run single-threaded, not via ProcessPool"

    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent
    from cls_option_tutor.tutor.tutor_agent import TutorAgent

    DATA_DIR = _resolve_data_dir(__file__)
    cfg = _make_config(cond, n_sup, grammar_id)
    cfg.seed = seed

    t0 = time.time()
    try:
        env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, grammar_id, seed=seed)
        elapsed = time.time() - t0
        obs_sr, teach_sr, eval_sr = _compute_phase_sr(block)

        return {
            "cond": cond_name, "grammar": grammar_id,
            "nsup": n_sup, "seed": seed,
            "OBS_SR": obs_sr, "TEACH_SR": teach_sr, "EVAL_SR": eval_sr,
            "actions": _tally_actions(block),
            "elapsed": round(elapsed, 1), "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": None, "TEACH_SR": None, "EVAL_SR": None,
            "actions": {}, "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}",
        }


def run_job_f5(cond_name: str, grammar_id: str, n_sup: int, seed: int) -> dict:
    """F5 single-thread runner. L0 tutor holds live LearnerAgent reference."""
    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent
    from cls_option_tutor.tutor.tutor_agent import TutorAgent

    DATA_DIR = _resolve_data_dir(__file__)
    cond = COND_BY_NAME[cond_name]
    cfg = _make_config(cond, n_sup, grammar_id)
    cfg.seed = seed

    t0 = time.time()
    try:
        env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, grammar_id, seed=seed)
        elapsed = time.time() - t0
        obs_sr, teach_sr, eval_sr = _compute_phase_sr(block)

        return {
            "cond": cond_name, "grammar": grammar_id,
            "nsup": n_sup, "seed": seed,
            "OBS_SR": obs_sr, "TEACH_SR": teach_sr, "EVAL_SR": eval_sr,
            "actions": _tally_actions(block),
            "elapsed": round(elapsed, 1), "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": None, "TEACH_SR": None, "EVAL_SR": None,
            "actions": {}, "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation & display
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list) -> dict:
    from collections import defaultdict
    import math

    groups: Dict[str, list] = defaultdict(list)
    for r in results:
        if r["error"] is None and r["EVAL_SR"] is not None:
            groups[r["cond"]].append(r)

    out = {}
    for cond_name, recs in sorted(groups.items()):
        n = len(recs)
        obs   = [r["OBS_SR"]   for r in recs]
        teach = [r["TEACH_SR"] for r in recs]
        eval_ = [r["EVAL_SR"]  for r in recs]
        gap   = [r["EVAL_SR"] - r["OBS_SR"] for r in recs]

        def _mean(xs): return sum(xs) / len(xs)
        def _se(xs):
            m = _mean(xs)
            if len(xs) < 2: return 0.0
            var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
            return math.sqrt(var / len(xs))

        action_totals: Dict[str, float] = {}
        for r in recs:
            for k, v in r.get("actions", {}).items():
                action_totals[k] = action_totals.get(k, 0) + v
        action_means = {k: round(v / n, 1) for k, v in sorted(action_totals.items())}

        out[cond_name] = {
            "n": n,
            "OBS_SR":      round(_mean(obs), 4),
            "TEACH_SR":    round(_mean(teach), 4),
            "EVAL_SR":     round(_mean(eval_), 4),
            "EVAL_SE":     round(_se(eval_), 4),
            "TransferGap": round(_mean(gap), 4),
            "AvgTime_s":   round(_mean([r["elapsed"] for r in recs]), 1),
            "Actions":     action_means,
        }
    return out


def print_table(agg: dict, n_errors: int = 0) -> None:
    header = f"{'Condition':<35} {'N':>4}  {'OBS_SR':>7} {'TEACH_SR':>8} {'EVAL_SR':>7} {'SE':>6} {'Gap':>6}  Actions"
    print(header)
    print("=" * 110)
    for cond_name, row in agg.items():
        act_str = " ".join(f"{k}:{v}%" for k, v in row["Actions"].items())
        print(f"{cond_name:<35} {row['n']:>4}  "
              f"{row['OBS_SR']:>7.3f} {row['TEACH_SR']:>8.3f} "
              f"{row['EVAL_SR']:>7.3f} {row['EVAL_SE']:>6.3f} "
              f"{row['TransferGap']:>6.3f}  {act_str}")
    print()
    if n_errors:
        print(f"[!] {n_errors} jobs failed")


def write_results(agg: dict, n_errors: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Condition                          N  OBS_SR  TEACH_SR  EVAL_SR     SE     Gap  Actions\n")
        f.write("=" * 110 + "\n")
        for cond_name, row in agg.items():
            act_str = " ".join(f"{k}:{v}%" for k, v in row["Actions"].items())
            f.write(f"{cond_name:<35} {row['n']:>4}  "
                    f"{row['OBS_SR']:>7.3f} {row['TEACH_SR']:>8.3f} "
                    f"{row['EVAL_SR']:>7.3f} {row['EVAL_SE']:>6.3f} "
                    f"{row['TransferGap']:>6.3f}  {act_str}\n")
        if n_errors:
            f.write(f"\n[!] {n_errors} jobs failed\n")
        f.write("\n--- JSON ---\n")
        f.write(json.dumps(agg, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp F: RSA L1 Learner evaluation")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default="cls_option_tutor/results/exp_rsa_learner.txt")
    parser.add_argument("--cond", nargs="+", default=None)
    args = parser.parse_args()

    DATA_DIR = os.path.join('BASIC', 'cls_learner', 'data')
    all_grammars = sorted(set(
        os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith('.txt')
    )) if os.path.isdir(DATA_DIR) else ["000001", "000002", "000003", "000004"]

    if args.smoke:
        grammars = all_grammars[:2]
        nsups    = [4, 6]
        seeds    = [42, 123]
    else:
        grammars = all_grammars[:4]
        nsups    = [4, 6]
        seeds    = list(range(42, 72))  # 30 seeds

    if args.cond:
        selected = [c for c in CONDITIONS if c.name in args.cond]
    else:
        selected = CONDITIONS

    parallel_conds = [c for c in selected if not c.use_l0_tutor]
    l0_conds       = [c for c in selected if c.use_l0_tutor]

    tag = "SMOKE" if args.smoke else "FULL"
    total_jobs = (len(parallel_conds) + len(l0_conds)) * len(grammars) * len(nsups) * len(seeds)
    print(f"Exp F [{tag}]: {total_jobs} total jobs | "
          f"{len(parallel_conds)} parallel conds | {len(l0_conds)} single-thread conds")
    print(f"  Grammars: {grammars}")
    print(f"  N_sup: {nsups}, Seeds: {len(seeds)}")
    print()

    all_results = []
    n_errors = 0
    t_start = time.time()

    # ── F0-F4: Parallel ──────────────────────────────────────────
    if parallel_conds:
        parallel_jobs = [
            (cond.name, grammar, nsup, seed)
            for cond in parallel_conds
            for grammar in grammars
            for nsup in nsups
            for seed in seeds
        ]

        n_done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_job, j): j for j in parallel_jobs}
            for fut in futures:
                res = fut.result()
                all_results.append(res)
                if res["error"]:
                    n_errors += 1
                n_done += 1
                if n_done % max(1, len(parallel_jobs) // 20) == 0:
                    pct = 100 * n_done / len(parallel_jobs)
                    elapsed = time.time() - t_start
                    print(f"  Progress: {n_done}/{len(parallel_jobs)} ({pct:.0f}%) | "
                          f"{n_errors} errors | {elapsed:.0f}s elapsed", end="\r")

        print(f"\n  F0-F4 done: {n_done} jobs, {n_errors} errors")

    # ── F5: Single-thread ─────────────────────────────────────────
    if l0_conds:
        l0_jobs = [
            (cond.name, grammar, nsup, seed)
            for cond in l0_conds
            for grammar in grammars
            for nsup in nsups
            for seed in seeds
        ]

        print(f"  F5: running {len(l0_jobs)} jobs single-threaded...")
        for i, (cond_name, grammar, nsup, seed) in enumerate(l0_jobs):
            res = run_job_f5(cond_name, grammar, nsup, seed)
            all_results.append(res)
            if res["error"]:
                n_errors += 1
            if (i + 1) % max(1, len(l0_jobs) // 10) == 0:
                print(f"    F5 {i+1}/{len(l0_jobs)}", end="\r")
        print(f"\n  F5 done: {len(l0_jobs)} jobs")

    # ── Results ───────────────────────────────────────────────────
    agg = aggregate(all_results)
    print_table(agg, n_errors)
    write_results(agg, n_errors, args.output)

    print(f"\nResults saved to: {args.output}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")

    # Signal check
    print("\n--- Signal check ---")
    if "F2_rsa_reveal" in agg and "F0_legacy" in agg:
        f0 = agg["F0_legacy"]["EVAL_SR"]
        f2 = agg["F2_rsa_reveal"]["EVAL_SR"]
        delta = f2 - f0
        print(f"F2 vs F0 EVAL_SR: {f2:.3f} vs {f0:.3f} (Δ={delta:+.3f})")
        if delta > 0:
            print("  ✓ RSA learner shows positive EVAL transfer")
        else:
            print("  ✗ RSA not yet lifting EVAL_SR — check F1 vs F0 for pure action signal")
    if "F1_rsa_noreveal" in agg and "F0_legacy" in agg:
        f0 = agg["F0_legacy"]["EVAL_SR"]
        f1 = agg["F1_rsa_noreveal"]["EVAL_SR"]
        print(f"F1 (no reveal) vs F0: {f1:.3f} vs {f0:.3f} (Δ={f1-f0:+.3f})")
        if f1 > f0:
            print("  ✓ Pure tutor-action RSA produces nonzero eval transfer!")


if __name__ == "__main__":
    main()
