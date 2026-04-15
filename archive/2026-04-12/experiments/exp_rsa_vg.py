"""
exp_rsa_vg.py — Exp G: Layer 1 diagnostic ablations.

EXPERIMENT MATRIX:

G1: Gate efficacy (Q: does semantic gate fix F1-type misdirection?)
    G1a_sem_gated      F3 + gate=entropy (use_sem_gate=True)     ← new canonical
    G1b_sem_nogate     F3 + gate=none   (use_sem_gate=False)     ← = old F3 baseline
    G1c_noreveal_gated F1 + gate=entropy                         ← fix F1 negative

G2: Meta-attention ablation (Q: is meta-attention necessary?)
    G2a_no_meta        F3 + rho_attn=0, gamma_attn=0  (no meta)
    G2b_with_meta      F3 + current meta settings     (= G1a)
    G2c_meta_only      omega_hl=0, meta_attn active    (pure meta, no immediate bias)

G3: BAN value (Q: does ban_teaches_risk add anything? does penalty help L0?)
    G3a_ban_teach      F4 + ban_teaches_risk=True       [research flag ON]
    G3b_ban_no_teach   F4 + ban_teaches_risk=False      [canonical: no parametric]
    G3c_ban_l0_penalty F5 + ban_parametric_penalty=-0.3 [L0 tutor, single-thread]

KEY COMPARISONS:
    G1a vs G1b: is gate beneficial in the "easy" case (sem_only+reveal)?
    G1c vs F0 : does gate at minimum recover baseline? (target: G1c ≥ F0=0.690)
    G2a vs G2b: is meta-attention contributing? (target: know definitively)
    G2c alone : how much does meta carry without immediate bias?
    G3a vs G3b: does update_from_ban add eval gain? (hypothesis: no)
    G3c vs F5 : does BAN penalty shift tutor toward HIGHLIGHT? (target: G3c ≥ F2=0.714)

PARALLELISM:
    G1-G3a/G3b: ProcessPoolExecutor (no live learner reference)
    G3c:        Single-thread ONLY (L0 tutor holds live LearnerAgent reference)

USAGE:
    python exp_rsa_vg.py --smoke           # quick sanity
    python exp_rsa_vg.py --workers 12      # full 9 conditions × 240 each
    python exp_rsa_vg.py --cond G1a G1b G1c  # specific subset
    python exp_rsa_vg.py --cond G3c          # single-thread L0 (no --workers effect)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig


# ─────────────────────────────────────────────────────────────────────────────
# Exp G conditions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GCondition:
    name: str
    # Semantic channels
    use_rsa: bool
    omega_hl: float
    omega_ban: float
    reveal_mode: str
    # Gate
    use_sem_gate: bool
    sem_gate_type: str   # "entropy" | "none"
    # Meta-attention
    rho_attn: float
    gamma_attn: float
    # BAN wiring
    ban_teaches_risk: bool
    # L0 tutor
    use_l0_tutor: bool
    ban_parametric_penalty: float = 0.0

G_CONDITIONS = [
    # ── G1: Gate efficacy ──────────────────────────────────────────────────
    GCondition("G1a_sem_gated",
               use_rsa=True, omega_hl=2.0, omega_ban=3.0,
               reveal_mode="cortex_em",
               use_sem_gate=True, sem_gate_type="entropy",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=False,
               use_l0_tutor=False),
    GCondition("G1b_sem_nogate",
               use_rsa=True, omega_hl=2.0, omega_ban=3.0,
               reveal_mode="cortex_em",
               use_sem_gate=False, sem_gate_type="none",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=False,
               use_l0_tutor=False),
    GCondition("G1c_noreveal_gated",
               use_rsa=True, omega_hl=2.0, omega_ban=3.0,
               reveal_mode="off",
               use_sem_gate=True, sem_gate_type="entropy",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=False,
               use_l0_tutor=False),

    # ── G2: Meta-attention ablation ────────────────────────────────────────
    GCondition("G2a_no_meta",
               use_rsa=True, omega_hl=2.0, omega_ban=0.0,
               reveal_mode="cortex_em",
               use_sem_gate=True, sem_gate_type="entropy",
               rho_attn=0.0, gamma_attn=0.0,   # ← meta disabled
               ban_teaches_risk=False,
               use_l0_tutor=False),
    # G2b_with_meta is identical to G1a — reuse those results, no separate run
    GCondition("G2c_meta_only",
               use_rsa=True, omega_hl=0.0, omega_ban=0.0,  # ← no immediate RSA bias
               reveal_mode="cortex_em",
               use_sem_gate=False, sem_gate_type="none",    # gate N/A (no HIGHLIGHT bias)
               rho_attn=0.3, gamma_attn=0.3,   # ← meta active
               ban_teaches_risk=False,
               use_l0_tutor=False),

    # ── G3: BAN value ──────────────────────────────────────────────────────
    GCondition("G3a_ban_teach",
               use_rsa=True, omega_hl=0.0, omega_ban=3.0,
               reveal_mode="cortex_em",
               use_sem_gate=False, sem_gate_type="none",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=True,   # ← research flag ON
               use_l0_tutor=False),
    GCondition("G3b_ban_no_teach",
               use_rsa=True, omega_hl=0.0, omega_ban=3.0,
               reveal_mode="cortex_em",
               use_sem_gate=False, sem_gate_type="none",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=False,  # ← canonical: no parametric
               use_l0_tutor=False),
    GCondition("G3c_ban_l0_penalty",
               use_rsa=True, omega_hl=2.0, omega_ban=3.0,
               reveal_mode="cortex_em",
               use_sem_gate=True, sem_gate_type="entropy",
               rho_attn=0.3, gamma_attn=0.3,
               ban_teaches_risk=False,
               use_l0_tutor=True,      # ← L0 tutor: SINGLE-THREAD ONLY
               ban_parametric_penalty=-0.3),
]

GCOND_BY_NAME = {c.name: c for c in G_CONDITIONS}

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(cond: GCondition, n_sup: int) -> FullConfig:
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
    # RSA
    cfg.rsa.use_rsa = cond.use_rsa
    cfg.rsa.omega_hl = cond.omega_hl
    cfg.rsa.omega_ban = cond.omega_ban
    cfg.rsa.ban_teaches_risk = cond.ban_teaches_risk
    cfg.rsa.use_sem_gate = cond.use_sem_gate
    cfg.rsa.sem_gate_type = cond.sem_gate_type
    cfg.rsa.rho_attn = cond.rho_attn
    cfg.rsa.gamma_attn = cond.gamma_attn
    cfg.rsa.use_l0_tutor = cond.use_l0_tutor
    cfg.rsa.ban_parametric_penalty = cond.ban_parametric_penalty
    return cfg


def _resolve_data_dir(anchor: str) -> str:
    d = os.path.join(os.path.dirname(anchor), '..', 'BASIC', 'cls_learner', 'data')
    if os.path.isdir(d):
        return d
    return os.path.join('BASIC', 'cls_learner', 'data')


# ─────────────────────────────────────────────────────────────────────────────
# Phase SR helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_phase_sr(block) -> tuple:
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries

    def _sr(start, end):
        qs = block.queries[start:end]
        if not qs: return 0.0
        return sum(1 for q in qs if q.success) / len(qs)

    return _sr(0, obs_end), _sr(obs_end, teach_end), _sr(teach_end, len(block.queries))


def _tally_actions(block) -> Dict[str, float]:
    total = len(block.tutor_trace) or 1
    counts: Dict[str, int] = {}
    for ts in block.tutor_trace:
        counts[ts.action] = counts.get(ts.action, 0) + 1
    return {k: round(v * 100 / total, 1) for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Job runners
# ─────────────────────────────────────────────────────────────────────────────

def run_job(args):
    """G1-G3b runner (ProcessPool-safe, no L0 tutor)."""
    cond_name, grammar_id, n_sup, seed = args
    cond = GCOND_BY_NAME[cond_name]
    assert not cond.use_l0_tutor, "L0 conditions must use run_job_l0, not run_job"

    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent
    from cls_option_tutor.tutor.tutor_agent import TutorAgent

    DATA_DIR = _resolve_data_dir(__file__)
    cfg = _make_config(cond, n_sup)
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
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": obs_sr, "TEACH_SR": teach_sr, "EVAL_SR": eval_sr,
            "actions": _tally_actions(block), "elapsed": round(elapsed, 1), "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": None, "TEACH_SR": None, "EVAL_SR": None,
            "actions": {}, "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
        }


def run_job_l0(cond_name: str, grammar_id: str, n_sup: int, seed: int) -> dict:
    """G3c runner — SINGLE-THREAD ONLY. L0 tutor holds live learner reference."""
    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent
    from cls_option_tutor.tutor.tutor_agent import TutorAgent

    DATA_DIR = _resolve_data_dir(__file__)
    cond = GCOND_BY_NAME[cond_name]
    cfg = _make_config(cond, n_sup)
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
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": obs_sr, "TEACH_SR": teach_sr, "EVAL_SR": eval_sr,
            "actions": _tally_actions(block), "elapsed": round(elapsed, 1), "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "cond": cond_name, "grammar": grammar_id, "nsup": n_sup, "seed": seed,
            "OBS_SR": None, "TEACH_SR": None, "EVAL_SR": None,
            "actions": {}, "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list) -> dict:
    from collections import defaultdict
    import math

    groups = defaultdict(list)
    for r in results:
        if r["error"] is None and r["EVAL_SR"] is not None:
            groups[r["cond"]].append(r)

    out = {}
    for cond_name, recs in sorted(groups.items()):
        n = len(recs)
        if n == 0:
            continue
        obs   = [r["OBS_SR"]   for r in recs]
        teach = [r["TEACH_SR"] for r in recs]
        eval_ = [r["EVAL_SR"]  for r in recs]
        gap   = [r["EVAL_SR"] - r["OBS_SR"] for r in recs]

        def _mean(xs): return sum(xs) / len(xs)
        def _se(xs):
            m = _mean(xs)
            if len(xs) < 2: return 0.0
            var = sum((x - m)**2 for x in xs) / (len(xs) - 1)
            return math.sqrt(var / len(xs))

        act: Dict[str, float] = {}
        for r in recs:
            for k, v in r.get("actions", {}).items():
                act[k] = act.get(k, 0) + v
        act_means = {k: round(v / n, 1) for k, v in sorted(act.items())}

        out[cond_name] = {
            "n": n,
            "OBS_SR":      round(_mean(obs), 4),
            "TEACH_SR":    round(_mean(teach), 4),
            "EVAL_SR":     round(_mean(eval_), 4),
            "EVAL_SE":     round(_se(eval_), 4),
            "TransferGap": round(_mean(gap), 4),
            "AvgTime_s":   round(_mean([r["elapsed"] for r in recs]), 1),
            "Actions":     act_means,
        }
    return out


def print_table(agg: dict, n_errors: int = 0) -> None:
    # Reference values from Exp F
    refs = {"F0": 0.690, "F2": 0.714, "F3": 0.721, "F5": 0.649, "F1": 0.628}
    print(f"\n{'Condition':<35} {'N':>4}  {'OBS':>6} {'TEACH':>6} {'EVAL':>6} {'SE':>6} {'Gap':>6}  {'vs F0':>6}  Actions")
    print("=" * 120)
    for cond_name, row in agg.items():
        vs_f0 = row['EVAL_SR'] - refs['F0']
        act_str = " ".join(f"{k}:{v}%" for k, v in row["Actions"].items())
        sign = "+" if vs_f0 >= 0 else ""
        print(f"{cond_name:<35} {row['n']:>4}  "
              f"{row['OBS_SR']:>6.3f} {row['TEACH_SR']:>6.3f} "
              f"{row['EVAL_SR']:>6.3f} {row['EVAL_SE']:>6.3f} "
              f"{row['TransferGap']:>6.3f}  {sign}{vs_f0:.3f}   {act_str}")
    print()
    print(f"Reference:  F0=0.690  F1=0.628  F2=0.714  F3=0.721  F5=0.649")
    if n_errors:
        print(f"\n[!] {n_errors} jobs failed")


def write_results(agg: dict, n_errors: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for cond_name, row in agg.items():
            act_str = " ".join(f"{k}:{v}%" for k, v in row["Actions"].items())
            f.write(f"{cond_name:<35} n={row['n']}  "
                    f"OBS={row['OBS_SR']:.3f} TEACH={row['TEACH_SR']:.3f} "
                    f"EVAL={row['EVAL_SR']:.3f}±{row['EVAL_SE']:.3f} "
                    f"Gap={row['TransferGap']:+.3f}  {act_str}\n")
        if n_errors:
            f.write(f"\n[!] {n_errors} errors\n")
        f.write("\n--- JSON ---\n")
        f.write(json.dumps(agg, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp G: RSA Layer 1 ablations")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=12,
                        help="ProcessPool workers for G1-G3b (G3c always single-thread)")
    parser.add_argument("--output", default="cls_option_tutor/results/exp_rsa_vg.txt")
    parser.add_argument("--cond", nargs="+", default=None,
                        help="Run specific conditions e.g. G1a G2a G3c")
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
        selected = [c for c in G_CONDITIONS
                    if any(c.name.startswith(a) or c.name == a for a in args.cond)]
    else:
        selected = G_CONDITIONS

    parallel_conds = [c for c in selected if not c.use_l0_tutor]
    l0_conds       = [c for c in selected if c.use_l0_tutor]

    tag = "SMOKE" if args.smoke else "FULL"
    n_parallel = len(parallel_conds) * len(grammars) * len(nsups) * len(seeds)
    n_l0       = len(l0_conds) * len(grammars) * len(nsups) * len(seeds)
    print(f"Exp G [{tag}]: {n_parallel + n_l0} total jobs")
    print(f"  Parallel (G1-G3b): {n_parallel} jobs, {args.workers} workers")
    print(f"  Single-thread (G3c): {n_l0} jobs")
    print(f"  Grammars: {grammars}, N_sup: {nsups}, Seeds: {len(seeds)}")
    print()

    all_results = []
    n_errors = 0
    t_start = time.time()

    # ── G1-G3b: Parallel ─────────────────────────────────────────────────────
    if parallel_conds:
        jobs = [
            (cond.name, g, ns, s)
            for cond in parallel_conds
            for g in grammars
            for ns in nsups
            for s in seeds
        ]
        n_done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_job, j): j for j in jobs}
            for fut in futures:
                res = fut.result()
                all_results.append(res)
                if res["error"]: n_errors += 1
                n_done += 1
                if n_done % max(1, len(jobs) // 20) == 0:
                    pct = 100 * n_done / len(jobs)
                    print(f"  [{n_done}/{len(jobs)} {pct:.0f}%] {n_errors} errors "
                          f"| {time.time()-t_start:.0f}s", end="\r")
        print(f"\n  G1-G3b done: {n_done} jobs, {n_errors} errors")

    # ── G3c: Single-thread (L0 tutor) ────────────────────────────────────────
    if l0_conds:
        l0_jobs = [
            (cond.name, g, ns, s)
            for cond in l0_conds
            for g in grammars
            for ns in nsups
            for s in seeds
        ]
        print(f"  G3c: {len(l0_jobs)} jobs single-threaded (L0 tutor)...")
        for i, (cond_name, g, ns, s) in enumerate(l0_jobs):
            res = run_job_l0(cond_name, g, ns, s)
            all_results.append(res)
            if res["error"]: n_errors += 1
            if (i + 1) % max(1, len(l0_jobs) // 10) == 0:
                print(f"    G3c {i+1}/{len(l0_jobs)}", end="\r")
        print(f"\n  G3c done: {len(l0_jobs)} jobs")

    # ── Results ──────────────────────────────────────────────────────────────
    agg = aggregate(all_results)
    print_table(agg, n_errors)
    write_results(agg, n_errors, args.output)

    print(f"\nSaved to: {args.output}")
    print(f"Total elapsed: {time.time() - t_start:.0f}s")

    # Decision hints
    print("\n── Layer 1 Decision Points ──")
    if "G1a_sem_gated" in agg and "G1b_sem_nogate" in agg:
        g1a = agg["G1a_sem_gated"]["EVAL_SR"]
        g1b = agg["G1b_sem_nogate"]["EVAL_SR"]
        print(f"G1a(gated) vs G1b(nogate): {g1a:.3f} vs {g1b:.3f} "
              f"(Δ={g1a-g1b:+.3f}) → gate {'HELPS' if g1a>=g1b else 'HURTS'}")
    if "G1c_noreveal_gated" in agg:
        g1c = agg["G1c_noreveal_gated"]["EVAL_SR"]
        f0  = 0.690
        print(f"G1c(noreveal+gate) vs F0: {g1c:.3f} vs {f0:.3f} "
              f"→ gate {'RECOVERS baseline ✓' if g1c>=f0 else 'does NOT recover baseline ✗'}")
    if "G2a_no_meta" in agg and "G1a_sem_gated" in agg:
        g2a = agg["G2a_no_meta"]["EVAL_SR"]
        g2b = agg["G1a_sem_gated"]["EVAL_SR"]  # G2b = G1a
        print(f"G2a(no_meta) vs G2b(with_meta): {g2a:.3f} vs {g2b:.3f} "
              f"(Δ={g2b-g2a:+.3f}) → meta_attn {'NECESSARY' if g2b-g2a>0.01 else 'NOT necessary'}")
    if "G3a_ban_teach" in agg and "G3b_ban_no_teach" in agg:
        g3a = agg["G3a_ban_teach"]["EVAL_SR"]
        g3b = agg["G3b_ban_no_teach"]["EVAL_SR"]
        print(f"G3a(ban_teach) vs G3b(ban_no_teach): {g3a:.3f} vs {g3b:.3f} "
              f"→ update_from_ban {'HAS VALUE' if g3a-g3b>0.01 else 'USELESS → delete canonical'}")
    if "G3c_ban_l0_penalty" in agg:
        g3c = agg["G3c_ban_l0_penalty"]["EVAL_SR"]
        f5  = 0.649; f2 = 0.714
        print(f"G3c(L0+penalty) vs F5/F2: {g3c:.3f} vs F5={f5} / F2={f2:.3f} "
              f"→ penalty {'SUFFICIENT' if g3c>=f2-0.01 else 'INSUFFICIENT → redesign L0 objective'}")


if __name__ == "__main__":
    main()
