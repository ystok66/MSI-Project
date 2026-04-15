"""
benchmark.py — Within-grammar multi-task evaluation harness.

Runs tutor-vs-baseline comparisons using grammar-synthesized queries.
Each block gets novel query compositions from the same grammar,
testing generalization within a single production system.

Usage:
    python -m cls_option_tutor.eval.benchmark --task 000001 --blocks 5 --seeds 3
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np

from ..config import FullConfig
from ..env.option_env import OptionEnv
from ..learner.learner_agent import LearnerAgent
from ..tutor.tutor_agent import TutorAgent


@dataclass
class RunResult:
    """Result from a single block run."""
    task_id: str
    block_id: int
    seed: int
    condition: str              # "baseline" or "tutor"
    synthesized: bool           # True if queries were grammar-synthesized
    solve_rate: float
    total_correct: int
    total_damage: int
    n_queries: int
    total_refreshes: int
    tutor_interventions: int
    elapsed_ms: float

    @property
    def damage_per_query(self) -> float:
        return self.total_damage / max(self.n_queries, 1)


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""
    runs: List[RunResult] = field(default_factory=list)

    def add(self, r: RunResult):
        self.runs.append(r)

    def filter(self, condition: Optional[str] = None,
               synthesized: Optional[bool] = None) -> List[RunResult]:
        out = self.runs
        if condition is not None:
            out = [r for r in out if r.condition == condition]
        if synthesized is not None:
            out = [r for r in out if r.synthesized == synthesized]
        return out

    def summary_table(self) -> str:
        """Generate human-readable summary table."""
        lines = []

        for synth_label, synth_val in [("file-queries", False),
                                        ("synth-queries", True)]:
            runs_s = [r for r in self.runs if r.synthesized == synth_val]
            if not runs_s:
                continue

            lines.append(f"\n=== {synth_label} ===")
            conditions = sorted(set(r.condition for r in runs_s))
            header = f"{'Condition':<12} | {'SR':>6} {'Dmg':>6} {'Dmg/Q':>6} {'Int':>5} {'N':>4}"
            lines.append(header)
            lines.append("-" * len(header))

            for cond in conditions:
                rs = [r for r in runs_s if r.condition == cond]
                sr = np.mean([r.solve_rate for r in rs])
                dmg = np.mean([r.total_damage for r in rs])
                dpq = np.mean([r.damage_per_query for r in rs])
                intv = np.mean([r.tutor_interventions for r in rs])
                lines.append(
                    f"{cond:<12} | {sr:6.3f} {dmg:6.1f} {dpq:6.2f} {intv:5.1f} {len(rs):4d}")

        return "\n".join(lines)

    def delta_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute tutor - baseline deltas, grouped by synthesized."""
        result = {}
        for synth_label, synth_val in [("file", False), ("synth", True)]:
            base = [r for r in self.runs
                    if r.condition == "baseline" and r.synthesized == synth_val]
            tutor = [r for r in self.runs
                     if r.condition == "tutor" and r.synthesized == synth_val]
            if not base or not tutor:
                continue
            result[synth_label] = {
                "delta_solve_rate": float(
                    np.mean([r.solve_rate for r in tutor])
                    - np.mean([r.solve_rate for r in base])),
                "delta_damage": float(
                    np.mean([r.total_damage for r in tutor])
                    - np.mean([r.total_damage for r in base])),
                "mean_interventions": float(
                    np.mean([r.tutor_interventions for r in tutor])),
            }
        return result


def run_benchmark(
    task_id: str,
    n_blocks: int = 5,
    seeds: Optional[List[int]] = None,
    cfg: Optional[FullConfig] = None,
    data_dir: str = "",
    test_synthesized: bool = True,
    test_file_queries: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run within-grammar multi-task benchmark.

    For each (seed, synthesize_mode):
    - Runs n_blocks blocks with different generated queries
    - Compares baseline (learner-only) vs tutor

    Args:
        task_id: single grammar to evaluate (e.g. "000001")
        n_blocks: number of blocks per condition
        seeds: random seeds (default: [0,1,2])
        cfg: configuration
        data_dir: path to CLS data directory
        test_synthesized: include grammar-synthesized queries
        test_file_queries: include original file queries
        verbose: print progress

    Returns:
        BenchmarkResult with all runs
    """
    cfg = cfg or FullConfig()
    seeds = seeds or list(range(3))
    result = BenchmarkResult()

    modes = []
    if test_file_queries:
        modes.append(("file", False))
    if test_synthesized:
        modes.append(("synth", True))

    for mode_name, synthesize in modes:
        if verbose:
            print(f"\n--- Mode: {mode_name} (grammar {task_id}) ---")

        for seed in seeds:
            for block_id in range(n_blocks):
                # Unique seed per block
                block_seed = seed * 1000 + block_id

                env = OptionEnv(cfg=cfg, data_dir=data_dir)

                # ── Baseline ──
                t0 = time.time()
                try:
                    learner_b = LearnerAgent(cfg=cfg, seed=block_seed)
                    block_b = learner_b.run_block(
                        env, task_id, seed=block_seed,
                        synthesize=synthesize)
                    m_b = OptionEnv.get_block_metrics(block_b)
                    result.add(RunResult(
                        task_id=task_id, block_id=block_id,
                        seed=seed, condition="baseline",
                        synthesized=synthesize,
                        solve_rate=m_b["solve_rate"],
                        total_correct=m_b["total_correct"],
                        total_damage=m_b["total_damage"],
                        n_queries=m_b["n_queries"],
                        total_refreshes=m_b.get("total_refreshes", 0),
                        tutor_interventions=0,
                        elapsed_ms=(time.time() - t0) * 1000,
                    ))
                except Exception as e:
                    if verbose:
                        print(f"  WARN: baseline seed={seed} block={block_id}: {e}")
                    continue

                # ── Tutor ──
                t0 = time.time()
                try:
                    learner_t = LearnerAgent(cfg=cfg, seed=block_seed)
                    tutor = TutorAgent(cfg=cfg)
                    block_t = tutor.run_block(
                        env, learner_t, task_id, seed=block_seed,
                        synthesize=synthesize)
                    m_t = OptionEnv.get_block_metrics(block_t)
                    non_wait = sum(1 for s in block_t.tutor_trace
                                   if s.action != "WAIT")
                    result.add(RunResult(
                        task_id=task_id, block_id=block_id,
                        seed=seed, condition="tutor",
                        synthesized=synthesize,
                        solve_rate=m_t["solve_rate"],
                        total_correct=m_t["total_correct"],
                        total_damage=m_t["total_damage"],
                        n_queries=m_t["n_queries"],
                        total_refreshes=m_t.get("total_refreshes", 0),
                        tutor_interventions=non_wait,
                        elapsed_ms=(time.time() - t0) * 1000,
                    ))
                except Exception as e:
                    if verbose:
                        print(f"  WARN: tutor seed={seed} block={block_id}: {e}")

                if verbose and (block_id + 1) % n_blocks == 0:
                    b_rs = result.filter("baseline", synthesize)
                    t_rs = result.filter("tutor", synthesize)
                    if b_rs and t_rs:
                        b_sr = np.mean([r.solve_rate for r in b_rs])
                        t_sr = np.mean([r.solve_rate for r in t_rs])
                        print(f"  seed={seed}: base_sr={b_sr:.3f} tutor_sr={t_sr:.3f}")

    return result


def run_danger_convergence(
    task_id: str = "000001",
    seed: int = 42,
    data_dir: str = "",
    cfg: Optional[FullConfig] = None,
) -> Dict[str, list]:
    """Track danger head prediction error over observations.

    Returns dict with per-checkpoint MSE and uncertainty.
    """
    from ..learner.danger_head import create_danger_head
    from ..env.danger_model import generate_danger_model, generate_danger_vector

    cfg = cfg or FullConfig()
    rng = np.random.default_rng(seed)
    dm = generate_danger_model(m=cfg.env.danger_dim, rng=rng)
    dh = create_danger_head(m=cfg.env.danger_dim)

    # Test set
    test_rng = np.random.default_rng(seed + 1000)
    test_vs = [generate_danger_vector(cfg.env.danger_dim, test_rng)
               for _ in range(20)]
    true_ds = [dm.expected_damage(v) for v in test_vs]

    mse_hist, unc_hist, n_hist = [], [], []

    # Prior checkpoint
    preds = [dh.predict(v)[0] for v in test_vs]
    uncs = [dh.predict(v)[1] for v in test_vs]
    mse_hist.append(float(np.mean([(p - t)**2 for p, t in zip(preds, true_ds)])))
    unc_hist.append(float(np.mean(uncs)))
    n_hist.append(0)

    # Train and checkpoint
    for i in range(50):
        v = generate_danger_vector(cfg.env.danger_dim, rng)
        d = dm.sample_damage(v, rng)
        dh.update(v, d)
        if (i + 1) % 5 == 0:
            preds = [dh.predict(v)[0] for v in test_vs]
            uncs = [dh.predict(v)[1] for v in test_vs]
            mse_hist.append(float(np.mean(
                [(p - t)**2 for p, t in zip(preds, true_ds)])))
            unc_hist.append(float(np.mean(uncs)))
            n_hist.append(i + 1)

    return {"n_obs": n_hist, "mse": mse_hist, "uncertainty": unc_hist}


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="CLS Option Tutor — Within-Grammar Benchmark")
    parser.add_argument("--task", type=str, default="000001",
                        help="Grammar/task ID to evaluate")
    parser.add_argument("--blocks", type=int, default=5,
                        help="Number of blocks per condition")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--data-dir", type=str,
                        default="BASIC/cls_learner/data",
                        help="Path to CLS data directory")
    args = parser.parse_args()

    print(f"=== Within-Grammar Benchmark ===")
    print(f"Grammar: {args.task}, Blocks: {args.blocks}, Seeds: {args.seeds}")

    result = run_benchmark(
        task_id=args.task,
        n_blocks=args.blocks,
        seeds=list(range(args.seeds)),
        data_dir=args.data_dir,
    )

    print(result.summary_table())
    print()
    deltas = result.delta_summary()
    for k, v in deltas.items():
        print(f"  {k}: Δ_sr={v['delta_solve_rate']:+.3f}, "
              f"Δ_dmg={v['delta_damage']:+.1f}, "
              f"intv={v['mean_interventions']:.1f}")
