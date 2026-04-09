"""
experiment_harness.py — Automated experiment runner for convergence ablations.

Runs E1-E5 experiments from the convergence plan and produces
structured results in JSON for analysis.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np

from ..config import FullConfig, TutorConfig
from ..env.option_env import OptionEnv
from ..learner.learner_agent import LearnerAgent
from ..tutor.tutor_agent import TutorAgent
from .pre_post_eval import run_pre_post_eval, PrePostResult


@dataclass
class ExperimentResult:
    """Single experiment condition result."""
    condition: str
    seeds: List[int] = field(default_factory=list)
    solve_rates: List[float] = field(default_factory=list)
    damages: List[float] = field(default_factory=list)
    avg_attempts: List[float] = field(default_factory=list)
    pre_post: Optional[List[dict]] = None

    @property
    def mean_sr(self) -> float:
        return float(np.mean(self.solve_rates)) if self.solve_rates else 0.0

    @property
    def std_sr(self) -> float:
        return float(np.std(self.solve_rates)) if self.solve_rates else 0.0

    @property
    def mean_dmg(self) -> float:
        return float(np.mean(self.damages)) if self.damages else 0.0


def run_e1_repair_validation(
    data_dir: str,
    task_id: str = "000001",
    seeds: List[int] = None,
) -> Dict[str, ExperimentResult]:
    """E1: Repair validation — NoTutor vs Repaired Tutor.

    Conditions:
    - baseline: learner only, tutor = WAIT
    - repaired: R1+R2+R3+R4 tutor
    """
    seeds = seeds or list(range(10))
    results = {}

    # Baseline (no tutor)
    res_base = ExperimentResult(condition="baseline", seeds=seeds)
    for seed in seeds:
        env = OptionEnv(data_dir=data_dir)
        learner = LearnerAgent(seed=seed)
        block = learner.run_block(env, task_id, seed=seed)
        m = OptionEnv.get_block_metrics(block)
        res_base.solve_rates.append(m["solve_rate"])
        res_base.damages.append(m["total_damage"])
        res_base.avg_attempts.append(m["total_rounds"] / max(m["n_queries"], 1))
    results["baseline"] = res_base

    # Repaired tutor (all R1-R4)
    res_tutor = ExperimentResult(condition="repaired_tutor", seeds=seeds)
    for seed in seeds:
        env = OptionEnv(data_dir=data_dir)
        learner = LearnerAgent(seed=seed)
        tutor = TutorAgent()
        block = tutor.run_block(env, learner, task_id, seed=seed)
        m = OptionEnv.get_block_metrics(block)
        res_tutor.solve_rates.append(m["solve_rate"])
        res_tutor.damages.append(m["total_damage"])
        res_tutor.avg_attempts.append(m["total_rounds"] / max(m["n_queries"], 1))
    results["repaired_tutor"] = res_tutor

    # Pre->Teach->Post
    res_ptp = ExperimentResult(condition="pre_teach_post", seeds=seeds)
    res_ptp.pre_post = []
    for seed in seeds:
        r = run_pre_post_eval(task_id=task_id, seed=seed, data_dir=data_dir)
        res_ptp.pre_post.append({
            "pre_sr": r.pre.solve_rate,
            "post_sr": r.post.solve_rate,
            "teach_sr": r.teach.solve_rate,
            "delta_sr": r.delta_sr,
            "delta_ad": r.delta_ad,
        })
        res_ptp.solve_rates.append(r.post.solve_rate)
        res_ptp.damages.append(r.post.avg_damage)
    results["pre_teach_post"] = res_ptp

    return results


def run_e2_action_ablation(
    data_dir: str,
    task_id: str = "000001",
    seeds: List[int] = None,
) -> Dict[str, ExperimentResult]:
    """E2: Tutor action ablation.

    Conditions: Full / BAN-only / HL-only / SKIP-only / No-SKIP / WAIT-only
    Implemented via config overrides:
    - BAN-only: c_hl=100, c_skip=100
    - HL-only: c_ban=100, c_skip=100
    - SKIP-only: c_ban=100, c_hl=100
    - No-SKIP: c_skip=100
    """
    seeds = seeds or list(range(10))
    results = {}

    configs = {
        "full": FullConfig(),
        "ban_only": _override(c_ban=0.2, c_hl=100.0, c_skip=100.0),
        "hl_only": _override(c_ban=100.0, c_hl=0.2, c_skip=100.0),
        "skip_only": _override(c_ban=100.0, c_hl=100.0, c_skip=1.5),
        "no_skip": _override(c_ban=0.2, c_hl=0.2, c_skip=100.0),
        "wait_only": _override(c_ban=100.0, c_hl=100.0, c_skip=100.0),
    }

    for cond_name, cfg in configs.items():
        res = ExperimentResult(condition=cond_name, seeds=seeds)
        for seed in seeds:
            env = OptionEnv(cfg=cfg, data_dir=data_dir)
            learner = LearnerAgent(cfg=cfg, seed=seed)
            tutor = TutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, task_id, seed=seed)
            m = OptionEnv.get_block_metrics(block)
            res.solve_rates.append(m["solve_rate"])
            res.damages.append(m["total_damage"])
            res.avg_attempts.append(m["total_rounds"] / max(m["n_queries"], 1))
        results[cond_name] = res

    return results


def _override(**tutor_kwargs) -> FullConfig:
    """Create a FullConfig with TutorConfig overrides."""
    cfg = FullConfig()
    for k, v in tutor_kwargs.items():
        setattr(cfg.tutor, k, v)
    return cfg


def format_results(results: Dict[str, ExperimentResult]) -> str:
    """Pretty-print experiment results as a table."""
    lines = [
        f"{'Condition':<20} | {'SR':>8} {'sSD':>6} | {'Dmg':>6} | {'AAS':>6} | {'N':>3}",
        "-" * 65,
    ]
    for name, r in results.items():
        n = len(r.seeds)
        lines.append(
            f"{name:<20} | {r.mean_sr:8.3f} {r.std_sr:6.3f} | "
            f"{r.mean_dmg:6.1f} | "
            f"{np.mean(r.avg_attempts) if r.avg_attempts else 0:6.2f} | {n:3d}")
    return "\n".join(lines)


def save_results(results: Dict[str, ExperimentResult],
                 path: str) -> None:
    """Save results to JSON."""
    data = {}
    for name, r in results.items():
        d = {
            "condition": r.condition,
            "seeds": r.seeds,
            "solve_rates": r.solve_rates,
            "damages": [float(x) for x in r.damages],
            "avg_attempts": r.avg_attempts,
            "mean_sr": r.mean_sr,
            "std_sr": r.std_sr,
            "mean_dmg": r.mean_dmg,
        }
        if r.pre_post:
            d["pre_post"] = r.pre_post
        data[name] = d

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
