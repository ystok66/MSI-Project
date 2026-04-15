"""
pre_post_eval.py — Pre -> Teach -> Post evaluation protocol (R3).

Implements the canonical pedagogical evaluation:
    1. Pre-eval: frozen learner, no tutor, N eval queries -> baseline metrics
    2. Teaching: tutor + learner, M teaching queries -> learner adapts
    3. Post-eval: same learner (carrying over danger head), no tutor -> post metrics
    4. Compare: delta_SR, delta_AAS, delta_AD

"Frozen learner" means:
    - Semantic scorer params: FROZEN (no updates)
    - Danger head posterior: CARRIES OVER from teaching -> post
    - Episodic memory: BLOCK-SCOPED only (reset per block)
    - Attention: RESET per query
    - Pre-eval uses an independent CLONE (no leakage into teaching)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import copy
import numpy as np

from ..config import FullConfig
from ..env.option_env import OptionEnv
from ..learner.learner_agent import LearnerAgent
from ..tutor.tutor_agent import TutorAgent


@dataclass
class EvalMetrics:
    """Metrics from one evaluation block."""
    solve_rate: float = 0.0
    avg_attempts: float = 0.0    # average attempts per query (AAS)
    avg_damage: float = 0.0      # average damage per query (AD)
    total_damage: int = 0
    n_queries: int = 0
    total_correct: int = 0
    total_rounds: int = 0


@dataclass
class PrePostResult:
    """Result of a pre -> teach -> post evaluation."""
    pre: EvalMetrics = field(default_factory=EvalMetrics)
    post: EvalMetrics = field(default_factory=EvalMetrics)
    teach: EvalMetrics = field(default_factory=EvalMetrics)

    # Deltas
    @property
    def delta_sr(self) -> float:
        return self.post.solve_rate - self.pre.solve_rate

    @property
    def delta_aas(self) -> float:
        return self.post.avg_attempts - self.pre.avg_attempts

    @property
    def delta_ad(self) -> float:
        return self.post.avg_damage - self.pre.avg_damage

    def summary(self) -> str:
        lines = [
            "=== Pre -> Teach -> Post ===",
            f"Pre:   SR={self.pre.solve_rate:.3f}  AAS={self.pre.avg_attempts:.2f}  AD={self.pre.avg_damage:.2f}",
            f"Teach: SR={self.teach.solve_rate:.3f}  AAS={self.teach.avg_attempts:.2f}  AD={self.teach.avg_damage:.2f}",
            f"Post:  SR={self.post.solve_rate:.3f}  AAS={self.post.avg_attempts:.2f}  AD={self.post.avg_damage:.2f}",
            f"Delta: dSR={self.delta_sr:+.3f}  dAAS={self.delta_aas:+.2f}  dAD={self.delta_ad:+.2f}",
        ]
        return "\n".join(lines)


def _block_to_eval_metrics(block) -> EvalMetrics:
    """Extract EvalMetrics from a completed BlockState."""
    metrics = OptionEnv.get_block_metrics(block)
    n_q = metrics["n_queries"]
    return EvalMetrics(
        solve_rate=metrics["solve_rate"],
        avg_attempts=metrics["total_rounds"] / max(n_q, 1),
        avg_damage=metrics["total_damage"] / max(n_q, 1),
        total_damage=metrics["total_damage"],
        n_queries=n_q,
        total_correct=metrics["total_correct"],
        total_rounds=metrics["total_rounds"],
    )


def run_pre_post_eval(
    task_id: str,
    seed: int = 42,
    data_dir: str = "",
    cfg: Optional[FullConfig] = None,
    synthesize: bool = False,
    n_teach_blocks: int = 1,
) -> PrePostResult:
    """Run the canonical pre -> teach -> post evaluation.

    Args:
        task_id: grammar/task file ID
        seed: RNG seed for reproducibility
        data_dir: path to CLS data directory
        cfg: configuration (uses defaults if None)
        synthesize: if True, use grammar-synthesized queries
        n_teach_blocks: number of teaching blocks

    Returns:
        PrePostResult with pre, teach, post metrics and deltas
    """
    cfg = cfg or FullConfig()
    result = PrePostResult()

    # ── 1. Pre-eval: independent clone, no tutor ──
    env_pre = OptionEnv(cfg=cfg, data_dir=data_dir)
    learner_pre = LearnerAgent(cfg=cfg, seed=seed)
    block_pre = learner_pre.run_block(
        env_pre, task_id, seed=seed + 1000, synthesize=synthesize)
    result.pre = _block_to_eval_metrics(block_pre)
    # learner_pre is discarded — no leakage into teaching

    # ── 2. Teaching: tutor + fresh learner ──
    env_teach = OptionEnv(cfg=cfg, data_dir=data_dir)
    learner_teach = LearnerAgent(cfg=cfg, seed=seed)
    tutor = TutorAgent(cfg=cfg)

    teach_metrics_list = []
    for tb in range(n_teach_blocks):
        block_teach = tutor.run_block(
            env_teach, learner_teach, task_id,
            seed=seed + tb, synthesize=synthesize)
        teach_metrics_list.append(_block_to_eval_metrics(block_teach))

    # Average teaching metrics
    if teach_metrics_list:
        result.teach = EvalMetrics(
            solve_rate=np.mean([m.solve_rate for m in teach_metrics_list]),
            avg_attempts=np.mean([m.avg_attempts for m in teach_metrics_list]),
            avg_damage=np.mean([m.avg_damage for m in teach_metrics_list]),
            total_damage=sum(m.total_damage for m in teach_metrics_list),
            n_queries=sum(m.n_queries for m in teach_metrics_list),
            total_correct=sum(m.total_correct for m in teach_metrics_list),
            total_rounds=sum(m.total_rounds for m in teach_metrics_list),
        )

    # ── 3. Post-eval: same learner (danger head carries over), no tutor ──
    # The learner_teach has been trained during teaching.
    # Its danger_head posterior carries over. Episodic memory resets per block.
    # Semantic scorer is frozen (same as init).
    env_post = OptionEnv(cfg=cfg, data_dir=data_dir)
    block_post = learner_teach.run_block(
        env_post, task_id, seed=seed + 2000, synthesize=synthesize)
    result.post = _block_to_eval_metrics(block_post)

    return result


def run_multi_seed_pre_post(
    task_id: str,
    seeds: List[int],
    data_dir: str = "",
    cfg: Optional[FullConfig] = None,
    synthesize: bool = False,
) -> List[PrePostResult]:
    """Run pre->teach->post across multiple seeds."""
    results = []
    for seed in seeds:
        r = run_pre_post_eval(
            task_id=task_id, seed=seed, data_dir=data_dir,
            cfg=cfg, synthesize=synthesize)
        results.append(r)
    return results
