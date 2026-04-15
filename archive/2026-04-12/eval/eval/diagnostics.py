"""
diagnostics.py — Structured diagnostic data classes for root-cause analysis.

Captures per-round learner/tutor internals, per-reveal cortex health,
and block-level aggregate metrics for the root-cause experiment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RoundDiagnostics:
    """Per-round diagnostic snapshot."""
    query_id: int
    round_idx: int

    # Learner internal state
    semantic_scores: Optional[np.ndarray] = None
    danger_preds: Optional[np.ndarray] = None
    danger_uncs: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    U_pick: Optional[np.ndarray] = None
    refresh_gate_triggered: bool = False
    picked_index: Optional[int] = None
    hp: int = 5
    action: str = ""

    # Tutor state
    tutor_P_L: Optional[np.ndarray] = None
    tutor_Q_decomposition: Dict[str, float] = field(default_factory=dict)
    tutor_best_action: str = ""
    tutor_second_best: str = ""
    tutor_margin: float = 0.0
    nll_tom: float = 0.0  # -log P_tutor(actual learner action)

    # Highlight effectiveness (same-query)
    highlight_delta_P_corr: float = 0.0
    highlight_cells: Optional[Tuple[int, ...]] = None


@dataclass
class RevealDiagnostics:
    """Per-reveal diagnostic for cortex health."""
    query_id: int
    round_idx: int
    program: List[str] = field(default_factory=list)
    rendered_output: List[str] = field(default_factory=list)
    damage: int = 0
    reveal_learning_mode: str = "cortex_em"
    probe_score_before: float = 0.0
    probe_score_after: float = 0.0
    probe_delta: float = 0.0  # ProbeDelta


@dataclass
class BlockDiagnostics:
    """Block-level aggregate diagnostics."""
    task_id: str = ""
    seed: int = 0
    condition: str = ""
    n_sup: int = 0
    n_teach: int = 0

    # Phase SRs
    obs_sr: float = 0.0
    teach_sr: float = 0.0
    eval_sr: float = 0.0
    transfer_gap: float = 0.0  # EVAL_SR - OBS_SR

    # Cortex health
    cortex_poison_rate: float = 0.0  # fraction of reveals with ProbeDelta < 0
    mean_probe_delta: float = 0.0

    # ToM quality
    mean_nll_tom: float = 0.0

    # Highlight effectiveness
    highlight_usefulness_teach: float = 0.0  # mean ΔP_corr in teaching
    highlight_usefulness_eval: float = 0.0   # mean ΔP_corr at eval start

    # Action counts
    n_ban: int = 0
    n_highlight: int = 0
    n_risk_hint: int = 0
    n_skip: int = 0
    n_wait: int = 0

    # Detailed logs
    rounds: List[RoundDiagnostics] = field(default_factory=list)
    reveals: List[RevealDiagnostics] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from detailed logs."""
        self.transfer_gap = self.eval_sr - self.obs_sr

        # Cortex poison rate
        if self.reveals:
            poisoned = sum(1 for r in self.reveals if r.probe_delta < 0)
            self.cortex_poison_rate = poisoned / len(self.reveals)
            self.mean_probe_delta = float(np.mean([
                r.probe_delta for r in self.reveals
            ]))

        # Mean NLL_ToM
        tom_values = [r.nll_tom for r in self.rounds if r.nll_tom > 0]
        if tom_values:
            self.mean_nll_tom = float(np.mean(tom_values))

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "condition": self.condition,
            "n_sup": self.n_sup,
            "n_teach": self.n_teach,
            "obs_sr": self.obs_sr,
            "teach_sr": self.teach_sr,
            "eval_sr": self.eval_sr,
            "transfer_gap": self.transfer_gap,
            "cortex_poison_rate": self.cortex_poison_rate,
            "mean_probe_delta": self.mean_probe_delta,
            "mean_nll_tom": self.mean_nll_tom,
            "highlight_usefulness_teach": self.highlight_usefulness_teach,
            "highlight_usefulness_eval": self.highlight_usefulness_eval,
            "n_ban": self.n_ban,
            "n_highlight": self.n_highlight,
            "n_risk_hint": self.n_risk_hint,
            "n_skip": self.n_skip,
            "n_wait": self.n_wait,
            "n_reveals": len(self.reveals),
        }
