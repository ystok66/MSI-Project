"""
cerebellum.py — Cerebellum Executor: trace execution + error tracking.

Wraps ns_primitives execution with prediction error statistics.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from cls_learner.config import CLSConfig
from ns_learner.ns_concept import COLOR_VECS, vec_to_color, N_COLORS


class CerebellumExecutor:
    """
    Cerebellum: executes beam traces and tracks prediction errors.

    Provides:
      - execute(): run a trace through the stack machine
      - prediction_error(): compare predicted vs gold output
      - error_stats: accumulated error statistics for analysis
    """

    def __init__(self, cfg: CLSConfig):
        self.cfg = cfg
        self.error_stats = defaultdict(int)  # {error_type: count}
        self.total_predictions = 0
        self.total_correct = 0

    def execute_trace(self, trace: list) -> List[str]:
        """
        Execute a beam trace through the stack machine.
        Returns list of color names.
        """
        from ns_learner.ns_primitives import (
            StackState, StackItem, PrimEmit, PrimRepeat,
            PrimSwapInfix, PrimConcatInfix, PrimOverInfix,
        )

        prims = {
            'EMIT': PrimEmit(),
            'REPEAT': PrimRepeat(),
            'SWAP_INFIX': PrimSwapInfix(),
            'CONCAT_INFIX': PrimConcatInfix(),
            'OVER_INFIX': PrimOverInfix(),
        }

        state = StackState()
        for step in trace:
            prim = prims.get(step.role)
            if prim is None:
                continue

            if step.role == 'EMIT':
                result = prim.execute(state, step.emit_vec)
            elif step.role == 'REPEAT':
                result = prim.execute(state, step.repeat_k or 1,
                                      arity=step.arity if hasattr(step, 'arity') else 1)
            elif step.role in ('SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX'):
                result = prim.execute(state, step.b_vec,
                                      arity=step.arity if hasattr(step, 'arity') else 1)
            else:
                continue

            if result is not None:
                state = result

        # Flatten stack to color names
        vecs = state.flatten()
        return [vec_to_color(v) for v in vecs]

    def prediction_error(self, pred: List[str], gold: List[str]) -> Dict[str, float]:
        """
        Compute prediction error metrics.
        Returns dict with error type and magnitude.
        """
        self.total_predictions += 1
        correct = (pred == gold)
        if correct:
            self.total_correct += 1
            return {'type': 'correct', 'error': 0.0}

        # Classify error
        if len(pred) != len(gold):
            err_type = 'length_mismatch'
        elif sorted(pred) == sorted(gold):
            err_type = 'order_error'
        else:
            # Count substitutions
            n_sub = sum(1 for p, g in zip(pred, gold) if p != g)
            err_type = f'substitution_{n_sub}'

        self.error_stats[err_type] += 1
        error_magnitude = sum(1 for p, g in zip(pred, gold) if p != g) + abs(len(pred) - len(gold))

        return {'type': err_type, 'error': float(error_magnitude)}

    def reset_stats(self):
        """Reset accumulated error statistics."""
        self.error_stats.clear()
        self.total_predictions = 0
        self.total_correct = 0

    @property
    def accuracy(self) -> float:
        """Current overall accuracy."""
        return self.total_correct / max(1, self.total_predictions)
