"""
control.py — ControlSystem: PFC-BG-Cerebellum glue for Layer 3.

Provides unified e_step/predict/execute interface per user spec.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from cls_learner.interfaces import Example, MemBias, TraceSummary
from cls_learner.config import CLSConfig
from cls_learner.layer3_control.pfc import PFCPlanner
from cls_learner.layer3_control.bg import BGSelector
from cls_learner.layer3_control.cerebellum import CerebellumExecutor

from ns_learner.ns_learner import GlobalPriors
from ns_learner.ns_concept import NeuroConcept, COLOR_VECS, N_COLORS


class ControlSystem:
    """
    Layer 3 glue: coordinates PFC planning, BG selection, Cerebellum execution.

    Three key methods:
      - e_step(): run inference on all support examples (for EM learning)
      - predict(): run inference on a query (for evaluation)
      - execute(): run a trace to produce output
    """

    def __init__(self, cfg: CLSConfig, priors: GlobalPriors):
        self.cfg = cfg
        self.priors = priors
        self.pfc = PFCPlanner(cfg)
        self.bg = BGSelector(cfg)
        self.cerebellum = CerebellumExecutor(cfg)

    def e_step(self, examples: List[Example],
               library: Dict[str, NeuroConcept],
               hpc_bias_fn: Optional[Callable] = None,
               ) -> List[Optional[list]]:
        """
        E-step: run beam search on all support examples.

        Args:
            examples: support examples
            library: concept library
            hpc_bias_fn: optional callable(words) -> MemBias

        Returns:
            list of trace lists (one per example), None if failed.
        """
        all_traces: List[Optional[list]] = []

        for ex in examples:
            target_vecs = [COLOR_VECS.get(c, np.zeros(N_COLORS))
                           for c in ex.output]

            # Optional HPC bias during E-step
            mem_bias = hpc_bias_fn(ex.words) if hpc_bias_fn else None

            # BG adjusts beam width
            beam_width = self.bg.adjust_beam(self.cfg.beam_width, mem_bias)

            try:
                if self.cfg.mode == 'ast':
                    traces = self.pfc.infer_top_k_ast(
                        ex.words, target_vecs, library, self.priors,
                        mem_bias=mem_bias, beam_width=beam_width,
                    )
                else:
                    traces = self.pfc.infer_top_k_stack(
                        ex.words, target_vecs, library, self.priors,
                        mem_bias=mem_bias, beam_width=beam_width,
                    )
                all_traces.append(traces if traces else None)
            except Exception:
                all_traces.append(None)

        return all_traces

    def predict(self, words: List[str],
                library: Dict[str, NeuroConcept],
                mem_bias: Optional[MemBias] = None,
                priors: Optional[GlobalPriors] = None,
                ) -> Tuple[List[str], Optional[TraceSummary]]:
        """
        Predict output for a query word sequence.

        Two-stage inference (proposal → target rerank):
          Stage 1: beam search with HPC bias → top-K candidates (proposal)
          Stage 2: rerank by cortex-only score (target = score - S_mem)
        This ensures HPC only affects recall, not the final MAP selection.

        Returns (output_colors, trace_summary).
        """
        from ns_learner.ns_concept import vec_to_color

        p = priors or self.priors
        _gauss = getattr(p, 'gauss', False)

        # Choose vec→color decoder based on emission model
        if _gauss:
            from ns_learner.ns_colors import nearest_color as _v2c
        else:
            _v2c = vec_to_color

        # Fallback: MAP color for each word
        def _fallback():
            return [library[w].map_color(
                p.nig, p.eps_obj, p.tau_inc, delta=p.delta, gauss=_gauss
            ) if w in library else 'BLUE' for w in words]

        try:
            if self.cfg.mode == 'ast':
                results = self.pfc.infer_top_k_ast(
                    words, None,  # None = unconstrained (prediction mode)
                    library, p,
                    mem_bias=mem_bias,
                )

                if not results:
                    return _fallback(), None

                # AST returns (score, roots, trace) tuples
                # Rerank by target score: score - S_mem
                from ns_learner.ns_ast import eval_ast
                if mem_bias is not None:
                    best = max(results,
                               key=lambda r: r[0] - mem_bias.log_q_trace(r[2]))
                else:
                    best = results[0]
                best_score, best_roots, best_trace = best
                output_vecs = []
                for root in best_roots:
                    output_vecs.extend(eval_ast(root))
                if not output_vecs:
                    return _fallback(), None
                output = [_v2c(v) for v in output_vecs]
                return output, None
            else:
                traces = self.pfc.infer_top_k_stack(
                    words, None,  # None = unconstrained (prediction mode)
                    library, p,
                    mem_bias=mem_bias,
                )

                if not traces:
                    return _fallback(), None

                # Stack returns (score, trace) tuples → rerank + execute
                # Rerank by target score: score - S_mem
                from ns_learner.ns_inference import execute_trace
                if mem_bias is not None:
                    best = max(traces,
                               key=lambda t: t[0] - mem_bias.log_q_trace(t[1]))
                else:
                    best = traces[0]
                best_score, best_trace = best
                output_vecs = execute_trace(
                    best_trace, library, p.nig, delta=p.delta, gauss=_gauss)
                if not output_vecs:
                    return _fallback(), None
                output = [_v2c(v) for v in output_vecs]
                return output, None
        except Exception:
            return _fallback(), None

    def execute(self, trace: list) -> List[str]:
        """Execute a trace through the Cerebellum."""
        return self.cerebellum.execute_trace(trace)
