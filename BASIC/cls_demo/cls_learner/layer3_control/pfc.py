"""
pfc.py — PFC Planner: beam search planning (Stack & AST paths).

Calls existing ns_inference.infer_top_k / ns_ast.infer_top_k_ast
with optional mem_bias injection from HPC.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from cls_learner.interfaces import MemBias
from cls_learner.config import CLSConfig
from ns_learner.ns_learner import GlobalPriors
from ns_learner.ns_concept import NeuroConcept, COLOR_VECS, N_COLORS


class PFCPlanner:
    """
    Prefrontal Cortex planner: runs beam search inference.

    Wraps ns_inference and ns_ast, injecting mem_bias from HPC.
    """

    def __init__(self, cfg: CLSConfig):
        self.cfg = cfg

    def infer_top_k_stack(self, words: List[str],
                          target_vecs: List[np.ndarray],
                          library: Dict[str, NeuroConcept],
                          priors: GlobalPriors,
                          mem_bias: Optional[MemBias] = None,
                          k: Optional[int] = None,
                          beam_width: Optional[int] = None) -> list:
        """
        Run stack-based beam search.
        Returns list of (score, trace) tuples.
        """
        from ns_learner.ns_inference import infer_top_k

        return infer_top_k(
            words, target_vecs, library, priors,
            k=k or self.cfg.beam_k,
            beam_width=beam_width or self.cfg.beam_width,
            mem_bias=mem_bias,
        )

    def infer_top_k_ast(self, words: List[str],
                        target_vecs: List[np.ndarray],
                        library: Dict[str, NeuroConcept],
                        priors: GlobalPriors,
                        mem_bias: Optional[MemBias] = None,
                        k: Optional[int] = None,
                        beam_width: Optional[int] = None) -> list:
        """
        Run AST-based beam search with deferred hole-filling.
        Returns list of (score, ast_node) tuples.
        """
        from ns_learner.ns_ast import infer_top_k_ast

        return infer_top_k_ast(
            words, target_vecs, library, priors,
            k=k or self.cfg.beam_k,
            beam_width=beam_width or self.cfg.beam_width,
            mem_bias=mem_bias,
        )
