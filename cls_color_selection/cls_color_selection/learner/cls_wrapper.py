"""
cls_wrapper.py — Wrapper around BASIC/cls_learner for grammar prediction.

Provides:
  - fit_support(examples): learn grammar from support
  - predict_target(words) → Y*: predict output sequence
  - beam_posterior(words) → [(score, trace, Y_k), ...]: top-K candidates
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import sys
import numpy as np

from ..config import LearnerConfig
from ..interfaces import Example


def _ensure_basic_on_path():
    """Add BASIC/ to sys.path so cls_learner and ns_learner are importable."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    basic_dir = os.path.normpath(os.path.join(
        pkg_dir, '..', '..', '..', 'BASIC'))
    if basic_dir not in sys.path:
        sys.path.insert(0, basic_dir)


class CLSSequencePredictor:
    """Grammar learner wrapper around CLSAgent.

    Does NOT modify CLSAgent internals. All grammar knowledge lives
    inside the CLS cortex via study() / predict().
    """

    def __init__(self, cfg: LearnerConfig):
        self.cfg = cfg
        self._agent = None
        self._studied = False

    def fit_support(self, examples: List[Example]) -> None:
        """Learn grammar from support examples.

        Args:
            examples: support Example objects with words + output
        """
        _ensure_basic_on_path()
        from cls_learner.agent import CLSAgent
        from cls_learner.config import CLSConfig
        from cls_learner.interfaces import Example as CLSExample

        cls_cfg = CLSConfig(
            mode=self.cfg.cls_mode,
            use_hpc=self.cfg.use_hpc,
            n_em=self.cfg.n_em,
        )
        self._agent = CLSAgent(cls_cfg)
        self._agent.reset_episode()

        # Convert to CLS Example format
        cls_examples = [
            CLSExample(words=ex.words, output=ex.output)
            for ex in examples[:self.cfg.n_sup]
        ]
        self._agent.study(cls_examples, verbose=False)
        self._studied = True

    def predict_target(self, words: List[str]) -> List[str]:
        """Predict output sequence Y* for given words.

        Returns:
            List of color strings (the learner's current belief about the target).
        """
        if self._agent is None or not self._studied:
            raise RuntimeError("Must call fit_support() first.")
        return self._agent.predict(words, verbose=False)

    def beam_posterior(self, words: List[str]) -> List[Tuple[float, list, List[str]]]:
        """Get top-K beam candidates with scores and rendered outputs.

        Returns:
            List of (score, trace, Y_k) tuples, sorted by score descending.
            - score: log-likelihood under cortex model
            - trace: beam search trace (opaque list of steps)
            - Y_k: rendered output of this trace
        """
        if self._agent is None or not self._studied:
            raise RuntimeError("Must call fit_support() first.")

        _ensure_basic_on_path()

        # Run beam search in unconstrained mode (prediction mode)
        library = self._agent.cortex.library
        priors = self._agent.priors
        control = self._agent.control

        # Ensure vocabulary
        for w in words:
            self._agent.cortex._ensure_concept(w)

        # Get HPC bias if enabled
        mem_bias = None
        if self._agent.hpc is not None:
            mem_bias = self._agent.hpc.get_bias(words)

        results = []
        try:
            if self.cfg.cls_mode == 'ast':
                raw = control.pfc.infer_top_k_ast(
                    words, None, library, priors, mem_bias=mem_bias)
                if raw:
                    from ns_learner.ns_ast import eval_ast
                    from ns_learner.ns_concept import vec_to_color
                    _gauss = getattr(priors, 'gauss', False)
                    if _gauss:
                        from ns_learner.ns_colors import nearest_color as _v2c
                    else:
                        _v2c = vec_to_color

                    for item in raw:
                        score, roots, trace = item
                        # Render output
                        output_vecs = []
                        for root in roots:
                            output_vecs.extend(eval_ast(root))
                        y_k = [_v2c(v) for v in output_vecs] if output_vecs else []
                        results.append((score, trace, y_k))
            else:
                raw = control.pfc.infer_top_k_stack(
                    words, None, library, priors, mem_bias=mem_bias)
                if raw:
                    from ns_learner.ns_inference import execute_trace
                    from ns_learner.ns_concept import vec_to_color
                    _gauss = getattr(priors, 'gauss', False)
                    if _gauss:
                        from ns_learner.ns_colors import nearest_color as _v2c
                    else:
                        _v2c = vec_to_color

                    for item in raw:
                        score, trace = item
                        output_vecs = execute_trace(
                            trace, library, priors.nig,
                            delta=priors.delta, gauss=_gauss)
                        y_k = [_v2c(v) for v in output_vecs] if output_vecs else []
                        results.append((score, trace, y_k))
        except Exception:
            pass

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def get_library(self):
        """Return the cortex concept library for direct manipulation.

        Used by feedback_update.py's differential M-step.
        """
        if self._agent is None:
            return {}
        return self._agent.cortex.library

    def get_agent(self):
        """Return the underlying CLSAgent."""
        return self._agent
