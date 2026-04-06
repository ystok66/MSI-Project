"""
agent.py — CLSAgent: unified entry point for the CLS three-layer system.

Orchestrates:
  reset_episode() → study(support) → predict(query)

Phase 0 (v0): CLSAgent wraps ns_learner methods for bootstrap/EM,
with HPC hooks and BG beam modulation layered on top.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import logsumexp

from cls_learner.interfaces import Example, Episode, TraceSummary, MemBias
from cls_learner.config import CLSConfig
from cls_learner.layer1_cortex.cortex import CortexMemory
from cls_learner.layer2_hpc.hpc import EpisodeHPC
from cls_learner.layer3_control.control import ControlSystem

from ns_learner.ns_learner import GlobalPriors
from ns_learner.ns_concept import ROLES, REPEAT_RANGE, COLOR_VECS, N_COLORS, NIGParams


class CLSAgent:
    """
    CLS (Complementary Learning Systems) Agent.

    Three-layer architecture:
      Layer 1 (Cortex):   Slow, generalizable concept learning
      Layer 2 (HPC):      Fast, episode-internal memory
      Layer 3 (Control):  PFC-BG-Cerebellum search/select/execute

    Lifecycle per episode:
      1. reset_episode()
      2. study(support)   — bootstrap → EM loop with HPC hooks
      3. predict(query)   — inference with HPC bias
    """

    def __init__(self, cfg: Optional[CLSConfig] = None,
                 priors: Optional[GlobalPriors] = None):
        self.cfg = cfg or CLSConfig()

        # Build priors from config
        self.priors = priors or GlobalPriors()
        self.priors.rsa_alpha = self.cfg.rsa_alpha
        self.priors.gauss = self.cfg.gauss
        self.priors.delta = self.cfg.delta

        # If using Lab Gaussian mode, update NIG mu0 to 3D Lab palette mean
        if self.cfg.gauss:
            from ns_learner.ns_colors import lab_palette_mean
            self.priors.nig = NIGParams(d=3, mu0=lab_palette_mean())

        # Three layers
        self.cortex = CortexMemory(self.cfg, self.priors)
        self.hpc = EpisodeHPC(self.cfg) if self.cfg.use_hpc else None
        self.control = ControlSystem(self.cfg, self.priors)

    def reset_episode(self):
        """Reset for a new episode (clear HPC, reset cortex library)."""
        self.cortex.library.clear()
        if self.hpc is not None:
            self.hpc.reset()
        self.control.cerebellum.reset_stats()

    def study(self, support: List[Example], verbose: bool = False):
        """
        Learn from support examples.

        1. Cortex bootstrap (1:1 noun detection)
        2. HPC write after bootstrap
        3. EM loop: E-step → IS-corrected M-step → HPC reconsolidate → replay → decay
        4. (Optional) Cerebellum episode-level weight tuning
        """
        # Ensure vocabulary
        self.cortex.ensure_vocabulary(support)

        # Phase 0: Bootstrap
        self.cortex.bootstrap(support, verbose)

        # HPC: write examples after bootstrap
        hpc_indices: List[int] = []
        if self.hpc is not None:
            for ex in support:
                ts = self.cortex.extract_trace_summary(ex)
                idx = self.hpc.write_example(ex.words, ex.output, ts)
                hpc_indices.append(idx)
            self.hpc.calibrate_gate()

        # EM iterations — interleaved E+M per example (online EM)
        # This matches NSLearner: each example's M-step updates
        # benefit subsequent examples' E-step in the same iteration.
        # HPC is PASSIVE during study — write-only in phase 0, retrieval
        # only at predict time. No beam modulation, replay, or IS
        # correction during EM to avoid corrupting cortex learning.
        for em_iter in range(self.cfg.n_em):
            if verbose:
                print(f"  EM iter {em_iter}:")

            # Decay counts (not first iter)
            if em_iter > 0:
                self.cortex.decay()

            all_traces: List[Optional[list]] = []

            for ex_idx, ex in enumerate(support):
                # E-step: infer traces — no HPC bias, vanilla beam
                target_vecs = self._color_to_vecs(ex.output)

                try:
                    if self.cfg.mode == 'ast':
                        traces = self.control.pfc.infer_top_k_ast(
                            ex.words, target_vecs, self.cortex.library,
                            self.priors,
                        )
                    else:
                        traces = self.control.pfc.infer_top_k_stack(
                            ex.words, target_vecs, self.cortex.library,
                            self.priors,
                        )
                    traces = traces if traces else None
                except Exception:
                    traces = None

                all_traces.append(traces)

                # M-step: immediately update counts from this example
                if traces:
                    self.cortex.m_step_from_traces(
                        [traces],
                        n_support=len(support),
                    )

            if verbose:
                for w, c in sorted(self.cortex.library.items()):
                    mr = c.map_role(self.priors.alpha)
                    mc = c.map_color(self.priors.nig,
                                     self.priors.eps_obj,
                                     self.priors.tau_inc) if mr == 'EMIT' else ''
                    print(f"    {w:>12}: {mr:15s} {mc}")

    def _color_to_vecs(self, colors: List[str]) -> List[np.ndarray]:
        """Convert color names to target vectors.

        When gauss=True, uses CIELAB 3D vectors with optional noise injection.
        Otherwise, uses standard one-hot 6D COLOR_VECS.
        """
        if self.cfg.gauss:
            from ns_learner.ns_colors import lab_vec, add_noise
            vecs = []
            for c in colors:
                v = lab_vec(c)
                if self.cfg.lab_sigma > 0:
                    v = add_noise(v, self.cfg.lab_sigma)
                vecs.append(v)
            return vecs
        else:
            return [COLOR_VECS.get(c, np.zeros(N_COLORS)) for c in colors]

    def predict(self, words: List[str],
                verbose: bool = False) -> List[str]:
        """
        Predict output for a query word sequence.

        Uses HPC bias for memory-augmented inference.
        """
        for w in words:
            self.cortex._ensure_concept(w)

        # HPC: retrieve bias
        mem_bias = self.hpc.get_bias(words) if self.hpc is not None else None

        output, _ = self.control.predict(
            words, self.cortex.library, mem_bias)

        return output

    def learn(self, support_dicts: List[Dict], verbose: bool = False):
        """
        Convenience wrapper: learn from dict-format support.
        Compatible with existing evaluation scripts.
        """
        examples = [Example(words=d['input'], output=d['output'])
                    for d in support_dicts]
        self.reset_episode()
        self.study(examples, verbose)

    def evaluate_episode(self, episode: Episode,
                         verbose: bool = False) -> Dict:
        """
        Full episode evaluation: study support, predict queries.
        Returns {accuracy, correct, total, predictions}.
        """
        self.reset_episode()
        self.study(episode.support, verbose)

        correct = 0
        predictions = []
        for q in episode.query:
            pred = self.predict(q.words, verbose)
            ok = (pred == q.output)
            if ok:
                correct += 1
            predictions.append({
                'input': q.words,
                'predicted': pred,
                'expected': q.output,
                'correct': ok,
            })

        total = len(episode.query)
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'predictions': predictions,
        }
