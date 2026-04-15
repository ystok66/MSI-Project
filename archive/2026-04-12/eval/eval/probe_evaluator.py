"""
probe_evaluator.py — Eval surrogate for tutor planning + cortex health.

P1 upgrade: dual ID/OOD probe sets + classification-accuracy main surrogate.

Generates held-out probe queries and measures:
  - eval_score(): combined ID+OOD classification accuracy (main P1 surrogate)
  - eval_score_breakdown(): per-bucket scores for calibration logging
  - score(): legacy semantic margin (kept for backward compat / Exp B baseline)
  - probe_accuracy(): single-bucket accuracy (kept for compat)

Design:
  ID probes:  max_depth=3, max_len=6  (same distribution as training queries)
  OOD probes: max_depth=5, max_len=8  (longer programs, rarer compositions)

  eval_score = (1 - ood_ratio) * acc_id + ood_ratio * acc_ood

This replaces probe_eval.score() in shadow_learner as the main ΔProbe signal.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..interfaces import Example, Option
from ..grammar.query_synthesizer import synthesize_queries
from ..grammar.task_adapter import Grammar, TaskAdapter


class ProbeEvaluator:
    """Fixed dual probe set for measuring cortex health and eval surrogate.

    ID probes:  synthesized at max_depth=3 (same distribution as training).
    OOD probes: synthesized at max_depth=5, max_len=8 (harder, rarer combos).

    Main surrogate (P1): eval_score() — weighted accuracy over both buckets.
    Legacy surrogate (Exp B baseline): score() — semantic margin.
    """

    def __init__(self, grammar: Grammar, n_probes: int = 30,
                 seed: int = 99, existing: Optional[List[Example]] = None,
                 ood_ratio: float = 0.5):
        """
        Args:
            grammar:   parsed grammar for synthesizing probes
            n_probes:  total probe count (split into ID + OOD by ood_ratio)
            seed:      RNG seed (fixed for reproducibility)
            existing:  block queries to exclude from probe set
            ood_ratio: fraction of probes that are OOD (longer programs)
        """
        self.grammar = grammar
        self.ood_ratio = ood_ratio

        n_id  = max(4, int(round(n_probes * (1.0 - ood_ratio))))
        n_ood = n_probes - n_id

        rng_id  = np.random.default_rng(seed)
        rng_ood = np.random.default_rng(seed + 1)

        # ID probes: same depth distribution as training queries
        self._id_probes: List[Example] = synthesize_queries(
            grammar, n=n_id, max_depth=3, max_len=6,
            rng=rng_id, existing=existing,
        )

        # OOD probes: deeper, longer programs (rarer compositions)
        combined_seen = list(existing or []) + self._id_probes
        self._ood_probes: List[Example] = []
        if n_ood > 0:
            self._ood_probes = synthesize_queries(
                grammar, n=n_ood, max_depth=5, max_len=8,
                rng=rng_ood, existing=combined_seen,
            )

        # Fallback: if OOD synthesis returned too few, pad with more ID
        if len(self._ood_probes) < max(1, n_ood // 2):
            extra = synthesize_queries(
                grammar, n=n_ood - len(self._ood_probes),
                max_depth=4, max_len=7,
                rng=np.random.default_rng(seed + 2),
                existing=combined_seen + self._ood_probes,
            )
            self._ood_probes.extend(extra)

        # Combined list (ID first for backward compat access via .probes)
        self._probes = self._id_probes + self._ood_probes

        # Fallback: if synthesis produced nothing, use existing
        if len(self._probes) < 4 and existing:
            self._probes = existing[:min(n_probes, len(existing))]
            self._id_probes = self._probes
            self._ood_probes = []

    # ── Capacity reporting ──────────────────────────────────────────

    @property
    def capacity(self) -> dict:
        """Return probe counts per bucket for diagnostics."""
        return {
            "n_id":  len(self._id_probes),
            "n_ood": len(self._ood_probes),
            "n_total": len(self._probes),
        }

    @property
    def probes(self) -> List[Example]:
        return self._probes

    # ── Main P1 surrogate ───────────────────────────────────────────

    def eval_score(self, scorer, n_distractors: int = 5,
                   alpha: Optional[float] = None) -> float:
        """Main eval surrogate: weighted accuracy over ID + OOD probe sets.

        eval_score = (1 - ood_ratio) * acc_id + ood_ratio * acc_ood

        Replaces score() in shadow_learner for P1 eval-aware objective.
        Returns float in [0, 1].
        """
        if alpha is None:
            alpha = self.ood_ratio

        acc_id  = self._probe_accuracy_on(self._id_probes,  scorer, n_distractors)
        acc_ood = self._probe_accuracy_on(self._ood_probes, scorer, n_distractors)

        return (1.0 - alpha) * acc_id + alpha * acc_ood

    def eval_score_breakdown(self, scorer, n_distractors: int = 5,
                             alpha: Optional[float] = None
                             ) -> Tuple[float, float, float]:
        """Return (combined, acc_id, acc_ood) for calibration logging."""
        if alpha is None:
            alpha = self.ood_ratio

        acc_id  = self._probe_accuracy_on(self._id_probes,  scorer, n_distractors)
        acc_ood = self._probe_accuracy_on(self._ood_probes, scorer, n_distractors)
        combined = (1.0 - alpha) * acc_id + alpha * acc_ood

        return combined, acc_id, acc_ood

    # ── Legacy / Exp B baseline ─────────────────────────────────────

    def score(self, scorer) -> float:
        """Legacy: mean semantic margin on full probe set.

        Kept as Exp B baseline (B0_margin condition).
        In P1 main path, use eval_score() instead.
        """
        if not self._probes:
            return 0.0

        margins = []
        for probe in self._probes:
            target = probe.output
            L = len(target)
            if L == 0:
                continue

            predicted = scorer.predict_output(probe.words)
            if not predicted:
                margins.append(0.0)
                continue

            if len(predicted) != L:
                margins.append(0.0)
                continue

            matches = sum(1 for i in range(L)
                          if predicted[i] == target[i])
            margins.append(matches / L)

        return float(np.mean(margins)) if margins else 0.0

    def probe_accuracy(self, scorer, n_distractors: int = 5) -> float:
        """Single-bucket accuracy (full probe set). Legacy compat."""
        return self._probe_accuracy_on(self._probes, scorer, n_distractors)

    # ── Internal helpers ────────────────────────────────────────────

    def _probe_accuracy_on(self, probes: List[Example],
                           scorer, n_distractors: int = 5) -> float:
        """Classification accuracy on a given list of probe examples.

        For each probe:
          1. correct option = probe.words (renders to probe.output)
          2. Generate n_distractors wrong options from grammar noun bank
          3. Score all via scorer.score_option()
          4. Check if index-0 (correct) wins

        Returns fraction correct in [0, 1].
        """
        if not probes:
            return 0.0

        nouns = list(self.grammar.nouns.keys())
        rng = np.random.default_rng(42)   # fixed seed: deterministic distractor set
        correct_count = 0
        total = 0

        for probe in probes:
            target = probe.output
            L = len(target)
            if L == 0:
                continue

            correct_text = probe.words
            menu_texts = [correct_text]

            n_dist = min(n_distractors, max(1, len(nouns) - 1))
            distractor_set = {tuple(correct_text)}
            attempts = 0
            while len(menu_texts) < n_dist + 1 and attempts < 50:
                attempts += 1
                prog_len = rng.integers(1, min(4, len(nouns) + 1))
                prog = list(rng.choice(nouns, size=prog_len, replace=True))
                key = tuple(prog)
                if key in distractor_set:
                    continue
                rendered = TaskAdapter.render(prog, self.grammar)
                if rendered is not None and rendered != target:
                    menu_texts.append(prog)
                    distractor_set.add(key)

            if len(menu_texts) < 2:
                continue

            scores = np.array([
                scorer.score_option(target, text)
                for text in menu_texts
            ])

            if int(np.argmax(scores)) == 0:
                correct_count += 1
            total += 1

        return correct_count / max(total, 1)

    def measure_delta(self, scorer, update_fn) -> float:
        """Measure ProbeDelta around an update (legacy)."""
        before = self.score(scorer)
        update_fn()
        after = self.score(scorer)
        return after - before
