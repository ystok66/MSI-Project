"""
shadow_learner.py — Approximate belief-conditioned shadow learner.

Tutor maintains a shadow CLS-based pedagogical simulator initialized
from the same grammar/support and updated only with externally
observable evidence. Used only for action ranking on expected
probe/eval gain.

CONSTRAINTS (per design agreement):
  1. Shadow only consumes tutor-observable external information:
     same grammar, support, observed reveals, tutor actions, public query state.
     NEVER reads learner's real semantic scores, CLS params, or hidden states.
  2. Must pass calibration tests (rank correlation, sign agreement)
     between shadow ΔProbe predictions and actual eval/probe changes.
  3. This is an "approximate belief-conditioned shadow learner",
     NOT a "faithful clone" of the real learner.

PERFORMANCE:
  - Pre-compute all K reveal-outcome scorers ONCE per query
  - Cache current scorer and probe scores (rebuild only after observe_reveal)
  - Use fast probe score() (margin) not probe_accuracy (synthetic menus)
  - Each action simulation = O(K) weighted sum over cached values
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np

from ..interfaces import Example
from ..grammar.task_adapter import Grammar, TaskAdapter
from ..learner.cls_adapter import create_scorer
from ..learner.semantic_scorer import DeterministicSemanticScorer
from ..eval.probe_evaluator import ProbeEvaluator


class ShadowLearner:
    """Tutor-side approximate latent-state simulator.

    Initialized from the same grammar/support and updated only
    with externally observable evidence (reveals observed by tutor).

    Used by the eval-aware tutor to estimate:
        ΔProbe(action) = ProbeAcc(after_action) - ProbeAcc(before)

    Performance: pre-computes all K reveal-outcome probe scores once
    per query, then each action simulation is just a weighted sum.
    """

    def __init__(
        self,
        grammar: Grammar,
        support: List[Example],
        n_sup: int = 5,
        n_em: int = 2,
        use_hpc: bool = True,
        tau_sem: float = 1.0,
        use_accuracy: bool = True,
        rollout_horizon: int = 1,
        rollout_gamma: float = 1.0,
    ):
        self.grammar = grammar
        self._base_support = list(support[:min(n_sup, len(support))])
        self._observed_reveals: List[Example] = []
        # Shadow uses reduced EM config for speed:
        #   n_em=1 (single EM pass), use_hpc=False (~20x faster vs full config)
        #   Trade: less accurate posterior, but sufficient for action ranking.
        self._n_em = min(n_em, 1)     # cap at 1 for shadow
        self._use_hpc = False          # no HPC for shadow (speed)
        self._tau_sem = tau_sem
        self._n_sup = n_sup
        # P1: use accuracy-based eval_score() instead of legacy margin score()
        self._use_accuracy = use_accuracy
        # P2: short-horizon rollout
        self._rollout_horizon = rollout_horizon   # H=1 (P1) or H=2 (P2)
        self._rollout_gamma = rollout_gamma       # discount for step t+1
        # O5: DeterministicScorer for p_pick re-computation in BAN/HIGHLIGHT
        # (grammar render is oracle — no CLS needed for action ranking)
        self._det_scorer = DeterministicSemanticScorer(grammar, tau_sem)

        # Build and cache initial scorer
        self._current_scorer = create_scorer(
            grammar, support, use_cls=True,
            n_sup=n_sup, n_em=n_em, use_hpc=use_hpc,
            tau_sem=tau_sem,
        )
        self._scorer_dirty = False

        # Per-query cache: maps option_key -> probe_score_after_reveal
        self._query_reveal_cache: Dict[tuple, float] = {}
        self._cached_probe_before: Optional[float] = None
        self._cached_best_idx: Optional[int] = None

    def observe_reveal(self, example: Example) -> None:
        """Record an externally observed reveal event."""
        self._observed_reveals.append(example)
        self._scorer_dirty = True
        self.invalidate_query_cache()

    def invalidate_query_cache(self) -> None:
        """Clear per-query caches (call at start of new query)."""
        self._query_reveal_cache.clear()
        self._cached_probe_before = None
        self._cached_best_idx = None

    def _ensure_current_scorer(self):
        """Rebuild current scorer if dirty."""
        if self._scorer_dirty:
            all_examples = self._base_support + self._observed_reveals
            self._current_scorer = create_scorer(
                self.grammar, all_examples, use_cls=True,
                n_sup=len(all_examples), n_em=self._n_em,
                use_hpc=self._use_hpc, tau_sem=self._tau_sem,
            )
            self._scorer_dirty = False

    def _probe_score(self, probe_eval: ProbeEvaluator, scorer) -> float:
        """Route to accuracy or margin probe score based on use_accuracy flag."""
        if self._use_accuracy:
            return probe_eval.eval_score(scorer)
        else:
            return probe_eval.score(scorer)

    def current_probe_accuracy(self, probe_eval: ProbeEvaluator) -> float:
        """Probe accuracy of shadow scorer in current state."""
        self._ensure_current_scorer()
        return probe_eval.probe_accuracy(self._current_scorer)

    def precompute_query(
        self,
        probe_eval: ProbeEvaluator,
        active_texts: List[List[str]],
        target_output: List[str],
    ) -> None:
        """Pre-compute probe scores for ALL possible reveal outcomes.

        Called once per query. Builds K-1 updated scorers (one per
        wrong option) and caches their probe scores. This amortizes
        the CLS study() cost across all candidate actions.
        """
        self._ensure_current_scorer()
        self.invalidate_query_cache()

        # Current probe score (accuracy-based in P1, margin in B0 baseline)
        self._cached_probe_before = self._probe_score(probe_eval, self._current_scorer)

        # Identify correct option via DeterministicScorer (O5: oracle, no CLS needed)
        K = len(active_texts)
        oracle_scores = np.array([
            self._det_scorer.score_option(target_output, text)
            for text in active_texts
        ])
        self._cached_best_idx = int(np.argmax(oracle_scores))

        # Pre-compute probe score for each wrong-pick reveal (CLS study needed here)
        # P2 (H=2): additionally simulate one more learning step per branch
        for j in range(K):
            if j == self._cached_best_idx:
                continue  # correct pick = no reveal
            key = tuple(active_texts[j])
            if key in self._query_reveal_cache:
                continue  # already computed (duplicate program)

            rendered = TaskAdapter.render(active_texts[j], self.grammar)
            if rendered is not None:
                reveal_ex = Example(
                    words=list(active_texts[j]),
                    output=rendered,
                )
                updated = self._build_updated_scorer([reveal_ex])
                probe_t1 = self._probe_score(probe_eval, updated)

                if self._rollout_horizon >= 2:
                    # ── H=2: simulate one more learning step ──────────────────
                    # Find hardest probe example for this updated state (proxy
                    # for the next-query reveal that the learner would get wrong)
                    proxy_reveal = self._find_hardest_probe(
                        probe_eval, updated)

                    if proxy_reveal is not None:
                        # Build θ'' = base + reveal_j + proxy_next_reveal
                        updated2 = self._build_updated_scorer(
                            [reveal_ex, proxy_reveal])
                        probe_t2 = self._probe_score(probe_eval, updated2)

                        # Cumulative 2-step value stored as a single effective
                        # probe_after (so _expected_probe_from_cache is unchanged):
                        #   effective = probe_before
                        #               + step1_gain
                        #               + γ * step2_gain
                        step1 = probe_t1 - self._cached_probe_before
                        step2 = probe_t2 - probe_t1
                        effective_after = (self._cached_probe_before
                                          + step1
                                          + self._rollout_gamma * step2)
                        self._query_reveal_cache[key] = effective_after
                    else:
                        self._query_reveal_cache[key] = probe_t1
                else:
                    # H=1: original single-step behavior
                    self._query_reveal_cache[key] = probe_t1
            else:
                self._query_reveal_cache[key] = self._cached_probe_before

    def simulate_action_probe_delta(
        self,
        action: str,
        probe_eval: ProbeEvaluator,
        sem_scores_tutor: Optional[np.ndarray] = None,
        danger_preds: Optional[np.ndarray] = None,
        p_pick: Optional[np.ndarray] = None,
        active_texts: Optional[List[List[str]]] = None,
        target_output: Optional[List[str]] = None,
        ban_index: Optional[int] = None,
        highlight_cells: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[float, float, float]:
        """Simulate one action and estimate ΔProbe.

        Returns (delta_probe, probe_before, probe_after).

        FAST PATH: uses pre-computed reveal outcome cache.
        Each call is just a weighted sum over cached values.
        """
        if self._cached_probe_before is None:
            # Fallback: precompute wasn't called
            if active_texts and target_output:
                self.precompute_query(probe_eval, active_texts, target_output)
            else:
                self._ensure_current_scorer()
                pb = self._probe_score(probe_eval, self._current_scorer)
                return 0.0, pb, pb

        probe_before = self._cached_probe_before

        if action == "SKIP":
            return 0.0, probe_before, probe_before

        if p_pick is None or active_texts is None or target_output is None:
            return 0.0, probe_before, probe_before

        # Compute effective p_pick for this action
        if action == "BAN" and ban_index is not None:
            p_pick_eff, texts_eff = self._effective_ban(
                ban_index, p_pick, active_texts, target_output, danger_preds)
        elif action == "HIGHLIGHT" and highlight_cells is not None:
            p_pick_eff, texts_eff = self._effective_highlight(
                highlight_cells, p_pick, active_texts, target_output, danger_preds)
        else:
            # WAIT, RISK_HINT: use original p_pick
            p_pick_eff = p_pick
            texts_eff = active_texts

        # Compute expected probe using cached reveal outcomes
        probe_after = self._expected_probe_from_cache(
            p_pick_eff, texts_eff, target_output)

        delta = probe_after - probe_before
        return delta, probe_before, probe_after

    def _expected_probe_from_cache(
        self,
        p_pick: np.ndarray,
        active_texts: List[List[str]],
        target_output: List[str],
    ) -> float:
        """Expected probe score from cached reveal outcomes. O(K) weighted sum."""
        K = len(active_texts)
        if K == 0 or self._cached_probe_before is None:
            return self._cached_probe_before if self._cached_probe_before is not None else 0.0

        # O5: use DeterministicScorer (oracle) to re-identify best_idx
        #     Faster than CLS and equally accurate (grammar is the ground truth)
        oracle_scores = np.array([
            self._det_scorer.score_option(target_output, text)
            for text in active_texts
        ])
        best_idx = int(np.argmax(oracle_scores))

        expected = 0.0
        for j in range(K):
            p_j = float(p_pick[j]) if j < len(p_pick) else 0.0
            if p_j < 1e-6:
                continue

            if j == best_idx:
                expected += p_j * self._cached_probe_before
            else:
                key = tuple(active_texts[j])
                reveal_probe = self._query_reveal_cache.get(
                    key, self._cached_probe_before)
                expected += p_j * reveal_probe

        return expected

    def _effective_ban(
        self,
        ban_index: int,
        p_pick: np.ndarray,
        active_texts: List[List[str]],
        target_output: List[str],
        danger_preds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Compute effective p_pick after banning an option.

        O5: uses DeterministicScorer (grammar oracle) to re-score remaining
        options — no CLS study needed for p_pick re-computation.
        """
        K = len(active_texts)
        if ban_index < 0 or ban_index >= K:
            return p_pick, active_texts

        remaining_idx = [i for i in range(K) if i != ban_index]
        remaining_texts = [active_texts[i] for i in remaining_idx]

        # O5: DeterministicScorer instead of self._current_scorer (CLS)
        remaining_scores = np.array([
            self._det_scorer.score_option(target_output, text)
            for text in remaining_texts
        ])
        if danger_preds is not None:
            remaining_danger = np.array([danger_preds[i] for i in remaining_idx])
        else:
            remaining_danger = np.zeros(len(remaining_idx))

        beta = 4.0
        shifted = remaining_scores - remaining_danger
        shifted = shifted - np.max(shifted)
        new_p = np.exp(beta * shifted)
        new_p = new_p / (new_p.sum() + 1e-10)
        return new_p, remaining_texts

    def _effective_highlight(
        self,
        highlight_cells: Tuple[int, ...],
        p_pick: np.ndarray,
        active_texts: List[List[str]],
        target_output: List[str],
        danger_preds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Compute effective p_pick after highlighting cells.

        O5: uses DeterministicScorer with attention_weights for p_pick
        re-computation — no CLS needed.
        """
        L = len(target_output)
        rho_H = 2.0

        w_hl = np.ones(L) / L
        for c in highlight_cells:
            if 0 <= c < L:
                w_hl[c] *= np.exp(rho_H)
        w_hl = w_hl / (w_hl.sum() + 1e-10)

        # O5: DeterministicScorer instead of self._current_scorer (CLS)
        hl_scores = np.array([
            self._det_scorer.score_option(target_output, text,
                                          attention_weights=w_hl)
            for text in active_texts
        ])

        if danger_preds is not None:
            shifted = hl_scores - danger_preds
        else:
            shifted = hl_scores
        shifted = shifted - np.max(shifted)
        beta = 4.0
        new_p = np.exp(beta * shifted)
        new_p = new_p / (new_p.sum() + 1e-10)
        return new_p, active_texts

    def _build_updated_scorer(self, extra_examples: List[Example]):
        """Build scorer from support + observed reveals + extra examples."""
        all_examples = (self._base_support
                        + self._observed_reveals
                        + extra_examples)
        return create_scorer(
            self.grammar, all_examples, use_cls=True,
            n_sup=len(all_examples), n_em=self._n_em,
            use_hpc=self._use_hpc, tau_sem=self._tau_sem,
        )

    def _find_hardest_probe(
        self,
        probe_eval: ProbeEvaluator,
        scorer,
        n_distractors: int = 3,
    ) -> Optional[Example]:
        """Find the probe example with lowest classification accuracy under scorer.

        Used by H=2 rollout as a proxy for the 'next teaching query' that
        the learner would most likely get wrong. Lower accuracy = harder = more
        informative as a next reveal.

        Returns the hardest Example, or None if probe set is empty.
        """
        probes = probe_eval.probes
        if not probes:
            return None

        rng = np.random.default_rng(42)
        nouns = list(probe_eval.grammar.nouns.keys())
        worst_ex: Optional[Example] = None
        worst_score = float('inf')

        for probe in probes:
            target = probe.output
            L = len(target)
            if L == 0:
                continue

            # Classification score (0=wrong, 1=correct) for this probe
            correct_text = probe.words
            menu_texts = [correct_text]

            # Build small distractor menu
            distractor_set = {tuple(correct_text)}
            attempts = 0
            while len(menu_texts) < n_distractors + 1 and attempts < 20:
                attempts += 1
                prog_len = rng.integers(1, min(4, len(nouns) + 1))
                prog = list(rng.choice(nouns, size=prog_len, replace=True))
                key = tuple(prog)
                if key in distractor_set:
                    continue
                from ..grammar.task_adapter import TaskAdapter as TA
                rendered = TA.render(prog, probe_eval.grammar)
                if rendered is not None and rendered != target:
                    menu_texts.append(prog)
                    distractor_set.add(key)

            if len(menu_texts) < 2:
                continue

            scores = np.array([
                scorer.score_option(target, text)
                for text in menu_texts
            ])
            # Individual accuracy: 1 if correct wins else 0
            acc = float(int(np.argmax(scores)) == 0)

            if acc < worst_score:
                worst_score = acc
                worst_ex = probe

        return worst_ex
