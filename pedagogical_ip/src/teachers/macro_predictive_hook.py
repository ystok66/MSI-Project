"""Macro Predictive Hook — Action-prediction-aware lesson reranking.

Shadow-mode hook that scores lessons by their predicted effect on
future action quality. Does NOT change canonical controller by default.

G_pred(ℓ) = (1/|P_ℓ|) Σ [log P(a* | x, b̃^{A,ℓ}) - log P(a* | x, b^A)]

S_macro^shadow(ℓ) = S_macro^base(ℓ) + β_pred · G_pred(ℓ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

from ..agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from ..agents.agent_belief_state import AgentBelief
from .action_predictor import ActionPredictor


@dataclass
class PredictiveScore:
    """Shadow predictive gain for a lesson."""
    lesson_name: str = ""
    base_score: float = 0.0
    predictive_gain: float = 0.0
    shadow_score: float = 0.0
    rank_base: int = 0
    rank_shadow: int = 0
    rank_changed: bool = False


class MacroPredictiveHook:
    """Shadow-mode macro lesson reranking via action prediction gain.

    For each candidate lesson, estimates how much that lesson would
    improve the agent's future action quality (measured by oracle-safe
    action log-likelihood).

    Usage:
        hook = MacroPredictiveHook(predictor)
        scores = hook.score_lessons(lessons, base_scores, agent_belief, probes)
        report = hook.get_report()
    """

    def __init__(self, action_predictor: Optional[ActionPredictor] = None,
                 beta_pred: float = 0.5,
                 params: Optional[AgentPolicyParams] = None):
        self.predictor = action_predictor or ActionPredictor(params=params)
        self.beta_pred = beta_pred
        self._scores: List[PredictiveScore] = []
        self._call_count = 0

    def score_predictive_gain(self, lesson_name: str,
                               agent_belief: AgentBelief,
                               probe_branches: list[list[BranchAttributes]],
                               oracle_safe_actions: list[int],
                               post_lesson_belief: Optional[AgentBelief] = None,
                               ) -> float:
        """Compute predictive gain for a single lesson.

        G_pred = mean improvement in log P(a*|x,b) after lesson.

        Args:
            lesson_name: lesson identifier
            agent_belief: current belief
            probe_branches: list of branch-sets for probe states
            oracle_safe_actions: correct action index for each probe
            post_lesson_belief: hypothetical belief after lesson
        """
        if not probe_branches or not oracle_safe_actions:
            return 0.0

        gains = []
        for branches, a_star in zip(probe_branches, oracle_safe_actions):
            # Current action log-likelihood
            ll_before = self.predictor.score(None, agent_belief, branches, a_star)

            # Post-lesson action log-likelihood
            if post_lesson_belief is not None:
                ll_after = self.predictor.score(None, post_lesson_belief,
                                                branches, a_star)
            else:
                # No post-lesson belief: assume marginal improvement
                ll_after = ll_before + 0.05  # small default gain

            gains.append(ll_after - ll_before)

        return float(np.mean(gains)) if gains else 0.0

    def rerank_lessons_shadow(self, lesson_names: list[str],
                               base_scores: list[float],
                               predictive_gains: list[float],
                               ) -> list[PredictiveScore]:
        """Rerank lessons using shadow predictive scores.

        S_shadow = S_base + β_pred · G_pred
        """
        results = []
        shadow_scores = [
            b + self.beta_pred * g
            for b, g in zip(base_scores, predictive_gains)
        ]

        # Compute ranks
        base_order = np.argsort(base_scores)[::-1]  # descending
        shadow_order = np.argsort(shadow_scores)[::-1]
        base_ranks = np.empty_like(base_order)
        shadow_ranks = np.empty_like(shadow_order)
        base_ranks[base_order] = np.arange(len(base_scores))
        shadow_ranks[shadow_order] = np.arange(len(shadow_scores))

        for i, name in enumerate(lesson_names):
            ps = PredictiveScore(
                lesson_name=name,
                base_score=base_scores[i],
                predictive_gain=predictive_gains[i],
                shadow_score=shadow_scores[i],
                rank_base=int(base_ranks[i]),
                rank_shadow=int(shadow_ranks[i]),
                rank_changed=(int(base_ranks[i]) != int(shadow_ranks[i])),
            )
            results.append(ps)

        self._scores.extend(results)
        self._call_count += 1
        return results

    def get_report(self) -> Dict:
        """Aggregate report over all calls."""
        if not self._scores:
            return {"n_calls": 0}

        n = len(self._scores)
        n_changed = sum(1 for s in self._scores if s.rank_changed)
        gains = [s.predictive_gain for s in self._scores]

        # Top-1 agreement
        top1_agree = 0
        # Group by call batch
        batch_size = n // max(self._call_count, 1)
        for batch_start in range(0, n, max(batch_size, 1)):
            batch = self._scores[batch_start:batch_start + batch_size]
            if batch:
                base_top = min(batch, key=lambda s: s.rank_base)
                shadow_top = min(batch, key=lambda s: s.rank_shadow)
                if base_top.lesson_name == shadow_top.lesson_name:
                    top1_agree += 1

        return {
            "n_calls": self._call_count,
            "n_scores": n,
            "n_rank_changed": n_changed,
            "rank_change_rate": n_changed / max(n, 1),
            "mean_gain": float(np.mean(gains)),
            "top1_agreement": top1_agree / max(self._call_count, 1),
        }

    def reset(self):
        self._scores = []
        self._call_count = 0
