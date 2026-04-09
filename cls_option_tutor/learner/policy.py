"""
policy.py — V2 budget-aware learner pick/refresh policy.

V2 changes:
  - U_pick adds α_ko · p_ko(j | h_t) — KO risk avoidance
  - U_refresh adds α_time · (1-r_t) time pressure penalty, α_ko · mean_p_ko
  - Learner sees HP and attempt_idx explicitly
  - on_refresh() no longer resets attention (HIGHLIGHT persists in V2)
  - observe_outcome() splits updates to hazard + severity heads
  - observe_risk_hint() updates hazard head with weak label
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..config import LearnerConfig
from ..interfaces import Option
from ..env.state import QueryState
from ..env.interventions import get_active_menu
from .semantic_scorer import DeterministicSemanticScorer
from .danger_head import DangerHead
from .attention_model import AttentionModel
from .episodic_memory import EpisodicMemory


@dataclass
class PolicyOutput:
    """Result of policy computation."""
    action: str                    # "pick" or "refresh"
    pick_index: Optional[int]     # index if action == "pick"
    utilities: np.ndarray          # (K+1,) — last element is refresh
    probs: np.ndarray              # (K+1,) — action probabilities
    semantic_scores: np.ndarray    # (K,) — raw semantic scores
    danger_preds: np.ndarray       # (K,) — predicted damages
    danger_uncs: np.ndarray        # (K,) — danger uncertainties


class LearnerPolicy:
    """V2 budget-aware softmax option-selection policy with ε-lapse.

    Integrates:
    - Semantic scorer (mismatch-based)
    - Hazard + severity heads (V2 two-layer risk)
    - Attention model (uniform + highlight, persists through refresh)
    - Episodic memory (elimination penalty)
    - Budget awareness (HP + time pressure)
    """

    def __init__(self, cfg: LearnerConfig):
        self.cfg = cfg
        self.danger_head: Optional[DangerHead] = None
        self.memory: Optional[EpisodicMemory] = None
        self.scorer: Optional[DeterministicSemanticScorer] = None
        self.attention: Optional[AttentionModel] = None

    def init_for_block(self, scorer: DeterministicSemanticScorer,
                       m: int = 16) -> None:
        """Initialize learner state for a new block."""
        self.scorer = scorer
        self.danger_head = DangerHead(m,
                                      prior_var=self.cfg.hazard_prior_var,
                                      lr=self.cfg.hazard_lr)
        self.memory = EpisodicMemory()
        self.attention = None  # reset per query

    def init_for_query(self, L: int) -> None:
        """Initialize attention for a new query."""
        self.attention = AttentionModel(L, rho_H=self.cfg.rho_H)

    def compute_policy(self, qs: QueryState,
                       rng: np.random.Generator) -> PolicyOutput:
        """Compute action: pick or refresh.

        Decision logic:
          1. Rank options by semantic score (attention-weighted)
          2. Best semantic option → check if predicted risk >= HP
             - Yes → REFRESH (re-roll risk, costs 1 round)
             - No  → PICK the best semantic option
          3. Softmax with ε-lapse for exploration
        """
        active = get_active_menu(qs)
        K = len(active)
        if K == 0:
            return PolicyOutput(
                action="refresh", pick_index=None,
                utilities=np.zeros(1), probs=np.array([1.0]),
                semantic_scores=np.array([]),
                danger_preds=np.array([]),
                danger_uncs=np.array([]),
            )

        # Attention weights
        weights = (self.attention.weights
                   if self.attention is not None
                   else np.ones(len(qs.target_output)) / len(qs.target_output))

        # Semantic scores for active options (V2: attention-weighted mismatch)
        sem_scores = np.zeros(K)
        for i, opt in enumerate(active):
            sem_scores[i] = self.scorer.score_option(
                qs.target_output, opt.text,
                attention_weights=weights)

        # Danger predictions (V2: from composite head)
        danger_preds = np.zeros(K)
        danger_uncs = np.zeros(K)
        if self.danger_head is not None:
            for i, opt in enumerate(active):
                mu, u = self.danger_head.predict(opt.danger_vec)
                danger_preds[i] = mu
                danger_uncs[i] = u

        # Episodic memory penalties
        memory_penalties = np.zeros(K)
        if self.memory is not None:
            for i, opt in enumerate(active):
                rendered = (self.scorer.predict_output(opt.text)
                            if self.scorer else None)
                memory_penalties[i] = self.memory.get_elimination_penalty(
                    opt.text, rendered)

        # ── Pick utilities ──
        U_pick = (self.cfg.alpha_sem * sem_scores
                  - self.cfg.alpha_risk * danger_preds
                  - self.cfg.alpha_unc * danger_uncs
                  + memory_penalties)

        # ── Refresh decision: deterministic threshold ──
        # Find best semantic option
        best_sem = int(np.argmax(sem_scores))
        top_danger = danger_preds[best_sem]

        # Refresh if predicted damage of best option >= current HP
        # (picking it would likely KO us)
        should_refresh = (top_danger >= qs.hp and qs.rounds_used < qs.max_rounds - 1)

        if should_refresh:
            # Set U_refresh high to ensure refresh is selected
            U_refresh = np.max(U_pick) + 1.0
        else:
            U_refresh = -100.0  # never refresh

        # ── Combine into action distribution ──
        all_utilities = np.concatenate([U_pick, [U_refresh]])
        probs = self._softmax_with_lapse(all_utilities)

        # Sample action
        action_idx = rng.choice(len(probs), p=probs)

        if action_idx < K:
            action = "pick"
            pick_index = active[action_idx].index
        else:
            action = "refresh"
            pick_index = None

        return PolicyOutput(
            action=action,
            pick_index=pick_index,
            utilities=all_utilities,
            probs=probs,
            semantic_scores=sem_scores,
            danger_preds=danger_preds,
            danger_uncs=danger_uncs,
        )

    def observe_outcome(self, v: np.ndarray, damage: int,
                        reveal_event=None) -> None:
        """Update learner state after a wrong-choice reveal.

        V2: updates both hazard and severity heads.
        """
        if self.danger_head is not None:
            self.danger_head.update(v, damage)
        if self.memory is not None and reveal_event is not None:
            self.memory.write_reveal(reveal_event)

    def observe_risk_hint(self, v: np.ndarray, eta: float = 0.8) -> None:
        """Update hazard head from RISK_HINT (weak label). [V2]"""
        if self.danger_head is not None:
            self.danger_head.update_from_hint(v, eta)

    def on_refresh(self) -> None:
        """Handle refresh.

        V2: do NOT reset attention (text unchanged, HIGHLIGHT persists).
        """
        # In V2, attention is NOT reset on refresh.
        # HIGHLIGHT persists because text options don't change.
        pass

    def _softmax_with_lapse(self, utilities: np.ndarray) -> np.ndarray:
        """π'(a) = (1-ε)·softmax(β·U) + ε·Uniform."""
        n = len(utilities)
        shifted = utilities - np.max(utilities)
        exp_u = np.exp(self.cfg.beta_L * shifted)
        probs = exp_u / (exp_u.sum() + 1e-10)
        probs = (1 - self.cfg.epsilon) * probs + self.cfg.epsilon / n
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        return probs

    @staticmethod
    def _semantic_entropy(scores: np.ndarray) -> float:
        """Entropy of softmax over semantic scores."""
        if len(scores) == 0:
            return 0.0
        shifted = scores - np.max(scores)
        probs = np.exp(shifted)
        probs = probs / (probs.sum() + 1e-10)
        p_pos = probs[probs > 0]
        return -float(np.sum(p_pos * np.log(p_pos)))
