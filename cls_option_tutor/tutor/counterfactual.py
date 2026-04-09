"""
counterfactual.py — V2 counterfactual intervention scoring.

V2 action space: {WAIT, BAN, RISK_HINT, HIGHLIGHT, SKIP}
  Q(WAIT)          = 0  (baseline)
  Q(BAN, j)        = beta_safe * danger_j * P_L(j) - c_ban
  Q(RISK_HINT, j)  = beta_safe * p_h^tutor(v_j) * P_L(j) - c_hint
  Q(HL, H)         = beta_IG * IG(H) - beta_over * |H|/L - c_hl
  Q(SKIP)          = beta_mastery * P_corr + beta_certainty * (1-H/Hmax) + ...

BAN and HIGHLIGHT have equal cost (c_ban = c_hl = 0.0).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..config import TutorConfig
from ..interfaces import Option
from ..env.state import QueryState, ProfileState
from ..env.interventions import get_active_menu
from ..learner.semantic_scorer import DeterministicSemanticScorer
from ..learner.danger_head import DangerHead


@dataclass
class InterventionScore:
    """Score breakdown for one candidate intervention."""
    action: str
    total_q: float
    ban_index: Optional[int] = None
    hint_index: Optional[int] = None
    highlight_cells: Optional[Tuple[int, ...]] = None
    components: Dict[str, float] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}


class CounterfactualScorer:
    """V2 counterfactual intervention scorer.

    Scores WAIT, RISK_HINT, HIGHLIGHT, SKIP.
    ANTI-ORACLE: never accesses option.is_correct directly.
    """

    def __init__(self, cfg: TutorConfig):
        self.cfg = cfg

    def score_all(
        self,
        qs: QueryState,
        profile: ProfileState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
    ) -> List[InterventionScore]:
        """Score all legal interventions for the current state."""
        active = get_active_menu(qs)
        if not active:
            return [InterventionScore(action="WAIT", total_q=0.0)]

        target = qs.target_output
        L = len(target)
        K = len(active)

        # ── Gather per-option metrics ──
        sem_scores = np.array([
            scorer.score_option(target, o.text) for o in active
        ])

        # Danger predictions from tutor's model
        danger_preds = np.zeros(K)
        ko_probs = np.zeros(K)
        if danger_head is not None:
            for i, o in enumerate(active):
                mu, u = danger_head.predict(o.danger_vec)
                danger_preds[i] = mu
                ko_probs[i] = danger_head.predict_ko_prob(
                    o.danger_vec, qs.hp)

        # Predicted learner pick probabilities
        beta = 4.0
        sc = profile.semantic_competence
        shifted = sc * sem_scores - danger_preds
        shifted = shifted - np.max(shifted)
        p_pick = np.exp(beta * shifted)
        p_pick = p_pick / (p_pick.sum() + 1e-10)

        # Expected damage under current learner policy
        E_damage = float(p_pick @ danger_preds)

        # ── Score each intervention ──
        candidates: List[InterventionScore] = []

        # 1. WAIT
        q_wait = self._score_wait(E_damage, qs)
        candidates.append(q_wait)

        # 2. RISK_HINT candidates [V2]
        for i, opt in enumerate(active):
            if opt.index in qs.risk_hints:
                continue  # already hinted
            q_hint = self._score_risk_hint(
                i, opt, active, danger_preds, ko_probs,
                p_pick, qs)
            candidates.append(q_hint)

        # 3. BAN candidates
        for i, opt in enumerate(active):
            if opt.index in qs.banned_indices:
                continue
            q_ban = self._score_ban(
                i, opt, active, danger_preds, ko_probs,
                p_pick, sem_scores, qs)
            candidates.append(q_ban)

        # 4. HIGHLIGHT candidates
        if L > 0:
            hl_candidates = self._generate_highlight_candidates(
                L, self.cfg.max_highlight_cells)
            for cells in hl_candidates:
                q_hl = self._score_highlight(
                    cells, target, active, scorer, sem_scores,
                    p_pick, profile, qs)
                candidates.append(q_hl)

        # 5. SKIP
        q_skip = self._score_skip(qs, E_damage, sem_scores, p_pick)
        candidates.append(q_skip)

        # Sort by Q-value descending
        candidates.sort(key=lambda c: c.total_q, reverse=True)
        return candidates

    def _score_wait(self, E_damage: float,
                    qs: QueryState) -> InterventionScore:
        """Q(WAIT) = 0 (baseline)."""
        return InterventionScore(
            action="WAIT", total_q=0.0,
            components={"E_damage": E_damage, "note": "baseline"})

    def _score_risk_hint(
        self,
        idx_in_active: int,
        opt: Option,
        active: List[Option],
        danger_preds: np.ndarray,
        ko_probs: np.ndarray,
        p_pick: np.ndarray,
        qs: QueryState,
    ) -> InterventionScore:
        """Q(RISK_HINT, j) = β_safe · p_h^tutor(v_j) · P_L(j) - c_hint.

        Tutor hints risk when:
          - option is predicted risky (high danger)
          - learner is likely to pick it
        """
        # Tutor's hazard estimate for this option
        p_h_tutor = min(1.0, danger_preds[idx_in_active] / 2.0)  # rough hazard proxy
        p_j = float(p_pick[idx_in_active])

        # Value of hint: proportional to how risky AND how likely to be chosen
        hint_value = p_h_tutor * p_j

        q = self.cfg.beta_safe * hint_value - self.cfg.c_hint

        return InterventionScore(
            action="RISK_HINT", total_q=q,
            hint_index=opt.index,
            components={
                "p_h_tutor": p_h_tutor,
                "p_pick_j": p_j,
                "danger_j": float(danger_preds[idx_in_active]),
                "ko_prob_j": float(ko_probs[idx_in_active]),
            })

    def _score_ban(
        self,
        idx_in_active: int,
        opt: Option,
        active: List[Option],
        danger_preds: np.ndarray,
        ko_probs: np.ndarray,
        p_pick: np.ndarray,
        sem_scores: np.ndarray,
        qs: QueryState,
    ) -> InterventionScore:
        """Q(BAN, j) = beta_safe * danger_j * P_L(j) - c_ban.

        BAN removes an option from the menu. Best used on:
          - Dangerous options the learner is likely to pick
          - NOT the correct option (oracle renders best)
        Anti-oracle: uses danger prediction, not is_correct.
        """
        p_j = float(p_pick[idx_in_active])
        danger_j = float(danger_preds[idx_in_active])

        # Value: remove danger weighted by pick probability
        ban_value = danger_j * p_j

        # Penalty: semantic score — if this is the best semantic option,
        # banning it hurts the learner. Higher sem score = less ban value.
        sem_rank = float(sem_scores[idx_in_active])
        sem_penalty = max(0.0, sem_rank / max(abs(np.min(sem_scores)), 1e-5))

        q = (self.cfg.beta_safe * ban_value
             - 0.5 * sem_penalty  # don't ban the likely-correct option
             - self.cfg.c_ban)

        return InterventionScore(
            action="BAN", total_q=q,
            ban_index=opt.index,
            components={
                "danger_j": danger_j,
                "p_pick_j": p_j,
                "ban_value": ban_value,
                "sem_penalty": sem_penalty,
                "ko_prob": float(ko_probs[idx_in_active]),
            })

    def _score_highlight(
        self,
        cells: Tuple[int, ...],
        target: List[str],
        active: List[Option],
        scorer: DeterministicSemanticScorer,
        sem_scores: np.ndarray,
        p_pick: np.ndarray,
        profile: ProfileState,
        qs: QueryState,
    ) -> InterventionScore:
        """Q(HL, H) — discrimination-based highlight scoring.

        For each candidate cell set H, compute how much weighting those cells
        improves the ranking of the correct option vs incorrect ones.

        Discrimination(H) = Σ_{ℓ∈H} disc(ℓ)
        where disc(ℓ) = fraction of incorrect options with ŷ_ℓ ≠ y*_ℓ
                        × 1[correct option has ŷ_ℓ = y*_ℓ]

        This is positive only when the cell separates correct from incorrect.
        """
        L = len(target)
        K = len(active)

        # Render all options via oracle
        renders = []
        for o in active:
            r = scorer._render(o.text)
            renders.append(r if r is not None else [])

        # Find the correct option (best oracle score)
        best_idx = int(np.argmax(sem_scores))

        # Per-cell discrimination
        disc = np.zeros(L)
        for ell in range(L):
            correct_render = renders[best_idx]
            # Does correct option match target at this cell?
            correct_matches = (ell < len(correct_render) and
                               correct_render[ell] == target[ell])
            if not correct_matches:
                continue  # cell doesn't help if correct option is wrong here too

            # Count how many incorrect options mismatch at this cell
            n_incorrect_mismatch = 0
            for j in range(K):
                if j == best_idx:
                    continue
                if ell >= len(renders[j]) or renders[j][ell] != target[ell]:
                    n_incorrect_mismatch += 1
            disc[ell] = n_incorrect_mismatch / max(K - 1, 1)

        # Discrimination for the cell set
        disc_value = sum(disc[c] for c in cells if 0 <= c < L)

        # Also compute IG for logging (legacy)
        w_hl = np.ones(L) / L
        rho = self.cfg.rho_H
        for c in cells:
            if 0 <= c < L:
                w_hl[c] *= np.exp(rho * profile.g_highlight)
        w_hl /= w_hl.sum() + 1e-10
        hl_scores = scorer.score_menu(target, active, weights=w_hl)
        H_before = _entropy_from_scores(sem_scores)
        H_after = _entropy_from_scores(hl_scores)
        IG = max(0.0, H_before - H_after)

        # Over-reveal penalty
        over_reveal = len(cells) / max(L, 1)

        q = (self.cfg.beta_IG * disc_value
             - self.cfg.beta_over * over_reveal
             - self.cfg.c_hl)

        return InterventionScore(
            action="HIGHLIGHT", total_q=q,
            highlight_cells=cells,
            components={
                "disc_value": disc_value,
                "IG": IG,
                "H_before": H_before,
                "H_after": H_after,
                "over_reveal": over_reveal,
                "per_cell_disc": {c: float(disc[c]) for c in cells if 0 <= c < L},
            })

    def _score_skip(
        self,
        qs: QueryState,
        E_damage: float,
        sem_scores: np.ndarray,
        p_pick: np.ndarray,
    ) -> InterventionScore:
        """Q(SKIP) — mastery-preserving skip.

        Q_skip = beta_mastery * P_corr
               + beta_certainty * (1 - H/H_max)
               + beta_time * r
               + beta_hp * h
               - beta_learn * LG
               - c_skip
        """
        K = len(sem_scores)
        if K == 0:
            return InterventionScore(
                action="SKIP", total_q=-self.cfg.c_skip,
                components={"note": "empty_menu"})

        P_corr = float(np.max(p_pick))
        H = _entropy_from_scores(sem_scores)
        max_H = np.log(max(K, 2))
        certainty = 1.0 - H / max(max_H, 1.0)
        r = qs.rounds_used / max(qs.max_rounds, 1)
        h = 1.0 - qs.hp / max(self.cfg.beta_hp * 5.0, 1.0)  # V2: HP_0=5
        LG = H / max(max_H, 1.0)

        q = (self.cfg.beta_mastery * P_corr
             + self.cfg.beta_certainty * certainty
             + self.cfg.beta_time * r
             + self.cfg.beta_hp * h
             - self.cfg.beta_learn * LG
             - self.cfg.c_skip)

        return InterventionScore(
            action="SKIP", total_q=q,
            components={
                "P_corr": P_corr,
                "certainty": certainty,
                "time_pressure": r,
                "hp_pressure": h,
                "learning_gain": LG,
            })

    def _generate_highlight_candidates(
        self, L: int, max_cells: int,
    ) -> List[Tuple[int, ...]]:
        """Generate candidate highlight cell sets."""
        candidates = []
        for i in range(L):
            candidates.append((i,))
        if max_cells >= 2 and L >= 2:
            for i in range(L - 1):
                candidates.append((i, i + 1))
        return candidates


def _entropy_from_scores(scores: np.ndarray) -> float:
    """Softmax entropy from raw scores."""
    if len(scores) == 0:
        return 0.0
    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs = probs / (probs.sum() + 1e-10)
    p_pos = probs[probs > 0]
    return -float(np.sum(p_pos * np.log(p_pos)))
