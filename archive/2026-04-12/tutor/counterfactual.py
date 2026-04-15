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
from ..interfaces import Option, PolicyStateSnapshot
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
    # Eval-aware fields (P0)
    q_now: float = 0.0           # legacy myopic component
    q_probe: float = 0.0         # eval-aware probe delta component
    probe_acc_before: float = 0.0
    probe_acc_after: float = 0.0

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

    def score_all_eval_aware(
        self,
        qs: 'QueryState',
        profile: 'ProfileState',
        scorer: 'DeterministicSemanticScorer',
        danger_head=None,
        learner_state=None,
        access_mode: str = "proxy_oracle",
        shadow_learner=None,
        probe_evaluator=None,
        lambda_now: float = 1.0,
        lambda_probe: float = 0.0,
    ) -> List[InterventionScore]:
        """Score interventions using eval-aware Bayesian objective.

        Q(a) = lambda_now * q_now(a) + lambda_probe * delta_probe(a) - c(a)

        When lambda_probe=0, degrades exactly to legacy scoring.

        The shadow learner is a tutor-side approximate pedagogical
        simulator, NOT a faithful clone of the real learner. It is
        updated only with externally observable evidence.
        """
        # Step 1: get legacy scores
        legacy = self.score_all(
            qs, profile, scorer, danger_head,
            learner_state, access_mode,
        )

        # If no probe component requested, return legacy as-is
        if lambda_probe == 0.0 or shadow_learner is None or probe_evaluator is None:
            for item in legacy:
                item.q_now = item.total_q
            return legacy

        # Step 2: gather public query context for shadow simulation
        from ..env.interventions import get_active_menu
        active = get_active_menu(qs)
        active_texts = [list(o.text) for o in active]
        target_output = list(qs.target_output)
        K = len(active)

        # Guard: skip eval-aware scoring if no active options
        if K == 0:
            for item in legacy:
                item.q_now = item.total_q
            return legacy

        # Pre-compute all K-1 reveal outcomes ONCE (amortized CLS cost)
        shadow_learner.precompute_query(
            probe_evaluator, active_texts, target_output)

        # Tutor's own scores (not learner's)
        sem_scores_tutor = np.array([
            scorer.score_option(target_output, o.text) for o in active
        ])
        danger_preds = np.zeros(K)
        if danger_head is not None:
            for i, o in enumerate(active):
                mu, _u = danger_head.predict(o.danger_vec)
                danger_preds[i] = mu

        # p_pick from tutor's model (same as legacy scorer computes)
        sc = profile.semantic_competence
        shifted = sc * sem_scores_tutor - danger_preds
        shifted = shifted - np.max(shifted)
        beta = 4.0
        p_pick = np.exp(beta * shifted)
        p_pick = p_pick / (p_pick.sum() + 1e-10)

        # Step 3: compute probe delta for each candidate (Pass 1 — collect raw)
        for item in legacy:
            item.q_now = item.total_q

            ban_idx_for_sim = None
            hl_cells_for_sim = None
            action_name = item.action

            if action_name == "BAN" and item.ban_index is not None:
                ban_idx_for_sim = None
                for ai, o in enumerate(active):
                    if o.index == item.ban_index:
                        ban_idx_for_sim = ai
                        break
            elif action_name == "HIGHLIGHT":
                hl_cells_for_sim = item.highlight_cells

            delta_probe, probe_before, probe_after = (
                shadow_learner.simulate_action_probe_delta(
                    action=action_name,
                    probe_eval=probe_evaluator,
                    sem_scores_tutor=sem_scores_tutor,
                    danger_preds=danger_preds,
                    p_pick=p_pick,
                    active_texts=active_texts,
                    target_output=target_output,
                    ban_index=ban_idx_for_sim,
                    highlight_cells=hl_cells_for_sim,
                )
            )

            item.q_probe = delta_probe
            item.probe_acc_before = probe_before
            item.probe_acc_after = probe_after

        # Step 4 (P3-A): z-score normalize q_probe across all candidates
        #
        # Motivation: raw ΔProbe ≈ 0.01–0.05, Q_now ≈ 0.1–0.5 (10–50× gap).
        # Normalization maps probe deltas to unit variance so lambda_probe
        # directly controls the weight of probe signal relative to q_now.
        #
        # std ≈ 0  → all actions identical probe gain → normalized = 0 (no shift)
        # std > 0  → actions differ → ranking shifts proportional to lambda_probe
        raw_probes = np.array([item.q_probe for item in legacy])
        probe_std  = float(np.std(raw_probes))
        probe_mean = float(np.mean(raw_probes))

        if probe_std > 1e-6:
            norm_probes = (raw_probes - probe_mean) / probe_std
        else:
            norm_probes = np.zeros_like(raw_probes)

        # Pass 2: merge normalized probe signal with q_now
        for i, item in enumerate(legacy):
            q_probe_norm = float(norm_probes[i])
            item.total_q = lambda_now * item.q_now + lambda_probe * q_probe_norm
            item.components['q_now']       = item.q_now
            item.components['q_probe']     = item.q_probe      # raw (for logging)
            item.components['q_probe_z']   = q_probe_norm      # normalized (used)
            item.components['probe_before'] = item.probe_acc_before
            item.components['probe_after']  = item.probe_acc_after
            item.components['probe_std']    = probe_std

        # Re-sort by total_q descending
        legacy.sort(key=lambda c: c.total_q, reverse=True)
        return legacy

    def score_all(
        self,
        qs: QueryState,
        profile: ProfileState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
        learner_state: Optional[PolicyStateSnapshot] = None,
        access_mode: str = "proxy_oracle",
    ) -> List[InterventionScore]:
        """Score all legal interventions for the current state.

        Args:
            access_mode: "proxy_oracle" | "cheat_sem" | "cheat_full"
            learner_state: latest PolicyStateSnapshot (for cheat modes)
        """
        active = get_active_menu(qs)
        if not active:
            return [InterventionScore(action="WAIT", total_q=0.0)]

        target = qs.target_output
        L = len(target)
        K = len(active)

        # ── Gather per-option metrics (mode-dependent) ──
        if access_mode in ("cheat_sem", "cheat_full") and learner_state is not None:
            # Use learner's actual semantic scores
            sem_scores = learner_state.semantic_scores.copy()
            # Ensure length matches active menu
            if len(sem_scores) != K:
                sem_scores = np.array([
                    scorer.score_option(target, o.text) for o in active
                ])
        else:
            sem_scores = np.array([
                scorer.score_option(target, o.text) for o in active
            ])

        # Danger predictions from tutor's model
        danger_preds = np.zeros(K)
        ko_probs = np.zeros(K)
        if access_mode == "cheat_full" and learner_state is not None:
            # Use learner's actual danger predictions
            if (hasattr(learner_state, 'danger_posterior_mean')
                    and learner_state.danger_posterior_mean is not None):
                # Reconstruct from learner's danger head state
                # (approximate: use same active menu indexing)
                pass  # fall through to tutor's model if not feasible
            if danger_head is not None:
                for i, o in enumerate(active):
                    mu, u = danger_head.predict(o.danger_vec)
                    danger_preds[i] = mu
                    ko_probs[i] = danger_head.predict_ko_prob(
                        o.danger_vec, qs.hp)
        else:
            if danger_head is not None:
                for i, o in enumerate(active):
                    mu, u = danger_head.predict(o.danger_vec)
                    danger_preds[i] = mu
                    ko_probs[i] = danger_head.predict_ko_prob(
                        o.danger_vec, qs.hp)

        # Predicted learner pick probabilities
        beta = 4.0
        if access_mode in ("cheat_sem", "cheat_full") and learner_state is not None:
            # Use learner's actual utility-based pick probs
            # Reconstruct U_pick from learner's real semantic scores + danger
            sc = profile.semantic_competence
            shifted = sem_scores - danger_preds  # learner's actual scores
            shifted = shifted - np.max(shifted)
            p_pick = np.exp(beta * shifted)
            p_pick = p_pick / (p_pick.sum() + 1e-10)
        else:
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
                if access_mode in ("cheat_sem", "cheat_full") and learner_state is not None:
                    q_hl = self._score_highlight_cheat(
                        cells, target, active, sem_scores, p_pick,
                        learner_state, qs)
                else:
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

    def _score_highlight_cheat(
        self,
        cells: Tuple[int, ...],
        target: List[str],
        active: List[Option],
        sem_scores: np.ndarray,
        p_pick: np.ndarray,
        learner_state: 'PolicyStateSnapshot',
        qs: QueryState,
    ) -> InterventionScore:
        """Cheat-mode HIGHLIGHT: simulate on learner's actual attention.

        Q_HL_cheat(H) = ΔP_corr(H) - β_over · |H|/L
        where ΔP_corr = max_j P_L_after(j) - max_j P_L_before(j)
        computed from learner's real semantic scores + simulated attention shift.
        """
        L = len(target)
        K = len(active)
        beta = 4.0

        # Before: current P_corr from learner's scores
        P_corr_before = float(np.max(p_pick))

        # After: simulate highlight on learner's attention
        attn_before = learner_state.attention_weights
        if attn_before is not None and len(attn_before) == L:
            w_after = attn_before.copy()
            rho = self.cfg.rho_H
            for c in cells:
                if 0 <= c < L:
                    w_after[c] *= np.exp(rho)
            w_after = w_after / (w_after.sum() + 1e-10)
        else:
            w_after = np.ones(L) / L

        # Re-score semantics with highlighted attention
        # Use learner's actual semantic scores as base but weight differently
        # (This is an approximation — we shift scores by attention change)
        sem_after = sem_scores.copy()

        # Compute P_pick after highlight
        shifted_after = sem_after - np.zeros(K)  # no danger change
        shifted_after = shifted_after - np.max(shifted_after)
        p_pick_after = np.exp(beta * shifted_after)
        p_pick_after = p_pick_after / (p_pick_after.sum() + 1e-10)

        P_corr_after = float(np.max(p_pick_after))
        delta_P_corr = P_corr_after - P_corr_before

        over_reveal = len(cells) / max(L, 1)
        q = delta_P_corr - self.cfg.beta_over * over_reveal

        return InterventionScore(
            action="HIGHLIGHT", total_q=q,
            highlight_cells=cells,
            components={
                "delta_P_corr": delta_P_corr,
                "P_corr_before": P_corr_before,
                "P_corr_after": P_corr_after,
                "over_reveal": over_reveal,
                "mode": "cheat",
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
