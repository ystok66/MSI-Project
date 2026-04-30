"""
Shadow learner model for inverse profile inference.

This module intentionally stays independent from the real learner. It can be
updated only from public events and is safe to use inside inverse fitting and
planning.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ProfileHypothesis:
    """One discrete learner profile hypothesis."""

    name: str
    alpha_sem: float
    alpha_risk: float
    alpha_unc: float
    beta_L: float
    epsilon: float
    c_refresh: float
    risk_scale: float


PROFILE_GRID: List[ProfileHypothesis] = [
    ProfileHypothesis(
        name="sem_strong_risk_moderate",
        alpha_sem=1.0,
        alpha_risk=0.5,
        alpha_unc=0.2,
        beta_L=4.0,
        epsilon=0.05,
        c_refresh=0.4,
        risk_scale=1.0,
    ),
    ProfileHypothesis(
        name="sem_weak_risk_averse",
        alpha_sem=0.6,
        alpha_risk=1.0,
        alpha_unc=0.3,
        beta_L=3.0,
        epsilon=0.08,
        c_refresh=0.4,
        risk_scale=1.0,
    ),
    ProfileHypothesis(
        name="uncertainty_averse",
        alpha_sem=1.0,
        alpha_risk=0.3,
        alpha_unc=0.8,
        beta_L=4.0,
        epsilon=0.05,
        c_refresh=0.4,
        risk_scale=1.0,
    ),
    ProfileHypothesis(
        name="refresh_prone",
        alpha_sem=0.8,
        alpha_risk=0.5,
        alpha_unc=0.3,
        beta_L=3.0,
        epsilon=0.10,
        c_refresh=0.1,
        risk_scale=1.0,
    ),
    ProfileHypothesis(
        name="risk_tolerant",
        alpha_sem=1.0,
        alpha_risk=0.2,
        alpha_unc=0.2,
        beta_L=5.0,
        epsilon=0.03,
        c_refresh=0.5,
        risk_scale=0.7,
    ),
]


class ShadowLearnerModel:
    """Shadow semantic + risk + attention model updated from public events."""

    def __init__(
        self,
        scorer,
        danger_head,
        attention_L: int,
        rho_H: float = 2.0,
    ):
        self._scorer = scorer
        self._danger_head = danger_head
        self._rho_H = rho_H
        self._attention = np.ones(max(attention_L, 1)) / max(attention_L, 1)
        self._attention_L = attention_L

    @property
    def scorer(self):
        return self._scorer

    @property
    def danger_head(self):
        return self._danger_head

    @property
    def attention(self) -> np.ndarray:
        return self._attention

    def reset_attention(self, L: int) -> None:
        self._attention = np.ones(max(L, 1)) / max(L, 1)
        self._attention_L = L

    def update_attention_highlight(self, highlighted_cells: Tuple[int, ...]) -> None:
        if not highlighted_cells:
            return
        attn = self._attention.copy()
        for ell in highlighted_cells:
            if 0 <= ell < len(attn):
                attn[ell] *= np.exp(self._rho_H)
        s = attn.sum()
        if s > 0:
            attn /= s
        self._attention = attn

    def update_from_reveal(
        self,
        wrong_text: List[str],
        revealed_output: List[str],
        danger_vec: Optional[np.ndarray],
        damage: Optional[int],
        update_semantic: bool = True,
        update_risk: bool = True,
    ) -> None:
        if (
            update_semantic
            and self._scorer is not None
            and revealed_output
            and hasattr(self._scorer, "incremental_study")
        ):
            from ..interfaces import Example

            ex = Example(words=list(wrong_text), output=list(revealed_output))
            try:
                self._scorer.incremental_study([ex])
            except Exception:
                pass

        if (
            update_risk
            and self._danger_head is not None
            and danger_vec is not None
            and damage is not None
        ):
            self._danger_head.update(danger_vec, float(damage))

    def update_from_risk_hint(
        self,
        danger_vec: np.ndarray,
        eta: float = 0.8,
    ) -> None:
        if self._danger_head is not None:
            self._danger_head.update_from_hint(danger_vec, eta=eta)

    def _attention_under_spec(
        self,
        target_output: List[str],
        spec: Optional[dict],
        highlighted_cells: Optional[Tuple[int, ...]],
    ) -> np.ndarray:
        spec = spec or {"action": "WAIT"}
        action = spec.get("action", "WAIT")
        hl_cells = spec.get("highlight_cells") or highlighted_cells

        L = len(target_output)
        if L != self._attention_L:
            self.reset_attention(L)
        attn = self._attention.copy()
        if action in ("HIGHLIGHT", "MIX") and hl_cells:
            for ell in hl_cells:
                if 0 <= ell < len(attn):
                    attn[ell] *= np.exp(self._rho_H)
            s = attn.sum()
            if s > 0:
                attn /= s
        return attn

    def _active_positions_for_spec(
        self,
        K_full: int,
        spec: Optional[dict],
        banned_indices: Optional[set],
        option_indices: Optional[List[int]],
    ) -> List[int]:
        spec = spec or {"action": "WAIT"}
        action = spec.get("action", "WAIT")
        ban_idx = spec.get("ban_index")

        if option_indices is None:
            option_indices = list(range(K_full))

        banned_global = set(banned_indices or set())
        if ban_idx is not None and action in ("BAN", "MIX"):
            banned_global.add(ban_idx)

        return [
            pos for pos in range(K_full)
            if option_indices[pos] not in banned_global
        ]

    def score_option_components(
        self,
        target_output: List[str],
        option_texts: List[List[str]],
        option_danger_vecs: List[np.ndarray],
        spec: Optional[dict] = None,
        banned_indices: Optional[set] = None,
        highlighted_cells: Optional[Tuple[int, ...]] = None,
        option_indices: Optional[List[int]] = None,
    ) -> dict:
        K_full = len(option_texts)
        if K_full == 0:
            return {
                "sem": np.array([]),
                "mu_d": np.array([]),
                "u_d": np.array([]),
                "active_positions": [],
                "attention": np.array([]),
            }

        attn = self._attention_under_spec(
            target_output=target_output,
            spec=spec,
            highlighted_cells=highlighted_cells,
        )
        active_positions = self._active_positions_for_spec(
            K_full=K_full,
            spec=spec,
            banned_indices=banned_indices,
            option_indices=option_indices,
        )

        sem = np.zeros(K_full)
        mu_d = np.zeros(K_full)
        u_d = np.zeros(K_full)

        for pos in active_positions:
            text = list(option_texts[pos])
            if self._scorer is not None:
                try:
                    sem[pos] = self._scorer.score_option(
                        list(target_output),
                        text,
                        attention_weights=attn,
                    )
                except TypeError:
                    sem[pos] = self._scorer.score_option(
                        list(target_output),
                        text,
                    )
            if self._danger_head is not None:
                dv = np.asarray(option_danger_vecs[pos])
                mu, u = self._danger_head.predict(dv)
                mu_d[pos] = mu
                u_d[pos] = u

        return {
            "sem": sem,
            "mu_d": mu_d,
            "u_d": u_d,
            "active_positions": active_positions,
            "attention": attn,
        }

    @staticmethod
    def probs_from_components(
        sem: np.ndarray,
        mu_d: np.ndarray,
        u_d: np.ndarray,
        profile: ProfileHypothesis,
        active_positions: List[int],
        K_full: Optional[int] = None,
    ) -> np.ndarray:
        if K_full is None:
            K_full = len(sem)
        if len(active_positions) == 0:
            return np.zeros(K_full)

        sem_act = sem[active_positions]
        mu_d_act = mu_d[active_positions] * profile.risk_scale
        u_d_act = u_d[active_positions]
        U = (
            profile.alpha_sem * sem_act
            - profile.alpha_risk * mu_d_act
            - profile.alpha_unc * u_d_act
        )
        U_shifted = U - U.max()
        exp_u = np.exp(profile.beta_L * U_shifted)
        probs = exp_u / (exp_u.sum() + 1e-30)
        eps = profile.epsilon
        probs = (1 - eps) * probs + eps / max(len(active_positions), 1)
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()

        p = np.zeros(K_full)
        for i, pos in enumerate(active_positions):
            p[pos] = probs[i]
        return p

    def predict_pick_probs(
        self,
        target_output: List[str],
        option_texts: List[List[str]],
        option_danger_vecs: List[np.ndarray],
        profile: ProfileHypothesis,
        spec: dict,
        banned_indices: Optional[set] = None,
        highlighted_cells: Optional[Tuple[int, ...]] = None,
        option_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        components = self.score_option_components(
            target_output=target_output,
            option_texts=option_texts,
            option_danger_vecs=option_danger_vecs,
            spec=spec,
            banned_indices=banned_indices,
            highlighted_cells=highlighted_cells,
            option_indices=option_indices,
        )
        return self.probs_from_components(
            sem=components["sem"],
            mu_d=components["mu_d"],
            u_d=components["u_d"],
            profile=profile,
            active_positions=components["active_positions"],
            K_full=len(option_texts),
        )

    def predict_refresh_prob(
        self,
        option_danger_vecs: List[np.ndarray],
        hp: int,
        profile: ProfileHypothesis,
        refreshes_used: int = 0,
        max_refreshes: int = 2,
    ) -> float:
        if refreshes_used >= max_refreshes:
            return 0.0
        if self._danger_head is None or len(option_danger_vecs) == 0:
            return 0.0

        max_mu = 0.0
        for dv in option_danger_vecs:
            mu, _ = self._danger_head.predict(np.asarray(dv))
            max_mu = max(max_mu, mu * profile.risk_scale)

        threshold = hp * profile.c_refresh
        x = (max_mu - threshold) * 2.0
        p_ref = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(p_ref, 0.01, 0.5))

    def deep_copy(self) -> "ShadowLearnerModel":
        copied = ShadowLearnerModel(
            scorer=copy.deepcopy(self._scorer),
            danger_head=copy.deepcopy(self._danger_head),
            attention_L=self._attention_L,
            rho_H=self._rho_H,
        )
        copied._attention = self._attention.copy()
        return copied
