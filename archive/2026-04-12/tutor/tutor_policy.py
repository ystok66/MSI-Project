"""
tutor_policy.py — Tutor action selection from scored interventions.

Implements §11.4:
    a*_T = argmax_a Q(a | s, profile)
    with WAIT as the default (Q(WAIT) = 0 baseline).

Only acts during teaching phase (observation phase → always WAIT).

P0 eval-aware mode:
    Q(a) = λ_now · q_now(a) + λ_probe · ΔProbe(a)
    where ΔProbe is estimated via shadow learner simulation.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..config import TutorConfig
from ..env.state import BlockState, QueryState, ProfileState
from ..interfaces import TutorStep, PolicyStateSnapshot
from .counterfactual import CounterfactualScorer, InterventionScore
from ..learner.semantic_scorer import DeterministicSemanticScorer
from ..learner.danger_head import DangerHead


class TutorPolicy:
    """Tutor action selection policy.

    Lifecycle per block:
        1. Observation phase: WAIT on all queries, collect learner trace
        2. Profile inference (done externally)
        3. Teaching phase: score interventions → pick best action
    """

    def __init__(self, cfg: TutorConfig):
        self.cfg = cfg
        self.cf_scorer = CounterfactualScorer(cfg)

    def select_action(
        self,
        block: BlockState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
        learner_state: Optional[PolicyStateSnapshot] = None,
        access_mode: str = "proxy_oracle",
        shadow_learner=None,
        probe_evaluator=None,
    ) -> Tuple[str, dict]:
        """Select the best tutor action for the current query.

        Returns (action_name, kwargs) for env.tutor_act().
        """
        # Observation phase → always WAIT
        if block.in_observation_phase:
            return "WAIT", {}

        qs = block.current_query
        if qs is None or qs.done:
            return "WAIT", {}

        # Score all interventions (eval-aware or legacy)
        # PERF: eval-aware only in teaching phase; obs/eval always use legacy
        use_eval_aware = (
            self.cfg.tutor_scorer_mode == "eval_aware"
            and shadow_learner is not None
            and probe_evaluator is not None
            and block.in_teaching_phase
        )
        if use_eval_aware:
            candidates = self.cf_scorer.score_all_eval_aware(
                qs,
                profile=block.profile_state,
                scorer=scorer,
                danger_head=danger_head,
                learner_state=learner_state,
                access_mode=access_mode,
                shadow_learner=shadow_learner,
                probe_evaluator=probe_evaluator,
                lambda_now=self.cfg.lambda_now,
                lambda_probe=self.cfg.lambda_probe,
            )
        else:
            candidates = self.cf_scorer.score_all(
                qs,
                profile=block.profile_state,
                scorer=scorer,
                danger_head=danger_head,
                learner_state=learner_state,
                access_mode=access_mode,
            )

        if not candidates:
            return "WAIT", {}

        # Pick best (already sorted descending)
        best = candidates[0]

        # Only intervene if strictly better than WAIT Q-value
        wait_q = next((c.total_q for c in candidates if c.action == "WAIT"), 0.0)
        if best.total_q <= wait_q:
            return "WAIT", {}

        # Capture eval-aware diagnostics for calibration logging
        q_probe_chosen = getattr(best, 'q_probe', 0.0)
        # Also capture normalized probe value and probe std (P3-A)
        q_probe_z_chosen  = best.components.get('q_probe_z', None) if best.components else None
        probe_std_chosen   = best.components.get('probe_std', None) if best.components else None

        def _diag():
            d = {"_q_probe": q_probe_chosen}
            if q_probe_z_chosen is not None:
                d["_q_probe_z"]   = q_probe_z_chosen
                d["_probe_std"]   = probe_std_chosen
            return d

        # Convert to env action + kwargs
        if best.action == "RISK_HINT":
            return "RISK_HINT", {"hint_index": best.hint_index, **_diag()}
        elif best.action == "BAN":
            return "BAN", {"ban_index": best.ban_index, **_diag()}
        elif best.action == "HIGHLIGHT":
            return "HIGHLIGHT", {"highlight_cells": best.highlight_cells, **_diag()}
        elif best.action == "SKIP":
            return "SKIP", _diag()
        else:
            return "WAIT", _diag()

    def get_diagnostics(
        self,
        block: BlockState,
        scorer: DeterministicSemanticScorer,
        danger_head: Optional[DangerHead] = None,
        learner_state: Optional[PolicyStateSnapshot] = None,
        access_mode: str = "proxy_oracle",
        shadow_learner=None,
        probe_evaluator=None,
    ) -> List[InterventionScore]:
        """Get full Q-value breakdown for diagnostics."""
        qs = block.current_query
        if qs is None or qs.done:
            return []

        if (self.cfg.tutor_scorer_mode == "eval_aware"
                and shadow_learner is not None
                and probe_evaluator is not None):
            return self.cf_scorer.score_all_eval_aware(
                qs,
                profile=block.profile_state,
                scorer=scorer,
                danger_head=danger_head,
                learner_state=learner_state,
                access_mode=access_mode,
                shadow_learner=shadow_learner,
                probe_evaluator=probe_evaluator,
                lambda_now=self.cfg.lambda_now,
                lambda_probe=self.cfg.lambda_probe,
            )

        return self.cf_scorer.score_all(
            qs,
            profile=block.profile_state,
            scorer=scorer,
            danger_head=danger_head,
            learner_state=learner_state,
            access_mode=access_mode,
        )

    # ─────────────────────────────────────────────────────────────────────
    # L0 Speaker (RSA mode)
    # ─────────────────────────────────────────────────────────────────────

    def select_action_l0(
        self,
        block: BlockState,
        learner_agent,          # LearnerAgent — read-only access to live state
        rsa_cfg,                # RSAConfig from FullConfig
    ) -> Tuple[str, dict]:
        """L0 speaker: selects action by maximizing U_S0 over real learner state.

        U_S0(a) = λ_task * G_task(a) + λ_teach * G_teach(a)

        G_task(a) = ΔP_corr(a) - λ_ko * ΔE[dmg](a)
        G_teach^HL(H) = log q_post(j*) - log q_pre(j*)
        G_teach^BAN(j) = Δlogit P(r_j=1) * P_L(j)

        Runs in single-thread context only (F5 condition).
        learner_agent is NOT shared across calls; no mutation happens here.
        """
        qs = block.current_query
        if qs is None or qs.done or block.in_observation_phase:
            return "WAIT", {}

        from ..env.interventions import get_active_menu
        from .rsa_l0_speaker import compute_l0_utility
        import numpy as np

        active = get_active_menu(qs)
        K = len(active)
        if K == 0:
            return "WAIT", {}

        # Build read-only learner snapshot
        snap = self._build_learner_snapshot(learner_agent, qs, active)
        if snap is None:
            return "WAIT", {}

        # Score all candidate actions
        best_action, best_kwargs, best_u = "WAIT", {}, 0.0

        # --- HIGHLIGHT candidates ---
        L = len(qs.target_output)
        for cell_combo in self._candidate_highlight_cells(L, max_cells=self.cfg.max_highlight_cells):
            u = compute_l0_utility(
                action="HIGHLIGHT",
                action_cells=cell_combo,
                qs=qs, snap=snap, active=active,
                rsa_cfg=rsa_cfg,
                scorer=learner_agent._scorer,
            )
            if u > best_u:
                best_u = u
                best_action = "HIGHLIGHT"
                best_kwargs = {"highlight_cells": cell_combo}

        # --- BAN candidates ---
        for j, opt in enumerate(active):
            if opt.is_correct:
                continue  # never ban correct answer (tutor knows)
            u = compute_l0_utility(
                action="BAN", action_arg=j,
                qs=qs, snap=snap, active=active,
                rsa_cfg=rsa_cfg,
                scorer=learner_agent._scorer,
            )
            if u > best_u:
                best_u = u
                best_action = "BAN"
                best_kwargs = {"ban_index": opt.index}

        return best_action, best_kwargs

    @staticmethod
    def _build_learner_snapshot(learner_agent, qs, active):
        """Extract read-only learner state into LearnerStateSnapshot."""
        import numpy as np
        from ..interfaces import LearnerStateSnapshot
        from ..env.interventions import get_active_menu

        policy = learner_agent.policy
        scorer = learner_agent._scorer
        if scorer is None or policy.danger_head is None:
            return None

        K = len(active)
        L = len(qs.target_output)

        # Attention weights
        attn = (policy.attention.weights
                if policy.attention is not None
                else np.ones(L) / L)

        # Semantic scores
        sem = np.zeros(K)
        for i, opt in enumerate(active):
            sem[i] = scorer.score_option(qs.target_output, opt.text,
                                         attention_weights=attn)

        # Danger predictions
        d_preds = np.zeros(K)
        d_uncs  = np.zeros(K)
        d_p_h   = np.zeros(K)
        for i, opt in enumerate(active):
            mu, u = policy.danger_head.predict(opt.danger_vec)
            d_preds[i] = mu
            d_uncs[i]  = u
            d_p_h[i]   = policy.danger_head.hazard.predict(opt.danger_vec)

        # Pick probabilities (from current softmax)
        U_pick = (policy.cfg.alpha_sem * sem
                  - policy.cfg.alpha_risk * d_preds
                  - policy.cfg.alpha_unc  * d_uncs)
        shifted = U_pick - np.max(U_pick)
        exp_u   = np.exp(policy.cfg.beta_L * shifted)
        pick_pr = exp_u / (exp_u.sum() + 1e-10)

        return LearnerStateSnapshot(
            semantic_scores=sem.tolist(),
            danger_preds=d_preds.tolist(),
            danger_uncs=d_uncs.tolist(),
            hazard_probs=d_p_h.tolist(),
            pick_probs=pick_pr.tolist(),
            hp=qs.hp,
            active_option_indices=[o.index for o in active],
            danger_vecs=[o.danger_vec.tolist() for o in active],
            option_texts=[list(o.text) for o in active],
            attention_weights=attn.tolist(),
        )

    @staticmethod
    def _candidate_highlight_cells(L: int, max_cells: int = 2):
        """Generate candidate highlight subsets (single cells only for now)."""
        import itertools
        # First pass: single cells; Second: pairs if L >= 2 and max_cells >= 2
        cells = [(i,) for i in range(L)]
        if max_cells >= 2 and L >= 2:
            cells += list(itertools.combinations(range(L), 2))
        return cells


