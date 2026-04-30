"""
sparse_tutor.py â€” Bayes Gate Tutor with sparse BAN / HIGHLIGHT / MIX actions.

Replaces SHORTLIST as the primary teaching intervention.

Design:
  - Action space: WAIT | BAN(j) | HIGHLIGHT(cells) | MIX(j, cells)
  - Probability models are now LEARNER-CONSISTENT (Option A upgrade):
      _compute_learner_probs(): uses actual learner scoring path
        (scorer.score_option with attention-weighted mismatch + danger_head.predict)
        Under HIGHLIGHT: re-applies attention boost exp(rho_H) to highlighted cells
        then recomputes semantic scores â€” matches learner_agent.act() exactly.
        Under BAN: excludes banned option from active menu.
        NO hard-tier p(j*)=1 assumption.
      _rollout_estimate(): short N-step rollouts using learner scoring for
        calibrated multi-step P_death / P_timeout.
  - Q_use = Î»_evalÂ·G_eval + Î»_expÂ·G_exp
            - Î²Â·P_death - Î³Â·P_timeout
            - Î»_shiftÂ·D_shift - c(a)
  - D_shift = JS(learner_probs_WAIT || learner_probs_action)  [NOT tier-based]
  - HIGHLIGHT candidate only generated when P_timeout(WAIT) > hl_timeout_threshold
  - MIX = BAN(top_lethal_or_confusing) + HIGHLIGHT(correct_cells)
  - j* is never banned

rollout_mode (TutorConfig):
  "proxy"  : old static approximation (backward-compatible)
  "hybrid" : rollout for rescue + decision-boundary; proxy elsewhere
  "full"   : rollout for all non-WAIT candidates
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np

from ..config import FullConfig
from ..env.state import BlockState, QueryState
from ..env.option_env import OptionEnv
from ..env.interventions import get_active_menu
from ..interfaces import TutorStep, Option
from ..learner.learner_agent import LearnerAgent
from .sparse_tutor_ban import build_learning_ban_pool
from .sparse_tutor_candidates import enumerate_sparse_candidates
from .sparse_tutor_direct import (
    compute_direct_pick_probs,
    rollout_estimate_direct,
)
from .sparse_tutor_grace import handle_grace_round
from .allow_family import (
    FAMILY_NATIVE_LIKE_ALLOW,
    build_prereveal_allow_features,
    classify_prereveal_family,
)
from .sparse_tutor_phase import (
    infer_pedagogical_phase,
    maybe_record_audit_candidate,
    update_post_action_phase_flags,
)
from .sparse_tutor_routing import (
    compute_p_death_proxy,
    compute_p_timeout_proxy,
    route_pick_distribution,
    route_rollout_estimate,
    should_use_rollout,
)
from .sparse_tutor_scoring import (
    build_option_mass_records,
    compute_ban_oracle_stats,
    compute_outcome_conditioned_grace_conversion,
    compute_postreveal_consolidation_value,
    compute_postreveal_shift_decomp,
    compute_postreveal_q,
    compute_sparse_g_exp,
    summarize_option_mass_records,
)


# â”€â”€ JS divergence helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> float:
    """Jensen-Shannon divergence JS(p || q), bounded [0, ln2]."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(np.clip(0.5 * kl_pm + 0.5 * kl_qm, 0.0, np.log(2.0)))


def _spec_cache_key(spec: Dict[str, Any]) -> str:
    """Produce a hashable string key for a candidate spec dict.

    Used by the request-scoped learner_probs cache in _act_teaching().
    The key must uniquely identify the modified menu state that the spec
    induces, so that two different call sites requesting the same logical
    distribution receive the same cached array.

    Key format: "<action>|<sorted_extra_fields>"

    Examples:
      WAIT                     -> "WAIT"
      BAN(ban_index=2)         -> "BAN|ban_index=2"
      HIGHLIGHT(cells=(0,1,2)) -> "HIGHLIGHT|highlight_cells=(0, 1, 2)"
      MIX(bi=2, hc=(0,1))     -> "MIX|ban_index=2|highlight_cells=(0, 1)"
    """
    action = spec.get("action", "WAIT")
    extras = sorted(
        f"{k}={v}" for k, v in spec.items()
        if k not in ("action", "_q_detail")
    )
    return "|".join([action] + extras)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class SparseTutorAgent:
    """Bayes Gate Tutor: sparse BAN / HIGHLIGHT / MIX interventions.

    Interface mirrors OptionLevelTutorAgent for drop-in comparison.

    Args:
        cfg: FullConfig. Uses cfg.tutor (TutorConfig Stage A fields).
        g_learn_mode: G_learn estimation mode ("none" | "probe" | "oracle_surrogate").
                      Overrides cfg.tutor.sparse_g_learn_mode if provided.
        beta: P_death weight in Q_use (default from J objective: 0.5)
        gamma: P_timeout weight in Q_use (default from J objective: 0.2)
    """

    def __init__(
        self,
        cfg: Optional[FullConfig] = None,
        g_learn_mode: Optional[str] = None,
        beta: float = 0.5,
        gamma: float = 0.2,
        predictor=None,
    ):
        self.cfg = cfg or FullConfig()
        tcfg = self.cfg.tutor

        # Optional LearnerPredictor (predictor.py protocol).
        # When set, _compute_learner_probs() and _rollout_estimate()
        # delegate to predictor.pick_dist() / predictor.rollout().
        # When None, falls back to direct learner-access (legacy behavior).
        self._predictor = predictor

        self.g_learn_mode = g_learn_mode if g_learn_mode is not None else tcfg.sparse_g_learn_mode
        self.beta = beta
        self.gamma = gamma

        self.lambda_eval = tcfg.lambda_eval
        self.lambda_exp = tcfg.lambda_exp
        self.lambda_shift = tcfg.lambda_shift
        self.c_I = tcfg.c_I
        self.tutor_rho_H = tcfg.tutor_rho_H
        self.M_ban = tcfg.M_ban
        self.hl_timeout_threshold = tcfg.hl_timeout_threshold

        # â”€â”€ Rescue mode parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # theta_rescue: P_timeout(WAIT) threshold that triggers rescue mode
        # lambda_to:    weight on delta_P_timeout in Q_rescue
        # lambda_shift_res: shift penalty in rescue mode (lower than learning mode)
        self.theta_rescue = getattr(tcfg, 'theta_rescue', 0.5)
        self.lambda_to = getattr(tcfg, 'lambda_to', 1.0)
        self.lambda_shift_res = getattr(tcfg, 'lambda_shift_res', self.lambda_shift * 0.5)

        # â”€â”€ Rollout proxy parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.rollout_mode = getattr(tcfg, 'rollout_mode', 'hybrid')
        self.rollout_n = getattr(tcfg, 'rollout_n', 8)
        self.rollout_delta = getattr(tcfg, 'rollout_delta', 0.05)

        # G_learn estimator
        from .g_learn import GLearnEstimator
        self._g_learn = GLearnEstimator(
            mode=self.g_learn_mode,
            n_probe=tcfg.sparse_n_probe,
            seed=42,
        )
        self._probe_queries: List = []
        self._decision_trace: List[dict] = []
        self._mix_target_audit_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._productive_allow_meta_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._productive_allow_diagnostic_failures: int = 0

    # â”€â”€ Block init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize per-block state."""
        self._decision_trace = []
        self._mix_target_audit_cache = {}
        self._productive_allow_meta_cache = {}
        self._productive_allow_diagnostic_failures = 0
        block._decision_trace = self._decision_trace
        block._productive_allow_diagnostic_failures = 0

        if self.g_learn_mode == "probe" and support:
            from ..interfaces import Example
            self._probe_queries = [
                {'target_output': list(ex.output), 'menu': []}
                for ex in support
            ]
        else:
            self._probe_queries = []

        self._g_learn.init_block(
            support=support or [],
            grammar=grammar,
            cfg=getattr(self.cfg, '_cls_cfg', None),
            probe_queries=self._probe_queries,
        )
        self._productive_allow_meta_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # â”€â”€ Main entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def act(
        self,
        block: BlockState,
        env: OptionEnv,
        learner_agent: Optional[LearnerAgent] = None,
    ) -> TutorStep:
        """Execute one tutor turn.

        Obs / Eval phase: always WAIT.
        Teaching phase: enumerate sparse candidates â†’ argmax Q_use.
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return env.tutor_act(block, "WAIT")

        if block.in_observation_phase or block.in_evaluation_phase:
            return env.tutor_act(block, "WAIT")

        # Phase 6H.5 / 6H.7: grace round â€” MUST come before existing-intervention check.
        # After HIGHLIGHT/MIX, qs.highlighted_cells is set AND grace_round=True.
        # We must handle grace FIRST; otherwise the 'highlighted_cells â†’ WAIT' path
        # fires before grace logic can execute.
        grace_result = handle_grace_round(block, qs)
        grace_status = grace_result["status"]  # "none" | "wait" | "override"
        if grace_status == "wait":
            active = get_active_menu(qs)
            diag_labels = getattr(qs, "option_diag_labels", {})
            p_safe_diag_wait = 0.0
            p_high_risk_wait = 0.0
            uniform_mass = 1.0 / max(len(active), 1)
            for opt in active:
                if opt.is_correct:
                    continue
                label = diag_labels.get(opt.index, "")
                if label == "safe_diagnostic_wrong":
                    p_safe_diag_wait += uniform_mass
                elif label == "high_risk_lure":
                    p_high_risk_wait += uniform_mass
            self._decision_trace.append({
                "query_id": qs.query_id,
                "round_t": qs.rounds_used,
                "hp": qs.hp,
                "tau_remaining": max(0, qs.max_rounds - qs.rounds_used),
                "pre_rounds_left": max(0, qs.max_rounds - qs.rounds_used),
                "pre_hp": qs.hp,
                "pre_post_reveal_phase": getattr(qs, "post_reveal_phase", False),
                "pre_n_safe_diag_wrong_reveals": getattr(qs, "n_safe_diag_wrong_reveals", 0),
                "pre_last_wrong_diag_label": getattr(qs, "last_wrong_diag_label", ""),
                "pre_last_reveal_option_index": getattr(qs, "last_reveal_option_index", None),
                "pre_p_safe_diag_wait": round(p_safe_diag_wait, 4),
                "pre_p_high_risk_wait": round(p_high_risk_wait, 4),
                "pre_has_safe_diag_opp": p_safe_diag_wait > 0.25,
                "pre_has_high_risk_opp": p_high_risk_wait > 0.25,
                "phase": "GRACE_WAIT",
                "mode": "grace_wait",
                "grace_status": "consumed",
                "grace_reason": grace_result["reason"],
                "wait_reason": "WAIT_GRACE",
                "generation": {},
                "scoring": {"forced": False, "action": "WAIT", "grace_wait": True},
                "chosen_action": "WAIT",
                "p_correct_wait": 0.0,
                "outcome": None,
            })
            return env.tutor_act(block, "WAIT")

        # One intervention per query (BAN/MIX persist; don't double-intervene)
        # NOTE: must come AFTER grace check above.
        if qs.banned_indices or qs.highlighted_cells:
            return env.tutor_act(block, "WAIT")

        if learner_agent is None:
            return env.tutor_act(block, "WAIT")

        return self._act_teaching(block, env, learner_agent, grace_result=grace_result)

    def _act_teaching(
        self,
        block: BlockState,
        env: OptionEnv,
        learner: LearnerAgent,
        grace_result: Optional[Dict] = None,
    ) -> TutorStep:
        """Teaching-phase decision: dual-mode (learning vs rescue).

        Gate:
            if P_timeout(WAIT) > theta_rescue  â†’  rescue mode (deadline reduction)
            else                               â†’  learning mode (teaching Q_use)

        Records two-layer decision trace with mode field.
        """
        qs = block.current_query
        active = get_active_menu(qs)
        if not active:
            return env.tutor_act(block, "WAIT")

        # â”€â”€ Compute WAIT baseline bundle â€” once per _act_teaching() â”€
        # A+: compute wait_probs_lc / p_death_wait / p_timeout_wait ONCE.
        # All candidate scorers reference this shared bundle:
        #   _compute_q_use  â†’ base_probs, D_shift reference point
        #   pedagogical     â†’ dynamic d_max / t_max thresholds
        #   rescue gate     â†’ learner-consistent (already done)
        #   p_correct_wait  â†’ fraction of prob mass on correct option
        # This eliminates 1 full _compute_learner_probs call per non-WAIT candidate.
        self._learner_ref = learner

        # C: request-scoped learner_probs cache (local to this _act_teaching() call).
        # key = canonical spec action string; avoids re-calling score_option
        # for specs that produce the same distribution (e.g., WAIT already computed).
        # Keying by action string is safe because:
        #   - WAIT always maps to one fixed distribution
        #   - BAN(i), HL(H), MIX(i,H) each uniquely determine the modified menu
        # Cache is intentionally LOCAL: never survives across queries / rollout steps.
        _lp_cache: Dict[str, Any] = {}

        def _cached_learner_probs(spec_: Dict[str, Any]) -> Any:
            """Return learner_probs for spec_, computing once per spec action key."""
            key = _spec_cache_key(spec_)
            if key not in _lp_cache:
                _lp_cache[key] = self._compute_learner_probs(qs, active, spec_, learner)
            return _lp_cache[key]

        # Compute WAIT probs first (needed for rescue gate and baseline bundle).
        wait_probs_lc = _cached_learner_probs({"action": "WAIT"})
        p_timeout_wait = self._compute_p_timeout(qs, active, wait_probs_lc)
        p_death_wait   = self._compute_p_death(qs, active, wait_probs_lc)
        rescue_mode = p_timeout_wait > self.theta_rescue
        mode = "rescue" if rescue_mode else "learn"
        pre_rounds_left = max(0, qs.max_rounds - qs.rounds_used)
        diag_labels = getattr(qs, "option_diag_labels", {})
        p_safe_diag_wait = 0.0
        p_high_risk_wait = 0.0
        for i, opt in enumerate(active):
            if i >= len(wait_probs_lc) or opt.is_correct:
                continue
            p_j = float(wait_probs_lc[i])
            label = diag_labels.get(opt.index, "")
            if label == "safe_diagnostic_wrong":
                p_safe_diag_wait += p_j
            elif label == "high_risk_lure":
                p_high_risk_wait += p_j
        pre_phase = infer_pedagogical_phase(qs, active, wait_probs_lc, self.cfg)
        self._should_preserve_productive_allow(qs, active, learner)

        # â”€â”€ Phase 6H.6: Save audit candidate at post-reveal decision point â”€â”€
        # Snapshot the learner + query BEFORE action selection so that the
        # causal audit measures intervention effect on the real decision-time
        # learner state, not the final post-teach state.
        maybe_record_audit_candidate(
            block,
            qs,
            active,
            learner,
            self._select_highlight_cells,
        )

        # ── Phase 6I.1 P3: Forced post-reveal intervention ceiling ──
        force_action = getattr(self.cfg.tutor, 'force_postreveal_action', 'none')
        if force_action != 'none' and self._should_force_postreveal(qs):
            forced_spec = self._build_forced_action(qs, active, learner, force_action)
            if forced_spec is not None:
                action = forced_spec["action"]
                wait_q, wait_detail = (
                    self._compute_q_rescue(
                        qs, active, {"action": "WAIT"}, learner, p_timeout_wait,
                        _wait_probs=wait_probs_lc,
                        _spec_probs=wait_probs_lc,
                    )
                    if rescue_mode else
                    self._compute_q_use(
                        qs, active, {"action": "WAIT"}, learner,
                        _wait_probs=wait_probs_lc,
                        _spec_probs=wait_probs_lc,
                        p_death_wait=p_death_wait,
                        p_timeout_wait=p_timeout_wait,
                    )
                )
                forced_probs = _cached_learner_probs(forced_spec)
                forced_q, forced_detail = (
                    self._compute_q_rescue(
                        qs, active, forced_spec, learner, p_timeout_wait,
                        _wait_probs=wait_probs_lc,
                        _spec_probs=forced_probs,
                    )
                    if rescue_mode else
                    self._compute_q_use(
                        qs, active, forced_spec, learner,
                        _wait_probs=wait_probs_lc,
                        _spec_probs=forced_probs,
                        p_death_wait=p_death_wait,
                        p_timeout_wait=p_timeout_wait,
                    )
                )
                # Record trace
                trace_entry = {
                    "query_id": qs.query_id,
                    "round_t": qs.rounds_used,
                    "hp": qs.hp,
                    "tau_remaining": max(0, qs.max_rounds - qs.rounds_used),
                    "pre_rounds_left": pre_rounds_left,
                    "pre_hp": qs.hp,
                    "pre_post_reveal_phase": getattr(qs, "post_reveal_phase", False),
                    "pre_n_safe_diag_wrong_reveals": getattr(qs, "n_safe_diag_wrong_reveals", 0),
                    "pre_last_wrong_diag_label": getattr(qs, "last_wrong_diag_label", ""),
                    "pre_last_reveal_option_index": getattr(qs, "last_reveal_option_index", None),
                    "pre_p_safe_diag_wait": round(p_safe_diag_wait, 4),
                    "pre_p_high_risk_wait": round(p_high_risk_wait, 4),
                    "pre_has_safe_diag_opp": p_safe_diag_wait > 0.25,
                    "pre_has_high_risk_opp": p_high_risk_wait > 0.25,
                    "phase": pre_phase,
                    "n_active": len(active),
                    "n_lethal": 0,
                    "mode": "forced_ceiling",
                    "grace_status": grace_result["status"] if grace_result else "none",
                    "p_timeout_wait": round(p_timeout_wait, 4),
                    "scoring": {
                        "forced": True,
                        "action": action,
                        "force_type": force_action,
                        "q_wait": float(wait_q),
                        "candidates": [wait_detail, forced_detail],
                        "chosen_detail": forced_detail,
                    },
                    "chosen_action": action,
                    "chosen_ban_index": forced_spec.get("ban_index"),
                    "chosen_highlight_cells": forced_spec.get("highlight_cells"),
                    "p_correct_wait": 0.0,
                    "outcome": None,
                }
                self._decision_trace.append(trace_entry)
                # Execute
                if action == "HIGHLIGHT":
                    step = env.tutor_act(block, "HIGHLIGHT",
                                         highlight_cells=forced_spec["highlight_cells"])
                elif action == "MIX":
                    step = env.tutor_act(block, "MIX",
                                         ban_index=forced_spec["ban_index"],
                                         highlight_cells=forced_spec["highlight_cells"])
                elif action == "BAN":
                    step = env.tutor_act(block, "BAN",
                                         ban_index=forced_spec["ban_index"])
                else:
                    step = env.tutor_act(block, "WAIT")
                step.q_use_detail = forced_detail
                update_post_action_phase_flags(block, self.cfg, qs, action)
                return step

        # â”€â”€ Layer 1: Enumerate candidates with gate reasons â”€â”€â”€â”€â”€â”€
        candidates, gen_info = self._enumerate_candidates_traced(
            qs,
            active,
            learner,
            mode=mode,
            p_timeout_wait=p_timeout_wait,
            wait_probs_lc=wait_probs_lc,
        )

        # â”€â”€ Layer 2: Score all candidates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if not rescue_mode:
            candidates, direct_mix_info = self._apply_direct_mix_selector(
                qs,
                active,
                learner,
                candidates,
                wait_probs_lc=wait_probs_lc,
                p_death_wait=p_death_wait,
                p_timeout_wait=p_timeout_wait,
            )
            gen_info["mix_direct_selector"] = direct_mix_info
            candidates, joint_mix_info = self._apply_joint_mix_replay_gate(
                qs,
                active,
                learner,
                candidates,
                wait_probs_lc=wait_probs_lc,
                p_death_wait=p_death_wait,
                p_timeout_wait=p_timeout_wait,
            )
            gen_info["mix_joint_replay_gate"] = joint_mix_info

        q_wait = float('-inf')
        scored_candidates: List[Dict] = []
        best_q = float('-inf')
        best_action_spec: Dict[str, Any] = {"action": "WAIT"}

        for spec in candidates:
            if rescue_mode:
                q, detail = self._compute_q_rescue(
                    qs, active, spec, learner, p_timeout_wait,
                    _wait_probs=wait_probs_lc,
                    _spec_probs=_cached_learner_probs(spec),
                )
            else:
                # Pass cached WAIT probs (A+) and per-spec probs (C)
                spec_probs = _cached_learner_probs(spec)
                q, detail = self._compute_q_use(
                    qs, active, spec, learner,
                    _wait_probs=wait_probs_lc,
                    _spec_probs=spec_probs,
                    p_death_wait=p_death_wait,
                    p_timeout_wait=p_timeout_wait,
                )
            scored_candidates.append(detail)
            if spec["action"] == "WAIT":
                q_wait = q
            # â”€â”€ Guard filter (dual-mode only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # In protective mode: G_eval guard must pass.
            # In pedagogical mode: P_death/P_timeout hard constraints must pass.
            # WAIT always passes. Current mode: guard always True.
            guard_ok = detail.get('guard_passed', True)
            if q > best_q and guard_ok:
                best_q = q
                best_action_spec = spec
                best_action_spec["_q_detail"] = detail

        # â”€â”€ Compute margin_vs_wait for each candidate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for sc in scored_candidates:
            sc["margin_vs_wait"] = round(sc["q_use"] - q_wait, 6)

        # â”€â”€ Compute p(j* | WAIT) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        p_correct_wait = 0.0
        n_lethal = 0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_correct_wait = float(wait_probs_lc[i])  # learner-consistent
            if opt.risk_class >= qs.hp and not opt.is_correct:
                n_lethal += 1

        # â”€â”€ Compute p_timeout_after (chosen action) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # C: chosen action probs already in cache â€” no extra call needed.
        chosen_spec = best_action_spec
        chosen_probs_lc = _cached_learner_probs(chosen_spec)
        p_timeout_after = self._compute_p_timeout(qs, active, chosen_probs_lc)
        delta_p_timeout = round(p_timeout_wait - p_timeout_after, 6)

        # â”€â”€ Record trace entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        allow_meta = getattr(self, "_productive_allow_meta_cache", {}).get(
            (int(qs.query_id), int(qs.rounds_used)),
            {},
        )
        trace_entry = {
            "query_id": qs.query_id,
            "round_t": qs.rounds_used,
            "hp": qs.hp,
            "tau_remaining": max(0, qs.max_rounds - qs.rounds_used),
            "pre_rounds_left": pre_rounds_left,
            "pre_hp": qs.hp,
            "pre_post_reveal_phase": getattr(qs, "post_reveal_phase", False),
            "pre_contrastive_ticket_available": not bool(getattr(qs, "contrastive_update_used", False)),
            "pre_positive_ticket_available": not bool(getattr(qs, "positive_update_used", False)),
            "pre_both_tickets_available": (
                not bool(getattr(qs, "contrastive_update_used", False))
                and not bool(getattr(qs, "positive_update_used", False))
            ),
            "pre_n_safe_diag_wrong_reveals": getattr(qs, "n_safe_diag_wrong_reveals", 0),
            "pre_last_wrong_diag_label": getattr(qs, "last_wrong_diag_label", ""),
            "pre_last_reveal_option_index": getattr(qs, "last_reveal_option_index", None),
            "pre_p_safe_diag_wait": round(p_safe_diag_wait, 4),
            "pre_p_high_risk_wait": round(p_high_risk_wait, 4),
            "pre_has_safe_diag_opp": p_safe_diag_wait > 0.25,
            "pre_has_high_risk_opp": p_high_risk_wait > 0.25,
            "pre_productive_allow_preserved": bool(allow_meta.get("preserve", False)),
            "pre_productive_allow_preserve_base": bool(allow_meta.get("preserve_base", False)),
            "pre_productive_allow_reason": allow_meta.get("reason"),
            "pre_allow_reject_reason": allow_meta.get("allow_reject_reason"),
            "pre_allow_eligible": bool(allow_meta.get("eligible", False)),
            "pre_allow_phase_eligible": bool(allow_meta.get("phase_allows", False)),
            "pre_productive_mass_wait": round(float(allow_meta.get("productive_mass", 0.0)), 4),
            "pre_info_mass_wait": round(float(allow_meta.get("info_mass", 0.0)), 4),
            "pre_harm_mass_wait": round(float(allow_meta.get("harm_mass", 0.0)), 4),
            "pre_expected_damage_wait": round(float(allow_meta.get("expected_damage_wait", 0.0)), 4),
            "pre_allow_p_survive": round(float(allow_meta.get("p_survive", 0.0)), 4),
            "pre_allow_loop_value": round(float(allow_meta.get("allow_loop_value", 0.0)), 4),
            "pre_allow_best_cue_cate_estimate": round(float(allow_meta.get("best_cue_cate_estimate", 0.0)), 4),
            "pre_allow_post_reveal_best_value_estimate": round(float(allow_meta.get("post_reveal_best_value_estimate", 0.0)), 4),
            "pre_p_bounded_diag_wait": round(float(allow_meta.get("p_bounded_diag", 0.0)), 4),
            "pre_p_farwrong_wait": round(float(allow_meta.get("p_farwrong", 0.0)), 4),
            "pre_p_highrisk_wait": round(float(allow_meta.get("p_highrisk", 0.0)), 4),
            "phase": pre_phase,
            "n_active": len(active),
            "n_lethal": n_lethal,
            "mode": mode,
            "grace_status": grace_result["status"] if grace_result else "none",
            "p_timeout_wait": round(p_timeout_wait, 4),
            "p_timeout_after": round(p_timeout_after, 4),
            "delta_p_timeout": delta_p_timeout,
            "generation": gen_info,
            "scoring": {
                "q_wait": round(q_wait, 6),
                "candidates": scored_candidates,
            },
            "chosen_action": best_action_spec["action"],
            "chosen_ban_index": best_action_spec.get("ban_index"),
            "chosen_highlight_cells": best_action_spec.get("highlight_cells"),
            "p_correct_wait": round(p_correct_wait, 4),
            "outcome": None,  # filled in finalize_trace()
        }
        # 6I.5: WAIT reason codes
        chosen_act = best_action_spec["action"]
        if chosen_act == "WAIT":
            is_postreveal = getattr(qs, "post_reveal_phase", False)
            has_safe_diag_opp = p_safe_diag_wait > 0.25
            preserve_allow = bool(trace_entry.get("pre_productive_allow_preserved", False))
            # Check if any cue candidate had positive Q
            any_positive_cue = any(
                c.get("action") in ("HIGHLIGHT", "MIX") and c.get("q_use", -999) > q_wait
                for c in scored_candidates
            )
            any_cue_generated = gen_info.get("hl_generated", False) or gen_info.get("mix_generated", False)
            if grace_result and grace_result.get("status") == "override":
                if grace_result.get("protect_override"):
                    wait_reason = "WAIT_BLOCKED_BY_PROTECT"
                elif grace_result.get("deadline_override"):
                    wait_reason = "WAIT_BLOCKED_BY_DEADLINE"
                else:
                    wait_reason = "WAIT_GRACE_OVERRIDE"
            elif not is_postreveal and preserve_allow:
                wait_reason = "WAIT_ALLOW_SAFE_DIAG"
            elif is_postreveal and any_positive_cue:
                wait_reason = "WAIT_MISSED_POSITIVE_CUE"
            elif is_postreveal and any_cue_generated and not any_positive_cue:
                wait_reason = "WAIT_NO_GOOD_CUE"
            elif is_postreveal and not any_cue_generated:
                wait_reason = "WAIT_NO_GOOD_CUE"
            elif not is_postreveal and not has_safe_diag_opp:
                if p_correct_wait >= 0.5:
                    wait_reason = "WAIT_BORING_MASTERY"
                else:
                    wait_reason = "WAIT_NO_PED_OPPORTUNITY"
            else:
                wait_reason = "WAIT_GENERIC"
            trace_entry["wait_reason"] = wait_reason
        else:
            trace_entry["wait_reason"] = None
        self._decision_trace.append(trace_entry)

        # â”€â”€ Execute chosen action â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        action = best_action_spec["action"]

        if action == "WAIT":
            return env.tutor_act(block, "WAIT")
        elif action == "BAN":
            step = env.tutor_act(block, "BAN", ban_index=best_action_spec["ban_index"])
        elif action == "HIGHLIGHT":
            step = env.tutor_act(block, "HIGHLIGHT",
                                 highlight_cells=best_action_spec["highlight_cells"])
        elif action == "MIX":
            step = env.tutor_act(block, "MIX",
                                 ban_index=best_action_spec["ban_index"],
                                 highlight_cells=best_action_spec["highlight_cells"])
        else:
            step = env.tutor_act(block, "WAIT")

        step.q_use_detail = best_action_spec.get("_q_detail")

        update_post_action_phase_flags(block, self.cfg, qs, action)

        return step

    # â”€â”€ Candidate generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _enumerate_candidates_traced(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        mode: str = "learn",
        p_timeout_wait: float = 0.0,
        wait_probs_lc: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate candidates with Layer-1 gate diagnostics, mode-aware.

        In rescue mode:
          - BAN target is selected by BlockScore (timeout blocker), not danger
          - HIGHLIGHT is always generated (rescue gate bypasses hl_timeout_threshold)
          - MIX = timeout_blocker_BAN + HIGHLIGHT(j*)

        Returns:
            (candidates, generation_info)
        """
        return enumerate_sparse_candidates(
            self.cfg,
            qs,
            active,
            learner,
            mode=mode,
            p_timeout_wait=p_timeout_wait,
            hl_timeout_threshold=self.hl_timeout_threshold,
            wait_probs_lc=wait_probs_lc,
            select_ban_target=self._select_ban_target,
            select_timeout_blocker=self._select_timeout_blocker,
            select_highlight_cells=self._select_highlight_cells,
        )

    def _enumerate_candidates(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        mode: str = "learn",
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper used by older unit tests."""
        wait_probs_lc = self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
        p_timeout_wait = self._compute_p_timeout(qs, active, wait_probs_lc)
        candidates, _ = self._enumerate_candidates_traced(
            qs,
            active,
            learner,
            mode=mode,
            p_timeout_wait=p_timeout_wait,
            wait_probs_lc=wait_probs_lc,
        )
        return candidates

    def _compute_mix_target_audit(
        self,
        qs: QueryState,
        active: List[Option],
        candidate_pool: List[Option],
        learner: LearnerAgent,
    ) -> Dict[str, Any]:
        """Compute removed-vs-net-harm BAN oracles on the current post-reveal state."""
        labels = getattr(qs, "option_diag_labels", {}) or {}
        hp_scale = max(float(qs.hp), 1.0)
        last_wrong_index = getattr(qs, "last_reveal_option_index", None)
        wait_probs = self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
        ban_probs_by_index: Dict[int, np.ndarray] = {}
        for opt in candidate_pool:
            if opt.is_correct:
                continue
            ban_probs_by_index[int(opt.index)] = self._compute_learner_probs(
                qs,
                active,
                {"action": "BAN", "ban_index": opt.index},
                learner,
            )
        oracle = compute_ban_oracle_stats(
            active,
            wait_probs,
            ban_probs_by_index,
            labels,
            last_wrong_index=last_wrong_index,
            hp_scale=hp_scale,
        )
        wait_records = oracle.get("wait_records", []) or []
        top_wrong_idx = None
        top_wrong_prob = float("-inf")
        for rec in wait_records:
            if rec.get("is_correct"):
                continue
            p = float(rec.get("prob", 0.0))
            if p > top_wrong_prob:
                top_wrong_prob = p
                top_wrong_idx = int(rec["index"])
        for idx, rec in (oracle.get("per_target", {}) or {}).items():
            rec["target_is_top_prob_wrong"] = float(top_wrong_idx is not None and int(idx) == int(top_wrong_idx))
        oracle.update({
            "wait_probs": wait_probs,
            "top_wrong_index": top_wrong_idx,
            "target_mode": getattr(self.cfg.tutor, "mix_target_mode", "current"),
            "candidate_indices": [int(opt.index) for opt in candidate_pool if not opt.is_correct],
        })
        return oracle

    def _record_mix_target_audit(
        self,
        qs: QueryState,
        audit: Dict[str, Any],
        chosen_index: Optional[int],
    ) -> None:
        """Persist chosen-vs-oracle target audit for later scoring/metrics extraction."""
        key = (int(qs.query_id), int(qs.rounds_used))
        chosen_rec = (audit.get("per_target", {}) or {}).get(int(chosen_index)) if chosen_index is not None else None
        audit = dict(audit)
        audit["chosen_index"] = chosen_index
        audit["chosen_matches_removed_oracle"] = (
            int(chosen_index) == audit.get("removed_oracle_index")
            if chosen_index is not None and audit.get("removed_oracle_index") is not None
            else False
        )
        audit["chosen_matches_net_oracle"] = (
            int(chosen_index) == audit.get("net_oracle_index")
            if chosen_index is not None and audit.get("net_oracle_index") is not None
            else False
        )
        if chosen_rec is not None:
            audit["chosen_record"] = dict(chosen_rec)
            audit["removed_target_regret"] = max(
                0.0,
                float(audit.get("removed_oracle_mass", 0.0)) - float(chosen_rec.get("removed_harm_mass", 0.0)),
            )
            audit["net_target_regret"] = max(
                0.0,
                float(audit.get("net_oracle_drop", 0.0)) - float(chosen_rec.get("net_harm_drop", 0.0)),
            )
        else:
            audit["removed_target_regret"] = 0.0
            audit["net_target_regret"] = 0.0
        self._mix_target_audit_cache[key] = audit

    def _attach_mix_target_audit_to_detail(
        self,
        detail: Dict[str, Any],
        qs: QueryState,
    ) -> None:
        if detail.get("action") != "MIX":
            return
        mix_audit = self._mix_target_audit_cache.get((int(qs.query_id), int(qs.rounds_used)))
        if mix_audit is None:
            return
        chosen_rec = mix_audit.get("chosen_record") or {}
        detail["mix_target_audit"] = mix_audit
        detail.update({
            "mix_target_mode": mix_audit.get("target_mode", "current"),
            "mix_removed_oracle_index": mix_audit.get("removed_oracle_index"),
            "mix_net_oracle_index": mix_audit.get("net_oracle_index"),
            "mix_chosen_matches_removed_oracle": bool(mix_audit.get("chosen_matches_removed_oracle", False)),
            "mix_chosen_matches_net_oracle": bool(mix_audit.get("chosen_matches_net_oracle", False)),
            "mix_removed_target_regret": float(mix_audit.get("removed_target_regret", 0.0)),
            "mix_net_target_regret": float(mix_audit.get("net_target_regret", 0.0)),
            "mix_ban_target_policy_prob_wait": float(chosen_rec.get("target_prob", 0.0)),
            "mix_ban_target_badness": float(chosen_rec.get("target_harm", 0.0)),
            "mix_ban_target_label": chosen_rec.get("target_label", ""),
            "mix_ban_target_was_last_wrong": bool(chosen_rec.get("target_is_last_wrong", 0.0)),
            "mix_ban_target_was_highrisk": bool(chosen_rec.get("target_is_highrisk", 0.0)),
            "mix_ban_target_was_safe_diag": bool(chosen_rec.get("target_is_safe_diag", 0.0)),
            "mix_ban_target_was_far_wrong": bool(chosen_rec.get("target_is_far_wrong", 0.0)),
            "mix_ban_target_was_top_prob_wrong": bool(chosen_rec.get("target_is_top_prob_wrong", 0.0)),
            "mix_ban_target_was_correct": bool(chosen_rec.get("target_is_correct", 0.0)),
            "mix_removed_oracle_mass": float(mix_audit.get("removed_oracle_mass", 0.0)),
            "mix_net_oracle_drop": float(mix_audit.get("net_oracle_drop", 0.0)),
        })

    def _attach_postreveal_diagnostics_to_detail(
        self,
        detail: Dict[str, Any],
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        base_probs: np.ndarray,
        learner_probs: np.ndarray,
        *,
        p_death: float,
        p_timeout: float,
    ) -> None:
        learner_cfg = getattr(self.cfg, "learner", None)
        incidental_correct_credit = float(
            getattr(learner_cfg, "incidental_correct_credit", 0.5)
        )
        action = spec.get("action", detail.get("action", "WAIT"))
        detail["postreveal_consolidation_value"] = 0.0
        detail["postreveal_consolidation_effective"] = 0.0
        detail["postreveal_positive_ticket_available"] = False
        detail["postreveal_consolidation_source_weight"] = 0.0
        detail["postreveal_consolidation_reason"] = "off"
        detail["postreveal_traj_value"] = 0.0
        detail["postreveal_q_without_consolidate"] = 0.0
        detail["postreveal_q_with_consolidate"] = 0.0
        detail["postreveal_q_consolidate_delta"] = 0.0
        if not getattr(qs, "post_reveal_phase", False):
            self._attach_mix_target_audit_to_detail(detail, qs)
            return

        diag_labels = getattr(qs, "option_diag_labels", {}) or {}
        last_wrong_idx = getattr(qs, "last_reveal_option_index", None)
        value_mode = getattr(self.cfg.tutor, "postreveal_value_mode", "legacy")
        decomp = compute_postreveal_shift_decomp(
            active,
            base_probs,
            learner_probs,
            diag_labels,
            last_wrong_index=last_wrong_idx,
            hp_scale=max(qs.hp, 1),
            ban_target_index=spec.get("ban_index"),
        )
        rounds_left = max(0, qs.max_rounds - qs.rounds_used)
        p_terminal = min(1.0, max(0.0, p_death + p_timeout))
        grace_conversion = compute_outcome_conditioned_grace_conversion(
            active,
            learner_probs,
            diag_labels,
            last_wrong_index=getattr(qs, "last_reveal_option_index", None),
            rounds_left=rounds_left,
            p_terminal=p_terminal,
            hp_scale=max(qs.hp, 1),
        )
        consolidation_meta = compute_postreveal_consolidation_value(
            qs,
            p_correct_action=float(decomp.get("p_correct_action", 0.0)),
            action_name=action,
            incidental_correct_credit=incidental_correct_credit,
        )
        consolidation_value = 0.0
        if getattr(self.cfg.tutor, "use_postreveal_consolidation_value", False):
            consolidation_value = float(consolidation_meta.get("consolidation_value", 0.0))
        detail["postreveal_value_mode"] = value_mode
        detail["postreveal_decomp"] = decomp
        detail["postreveal_grace_conversion"] = grace_conversion
        detail["postreveal_consolidation_value"] = float(
            consolidation_meta.get("consolidation_value", 0.0)
        )
        detail["postreveal_consolidation_effective"] = float(consolidation_value)
        detail["postreveal_positive_ticket_available"] = bool(
            consolidation_meta.get("positive_ticket_available", False)
        )
        detail["postreveal_consolidation_source_weight"] = float(
            consolidation_meta.get("source_weight", 0.0)
        )
        detail["postreveal_consolidation_reason"] = str(
            consolidation_meta.get("reason", "none")
        )
        q_without_consolidate = compute_postreveal_q(
            decomp,
            action_name=action,
            value_mode=value_mode,
            lambda_info_post=getattr(self.cfg.tutor, "postreveal_info_weight", 0.0),
            grace_conversion=grace_conversion,
            consolidation_value=0.0,
            cost=0.0,
        )
        q_with_consolidate = compute_postreveal_q(
            decomp,
            action_name=action,
            value_mode=value_mode,
            lambda_info_post=getattr(self.cfg.tutor, "postreveal_info_weight", 0.0),
            grace_conversion=grace_conversion,
            consolidation_value=consolidation_value,
            cost=0.0,
        )
        detail["postreveal_traj_value"] = q_with_consolidate
        detail["postreveal_q_without_consolidate"] = float(q_without_consolidate)
        detail["postreveal_q_with_consolidate"] = float(q_with_consolidate)
        detail["postreveal_q_consolidate_delta"] = float(
            q_with_consolidate - q_without_consolidate
        )
        detail["g_consolidate"] = float(consolidation_value)
        detail["g_consolidate_raw"] = float(
            consolidation_meta.get("consolidation_value", 0.0)
        )
        detail["postreveal_cue_q"] = (
            detail["postreveal_traj_value"]
            if action in ("HIGHLIGHT", "MIX")
            else 0.0
        )
        p_correct_next = float(decomp.get("p_correct_action", 0.0))
        detail["postreveal_p_correct_next"] = p_correct_next
        detail["postreveal_p_correct_2r"] = min(
            1.0,
            max(0.0, p_correct_next + float(grace_conversion)),
        )
        detail.update(decomp)
        self._attach_mix_target_audit_to_detail(detail, qs)

    def _select_ban_target(
        self,
        qs: QueryState,
        non_correct: List[Option],
        learner: LearnerAgent,
    ) -> Optional[Option]:
        """Select the best distractor to BAN (learning mode: danger-first).

        Priority:
          1. Highest expected damage lethal options (risk >= HP_t)
          2. Among equally risky, highest learner pick_prob (most confusing)

        Phase 6G (tutor_lg_mode="diagnostic"):
          - Avoid banning SAFE_DIAGNOSTIC_WRONG or BOUNDED_DIAGNOSTIC_WRONG
          - Prefer banning HIGH_RISK_LURE

        Phase 6H.5 (tutor_lg_mode="self_correct"):
          - Hard guard: never BAN the first safe_diagnostic_wrong opportunity
            if learner can survive the pick and has >= 1 round left afterward
          - Only lifts if: hp <= 1, or no time left, or already revealed
        """
        if not non_correct:
            return None

        hp = qs.hp
        lg_mode = getattr(self.cfg.tutor, 'tutor_lg_mode', 'off')
        mix_target_mode = getattr(self.cfg.tutor, "mix_target_mode", "current")

        diag_labels = getattr(qs, 'option_diag_labels', {})
        hard_guard_enabled = getattr(self.cfg.tutor, 'protect_safe_diag_hard_guard', False)
        lures, pool = build_learning_ban_pool(
            qs,
            non_correct,
            diag_labels,
            lg_mode=lg_mode,
            hard_guard_enabled=hard_guard_enabled,
        )
        active = get_active_menu(qs)

        if self._should_preserve_productive_allow(qs, active, learner):
            return None

        # Phase 6I.6B: post-reveal bad-mass-aware target audit / selector.
        # Keep pre-reveal BAN behavior unchanged; only use HarmMass oracles in the
        # post-reveal consolidation regime where protection-vs-cue tradeoffs matter.
        if getattr(qs, "post_reveal_phase", False) and pool:
            try:
                audit = self._compute_mix_target_audit(qs, active, pool, learner)
            except Exception:
                audit = None
            if audit is not None:
                chosen_idx = None
                if mix_target_mode == "removed_badmass":
                    chosen_idx = audit.get("removed_oracle_index")
                elif mix_target_mode == "net_badmass":
                    chosen_idx = audit.get("net_oracle_index")
                if chosen_idx is not None:
                    chosen_opt = next((o for o in pool if int(o.index) == int(chosen_idx)), None)
                    if chosen_opt is not None:
                        self._record_mix_target_audit(qs, audit, chosen_opt.index)
                        return chosen_opt
                # Even when we keep legacy selection, preserve oracle diagnostics.
                self._record_mix_target_audit(qs, audit, None)

        if lures:
            try:
                probs = self._compute_pick_probs_for_opts(
                    qs, lures, learner, use_logit=True)
                chosen = lures[int(np.argmax(probs))]
                cached = self._mix_target_audit_cache.get((int(qs.query_id), int(qs.rounds_used)))
                if cached is not None:
                    self._record_mix_target_audit(qs, cached, chosen.index)
                return chosen
            except Exception:
                chosen = lures[0]
                cached = self._mix_target_audit_cache.get((int(qs.query_id), int(qs.rounds_used)))
                if cached is not None:
                    self._record_mix_target_audit(qs, cached, chosen.index)
                return chosen
        lethal = [o for o in pool if o.risk_class >= hp]
        pool = lethal if lethal else pool

        # Score by pick_prob (confusion)
        try:
            probs = self._compute_pick_probs_for_opts(qs, pool, learner,
                                                      use_logit=True)
            idx = int(np.argmax(probs))
            chosen = pool[idx]
            cached = self._mix_target_audit_cache.get((int(qs.query_id), int(qs.rounds_used)))
            if cached is not None:
                self._record_mix_target_audit(qs, cached, chosen.index)
            return chosen
        except Exception:
            chosen = pool[0]
            cached = self._mix_target_audit_cache.get((int(qs.query_id), int(qs.rounds_used)))
            if cached is not None:
                self._record_mix_target_audit(qs, cached, chosen.index)
            return chosen

    def _candidate_joint_mix_highlight_sets(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        current_hl_cells: Tuple[int, ...],
    ) -> List[Tuple[int, ...]]:
        variants: List[Tuple[int, ...]] = []
        seen: Set[Tuple[int, ...]] = set()

        def _add(cells: Optional[Tuple[int, ...]]) -> None:
            if not cells:
                return
            cells_t = tuple(int(c) for c in cells)
            if cells_t and cells_t not in seen:
                seen.add(cells_t)
                variants.append(cells_t)

        _add(current_hl_cells)
        try:
            from .highlight_selection import select_counterfactual_highlight_cells

            cf_cells = select_counterfactual_highlight_cells(
                qs,
                active,
                learner,
                max_cells=int(getattr(self.cfg.tutor, "max_highlight_cells", 2)),
                m_candidates=4,
            )
            _add(cf_cells)
        except Exception:
            pass
        try:
            j_star = next((o for o in active if o.is_correct), None)
            if j_star is not None:
                _add(self._select_highlight_cells(qs, j_star, learner))
        except Exception:
            pass
        try:
            L = len(qs.target_output) if qs.target_output else 0
            max_cells = int(getattr(self.cfg.tutor, "max_highlight_cells", 2))
            _add(tuple(range(min(max_cells, L))))
        except Exception:
            pass
        singleton_cells = sorted({int(c) for cells in variants for c in cells})
        for c in singleton_cells:
            _add((c,))
        return variants

    def _should_preserve_productive_allow(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
    ) -> bool:
        cache = getattr(self, "_productive_allow_meta_cache", None)
        if cache is None:
            cache = {}
            self._productive_allow_meta_cache = cache
        if not hasattr(self, "_productive_allow_diagnostic_failures"):
            self._productive_allow_diagnostic_failures = 0
        cache_key = (int(getattr(qs, "query_id", -1)), int(getattr(qs, "rounds_used", 0)))
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and "preserve" in cached:
            return bool(cached.get("preserve", False))

        if not self.cfg.tutor.productive_allow_planning:
            cache[cache_key] = {
                "preserve": False,
                "preserve_base": False,
                "eligible": False,
                "reason": "planning_disabled",
                "allow_reject_reason": "PLANNING_DISABLED",
            }
            return False
        if qs.post_reveal_phase or qs.success or qs.hp <= 0:
            cache[cache_key] = {
                "preserve": False,
                "preserve_base": False,
                "eligible": False,
                "reason": "not_prereveal",
                "allow_reject_reason": "NOT_PRE_REVEAL",
            }
            return False
        try:
            wait_probs = self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
            phase = infer_pedagogical_phase(qs, active, wait_probs, self.cfg)
        except Exception:
            cache[cache_key] = {
                "preserve": False,
                "preserve_base": False,
                "eligible": False,
                "reason": "phase_inference_failed",
                "allow_reject_reason": "PHASE_INFERENCE_FAILED",
            }
            return False

        allow_mode = self.cfg.tutor.productive_allow_mode
        phase_allows = phase == "PRE_REVEAL_ALLOW"
        if allow_mode not in ("controlled_v1", "controlled_v2", "native_like_v1"):
            preserve = bool(phase_allows)
            cache[cache_key] = {
                "preserve": preserve,
                "preserve_base": preserve,
                "eligible": False,
                "reason": "phase_only" if preserve else "phase_not_allow",
                "allow_reject_reason": "ALLOW_PRESERVED" if preserve else "NOT_PRE_REVEAL",
                "phase": phase,
            }
            return preserve

        try:
            records = build_option_mass_records(
                active,
                wait_probs,
                qs.option_diag_labels or {},
                last_wrong_index=qs.last_reveal_option_index,
                hp_scale=max(qs.hp, 1),
            )
            summary = summarize_option_mass_records(records)
        except Exception:
            # Fail open for pedagogy, but record that controlled_v1 could not
            # actually evaluate its diagnostic gate on this state.
            self._productive_allow_diagnostic_failures += 1
            cache[cache_key] = {
                "preserve": True,
                "preserve_base": True,
                "eligible": False,
                "reason": "diagnostic_failure",
                "allow_reject_reason": "DIAGNOSTIC_FAILURE",
                "phase": phase,
            }
            return True

        from .sparse_tutor_horizon import compute_pre_reveal_allow_value

        p_safe_diag = sum(float(rec["prob"]) for rec in records if rec["is_safe_diag"] > 0.0)
        p_bounded_diag = sum(float(rec["prob"]) for rec in records if rec["is_bounded_diag"] > 0.0)
        p_farwrong = float(summary["p_farwrong"])
        p_highrisk = float(summary["p_highrisk"])
        info_mass = float(summary["info_mass"])
        harm_mass = float(summary["harm_mass"])
        expected_damage_wait = float(summary.get("expected_damage", 0.0))
        p_correct_wait = float(summary.get("p_correct", 0.0))
        rounds_left = max(0, qs.max_rounds - qs.rounds_used)
        contrastive_ticket_available = not bool(getattr(qs, "contrastive_update_used", False))
        positive_ticket_available = not bool(getattr(qs, "positive_update_used", False))
        both_tickets_available = contrastive_ticket_available and positive_ticket_available
        p_survive = max(0.0, (float(qs.hp) - expected_damage_wait) / max(1.0, float(qs.hp)))
        p_time = 1.0 if rounds_left >= 3 else 0.0
        best_cue_cate = float(getattr(self.cfg.tutor, "_best_cue_cate_estimate", 0.03))
        if getattr(self.cfg.tutor, "oracle_horizon", False):
            best_cue_cate = float(getattr(self.cfg.tutor, "_oracle_cate", best_cue_cate))
        productive_mass = (p_safe_diag + 0.5 * p_bounded_diag)
        post_reveal_best_value_estimate = max(0.0, best_cue_cate) * max(0.0, 1.0 - p_correct_wait)
        allow_loop_value = float(
            compute_pre_reveal_allow_value(
                p_safe_diag,
                p_bounded_diag,
                p_survive,
                p_time,
                best_cue_cate,
                p_correct_wait,
                harm_mass_wait=harm_mass,
                contrastive_ticket_available=contrastive_ticket_available,
                positive_ticket_available=positive_ticket_available,
            )
        )

        # Minimal controlled allow rule:
        #   - still needs the PRE_REVEAL_ALLOW phase
        #   - safe/diagnostic info must be present
        #   - productive information must dominate expected harmful mass
        #   - far/high-risk mass must not dominate the reveal opportunity
        productive_opportunity = productive_mass > 0.0
        info_dominates_harm = info_mass >= harm_mass
        safe_diag_not_dominated = p_safe_diag >= (p_highrisk + p_farwrong)
        ratio_gate = productive_mass >= (0.5 * harm_mass)
        highrisk_dominates = p_highrisk > (p_safe_diag + p_bounded_diag)
        farwrong_dominates = p_farwrong > (p_safe_diag + p_bounded_diag)
        mastery_high = p_correct_wait >= 0.75
        enough_rounds_for_loop = rounds_left >= 3
        eligible_for_allow = bool(phase_allows and both_tickets_available and enough_rounds_for_loop)
        allow_family_features = build_prereveal_allow_features(
            post_reveal_phase=bool(getattr(qs, "post_reveal_phase", False)),
            success=bool(getattr(qs, "success", False)),
            hp=float(getattr(qs, "hp", 0.0)),
            n_safe_diag_wrong_reveals=int(getattr(qs, "n_safe_diag_wrong_reveals", 0)),
            both_tickets_available=both_tickets_available,
            rounds_left=rounds_left,
            p_safe_diag=p_safe_diag,
            p_bounded_diag=p_bounded_diag,
            p_farwrong=p_farwrong,
            p_highrisk=p_highrisk,
            p_correct_wait=p_correct_wait,
            harm_mass=harm_mass,
            expected_damage_wait=expected_damage_wait,
            productive_mass=productive_mass,
        )
        allow_family = classify_prereveal_family(allow_family_features)

        preserve = bool(
            phase_allows
            and
            productive_opportunity
            and info_dominates_harm
            and safe_diag_not_dominated
            and rounds_left >= 2
            and qs.hp > 1
        )
        preserve_native_like = bool(
            phase_allows and allow_family == FAMILY_NATIVE_LIKE_ALLOW
        )

        def _allow_reject_reason(preserve_flag: bool) -> str:
            if preserve_flag:
                return "ALLOW_PRESERVED"
            if allow_mode == "native_like_v1" and allow_family != FAMILY_NATIVE_LIKE_ALLOW:
                return f"FAMILY_{allow_family}"
            if not contrastive_ticket_available:
                return "NO_CONTRASTIVE_TICKET"
            if not positive_ticket_available:
                return "NO_POSITIVE_TICKET"
            if not enough_rounds_for_loop:
                return "NOT_ENOUGH_ROUNDS"
            if not productive_opportunity:
                return "NO_PRODUCTIVE_MASS"
            if not ratio_gate:
                return "HARM_DOMINATES"
            if highrisk_dominates:
                return "HIGH_RISK_DOMINATES"
            if farwrong_dominates:
                return "FAR_WRONG_DOMINATES"
            if allow_mode == "controlled_v2" and allow_loop_value <= 0.0:
                return "POST_REVEAL_VALUE_LOW"
            if mastery_high:
                return "MASTERY_ALREADY_HIGH"
            if phase == "PROTECT":
                return "PROTECT_REQUIRED"
            if not phase_allows:
                return "NOT_PRE_REVEAL"
            return "CONTROLLED_BLOCKED"

        meta = {
            "phase": phase,
            "reason": "controlled_base" if preserve else "controlled_blocked",
            "p_safe_diag": p_safe_diag,
            "p_bounded_diag": p_bounded_diag,
            "p_farwrong": p_farwrong,
            "p_highrisk": p_highrisk,
            "p_correct_wait": p_correct_wait,
            "info_mass": info_mass,
            "harm_mass": harm_mass,
            "expected_damage_wait": expected_damage_wait,
            "p_survive": p_survive,
            "productive_mass": productive_mass,
            "rounds_left": rounds_left,
            "contrastive_ticket_available": contrastive_ticket_available,
            "positive_ticket_available": positive_ticket_available,
            "both_tickets_available": both_tickets_available,
            "best_cue_cate_estimate": best_cue_cate,
            "post_reveal_best_value_estimate": post_reveal_best_value_estimate,
            "allow_loop_value": allow_loop_value,
            "preserve_base": preserve,
            "eligible": eligible_for_allow,
            "phase_allows": phase_allows,
            "info_dominates_harm": info_dominates_harm,
            "ratio_gate": ratio_gate,
            "safe_diag_not_dominated": safe_diag_not_dominated,
            "highrisk_dominates": highrisk_dominates,
            "farwrong_dominates": farwrong_dominates,
            "mastery_high": mastery_high,
            "allow_family_split": allow_family,
            "safe_diag_quality_gap": float(allow_family_features["safe_diag_quality_gap"]),
            "allow_reject_reason": _allow_reject_reason(False),
        }
        if allow_mode == "native_like_v1":
            meta["preserve"] = preserve_native_like
            meta["reason"] = "native_like_allow" if preserve_native_like else "native_like_blocked"
            meta["allow_reject_reason"] = _allow_reject_reason(preserve_native_like)
            cache[cache_key] = meta
            return preserve_native_like
        if allow_mode != "controlled_v2":
            meta["preserve"] = preserve
            meta["reason"] = "controlled_v1_allow" if preserve else "controlled_v1_blocked"
            meta["allow_reject_reason"] = _allow_reject_reason(preserve)
            cache[cache_key] = meta
            return preserve

        preserve_v2 = bool(
            preserve
            and both_tickets_available
            and enough_rounds_for_loop
            and allow_loop_value > 0.0
        )
        meta["preserve"] = preserve_v2
        meta["allow_reject_reason"] = _allow_reject_reason(preserve_v2)
        if preserve_v2:
            meta["reason"] = "controlled_v2_allow"
        elif not preserve:
            meta["reason"] = "controlled_v2_base_blocked"
        elif not both_tickets_available:
            meta["reason"] = "controlled_v2_missing_ticket"
        elif not enough_rounds_for_loop:
            meta["reason"] = "controlled_v2_not_enough_rounds"
        elif allow_loop_value <= 0.0:
            meta["reason"] = "controlled_v2_zero_loop_value"
        else:
            meta["reason"] = "controlled_v2_blocked"
        cache[cache_key] = meta
        return preserve_v2

    def _apply_direct_mix_selector(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        candidates: List[Dict[str, Any]],
        *,
        wait_probs_lc: np.ndarray,
        p_death_wait: float,
        p_timeout_wait: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "enabled": bool(getattr(self.cfg.tutor, "direct_mix_selector", False)),
            "evaluated": False,
            "applied": False,
            "suppressed": False,
            "current_q": None,
            "selected_q": None,
            "selected_ban_index": None,
            "selected_highlight_cells": None,
            "topk_target_indices": [],
            "selected_net_harm_drop": 0.0,
            "oracle_net_harm_drop": 0.0,
            "net_target_regret": 0.0,
        }
        if not info["enabled"] or not getattr(qs, "post_reveal_phase", False):
            return candidates, info

        mix_pos = next((i for i, spec in enumerate(candidates) if spec.get("action") == "MIX"), None)
        if mix_pos is None:
            return candidates, info

        current_spec = dict(candidates[mix_pos])
        current_ban_index = current_spec.get("ban_index")
        current_hl_cells = tuple(current_spec.get("highlight_cells") or ())
        if current_ban_index is None or not current_hl_cells:
            return candidates, info

        diag_labels = getattr(qs, "option_diag_labels", {}) or {}
        hard_guard_enabled = getattr(self.cfg.tutor, "protect_safe_diag_hard_guard", False)
        non_correct = [o for o in active if not o.is_correct]
        _, pool = build_learning_ban_pool(
            qs,
            non_correct,
            diag_labels,
            lg_mode=getattr(self.cfg.tutor, "tutor_lg_mode", "off"),
            hard_guard_enabled=hard_guard_enabled,
        )
        if not pool:
            return candidates, info

        try:
            audit = self._compute_mix_target_audit(qs, active, pool, learner)
        except Exception:
            audit = None
        if audit is None:
            return candidates, info

        info["evaluated"] = True
        info["oracle_net_harm_drop"] = float(audit.get("net_oracle_drop", 0.0))
        per_target = list((audit.get("per_target", {}) or {}).values())
        per_target = [rec for rec in per_target if not bool(rec.get("target_is_correct", 0.0))]
        if not per_target:
            return candidates, info
        per_target = sorted(
            per_target,
            key=lambda rec: (
                float(rec.get("net_harm_drop", 0.0)),
                float(rec.get("removed_harm_mass", 0.0)),
            ),
            reverse=True,
        )
        positive_targets = [rec for rec in per_target if float(rec.get("net_harm_drop", 0.0)) > 1e-9]
        ranked_targets = positive_targets if positive_targets else per_target
        top_k = max(1, int(getattr(self.cfg.tutor, "direct_mix_top_k", 3)))
        top_targets = ranked_targets[:top_k]
        info["topk_target_indices"] = [int(rec["target_index"]) for rec in top_targets]

        if not positive_targets:
            updated = [dict(spec) for spec in candidates if spec.get("action") != "MIX"]
            info["applied"] = True
            info["suppressed"] = True
            self._record_mix_target_audit(qs, audit, None)
            return updated, info

        labels = getattr(qs, "option_diag_labels", {}) or {}
        last_wrong_idx = getattr(qs, "last_reveal_option_index", None)
        hp_scale = max(qs.hp, 1)
        hl_variants = self._candidate_joint_mix_highlight_sets(qs, active, learner, current_hl_cells)
        if not hl_variants:
            return candidates, info
        top_m = max(1, int(getattr(self.cfg.tutor, "direct_mix_top_m", 3)))

        def _mix_probs_and_decomp(ban_index: int, hl_cells: Tuple[int, ...]):
            spec = {
                "action": "MIX",
                "ban_index": int(ban_index),
                "highlight_cells": tuple(hl_cells),
            }
            spec_probs = self._compute_learner_probs(qs, active, spec, learner)
            decomp = compute_postreveal_shift_decomp(
                active,
                wait_probs_lc,
                spec_probs,
                labels,
                last_wrong_index=last_wrong_idx,
                hp_scale=hp_scale,
                ban_target_index=int(ban_index),
            )
            return spec, spec_probs, decomp

        current_probs = self._compute_learner_probs(qs, active, current_spec, learner)
        current_q, _ = self._compute_q_use(
            qs,
            active,
            current_spec,
            learner,
            _wait_probs=wait_probs_lc,
            _spec_probs=current_probs,
            p_death_wait=p_death_wait,
            p_timeout_wait=p_timeout_wait,
        )
        best_q = float(current_q)
        best_spec = dict(current_spec)
        best_rec = next((rec for rec in top_targets if int(rec["target_index"]) == int(current_ban_index)), None)

        for rec in top_targets:
            ban_index = int(rec["target_index"])
            hl_ranked: List[Tuple[float, Tuple[int, ...], Any]] = []
            for hl_cells in hl_variants:
                _, spec_probs, decomp = _mix_probs_and_decomp(ban_index, tuple(hl_cells))
                hl_rank_score = (
                    max(0.0, float(decomp.get("delta_p_correct", 0.0)))
                    + max(0.0, float(decomp.get("log_margin_gain", decomp.get("correct_margin_gain", 0.0))))
                )
                hl_ranked.append((hl_rank_score, tuple(hl_cells), spec_probs))
            hl_ranked.sort(key=lambda item: item[0], reverse=True)
            for _, hl_cells, spec_probs in hl_ranked[:top_m]:
                spec = {
                    "action": "MIX",
                    "ban_index": ban_index,
                    "highlight_cells": tuple(hl_cells),
                }
                q, _ = self._compute_q_use(
                    qs,
                    active,
                    spec,
                    learner,
                    _wait_probs=wait_probs_lc,
                    _spec_probs=spec_probs,
                    p_death_wait=p_death_wait,
                    p_timeout_wait=p_timeout_wait,
                )
                if float(q) > best_q + 1e-9:
                    best_q = float(q)
                    best_spec = dict(spec)
                    best_rec = rec

        selected_ban_index = int(best_spec.get("ban_index", current_ban_index))
        selected_hl_cells = tuple(best_spec.get("highlight_cells") or current_hl_cells)
        self._record_mix_target_audit(qs, audit, selected_ban_index)
        info.update({
            "applied": True,
            "current_q": float(current_q),
            "selected_q": best_q,
            "selected_ban_index": selected_ban_index,
            "selected_highlight_cells": list(selected_hl_cells),
            "selected_net_harm_drop": float((best_rec or {}).get("net_harm_drop", 0.0)),
            "net_target_regret": max(
                0.0,
                float(audit.get("net_oracle_drop", 0.0)) - float((best_rec or {}).get("net_harm_drop", 0.0)),
            ),
        })

        updated_spec = dict(best_spec)
        updated_spec.update({
            "mix_direct_selector_applied": True,
            "mix_direct_current_ban_index": int(current_ban_index),
            "mix_direct_selected_ban_index": selected_ban_index,
            "mix_direct_selected_highlight_cells": list(selected_hl_cells),
            "mix_direct_current_q": float(current_q),
            "mix_direct_selected_q": best_q,
            "mix_direct_topk_target_indices": list(info["topk_target_indices"]),
            "mix_direct_selected_net_harm_drop": float(info["selected_net_harm_drop"]),
            "mix_direct_oracle_net_harm_drop": float(info["oracle_net_harm_drop"]),
            "mix_direct_net_target_regret": float(info["net_target_regret"]),
        })
        updated = list(candidates)
        updated[mix_pos] = updated_spec
        return updated, info

    def _apply_joint_mix_replay_gate(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        candidates: List[Dict[str, Any]],
        *,
        wait_probs_lc: np.ndarray,
        p_death_wait: float,
        p_timeout_wait: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "enabled": bool(getattr(self.cfg.tutor, "joint_mix_replay_gate", False)),
            "evaluated": False,
            "applied": False,
            "replaced": False,
            "current_q": None,
            "target_q": None,
            "highlight_q": None,
            "separate_q": None,
            "joint_q": None,
            "target_regret": 0.0,
            "highlight_regret": 0.0,
            "joint_regret": 0.0,
            "joint_interaction_regret": 0.0,
            "selected_ban_index": None,
            "selected_highlight_cells": None,
        }
        if not info["enabled"]:
            return candidates, info
        if not getattr(qs, "post_reveal_phase", False):
            return candidates, info

        mix_pos = next((i for i, spec in enumerate(candidates) if spec.get("action") == "MIX"), None)
        if mix_pos is None:
            return candidates, info

        current_spec = dict(candidates[mix_pos])
        current_ban_index = current_spec.get("ban_index")
        current_hl_cells = tuple(current_spec.get("highlight_cells") or ())
        if current_ban_index is None or not current_hl_cells:
            return candidates, info

        diag_labels = getattr(qs, "option_diag_labels", {}) or {}
        hard_guard_enabled = getattr(self.cfg.tutor, "protect_safe_diag_hard_guard", False)
        non_correct = [o for o in active if not o.is_correct]
        _, pool = build_learning_ban_pool(
            qs,
            non_correct,
            diag_labels,
            lg_mode=getattr(self.cfg.tutor, "tutor_lg_mode", "off"),
            hard_guard_enabled=hard_guard_enabled,
        )
        if not pool:
            return candidates, info

        hl_variants = self._candidate_joint_mix_highlight_sets(qs, active, learner, current_hl_cells)
        if not hl_variants:
            return candidates, info

        info["evaluated"] = True

        def _eval_mix(ban_index: int, hl_cells: Tuple[int, ...]) -> float:
            spec = {
                "action": "MIX",
                "ban_index": int(ban_index),
                "highlight_cells": tuple(hl_cells),
            }
            spec_probs = self._compute_learner_probs(qs, active, spec, learner)
            q, _ = self._compute_q_use(
                qs,
                active,
                spec,
                learner,
                _wait_probs=wait_probs_lc,
                _spec_probs=spec_probs,
                p_death_wait=p_death_wait,
                p_timeout_wait=p_timeout_wait,
            )
            return float(q)

        current_q = _eval_mix(int(current_ban_index), current_hl_cells)
        best_target_q = current_q
        best_target_idx = int(current_ban_index)
        for opt in pool:
            q = _eval_mix(int(opt.index), current_hl_cells)
            if q > best_target_q:
                best_target_q = q
                best_target_idx = int(opt.index)

        best_highlight_q = current_q
        best_highlight_cells = current_hl_cells
        for hl_cells in hl_variants:
            q = _eval_mix(int(current_ban_index), hl_cells)
            if q > best_highlight_q:
                best_highlight_q = q
                best_highlight_cells = tuple(hl_cells)

        separate_q = _eval_mix(best_target_idx, best_highlight_cells)

        best_joint_q = current_q
        best_joint_idx = int(current_ban_index)
        best_joint_cells = current_hl_cells
        for opt in pool:
            for hl_cells in hl_variants:
                q = _eval_mix(int(opt.index), tuple(hl_cells))
                if q > best_joint_q:
                    best_joint_q = q
                    best_joint_idx = int(opt.index)
                    best_joint_cells = tuple(hl_cells)

        info.update({
            "current_q": current_q,
            "target_q": best_target_q,
            "highlight_q": best_highlight_q,
            "separate_q": separate_q,
            "joint_q": best_joint_q,
            "target_regret": max(0.0, best_target_q - current_q),
            "highlight_regret": max(0.0, best_highlight_q - current_q),
            "joint_regret": max(0.0, best_joint_q - current_q),
            "joint_interaction_regret": max(0.0, best_joint_q - separate_q),
            "selected_ban_index": best_joint_idx,
            "selected_highlight_cells": list(best_joint_cells),
        })

        selected_index = best_joint_idx if best_joint_q > current_q + 1e-9 else int(current_ban_index)
        try:
            audit = self._compute_mix_target_audit(qs, active, pool, learner)
            self._record_mix_target_audit(qs, audit, selected_index)
        except Exception:
            pass

        updated_spec = dict(current_spec)
        updated_spec.update({
            "mix_joint_gate_applied": True,
            "mix_joint_gate_replaced": bool(best_joint_q > current_q + 1e-9),
            "mix_joint_current_q": current_q,
            "mix_joint_target_q": best_target_q,
            "mix_joint_highlight_q": best_highlight_q,
            "mix_joint_separate_q": separate_q,
            "mix_joint_best_q": best_joint_q,
            "mix_joint_target_regret": max(0.0, best_target_q - current_q),
            "mix_joint_highlight_regret": max(0.0, best_highlight_q - current_q),
            "mix_joint_regret": max(0.0, best_joint_q - current_q),
            "mix_joint_interaction_regret": max(0.0, best_joint_q - separate_q),
            "mix_joint_current_ban_index": int(current_ban_index),
            "mix_joint_current_highlight_cells": list(current_hl_cells),
            "mix_joint_selected_ban_index": best_joint_idx,
            "mix_joint_selected_highlight_cells": list(best_joint_cells),
        })

        if best_joint_q > current_q + 1e-9:
            updated_spec["ban_index"] = best_joint_idx
            updated_spec["highlight_cells"] = best_joint_cells
            info["applied"] = True
            info["replaced"] = True
        else:
            info["applied"] = True
            info["replaced"] = False

        candidates = list(candidates)
        candidates[mix_pos] = updated_spec
        return candidates, info

    def _select_timeout_blocker(
        self,
        qs: QueryState,
        active: List[Option],
        j_star: Option,
        non_correct: List[Option],
        learner: LearnerAgent,
    ) -> Tuple[Optional[Option], Optional[float]]:
        """Select the timeout blocker to BAN (rescue mode: BlockScore).

        BlockScore(j) = 1[j != j*] * 1[rank(j) < rank(j*) under WAIT] * p_wait(j)

        Selects the non-correct option with the highest pick probability
        that is currently ranked above j* (i.e., being selected before j*,
        causing timeout).

        Returns (blocker_option, blocker_score) or (fallback_ban, None).
        """
        if not non_correct:
            return None, None

        try:
            wait_probs = self._compute_tier_probs(qs, active, {"action": "WAIT"})
            # Map option.index -> wait_prob
            prob_by_idx = {opt.index: float(wait_probs[i])
                           for i, opt in enumerate(active)}
            p_jstar = prob_by_idx.get(j_star.index, 0.0)

            # BlockScore: non-correct opts with p_wait > p_jstar (ranked before j*)
            blockers = []
            for opt in non_correct:
                p_j = prob_by_idx.get(opt.index, 0.0)
                if p_j > p_jstar:  # ranked strictly above j* â†’ most likely to block
                    blockers.append((opt, p_j))

            if blockers:
                # Pick the one with the highest pick prob (biggest blocker)
                best = max(blockers, key=lambda x: x[1])
                return best[0], best[1]
            else:
                # j* already has the highest prob; fallback: highest pick prob non-correct
                probs_nc = self._compute_pick_probs_for_opts(
                    qs, non_correct, learner, use_logit=True
                )
                idx = int(np.argmax(probs_nc))
                return non_correct[idx], float(probs_nc[idx])
        except Exception:
            return non_correct[0], None

    # ── Phase 6I.1 P3: Forced post-reveal helpers ──────────────────

    def _should_force_postreveal(self, qs: QueryState) -> bool:
        """Check if current state qualifies for forced post-reveal intervention.

        Conditions:
          - post_reveal_phase = True
          - n_safe_diag_wrong_reveals >= 1
          - query not solved
          - rounds_left >= 1
          - hp > 0
          - this query hasn't been force-cued yet (prevent repeated forcing)
        """
        if not getattr(qs, 'post_reveal_phase', False):
            return False
        if getattr(qs, 'n_safe_diag_wrong_reveals', 0) < 1:
            return False
        if qs.success:
            return False
        if qs.hp <= 0:
            return False
        rounds_left = max(0, qs.max_rounds - qs.rounds_used)
        if rounds_left < 1:
            return False
        # Prevent repeated forcing on same query
        if getattr(qs, '_forced_postreveal_done', False):
            return False
        return True

    def _build_forced_action(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        force_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Build the action spec for forced post-reveal intervention.

        force_type:
          "HL_cf"      -> HIGHLIGHT with counterfactual cells
          "MIX_cf"     -> MIX with counterfactual cells + ban target
          "best_CATE"  -> whichever of HL_cf/MIX_cf is better
        """
        from .highlight_selection import select_counterfactual_highlight_cells

        # Get counterfactual highlight cells
        max_cells = int(getattr(self.cfg.tutor, "max_highlight_cells", 2))
        hl_cells = select_counterfactual_highlight_cells(
            qs, active, learner, max_cells=max_cells, m_candidates=4)
        if hl_cells is None:
            L = len(qs.target_output) if qs.target_output else 0
            hl_cells = tuple(range(min(max_cells, L)))

        # Find ban target for MIX
        ban_target_idx = None
        labels = getattr(qs, 'option_diag_labels', {})
        non_correct = [o for o in active if not o.is_correct]
        audit = None
        if non_correct:
            try:
                audit = self._compute_mix_target_audit(qs, active, non_correct, learner)
            except Exception:
                audit = None
        target_mode = getattr(self.cfg.tutor, "mix_target_mode", "current")
        if audit is not None:
            if target_mode == "removed_badmass":
                ban_target_idx = audit.get("removed_oracle_index")
            elif target_mode == "net_badmass":
                ban_target_idx = audit.get("net_oracle_index")
        if ban_target_idx is None:
            for opt in active:
                if not opt.is_correct and labels.get(opt.index, "") == "high_risk_lure":
                    ban_target_idx = opt.index
                    break
        if ban_target_idx is None:
            for opt in active:
                if not opt.is_correct and opt.risk_class >= qs.hp:
                    ban_target_idx = opt.index
                    break
        if ban_target_idx is None:
            # Fallback: pick highest-risk non-correct
            worst = None
            worst_risk = -1
            for opt in active:
                if not opt.is_correct and opt.risk_class > worst_risk:
                    worst = opt
                    worst_risk = opt.risk_class
            if worst is not None:
                ban_target_idx = worst.index
        if audit is not None:
            self._record_mix_target_audit(qs, audit, ban_target_idx)

        # Mark forced to prevent repetition
        qs._forced_postreveal_done = True

        if force_type == "HL_cf":
            return {"action": "HIGHLIGHT", "highlight_cells": hl_cells}

        elif force_type == "MIX_cf":
            if ban_target_idx is not None:
                return {"action": "MIX",
                        "ban_index": ban_target_idx,
                        "highlight_cells": hl_cells}
            else:
                return {"action": "HIGHLIGHT", "highlight_cells": hl_cells}

        elif force_type == "best_CATE":
            # Compare full post-reveal trajectory value when possible; otherwise
            # fall back to immediate P(correct) as the legacy ceiling.
            from .highlight_selection import _eval_p_correct
            if getattr(qs, "post_reveal_phase", False):
                base_probs = self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
                hl_probs = self._compute_learner_probs(
                    qs, active, {"action": "HIGHLIGHT", "highlight_cells": hl_cells}, learner
                )
                hl_decomp = compute_postreveal_shift_decomp(
                    active,
                    base_probs,
                    hl_probs,
                    labels,
                    last_wrong_index=getattr(qs, "last_reveal_option_index", None),
                    hp_scale=max(qs.hp, 1),
                    ban_target_index=None,
                )
                rounds_left = max(0, qs.max_rounds - qs.rounds_used)
                p_terminal_hl = max(0.0, min(1.0, self._compute_p_death(qs, active, hl_probs) + self._compute_p_timeout(qs, active, hl_probs)))
                grace_hl = max(
                    0.0,
                    (1.0 - float(hl_decomp.get("p_correct_action", 0.0)) - p_terminal_hl)
                    * (1.0 if rounds_left >= 2 else 0.0)
                    * float(hl_decomp.get("p_correct_action", 0.0))
                )
                consolidate_hl = 0.0
                learner_cfg = getattr(self.cfg, "learner", None)
                incidental_correct_credit = float(
                    getattr(learner_cfg, "incidental_correct_credit", 0.5)
                )
                if getattr(self.cfg.tutor, "use_postreveal_consolidation_value", False):
                    consolidate_hl = float(
                        compute_postreveal_consolidation_value(
                            qs,
                            p_correct_action=float(hl_decomp.get("p_correct_action", 0.0)),
                            action_name="HIGHLIGHT",
                            incidental_correct_credit=incidental_correct_credit,
                        ).get("consolidation_value", 0.0)
                    )
                value_mode = getattr(self.cfg.tutor, "postreveal_value_mode", "legacy")
                q_hl = compute_postreveal_q(
                    hl_decomp,
                    action_name="HIGHLIGHT",
                    value_mode=value_mode,
                    lambda_info_post=getattr(self.cfg.tutor, "postreveal_info_weight", 0.0),
                    grace_conversion=grace_hl,
                    consolidation_value=consolidate_hl,
                    cost=0.0,
                )
                q_mix = float("-inf")
                if ban_target_idx is not None:
                    mix_probs = self._compute_learner_probs(
                        qs,
                        active,
                        {"action": "MIX", "ban_index": ban_target_idx, "highlight_cells": hl_cells},
                        learner,
                    )
                    mix_decomp = compute_postreveal_shift_decomp(
                        active,
                        base_probs,
                        mix_probs,
                        labels,
                        last_wrong_index=getattr(qs, "last_reveal_option_index", None),
                        hp_scale=max(qs.hp, 1),
                        ban_target_index=ban_target_idx,
                    )
                    p_terminal_mix = max(
                        0.0,
                        min(1.0, self._compute_p_death(qs, active, mix_probs) + self._compute_p_timeout(qs, active, mix_probs)),
                    )
                    grace_mix = max(
                        0.0,
                        (1.0 - float(mix_decomp.get("p_correct_action", 0.0)) - p_terminal_mix)
                        * (1.0 if rounds_left >= 2 else 0.0)
                        * float(mix_decomp.get("p_correct_action", 0.0))
                    )
                    consolidate_mix = 0.0
                    if getattr(self.cfg.tutor, "use_postreveal_consolidation_value", False):
                        consolidate_mix = float(
                            compute_postreveal_consolidation_value(
                                qs,
                                p_correct_action=float(mix_decomp.get("p_correct_action", 0.0)),
                                action_name="MIX",
                                incidental_correct_credit=incidental_correct_credit,
                            ).get("consolidation_value", 0.0)
                        )
                    q_mix = compute_postreveal_q(
                        mix_decomp,
                        action_name="MIX",
                        value_mode=value_mode,
                        lambda_info_post=getattr(self.cfg.tutor, "postreveal_info_weight", 0.0),
                        grace_conversion=grace_mix,
                        consolidation_value=consolidate_mix,
                        cost=0.0,
                    )
                if q_mix >= q_hl and ban_target_idx is not None:
                    return {"action": "MIX", "ban_index": ban_target_idx, "highlight_cells": hl_cells}
            else:
                p_hl = _eval_p_correct(qs, active, learner, highlight_cells=hl_cells)
                if ban_target_idx is not None:
                    import copy
                    qs_mix = copy.deepcopy(qs)
                    qs_mix.banned_indices.add(ban_target_idx)
                    remaining = [o for o in active if o.index != ban_target_idx]
                    p_mix = _eval_p_correct(qs_mix, remaining, learner,
                                            highlight_cells=hl_cells)
                    if p_mix >= p_hl:
                        return {"action": "MIX",
                                "ban_index": ban_target_idx,
                                "highlight_cells": hl_cells}
            return {"action": "HIGHLIGHT", "highlight_cells": hl_cells}

        return None

    def _select_highlight_cells(
        self,
        qs: QueryState,
        j_star: Option,
        learner: LearnerAgent,
    ) -> Optional[Tuple[int, ...]]:
        """Select output cells to HIGHLIGHT for j* using diagnostic D_l score.

        Phase 6F: Supports three modes via cfg.env.highlight_mode:
          "diagnostic" : D_l-based selector (default)
          "fixed"      : first-N cells (old stub, ablation baseline)
          "none"       : suppress HIGHLIGHT entirely (returns None)
        """
        hl_mode = getattr(self.cfg.env, 'highlight_mode', 'diagnostic')
        max_cells = self.cfg.tutor.max_highlight_cells
        L = len(qs.target_output)
        if L == 0:
            return None

        # "none" mode: suppress HIGHLIGHT generation
        if hl_mode == "none":
            return None

        # "fixed" mode: old stub behavior
        if hl_mode == "fixed":
            cells = tuple(range(min(max_cells, L)))
            return cells if cells else None

        # 6I-B: "counterfactual_pcorrect" mode
        if hl_mode == "counterfactual_pcorrect":
            active = get_active_menu(qs)
            if not active:
                return tuple(range(min(max_cells, L)))
            try:
                from .highlight_selection import select_counterfactual_highlight_cells
                cells = select_counterfactual_highlight_cells(
                    qs, active, learner, max_cells=max_cells, m_candidates=4)
                return cells if cells else tuple(range(min(max_cells, L)))
            except Exception:
                cells = tuple(range(min(max_cells, L)))
                return cells if cells else None

        # "diagnostic" mode: D_l-based selector (default)
        active = get_active_menu(qs)
        if not active:
            return tuple(range(min(max_cells, L)))

        # Get pick distribution from predictor or learner
        try:
            if self._predictor is not None:
                pick_probs = self._predictor.pick_dist(qs, active, {"action": "WAIT"})
            else:
                pick_probs = self._compute_learner_probs(
                    qs, active, {"action": "WAIT"}, learner)

            from .highlight_selection import select_diagnostic_highlight_cells
            cells = select_diagnostic_highlight_cells(
                qs, active, pick_probs, max_cells)
            return cells if cells else tuple(range(min(max_cells, L)))
        except Exception:
            # Fallback to first-N cells on any error
            cells = tuple(range(min(max_cells, L)))
            return cells if cells else None

    # â”€â”€ Q_use computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _compute_q_use(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
        _q_wait_ref: Optional[float] = None,
        _wait_probs: Optional[Any] = None,    # A+: pre-computed WAIT baseline bundle
        _spec_probs: Optional[Any] = None,    # C:  pre-computed spec learner_probs
        p_death_wait: float = 0.0,
        p_timeout_wait: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Compute Q_use for a candidate action spec.

        Supports three tutor_mode values (TutorConfig.tutor_mode):

          "current" (default, backward-compat):
              Q_use = Î»_eval*G_eval + Î»_exp*G_exp
                    - Î²*P_death - Î³*P_timeout - Î»_shift*D_shift - c(a)
              guard_passed = True always.

          "protective":
              Q_score = U_teach(a) - Î»_shift*D_shift - c(a)
              guard_passed = (G_eval(a) >= -eps_eval_guard)  [or g_learn=none]
              Rollout forced for non-WAIT to get accurate p_success.

          "pedagogical":
              Q_score = G_eval(a) + Î·*U_teach(a) - Î»_shift*D_shift - c(a)
              guard_passed = P_death(a) â‰¤ d_max AND P_timeout(a) â‰¤ t_max
              d_max = min(p_death_wait + d_max_margin, d_max_cap)  [dynamic]
              t_max = min(p_timeout_wait + t_max_margin, t_max_cap)  [dynamic]
              Rollout forced for non-WAIT.

        p_death_wait / p_timeout_wait: pre-computed WAIT baselines from
        _act_teaching(), used for pedagogical dynamic constraint thresholds.

        Returns (q_value, detail_dict).
        """
        action = spec["action"]
        tcfg = self.cfg.tutor
        tutor_mode = getattr(tcfg, 'tutor_mode', 'current')

        # â”€â”€ Learner-consistent probability distributions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # C: use pre-computed spec probs if supplied (request-scoped cache).
        # A+: use pre-computed WAIT probs if supplied (avoids repeated beam search).
        learner_probs = (_spec_probs if _spec_probs is not None
                         else self._compute_learner_probs(qs, active, spec, learner))
        base_probs    = (_wait_probs if _wait_probs is not None
                         else self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner))

        # â”€â”€ G_eval (ProbeEvaluator / OracleSurrogate) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        g_eval = 0.0
        if action != "WAIT" and self.g_learn_mode != "none":
            g_eval = self._compute_g_eval(qs, spec, learner_probs, learner)

        # G_exp (safe exposure gain; only used in "current" mode)
        # Phase 6H.7: pass _spec_probs sidecar so _compute_g_exp can compute
        # actual delta P(correct) for HIGHLIGHT/MIX instead of d_cell_bar constant.
        feedback_mode = self.cfg.env.feedback_mode
        g_exp_spec = spec if "_spec_probs" in spec else {
            **spec, "_spec_probs": learner_probs}
        g_exp = self._compute_g_exp(
            qs, active, learner_probs, g_exp_spec, wait_probs=base_probs
        )
        if feedback_mode == "nonreveal":
            g_exp = 0.0
        g_exp_base = float(g_exp)
        g_exp_consolidation_bonus = 0.0

        # â”€â”€ P_death / P_timeout â€” proxy (cheap, always computed) â”€â”€
        p_death_proxy = self._compute_p_death(qs, active, learner_probs)
        p_timeout_proxy = self._compute_p_timeout(qs, active, learner_probs)

        # â”€â”€ Rollout refinement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Force rollout in dual-mode (non-WAIT) to get calibrated p_success
        # for U_teach. In current mode, defer to _use_rollout gate.
        if tutor_mode != 'current' and action != 'WAIT':
            use_ro = True
        else:
            use_ro = self._use_rollout(mode='normal', action=action,
                                       q_wait=_q_wait_ref,
                                       p_timeout_proxy=p_timeout_proxy)

        if use_ro:
            p_death, p_timeout, p_success = self._rollout_estimate(
                qs, active, spec, learner)
        else:
            p_death, p_timeout = p_death_proxy, p_timeout_proxy
            # WAIT or proxy path: approximate p_success from proxy estimates
            p_success = float(max(0.0, min(1.0, 1.0 - p_death - p_timeout)))

        # â”€â”€ D_shift (JS on learner-consistent distributions) â”€â”€â”€â”€â”€â”€
        d_shift = _js_divergence(base_probs, learner_probs)
        effective_d_shift = d_shift

        # â”€â”€ Action cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        raw_cost = self._action_cost(spec)
        cost = raw_cost

        # â”€â”€ U_teach (shared component) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        u_teach = self._compute_u_teach(p_success, p_death, p_timeout)

        # â”€â”€ Dual-mode Q computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        u_learn = g_eval   # alias: long-term learning signal

        if tutor_mode == 'protective':
            # Q = U_teach - D_shift - cost
            # Guard: filter any action where G_eval < -eps_eval_guard.
            # Exception: WAIT always passes (reference point).
            # Exception: g_learn=none â†’ no eval signal â†’ guard bypassed.
            if action == 'WAIT':
                guard_passed = True
            elif self.g_learn_mode == 'none':
                guard_passed = True  # no eval info â†’ can't guard
            else:
                guard_passed = (u_learn >= -getattr(tcfg, 'eps_eval_guard', 0.01))
            q = u_teach - self.lambda_shift * d_shift - cost

        elif tutor_mode == 'pedagogical':
            # Q = G_eval + Î·*U_teach - D_shift - cost
            # Hard constraints: disqualify if P_death > d_max OR P_timeout > t_max.
            # Thresholds are dynamic: WAIT baseline + margin, capped at absolute max.
            d_max = min(
                p_death_wait + getattr(tcfg, 'd_max_margin', 0.01),
                getattr(tcfg, 'd_max_cap', 0.20)
            )
            t_max = min(
                p_timeout_wait + getattr(tcfg, 't_max_margin', 0.03),
                getattr(tcfg, 't_max_cap', 0.70)
            )
            if action == 'WAIT':
                guard_passed = True
            else:
                guard_passed = (p_death <= d_max and p_timeout <= t_max)
            eta = getattr(tcfg, 'eta_pedagogical', 0.25)
            q = u_learn + eta * u_teach - self.lambda_shift * d_shift - cost

        else:  # "current" â€” original formula, full backward-compat
            guard_passed = True
            _phase = "DEFAULT"
            _phase_bias = 0.0
            _postreveal_decomp = None
            _grace_conversion = 0.0
            _cue_q = 0.0
            _consolidation_bonus_for_gexp = 0.0
            # 6I.4-C: phase controller + post-reveal shift decomposition
            try:
                from .sparse_tutor_phase import infer_pedagogical_phase, phase_action_prior
                _phase = infer_pedagogical_phase(qs, active, learner_probs, self.cfg)
                _prior = phase_action_prior(_phase)
                _phase_bias = _prior.get(action, 0.0)
            except Exception:
                _phase = "DEFAULT"
                _phase_bias = 0.0

            # 6I.4-C: In POST_REVEAL_CONSOLIDATE, replace raw D_shift with
            # harmful-shift-aware cue value for HIGHLIGHT/MIX
            _postreveal_decomp = None
            is_postreveal = getattr(qs, 'post_reveal_phase', False)
            if is_postreveal and action in ("HIGHLIGHT", "MIX"):
                from .sparse_tutor_scoring import compute_postreveal_shift_decomp, compute_postreveal_q
                diag_labels = getattr(qs, 'option_diag_labels', {})
                last_wrong_idx = getattr(qs, 'last_reveal_option_index', None)
                value_mode = getattr(tcfg, "postreveal_value_mode", "legacy")
                _postreveal_decomp = compute_postreveal_shift_decomp(
                    active, base_probs, learner_probs, diag_labels,
                    last_wrong_index=last_wrong_idx,
                    hp_scale=max(qs.hp, 1),
                    ban_target_index=spec.get("ban_index"),
                )
                if value_mode == "traj_v1":
                    # In trajectory mode, harmful shift is handled inside cue value
                    # instead of being double-counted as a generic JS penalty.
                    effective_d_shift = 0.0
                    cost = 0.5 * raw_cost
                elif value_mode == "traj_v2":
                    effective_d_shift = 0.0
                    cost = 0.5 * raw_cost
                else:
                    # Legacy path: replace JS penalty with harmful-shift proxy.
                    effective_d_shift = _postreveal_decomp["bad_shift"]
                rounds_left = max(0, qs.max_rounds - qs.rounds_used)
                p_terminal = min(1.0, max(0.0, p_death + p_timeout))
                _grace_conversion = compute_outcome_conditioned_grace_conversion(
                    active,
                    learner_probs,
                    diag_labels,
                    last_wrong_index=last_wrong_idx,
                    rounds_left=rounds_left,
                    p_terminal=p_terminal,
                    hp_scale=max(qs.hp, 1),
                )
                _consolidation_value = 0.0
                if getattr(tcfg, "use_postreveal_consolidation_value", False):
                    learner_cfg = getattr(self.cfg, "learner", None)
                    _consolidation_value = float(
                        compute_postreveal_consolidation_value(
                            qs,
                            p_correct_action=float(_postreveal_decomp.get("p_correct_action", 0.0)),
                            action_name=action,
                            incidental_correct_credit=float(
                                getattr(learner_cfg, "incidental_correct_credit", 0.5)
                            ),
                        ).get("consolidation_value", 0.0)
                    )
                _cue_q = compute_postreveal_q(
                    _postreveal_decomp,
                    action_name=action,
                    value_mode=value_mode,
                    lambda_info_post=getattr(tcfg, "postreveal_info_weight", 0.0),
                    grace_conversion=_grace_conversion,
                    consolidation_value=_consolidation_value,
                    cost=0.0,
                )  # cost handled separately

            if (
                is_postreveal
                and getattr(tcfg, "promote_postreveal_consolidation_into_gexp", False)
            ):
                learner_cfg = getattr(self.cfg, "learner", None)
                incidental_correct_credit = float(
                    getattr(learner_cfg, "incidental_correct_credit", 0.5)
                )
                diag_labels = getattr(qs, "option_diag_labels", {})
                last_wrong_idx = getattr(qs, "last_reveal_option_index", None)
                rounds_left = max(0, qs.max_rounds - qs.rounds_used)
                p_terminal = min(1.0, max(0.0, p_death + p_timeout))
                if _postreveal_decomp is not None:
                    p_correct_next_for_gexp = float(
                        _postreveal_decomp.get("p_correct_action", 0.0)
                    )
                    grace_for_gexp = float(_grace_conversion)
                else:
                    p_correct_next_for_gexp = 0.0
                    for i, opt in enumerate(active):
                        if opt.is_correct:
                            p_correct_next_for_gexp = float(learner_probs[i])
                            break
                    grace_for_gexp = compute_outcome_conditioned_grace_conversion(
                        active,
                        learner_probs,
                        diag_labels,
                        last_wrong_index=last_wrong_idx,
                        rounds_left=rounds_left,
                        p_terminal=p_terminal,
                        hp_scale=max(qs.hp, 1),
                    )
                p_correct_2r_for_gexp = min(
                    1.0,
                    max(0.0, p_correct_next_for_gexp + float(grace_for_gexp)),
                )
                _consolidation_bonus_for_gexp = float(
                    compute_postreveal_consolidation_value(
                        qs,
                        p_correct_action=p_correct_2r_for_gexp,
                        action_name=action,
                        incidental_correct_credit=incidental_correct_credit,
                    ).get("consolidation_value", 0.0)
                )

            # 6I-D: horizon_self_correct uses compute_horizon_g_exp
            _lg_mode = getattr(tcfg, 'tutor_lg_mode', 'off')
            if _lg_mode == 'horizon_self_correct':
                try:
                    from .sparse_tutor_scoring import compute_horizon_g_exp
                    g_exp = compute_horizon_g_exp(
                        self.cfg, qs, active, learner_probs,
                        compute_wait_tier_probs=lambda: base_probs,
                        spec=g_exp_spec, phase=_phase,
                    )
                except Exception:
                    pass
            if _consolidation_bonus_for_gexp > 0.0:
                g_exp += _consolidation_bonus_for_gexp
                g_exp_consolidation_bonus = float(_consolidation_bonus_for_gexp)

            # Final Q computation after all mode-specific overrides.
            q = (self.lambda_eval * g_eval
                 + self.lambda_exp * g_exp
                 - self.beta * p_death
                 - self.gamma * p_timeout
                 - self.lambda_shift * effective_d_shift
                 - cost + _phase_bias)
            if _postreveal_decomp is not None:
                q += self.lambda_exp * _cue_q

        detail = {
            "action": action,
            "tutor_mode": tutor_mode,
            "guard_passed": guard_passed,
            "u_learn": round(float(u_learn), 6),
            "u_teach": round(float(u_teach), 6),
            "g_eval": g_eval,
            "g_exp": g_exp,
            "g_exp_base": g_exp_base,
            "g_exp_consolidation_bonus": g_exp_consolidation_bonus,
            "g_exp_effective": g_exp,
            "p_success": round(float(p_success), 4),
            "p_death": p_death,
            "p_timeout": p_timeout,
            "p_death_proxy": p_death_proxy,
            "p_timeout_proxy": p_timeout_proxy,
            "used_rollout": use_ro,
            "d_shift": d_shift,
            "effective_d_shift": effective_d_shift,
            "phase_bias": locals().get("_phase_bias", 0.0),
            "cost": cost,
            "raw_cost": raw_cost,
            "q_use": q,
            "postreveal_value_mode": getattr(tcfg, "postreveal_value_mode", "legacy"),
            "postreveal_consolidation_enabled": bool(
                getattr(tcfg, "use_postreveal_consolidation_value", False)
            ),
        }
        self._attach_postreveal_diagnostics_to_detail(
            detail,
            qs,
            active,
            spec,
            base_probs,
            learner_probs,
            p_death=p_death,
            p_timeout=p_timeout,
        )
        q_use_consolidate_delta = 0.0
        if (
            tutor_mode == "current"
        ):
            q_use_consolidate_delta += float(self.lambda_exp) * float(
                detail.get("g_exp_consolidation_bonus", 0.0)
            )
            if getattr(qs, "post_reveal_phase", False) and action in ("HIGHLIGHT", "MIX"):
                q_use_consolidate_delta += float(self.lambda_exp) * float(
                    detail.get("postreveal_q_consolidate_delta", 0.0)
                )
        detail["q_use_with_consolidate"] = float(q)
        detail["q_use_consolidate_delta"] = float(q_use_consolidate_delta)
        detail["q_use_without_consolidate"] = float(q) - float(q_use_consolidate_delta)
        if action == "MIX":
            for key, value in spec.items():
                if key.startswith("mix_joint_") or key.startswith("mix_direct_"):
                    detail[key] = value
        return float(q), detail

    def _compute_q_rescue(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
        p_timeout_wait: float,
        _wait_probs: Optional[Any] = None,   # A+: pre-computed WAIT baseline
        _spec_probs: Optional[Any] = None,   # C:  pre-computed spec learner_probs
    ) -> Tuple[float, Dict]:
        """Compute Q_rescue for rescue mode candidates.

        Rescue objective (upgraded to learner-consistent model):
            Q_rescue(a) = lambda_to * delta_P_timeout(a)
                        + lambda_eval_res * G_eval(a)
                        - beta * P_death(a)
                        - lambda_shift_res * D_shift(a)
                        - c(a)

        Uses _compute_learner_probs() (not tier model) and _rollout_estimate()
        in rescue mode (always), since this is the highest-miscalibration regime.
        D_shift = JS(learner_probs_WAIT || learner_probs_action).
        """
        action = spec["action"]

        # A+/C: use pre-computed probs if supplied (same cache as _compute_q_use).
        learner_probs = (_spec_probs if _spec_probs is not None
                         else self._compute_learner_probs(qs, active, spec, learner))
        base_probs    = (_wait_probs if _wait_probs is not None
                         else self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner))

        # â”€â”€ G_eval (auxiliary; still runs probe/oracle if enabled) â”€
        g_eval = 0.0
        if action != "WAIT" and self.g_learn_mode != "none":
            g_eval = self._compute_g_eval(qs, spec, learner_probs, learner)

        # â”€â”€ Rollout for P_death and P_timeout (rescue: always) â”€â”€â”€â”€
        if self._use_rollout(mode='rescue', action=action):
            p_death, p_timeout_after, _ = self._rollout_estimate(
                qs, active, spec, learner)
        else:
            p_death = self._compute_p_death(qs, active, learner_probs)
            p_timeout_after = self._compute_p_timeout(qs, active, learner_probs)

        delta_p_timeout = p_timeout_wait - p_timeout_after  # positive = improvement

        # â”€â”€ D_shift (learner-consistent JS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        d_shift = _js_divergence(base_probs, learner_probs)

        # â”€â”€ Action cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cost = self._action_cost(spec)

        # â”€â”€ Q_rescue (no G_exp) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lambda_eval_res = self.lambda_eval * 0.5
        q = (self.lambda_to * delta_p_timeout
             + lambda_eval_res * g_eval
             - self.beta * p_death
             - self.lambda_shift_res * d_shift
             - cost)

        detail = {
            "action": action,
            "mode": "rescue",
            "g_eval": g_eval,
            "g_exp": 0.0,
            "delta_p_timeout": round(delta_p_timeout, 6),
            "p_timeout_wait": round(p_timeout_wait, 6),
            "p_timeout_after": round(p_timeout_after, 6),
            "p_death": p_death,
            "d_shift": d_shift,
            "cost": cost,
            "used_rollout": True,
            "q_use": q,
        }
        self._attach_postreveal_diagnostics_to_detail(
            detail,
            qs,
            active,
            spec,
            base_probs,
            learner_probs,
            p_death=p_death,
            p_timeout=p_timeout_after,
        )
        detail["q_use_with_consolidate"] = float(q)
        detail["q_use_consolidate_delta"] = 0.0
        detail["q_use_without_consolidate"] = float(q)
        return float(q), detail

    # â”€â”€ Probability models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _compute_learner_probs(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
    ) -> np.ndarray:
        """Route to predictor or direct learner access.

        When self._predictor is set, delegates to predictor.pick_dist().
        Otherwise falls through to _compute_learner_probs_direct().
        """
        return route_pick_distribution(
            self._predictor,
            qs,
            active,
            spec,
            learner,
            direct_fn=self._compute_learner_probs_direct,
        )

    def _compute_learner_probs_direct(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
    ) -> np.ndarray:
        """Learner-consistent pick distribution p^learner_a(j). [Option A]

        PRIVILEGED: reads learner._scorer, learner.policy.danger_head,
        learner.policy.attention directly.

        Matches the actual learner execution path in policy.compute_policy():
          U_j = Î±_sem * S_CLS(attention_weighted) - Î±_risk * Î¼_d - Î±_unc * u_d
          p  = softmax(Î²_L * U) with Îµ-lapse

        Under HIGHLIGHT: applies exp(rho_H) attention boost to highlighted cells,
          recomputes semantic scores with updated attention.  [Does NOT assume p(j*)=1]
        Under BAN: removes banned option from active set, renormalizes.
        Under MIX: BAN + HL combined.

        This replaces _compute_tier_probs() for ALL Q-value and D_shift computation.
        """
        return compute_direct_pick_probs(self.cfg, qs, active, spec, learner)

    def _compute_tier_probs(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
    ) -> np.ndarray:
        """[LEGACY] Tier-aware distribution. Kept for backward-compat ablation only.

        DEPRECATED for Q-value computation: use _compute_learner_probs() instead.
        Retained so that rollout_mode="proxy" code path can reference it.
        """
        action = spec["action"]
        ban_idx: Optional[int] = spec.get("ban_index")
        hl_cells: Optional[Tuple] = spec.get("highlight_cells")

        K = len(active)
        if K == 0:
            return np.array([])

        h_set: Set[int] = set()
        b_set: Set[int] = set()

        if action in ("HIGHLIGHT", "MIX") and hl_cells:
            for opt in active:
                if opt.is_correct:
                    h_set.add(opt.index)
                    break

        if action in ("BAN", "MIX") and ban_idx is not None:
            b_set.add(ban_idx)

        u = self._get_utilities(qs, active)
        beta_L = self.cfg.learner.beta_L

        def softmax_over(indices_in_active: List[int]) -> np.ndarray:
            if not indices_in_active:
                return np.array([])
            vals = np.array([u[i] for i in indices_in_active])
            vals = vals - vals.max()
            exp_v = np.exp(beta_L * vals)
            return exp_v / (exp_v.sum() + 1e-30)

        h_pos = [i for i, o in enumerate(active) if o.index in h_set]
        n_pos = [i for i, o in enumerate(active)
                 if o.index not in h_set and o.index not in b_set]
        b_pos = [i for i, o in enumerate(active) if o.index in b_set]

        p = np.zeros(K)
        if h_pos:
            sm = softmax_over(h_pos)
            for rank, idx in enumerate(h_pos):
                p[idx] = sm[rank]
        elif n_pos:
            sm = softmax_over(n_pos)
            for rank, idx in enumerate(n_pos):
                p[idx] = sm[rank]
        else:
            sm = softmax_over(b_pos)
            for rank, idx in enumerate(b_pos):
                p[idx] = sm[rank]
        return p

    def _compute_logit_probs(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
    ) -> np.ndarray:
        """Logit-surrogate probability distribution p^logit_a(j).

        Approximates tier-aware semantics via large logit perturbations:
          HIGHLIGHT(k): u_k += tutor_rho_H
          BAN(j):       u_j -= M_ban

        Used internally for fast Q_use planning (NOT for main-report MenuShift).
        """
        action = spec["action"]
        ban_idx: Optional[int] = spec.get("ban_index")
        hl_cells: Optional[Tuple] = spec.get("highlight_cells")

        K = len(active)
        if K == 0:
            return np.array([])

        u = self._get_utilities(qs, active).copy()
        beta_L = self.cfg.learner.beta_L

        if action in ("HIGHLIGHT", "MIX") and hl_cells:
            for i, opt in enumerate(active):
                if opt.is_correct:
                    u[i] += self.tutor_rho_H

        if action in ("BAN", "MIX") and ban_idx is not None:
            for i, opt in enumerate(active):
                if opt.index == ban_idx:
                    u[i] -= self.M_ban

        u_shifted = u - u.max()
        exp_u = np.exp(beta_L * u_shifted)
        return exp_u / (exp_u.sum() + 1e-30)

    def _get_utilities(self, qs: QueryState, active: List[Option]) -> np.ndarray:
        """Compute raw learner utility U_j for each active option.

        U_j = Î±_sem * S_CLS(j) - Î±_risk * Î¼_d(j) - Î±_unc * u_d(j)
        """
        K = len(active)
        lcfg = self.cfg.learner
        scorer = getattr(self._learner_ref, '_scorer', None) if hasattr(self, '_learner_ref') else None

        # Attention weights
        policy = getattr(self._learner_ref, 'policy', None) if hasattr(self, '_learner_ref') else None
        if policy is not None and policy.attention is not None:
            attn = policy.attention.weights
        else:
            L = len(qs.target_output)
            attn = np.ones(L) / max(L, 1)

        sem = np.zeros(K)
        danger = np.zeros(K)
        unc = np.zeros(K)

        for i, opt in enumerate(active):
            if scorer is not None:
                sem[i] = scorer.score_option(
                    qs.target_output, opt.text, attention_weights=attn
                )
            if policy is not None and policy.danger_head is not None:
                mu, u = policy.danger_head.predict(opt.danger_vec)
                danger[i] = mu
                unc[i] = u

        U = (lcfg.alpha_sem * sem
             - lcfg.alpha_risk * danger
             - lcfg.alpha_unc * unc)
        return U

    def _compute_pick_probs_for_opts(
        self,
        qs: QueryState,
        opts: List[Option],
        learner: LearnerAgent,
        use_logit: bool = True,
    ) -> np.ndarray:
        """Compute WAIT (no intervention) pick probs for a specific option list."""
        # Store learner ref for _get_utilities
        self._learner_ref = learner
        spec = {"action": "WAIT"}
        return self._compute_tier_probs(qs, opts, spec)

    # â”€â”€ Q_use sub-terms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _compute_g_eval(
        self,
        qs: QueryState,
        spec: Dict[str, Any],
        tier_probs: np.ndarray,
        learner: LearnerAgent,
    ) -> float:
        """G_eval via ProbeEvaluator or OracleSurrogate.

        Passes tier-aware p_a to _simulate_expected_reveals() via p_a parameter.
        Resolves feedback_mode from config so the underlying estimator simulates
        the correct update type (cortex restudy vs negative evidence).
        """
        scorer = getattr(learner, '_scorer', None)
        if scorer is None:
            return 0.0

        active = get_active_menu(qs)
        shortlist_indices = [o.index for o in active]

        # Resolve effective feedback mode
        fb_cfg = self.cfg.tutor.g_learn_feedback_mode
        if fb_cfg == "inherit":
            feedback_mode = self.cfg.env.feedback_mode
        else:
            feedback_mode = fb_cfg

        try:
            return self._g_learn.estimate(
                scorer, qs, shortlist_indices, learner,
                probe_queries=self._probe_queries or None,
                p_a=tier_probs,
                feedback_mode=feedback_mode,
            )
        except TypeError:
            # Fallback: g_learn doesn't support feedback_mode yet (legacy signature)
            return self._g_learn.estimate(
                scorer, qs, shortlist_indices, learner,
                probe_queries=self._probe_queries or None,
            )

    def _compute_g_exp(
        self,
        qs: QueryState,
        active: List[Option],
        tier_probs: np.ndarray,
        spec: Optional[Dict[str, Any]] = None,
        wait_probs: Optional[np.ndarray] = None,
    ) -> float:
        """Safe exposure gain G_exp.

        Phase 6G/6H extensions:
          tutor_lg_mode = "off"          : original formula (backward-compat)
          tutor_lg_mode = "diagnostic"   : bonus for diagnostic wrongs, penalty for lures
          tutor_lg_mode = "safety_only"  : returns 0.0 (optimize risk only)
          tutor_lg_mode = "learning_only": includes all wrongs (ignore risk)
          tutor_lg_mode = "self_correct" : ALLOW/CONSOLIDATE trajectory-aware planning

        For WAIT / BAN:
            G_exp = Î£_{j â‰  j*, j safe} p_tier(j)
            Uses tier_probs directly. BAN suppresses the banned option.

        For HIGHLIGHT / MIX:
            Residual estimate using WAIT baseline probs.
        """
        return compute_sparse_g_exp(
            self.cfg,
            qs,
            active,
            tier_probs,
            compute_wait_tier_probs=lambda: (
                wait_probs if wait_probs is not None else self._compute_tier_probs(
                    qs, active, {"action": "WAIT"}
                )
            ),
            spec=spec,
        )

    def _compute_p_death(
        self,
        qs: QueryState,
        active: List[Option],
        probs: np.ndarray,
    ) -> float:
        """Single-step P(death) proxy: sum_j p(j) * 1[j lethal, j != j*].

        Works with learner-consistent probs from _compute_learner_probs().
        Note: pick(j*) => immediate success, no damage.  Only non-correct lethal options count.
        """
        return compute_p_death_proxy(qs, active, probs)

    def _compute_p_timeout(
        self,
        qs: QueryState,
        active: List[Option],
        probs: np.ndarray,
    ) -> float:
        """Geometric P(timeout) proxy: (1 - p(j*))^tau_t.

        Used as cheap first-pass estimate. Works with learner-consistent probs.
        NOTE: This underestimates timeout because it ignores:
          - CLS posterior updates after reveals (p(j*) rises over rounds)
          - HP depletion causing early death termination (competing risk)
          - epsilon lapse dynamics
        Use _rollout_estimate() for calibrated estimates in rescue/boundary cases.
        """
        return compute_p_timeout_proxy(qs, active, probs)

    def _use_rollout(
        self,
        mode: str,
        action: str,
        q_wait: Optional[float] = None,
        p_timeout_proxy: Optional[float] = None,
    ) -> bool:
        """Gate: should we use rollout for this candidate?

        mode='rescue': always True (highest miscalibration zone)
        mode='normal': True when:
          - rollout_mode='full', OR
          - rollout_mode='hybrid' AND (
              action is non-WAIT AND (decision boundary close OR high-risk proxy)
            )
        rollout_mode='proxy': always False (backward-compat)
        """
        return should_use_rollout(
            self.rollout_mode,
            mode,
            action,
            q_wait=q_wait,
            p_timeout_proxy=p_timeout_proxy,
        )

    def _rollout_estimate(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
        n_override: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Route to predictor or direct learner rollout.

        When self._predictor is set, delegates to predictor.rollout().
        Otherwise falls through to _rollout_estimate_direct().
        """
        N = n_override if n_override is not None else self.rollout_n
        return route_rollout_estimate(
            self._predictor,
            qs,
            active,
            spec,
            learner,
            N,
            direct_fn=self._rollout_estimate_direct,
        )

    def _rollout_estimate_direct(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
        n_override: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Short learner rollout for calibrated (P_death, P_timeout, P_success).

        PRIVILEGED: deepcopies learner._scorer and learner.policy.danger_head.

        Uses the actual learner scoring path (scorer + danger_head + attention)
        to simulate N independent full queries from the current state.

        State copied: QueryState (hp, rounds_used, menu, highlighted_cells, banned_indices),
          scorer (CLS weights), danger_head (hazard+severity bayesian state),
          attention weights (from policy.attention).

        Under HIGHLIGHT: applies exp(rho_H) attention boost before each pick,
          consistent with learner_agent.act() behavior.
        Under BAN: banned option excluded from picks.
        Learner picks via _compute_learner_probs_direct() â€” same formula as real policy.

        Returns: (p_death, p_timeout, p_success) empirical means over N rollouts.
        """
        N = n_override if n_override is not None else self.rollout_n
        return rollout_estimate_direct(self.cfg, qs, active, spec, learner, N)

    def _estimate_p_timeout_wait(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
    ) -> float:
        """Estimate P_timeout under WAIT (no intervention) for HL gate.

        Uses learner-consistent probs (Option A) instead of tier model.
        """
        self._learner_ref = learner
        wait_probs = self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
        return self._compute_p_timeout(qs, active, wait_probs)

    def _action_cost(self, spec: Dict[str, Any]) -> float:
        """c(a) = c_I * (number of active intervention slots)."""
        action = spec["action"]
        if action == "WAIT":
            return 0.0
        elif action in ("BAN", "HIGHLIGHT"):
            return self.c_I
        elif action == "MIX":
            return 2.0 * self.c_I
        return 0.0

    def _compute_u_teach(
        self,
        p_success: float,
        p_death: float,
        p_timeout: float,
    ) -> float:
        """Short-term teaching utility U_teach.

        U_teach = w_succ * p_success
                - w_death_teach * p_death
                - w_tout_teach  * p_timeout

        Uses rollout-estimated p_success (callee responsibility to force
        rollout in dual-mode). In current mode, p_success may be 0.0.
        """
        tcfg = self.cfg.tutor
        return (
            getattr(tcfg, 'w_succ',        1.0) * float(p_success)
            - getattr(tcfg, 'w_death_teach', 0.5) * float(p_death)
            - getattr(tcfg, 'w_tout_teach',  0.2) * float(p_timeout)
        )

    def _eval_non_regression_guard(self, u_learn: float) -> bool:
        """True if G_eval(a) >= -eps_eval_guard (eval non-regression satisfied).

        Used by protective mode in _compute_q_use; exposed here for testing.
        """
        eps = getattr(self.cfg.tutor, 'eps_eval_guard', 0.01)
        return float(u_learn) >= -eps

    # â”€â”€ Block runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_block(
        self,
        env: OptionEnv,
        learner: LearnerAgent,
        task_id: str,
        seed: int = 42,
    ) -> BlockState:
        """Run a full block with sparse tutor + learner.

        Same interface as OptionLevelTutorAgent.run_block() for drop-in comparison.
        """
        block = env.reset_block(task_id, seed=seed)
        support, _, grammar = env.adapter.load_task(task_id)

        self.init_block(block, grammar, support)
        block._decision_trace = self._decision_trace
        learner.init_block(block, grammar, support)

        # Store learner ref for utility computation
        self._learner_ref = learner

        max_steps = len(block.queries) * 20
        steps = 0
        while not block.done and steps < max_steps:
            steps += 1
            qs = block.current_query
            if qs is None or qs.done:
                break

            # Tutor acts first
            self.act(block, env, learner_agent=learner)

            if qs.done:
                continue

            # Learner acts
            learner.act(block, env)

            # Feed public observation to predictor (if set)
            if self._predictor is not None:
                from .observation_adapter import ObservationAdapter
                latest = ObservationAdapter().extract_latest(block)
                if latest is not None:
                    self._predictor.observe(latest)

        if not block.done:
            block.done = True

        # Backfill outcomes into decision trace
        self.finalize_trace(block)
        block._productive_allow_diagnostic_failures = getattr(
            self, "_productive_allow_diagnostic_failures", 0
        )

        return block

    # â”€â”€ Decision Trace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def finalize_trace(self, block: BlockState) -> None:
        """Backfill outcome into each trace entry after block completion."""
        qid_to_qs = {q.query_id: q for q in block.queries}
        for entry in self._decision_trace:
            qid = entry.get("query_id")
            qs = qid_to_qs.get(qid)
            if qs is not None:
                entry["outcome"] = {
                    "success": qs.success,
                    "death": qs.hp <= 0 and not qs.success,
                    "timeout": qs.hp > 0 and not qs.success,
                    "n_reveals": len(qs.reveal_history) if hasattr(qs, 'reveal_history') else 0,
                }

    def get_decision_trace(self) -> List[dict]:
        """Return the full per-query decision trace (for JSON dump)."""
        return self._decision_trace

    def _extract_trace_summary(self) -> dict:
        """Extract lightweight aggregate diagnostics from decision trace.

        Always available (no --trace flag needed). Used by exp runner.

        Returns dict with:
            ban_generated_rate: fraction of traced queries where BAN was generated
            hl_generated_rate: fraction where HIGHLIGHT was generated
            hl_suppressed_rate: fraction where HL was suppressed by gate
            nonwait_beats_wait_rate: fraction where any non-WAIT Q > Q_wait
            mean_best_nonwait_margin: mean of max(Q_nonwait) - Q_wait
            mean_hl_gate_value: mean p_timeout_wait across traced queries
            rescue_trigger_rate: fraction of queries that entered rescue mode
            mean_delta_p_timeout: mean P_timeout reduction from chosen action
            timeout_blocker_selected_rate: fraction of rescue queries with blocker BAN
            highlight_in_rescue_rate: fraction of rescue queries with HIGHLIGHT
        """
        n = len(self._decision_trace)
        if n == 0:
            return {
                "ban_generated_rate": 0.0,
                "hl_generated_rate": 0.0,
                "hl_suppressed_rate": 0.0,
                "nonwait_beats_wait_rate": 0.0,
                "mean_best_nonwait_margin": 0.0,
                "mean_hl_gate_value": 0.0,
                "rescue_trigger_rate": 0.0,
                "mean_delta_p_timeout": 0.0,
                "timeout_blocker_selected_rate": 0.0,
                "highlight_in_rescue_rate": 0.0,
            }

        ban_gen = 0
        hl_gen = 0
        hl_supp = 0
        nonwait_wins = 0
        margins = []
        hl_gate_vals = []
        # Rescue-specific
        rescue_count = 0
        delta_p_timeouts = []
        blocker_selected = 0
        hl_in_rescue = 0

        for entry in self._decision_trace:
            gen = entry.get("generation", {})
            scoring = entry.get("scoring", {})
            mode = entry.get("mode", "learn")
            is_rescue = (mode == "rescue")

            if gen.get("ban_generated"):
                ban_gen += 1
            if gen.get("hl_generated"):
                hl_gen += 1
            if gen.get("hl_suppressed_by_gate"):
                hl_supp += 1
            hl_gate_vals.append(gen.get("hl_gate_value", 0.0))

            q_wait = scoring.get("q_wait", 0.0)
            cands = scoring.get("candidates", [])
            nonwait_qs = [c.get("q_use", float('-inf'))
                          for c in cands if c.get("action") != "WAIT"]
            if nonwait_qs:
                best_nw = max(nonwait_qs)
                margins.append(best_nw - q_wait)
                if best_nw > q_wait:
                    nonwait_wins += 1

            # Rescue diagnostics
            if is_rescue:
                rescue_count += 1
                # delta_p_timeout from trace entry
                dpt = entry.get("delta_p_timeout", 0.0)
                delta_p_timeouts.append(float(dpt))
                # blocker selected?
                if gen.get("timeout_blocker_idx") is not None:
                    blocker_selected += 1
                # HIGHLIGHT in rescue?
                chosen = entry.get("chosen_action", "WAIT")
                if chosen in ("HIGHLIGHT", "MIX"):
                    hl_in_rescue += 1

        rescue_rate = rescue_count / n
        return {
            "ban_generated_rate":       round(ban_gen / n, 4),
            "hl_generated_rate":        round(hl_gen / n, 4),
            "hl_suppressed_rate":       round(hl_supp / n, 4),
            "nonwait_beats_wait_rate":  round(nonwait_wins / n, 4),
            "mean_best_nonwait_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
            "mean_hl_gate_value":       round(float(np.mean(hl_gate_vals)), 4) if hl_gate_vals else 0.0,
            # Rescue-specific
            "rescue_trigger_rate":          round(rescue_rate, 4),
            "mean_delta_p_timeout":         round(float(np.mean(delta_p_timeouts)), 4) if delta_p_timeouts else 0.0,
            "timeout_blocker_selected_rate": round(blocker_selected / rescue_count, 4) if rescue_count > 0 else 0.0,
            "highlight_in_rescue_rate":     round(hl_in_rescue / rescue_count, 4) if rescue_count > 0 else 0.0,
        }

    # â”€â”€ Diagnostics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_block_summary(self, block: BlockState) -> dict:
        """Compute sparse tutor diagnostics for a completed block."""
        ban_steps = [t for t in block.tutor_trace if t.action in ("BAN", "MIX")]
        hl_steps  = [t for t in block.tutor_trace if t.action in ("HIGHLIGHT", "MIX")]
        mix_steps = [t for t in block.tutor_trace if t.action == "MIX"]
        wait_steps = [t for t in block.tutor_trace if t.action == "WAIT"]

        # Q_use detail stats
        q_details = [t.q_use_detail for t in block.tutor_trace
                     if t.q_use_detail is not None]
        avg_d_shift = (sum(d.get("d_shift", 0.0) for d in q_details) / len(q_details)
                       if q_details else 0.0)

        return {
            "n_ban": len(ban_steps),
            "n_highlight": len(hl_steps),
            "n_mix": len(mix_steps),
            "n_wait": len(wait_steps),
            "avg_d_shift": round(avg_d_shift, 4),
            "total_correct": block.total_correct,
            "solve_rate": block.total_correct / max(len(block.queries), 1),
            "total_damage": block.total_damage,
        }

