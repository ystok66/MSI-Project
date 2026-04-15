"""
sparse_tutor.py — Bayes Gate Tutor with sparse BAN / HIGHLIGHT / MIX actions.

Replaces SHORTLIST as the primary teaching intervention.

Design:
  - Action space: WAIT | BAN(j) | HIGHLIGHT(cells) | MIX(j, cells)
  - Probability models are now LEARNER-CONSISTENT (Option A upgrade):
      _compute_learner_probs(): uses actual learner scoring path
        (scorer.score_option with attention-weighted mismatch + danger_head.predict)
        Under HIGHLIGHT: re-applies attention boost exp(rho_H) to highlighted cells
        then recomputes semantic scores — matches learner_agent.act() exactly.
        Under BAN: excludes banned option from active menu.
        NO hard-tier p(j*)=1 assumption.
      _rollout_estimate(): short N-step rollouts using learner scoring for
        calibrated multi-step P_death / P_timeout.
  - Q_use = λ_eval·G_eval + λ_exp·G_exp
            - β·P_death - γ·P_timeout
            - λ_shift·D_shift - c(a)
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

import copy
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np

from ..config import FullConfig
from ..env.state import BlockState, QueryState
from ..env.option_env import OptionEnv
from ..env.interventions import get_active_menu
from ..interfaces import TutorStep, Option
from ..learner.learner_agent import LearnerAgent


# ── JS divergence helper ──────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────


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
    ):
        self.cfg = cfg or FullConfig()
        tcfg = self.cfg.tutor

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

        # ── Rescue mode parameters ────────────────────────────────
        # theta_rescue: P_timeout(WAIT) threshold that triggers rescue mode
        # lambda_to:    weight on delta_P_timeout in Q_rescue
        # lambda_shift_res: shift penalty in rescue mode (lower than learning mode)
        self.theta_rescue = getattr(tcfg, 'theta_rescue', 0.5)
        self.lambda_to = getattr(tcfg, 'lambda_to', 1.0)
        self.lambda_shift_res = getattr(tcfg, 'lambda_shift_res', self.lambda_shift * 0.5)

        # ── Rollout proxy parameters ──────────────────────────────
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

    # ── Block init ────────────────────────────────────────────────

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize per-block state."""
        self._decision_trace = []

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

    # ── Main entry ────────────────────────────────────────────────

    def act(
        self,
        block: BlockState,
        env: OptionEnv,
        learner_agent: Optional[LearnerAgent] = None,
    ) -> TutorStep:
        """Execute one tutor turn.

        Obs / Eval phase: always WAIT.
        Teaching phase: enumerate sparse candidates → argmax Q_use.
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return env.tutor_act(block, "WAIT")

        if block.in_observation_phase or block.in_evaluation_phase:
            return env.tutor_act(block, "WAIT")

        # One intervention per query (BAN/MIX persist; don't double-intervene)
        if qs.banned_indices or qs.highlighted_cells:
            return env.tutor_act(block, "WAIT")

        if learner_agent is None:
            return env.tutor_act(block, "WAIT")

        return self._act_teaching(block, env, learner_agent)

    def _act_teaching(
        self,
        block: BlockState,
        env: OptionEnv,
        learner: LearnerAgent,
    ) -> TutorStep:
        """Teaching-phase decision: dual-mode (learning vs rescue).

        Gate:
            if P_timeout(WAIT) > theta_rescue  →  rescue mode (deadline reduction)
            else                               →  learning mode (teaching Q_use)

        Records two-layer decision trace with mode field.
        """
        qs = block.current_query
        active = get_active_menu(qs)
        if not active:
            return env.tutor_act(block, "WAIT")

        # ── Compute WAIT baseline bundle — once per _act_teaching() ─
        # A+: compute wait_probs_lc / p_death_wait / p_timeout_wait ONCE.
        # All candidate scorers reference this shared bundle:
        #   _compute_q_use  → base_probs, D_shift reference point
        #   pedagogical     → dynamic d_max / t_max thresholds
        #   rescue gate     → learner-consistent (already done)
        #   p_correct_wait  → fraction of prob mass on correct option
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

        # ── Layer 1: Enumerate candidates with gate reasons ──────
        candidates, gen_info = self._enumerate_candidates_traced(
            qs, active, learner, mode=mode, p_timeout_wait=p_timeout_wait
        )

        # ── Layer 2: Score all candidates ────────────────────────
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
            # ── Guard filter (dual-mode only) ────────────────────
            # In protective mode: G_eval guard must pass.
            # In pedagogical mode: P_death/P_timeout hard constraints must pass.
            # WAIT always passes. Current mode: guard always True.
            guard_ok = detail.get('guard_passed', True)
            if q > best_q and guard_ok:
                best_q = q
                best_action_spec = spec
                best_action_spec["_q_detail"] = detail

        # ── Compute margin_vs_wait for each candidate ────────────
        for sc in scored_candidates:
            sc["margin_vs_wait"] = round(sc["q_use"] - q_wait, 6)

        # ── Compute p(j* | WAIT) ─────────────────────────────────
        p_correct_wait = 0.0
        n_lethal = 0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_correct_wait = float(wait_probs_lc[i])  # learner-consistent
            if opt.risk_class >= qs.hp and not opt.is_correct:
                n_lethal += 1

        # ── Compute p_timeout_after (chosen action) ──────────────
        # C: chosen action probs already in cache — no extra call needed.
        chosen_spec = best_action_spec
        chosen_probs_lc = _cached_learner_probs(chosen_spec)
        p_timeout_after = self._compute_p_timeout(qs, active, chosen_probs_lc)
        delta_p_timeout = round(p_timeout_wait - p_timeout_after, 6)

        # ── Record trace entry ───────────────────────────────────
        trace_entry = {
            "query_id": qs.query_id,
            "hp": qs.hp,
            "tau_remaining": max(0, qs.max_rounds - qs.rounds_used),
            "n_active": len(active),
            "n_lethal": n_lethal,
            "mode": mode,
            "p_timeout_wait": round(p_timeout_wait, 4),
            "p_timeout_after": round(p_timeout_after, 4),
            "delta_p_timeout": delta_p_timeout,
            "generation": gen_info,
            "scoring": {
                "q_wait": round(q_wait, 6),
                "candidates": scored_candidates,
            },
            "chosen_action": best_action_spec["action"],
            "p_correct_wait": round(p_correct_wait, 4),
            "outcome": None,  # filled in finalize_trace()
        }
        self._decision_trace.append(trace_entry)

        # ── Execute chosen action ────────────────────────────────
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
        return step

    # ── Candidate generation ──────────────────────────────────────

    def _enumerate_candidates_traced(
        self,
        qs: QueryState,
        active: List[Option],
        learner: LearnerAgent,
        mode: str = "learn",
        p_timeout_wait: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate candidates with Layer-1 gate diagnostics, mode-aware.

        In rescue mode:
          - BAN target is selected by BlockScore (timeout blocker), not danger
          - HIGHLIGHT is always generated (rescue gate bypasses hl_timeout_threshold)
          - MIX = timeout_blocker_BAN + HIGHLIGHT(j*)

        Returns:
            (candidates, generation_info)
        """
        candidates: List[Dict[str, Any]] = [{"action": "WAIT"}]

        gen_info: Dict[str, Any] = {
            "mode": mode,
            "ban_generated": False,
            "ban_gate_reason": "no_non_correct",
            "ban_target_idx": None,
            "ban_target_risk": None,
            "ban_target_pick_prob": None,
            "ban_is_timeout_blocker": mode == "rescue",
            "hl_generated": False,
            "hl_gate_reason": "no_correct_option",
            "hl_gate_value": round(p_timeout_wait, 4),
            "hl_gate_threshold": self.hl_timeout_threshold,
            "hl_suppressed_by_gate": False,
            "mix_generated": False,
            # Rescue-specific
            "timeout_blocker_idx": None,
            "timeout_blocker_score": None,
        }

        correct_opts = [o for o in active if o.is_correct]
        non_correct = [o for o in active if not o.is_correct]

        if not correct_opts:
            gen_info["ban_gate_reason"] = "no_correct_option"
            return candidates, gen_info

        j_star = correct_opts[0]

        # ── Ban target selection (mode-aware) ─────────────────────
        if not non_correct:
            gen_info["ban_gate_reason"] = "no_non_correct"
        else:
            if mode == "rescue":
                ban_target, blocker_score = self._select_timeout_blocker(
                    qs, active, j_star, non_correct, learner
                )
                gen_info["timeout_blocker_idx"] = (
                    ban_target.index if ban_target else None
                )
                gen_info["timeout_blocker_score"] = (
                    round(blocker_score, 4) if blocker_score is not None else None
                )
            else:
                ban_target = self._select_ban_target(qs, non_correct, learner)

            if ban_target is None:
                gen_info["ban_gate_reason"] = "selection_failed"
            else:
                gen_info["ban_generated"] = True
                gen_info["ban_gate_reason"] = "ok"
                gen_info["ban_target_idx"] = ban_target.index
                gen_info["ban_target_risk"] = ban_target.risk_class
                # Record pick prob of ban target under WAIT
                # Use wait_probs_lc from outer scope (already computed) to avoid extra call.
                try:
                    for ai, ao in enumerate(active):
                        if ao.index == ban_target.index:
                            gen_info["ban_target_pick_prob"] = round(float(wait_probs_lc[ai]), 4)
                            break
                except Exception:
                    pass
                candidates.append({
                    "action": "BAN",
                    "ban_index": ban_target.index,
                })

        # ── Highlight gate (mode-aware) ───────────────────────────
        hl_cells = self._select_highlight_cells(qs, j_star, learner)

        if hl_cells is None:
            gen_info["hl_gate_reason"] = "no_cells"
        elif mode == "rescue":
            # Rescue mode: always generate HIGHLIGHT (timeout reduction is the goal)
            gen_info["hl_generated"] = True
            gen_info["hl_gate_reason"] = "rescue_forced"
            candidates.append({
                "action": "HIGHLIGHT",
                "highlight_cells": hl_cells,
            })
            if gen_info["ban_generated"]:
                gen_info["mix_generated"] = True
                candidates.append({
                    "action": "MIX",
                    "ban_index": gen_info["ban_target_idx"],
                    "highlight_cells": hl_cells,
                })
        elif p_timeout_wait <= self.hl_timeout_threshold:
            gen_info["hl_gate_reason"] = "p_timeout_below_threshold"
            gen_info["hl_suppressed_by_gate"] = True
        else:
            gen_info["hl_generated"] = True
            gen_info["hl_gate_reason"] = "ok"
            candidates.append({
                "action": "HIGHLIGHT",
                "highlight_cells": hl_cells,
            })
            if gen_info["ban_generated"]:
                gen_info["mix_generated"] = True
                ban_idx = gen_info["ban_target_idx"]
                candidates.append({
                    "action": "MIX",
                    "ban_index": ban_idx,
                    "highlight_cells": hl_cells,
                })

        return candidates, gen_info

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
        """
        if not non_correct:
            return None

        hp = qs.hp
        lethal = [o for o in non_correct if o.risk_class >= hp]
        pool = lethal if lethal else non_correct

        # Score by pick_prob (confusion)
        try:
            probs = self._compute_pick_probs_for_opts(qs, pool, learner,
                                                      use_logit=True)
            idx = int(np.argmax(probs))
            return pool[idx]
        except Exception:
            return pool[0]

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
                if p_j > p_jstar:  # ranked strictly above j* → most likely to block
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

    def _select_highlight_cells(
        self,
        qs: QueryState,
        j_star: Option,
        learner: LearnerAgent,
    ) -> Optional[Tuple[int, ...]]:
        """Select output cells to HIGHLIGHT for j*.

        Returns 1-2 cells where j*'s rendered output differs from menu average,
        indicating the most diagnostic cells for distinguishing j*.
        Falls back to (0,) if rendering unavailable.
        """
        max_cells = self.cfg.tutor.max_highlight_cells
        L = len(qs.target_output)
        if L == 0:
            return None

        # Simple heuristic: highlight the first max_cells cells
        # (Richer version: compare j*.rendered_output vs distractors at each cell)
        cells = tuple(range(min(max_cells, L)))
        return cells if cells else None

    # ── Q_use computation ─────────────────────────────────────────

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
              Q_use = λ_eval*G_eval + λ_exp*G_exp
                    - β*P_death - γ*P_timeout - λ_shift*D_shift - c(a)
              guard_passed = True always.

          "protective":
              Q_score = U_teach(a) - λ_shift*D_shift - c(a)
              guard_passed = (G_eval(a) >= -eps_eval_guard)  [or g_learn=none]
              Rollout forced for non-WAIT to get accurate p_success.

          "pedagogical":
              Q_score = G_eval(a) + η*U_teach(a) - λ_shift*D_shift - c(a)
              guard_passed = P_death(a) ≤ d_max AND P_timeout(a) ≤ t_max
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

        # ── Learner-consistent probability distributions ───────────
        # C: use pre-computed spec probs if supplied (request-scoped cache).
        # A+: use pre-computed WAIT probs if supplied (avoids repeated beam search).
        learner_probs = (_spec_probs if _spec_probs is not None
                         else self._compute_learner_probs(qs, active, spec, learner))
        base_probs    = (_wait_probs if _wait_probs is not None
                         else self._compute_learner_probs(qs, active, {"action": "WAIT"}, learner))

        # ── G_eval (ProbeEvaluator / OracleSurrogate) ────────────
        g_eval = 0.0
        if action != "WAIT" and self.g_learn_mode != "none":
            g_eval = self._compute_g_eval(qs, spec, learner_probs, learner)

        # G_exp (safe exposure gain; only used in "current" mode)
        feedback_mode = self.cfg.env.feedback_mode
        g_exp = self._compute_g_exp(qs, active, learner_probs, spec)
        if feedback_mode == "nonreveal":
            g_exp = 0.0

        # ── P_death / P_timeout — proxy (cheap, always computed) ──
        p_death_proxy = self._compute_p_death(qs, active, learner_probs)
        p_timeout_proxy = self._compute_p_timeout(qs, active, learner_probs)

        # ── Rollout refinement ────────────────────────────────────
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

        # ── D_shift (JS on learner-consistent distributions) ──────
        d_shift = _js_divergence(base_probs, learner_probs)

        # ── Action cost ───────────────────────────────────────────
        cost = self._action_cost(spec)

        # ── U_teach (shared component) ────────────────────────────
        u_teach = self._compute_u_teach(p_success, p_death, p_timeout)

        # ── Dual-mode Q computation ───────────────────────────────
        u_learn = g_eval   # alias: long-term learning signal

        if tutor_mode == 'protective':
            # Q = U_teach - D_shift - cost
            # Guard: filter any action where G_eval < -eps_eval_guard.
            # Exception: WAIT always passes (reference point).
            # Exception: g_learn=none → no eval signal → guard bypassed.
            if action == 'WAIT':
                guard_passed = True
            elif self.g_learn_mode == 'none':
                guard_passed = True  # no eval info → can't guard
            else:
                guard_passed = (u_learn >= -getattr(tcfg, 'eps_eval_guard', 0.01))
            q = u_teach - self.lambda_shift * d_shift - cost

        elif tutor_mode == 'pedagogical':
            # Q = G_eval + η*U_teach - D_shift - cost
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

        else:  # "current" — original formula, full backward-compat
            guard_passed = True
            q = (self.lambda_eval * g_eval
                 + self.lambda_exp * g_exp
                 - self.beta * p_death
                 - self.gamma * p_timeout
                 - self.lambda_shift * d_shift
                 - cost)

        detail = {
            "action": action,
            "tutor_mode": tutor_mode,
            "guard_passed": guard_passed,
            "u_learn": round(float(u_learn), 6),
            "u_teach": round(float(u_teach), 6),
            "g_eval": g_eval,
            "g_exp": g_exp,
            "p_success": round(float(p_success), 4),
            "p_death": p_death,
            "p_timeout": p_timeout,
            "p_death_proxy": p_death_proxy,
            "p_timeout_proxy": p_timeout_proxy,
            "used_rollout": use_ro,
            "d_shift": d_shift,
            "cost": cost,
            "q_use": q,
        }
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

        # ── G_eval (auxiliary; still runs probe/oracle if enabled) ─
        g_eval = 0.0
        if action != "WAIT" and self.g_learn_mode != "none":
            g_eval = self._compute_g_eval(qs, spec, learner_probs, learner)

        # ── Rollout for P_death and P_timeout (rescue: always) ────
        if self._use_rollout(mode='rescue', action=action):
            p_death, p_timeout_after, _ = self._rollout_estimate(
                qs, active, spec, learner)
        else:
            p_death = self._compute_p_death(qs, active, learner_probs)
            p_timeout_after = self._compute_p_timeout(qs, active, learner_probs)

        delta_p_timeout = p_timeout_wait - p_timeout_after  # positive = improvement

        # ── D_shift (learner-consistent JS) ───────────────────────
        d_shift = _js_divergence(base_probs, learner_probs)

        # ── Action cost ───────────────────────────────────────────
        cost = self._action_cost(spec)

        # ── Q_rescue (no G_exp) ───────────────────────────────────
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
        return float(q), detail

    # ── Probability models ────────────────────────────────────────

    def _compute_learner_probs(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
    ) -> np.ndarray:
        """Learner-consistent pick distribution p^learner_a(j). [Option A]

        Matches the actual learner execution path in policy.compute_policy():
          U_j = α_sem * S_CLS(attention_weighted) - α_risk * μ_d - α_unc * u_d
          p  = softmax(β_L * U) with ε-lapse

        Under HIGHLIGHT: applies exp(rho_H) attention boost to highlighted cells,
          recomputes semantic scores with updated attention.  [Does NOT assume p(j*)=1]
        Under BAN: removes banned option from active set, renormalizes.
        Under MIX: BAN + HL combined.

        This replaces _compute_tier_probs() for ALL Q-value and D_shift computation.
        """
        action = spec["action"]
        ban_idx: Optional[int] = spec.get("ban_index")
        hl_cells: Optional[Tuple] = spec.get("highlight_cells")

        K_full = len(active)
        if K_full == 0:
            return np.array([])

        # ── Attention weights under this action ───────────────────
        # Start from current learner attention (or uniform if not set)
        lcfg = self.cfg.learner
        policy = getattr(learner, 'policy', None)
        if policy is not None and policy.attention is not None:
            attn = policy.attention.weights.copy()  # (L,)
        else:
            L = len(qs.target_output)
            attn = np.ones(L) / max(L, 1)

        # Apply HL attention boost (matches apply_highlight in attention_model.py)
        if action in ("HIGHLIGHT", "MIX") and hl_cells:
            attn = attn.copy()
            for ell in hl_cells:
                if 0 <= ell < len(attn):
                    attn[ell] *= np.exp(lcfg.rho_H)
            s = attn.sum()
            if s > 0:
                attn = attn / s

        # ── Build active subset (exclude banned) ──────────────────
        if action in ("BAN", "MIX") and ban_idx is not None:
            active_sub = [o for o in active if o.index != ban_idx]
        else:
            active_sub = list(active)

        K = len(active_sub)
        if K == 0:
            return np.zeros(K_full)

        # ── Scorer and danger head from learner ───────────────────
        scorer = getattr(learner, '_scorer', None)
        danger_head = policy.danger_head if policy is not None else None

        sem = np.zeros(K)
        mu_d = np.zeros(K)
        u_d = np.zeros(K)

        for i, opt in enumerate(active_sub):
            if scorer is not None:
                sem[i] = scorer.score_option(
                    qs.target_output, opt.text, attention_weights=attn)
            if danger_head is not None:
                mu, u = danger_head.predict(opt.danger_vec)
                mu_d[i] = mu
                u_d[i] = u

        U = (lcfg.alpha_sem * sem
             - lcfg.alpha_risk * mu_d
             - lcfg.alpha_unc * u_d)

        # Softmax with ε-lapse (matches policy._softmax_with_lapse)
        U_shifted = U - U.max()
        exp_u = np.exp(lcfg.beta_L * U_shifted)
        probs_sub = exp_u / (exp_u.sum() + 1e-30)
        eps = lcfg.epsilon
        probs_sub = (1 - eps) * probs_sub + eps / K
        probs_sub = np.clip(probs_sub, 0, 1)
        probs_sub /= probs_sub.sum()

        # Map back to full active vector (banned option gets prob 0)
        p = np.zeros(K_full)
        sub_idx_map = {o.index: i for i, o in enumerate(active_sub)}
        for j, o in enumerate(active):
            if o.index in sub_idx_map:
                p[j] = probs_sub[sub_idx_map[o.index]]
        return p

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

        U_j = α_sem * S_CLS(j) - α_risk * μ_d(j) - α_unc * u_d(j)
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

    # ── Q_use sub-terms ───────────────────────────────────────────

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
    ) -> float:
        """Safe exposure gain G_exp.

        For WAIT / BAN:
            G_exp = Σ_{j ≠ j*, j safe} p_tier(j)
            Uses tier_probs directly. BAN suppresses the banned option.

        For HIGHLIGHT / MIX:
            tier_probs has p(j*) ≈ 1.0 (j* in H tier), so the standard
            formula gives G_exp ≈ 0. This is a modelling artefact:
            HIGHLIGHT immediately solves the query, but doesn't destroy
            all exploration value—the learner still needs to distinguish
            options in other rounds.

            Residual estimate: use WAIT baseline probs, scaled by
            (1 - p_correct_wait) to credit only the fraction of exposure
            potential that HIGHLIGHT doesn't consume:

                wait_p = _compute_tier_probs(WAIT)
                p_correct_wait = wait_p[j*]
                g_exp_hl = (1 - p_correct_wait) * Σ_{j≠j*, j safe} wait_p[j]

            Intuition: if j* is easy under WAIT (high p_correct_wait),
            HIGHLIGHT costs little exploration; if j* is hard, HIGHLIGHT
            saves the learner but we discount the residual exposure.
        """
        if len(tier_probs) == 0 or len(tier_probs) != len(active):
            return 0.0

        action = (spec or {}).get("action", "WAIT")
        is_hl_action = action in ("HIGHLIGHT", "MIX")

        hp = qs.hp

        if is_hl_action:
            # Use WAIT probs for residual g_exp estimation
            wait_probs = self._compute_tier_probs(qs, active, {"action": "WAIT"})
            p_correct_wait = 0.0
            safe_wait_total = 0.0
            for i, opt in enumerate(active):
                if opt.is_correct:
                    p_correct_wait = float(wait_probs[i])
                elif opt.risk_class < hp:  # safe non-correct
                    safe_wait_total += float(wait_probs[i])
            return max(0.0, (1.0 - p_correct_wait) * safe_wait_total)
        else:
            # Standard: use tier_probs (works for WAIT and BAN)
            total = 0.0
            for i, opt in enumerate(active):
                if opt.is_correct:
                    continue
                is_lethal = (opt.risk_class >= hp)
                if is_lethal:
                    continue
                total += float(tier_probs[i])  # w(j) = 1
            return max(0.0, total)

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
        if len(probs) == 0 or len(probs) != len(active):
            return 0.0
        hp = qs.hp
        p_d = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                continue
            if opt.risk_class >= hp:
                p_d += float(probs[i])
        return max(0.0, p_d)

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
        tau_t = max(0, qs.max_rounds - qs.rounds_used)
        if tau_t <= 0:
            return 1.0

        if len(probs) == 0 or len(probs) != len(active):
            return 1.0

        p_j_star = 0.0
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_j_star = float(probs[i])
                break

        p_success = 1.0 - (1.0 - p_j_star) ** tau_t
        return float(max(0.0, 1.0 - p_success))

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
        rm = self.rollout_mode
        if rm == 'proxy':
            return False
        if action == 'WAIT':
            return False  # WAIT always uses cheap proxy
        if rm == 'full':
            return True
        # hybrid
        if mode == 'rescue':
            return True
        # normal mode: rollout only at decision boundary or high proxy risk
        if q_wait is not None:
            # Will be set after WAIT Q is computed
            pass  # boundary check happens in caller after WAIT Q known
        high_timeout = (p_timeout_proxy is not None and p_timeout_proxy > 0.5)
        return high_timeout

    def _rollout_estimate(
        self,
        qs: QueryState,
        active: List[Option],
        spec: Dict[str, Any],
        learner: LearnerAgent,
        n_override: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Short learner rollout for calibrated (P_death, P_timeout, P_success).

        Uses the actual learner scoring path (scorer + danger_head + attention)
        to simulate N independent full queries from the current state.

        State copied: QueryState (hp, rounds_used, menu, highlighted_cells, banned_indices),
          scorer (CLS weights), danger_head (hazard+severity bayesian state),
          attention weights (from policy.attention).

        Under HIGHLIGHT: applies exp(rho_H) attention boost before each pick,
          consistent with learner_agent.act() behavior.
        Under BAN: banned option excluded from picks.
        Learner picks via _compute_learner_probs() — same formula as real policy.

        Returns: (p_death, p_timeout, p_success) empirical means over N rollouts.
        """
        import copy as _copy
        N = n_override if n_override is not None else self.rollout_n
        action = spec.get("action", "WAIT")
        ban_idx: Optional[int] = spec.get("ban_index")
        hl_cells: Optional[Tuple] = spec.get("highlight_cells")
        lcfg = self.cfg.learner

        # ── Snapshot learner state ────────────────────────────────
        scorer_snap = _copy.deepcopy(getattr(learner, '_scorer', None))
        policy = getattr(learner, 'policy', None)
        danger_head_snap = _copy.deepcopy(policy.danger_head) if policy else None
        memory_snap = _copy.deepcopy(policy.memory) if policy else None

        # Attention weights at start of this query
        if policy is not None and policy.attention is not None:
            base_attn = policy.attention.weights.copy()
        else:
            L = len(qs.target_output)
            base_attn = np.ones(L) / max(L, 1)

        # Compute attention under this action (applied once at query start)
        attn_action = base_attn.copy()
        if action in ("HIGHLIGHT", "MIX") and hl_cells:
            for ell in hl_cells:
                if 0 <= ell < len(attn_action):
                    attn_action[ell] *= np.exp(lcfg.rho_H)
            s = attn_action.sum()
            if s > 0:
                attn_action /= s

        # ── Run N rollouts ────────────────────────────────────────
        deaths = 0
        timeouts = 0
        successes = 0
        rng_base = np.random.default_rng(seed=hash((qs.query_id, action)) & 0xFFFFFFFF)

        for roll_i in range(N):
            # Per-rollout deepcopy of scorer and danger_head (for EM updates)
            scorer_roll = _copy.deepcopy(scorer_snap)
            dh_roll = _copy.deepcopy(danger_head_snap)

            # Simulate QueryState
            hp = qs.hp
            rounds_used = qs.rounds_used
            max_rounds = qs.max_rounds
            target = qs.target_output

            # Build active menu for this rollout
            banned = set(qs.banned_indices)
            if action in ("BAN", "MIX") and ban_idx is not None:
                banned.add(ban_idx)
            active_roll = [o for o in qs.menu if o.index not in banned]

            # Current attention for this rollout (updated after each reveal)
            attn_roll = attn_action.copy()

            rng = np.random.default_rng(rng_base.integers(0, 2**32) + roll_i)
            outcome = 'timeout'

            while rounds_used < max_rounds and hp > 0 and active_roll:
                # Compute pick probs using learner scoring
                K = len(active_roll)
                sem = np.zeros(K)
                mu_d_arr = np.zeros(K)
                u_d_arr = np.zeros(K)

                for i, opt in enumerate(active_roll):
                    if scorer_roll is not None:
                        sem[i] = scorer_roll.score_option(
                            target, opt.text, attention_weights=attn_roll)
                    if dh_roll is not None:
                        mu, u = dh_roll.predict(opt.danger_vec)
                        mu_d_arr[i] = mu
                        u_d_arr[i] = u

                U = (lcfg.alpha_sem * sem
                     - lcfg.alpha_risk * mu_d_arr
                     - lcfg.alpha_unc * u_d_arr)
                U_shifted = U - U.max()
                exp_u = np.exp(lcfg.beta_L * U_shifted)
                probs = exp_u / (exp_u.sum() + 1e-30)
                eps = lcfg.epsilon
                probs = (1 - eps) * probs + eps / K
                probs /= probs.sum()

                pick_i = int(rng.choice(K, p=probs))
                picked = active_roll[pick_i]
                rounds_used += 1

                if picked.is_correct:
                    # ── Correct pick: sync with learner's correct_pick_learning ──
                    # If the real learner would update CLS on correct pick,
                    # the rollout must also update scorer_roll to stay consistent.
                    if (lcfg.correct_pick_learning_mode == "cortex_em"
                            and scorer_roll is not None
                            and hasattr(scorer_roll, 'incremental_study')):
                        from ..interfaces import Example
                        pos_ex = Example(
                            words=list(picked.text),
                            output=list(target),
                        )
                        n_em_ov = lcfg.correct_pick_n_em_override
                        # Lightweight EM (n_em_override=1), matches _handle_correct_pick
                        if lcfg.eta_correct_pick >= 1.0 or rng.random() < lcfg.eta_correct_pick:
                            scorer_roll.incremental_study([pos_ex], n_em_override=n_em_ov)
                    outcome = 'success'
                    break
                else:
                    damage = picked.risk_class
                    hp = max(0, hp - damage)
                    if hp <= 0:
                        outcome = 'death'
                        break
                    # Simulate reveal + incremental_study (CLS update).
                    #
                    # Semantics: in the real env, the wrong-pick reveal shows the
                    # GRAMMAR output (option.rendered_output), NOT the CLS prediction.
                    # The old predict_output() fallback was therefore doubly wrong:
                    #   (1) it called beam search (~15-25ms, the #1 rollout bottleneck)
                    #   (2) it used the learner's own prediction instead of the truth
                    # Fix: always use option.rendered_output (set at menu construction).
                    # If None (padding distractor), skip the study step — studying an
                    # unknown rendered output would add noise, not signal.
                    if scorer_roll is not None and hasattr(scorer_roll, 'incremental_study'):
                        rendered = picked.rendered_output
                        if rendered and lcfg.eta_reveal >= 1.0:
                            from ..interfaces import Example
                            ex = Example(words=list(picked.text), output=list(rendered))
                            scorer_roll.incremental_study([ex])

                    # Update danger_head
                    if dh_roll is not None:
                        dh_roll.update(picked.danger_vec, damage)

            if outcome == 'success':
                successes += 1
            elif outcome == 'death':
                deaths += 1
            else:
                timeouts += 1

        p_death = deaths / N
        p_timeout = timeouts / N
        p_success = successes / N
        return float(p_death), float(p_timeout), float(p_success)

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

    # ── Block runner ──────────────────────────────────────────────

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

        if not block.done:
            block.done = True

        # Backfill outcomes into decision trace
        self.finalize_trace(block)

        return block

    # ── Decision Trace ────────────────────────────────────────────

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

    # ── Diagnostics ───────────────────────────────────────────────

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

