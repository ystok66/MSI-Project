"""
tutor/causal_audit.py — Phase 6H.5: Clone-and-intervene action-effect audit.

Measures the causal effect of tutor actions on the next learner pick distribution.

Protocol:
    1. Collect post-reveal states (post_reveal_phase=True, query unsolved)
    2. For each state, clone learner + env
    3. Apply each candidate tutor action
    4. Measure next-step P(correct) shift = DeltaP_correct

This answers QH5-Q1/Q2:
    - Does HIGHLIGHT causally increase next-step P(correct)?
    - Is MIX better than standalone HIGHLIGHT or just acting like BAN?

Usage:
    from cls_option_tutor.tutor.causal_audit import audit_post_reveal_action_effects
"""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from ..env.state import QueryState
from ..env.interventions import (
    apply_wait, apply_ban, apply_highlight, apply_mix,
    get_active_menu,
)
from ..interfaces import Option


@dataclass
class ActionEffectResult:
    """Result of a single clone-and-intervene audit for one action."""
    action_name: str
    p_correct_before: float
    p_correct_after: float
    delta_p_correct: float
    p_safe_diag_after: float
    p_high_risk_after: float
    entropy_before: float
    entropy_after: float
    top1_before: int           # index of most likely option before action
    top1_after: int            # index of most likely option after action
    top1_changed: bool
    # Phase 6H.6: full accounting
    p_refresh: float = 0.0
    p_correct_conditional_pick: float = 0.0  # P(correct | action=pick)
    banned_option_pre_prob: float = 0.0      # P(banned opt) before BAN
    # Raw counts for computing denominators correctly
    n_active: int = 0
    hl_cells: Tuple[int, ...] = ()


@dataclass
class AuditSummary:
    """Aggregated summary across multiple audited states.

    Phase 6I-A: also stores per_state_results for CATE computation.
    Each element is the list of ActionEffectResult for one audit state.
    """
    n_states: int = 0
    results_by_action: Dict[str, List[ActionEffectResult]] = field(
        default_factory=dict)
    per_state_results: List[List[ActionEffectResult]] = field(
        default_factory=list)

    def mean_delta_p_correct(self, action: str) -> float:
        rs = self.results_by_action.get(action, [])
        if not rs:
            return 0.0
        return float(np.mean([r.delta_p_correct for r in rs]))

    def mean_p_correct_before(self) -> float:
        rs = self.results_by_action.get("WAIT", [])
        if not rs:
            return 0.0
        return float(np.mean([r.p_correct_before for r in rs]))

    def top1_change_rate(self, action: str) -> float:
        rs = self.results_by_action.get(action, [])
        if not rs:
            return 0.0
        return float(np.mean([r.top1_changed for r in rs]))

    # ------------------------------------------------------------------
    # 6I-A: per-state CATE aggregation
    # ------------------------------------------------------------------

    def _per_state_best(self, action_prefixes: Tuple[str, ...]) -> List[float]:
        """For each state, find max delta_p_correct across matching actions."""
        bests = []
        for state_results in self.per_state_results:
            deltas = [r.delta_p_correct for r in state_results
                      if any(r.action_name.startswith(p) for p in action_prefixes)]
            bests.append(max(deltas) if deltas else 0.0)
        return bests

    def cate_summary(self) -> dict:
        """Compute per-state CATE aggregation for HIGHLIGHT and MIX.

        Definitions:
          PositiveHL_CATE_OppRate  = frac of states where max_H DeltaP > 0
          BestHLCATE              = mean of per-state max_H DeltaP
          PositiveMIX_CATE_OppRate = frac of states where max_MIX DeltaP > 0
          BestMIXCATE             = mean of per-state max_MIX DeltaP
          BestBANCATE             = mean of per-state BAN DeltaP
        """
        if not self.per_state_results:
            return {}
        n = len(self.per_state_results)

        hl_bests = self._per_state_best(("HIGHLIGHT",))
        mix_bests = self._per_state_best(("MIX",))
        ban_bests = self._per_state_best(("BAN",))

        return {
            "CATE_PositiveHL_OppRate": round(
                sum(1 for v in hl_bests if v > 0) / n, 4),
            "CATE_BestHLCATE": round(
                float(np.mean(hl_bests)) if hl_bests else 0.0, 4),
            "CATE_PositiveMIX_OppRate": round(
                sum(1 for v in mix_bests if v > 0) / n, 4),
            "CATE_BestMIXCATE": round(
                float(np.mean(mix_bests)) if mix_bests else 0.0, 4),
            "CATE_BestBANCATE": round(
                float(np.mean(ban_bests)) if ban_bests else 0.0, 4),
        }

    def to_dict(self) -> dict:
        out = {"AuditNStates": self.n_states}
        for action, rs in self.results_by_action.items():
            prefix = f"Audit_{action}"
            if not rs:
                continue
            out[f"{prefix}_DeltaP"] = round(
                float(np.mean([r.delta_p_correct for r in rs])), 4)
            out[f"{prefix}_PcorrAfter"] = round(
                float(np.mean([r.p_correct_after for r in rs])), 4)
            out[f"{prefix}_Top1Change"] = round(
                float(np.mean([r.top1_changed for r in rs])), 4)
            out[f"{prefix}_Entropy"] = round(
                float(np.mean([r.entropy_after for r in rs])), 4)
            out[f"{prefix}_PRefresh"] = round(
                float(np.mean([r.p_refresh for r in rs])), 4)
            out[f"{prefix}_PcorrCondPick"] = round(
                float(np.mean([r.p_correct_conditional_pick for r in rs])), 4)
            out[f"{prefix}_PHiRisk"] = round(
                float(np.mean([r.p_high_risk_after for r in rs])), 4)
            out[f"{prefix}_DeltaRefresh"] = round(
                float(np.mean([r.__dict__.get('delta_refresh', 0.0) for r in rs])), 4)
            if action in ("BAN_high_risk", "MIX_diagnostic",
                          "MIX_counterfactual"):
                out[f"{prefix}_BannedPreProb"] = round(
                    float(np.mean([r.banned_option_pre_prob for r in rs])), 4)
            out[f"{prefix}_N"] = len(rs)
        # 6I-A: CATE aggregation
        out.update(self.cate_summary())
        return out


def _entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _extract_action_probs(
    policy_out,
    active_after: List[Option],
    active_before: List[Option],
) -> Tuple[np.ndarray, float, float]:
    """Extract pick+refresh probabilities from PolicyOutput.

    PolicyOutput.probs has shape (K+1,) where:
      probs[0..K-1] are pick probabilities over active_after options.
      probs[K]      is the refresh probability.

    Returns:
        full_pick_probs: (len(active_before),) array of pick probs, NOT renormalized.
            Options absent from active_after (e.g. banned) get prob 0.
        p_refresh: scalar refresh probability from probs[-1].
        p_pick_total: sum of full_pick_probs (== 1 - p_refresh if well-calibrated).
    """
    raw = np.asarray(policy_out.probs, dtype=float)
    K_after = len(active_after)

    if len(raw) == K_after + 1:
        pick_probs_after = raw[:-1]          # (K_after,)
        p_refresh = float(raw[-1])
    elif len(raw) == K_after:
        # Older code path without refresh slot — treat as pick-only
        pick_probs_after = raw
        p_refresh = 0.0
    else:
        # Unexpected shape — fall back to uniform over active_after
        pick_probs_after = np.ones(K_after) / max(K_after, 1)
        p_refresh = 0.0

    # Map active_after probabilities back into active_before index space
    after_pos = {opt.index: i for i, opt in enumerate(active_after)}
    full_pick_probs = np.zeros(len(active_before), dtype=float)
    for i_before, opt in enumerate(active_before):
        if opt.index in after_pos:
            full_pick_probs[i_before] = pick_probs_after[after_pos[opt.index]]

    # Do NOT renormalize — preserve absolute probability mass
    p_pick_total = float(full_pick_probs.sum())
    return full_pick_probs, p_refresh, p_pick_total


def _pick_distribution_after_action(
    qs_clone: QueryState,
    active: List[Option],
    learner_clone,
    action_spec: Dict[str, Any],
    highlight_strength: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Compute next-step distribution after applying an action to a cloned state.

    Uses learner_clone.get_policy_snapshot_for_query(qs_clone).
    PolicyOutput.probs is (K+1,); the last element is p_refresh.

    Returns:
        full_pick_probs: (len(active),) — absolute pick probs, NOT renormalized.
            Banned options get 0.0.  Sum == 1 - p_refresh (if model well-calibrated).
        p_refresh: refresh probability directly from probs[-1].
    """
    # Apply action to clone's query state
    action = action_spec.get("action", "WAIT")

    if action == "WAIT":
        pass
    elif action == "BAN":
        ban_idx = action_spec.get("ban_index")
        if ban_idx is not None:
            qs_clone.banned_indices.add(ban_idx)
    elif action in ("HIGHLIGHT", "HIGHLIGHT_fixed"):
        hl_cells = action_spec.get("highlight_cells", ())
        if hl_cells:
            qs_clone.highlighted_cells = tuple(hl_cells)
    elif action == "MIX":
        ban_idx = action_spec.get("ban_index")
        hl_cells = action_spec.get("highlight_cells", ())
        if ban_idx is not None:
            qs_clone.banned_indices.add(ban_idx)
        if hl_cells:
            qs_clone.highlighted_cells = tuple(hl_cells)

    # Get policy snapshot — processes attention/highlight correctly
    try:
        policy_out = learner_clone.get_policy_snapshot_for_query(qs_clone)
        active_after = get_active_menu(qs_clone)
        if not active_after:
            return np.zeros(len(active)), 0.0

        full_pick_probs, p_refresh, _ = _extract_action_probs(
            policy_out, active_after, active)

    except Exception:
        active_after = get_active_menu(qs_clone)
        K_af = max(len(active_after), 1)
        # Uniform fallback (pick-only, no refresh)
        after_set = {o.index for o in active_after}
        full_pick_probs = np.array(
            [1.0 / K_af if o.index in after_set else 0.0 for o in active],
            dtype=float)
        p_refresh = 0.0

    return full_pick_probs, p_refresh


def audit_post_reveal_action_effects(
    learner,
    qs: QueryState,
    active: List[Option],
    diag_labels: Optional[Dict[int, str]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    highlight_cells_diagnostic: Optional[Tuple[int, ...]] = None,
    highlight_cells_fixed: Optional[Tuple[int, ...]] = None,
    highlight_cells_counterfactual: Optional[Tuple[int, ...]] = None,
    highlight_strength: float = 1.0,
) -> List[ActionEffectResult]:
    """Audit causal effect of candidate actions on next-step P(correct).

    Args:
        learner: LearnerAgent with pick distribution access.
        qs: Current QueryState (post-reveal phase).
        active: Active menu options.
        diag_labels: {opt_index: label} sidecar labels.
        actions: List of action specs to test. If None, uses default set.
        highlight_cells_diagnostic: Pre-computed D_l highlight cells.
        highlight_cells_fixed: First-N highlight cells for baseline.
        highlight_cells_counterfactual: 6I-B counterfactual-selected cells.
        highlight_strength: Multiplier for highlight attention weight.

    Returns:
        List of ActionEffectResult — one per action.
    """
    labels = diag_labels or getattr(qs, 'option_diag_labels', {})

    # Default canonical action set
    if actions is None:
        # Find a ban target: prefer high_risk_lure
        ban_target_idx = None
        for opt in active:
            if not opt.is_correct and labels.get(opt.index, "") == "high_risk_lure":
                ban_target_idx = opt.index
                break
        if ban_target_idx is None:
            for opt in active:
                if not opt.is_correct and opt.risk_class >= qs.hp:
                    ban_target_idx = opt.index
                    break

        # Default highlight cells
        hl_diag = highlight_cells_diagnostic or tuple(range(min(2, len(qs.target_output))))
        hl_fixed = highlight_cells_fixed or tuple(range(min(2, len(qs.target_output))))
        hl_cf = highlight_cells_counterfactual  # may be None

        actions = [
            {"action": "WAIT"},
            {"action": "HIGHLIGHT", "highlight_cells": hl_diag,
             "label": "HIGHLIGHT_diagnostic"},
            {"action": "HIGHLIGHT", "highlight_cells": hl_fixed,
             "label": "HIGHLIGHT_fixed"},
        ]
        # 6I-A: counterfactual highlight (only if cells were computed)
        if hl_cf is not None:
            actions.append({"action": "HIGHLIGHT", "highlight_cells": hl_cf,
                            "label": "HIGHLIGHT_counterfactual"})
        if ban_target_idx is not None:
            actions.append({"action": "BAN", "ban_index": ban_target_idx,
                            "label": "BAN_high_risk"})
            actions.append({"action": "MIX", "ban_index": ban_target_idx,
                            "highlight_cells": hl_diag,
                            "label": "MIX_diagnostic"})
            # 6I-A: counterfactual MIX
            if hl_cf is not None:
                actions.append({"action": "MIX", "ban_index": ban_target_idx,
                                "highlight_cells": hl_cf,
                                "label": "MIX_counterfactual"})

    # ── Baseline: WAIT (no action) ───────────────────────────────
    baseline_full_probs = np.zeros(len(active))
    baseline_p_refresh = 0.0
    baseline_p_pick_total = 0.0
    try:
        baseline_qs = copy.deepcopy(qs)
        baseline_learner = copy.deepcopy(learner)
        baseline_full_probs, baseline_p_refresh = _pick_distribution_after_action(
            baseline_qs, list(active), baseline_learner,
            {"action": "WAIT"}, highlight_strength=1.0)
        baseline_p_pick_total = float(baseline_full_probs.sum())
    except Exception:
        # Uniform fallback
        baseline_full_probs = np.ones(len(active)) / max(len(active), 1)
        baseline_p_refresh = 0.0
        baseline_p_pick_total = 1.0

    # P_correct_full under WAIT — raw, not renormalized
    p_correct_before = 0.0
    top1_before = 0
    if len(baseline_full_probs) > 0:
        top1_before = int(np.argmax(baseline_full_probs))
        for i, opt in enumerate(active):
            if opt.is_correct:
                p_correct_before = float(baseline_full_probs[i])

    # P_correct_conditional_pick under WAIT
    p_correct_before_cond = (p_correct_before / max(baseline_p_pick_total, 1e-12)
                             if baseline_p_pick_total > 1e-12 else 0.0)

    entropy_before = _entropy(baseline_full_probs)

    results = []
    for action_spec in actions:
        action_name = action_spec.get("label", action_spec.get("action", "?"))
        try:
            qs_clone = copy.deepcopy(qs)
            learner_clone = copy.deepcopy(learner)

            after_probs, p_refresh = _pick_distribution_after_action(
                qs_clone, list(active), learner_clone,
                action_spec, highlight_strength=highlight_strength)

            p_pick_total_after = float(after_probs.sum())

            p_correct_after = 0.0
            p_safe_diag_after = 0.0
            p_high_risk_after = 0.0
            banned_pre_prob = 0.0
            top1_after = int(np.argmax(after_probs)) if len(after_probs) > 0 else 0

            for i, opt in enumerate(active):
                if i >= len(after_probs):
                    break
                p = float(after_probs[i])
                if opt.is_correct:
                    p_correct_after = p
                label = labels.get(opt.index, "")
                if label == "safe_diagnostic_wrong":
                    p_safe_diag_after += p
                elif label == "high_risk_lure":
                    p_high_risk_after += p
                ban_idx = action_spec.get("ban_index")
                if ban_idx is not None and opt.index == ban_idx:
                    # banned option's pre-prob from WAIT baseline
                    if i < len(baseline_full_probs):
                        banned_pre_prob = float(baseline_full_probs[i])

            entropy_after = _entropy(after_probs)

            # P_correct conditional on pick (excludes refresh mass)
            p_correct_cond = (p_correct_after / max(p_pick_total_after, 1e-12)
                              if p_pick_total_after > 1e-12 else 0.0)

            # DeltaRefresh: positive means this action makes learner more likely to refresh
            delta_refresh = p_refresh - baseline_p_refresh

            results.append(ActionEffectResult(
                action_name=action_name,
                p_correct_before=round(p_correct_before, 4),
                p_correct_after=round(p_correct_after, 4),
                delta_p_correct=round(p_correct_after - p_correct_before, 4),
                p_safe_diag_after=round(p_safe_diag_after, 4),
                p_high_risk_after=round(p_high_risk_after, 4),
                entropy_before=round(entropy_before, 4),
                entropy_after=round(entropy_after, 4),
                top1_before=top1_before,
                top1_after=top1_after,
                top1_changed=(top1_after != top1_before),
                p_refresh=round(p_refresh, 4),
                p_correct_conditional_pick=round(p_correct_cond, 4),
                banned_option_pre_prob=round(banned_pre_prob, 4),
                n_active=len(active),
                hl_cells=action_spec.get("highlight_cells", ()),
            ))
            # Annotate with delta_refresh for to_dict() if we extend later
            results[-1].__dict__['delta_refresh'] = round(delta_refresh, 4)
        except Exception:
            results.append(ActionEffectResult(
                action_name=action_name,
                p_correct_before=p_correct_before,
                p_correct_after=p_correct_before,
                delta_p_correct=0.0,
                p_safe_diag_after=0.0,
                p_high_risk_after=0.0,
                entropy_before=entropy_before,
                entropy_after=entropy_before,
                top1_before=top1_before,
                top1_after=top1_before,
                top1_changed=False,
            ))

    return results


def collect_post_reveal_states(block) -> List[Dict[str, Any]]:
    """Extract post-reveal audit states from a completed block.

    Returns list of {qs, active_at_time, round_t} dicts for audit states:
        post_reveal_phase=True
        n_safe_diag_wrong_reveals >= 1
        query not yet solved at that round
        rounds left >= 1
        hp > 0
    """
    from ..env.interventions import get_active_menu
    states = []

    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]

    for qs in teach_queries:
        n_safe = getattr(qs, 'n_safe_diag_wrong_reveals', 0)
        if n_safe < 1:
            continue
        if qs.hp <= 0:
            continue
        if qs.success:
            continue  # only count mid-query states pre-success
        rounds_left = qs.max_rounds - qs.rounds_used
        if rounds_left < 1:
            continue

        active = get_active_menu(qs)
        if not active:
            continue

        states.append({
            "qs": qs,
            "active": active,
            "n_safe_reveals": n_safe,
            "hp": qs.hp,
            "rounds_left": rounds_left,
            "post_reveal": getattr(qs, 'post_reveal_phase', False),
        })

    return states
