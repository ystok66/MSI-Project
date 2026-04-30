"""
scripted_protocols.py — Controlled experience sequences for learning-increment experiments.

Phase 6.4: Split protocols into two families:
  - `then_answer`: wrong reveal(s) → SHORTLIST(correct) → learner acts
  - `self_correct`: wrong reveal(s) → set assist_level='self_correct' → force correct pick

These are MECHANISM PROBES: they force exact learner experience sequences.

Wrong type tiers (Phase 6.3):
  - safe:         risk_class == 0
  - bounded_risk: risk_class in {1, 2}
  - high_risk:    risk_class in {3, 4}

Diagnostic wrong selection (Phase 6.4):
  - random:     first available wrong in risk band
  - diagnostic: highest D_learn score wrong in risk band
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ..env.option_env import OptionEnv
from ..env.state import BlockState, QueryState
from ..env.interventions import (
    apply_wait, apply_shortlist, get_active_menu, clear_menu_interventions,
)
from ..learner.learner_agent import LearnerAgent
from ..config import FullConfig
from ..interfaces import Option


# ── Skip logging ──────────────────────────────────────────────────────────────

@dataclass
class SkipRecord:
    query_id: int
    reason: str


@dataclass
class ScriptedBlockResult:
    """Result from a scripted protocol run."""
    block: BlockState
    skipped: List[SkipRecord] = field(default_factory=list)
    wrong_reveal_count: int = 0
    safe_wrong_count: int = 0
    risky_wrong_count: int = 0
    correct_pick_count: int = 0
    direct_answer_count: int = 0

    # Phase 6.3: per-risk-class wrong reveal counters
    wrong_reveal_risk0: int = 0
    wrong_reveal_risk1: int = 0
    wrong_reveal_risk2: int = 0
    wrong_reveal_risk3: int = 0
    wrong_reveal_risk4: int = 0
    correct_after_wrong_count: int = 0
    death_before_correct_count: int = 0
    script_violation_count: int = 0

    # Phase 6.4: experience semantics
    then_answer_count: int = 0
    self_correct_count: int = 0

    # Phase 6.4: forced step tracking for damage attribution
    forced_step_indices: List[int] = field(default_factory=list)


# ── Option selectors (Phase 6.3: 3-tier) ─────────────────────────────────────

def _choose_safe_wrong(qs: QueryState, diagnostic: bool = False) -> Optional[Option]:
    """Find a safe (risk_class == 0) wrong option."""
    candidates = [
        o for o in qs.menu
        if not o.is_correct and o.risk_class == 0
        and o.index not in qs.banned_indices
    ]
    if not candidates:
        return None
    if diagnostic and len(candidates) > 1:
        return _select_most_diagnostic(candidates, qs)
    return candidates[0]


def _choose_bounded_risk_wrong(qs: QueryState, diagnostic: bool = False) -> Optional[Option]:
    """Find a bounded-risk (risk_class in {1, 2}) wrong option."""
    candidates = [
        o for o in qs.menu
        if not o.is_correct and o.risk_class in (1, 2)
        and o.index not in qs.banned_indices
    ]
    if not candidates:
        return None
    if diagnostic and len(candidates) > 1:
        return _select_most_diagnostic(candidates, qs)
    return candidates[0]


def _choose_high_risk_wrong(qs: QueryState, diagnostic: bool = False) -> Optional[Option]:
    """Find a high-risk (risk_class in {3, 4}) wrong option."""
    candidates = [
        o for o in qs.menu
        if not o.is_correct and o.risk_class in (3, 4)
        and o.index not in qs.banned_indices
    ]
    if not candidates:
        return None
    if diagnostic and len(candidates) > 1:
        return _select_most_diagnostic(candidates, qs)
    return candidates[0]


def _choose_any_wrong(qs: QueryState, diagnostic: bool = False) -> Optional[Option]:
    """Find any wrong option."""
    candidates = [
        o for o in qs.menu
        if not o.is_correct and o.index not in qs.banned_indices
    ]
    if not candidates:
        return None
    if diagnostic and len(candidates) > 1:
        return _select_most_diagnostic(candidates, qs)
    return candidates[0]


def _find_correct(qs: QueryState) -> Optional[Option]:
    """Find the correct option."""
    for o in qs.menu:
        if o.is_correct:
            return o
    return None


def _record_wrong(result: ScriptedBlockResult, opt: Option) -> None:
    """Record a wrong-pick in the per-risk-class counters."""
    result.wrong_reveal_count += 1
    rc = opt.risk_class
    if rc == 0:
        result.wrong_reveal_risk0 += 1
        result.safe_wrong_count += 1
    elif rc == 1:
        result.wrong_reveal_risk1 += 1
        result.risky_wrong_count += 1
    elif rc == 2:
        result.wrong_reveal_risk2 += 1
        result.risky_wrong_count += 1
    elif rc == 3:
        result.wrong_reveal_risk3 += 1
        result.risky_wrong_count += 1
    elif rc == 4:
        result.wrong_reveal_risk4 += 1
        result.risky_wrong_count += 1


# ── Diagnostic wrong selection (Phase 6.4) ────────────────────────────────────

def _compute_d_learn(opt: Option, qs: QueryState) -> float:
    """Compute lightweight learning diagnosticity score for a wrong option.

    D_learn = (D_sem + D_risk) / 2
    D_sem = (near_output + order_like + scope_like + closeness) / 4
    D_risk: bounded(1-2)=1.0, safe(0)=0.5, high(3-4)=0.5
    """
    target = list(qs.target_output) if qs.target_output else []
    cand = list(opt.rendered_output) if hasattr(opt, 'rendered_output') and opt.rendered_output else []

    if not target or not cand:
        return 0.0

    # Hamming distance fraction
    max_len = max(len(target), len(cand))
    if max_len == 0:
        return 0.0
    h = sum(1 for a, b in zip(target, cand) if a != b) + abs(len(target) - len(cand))
    h_frac = h / max_len

    # Near output: h <= 0.25
    near_output = 1.0 if h_frac <= 0.25 else 0.0

    # Order-like: same bag of tokens
    from collections import Counter
    bag_same = 1.0 if Counter(target) == Counter(cand) and target != cand else 0.0

    # Scope-like: shared bigram ratio >= 0.5
    def bigrams(seq):
        return set(zip(seq, seq[1:])) if len(seq) >= 2 else set()
    t_bi = bigrams(target)
    c_bi = bigrams(cand)
    shared_bi_ratio = len(t_bi & c_bi) / max(len(t_bi | c_bi), 1)
    scope_like = 1.0 if shared_bi_ratio >= 0.5 else 0.0

    closeness = 1.0 - h_frac

    d_sem = (near_output + bag_same + scope_like + closeness) / 4.0

    # Risk diagnosticity
    rc = opt.risk_class
    if rc in (1, 2):
        d_risk = 1.0
    else:
        d_risk = 0.5

    return (d_sem + d_risk) / 2.0


def _select_most_diagnostic(candidates: List[Option], qs: QueryState) -> Option:
    """Select the wrong option with highest D_learn score."""
    scored = [(c, _compute_d_learn(c, qs)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


# ── Wrong type dispatch table ─────────────────────────────────────────────────

_WRONG_TYPE_SELECTORS = {
    "safe": _choose_safe_wrong,
    "bounded_risk": _choose_bounded_risk_wrong,
    "high_risk": _choose_high_risk_wrong,
    "any": _choose_any_wrong,
}


# ── Protocol registry ─────────────────────────────────────────────────────────

# Maps protocol name -> (wrong_sequence, finish_mode, diagnostic_selection)
# finish_mode: "then_answer" = SHORTLIST(correct), "self_correct" = force correct w/o shortlist
_PROTOCOL_SPECS = {
    # Phase 6.3 legacy — all are then_answer
    "script_wrong1_correct_safe":       (["safe"], "then_answer", False),
    "script_wrong1_correct_bounded_risk": (["bounded_risk"], "then_answer", False),
    "script_wrong1_correct_high_risk":  (["high_risk"], "then_answer", False),
    "script_wrong2_correct_safe":       (["safe", "safe"], "then_answer", False),
    "script_wrong2_mixed_safe_bounded": (["safe", "bounded_risk"], "then_answer", False),
    "script_wrong2_mixed_bounded_high": (["bounded_risk", "high_risk"], "then_answer", False),
    # Legacy aliases
    "script_wrong1_risky_correct":      (["high_risk"], "then_answer", False),
    "script_wrong2_mixed_correct":      (["safe", "high_risk"], "then_answer", False),

    # Phase 6.4: then_answer (explicit naming)
    "script_wrong1_then_answer_safe":     (["safe"], "then_answer", False),
    "script_wrong1_then_answer_bounded":  (["bounded_risk"], "then_answer", False),
    "script_wrong1_then_answer_high":     (["high_risk"], "then_answer", False),

    # Phase 6.4: self_correct
    "script_wrong1_self_correct_safe":     (["safe"], "self_correct", False),
    "script_wrong1_self_correct_bounded":  (["bounded_risk"], "self_correct", False),
    "script_wrong1_self_correct_high":     (["high_risk"], "self_correct", False),
    "script_wrong2_self_correct_safe":     (["safe", "safe"], "self_correct", False),
    "script_wrong2_self_correct_safe_bounded": (["safe", "bounded_risk"], "self_correct", False),
    "script_wrong2_self_correct_bounded_high": (["bounded_risk", "high_risk"], "self_correct", False),

    # Phase 6.4: diagnostic wrong selection variants
    "script_wrong1_self_correct_diagnostic_safe": (["safe"], "self_correct", True),
    "script_wrong1_self_correct_random_safe":     (["safe"], "self_correct", False),
}


class ScriptedProtocolRunner:
    """Runs a specific scripted experience sequence on teach queries."""

    def __init__(self, cfg: FullConfig, protocol: str):
        self.cfg = cfg
        self.protocol = protocol

    def run_block(
        self,
        env: OptionEnv,
        learner: LearnerAgent,
        task_id: str,
        seed: int = 42,
    ) -> ScriptedBlockResult:
        """Run a complete block with scripted teach intervention."""
        block = env.reset_block(task_id, seed=seed)
        support, _, grammar = env.adapter.load_task(task_id)
        learner.init_block(block, grammar, support)

        result = ScriptedBlockResult(block=block)
        max_steps = len(block.queries) * 30
        steps = 0

        while not block.done and steps < max_steps:
            steps += 1
            qs = block.current_query
            if qs is None:
                break
            if qs.done:
                block.current_query_idx += 1
                if block.current_query_idx >= len(block.queries):
                    block.done = True
                continue

            if block.in_teaching_phase:
                self._run_teach_query(qs, block, env, learner, result)
            else:
                ts = apply_wait(qs, round_t=qs.rounds_used)
                block.tutor_trace.append(ts)
                if not qs.done:
                    learner.act(block, env)

        if not block.done:
            block.done = True

        return result

    def _run_teach_query(
        self,
        qs: QueryState,
        block: BlockState,
        env: OptionEnv,
        learner: LearnerAgent,
        result: ScriptedBlockResult,
    ) -> None:
        """Execute the scripted protocol for one teach query."""
        protocol = self.protocol

        if protocol == "script_direct_answer":
            self._script_direct_answer(qs, block, env, learner, result)
        elif protocol == "script_direct_correct":
            self._script_direct_correct(qs, block, env, learner, result)
        elif protocol in _PROTOCOL_SPECS:
            wrong_seq, finish_mode, diagnostic = _PROTOCOL_SPECS[protocol]
            self._script_wrong_then_finish(
                qs, block, env, learner, result,
                wrong_sequence=wrong_seq,
                finish_mode=finish_mode,
                diagnostic=diagnostic,
            )
        elif protocol in ("no_tutor_reveal", "no_tutor_nonreveal_neg"):
            while not qs.done and qs.rounds_used < qs.max_rounds:
                ts = apply_wait(qs, round_t=qs.rounds_used)
                block.tutor_trace.append(ts)
                if qs.done:
                    break
                learner.act(block, env)
        else:
            raise ValueError(f"Unknown scripted protocol: {protocol}")

    # ── Script implementations ────────────────────────────────────────────

    def _script_direct_answer(self, qs, block, env, learner, result):
        """SHORTLIST([j*]) on first round."""
        j_star = _find_correct(qs)
        if j_star is None:
            result.skipped.append(SkipRecord(qs.query_id, "no_correct_option"))
            qs.done = True
            qs.skipped = True
            return
        ts = apply_shortlist(qs, [j_star.index], round_t=qs.rounds_used)
        block.tutor_trace.append(ts)
        result.direct_answer_count += 1
        result.then_answer_count += 1
        if not qs.done:
            qs.learning_event_source = "scripted_direct_answer"
            learner.act(block, env)
            qs.learning_event_source = "incidental"
        if qs.success:
            result.correct_pick_count += 1

    def _script_direct_correct(self, qs, block, env, learner, result):
        """Force learner to pick j* with assist_level='none' (unassisted)."""
        j_star = _find_correct(qs)
        if j_star is None:
            result.skipped.append(SkipRecord(qs.query_id, "no_correct_option"))
            qs.done = True
            qs.skipped = True
            return
        ts = apply_wait(qs, round_t=qs.rounds_used)
        block.tutor_trace.append(ts)
        if not qs.done:
            qs.learning_event_source = "scripted_direct_correct"
            step = env.force_learner_pick(block, j_star.index)
            learner.observe_forced_step(block, step, qs=qs)
            qs.learning_event_source = "incidental"
        if qs.success:
            result.correct_pick_count += 1
            result.self_correct_count += 1

    def _script_wrong_then_finish(
        self, qs, block, env, learner, result,
        wrong_sequence: List[str],
        finish_mode: str,
        diagnostic: bool,
    ) -> None:
        """Force wrong picks, then finish with either SHORTLIST or self-correct.

        finish_mode:
          "then_answer": SHORTLIST(correct) → learner.act()
          "self_correct": set assist_level='self_correct' → force correct pick
        """
        step_base = len(block.learner_trace)

        for i, wtype in enumerate(wrong_sequence):
            if qs.done or qs.hp <= 0:
                if not qs.success:
                    result.death_before_correct_count += 1
                break

            selector = _WRONG_TYPE_SELECTORS.get(wtype, _choose_any_wrong)
            wrong_opt = selector(qs, diagnostic=diagnostic)
            if wrong_opt is None:
                result.skipped.append(SkipRecord(
                    qs.query_id, f"no_{wtype}_wrong_for_step_{i}"
                ))
                qs.done = True
                qs.skipped = True
                return

            # WAIT, then force wrong pick
            ts = apply_wait(qs, round_t=qs.rounds_used)
            block.tutor_trace.append(ts)
            if not qs.done:
                qs.learning_event_source = "scripted_wrong_reveal"
                step = env.force_learner_pick(block, wrong_opt.index)
                learner.observe_forced_step(block, step, qs=qs)
                qs.learning_event_source = "incidental"
            _record_wrong(result, wrong_opt)
            # Track forced step for damage attribution
            result.forced_step_indices.append(len(block.learner_trace) - 1)

            if qs.hp <= 0:
                result.death_before_correct_count += 1
                return

        # Finish: then_answer or self_correct
        if not qs.done and qs.hp > 0:
            j_star = _find_correct(qs)
            if j_star is None:
                result.skipped.append(SkipRecord(qs.query_id, "no_correct_after_wrong"))
                result.script_violation_count += 1
                return

            if finish_mode == "then_answer":
                # SHORTLIST(correct) → learner acts
                ts = apply_shortlist(qs, [j_star.index], round_t=qs.rounds_used)
                block.tutor_trace.append(ts)
                result.direct_answer_count += 1
                result.then_answer_count += 1
                if not qs.done:
                    qs.learning_event_source = "scripted_then_answer"
                    learner.act(block, env)
                    qs.learning_event_source = "incidental"
                if qs.success:
                    result.correct_pick_count += 1
                    result.correct_after_wrong_count += 1
            elif finish_mode == "self_correct":
                # Set self_correct assist level → force correct pick
                qs.assist_level = "self_correct"
                ts = apply_wait(qs, round_t=qs.rounds_used)
                block.tutor_trace.append(ts)
                if not qs.done:
                    qs.learning_event_source = "scripted_self_correct"
                    step = env.force_learner_pick(block, j_star.index)
                    learner.observe_forced_step(block, step, qs=qs)
                    qs.learning_event_source = "incidental"
                if qs.success:
                    result.correct_pick_count += 1
                    result.correct_after_wrong_count += 1
                    result.self_correct_count += 1
