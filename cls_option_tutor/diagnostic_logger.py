"""
diagnostic_logger.py — Full-chain diagnostic logging for CLS Option Tutor.

Records per-step learner and tutor state for post-hoc analysis.
All data stored as plain dicts for JSON serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json
import os


@dataclass
class LearnerStepLog:
    """One learner decision step within a query."""
    query_id: int
    step_idx: int

    # Attention
    attention_weights: List[float]           # (L,) current weights
    highlighted_cells: Tuple[int, ...]       # which cells are highlighted
    is_highlight_active: bool

    # Semantic scoring (per active option)
    option_indices: List[int]
    option_texts: List[List[str]]
    semantic_scores: List[float]             # S_sem(j) for each active option
    cls_predictions: List[List[str]]         # CLS Ŷ_j for each option
    oracle_predictions: List[List[str]]      # Oracle Ŷ_j (if available)
    target_output: List[str]                 # Y*

    # Per-cell mismatch detail (for the top-3 options)
    cell_mismatch_detail: List[Dict]         # [{cell_idx, target, cls_pred, match, weight}, ...]

    # Danger predictions
    danger_preds: List[float]                # μ_d(j)
    danger_uncs: List[float]                 # u_d(j)
    ko_probs: List[float]                    # p_ko(j)
    hazard_w_norm: float                     # ||w_hazard||
    severity_w_norm: float                   # ||w_severity||

    # Utility computation
    utilities: List[float]                   # U(j) for each option
    u_refresh: float                         # U(refresh)
    policy_probs: List[float]               # π(j) softmax probs

    # Decision
    action: str                              # "pick" or "refresh"
    pick_index: Optional[int]                # which option picked (if pick)

    # Budget state
    hp_before: int
    refresh_count: int
    max_refreshes: int
    round_idx: int

    # Risk hint state
    risk_hints_received: List[int]           # option indices that got hints


@dataclass
class TutorStepLog:
    """One tutor decision step."""
    query_id: int
    step_idx: int
    phase: str                               # "observation" or "teaching"

    # All candidate scores
    candidate_scores: List[Dict]             # [{action, total_q, components, ...}]
    selected_action: str
    selected_kwargs: Dict[str, Any]

    # Profile state (if teaching phase)
    profile_semantic_competence: Optional[float]
    profile_g_highlight: Optional[float]

    # Learner model (tutor's view)
    tutor_sem_scores: List[float]           # tutor's semantic scores
    tutor_danger_preds: List[float]         # tutor's danger predictions
    tutor_p_pick: List[float]              # tutor's predicted learner pick probs
    tutor_e_damage: float                  # expected damage under learner policy

    # HIGHLIGHT specific (if applicable)
    hl_ig_values: List[Dict]               # [{cells, IG, H_before, H_after}, ...]

    # RISK_HINT specific (if applicable)
    hint_values: List[Dict]                # [{option_idx, p_h, p_pick, q}, ...]


@dataclass
class OutcomeLog:
    """Outcome after learner action."""
    query_id: int
    step_idx: int
    action: str
    correct: Optional[bool]
    damage: Optional[int]
    hp_after: int
    revealed_output: Optional[List[str]]
    option_risk_class: Optional[int]


@dataclass
class QuerySummaryLog:
    """Per-query summary."""
    query_id: int
    target_output: List[str]
    n_options: int
    n_safe: int
    n_risky: int
    risk_classes: List[int]
    correct_option_index: int
    correct_option_text: List[str]

    # CLS vs Oracle comparison
    cls_correct_score: Optional[float]      # CLS score for correct option
    oracle_correct_score: Optional[float]   # Oracle score for correct option
    cls_rank_of_correct: Optional[int]      # rank of correct in CLS ordering
    oracle_rank_of_correct: Optional[int]

    # Outcome
    solved: bool
    total_damage: int
    rounds_used: int
    was_skipped: bool
    highlight_cells_used: List[Tuple[int, ...]]
    risk_hints_given: List[int]

    # Where CLS disagrees with oracle (per-cell)
    cls_oracle_disagreement: List[Dict]     # [{option_idx, cell_idx, cls, oracle, target}]


@dataclass
class BlockDiagnosticLog:
    """Full block diagnostic log."""
    condition: str
    grammar_id: str
    seed: int
    learner_steps: List[LearnerStepLog] = field(default_factory=list)
    tutor_steps: List[TutorStepLog] = field(default_factory=list)
    outcomes: List[OutcomeLog] = field(default_factory=list)
    query_summaries: List[QuerySummaryLog] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Cannot serialize {type(obj)}")

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1, default=_convert)


class DiagnosticLogger:
    """Collects diagnostic data during a block run."""

    def __init__(self, condition: str, grammar_id: str, seed: int):
        self.log = BlockDiagnosticLog(
            condition=condition, grammar_id=grammar_id, seed=seed)
        self._oracle_scorer = None  # set externally if available

    def set_oracle_scorer(self, scorer):
        """Set oracle scorer for CLS vs oracle comparison."""
        self._oracle_scorer = scorer

    def log_learner_step(
        self,
        qs,                    # QueryState
        policy_out,            # PolicyOutput
        policy,                # LearnerPolicy
        active_options,        # List[Option]
    ) -> None:
        """Log one learner decision step."""
        L = len(qs.target_output)
        attn = policy.attention

        # Attention weights
        weights = attn.weights.tolist() if attn else [1.0/L]*L

        # CLS predictions
        cls_preds = []
        oracle_preds = []
        for opt in active_options:
            cls_pred = policy.scorer.predict_output(opt.text)
            cls_preds.append(cls_pred if cls_pred else [])
            if self._oracle_scorer:
                oracle_pred = self._oracle_scorer.predict_output(opt.text)
                oracle_preds.append(oracle_pred if oracle_pred else [])
            else:
                oracle_preds.append([])

        # Per-cell mismatch detail for top-3 options by score
        sorted_idx = np.argsort(policy_out.semantic_scores)[::-1]
        cell_detail = []
        for rank, idx in enumerate(sorted_idx[:3]):
            opt = active_options[idx]
            pred = cls_preds[idx]
            for cell_idx in range(L):
                target_c = qs.target_output[cell_idx] if cell_idx < L else "?"
                pred_c = pred[cell_idx] if cell_idx < len(pred) else "?"
                cell_detail.append({
                    "rank": rank,
                    "option_idx": opt.index,
                    "cell_idx": cell_idx,
                    "target": target_c,
                    "cls_pred": pred_c,
                    "match": target_c == pred_c,
                    "weight": weights[cell_idx] if cell_idx < len(weights) else 0,
                })

        # Hazard/severity norms
        h_norm = 0.0
        s_norm = 0.0
        if policy.danger_head is not None:
            h_norm = float(np.linalg.norm(policy.danger_head.hazard.w))
            s_norm = float(np.linalg.norm(policy.danger_head.severity.w))

        step = LearnerStepLog(
            query_id=qs.query_id,
            step_idx=qs.rounds_used,
            attention_weights=weights,
            highlighted_cells=qs.highlighted_cells if qs.highlighted_cells else (),
            is_highlight_active=bool(qs.highlighted_cells),
            option_indices=[o.index for o in active_options],
            option_texts=[list(o.text) for o in active_options],
            semantic_scores=policy_out.semantic_scores.tolist(),
            cls_predictions=cls_preds,
            oracle_predictions=oracle_preds,
            target_output=list(qs.target_output),
            cell_mismatch_detail=cell_detail,
            danger_preds=policy_out.danger_preds.tolist(),
            danger_uncs=policy_out.danger_uncs.tolist(),
            ko_probs=getattr(policy_out, 'ko_probs', np.zeros(len(active_options))).tolist()
                if hasattr(policy_out, 'ko_probs') else [],
            hazard_w_norm=h_norm,
            severity_w_norm=s_norm,
            utilities=policy_out.utilities.tolist(),
            u_refresh=float(policy_out.utilities[-1]) if policy_out.action == "refresh" else 0.0,
            policy_probs=policy_out.probs.tolist(),
            action=policy_out.action,
            pick_index=policy_out.pick_index,
            hp_before=qs.hp,
            refresh_count=qs.refreshes_used,
            max_refreshes=qs.max_refreshes,
            round_idx=qs.rounds_used,
            risk_hints_received=list(qs.risk_hints),
        )
        self.log.learner_steps.append(step)

    def log_tutor_step(
        self,
        qs,                    # QueryState
        candidates,            # List[InterventionScore]
        selected_action: str,
        selected_kwargs: dict,
        profile,               # ProfileState
        tutor_sem_scores,      # np.ndarray
        tutor_danger_preds,    # np.ndarray
        tutor_p_pick,         # np.ndarray
        tutor_e_damage: float,
        phase: str = "teaching",
    ) -> None:
        """Log one tutor decision step."""
        # Separate HL and hint candidates
        hl_vals = []
        hint_vals = []
        for c in candidates:
            if c.action == "HIGHLIGHT" and c.components:
                hl_vals.append({
                    "cells": list(c.highlight_cells) if c.highlight_cells else [],
                    "IG": c.components.get("IG", 0),
                    "H_before": c.components.get("H_before", 0),
                    "H_after": c.components.get("H_after", 0),
                    "total_q": c.total_q,
                })
            if c.action == "RISK_HINT" and c.components:
                hint_vals.append({
                    "option_idx": c.hint_index,
                    "p_h": c.components.get("p_h_tutor", 0),
                    "p_pick": c.components.get("p_pick_j", 0),
                    "danger": c.components.get("danger_j", 0),
                    "total_q": c.total_q,
                })

        step = TutorStepLog(
            query_id=qs.query_id,
            step_idx=qs.rounds_used,
            phase=phase,
            candidate_scores=[
                {"action": c.action, "total_q": c.total_q,
                 "ban_index": c.ban_index, "hint_index": c.hint_index,
                 "highlight_cells": list(c.highlight_cells) if c.highlight_cells else None,
                 "components": c.components}
                for c in candidates[:10]  # top 10
            ],
            selected_action=selected_action,
            selected_kwargs=selected_kwargs,
            profile_semantic_competence=profile.semantic_competence if profile else None,
            profile_g_highlight=profile.g_highlight if profile else None,
            tutor_sem_scores=tutor_sem_scores.tolist() if tutor_sem_scores is not None else [],
            tutor_danger_preds=tutor_danger_preds.tolist() if tutor_danger_preds is not None else [],
            tutor_p_pick=tutor_p_pick.tolist() if tutor_p_pick is not None else [],
            tutor_e_damage=tutor_e_damage,
            hl_ig_values=hl_vals,
            hint_values=hint_vals,
        )
        self.log.tutor_steps.append(step)

    def log_outcome(
        self,
        qs,                    # QueryState
        step,                  # LearnerStep
        option=None,           # Option (picked)
    ) -> None:
        """Log outcome of a learner action."""
        # Get revealed output from reveal_history if available
        revealed = None
        if qs.reveal_history:
            last_rev = qs.reveal_history[-1]
            if hasattr(last_rev, 'rendered_output') and last_rev.rendered_output:
                revealed = list(last_rev.rendered_output)

        out = OutcomeLog(
            query_id=qs.query_id,
            step_idx=qs.rounds_used,
            action=step.action,
            correct=step.correct,
            damage=step.damage,
            hp_after=step.hp_after,
            revealed_output=revealed,
            option_risk_class=option.risk_class if option else None,
        )
        self.log.outcomes.append(out)

    def log_query_summary(
        self,
        qs,                    # QueryState
        scorer=None,           # CLS scorer
    ) -> None:
        """Log per-query summary after query is done."""
        correct_opt = next((o for o in qs.menu if o.is_correct), None)

        # CLS vs oracle comparison
        cls_scores = []
        oracle_scores = []
        disagreements = []

        if scorer:
            L = len(qs.target_output)
            for opt in qs.menu:
                cls_s = scorer.score_option(qs.target_output, opt.text)
                cls_scores.append((opt.index, cls_s))

                cls_pred = scorer.predict_output(opt.text)
                oracle_pred = None
                if self._oracle_scorer:
                    oracle_pred = self._oracle_scorer.predict_output(opt.text)
                    oracle_s = self._oracle_scorer.score_option(
                        qs.target_output, opt.text)
                    oracle_scores.append((opt.index, oracle_s))

                    # Find disagreements
                    if cls_pred and oracle_pred:
                        for ci in range(min(L, len(cls_pred), len(oracle_pred))):
                            if cls_pred[ci] != oracle_pred[ci]:
                                disagreements.append({
                                    "option_idx": opt.index,
                                    "cell_idx": ci,
                                    "cls": cls_pred[ci],
                                    "oracle": oracle_pred[ci],
                                    "target": qs.target_output[ci] if ci < L else "?",
                                })

        # Ranks
        cls_rank = None
        oracle_rank = None
        if correct_opt and cls_scores:
            sorted_cls = sorted(cls_scores, key=lambda x: x[1], reverse=True)
            for rank, (idx, _) in enumerate(sorted_cls):
                if idx == correct_opt.index:
                    cls_rank = rank
                    break
        if correct_opt and oracle_scores:
            sorted_oracle = sorted(oracle_scores, key=lambda x: x[1], reverse=True)
            for rank, (idx, _) in enumerate(sorted_oracle):
                if idx == correct_opt.index:
                    oracle_rank = rank
                    break

        risk_classes = [o.risk_class for o in qs.menu]

        summary = QuerySummaryLog(
            query_id=qs.query_id,
            target_output=list(qs.target_output),
            n_options=len(qs.menu),
            n_safe=sum(1 for rc in risk_classes if rc == 0),
            n_risky=sum(1 for rc in risk_classes if rc > 0),
            risk_classes=risk_classes,
            correct_option_index=correct_opt.index if correct_opt else -1,
            correct_option_text=list(correct_opt.text) if correct_opt else [],
            cls_correct_score=dict(cls_scores).get(
                correct_opt.index) if correct_opt and cls_scores else None,
            oracle_correct_score=dict(oracle_scores).get(
                correct_opt.index) if correct_opt and oracle_scores else None,
            cls_rank_of_correct=cls_rank,
            oracle_rank_of_correct=oracle_rank,
            solved=qs.success,
            total_damage=sum(r.damage for r in qs.reveal_history),
            rounds_used=qs.rounds_used,
            was_skipped=qs.skipped,
            highlight_cells_used=[qs.highlighted_cells] if qs.highlighted_cells else [],
            risk_hints_given=list(qs.risk_hints),
            cls_oracle_disagreement=disagreements,
        )
        self.log.query_summaries.append(summary)
