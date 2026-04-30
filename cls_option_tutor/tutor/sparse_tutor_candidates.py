from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import FullConfig
from ..env.state import QueryState
from ..interfaces import Option
from ..learner.learner_agent import LearnerAgent


def _init_generation_info(
    mode: str,
    p_timeout_wait: float,
    hl_timeout_threshold: float,
) -> Dict[str, Any]:
    return {
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
        "hl_gate_threshold": hl_timeout_threshold,
        "hl_suppressed_by_gate": False,
        "mix_generated": False,
        "timeout_blocker_idx": None,
        "timeout_blocker_score": None,
    }


def _pedagogical_highlight_gate(cfg: FullConfig, qs: QueryState) -> bool:
    return (
        getattr(qs, "post_reveal_phase", False)
        and getattr(qs, "n_safe_diag_wrong_reveals", 0) >= 1
        and not qs.success
        and qs.hp > 0
        and qs.rounds_used < (qs.max_rounds - 1)
        and getattr(cfg.tutor, "tutor_lg_mode", "off") in (
            "self_correct", "horizon_self_correct", "diagnostic"
        )
    )


def enumerate_sparse_candidates(
    cfg: FullConfig,
    qs: QueryState,
    active: List[Option],
    learner: LearnerAgent,
    *,
    mode: str,
    p_timeout_wait: float,
    hl_timeout_threshold: float,
    wait_probs_lc: Any,
    select_ban_target: Callable[[QueryState, List[Option], LearnerAgent], Optional[Option]],
    select_timeout_blocker: Callable[[QueryState, List[Option], Option, List[Option], LearnerAgent], Tuple[Optional[Option], Optional[float]]],
    select_highlight_cells: Callable[[QueryState, Option, LearnerAgent], Optional[Tuple[int, ...]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Enumerate tutor candidates and gate diagnostics without Q scoring.

    This keeps the candidate/gating policy separate from downstream Q-value
    computation so SparseTutor._act_teaching() can remain a readable high-level
    decision pipeline.
    """
    candidates: List[Dict[str, Any]] = [{"action": "WAIT"}]
    gen_info = _init_generation_info(mode, p_timeout_wait, hl_timeout_threshold)

    correct_opts = [o for o in active if o.is_correct]
    non_correct = [o for o in active if not o.is_correct]
    if not correct_opts:
        gen_info["ban_gate_reason"] = "no_correct_option"
        return candidates, gen_info

    j_star = correct_opts[0]

    if not non_correct:
        gen_info["ban_gate_reason"] = "no_non_correct"
    else:
        if mode == "rescue":
            ban_target, blocker_score = select_timeout_blocker(
                qs, active, j_star, non_correct, learner
            )
            gen_info["timeout_blocker_idx"] = (
                ban_target.index if ban_target else None
            )
            gen_info["timeout_blocker_score"] = (
                round(blocker_score, 4) if blocker_score is not None else None
            )
        else:
            ban_target = select_ban_target(qs, non_correct, learner)

        if ban_target is None:
            gen_info["ban_gate_reason"] = "selection_failed"
        else:
            gen_info["ban_generated"] = True
            gen_info["ban_gate_reason"] = "ok"
            gen_info["ban_target_idx"] = ban_target.index
            gen_info["ban_target_risk"] = ban_target.risk_class
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

    hl_cells = select_highlight_cells(qs, j_star, learner)
    pedagogical_hl_gate = _pedagogical_highlight_gate(cfg, qs)

    if hl_cells is None:
        gen_info["hl_gate_reason"] = "no_cells"
        return candidates, gen_info

    if mode == "rescue":
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
        return candidates, gen_info

    if pedagogical_hl_gate:
        gen_info["hl_generated"] = True
        gen_info["hl_gate_reason"] = "pedagogical_consolidate"
        gen_info["hl_suppressed_by_gate"] = False
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
        return candidates, gen_info

    if p_timeout_wait <= hl_timeout_threshold:
        gen_info["hl_gate_reason"] = "p_timeout_below_threshold"
        gen_info["hl_suppressed_by_gate"] = True
        return candidates, gen_info

    gen_info["hl_generated"] = True
    gen_info["hl_gate_reason"] = "ok"
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

    return candidates, gen_info
