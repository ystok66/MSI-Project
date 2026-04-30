from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cls_option_tutor.experiments.condition_overrides import (
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.run_learning_increment_micro import (
    DATA_DIR,
    make_cfg,
    _apply_condition_overrides,
)
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.interventions import get_active_menu
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
from cls_option_tutor.tutor.highlight_selection import select_counterfactual_highlight_cells
from cls_option_tutor.tutor.sparse_tutor_scoring import (
    build_option_mass_records,
    compute_ban_oracle_stats,
    compute_postreveal_q,
    compute_postreveal_shift_decomp,
)

def _run_teach_block(task_id: str, seed: int, condition: str, *, rho: float, generator: str):
    cfg = make_cfg(n_sup=4, rho_assist=rho, generator_mode=generator, tutor_lg_mode="off", highlight_mode="diagnostic")
    condition_eff = resolve_condition_alias(condition)
    cfg = _apply_condition_overrides(copy.deepcopy(cfg), condition_eff)

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=seed)
    support, _, grammar = env.adapter.load_task(task_id)
    init_block = env.reset_block(task_id, seed=seed)
    learner.init_block(init_block, grammar, support)

    if condition_eff.startswith("script_") or condition_eff.startswith("no_tutor_"):
        teach_cfg = copy.deepcopy(cfg)
        protocol = extract_scripted_protocol_name(condition_eff)
        if protocol == "no_tutor_nonreveal_neg":
            teach_cfg.env.feedback_mode = "nonreveal"
            teach_cfg.learner.reveal_learning_mode = "nonreveal_negative"
            teach_cfg.learner.negative_evidence_mode = "exact_program_target"
        runner = ScriptedProtocolRunner(cfg=teach_cfg, protocol=protocol)
        result = runner.run_block(OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR), learner, task_id, seed=seed)
        return result.block, teach_cfg

    teach_cfg = copy.deepcopy(cfg)
    tutor = SparseTutorAgent(cfg=teach_cfg)
    block = tutor.run_block(OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR), learner, task_id, seed=seed)
    return block, teach_cfg


def _wrong_label_family(qs, pick_index: Optional[int]) -> str:
    if qs is None or pick_index is None:
        return "none"
    labels = getattr(qs, "option_diag_labels", {}) or {}
    label = labels.get(pick_index, "")
    if label == "safe_diagnostic_wrong":
        return "safe_diag"
    if label == "bounded_diagnostic_wrong":
        return "bounded_diag"
    if label == "high_risk_lure":
        return "high_risk"
    if label in ("safe_far", "safe_random_wrong", "risky_far"):
        return "far"
    return "other"


def _collect_wrong_reveal_stats(block) -> Dict[str, float]:
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]
    query_by_qid = {qs.query_id: qs for qs in teach_queries}
    total = safe = bounded = high = far = same_repeat = 0
    risks: List[float] = []
    damages: List[float] = []
    seen_wrong_by_query: Dict[int, set[int]] = {}
    for ls in getattr(block, "learner_trace", []) or []:
        if getattr(ls, "action", None) != "pick" or getattr(ls, "correct", False):
            continue
        qid = getattr(ls, "query_id", None)
        qs = query_by_qid.get(qid)
        fam = _wrong_label_family(qs, getattr(ls, "pick_index", None))
        total += 1
        if fam == "safe_diag":
            safe += 1
        elif fam == "bounded_diag":
            bounded += 1
        elif fam == "high_risk":
            high += 1
        elif fam == "far":
            far += 1
        seen = seen_wrong_by_query.setdefault(int(qid or -1), set())
        if getattr(ls, "pick_index", None) in seen:
            same_repeat += 1
        else:
            seen.add(getattr(ls, "pick_index", None))
        risks.append(float(getattr(ls, "damage", 0) or 0))
        damages.append(float(getattr(ls, "damage", 0) or 0))
    return {
        "WrongRevealTotal": total,
        "WR_safe_diag_rate": safe / max(total, 1),
        "WR_bounded_diag_rate": bounded / max(total, 1),
        "WR_highrisk_rate": high / max(total, 1),
        "WR_far_rate": far / max(total, 1),
        "WR_same_wrong_repeat_rate": same_repeat / max(total, 1),
        "MeanRiskOfWrongReveal": float(np.mean(risks)) if risks else 0.0,
        "MeanDamagePerWrongReveal": float(np.mean(damages)) if damages else 0.0,
    }


def _current_highlight_cells(tutor: SparseTutorAgent, qs, active, learner, cfg) -> Optional[Tuple[int, ...]]:
    correct = next((o for o in active if o.is_correct), None)
    if correct is None:
        return None
    cells = tutor._select_highlight_cells(qs, correct, learner)
    if cells is None:
        return None
    return tuple(cells)


def _candidate_highlight_sets(
    tutor: SparseTutorAgent,
    qs,
    active,
    learner,
    cfg,
    *,
    max_joint_combos: int = 64,
) -> List[Tuple[int, ...]]:
    variants = set()
    cur = _current_highlight_cells(tutor, qs, active, learner, cfg)
    if cur:
        variants.add(tuple(cur))
    cf = select_counterfactual_highlight_cells(
        qs, active, learner,
        max_cells=int(getattr(cfg.tutor, "max_highlight_cells", 2)),
        m_candidates=4,
    )
    if cf:
        variants.add(tuple(cf))
    L = len(getattr(qs, "target_output", []) or [])
    fixed = tuple(range(min(int(getattr(cfg.tutor, "max_highlight_cells", 2)), L)))
    if fixed:
        variants.add(fixed)
    all_combos: List[Tuple[int, ...]] = []
    for r in range(1, min(int(getattr(cfg.tutor, "max_highlight_cells", 2)), L) + 1):
        all_combos.extend(itertools.combinations(range(L), r))
    if 0 < len(all_combos) <= max_joint_combos:
        variants.update(tuple(v) for v in all_combos)
    return sorted(variants)


def _grace_bonus_scalar(qs, decomp: Dict[str, float], p_terminal: float) -> float:
    rounds_left = max(0, qs.max_rounds - qs.rounds_used)
    if rounds_left < 2:
        return 0.0
    p_correct = float(decomp.get("p_correct_action", 0.0))
    return max(0.0, (1.0 - p_correct - p_terminal) * p_correct)


def _grace_bonus_outcome(qs, active, probs, decomp: Dict[str, float]) -> float:
    rounds_left = max(0, qs.max_rounds - qs.rounds_used)
    if rounds_left < 2:
        return 0.0
    labels = getattr(qs, "option_diag_labels", {}) or {}
    recs = build_option_mass_records(
        active,
        probs,
        labels,
        last_wrong_index=getattr(qs, "last_reveal_option_index", None),
        hp_scale=max(qs.hp, 1),
    )
    p_correct = float(decomp.get("p_correct_action", 0.0))
    p_same = sum(float(r["prob"]) for r in recs if r["is_last_wrong"])
    p_safe = sum(float(r["prob"]) for r in recs if r["is_safe_diag"])
    p_bounded = sum(float(r["prob"]) for r in recs if r["is_bounded_diag"])
    p_far = sum(float(r["prob"]) for r in recs if r["is_far_wrong"])
    # Deliberately conservative: high-risk wrongs get no grace credit.
    return p_correct * (p_same + p_safe + 0.5 * p_bounded + 0.25 * p_far)


def _evaluate_spec(
    tutor: SparseTutorAgent,
    cfg,
    qs,
    active,
    learner,
    spec: Dict[str, Any],
    *,
    value_mode: Optional[str] = None,
    use_outcome_grace: bool = False,
) -> Dict[str, Any]:
    wait_probs = tutor._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
    action_probs = tutor._compute_learner_probs(qs, active, spec, learner)
    decomp = compute_postreveal_shift_decomp(
        active,
        wait_probs,
        action_probs,
        getattr(qs, "option_diag_labels", {}) or {},
        last_wrong_index=getattr(qs, "last_reveal_option_index", None),
        hp_scale=max(qs.hp, 1),
        ban_target_index=spec.get("ban_index"),
    )
    p_death = tutor._compute_p_death(qs, active, action_probs)
    p_timeout = tutor._compute_p_timeout(qs, active, action_probs)
    p_terminal = max(0.0, min(1.0, p_death + p_timeout))
    grace_bonus = (
        _grace_bonus_outcome(qs, active, action_probs, decomp)
        if use_outcome_grace
        else _grace_bonus_scalar(qs, decomp, p_terminal)
    )
    q = compute_postreveal_q(
        decomp,
        action_name=spec.get("action", "WAIT"),
        value_mode=value_mode or getattr(cfg.tutor, "postreveal_value_mode", "legacy"),
        lambda_info_post=getattr(cfg.tutor, "postreveal_info_weight", 0.0),
        grace_conversion=grace_bonus,
        cost=0.0,
    )
    return {
        "spec": spec,
        "q": float(q),
        "p_death": float(p_death),
        "p_timeout": float(p_timeout),
        "grace_bonus": float(grace_bonus),
        **decomp,
    }


def _best_joint_mix(
    tutor: SparseTutorAgent,
    cfg,
    qs,
    active,
    learner,
    candidate_indices: Sequence[int],
    highlight_sets: Sequence[Tuple[int, ...]],
    *,
    value_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    best = None
    best_q = float("-inf")
    for idx in candidate_indices:
        for hl_cells in highlight_sets:
            rec = _evaluate_spec(
                tutor, cfg, qs, active, learner,
                {"action": "MIX", "ban_index": idx, "highlight_cells": tuple(hl_cells)},
                value_mode=value_mode,
            )
            if rec["q"] > best_q:
                best_q = rec["q"]
                best = rec
    return best


def _label_postreveal_family(state: Dict[str, Any]) -> str:
    if state.get("net_oracle_drop", 0.0) > 0.02:
        return "MIX_CRITICAL"
    if state.get("hl_margin_gain", 0.0) > 0.02 or state.get("hl_delta_p", 0.0) > 0.02:
        return "HL_CRITICAL"
    if state.get("p_correct_wait", 0.0) >= 0.6 and state.get("harm_mass_wait", 0.0) < 0.1:
        return "BORING_MASTERY"
    if state.get("rounds_left", 0) >= 2 and state.get("current_mix_q", 0.0) > 0.0:
        return "GRACE_CRITICAL"
    if max(state.get("current_mix_q", 0.0), state.get("current_hl_q", 0.0), 0.0) <= 0.0:
        return "LOW_LEVERAGE"
    return "PROTECT_CRITICAL" if state.get("harm_mass_wait", 0.0) > 0.2 else "OTHER"


def _collect_replay_records(block, cfg) -> List[Dict[str, Any]]:
    tutor = SparseTutorAgent(cfg=cfg)
    out: List[Dict[str, Any]] = []
    for ac in getattr(block, "_audit_candidates", []) or []:
        learner = ac["learner_snapshot"]
        qs = ac["qs_snapshot"]
        active = ac["active"]
        non_correct = [o for o in active if not o.is_correct]
        if not non_correct:
            continue

        current_hl = _current_highlight_cells(tutor, qs, active, learner, cfg)
        if current_hl is None:
            continue
        current_target = tutor._select_ban_target(qs, non_correct, learner)
        audit = tutor._compute_mix_target_audit(qs, active, non_correct, learner)
        removed_idx = audit.get("removed_oracle_index")
        net_idx = audit.get("net_oracle_index")
        current_mix = None
        if current_target is not None:
            current_mix = _evaluate_spec(
                tutor, cfg, qs, active, learner,
                {"action": "MIX", "ban_index": current_target.index, "highlight_cells": current_hl},
            )
        current_hl_rec = _evaluate_spec(
            tutor, cfg, qs, active, learner,
            {"action": "HIGHLIGHT", "highlight_cells": current_hl},
        )
        oracle_target_mix = None
        if net_idx is not None:
            oracle_target_mix = _evaluate_spec(
                tutor, cfg, qs, active, learner,
                {"action": "MIX", "ban_index": int(net_idx), "highlight_cells": current_hl},
            )
        hl_variants = _candidate_highlight_sets(tutor, qs, active, learner, cfg)
        cf_hl = select_counterfactual_highlight_cells(
            qs, active, learner,
            max_cells=int(getattr(cfg.tutor, "max_highlight_cells", 2)),
            m_candidates=4,
        )
        oracle_hl_mix = None
        if current_target is not None and cf_hl:
            oracle_hl_mix = _evaluate_spec(
                tutor, cfg, qs, active, learner,
                {"action": "MIX", "ban_index": current_target.index, "highlight_cells": tuple(cf_hl)},
            )
        oracle_sep_mix = None
        if net_idx is not None and cf_hl:
            oracle_sep_mix = _evaluate_spec(
                tutor, cfg, qs, active, learner,
                {"action": "MIX", "ban_index": int(net_idx), "highlight_cells": tuple(cf_hl)},
            )
        joint_best = _best_joint_mix(
            tutor,
            cfg,
            qs,
            active,
            learner,
            [int(o.index) for o in non_correct],
            hl_variants,
        )

        wait_probs = tutor._compute_learner_probs(qs, active, {"action": "WAIT"}, learner)
        wait_records = build_option_mass_records(
            active,
            wait_probs,
            getattr(qs, "option_diag_labels", {}) or {},
            last_wrong_index=getattr(qs, "last_reveal_option_index", None),
            hp_scale=max(qs.hp, 1),
        )
        harm_wait = sum(float(r["removed_harm_mass"]) for r in wait_records)
        info_wait = sum(float(r["removed_info_mass"]) for r in wait_records)

        current_q = float(current_mix["q"]) if current_mix is not None else float("-inf")
        target_q = float(oracle_target_mix["q"]) if oracle_target_mix is not None else current_q
        hl_q = float(oracle_hl_mix["q"]) if oracle_hl_mix is not None else current_q
        sep_q = float(oracle_sep_mix["q"]) if oracle_sep_mix is not None else max(target_q, hl_q)
        joint_q = float(joint_best["q"]) if joint_best is not None else sep_q
        rec = {
            "query_id": qs.query_id,
            "round_t": ac["round_t"],
            "current_mix_q": current_q,
            "current_hl_q": float(current_hl_rec["q"]),
            "target_q": target_q,
            "highlight_q": hl_q,
            "separate_q": sep_q,
            "joint_q": joint_q,
            "target_regret": max(0.0, target_q - current_q) if np.isfinite(current_q) else 0.0,
            "highlight_regret": max(0.0, hl_q - current_q) if np.isfinite(current_q) else 0.0,
            "separate_regret": max(0.0, sep_q - current_q) if np.isfinite(current_q) else 0.0,
            "joint_regret": max(0.0, joint_q - current_q) if np.isfinite(current_q) else 0.0,
            "joint_interaction_regret": max(0.0, joint_q - sep_q) if np.isfinite(sep_q) else 0.0,
            "chosen_index": getattr(current_target, "index", None),
            "removed_oracle_index": removed_idx,
            "net_oracle_index": net_idx,
            "chosen_equals_removed_oracle": int(getattr(current_target, "index", None) == removed_idx) if current_target is not None and removed_idx is not None else 0,
            "chosen_equals_net_oracle": int(getattr(current_target, "index", None) == net_idx) if current_target is not None and net_idx is not None else 0,
            "removed_oracle_mass": float(audit.get("removed_oracle_mass", 0.0)),
            "net_oracle_drop": float(audit.get("net_oracle_drop", 0.0)),
            "chosen_removed_harm_mass": float(current_mix["removed_bad_mass"]) if current_mix is not None else 0.0,
            "chosen_net_harm_drop": float(current_mix["harm_mass_drop"]) if current_mix is not None else 0.0,
            "harm_mass_wait": float(harm_wait),
            "info_mass_wait": float(info_wait),
            "p_correct_wait": float(current_hl_rec["p_correct_wait"]),
            "hl_delta_p": float(current_hl_rec["delta_p_correct"]),
            "hl_margin_gain": float(current_hl_rec.get("log_margin_gain", 0.0)),
            "rounds_left": max(0, qs.max_rounds - qs.rounds_used),
        }
        rec["family"] = _label_postreveal_family(rec)
        out.append(rec)
    return out


def _forced_trace_coverage(block) -> Dict[str, float]:
    chosen_mix = 0
    with_decomp = 0
    removed_bad = []
    bad_drop = []
    for tr in getattr(block, "_decision_trace", []) or []:
        scoring = tr.get("scoring", {}) or {}
        if not scoring.get("forced"):
            continue
        chosen = scoring.get("chosen_detail", {}) or {}
        if chosen.get("action") == "MIX" or tr.get("chosen_action") == "MIX":
            chosen_mix += 1
            if "removed_bad_mass" in chosen or "postreveal_decomp" in chosen:
                with_decomp += 1
                removed_bad.append(float(chosen.get("removed_bad_mass", chosen.get("postreveal_decomp", {}).get("removed_bad_mass", 0.0))))
                bad_drop.append(float(chosen.get("bad_mass_drop", chosen.get("postreveal_decomp", {}).get("bad_mass_drop", 0.0))))
    return {
        "n_chosen_mix": chosen_mix,
        "n_mix_with_decomp_trace": with_decomp,
        "trace_coverage_rate": with_decomp / max(chosen_mix, 1),
        "mean_removed_bad_mass": float(np.mean(removed_bad)) if removed_bad else 0.0,
        "mean_badmass_drop": float(np.mean(bad_drop)) if bad_drop else 0.0,
    }


def _format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["000001", "000002", "000003", "000004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--generator", default="diagnostic_quota")
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "SIS_horizon_self_correct_cf_mix",
            "SIS_force_MIX_cf_after_safe_diag_reveal",
            "no_tutor_reveal",
            "w1_sc_diagnostic_safe",
        ],
    )
    ap.add_argument("--max-states", type=int, default=120)
    ap.add_argument("--out", default=os.path.join("cls_option_tutor", "results", "e6_micro", "phase6i6b_localization_report.md"))
    args = ap.parse_args()

    replay_records: List[Dict[str, Any]] = []
    wrong_stats: Dict[str, List[Dict[str, float]]] = {}
    forced_cov: List[Dict[str, float]] = []

    for condition in args.conditions:
        wrong_stats[condition] = []
        for task_id in args.tasks:
            for seed in args.seeds:
                block, eff_cfg = _run_teach_block(task_id, seed, condition, rho=args.rho, generator=args.generator)
                wrong_stats[condition].append(_collect_wrong_reveal_stats(block))
                if "force_mix" in condition.lower():
                    forced_cov.append(_forced_trace_coverage(block))
                if "cf_mix" in condition.lower() and len(replay_records) < args.max_states:
                    recs = _collect_replay_records(block, eff_cfg)
                    remaining = max(0, args.max_states - len(replay_records))
                    replay_records.extend(recs[:remaining])

    out_lines = ["# Phase 6I.6B Localization Report", ""]

    if forced_cov:
        total_chosen = int(sum(r["n_chosen_mix"] for r in forced_cov))
        total_with_decomp = int(sum(r["n_mix_with_decomp_trace"] for r in forced_cov))
        weighted_removed_num = sum(
            float(r["mean_removed_bad_mass"]) * int(r["n_mix_with_decomp_trace"]) for r in forced_cov
        )
        weighted_drop_num = sum(
            float(r["mean_badmass_drop"]) * int(r["n_mix_with_decomp_trace"]) for r in forced_cov
        )
        cov = {
            "n_chosen_mix": total_chosen,
            "n_mix_with_decomp_trace": total_with_decomp,
            "trace_coverage_rate": total_with_decomp / max(total_chosen, 1),
            "mean_removed_bad_mass": weighted_removed_num / max(total_with_decomp, 1),
            "mean_badmass_drop": weighted_drop_num / max(total_with_decomp, 1),
        }
        out_lines.append("## Forced Trace Coverage")
        out_lines.extend(_format_table(
            ["n_chosen_mix", "n_mix_with_decomp_trace", "trace_coverage_rate", "mean_removed_bad_mass", "mean_badmass_drop"],
            [[cov["n_chosen_mix"], cov["n_mix_with_decomp_trace"], f"{cov['trace_coverage_rate']:.4f}", f"{cov['mean_removed_bad_mass']:.4f}", f"{cov['mean_badmass_drop']:.4f}"]],
        ))
        out_lines.append("")

    out_lines.append("## no_tutor / cf_mix Wrong Reveal Comparison")
    reveal_rows = []
    for condition, stats_list in wrong_stats.items():
        if not stats_list:
            continue
        agg = {
            k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0].keys()
        }
        reveal_rows.append([
            condition,
            f"{agg['WrongRevealTotal']:.2f}",
            f"{agg['WR_safe_diag_rate']:.4f}",
            f"{agg['WR_bounded_diag_rate']:.4f}",
            f"{agg['WR_highrisk_rate']:.4f}",
            f"{agg['WR_far_rate']:.4f}",
            f"{agg['WR_same_wrong_repeat_rate']:.4f}",
            f"{agg['MeanRiskOfWrongReveal']:.4f}",
            f"{agg['MeanDamagePerWrongReveal']:.4f}",
        ])
    out_lines.extend(_format_table(
        [
            "condition", "wrong_total", "safe_diag", "bounded", "highrisk",
            "far", "same_repeat", "mean_risk", "mean_damage",
        ],
        reveal_rows,
    ))
    out_lines.append("")

    if replay_records:
        out_lines.append("## Common-State Replay")
        current_mix_q = np.mean([r["current_mix_q"] for r in replay_records if np.isfinite(r["current_mix_q"])]) if replay_records else 0.0
        target_regret = np.mean([r["target_regret"] for r in replay_records]) if replay_records else 0.0
        highlight_regret = np.mean([r["highlight_regret"] for r in replay_records]) if replay_records else 0.0
        joint_regret = np.mean([r["joint_regret"] for r in replay_records]) if replay_records else 0.0
        joint_interaction = np.mean([r["joint_interaction_regret"] for r in replay_records]) if replay_records else 0.0
        out_lines.extend(_format_table(
            ["n_states", "current_mix_q", "target_regret", "highlight_regret", "joint_regret", "joint_interaction_regret"],
            [[len(replay_records), f"{current_mix_q:.4f}", f"{target_regret:.4f}", f"{highlight_regret:.4f}", f"{joint_regret:.4f}", f"{joint_interaction:.4f}"]],
        ))
        out_lines.append("")
        out_lines.append("## MIX Oracle / Target Audit")
        chosen_net = np.mean([r["chosen_equals_net_oracle"] for r in replay_records]) if replay_records else 0.0
        chosen_removed = np.mean([r["chosen_equals_removed_oracle"] for r in replay_records]) if replay_records else 0.0
        chosen_drop = np.mean([r["chosen_net_harm_drop"] for r in replay_records]) if replay_records else 0.0
        oracle_drop = np.mean([r["net_oracle_drop"] for r in replay_records]) if replay_records else 0.0
        out_lines.extend(_format_table(
            ["chosen_equals_removed_oracle", "chosen_equals_net_oracle", "chosen_net_harm_drop", "oracle_net_harm_drop"],
            [[f"{chosen_removed:.4f}", f"{chosen_net:.4f}", f"{chosen_drop:.4f}", f"{oracle_drop:.4f}"]],
        ))
        out_lines.append("")
        out_lines.append("## Family Tagging")
        family_rows = []
        for family in sorted({r["family"] for r in replay_records}):
            subset = [r for r in replay_records if r["family"] == family]
            family_rows.append([
                family,
                len(subset),
                f"{np.mean([r['current_mix_q'] for r in subset if np.isfinite(r['current_mix_q'])]) if subset else 0.0:.4f}",
                f"{np.mean([r['target_regret'] for r in subset]) if subset else 0.0:.4f}",
                f"{np.mean([r['joint_regret'] for r in subset]) if subset else 0.0:.4f}",
            ])
        out_lines.extend(_format_table(
            ["family", "state_count", "current_mix_q", "target_regret", "joint_regret"],
            family_rows,
        ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
