"""Central registry for active mainline experiment aliases and runtime tags.

This module is the single source of truth for the frozen active mainline used
throughout Phase 6I+ experiments. Keeping aliases here prevents long condition
tags from leaking into tests, reports, and package docs.
"""

from __future__ import annotations

from typing import Dict, Tuple

ACTIVE_MAINLINE_ALIAS = "SIS_cf_mix_loop_v1"
ACTIVE_MAINLINE_CANONICAL = (
    "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
    "budgeted_allowctl2_consolidate_tmax5"
)

ACTIVE_MAINLINE_NO_CONSOLIDATE_ALIAS = "SIS_cf_mix_loop_v1_no_consolidate"
ACTIVE_MAINLINE_NO_CONSOLIDATE_CANONICAL = (
    "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
    "budgeted_allowctl2_tmax5"
)

ACTIVE_MAINLINE_Q_PROMOTED_ALIAS = "SIS_cf_mix_loop_v2"
ACTIVE_MAINLINE_Q_PROMOTED_CANONICAL = (
    "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
    "budgeted_allowctl2_consolidateq_tmax5"
)

ACTIVE_MAINLINE_NATIVEALLOW_ALIAS = "SIS_cf_mix_loop_v1_nativeallow"
ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL = (
    "SIS_horizon_self_correct_cf_mix_netharm_direct_allow_"
    "budgeted_allowctl2_consolidate_tmax5_nativeallow"
)

SCRIPTED_SAFE_GOLD_CONDITION = "w1_sc_diagnostic_safe_budgeted_tmax5"
NO_TUTOR_BUDGETED_CONDITION = "no_tutor_reveal_budgeted_tmax5"

ACTIVE_BASELINE_CONDITIONS: Tuple[str, ...] = (
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)

CONDITION_ALIASES: Dict[str, str] = {
    "w1_sc_diagnostic_safe": "script_wrong1_self_correct_diagnostic_safe",
    ACTIVE_MAINLINE_ALIAS: ACTIVE_MAINLINE_CANONICAL,
    ACTIVE_MAINLINE_ALIAS.lower(): ACTIVE_MAINLINE_CANONICAL,
    ACTIVE_MAINLINE_NO_CONSOLIDATE_ALIAS: ACTIVE_MAINLINE_NO_CONSOLIDATE_CANONICAL,
    ACTIVE_MAINLINE_NO_CONSOLIDATE_ALIAS.lower(): (
        ACTIVE_MAINLINE_NO_CONSOLIDATE_CANONICAL
    ),
    ACTIVE_MAINLINE_Q_PROMOTED_ALIAS: ACTIVE_MAINLINE_Q_PROMOTED_CANONICAL,
    ACTIVE_MAINLINE_Q_PROMOTED_ALIAS.lower(): ACTIVE_MAINLINE_Q_PROMOTED_CANONICAL,
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS: ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL,
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS.lower(): ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL,
}

RUNTIME_OVERRIDE_TAGS: Tuple[str, ...] = (
    "tmax4",
    "tmax5",
    "er05",
    "er0",
    "cpoff",
    "budgeted",
    "consolidate",
    "consolidateq",
    "rfcap",
    "refreshcap",
    "allowctl",
    "allowctl2",
    "allowctlv2",
    "productiveallowv2",
    "phasecalib",
    "nativeallow",
    "nativeallowv1",
    "nativelike",
)

__all__ = [
    "ACTIVE_BASELINE_CONDITIONS",
    "ACTIVE_MAINLINE_ALIAS",
    "ACTIVE_MAINLINE_CANONICAL",
    "ACTIVE_MAINLINE_NO_CONSOLIDATE_ALIAS",
    "ACTIVE_MAINLINE_NO_CONSOLIDATE_CANONICAL",
    "ACTIVE_MAINLINE_NATIVEALLOW_ALIAS",
    "ACTIVE_MAINLINE_NATIVEALLOW_CANONICAL",
    "ACTIVE_MAINLINE_Q_PROMOTED_ALIAS",
    "ACTIVE_MAINLINE_Q_PROMOTED_CANONICAL",
    "CONDITION_ALIASES",
    "NO_TUTOR_BUDGETED_CONDITION",
    "RUNTIME_OVERRIDE_TAGS",
    "SCRIPTED_SAFE_GOLD_CONDITION",
]
