"""Condition-name parsing helpers for learning-increment experiments.

Naming policy:
  - short active-mainline aliases live in ``mainline_registry.py``
  - this module only resolves aliases and mutates config objects
  - long canonical tags remain supported for backward-compatible replay
"""

from __future__ import annotations

from .mainline_registry import CONDITION_ALIASES, RUNTIME_OVERRIDE_TAGS


def resolve_condition_alias(condition: str) -> str:
    """Map short aliases to canonical condition names, preserving suffix tags."""
    for alias, canonical in sorted(CONDITION_ALIASES.items(), key=lambda item: -len(item[0])):
        if condition == alias:
            return canonical
        if condition.startswith(alias + "_"):
            return canonical + condition[len(alias):]
    return condition


def extract_scripted_protocol_name(condition: str) -> str:
    """Strip runtime override suffixes from scripted/no-tutor protocol names."""
    protocol = resolve_condition_alias(condition)
    changed = True
    while changed:
        changed = False
        for tag in RUNTIME_OVERRIDE_TAGS:
            suffix = "_" + tag
            if protocol.endswith(suffix):
                protocol = protocol[: -len(suffix)]
                changed = True
    return protocol


def apply_condition_overrides(cfg, condition: str):
    """Apply config overrides encoded in condition names."""
    c = condition.lower()
    tokens = set(c.split("_"))

    # Highlight mode
    if "cf_highlight" in c or "cf_mix" in c:
        cfg.env.highlight_mode = "counterfactual_pcorrect"
    elif "fixed_highlight" in c:
        cfg.env.highlight_mode = "fixed"
    elif "no_highlight" in c:
        cfg.env.highlight_mode = "none"
    else:
        cfg.env.highlight_mode = "diagnostic"

    # Learning-gain mode
    if "horizon_self_correct" in c:
        cfg.tutor.tutor_lg_mode = "horizon_self_correct"
    elif "self_correct" in c:
        cfg.tutor.tutor_lg_mode = "self_correct"
    elif "no_lg" in c:
        cfg.tutor.tutor_lg_mode = "off"
    elif "safety_only" in c:
        cfg.tutor.tutor_lg_mode = "safety_only"
    elif "learning_only" in c:
        cfg.tutor.tutor_lg_mode = "learning_only"
    elif "inverse_shadow" in c:
        cfg.tutor.tutor_lg_mode = "diagnostic"

    # Post-reveal trajectory-value routing
    cfg.tutor.postreveal_value_mode = "legacy"
    if "traj_v2" in c or "netbadmass" in c or "harminfo" in c or "bayes" in c or "netharm_direct" in c:
        cfg.tutor.postreveal_value_mode = "traj_v2"
    elif "cf_mix" in c or "traj_v1" in c or "qrepair" in c:
        cfg.tutor.postreveal_value_mode = "traj_v1"
    cfg.tutor.mix_target_mode = "current"
    if "netbadmass" in c or "netharm_direct" in c:
        cfg.tutor.mix_target_mode = "net_badmass"
    elif "removedbadmass" in c or "badmass" in c:
        cfg.tutor.mix_target_mode = "removed_badmass"
    cfg.tutor.postreveal_info_weight = 0.0
    if "harminfo" in c:
        cfg.tutor.postreveal_info_weight = 0.25
    cfg.tutor.use_postreveal_consolidation_value = (
        "consolidate" in tokens or "consolidateq" in tokens
    )
    cfg.tutor.promote_postreveal_consolidation_into_gexp = (
        "consolidateq" in tokens or "loopv2" in tokens
    )
    cfg.tutor.use_bayesian_postreveal_value = "bayes" in c
    cfg.tutor.joint_mix_replay_gate = ("jointmix" in c or "joint_gate" in c)
    cfg.tutor.direct_mix_selector = ("netharm_direct" in c or "directmix" in c)
    # `allowctl` implies productive-allow planning which in turn implies the
    # direct MIX selector should be active for the cf_mix controlled mainline.
    cfg.tutor.productive_allow_planning = (
        "netharm_direct_allow" in c
        or "productive_allow" in c
        or "allowctl" in tokens
        or "nativeallow" in tokens
        or "nativeallowv1" in tokens
        or "nativelike" in tokens
    )
    cfg.tutor.productive_allow_mode = "phase"
    # Keep v2 aliases ahead of v1 aliases. Tokens are split on "_", so
    # "allowctl2" is distinct from "allowctl", but this ordering documents the
    # intended precedence for future alias additions.
    if (
        "nativeallow" in tokens
        or "nativeallowv1" in tokens
        or "nativelike" in tokens
    ):
        cfg.tutor.productive_allow_mode = "native_like_v1"
    elif (
        "allowctl2" in tokens
        or "allowctlv2" in tokens
        or "productiveallowv2" in tokens
    ):
        cfg.tutor.productive_allow_mode = "controlled_v2"
    elif "allowctl" in tokens or "productiveallowv1" in tokens:
        cfg.tutor.productive_allow_mode = "controlled_v1"
    if cfg.tutor.productive_allow_planning:
        cfg.tutor.direct_mix_selector = True
    cfg.tutor.phase_allow_family_override = ("phasecalib" in tokens)

    # Highlight strength
    if "highlight_4x" in c:
        cfg.env.highlight_strength = 4.0
    elif "highlight_2x" in c:
        cfg.env.highlight_strength = 2.0
    else:
        cfg.env.highlight_strength = 1.0

    # Highlight cell budget
    if "cells4" in c:
        cfg.tutor.max_highlight_cells = 4
    elif "cells3" in c:
        cfg.tutor.max_highlight_cells = 3
    elif "cells2" in c:
        cfg.tutor.max_highlight_cells = 2

    # Oracle horizon: allow reading counterfactual CATE directly for scoring
    cfg.tutor.oracle_horizon = "oracle_horizon" in c

    cfg.env.diagnostic_quota_strict = ("strict" in c)
    cfg.tutor.protect_safe_diag_hard_guard = "protect" in c

    # Runtime ablations shared by sparse/scripted/no_tutor conditions.
    if "tmax5" in tokens:
        cfg.env.T_max = 5
    elif "tmax4" in tokens:
        cfg.env.T_max = 4

    if "er05" in tokens:
        cfg.learner.eta_reveal = 0.5
    elif "er0" in tokens:
        cfg.learner.eta_reveal = 0.0

    if "cpoff" in tokens:
        cfg.learner.correct_pick_learning_mode = "off"
    if "budgeted" in tokens:
        cfg.learner.pedagogical_feedback_mode = "budgeted_v1"
    if "rfcap" in tokens or "refreshcap" in tokens:
        cfg.env.enforce_max_refreshes = True

    # Phase 6I.1 P3: forced post-reveal intervention
    # force_hl_cf: force HIGHLIGHT_cf after safe diag reveal
    # force_mix_cf: force MIX_cf after safe diag reveal
    # force_best_cate: force whichever of HL_cf/MIX_cf has better local CATE
    cfg.tutor.force_postreveal_action = "none"
    if "force_mix_cf" in c:
        cfg.tutor.force_postreveal_action = "MIX_cf"
    elif "force_hl_cf" in c:
        cfg.tutor.force_postreveal_action = "HL_cf"
    elif "force_best_cate" in c:
        cfg.tutor.force_postreveal_action = "best_CATE"

    # Forced post-reveal cue ceilings are still self-correct tutor scenarios.
    # If no explicit LG mode was requested in the condition name, keep them on
    # the self_correct mainline rather than silently downgrading to "off".
    if cfg.tutor.force_postreveal_action != "none" and cfg.tutor.tutor_lg_mode == "off":
        cfg.tutor.tutor_lg_mode = "self_correct"

    return cfg
