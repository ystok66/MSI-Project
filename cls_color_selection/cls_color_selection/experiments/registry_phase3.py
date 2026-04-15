"""
registry_phase3.py — Phase 3 experiment condition registry.

Covers: tutor hierarchy, shadow fidelity, obs ablation, robustness.
"""
from __future__ import annotations
from typing import Any, Dict


REGISTRY_P3: Dict[str, Dict[str, Any]] = {
    # ── Baselines ──
    # Baseline 1: death = fail, normal timeout
    'no_tutor': {
        'tutor.tutor_policy_mode': 'none',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
    },
    # Baseline 2: immortal (warning on danger, learns risk), normal timeout
    'no_tutor_immortal_warnlike': {
        'tutor.tutor_policy_mode': 'immortal_warnlike',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
    },
    # Baseline 3: immortal + unlimited confirms (upper bound)
    'no_tutor_immortal_no_timeout': {
        'tutor.tutor_policy_mode': 'immortal_no_timeout',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
        'env.n_confirm_max': 999,
    },
    # Generated-query versions
    'no_tutor_generated': {
        'tutor.tutor_policy_mode': 'none',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'no_tutor_immortal_warnlike_gen': {
        'tutor.tutor_policy_mode': 'immortal_warnlike',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'no_tutor_immortal_no_timeout_gen': {
        'tutor.tutor_policy_mode': 'immortal_no_timeout',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
        'env.n_confirm_max': 999,
        'exp.query_source_mode': 'generated',
    },
    'T0_rule': {
        'tutor.tutor_policy_mode': 'rule',
        '_shadow_fidelity': 'none',
    },
    'T1_proxy': {
        'tutor.tutor_policy_mode': 'proxy',
        '_shadow_fidelity': 'none',
    },

    # ── T2: Oracle Shadow tutor (full-access upper bound) ──
    # NOTE: T2 deep-copies learner internal state. This is an oracle/debug
    # baseline, NOT a realistic ToM tutor. See T3 for realistic version.
    'T2_exact': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
    },
    'T2_oracle': {   # alias for T2_exact with explicit naming
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
    },
    'T2_compressed': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'compressed',
    },

    # ── T3: Behavioral tutor (realistic ToM, behavior-only) ──
    # T3 builds its OWN grammar independently, infers learner competence
    # from observed behavior. Does NOT read learner internal state.
    'T3_behavioral': {
        'tutor.tutor_policy_mode': 'behavioral',
        'tutor.use_observation_phase': True,
        '_shadow_fidelity': 'none',
    },
    'T3_behavioral_gen': {
        'tutor.tutor_policy_mode': 'behavioral',
        'tutor.use_observation_phase': True,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'T3_hint_on': {
        'tutor.tutor_policy_mode': 'behavioral',
        'tutor.use_observation_phase': True,
        'tutor.hint_after_confirm_fail': True,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'T3_hint_off': {
        'tutor.tutor_policy_mode': 'behavioral',
        'tutor.use_observation_phase': True,
        'tutor.hint_after_confirm_fail': False,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },

    # ── Shadow fidelity ──
    'shadow_exact_obs4': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 4,
    },
    'shadow_compressed_obs4': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'compressed',
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 4,
    },

    # ── Obs ablation under T2 ──
    'T2_exact_obs_off': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'tutor.use_observation_phase': False,
    },
    'T2_exact_obs_n4': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 4,
    },

    # ── Hint ablation ──
    'T2_hint_off': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'tutor.hint_after_confirm_fail': False,
    },
    'T2_hint_on': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'tutor.hint_after_confirm_fail': True,
    },

    # ── Robustness: support scarcity ──
    'T2_nsup_4': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'learner.n_sup': 4,
    },
    'T2_nsup_8': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'learner.n_sup': 8,
    },
    'T2_nsup_14': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'learner.n_sup': 14,
    },

    # ── Robustness: danger density ──
    'T2_danger_low': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'env.danger_ratio': 0.1,
    },
    'T2_danger_mid': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'env.danger_ratio': 0.3,
    },
    'T2_danger_high': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'env.danger_ratio': 0.5,
    },

    # ── Robustness: learner heterogeneity ──
    'T2_learner_balanced': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'learner.alpha_risk': 0.5,
    },
    'T2_learner_risk_averse': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'learner.alpha_risk': 1.5,
    },

    # ── Query source ablation ──
    'T2_txt_only': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'exp.query_source_mode': 'txt_only',
    },
    'T2_txt_resample': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'exp.query_source_mode': 'txt_resample',
    },
    'T2_generated': {
        'tutor.tutor_policy_mode': 'shadow',
        '_shadow_fidelity': 'exact',
        'exp.query_source_mode': 'generated',
    },
    'T0_generated': {
        'tutor.tutor_policy_mode': 'rule',
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'T1_generated': {
        'tutor.tutor_policy_mode': 'proxy',
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
    'no_tutor_generated': {
        'tutor.tutor_policy_mode': 'none',
        'tutor.use_observation_phase': False,
        '_shadow_fidelity': 'none',
        'exp.query_source_mode': 'generated',
    },
}


def apply_overrides(cfg, overrides: Dict[str, Any]) -> None:
    """Apply flat key overrides to FullConfig."""
    for key, value in overrides.items():
        if key.startswith('_'):
            continue
        parts = key.split('.')
        if len(parts) == 2:
            section, fld = parts
            sub = getattr(cfg, section, None)
            if sub is not None and hasattr(sub, fld):
                setattr(sub, fld, value)
