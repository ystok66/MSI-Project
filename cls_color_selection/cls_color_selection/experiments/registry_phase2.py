"""
registry_phase2.py — Phase 2 experiment condition registry.

Covers all experiment categories from the Phase 2 specification:
  S2: sanity, tutor compare, observation ablation
  B2: belief estimator, type inference
  R2: robustness
"""
from __future__ import annotations
from typing import Any, Dict


REGISTRY_P2: Dict[str, Dict[str, Any]] = {
    # ── Sanity ──
    'sanity_p2': {},

    # ── Baselines (Phase 1 carry-forward) ──
    'no_tutor': {
        'tutor.tutor_policy_mode': 'none',
        'tutor.use_observation_phase': False,
    },
    'oracle_warning_only': {
        'tutor.tutor_policy_mode': 'oracle_warning',
        'tutor.use_observation_phase': False,
    },

    # ── Tutor compare (core experiment) ──
    'tutor_rule': {
        'tutor.tutor_policy_mode': 'rule',
    },
    'tutor_proxy': {
        'tutor.tutor_policy_mode': 'proxy',
    },
    'tutor_rule_no_obs': {
        'tutor.tutor_policy_mode': 'rule',
        'tutor.use_observation_phase': False,
    },
    'tutor_proxy_no_obs': {
        'tutor.tutor_policy_mode': 'proxy',
        'tutor.use_observation_phase': False,
    },

    # ── Observation ablation ──
    'obs_off': {
        'tutor.use_observation_phase': False,
    },
    'obs_on_n2': {
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 2,
    },
    'obs_on_n4': {
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 4,
    },
    'obs_on_n8': {
        'tutor.use_observation_phase': True,
        'exp.n_obs_queries': 8,
    },

    # ── Belief estimator compare ──
    'belief_probe': {
        'belief.sem_estimator': 'probe',
    },
    'belief_surrogate': {
        'belief.sem_estimator': 'surrogate',
    },

    # ── Type inference ──
    'type_off': {
        'belief.enable_type_inference': False,
    },
    'type_on': {
        'belief.enable_type_inference': True,
    },

    # ── Hint ablation ──
    'hint_off': {
        'tutor.hint_after_confirm_fail': False,
    },
    'hint_on_max1': {
        'tutor.hint_after_confirm_fail': True,
        'tutor.max_hint_balls': 1,
    },
    'hint_on_max2': {
        'tutor.hint_after_confirm_fail': True,
        'tutor.max_hint_balls': 2,
    },

    # ── Courage ablation ──
    'courage_off': {
        'tutor.n_retry_courage': 999,  # effectively disabled
    },
    'courage_on_n3': {
        'tutor.n_retry_courage': 3,
    },
    'courage_on_n5': {
        'tutor.n_retry_courage': 5,
    },

    # ── Utility weight ablation ──
    'util_death_heavy': {
        'tutor.lambda_death': 5.0,
        'tutor.lambda_teach': 1.0,
    },
    'util_teach_heavy': {
        'tutor.lambda_death': 1.0,
        'tutor.lambda_teach': 3.0,
    },
    'util_over_heavy': {
        'tutor.lambda_over': 2.0,
    },
    'util_over_light': {
        'tutor.lambda_over': 0.3,
    },

    # ── Robustness: learner heterogeneity ──
    'robust_learner_balanced': {
        'learner.alpha_risk': 0.5,
        'learner.epsilon_policy': 0.05,
    },
    'robust_learner_risk_averse': {
        'learner.alpha_risk': 1.5,
        'learner.epsilon_policy': 0.01,
    },
    'robust_learner_slow': {
        'learner.alpha_risk': 0.3,
        'learner.epsilon_policy': 0.15,
    },

    # ── Robustness: support scarcity ──
    'robust_nsup_2': {'learner.n_sup': 2},
    'robust_nsup_4': {'learner.n_sup': 4},
    'robust_nsup_8': {'learner.n_sup': 8},
    'robust_nsup_14': {'learner.n_sup': 14},

    # ── Robustness: danger density ──
    'robust_danger_low': {'env.danger_ratio': 0.1},
    'robust_danger_mid': {'env.danger_ratio': 0.3},
    'robust_danger_high': {'env.danger_ratio': 0.5},
}


def apply_overrides(cfg, overrides: Dict[str, Any]) -> None:
    """Apply flat key overrides to FullConfig."""
    for key, value in overrides.items():
        if key.startswith('_'):
            continue
        parts = key.split('.')
        if len(parts) == 2:
            section, field = parts
            sub = getattr(cfg, section, None)
            if sub is not None and hasattr(sub, field):
                setattr(sub, field, value)
