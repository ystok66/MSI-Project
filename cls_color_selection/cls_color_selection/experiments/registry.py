"""
registry.py — Experiment condition registry.

Maps condition names to config overrides for systematic ablation.
"""
from __future__ import annotations
from typing import Any, Dict

# Each entry: condition_name → dict of nested config overrides
# e.g. {'learner.feedback_mode': 'wrong_only', 'learner.alpha_risk': 0.0}
REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Sanity ──
    'sanity_basic': {},

    # ── Grammar ablation (Exp-L1) ──
    'grammar_no_feedback': {
        'learner.feedback_mode': 'none',
    },
    'grammar_wrong_only': {
        'learner.feedback_mode': 'wrong_only',
    },
    'grammar_wrong_positions': {
        'learner.feedback_mode': 'wrong_positions',
    },

    # ── Risk ablation (Exp-L2) ──
    'risk_none': {
        'learner.alpha_risk': 0.0,
    },
    'risk_death_only': {
        'learner.alpha_risk': 0.5,
        # No warning — use NoTutor
        '_tutor_class': 'NoTutor',
    },
    'risk_warning_bayes': {
        'learner.alpha_risk': 0.5,
        '_tutor_class': 'OracleWarningTutor',
    },
    'risk_warning_courage': {
        'learner.alpha_risk': 0.5,
        'learner.enable_courage': True,
        '_tutor_class': 'OracleWarningTutor',
    },

    # ── Joint (Exp-L3) ──
    'joint_grammar_only': {
        'learner.alpha_risk': 0.0,
        'learner.feedback_mode': 'wrong_positions',
    },
    'joint_risk_only': {
        'learner.feedback_mode': 'none',
        'learner.alpha_risk': 0.5,
        '_tutor_class': 'OracleWarningTutor',
    },
    'joint_grammar_risk': {
        'learner.feedback_mode': 'wrong_positions',
        'learner.alpha_risk': 0.5,
        '_tutor_class': 'OracleWarningTutor',
    },

    # ── Feedback mode ablation (A1) ──
    'ablation_fb_wrong_only': {
        'learner.feedback_mode': 'wrong_only',
    },
    'ablation_fb_wrong_positions': {
        'learner.feedback_mode': 'wrong_positions',
    },

    # ── Danger type ablation (A2) ──
    'ablation_danger_1type': {
        'env.n_danger_types': 1,
    },
    'ablation_danger_3types': {
        'env.n_danger_types': 3,
    },

    # ── Courage ablation (A3) ──
    'ablation_courage_off': {
        'learner.enable_courage': False,
    },
    'ablation_courage_on': {
        'learner.enable_courage': True,
        'learner.n_retry_courage': 5,
    },

    # ── HPC ablation (A4) ──
    'ablation_hpc_off': {
        'learner.use_hpc': False,
    },
    'ablation_hpc_on': {
        'learner.use_hpc': True,
    },

    # ── Robustness: sigma sweep (R1) ──
    'robust_sigma_01': {'env.obs_sigma': 0.1},
    'robust_sigma_03': {'env.obs_sigma': 0.3},
    'robust_sigma_05': {'env.obs_sigma': 0.5},
    'robust_sigma_08': {'env.obs_sigma': 0.8},
    'robust_sigma_10': {'env.obs_sigma': 1.0},

    # ── Robustness: support sparsity (R2) ──
    'robust_nsup_2': {'learner.n_sup': 2},
    'robust_nsup_4': {'learner.n_sup': 4},
    'robust_nsup_8': {'learner.n_sup': 8},
    'robust_nsup_14': {'learner.n_sup': 14},

    # ── Robustness: danger ratio (R3) ──
    'robust_danger_01': {'env.danger_ratio': 0.1},
    'robust_danger_03': {'env.danger_ratio': 0.3},
    'robust_danger_05': {'env.danger_ratio': 0.5},
}


def apply_overrides(cfg, overrides: Dict[str, Any]) -> None:
    """Apply flat key overrides to a FullConfig object.

    Keys are dotted paths like 'learner.feedback_mode'.
    Keys starting with '_' are metadata, not config fields.
    """
    for key, value in overrides.items():
        if key.startswith('_'):
            continue
        parts = key.split('.')
        if len(parts) == 2:
            section, field = parts
            sub = getattr(cfg, section, None)
            if sub is not None and hasattr(sub, field):
                setattr(sub, field, value)
