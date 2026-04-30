"""
config.py — Layered configuration for cls_color_selection.

Follows the FullConfig(env, learner, tutor, exp) pattern from cls_option_tutor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml
import os


@dataclass
class EnvConfig:
    """Environment parameters."""
    # Candidate pool
    n_candidates: int = 8           # balls per candidate pool refresh
    # Timeout
    n_confirm_max: int = 5          # confirm attempts before timeout
    max_retry_per_confirm_window: int = 10  # stuck-retry guard (log only in Phase 1)
    # Risk model
    danger_dim: int = 10            # dimension of risk vector
    n_safe_types: int = 1           # number of safe prototype clusters
    n_danger_types: int = 3         # number of danger type clusters
    danger_ratio: float = 0.3       # P(danger) per generated ball
    obs_sigma: float = 0.3          # observation noise σ on danger vectors
    cluster_sigma: float = 0.5      # intra-cluster noise for prototypes


@dataclass
class LearnerConfig:
    """Learner policy and learning parameters."""
    # ── CLS grammar ──
    n_sup: int = 14                 # support examples to learn from
    n_em: int = 3                   # EM iterations for CLS
    use_hpc: bool = False           # Phase 1 default: off (ablation A4 tests on)
    cls_mode: str = 'ast'           # 'ast' or 'stack'

    # ── Risk belief ──
    risk_prior_safe: float = 0.7    # prior P(z_i = 0) per ball
    risk_update_lr: float = 0.1     # prototype moment-matching learning rate

    # ── Feedback ──
    feedback_mode: str = 'wrong_positions'  # 'wrong_only' | 'wrong_positions'
    eta_fb: float = 1.0             # feedback differential M-step learning rate
    eps_wrong: float = 0.01         # ε for wrong_only exact-match
    eps_eq: float = 0.05            # ε for position-match probability
    beta_err: float = 2.0           # distance scaling for wrong_only soft version
    rho_assist: float = 0.3         # assist evidence discount (0=ignore hints, 1=no discount)

    # ── Policy ──
    alpha_fill: float = 1.0         # fill-gap utility weight
    alpha_risk: float = 2.0         # risk penalty weight
    alpha_waste: float = 0.3        # waste/redundant-color penalty weight
    confirm_fill_threshold: float = 1.0  # confirm when fill ratio ≥ this
    beta_policy: float = 4.0        # softmax temperature for select scoring
    epsilon_policy: float = 0.05    # exploration lapse rate
    risk_gate_tau: float = 0.0      # 0 = disabled; >0 = skip needed balls with p_danger > tau
                                    # if a safer needed alternative exists

    # ── Courage ──
    enable_courage: bool = False    # Phase 1 default: off (ablation A3 tests on)
    n_retry_courage: int = 5        # consecutive retries before courage trigger

    # ── Hint-aware inference (Step 1: hint bias) ──
    enable_hint_bias: bool = False   # if True, hint payload biases target prediction
    hint_infer_mode: str = 'hard'    # 'hard' = filter traces, 'soft' = reweight
    beta_hint: float = 2.0           # soft-mode reweighting strength

    # ── Hint-induced autonomy shift (Step 2) ──
    enable_hint_autonomy_shift: bool = False  # if True, hints change policy behavior
    hint_confirm_bonus: float = 0.25          # lower confirm threshold after hint
    hint_exploration_drop: float = 0.8        # reduce exploration epsilon after hint
    hint_stop_shift: float = 0.5              # raise util stop bar after hint


@dataclass
class TutorConfig:
    """Tutor parameters — Phase 2: warning / hint / courage decisions."""
    enabled: bool = True
    # ── Observation phase ──
    use_observation_phase: bool = True
    n_obs: int = 4                  # observation queries before teaching
    # ── Tutor policy mode ──
    tutor_policy_mode: str = 'rule'  # 'rule' | 'proxy' | 'short_rollout'
    # ── Warning ──
    tau_warn: float = 0.0           # P(∃danger)>τ → WARNING (0 = always warn)
    # ── Courage ──
    tau_courage: float = 0.5        # P(∃safe-needed)>τ → COURAGE
    n_retry_courage: int = 5        # min retries before courage considered
    # ── Hint ──
    max_hint_balls: int = 2         # max balls per HINT action
    hint_after_confirm_fail: bool = True  # only hint after failed confirm
    # ── Utility weights (for proxy tutor) ──
    lambda_eval: float = 1.0        # eval generalization gain
    lambda_teach: float = 1.5       # current teaching success gain
    lambda_death: float = 3.0       # death prevention gain
    lambda_to: float = 1.0          # timeout prevention gain
    lambda_over: float = 0.8        # over-help penalty
    lambda_int: float = 0.2         # intervention fixed cost


@dataclass
class BeliefConfig:
    """Tutor belief model parameters."""
    # ── B_sem: grammar competence ──
    sem_estimator: str = 'surrogate'  # 'probe' | 'surrogate'
    n_probe_tutor: int = 4            # held-out probes for probe estimator
    sem_beta_prior: tuple = (1.0, 1.0)  # Beta prior (α, β) for success rate
    # ── B_risk: risk competence ──
    risk_beta_prior: tuple = (1.0, 1.0)  # Beta prior for danger detection rate
    over_beta_prior: tuple = (1.0, 3.0)  # Beta prior for over-avoidance (skewed safe)
    # ── B_type: learner type ──
    enable_type_inference: bool = False    # Phase 2: off by default
    type_set: list = None                  # populated at init
    type_prior: list = None               # uniform by default

    def __post_init__(self):
        if self.type_set is None:
            self.type_set = ['balanced', 'risk_averse', 'slow_uncertain']
        if self.type_prior is None:
            self.type_prior = [1.0 / len(self.type_set)] * len(self.type_set)


@dataclass
class ExpConfig:
    """Experiment parameters."""
    seed: int = 42
    n_seeds: int = 5                # seeds per condition
    n_obs_queries: int = 4          # queries in observation phase
    n_teach_queries: int = 8        # queries in teach phase
    n_eval_queries: int = 8         # queries in eval phase
    n_workers: int = 16             # parallel workers
    # Query source mode: 'txt_only' | 'txt_resample' | 'generated' | 'hybrid'
    query_source_mode: str = 'generated'
    # Task data
    task_ids: list = field(default_factory=lambda: [
        f'{i:06d}' for i in range(1, 21)
    ])


@dataclass
class FullConfig:
    """Top-level config combining all subsystems."""
    env: EnvConfig = field(default_factory=EnvConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    tutor: TutorConfig = field(default_factory=TutorConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    # Path to BASIC/cls_learner/data
    cls_data_dir: str = ''

    def to_dict(self) -> dict:
        """Serialize to nested dict for YAML / JSON."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'FullConfig':
        """Load config from YAML file with defaults."""
        with open(path, 'r') as f:
            d = yaml.safe_load(f) or {}
        cfg = cls()
        if 'env' in d:
            for k, v in d['env'].items():
                if hasattr(cfg.env, k):
                    setattr(cfg.env, k, v)
        if 'learner' in d:
            for k, v in d['learner'].items():
                if hasattr(cfg.learner, k):
                    setattr(cfg.learner, k, v)
        if 'tutor' in d:
            for k, v in d['tutor'].items():
                if hasattr(cfg.tutor, k):
                    setattr(cfg.tutor, k, v)
        if 'belief' in d:
            for k, v in d['belief'].items():
                if hasattr(cfg.belief, k):
                    setattr(cfg.belief, k, v)
        if 'exp' in d:
            for k, v in d['exp'].items():
                if hasattr(cfg.exp, k):
                    setattr(cfg.exp, k, v)
        if 'cls_data_dir' in d:
            cfg.cls_data_dir = d['cls_data_dir']
        return cfg

    def resolve_data_dir(self) -> str:
        """Resolve cls_data_dir relative to project root."""
        if self.cls_data_dir:
            return self.cls_data_dir
        # Auto-detect from package location
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(
            pkg_dir, '..', '..', 'BASIC', 'cls_learner', 'data'))
