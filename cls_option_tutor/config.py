"""
config.py — All hyperparameters for the CLS Option Tutor V2.

V2 changes:
  - HP_0 = 5 (from 10)
  - max_refreshes = 2 (from 3)
  - Discrete risk classes: 6 safe + 4 risky
  - Budget-aware learner (alpha_ko, alpha_time)
  - RISK_HINT replaces BAN (c_ban archived, c_hint added)
  - Hazard head + severity head replaces single danger head
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Core environment parameters."""
    # Menu
    K: int = 10                        # candidates per menu
    T_max: int = 5                     # max rounds per query
    H_0: int = 5                       # initial HP per query (V2: was 10)

    # Block structure
    N_sup: int = 14                    # support examples per block
    M_queries: int = 8                 # queries per block
    N_obs: int = 2                     # Phase 2: observation (tutor watches frozen learner)
    N_teach: int = 3                   # Phase 3: teaching (tutor intervenes, learner learns)
    N_eval: int = 3                    # Phase 4: evaluation (frozen learner, no tutor)

    # Risk model (V2 discrete)
    n_safe: int = 6                    # safe options per menu
    n_risky: int = 4                   # risky options per menu
    risk_classes: tuple = (1, 2, 3, 4) # possible risk values for risky options
    danger_dim: int = 16               # dimension of danger vector v
    cluster_sigma: float = 0.5         # intra-cluster noise for danger vectors

    # Refresh (V2: risk-only)
    max_refreshes: int = 2             # max refreshes per query (V2: was 3)


@dataclass
class LearnerConfig:
    """Learner policy parameters."""
    # Semantic weights
    alpha_sem: float = 1.0             # semantic score weight
    alpha_risk: float = 0.5            # danger prediction weight
    alpha_unc: float = 0.2             # danger uncertainty weight
    alpha_nov: float = 0.0             # novelty bonus (off in v2)
    alpha_ko: float = 1.0             # KO risk weight (V2 new)
    alpha_time: float = 0.3            # time pressure penalty (V2 new)
    beta_L: float = 4.0               # softmax temperature
    epsilon: float = 0.05              # lapse rate
    c_refresh: float = 0.3            # refresh cost
    tau_sem: float = 1.0               # semantic mismatch temperature

    # CLS semantic subsystem
    use_cls: bool = False              # False = deterministic baseline
    n_sup: int = 5                     # support examples to show (1/3/5)
    n_em: int = 2                      # EM iterations for CLS
    use_hpc: bool = True               # HPC memory for CLS
    rho_H: float = 2.0                  # highlight attention boost (mirrored from tutor)

    # Hazard head (V2: binary safe/risky classifier)
    hazard_lr: float = 0.1             # learning rate for hazard head
    hazard_prior_var: float = 1.0      # prior variance
    eta_hint: float = 0.8             # RISK_HINT weak label confidence

    # Severity head (V2: damage regression given risky)
    severity_prior_var: float = 1.0
    severity_lr: float = 0.1


@dataclass
class TutorConfig:
    """Tutor policy parameters."""
    # Scoring weights
    beta_corr: float = 2.0             # correctness improvement weight
    beta_hp: float = 1.0               # HP safety weight
    beta_learn: float = 0.5            # learning gain weight
    beta_time: float = 0.3             # time pressure weight
    beta_safe: float = 1.5             # safety gain weight
    beta_IG: float = 1.0               # HIGHLIGHT info gain weight
    beta_over: float = 0.2             # HIGHLIGHT over-reveal penalty

    # Intervention fixed costs
    c_ban: float = 0.0                 # BAN cost (equal to HIGHLIGHT)
    c_hint: float = 0.3                # RISK_HINT cost (V2 new)
    c_hl: float = 0.0                  # V2: free (cost via beta_over only)
    c_skip: float = 1.5

    # SKIP mastery parameters
    beta_mastery: float = 0.8
    beta_certainty: float = 0.4

    # HIGHLIGHT
    max_highlight_cells: int = 2
    rho_H: float = 2.0                 # highlight attention boost

    # Profile inference
    profile_grid_size: int = 5         # grid points per profile dim


@dataclass
class TrustState:
    """Future trust hook — reserved but disabled in V2."""
    enabled: bool = False
    tau: float = 0.3
    compliance_count: int = 0
    defiance_count: int = 0


@dataclass
class FullConfig:
    """Top-level config combining all subsystems."""
    env: EnvConfig = field(default_factory=EnvConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    tutor: TutorConfig = field(default_factory=TutorConfig)

    # CLS backend
    cls_mode: str = "ast"
    cls_use_hpc: bool = True
    cls_gauss: bool = False
    cls_data_dir: str = ""

    # Reproducibility
    seed: int = 42
