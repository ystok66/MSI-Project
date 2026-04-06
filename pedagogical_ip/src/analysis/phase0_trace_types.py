"""
Phase 0 Trace Types — Unified dataclasses for convergence audit.

All trace types are read-only diagnostics, used by Phase 0 audit scripts.
They do NOT influence canonical execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


# ═══════════════════════════════════════════════════════════
# Q1: Warning Path Attribution
# ═══════════════════════════════════════════════════════════

@dataclass
class WarningCounterfactual:
    """One counterfactual branch for a single warning event."""
    variant: str                          # "none" | "legacy_only" | "rsa_only" | "hybrid"
    delta_rho: float                      # ρ_wait - ρ_warn_variant
    delta_J: float                        # J_next^variant - J_next^none
    first_step_action: str                # planned first step after this variant
    flip: bool                            # first_step differs from 'none' variant


@dataclass
class WarningPathTrace:
    """Per-warning-event trace for path attribution audit."""
    t: int                                # timestep of warning
    seed: int
    family: str
    segment_index: int
    # Source quantities
    n_pseudolabel_updates: int            # how many cells got pseudo-label injection
    lane_bias_mass: float                 # aggregate lane bias value
    rsa_delta_rho: float                  # RSA belief-to-risk adapter delta
    # Counterfactual results
    counterfactuals: Dict[str, WarningCounterfactual] = field(default_factory=dict)


@dataclass
class WarningAuditSummary:
    """Aggregate summary across all warning events for one variant config."""
    variant: str
    n_events: int
    mean_delta_rho_legacy: float
    mean_delta_rho_rsa: float
    mean_delta_rho_hybrid: float
    flip_rate_legacy: float
    flip_rate_rsa: float
    flip_rate_hybrid: float
    # Attribution: fraction of flips explained by each mechanism
    rsa_explains_fraction: float          # fraction where |rsa_delta_rho| > half of total delta
    legacy_explains_fraction: float


# ═══════════════════════════════════════════════════════════
# Q2: Transfer Capacity
# ═══════════════════════════════════════════════════════════

@dataclass
class TransferEpisodeTrace:
    """Per-episode trace for transfer capacity audit."""
    block_id: str                         # "A" | "B" | "C" | "D"
    episode_index: int
    model_type: str                       # "linear_current" | "basis_shadow" etc.
    stateful: bool
    tbsr: float                           # 0.0 or 1.0
    survived: bool
    steps: int
    n_model_updates: int                  # how many updates the head received
    # Weight snapshot at episode end (for convergence visualization)
    cost_w_norm: float
    risk_w_norm: float


@dataclass
class TransferAuditSummary:
    """Summary for one model×stateful condition."""
    model_type: str
    stateful: bool
    block_tbsr: Dict[str, float]          # block_id → mean TBSR
    state_gain: Dict[str, float]          # block_id → StatGain
    block_a_tbsr: float


# ═══════════════════════════════════════════════════════════
# Q3: GTET Temptation
# ═══════════════════════════════════════════════════════════

@dataclass
class GTETPosteriorStepTrace:
    """Per-step posterior update trace for GTET factor audit."""
    t: int
    factor_mode: str
    # Posterior update mass (L1 change from previous step)
    delta_g_mass: float                   # |q_t(g) - q_{t-1}(g)|_1
    delta_theta_mass: float               # |q_t(θ) - q_{t-1}(θ)|_1
    delta_z_mass: float                   # |q_t(z) - q_{t-1}(z)|_1


@dataclass
class GTETEpisodeTrace:
    """Per-episode trace for GTET factor audit."""
    seed: int
    factor_mode: str
    survived: bool
    reached_goal: bool
    steps: int
    per_step: List[GTETPosteriorStepTrace] = field(default_factory=list)
    # Aggregates
    mean_delta_g: float = 0.0
    mean_delta_theta: float = 0.0
    mean_delta_z: float = 0.0
    # Task performance
    route_top1_correct: Optional[bool] = None
    lift_u: Optional[float] = None


@dataclass
class GTETAuditSummary:
    """Summary for one factor_mode condition."""
    factor_mode: str
    n_episodes: int
    survival_rate: float
    goal_rate: float
    mean_delta_g: float
    mean_delta_theta: float
    mean_delta_z: float
    mean_lift_u: float


# ═══════════════════════════════════════════════════════════
# Q4: Time-Learning Closure
# ═══════════════════════════════════════════════════════════

@dataclass
class TimeLearningStepTrace:
    """Per-step trace for time-learning closure audit."""
    t: int
    # Uncertainty quantities
    U_t: float                            # mean directional uncertainty of observed patch
    IG_t: float                           # U_{t-1} - U_t  (info gain)
    FC_t: float                           # 1 + max(0, c_t - 1)  (flow cost)
    is_stall: bool                        # IG_t <= 0
    # Tutor decision
    tutor_action: str                     # "WAIT" | "WARN" | "UNLOCK" | "ITEM_DROP"
    # FP_wait: tutor chose WAIT but next step had no info gain
    fp_wait: bool                         # a_t=WAIT ∧ IG_{t+1}<=0 ∧ FC_{t+1}>0


@dataclass
class TimeLearningEpisodeTrace:
    """Per-episode trace for time-learning closure."""
    seed: int
    family: str
    tutor_mode: str                       # "selective" | "always_warn" | "no_tutor"
    survived: bool
    reached_goal: bool
    steps: int
    per_step: List[TimeLearningStepTrace] = field(default_factory=list)
    # Aggregates
    stall_cost: float = 0.0               # Σ 1[IG≤0]·FC
    total_cost: float = 0.0               # Σ FC
    bore_ratio: float = 0.0               # stall_cost / total_cost
    fp_wait_rate: float = 0.0             # P(WAIT ∧ IG≤0 ∧ FC>0)


@dataclass
class TimeLearningAuditSummary:
    """Summary for one tutor_mode × family condition."""
    family: str
    tutor_mode: str
    n_episodes: int
    mean_bore_ratio: float
    mean_fp_wait_rate: float
    mean_stall_cost: float
