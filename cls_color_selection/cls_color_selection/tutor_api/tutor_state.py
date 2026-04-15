"""
tutor_state.py — Tutor belief state dataclasses.

Three-layer decomposition:
  B_sem:  grammar competence belief (Beta on success rate + beam entropy)
  B_risk: risk competence belief (Beta on detection, over-avoidance, calibration)
  B_type: learner type posterior (categorical over discrete types)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..config import BeliefConfig


@dataclass
class BetaPosterior:
    """Beta(α, β) posterior for Bernoulli-type beliefs."""
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def n_obs(self) -> float:
        return self.alpha + self.beta - 2.0  # subtract prior counts

    def update_success(self, n: float = 1.0):
        self.alpha += n

    def update_failure(self, n: float = 1.0):
        self.beta += n

    def clone(self) -> 'BetaPosterior':
        return BetaPosterior(alpha=self.alpha, beta=self.beta)


@dataclass
class BSem:
    """Grammar competence belief.

    Attributes:
        success_rate: Beta posterior on P(Y* = Y_gold)
        beam_entropy: running mean of beam posterior entropy
        confirm_risk: running mean of P(timeout given current grammar)
    """
    success_rate: BetaPosterior = field(default_factory=BetaPosterior)
    beam_entropy_mean: float = 0.0  # H(beam) averaged over observed queries
    beam_entropy_n: int = 0
    confirm_timeout_rate: float = 0.0

    @property
    def a_probe(self) -> float:
        """Estimated grammar accuracy."""
        return self.success_rate.mean

    @property
    def e_beam(self) -> float:
        """Estimated beam uncertainty."""
        return self.beam_entropy_mean

    def update_beam_entropy(self, h: float):
        """Running mean update for beam entropy."""
        self.beam_entropy_n += 1
        alpha = 1.0 / self.beam_entropy_n
        self.beam_entropy_mean = (1 - alpha) * self.beam_entropy_mean + alpha * h


@dataclass
class BRisk:
    """Risk competence belief.

    Attributes:
        p_detect: P(learner correctly avoids danger)
        p_overavoid: P(learner incorrectly avoids safe balls)
        p_calib: overall calibration quality
    """
    p_detect: BetaPosterior = field(default_factory=BetaPosterior)
    p_overavoid: BetaPosterior = field(default_factory=lambda: BetaPosterior(1.0, 3.0))
    # Risk event counters for calibration
    n_danger_encountered: int = 0
    n_danger_avoided: int = 0
    n_safe_skipped: int = 0
    n_safe_selected: int = 0

    @property
    def detect_rate(self) -> float:
        return self.p_detect.mean

    @property
    def overavoid_rate(self) -> float:
        return self.p_overavoid.mean


@dataclass
class BType:
    """Learner type posterior (categorical).

    Types: ['balanced', 'risk_averse', 'slow_uncertain']
    """
    type_names: List[str] = field(default_factory=lambda: [
        'balanced', 'risk_averse', 'slow_uncertain'
    ])
    log_posterior: np.ndarray = field(default=None)  # log P(T=t | H)

    def __post_init__(self):
        if self.log_posterior is None:
            n = len(self.type_names)
            self.log_posterior = np.full(n, -np.log(n))  # uniform

    @property
    def posterior(self) -> np.ndarray:
        """Normalized posterior P(T=t | H)."""
        lp = self.log_posterior - np.max(self.log_posterior)
        p = np.exp(lp)
        return p / p.sum()

    @property
    def map_type(self) -> str:
        """Maximum a posteriori type."""
        return self.type_names[int(np.argmax(self.log_posterior))]

    def update_log_likelihood(self, log_lik: np.ndarray):
        """Update with log-likelihood vector for each type."""
        self.log_posterior = self.log_posterior + log_lik
        # Normalize log space
        self.log_posterior -= np.max(self.log_posterior)


@dataclass
class TutorBelief:
    """Full tutor belief state.

    Combines B_sem, B_risk, B_type into one structure.
    Initialized from observation phase, updated during teaching.
    """
    sem: BSem = field(default_factory=BSem)
    risk: BRisk = field(default_factory=BRisk)
    type: BType = field(default_factory=BType)
    # Episode-level counters
    n_warnings_issued: int = 0
    n_hints_issued: int = 0
    n_courage_issued: int = 0
    n_queries_seen: int = 0

    @classmethod
    def from_config(cls, cfg: BeliefConfig) -> 'TutorBelief':
        """Initialize belief from config."""
        sem = BSem(
            success_rate=BetaPosterior(
                alpha=cfg.sem_beta_prior[0],
                beta=cfg.sem_beta_prior[1]),
        )
        risk = BRisk(
            p_detect=BetaPosterior(
                alpha=cfg.risk_beta_prior[0],
                beta=cfg.risk_beta_prior[1]),
            p_overavoid=BetaPosterior(
                alpha=cfg.over_beta_prior[0],
                beta=cfg.over_beta_prior[1]),
        )
        btype = BType(type_names=list(cfg.type_set))
        if cfg.type_prior:
            btype.log_posterior = np.log(
                np.array(cfg.type_prior) + 1e-30)

        return cls(sem=sem, risk=risk, type=btype)

    def summary_dict(self) -> Dict[str, float]:
        """Flat summary for logging."""
        return {
            'B_sem_a_probe': self.sem.a_probe,
            'B_sem_e_beam': self.sem.e_beam,
            'B_risk_detect': self.risk.detect_rate,
            'B_risk_overavoid': self.risk.overavoid_rate,
            'B_type_map': self.type.map_type,
            'B_type_posterior': list(self.type.posterior),
            'n_warnings': self.n_warnings_issued,
            'n_hints': self.n_hints_issued,
            'n_courage': self.n_courage_issued,
        }
