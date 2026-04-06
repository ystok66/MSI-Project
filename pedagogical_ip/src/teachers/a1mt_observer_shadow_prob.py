"""Shadow Probabilistic Observer for 5D internalization state.

Phase 0: NullShadowObserver — copies frozen output, validates bridge.
Phase 1: ProbShadowObserver — factorized 1D grid posteriors.

Hard constraints:
  - Does NOT import or modify internalization_observer.py internals
  - Does NOT modify internalization_state_v3.py
  - Does NOT modify stochastic_agent_policy.py
  - All dims accessed by NAME (tau, nu, gamma_gen, gamma_spec, kappa)
  - kappa is transition-only in Phase 1 (no emission)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
from scipy import stats as sp_stats

from .a1mt_observer_shadow_types import (
    DIM_NAMES, DIM_BOUNDS, DIM_PRIORS,
    ShadowDimConfig, DimPosterior, ShadowSnapshot, ShadowDiagnostics,
)
from .internalization_observer import ObsEvent


# ═══════════════════════════════════════════════════════════════
#  Phase 0: Null Shadow Observer (deterministic copy of frozen)
# ═══════════════════════════════════════════════════════════════

class NullShadowObserver:
    """Shadow that copies frozen observer output exactly.

    Purpose: validate bridge, logging, and report pipeline before
    introducing probabilistic updates.
    """

    def __init__(self, n_grid: int = 32):
        self.n_grid = n_grid
        self._state: Dict[str, float] = dict(DIM_PRIORS)
        self.history: List[ShadowSnapshot] = []

    def reset(self):
        self._state = dict(DIM_PRIORS)
        self.history = []

    def step(self, ev: ObsEvent, frozen_estimate: dict) -> ShadowSnapshot:
        """Copy frozen estimate into a grid posterior (delta at frozen value)."""
        posteriors = {}
        for dim in DIM_NAMES:
            lo, hi = DIM_BOUNDS[dim]
            grid = np.linspace(lo, hi, self.n_grid)
            val = frozen_estimate.get(dim, DIM_PRIORS[dim])
            # Delta-like: put all weight on nearest grid point
            dists = np.abs(grid - val)
            weights = np.zeros(self.n_grid)
            weights[np.argmin(dists)] = 1.0
            posteriors[dim] = DimPosterior(name=dim, grid=grid, weights=weights)
            self._state[dim] = val

        snap = ShadowSnapshot(posteriors=posteriors, event_loglik=0.0)
        self.history.append(snap)
        return snap

    def get_estimate(self) -> dict:
        return {d: round(v, 6) for d, v in self._state.items()}


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Probabilistic Shadow Observer (factorized grid)
# ═══════════════════════════════════════════════════════════════

# ── Frozen update equations (replicated as transition means) ──

def _frozen_tau_mean(tau_prev: float, ev: ObsEvent,
                     alpha_plus: float = 0.22,
                     alpha_minus: float = 0.10,
                     lambda_tau: float = 0.005,
                     tau_0: float = 0.3) -> float:
    """Replicate A1MtObserver tau update as predicted mean."""
    e_trust_plus = float(ev.warned and ev.follow_warn and ev.warn_correct)
    e_trust_minus = float(ev.warned and ev.follow_warn and ev.warn_wrong)
    tau_new = tau_prev
    tau_new += alpha_plus * e_trust_plus * (1.0 - tau_prev)
    tau_new -= alpha_minus * e_trust_minus * tau_prev
    # Conditional reversion (simplified: always apply weak reversion)
    tau_new += lambda_tau * (tau_0 - tau_new)
    return float(np.clip(tau_new, 0.0, 1.0))


def _frozen_nu_mean(nu_prev: float, ev: ObsEvent,
                    alpha_plus: float = 0.18,
                    alpha_minus: float = 0.13,
                    lambda_nu: float = 0.005,
                    nu_0: float = 0.1,
                    nu_max: float = 0.8) -> float:
    """Replicate A1MtObserver nu update as predicted mean."""
    e_blind = float(ev.warned and ev.follow_warn) * (1.0 - ev.p_self)
    e_selfdisc = float(ev.self_discovery) * ev.p_self
    nu_new = nu_prev
    nu_new += alpha_plus * e_blind * (nu_max - nu_prev)
    nu_new -= alpha_minus * e_selfdisc * nu_prev
    nu_new += lambda_nu * (nu_0 - nu_new)
    return float(np.clip(nu_new, 0.0, nu_max))


def _frozen_gamma_gen_mean(gg_prev: float, ev: ObsEvent,
                           pressure_ema: float,
                           alpha_plus: float = 0.07,
                           alpha_minus: float = 0.10,
                           lambda_gg: float = 0.005,
                           gg_0: float = 0.0,
                           gg_max: float = 0.5) -> float:
    """Replicate A1MtObserver gamma_gen update."""
    e_explore_plus = float(ev.beneficial_novelty or ev.self_discovery)
    gg_new = gg_prev
    gg_new += alpha_plus * pressure_ema * (gg_max - gg_prev)
    gg_new -= alpha_minus * e_explore_plus * gg_prev
    gg_new += lambda_gg * (gg_0 - gg_new)
    return float(np.clip(gg_new, 0.0, gg_max))


def _frozen_gamma_spec_mean(gs_prev: float, ev: ObsEvent,
                            alpha_resist: float = 0.03,
                            alpha_follow: float = 0.025,
                            lure_threshold: float = 0.3,
                            gs_max: float = 1.0) -> float:
    """Replicate A1MtObserver gamma_spec update."""
    if ev.lure < lure_threshold:
        return gs_prev  # no update
    correct = (ev.agent_choice == ev.oracle_safe)
    if correct:
        e_resist = ev.lure * (1.0 - gs_prev)
        return float(np.clip(gs_prev + alpha_resist * e_resist, 0.0, gs_max))
    else:
        e_follow = ev.lure * gs_prev
        return float(np.clip(gs_prev - alpha_follow * e_follow, 0.0, gs_max))


def _frozen_kappa_mean(kappa_prev: float, ev: ObsEvent,
                       lambda_kappa: float = 0.02,
                       kappa_0: float = 0.3,
                       alpha_pos: float = 0.015,
                       alpha_neg: float = 0.012,
                       risk_gate: float = 0.1,
                       kappa_min: float = 0.0,
                       kappa_max: float = 1.0) -> float:
    """Replicate A1MtObserver kappa update (transition-only)."""
    if ev.risk < risk_gate or ev.risk_hat is None:
        # Mean-reversion only
        kap = (1 - lambda_kappa) * kappa_prev + lambda_kappa * kappa_0
        return float(np.clip(kap, kappa_min, kappa_max))
    delta_risk = ev.risk - ev.risk_hat
    kap = (1 - lambda_kappa) * kappa_prev + lambda_kappa * kappa_0
    if delta_risk > 0:
        kap += alpha_pos * delta_risk * (kappa_max - kappa_prev)
    else:
        kap += alpha_neg * delta_risk * (kappa_prev - kappa_min)
    return float(np.clip(kap, kappa_min, kappa_max))


# ── Emission likelihoods ──────────────────────────────────────

def _tau_emission(grid: np.ndarray, ev: ObsEvent) -> np.ndarray:
    """Trust emission: P(z | tau).

    z = trust+  if warned & follow & correct  -> P = tau
    z = trust-  if warned & follow & wrong     -> P = 1 - tau
    z = none    otherwise                      -> P = 1 (uninformative)
    """
    if ev.warned and ev.follow_warn and ev.warn_correct:
        return np.clip(grid, 1e-8, 1.0 - 1e-8)
    elif ev.warned and ev.follow_warn and ev.warn_wrong:
        return np.clip(1.0 - grid, 1e-8, 1.0 - 1e-8)
    return np.ones_like(grid)


def _nu_emission(grid: np.ndarray, ev: ObsEvent) -> np.ndarray:
    """Dependence emission: P(z | nu).

    z = blind    if warned & follow & low p_self -> P = nu
    z = selfdisc if self_discovery & high p_self -> P = 1 - nu
    z = none     otherwise                       -> P = 1
    """
    e_blind = float(ev.warned and ev.follow_warn) * (1.0 - ev.p_self)
    e_selfdisc = float(ev.self_discovery) * ev.p_self

    if e_blind > 0.05:
        # Soft likelihood: weight by e_blind strength
        return np.clip(grid, 1e-8, 1.0) ** e_blind
    elif e_selfdisc > 0.05:
        return np.clip(1.0 - grid, 1e-8, 1.0) ** e_selfdisc
    return np.ones_like(grid)


def _gamma_gen_emission(grid: np.ndarray, ev: ObsEvent) -> np.ndarray:
    """General suppression emission: P(explore+ | gamma_gen) = 1 - gamma_gen.

    Only fires when exploration actually occurred.
    """
    e_explore = float(ev.beneficial_novelty or ev.self_discovery)
    if e_explore > 0:
        # Exploration happened -> likelihood = 1 - gamma_gen
        return np.clip(1.0 - grid, 1e-8, 1.0)
    return np.ones_like(grid)


def _gamma_spec_emission(grid: np.ndarray, ev: ObsEvent) -> np.ndarray:
    """Temptation resistance emission.

    Only fires when lure >= 0.3:
      resist      -> P = gamma_spec
      follow_lure -> P = 1 - gamma_spec
    """
    if ev.lure < 0.3:
        return np.ones_like(grid)
    correct = (ev.agent_choice == ev.oracle_safe)
    if correct:
        return np.clip(grid, 1e-8, 1.0)
    else:
        return np.clip(1.0 - grid, 1e-8, 1.0)


# ── Core observer ─────────────────────────────────────────────

class ProbShadowObserver:
    """Factorized assumed-density filter over 5D internalization state.

    q_t(m_t) = prod_d q_t^d(m_t^d)

    Bounded dims: 1D grid, Beta transition kernel.
    Kappa: 1D grid, truncated Gaussian transition (no emission).

    New hyperparameters (only 3):
      c_bnd:       shared Beta concentration for bounded transitions
      sigma_kappa: Gaussian process noise for kappa transition
      n_grid:      grid points per dimension
    """

    def __init__(self, c_bnd: float = 20.0,
                 sigma_kappa: float = 0.1,
                 n_grid: int = 32,
                 use_kappa_emission: bool = False,
                 use_action_likelihood: bool = False):
        self.c_bnd = c_bnd
        self.sigma_kappa = sigma_kappa
        self.n_grid = n_grid
        self.use_kappa_emission = use_kappa_emission
        self.use_action_likelihood = use_action_likelihood

        # Per-dim grids and weights (initialized in reset)
        self._grids: Dict[str, np.ndarray] = {}
        self._weights: Dict[str, np.ndarray] = {}

        # Pressure EMA (for gamma_gen transition)
        self._pressure_ema: float = 0.0
        self._pressure_alpha: float = 0.3

        # History
        self.history: List[ShadowSnapshot] = []
        self.reset()

    def reset(self):
        """Initialize all dimensions to prior-centered distribution.

        Uses Beta(c_init * mu, c_init * (1-mu)) for bounded dims,
        and Gaussian for kappa, centered on DIM_PRIORS.
        """
        self._grids = {}
        self._weights = {}
        c_init = 10.0  # moderate prior concentration (not too tight, not uniform)

        for dim in DIM_NAMES:
            lo, hi = DIM_BOUNDS[dim]
            grid = np.linspace(lo + 1e-6, hi - 1e-6, self.n_grid)
            self._grids[dim] = grid

            prior_mean = DIM_PRIORS[dim]
            rng = hi - lo

            if dim == "kappa":
                # Gaussian prior centered on kappa_0
                log_w = -0.5 * ((grid - prior_mean) / self.sigma_kappa) ** 2
                w = np.exp(log_w - np.max(log_w))
            else:
                # Beta prior centered on prior_mean
                mu_tilde = np.clip((prior_mean - lo) / rng, 0.05, 0.95)
                a = c_init * mu_tilde
                b = c_init * (1.0 - mu_tilde)
                grid_norm = np.clip((grid - lo) / rng, 1e-6, 1.0 - 1e-6)
                log_w = sp_stats.beta.logpdf(grid_norm, a, b)
                w = np.exp(log_w - np.max(log_w))

            w /= w.sum()
            self._weights[dim] = w

        self._pressure_ema = 0.0
        self.history = []

    def _frozen_update_per_point(self, dim: str, x_val: float,
                                    ev: ObsEvent) -> float:
        """Compute frozen-observer predicted value for a specific x value."""
        if dim == "tau":
            return _frozen_tau_mean(x_val, ev)
        elif dim == "nu":
            return _frozen_nu_mean(x_val, ev)
        elif dim == "gamma_gen":
            return _frozen_gamma_gen_mean(x_val, ev, self._pressure_ema)
        elif dim == "gamma_spec":
            return _frozen_gamma_spec_mean(x_val, ev)
        elif dim == "kappa":
            return _frozen_kappa_mean(x_val, ev)
        return x_val

    def _predict_bounded(self, dim: str, ev: ObsEvent):
        """Predict step using proper transition matrix convolution.

        For each source point x_j:
          mu_j = frozen_update(x_j, ev)  (deterministic)
          T(x_i | x_j) = Beta(c_bnd * mu_j_tilde, c_bnd * (1-mu_j_tilde))
        Then:
          w_pred[i] = sum_j T(x_i | x_j) * w_prev[j]
        """
        lo, hi = DIM_BOUNDS[dim]
        rng = hi - lo
        grid = self._grids[dim]
        w_prev = self._weights[dim]
        n = self.n_grid

        grid_norm = np.clip((grid - lo) / rng, 1e-6, 1.0 - 1e-6)

        # Build transition matrix T[i, j] = P(x_i | x_j)
        w_pred = np.zeros(n)
        for j in range(n):
            if w_prev[j] < 1e-15:
                continue
            # Predicted mean from source point x_j
            mu_j = self._frozen_update_per_point(dim, grid[j], ev)
            mu_j_tilde = np.clip((mu_j - lo) / rng, 1e-4, 1.0 - 1e-4)
            a = self.c_bnd * mu_j_tilde
            b = self.c_bnd * (1.0 - mu_j_tilde)
            # Transition kernel from x_j
            log_k = sp_stats.beta.logpdf(grid_norm, a, b)
            k = np.exp(log_k - np.max(log_k))
            k /= (k.sum() + 1e-15)
            w_pred += w_prev[j] * k

        w_sum = w_pred.sum()
        if w_sum > 1e-15:
            w_pred /= w_sum
        else:
            w_pred = np.ones(n) / n
        self._weights[dim] = w_pred

    def _predict_kappa(self, ev: ObsEvent):
        """Predict step for kappa using transition matrix convolution."""
        grid = self._grids["kappa"]
        w_prev = self._weights["kappa"]
        n = self.n_grid

        w_pred = np.zeros(n)
        for j in range(n):
            if w_prev[j] < 1e-15:
                continue
            mu_j = self._frozen_update_per_point("kappa", grid[j], ev)
            log_k = -0.5 * ((grid - mu_j) / self.sigma_kappa) ** 2
            k = np.exp(log_k - np.max(log_k))
            k /= (k.sum() + 1e-15)
            w_pred += w_prev[j] * k

        w_sum = w_pred.sum()
        if w_sum > 1e-15:
            w_pred /= w_sum
        else:
            w_pred = np.ones(n) / n
        self._weights["kappa"] = w_pred

    def _emission(self, dim: str, ev: ObsEvent) -> np.ndarray:
        """Get emission likelihood for a dimension."""
        grid = self._grids[dim]
        if dim == "tau":
            return _tau_emission(grid, ev)
        elif dim == "nu":
            return _nu_emission(grid, ev)
        elif dim == "gamma_gen":
            return _gamma_gen_emission(grid, ev)
        elif dim == "gamma_spec":
            return _gamma_spec_emission(grid, ev)
        elif dim == "kappa":
            # Phase 1: transition-only, no emission
            if self.use_kappa_emission:
                pass  # TODO: Phase 2 ablation
            return np.ones_like(grid)
        return np.ones_like(grid)

    def _update_dim(self, dim: str, ev: ObsEvent) -> float:
        """Bayesian update for one dimension. Returns event log-likelihood."""
        lik = self._emission(dim, ev)
        w = self._weights[dim]
        # Marginal likelihood for this dim
        marginal = float(np.dot(w, lik))
        if marginal > 1e-15:
            self._weights[dim] = (w * lik) / marginal
            return float(np.log(marginal))
        else:
            # All likelihoods near zero — keep prior
            return float(np.log(1e-15))

    def step(self, ev: ObsEvent) -> ShadowSnapshot:
        """Full predict-update cycle for all 5 dimensions."""
        # Update pressure EMA (used by gamma_gen transition)
        self._pressure_ema = ((1 - self._pressure_alpha) * self._pressure_ema
                              + self._pressure_alpha * ev.dose)

        total_loglik = 0.0

        for dim in DIM_NAMES:
            # Predict (proper transition matrix convolution)
            if dim == "kappa":
                self._predict_kappa(ev)
            else:
                self._predict_bounded(dim, ev)

            # Update (emission likelihood reweighting)
            ll = self._update_dim(dim, ev)
            total_loglik += ll

        # Build snapshot
        posteriors = {}
        for dim in DIM_NAMES:
            posteriors[dim] = DimPosterior(
                name=dim,
                grid=self._grids[dim].copy(),
                weights=self._weights[dim].copy(),
            )

        snap = ShadowSnapshot(
            posteriors=posteriors,
            event_loglik=total_loglik,
            events_used={
                "trust+": float(ev.warned and ev.follow_warn and ev.warn_correct),
                "trust-": float(ev.warned and ev.follow_warn and ev.warn_wrong),
                "blind": float(ev.warned and ev.follow_warn) * (1.0 - ev.p_self),
                "selfdisc": float(ev.self_discovery) * ev.p_self,
                "pressure": self._pressure_ema,
                "explore+": float(ev.beneficial_novelty or ev.self_discovery),
                "lure": ev.lure,
            },
        )
        self.history.append(snap)
        return snap

    def get_estimate(self) -> dict:
        """Current posterior means as dict."""
        return {dim: round(float(np.dot(self._grids[dim], self._weights[dim])), 6)
                for dim in DIM_NAMES}

    def compute_diagnostics(
        self,
        m_true_history: List[dict],
        frozen_history: List[dict],
    ) -> ShadowDiagnostics:
        """Compute RMSE, MAE, 90% coverage, event-NLL vs true and frozen."""
        n = min(len(self.history), len(m_true_history), len(frozen_history))
        if n == 0:
            return ShadowDiagnostics()

        diag = ShadowDiagnostics(n_steps=n)
        total_nll = 0.0

        for dim in DIM_NAMES:
            errs_shadow = []
            errs_frozen = []
            covers = []
            for t in range(n):
                true_val = m_true_history[t].get(dim, DIM_PRIORS[dim])
                shadow_mean = self.history[t].mean(dim)
                frozen_val = frozen_history[t].get(dim, DIM_PRIORS[dim])

                errs_shadow.append((shadow_mean - true_val) ** 2)
                errs_frozen.append((frozen_val - true_val) ** 2)
                covers.append(self.history[t].posteriors[dim].covers(true_val, 0.90))

            diag.rmse[dim] = float(np.sqrt(np.mean(errs_shadow)))
            diag.mae[dim] = float(np.mean(np.sqrt(errs_shadow)))  # MAE via abs
            diag.rmse_frozen[dim] = float(np.sqrt(np.mean(errs_frozen)))
            diag.mae_frozen[dim] = float(np.mean(np.sqrt(errs_frozen)))
            diag.coverage_90[dim] = float(np.mean(covers))

        # Recompute MAE correctly (absolute errors)
        for dim in DIM_NAMES:
            abs_errs = []
            abs_errs_f = []
            for t in range(n):
                true_val = m_true_history[t].get(dim, DIM_PRIORS[dim])
                shadow_mean = self.history[t].mean(dim)
                frozen_val = frozen_history[t].get(dim, DIM_PRIORS[dim])
                abs_errs.append(abs(shadow_mean - true_val))
                abs_errs_f.append(abs(frozen_val - true_val))
            diag.mae[dim] = float(np.mean(abs_errs))
            diag.mae_frozen[dim] = float(np.mean(abs_errs_f))

        # Event NLL
        total_nll = sum(snap.event_loglik for snap in self.history[:n])
        diag.total_event_nll = -total_nll  # report as positive NLL
        diag.mean_event_nll = -total_nll / n

        return diag
