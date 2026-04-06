"""Persistent Learner Profile — Bootstrap & Finalize.

Supports:
  - Pure carry-over (η=1.0, use_calibration=False)
  - Confidence-gated shrinkage (use_calibration=True, Task 3A)
"""

from __future__ import annotations
from typing import Optional, Callable, Dict
import numpy as np

from .profile_state import ProfileState, SessionSummary
from ..agents.internalization_state_v3 import FactoredInternalizationState


def bootstrap_observer(observer, profile: ProfileState,
                       eta: float = 1.0,
                       use_calibration: bool = False,
                       calib_fn: Optional[Callable] = None,
                       lambda_c: float = 0.3):
    """Initialize observer from previous session terminal estimate.

    Sets observer hat-state to profile's m̂_T (with optional blending via η).
    Does NOT modify frozen per-step update logic.

    Args:
        observer: A1MtObserver or subclass. Only sets hat-state fields
                  (tau_hat, nu_hat, etc.), which are not frozen-guarded.
        profile: Previous session's ProfileState.
        eta: Carry-over weight. 1.0 = full carry-over (default, no forgetting).
        use_calibration: If True, apply confidence-gated shrinkage.
            m̂_0^(s+1) = (1-ρ_c)·m̂_T^(s) + ρ_c·prior
            where ρ_c = λ_c · (1 - c̄_T).
        calib_fn: Optional custom calibration function(observer, profile).
            If provided and use_calibration=True, called AFTER shrinkage.
        lambda_c: Shrinkage strength. Higher = more aggressive shrinkage
            when confidence is low. Default 0.3.
    """
    m_hat = profile.m_hat_terminal
    conf = profile.confidence
    n_prior = profile.session_idx  # how many sessions completed before this one

    # Observer prior defaults (same as A1MtObserver.__init__)
    prior = {"tau": 0.3, "nu": 0.1, "gamma_gen": 0.0,
             "gamma_spec": 0.0, "kappa": 0.3}
    prior_conf = {"tau": 0.2, "nu": 0.2, "gamma_gen": 0.2}

    # Per-dimension confidence-gated shrinkage
    # ρ_{c,d} = λ_c · (1 - c_d) · session_scale
    # session_scale grows logarithmically: log(1 + n_sessions) / log(4)
    # so session 0→1: scale ~0.5, session 3→4: scale ~1.0
    if use_calibration:
        session_scale = min(np.log1p(max(n_prior, 0)) / np.log(4), 2.0)
    else:
        session_scale = 0.0

    def _dim_weight(dim_name):
        if not use_calibration:
            return eta, 1.0 - eta
        c_d = min(conf.get(dim_name, 0.2), 1.0)
        rho_d = lambda_c * (1.0 - c_d) * session_scale
        rho_d = min(rho_d, 0.8)  # cap: never shrink more than 80%
        w_c = (1.0 - rho_d) * eta
        return w_c, 1.0 - w_c

    w_tau, wp_tau = _dim_weight("tau")
    w_nu, wp_nu = _dim_weight("nu")
    w_gg, wp_gg = _dim_weight("gamma_gen")
    # γ_spec and κ have no confidence tracking, use scalar shrinkage
    w_other = (1.0 - (lambda_c * 0.5 * session_scale if use_calibration else 0.0)) * eta
    w_other = max(w_other, 0.2)
    wp_other = 1.0 - w_other

    observer.tau_hat = w_tau * m_hat.get("tau", prior["tau"]) + wp_tau * prior["tau"]
    observer.nu_hat = w_nu * m_hat.get("nu", prior["nu"]) + wp_nu * prior["nu"]
    observer.gamma_gen_hat = w_gg * m_hat.get("gamma_gen", prior["gamma_gen"]) + wp_gg * prior["gamma_gen"]
    observer.gamma_spec_hat = w_other * m_hat.get("gamma_spec", prior["gamma_spec"]) + wp_other * prior["gamma_spec"]
    observer.kappa_hat = w_other * m_hat.get("kappa", prior["kappa"]) + wp_other * prior["kappa"]

    # Confidence carry-over (also shrunk per-dimension)
    observer.conf_tau = w_tau * conf.get("tau", prior_conf["tau"]) + wp_tau * prior_conf["tau"]
    observer.conf_nu = w_nu * conf.get("nu", prior_conf["nu"]) + wp_nu * prior_conf["nu"]
    observer.conf_gamma = w_gg * conf.get("gamma_gen", prior_conf["gamma_gen"]) + wp_gg * prior_conf["gamma_gen"]

    # Optional custom calibration (applied after shrinkage)
    if use_calibration and calib_fn is not None:
        calib_fn(observer, profile)

    # Reset step counter, event counters, and history (new session)
    observer._step_counter = 0
    observer._recent_events_tau = 0
    observer._recent_events_nu = 0
    observer._recent_events_gamma = 0
    observer.history = []


def bootstrap_agent_state(state: FactoredInternalizationState,
                          profile: ProfileState,
                          eta: float = 1.0):
    """Initialize agent true state from previous session terminal.

    For simulation: the learner's actual internal state carries over.
    η=1.0 means full carry-over (learner remembers everything).
    """
    m_T = profile.m_terminal
    prior = {"kappa": 1.0, "tau": 0.3, "nu": 0.1,
             "gamma_spec": 0.0, "gamma_gen": 0.0}

    state.kappa = eta * m_T.get("kappa", prior["kappa"]) + (1 - eta) * prior["kappa"]
    state.tau = eta * m_T.get("tau", prior["tau"]) + (1 - eta) * prior["tau"]
    state.nu = eta * m_T.get("nu", prior["nu"]) + (1 - eta) * prior["nu"]
    state.gamma_spec = eta * m_T.get("gamma_spec", prior["gamma_spec"]) + (1 - eta) * prior["gamma_spec"]
    state.gamma_gen = eta * m_T.get("gamma_gen", prior["gamma_gen"]) + (1 - eta) * prior["gamma_gen"]

    # Clear history for new session
    state.kappa_history = []
    state.tau_history = []
    state.nu_history = []
    state.gs_history = []
    state.gg_history = []
    state.snapshot()


def finalize_session(observer, m_true: FactoredInternalizationState,
                     session_idx: int, theta: str,
                     learner_id: str = "default",
                     session_summary: Optional[SessionSummary] = None,
                     ) -> ProfileState:
    """Export current observer + true state as a ProfileState.

    Call at end of each session to create the persistent profile entry.
    """
    est = observer.get_estimate()
    conf = observer.get_confidence()

    # Compute calibration error
    m_true_dict = m_true.as_dict
    cal_err = {}
    for dim in ["tau", "nu", "gamma_gen"]:
        est_key = dim
        true_key = dim
        cal_err[dim] = abs(est.get(est_key, 0) - m_true_dict.get(true_key, 0))
    # gamma_spec: observer uses gamma_spec, true state uses gamma_spec
    cal_err["gamma_spec"] = abs(est.get("gamma_spec", 0) - m_true_dict.get("gamma_spec", 0))
    # kappa: observer uses kappa, true state uses kappa
    cal_err["kappa"] = abs(est.get("kappa", 0) - m_true_dict.get("kappa", 0))

    if session_summary is None:
        session_summary = SessionSummary()
    session_summary.calibration_error = cal_err
    session_summary.n_steps = observer._step_counter

    return ProfileState(
        learner_id=learner_id,
        session_idx=session_idx,
        theta=theta,
        m_terminal=dict(m_true_dict),
        m_hat_terminal=dict(est),
        confidence=dict(conf),
        history=session_summary,
    )


def make_need_hook(z_bar: dict, z_star: dict = None,
                   lambda_need: float = 0.3):
    """Create a profile-aware need bonus hook for CurriculumControllerV13.

    Δ_need(ℓ|P_s) = Σ_p w_{ℓ,p} · [z*_p - z̄_{s,p}]+

    where w_{ℓ,p} = lesson.gain[p] (main gain mapping).

    Args:
        z_bar: Current EMA-smoothed probe weakness (from ProfileManager).
        z_star: Target probe levels. Default: {"RC":0.7, "TR":0.65, "EP":0.55, "VA":0.7, "IA":0.7}.
        lambda_need: Weight for need bonus. 0.3 = moderate.

    Returns:
        Hook function: (lesson, base_J, mastery) -> float (additive).
    """
    from ..curriculum.lesson_library_v2 import PROBE_NAMES

    if z_star is None:
        z_star = {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.70}

    def hook(lesson, base_J, mastery):
        delta_need = 0.0
        for i, p in enumerate(PROBE_NAMES):
            target = z_star.get(p, 0.5)
            current = z_bar.get(p, 0.5)
            deficit = max(target - current, 0.0)
            # w_{ℓ,p} from lesson gain vector
            w_lp = float(lesson.gain[i]) if i < len(lesson.gain) else 0.0
            delta_need += w_lp * deficit
        return lambda_need * delta_need

    return hook

