"""Shadow Observer Bridge — runs frozen + shadow in parallel.

Does NOT modify any frozen module. Purely diagnostic adapter.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np

from .internalization_observer import A1MtObserverFrozen, ObsEvent, ObserverSnapshot
from .a1mt_observer_shadow_prob import NullShadowObserver, ProbShadowObserver
from .a1mt_observer_shadow_types import (
    DIM_NAMES, DIM_PRIORS, ShadowSnapshot, ShadowDiagnostics,
)


class ShadowObserverBridge:
    """Runs frozen A1MtObserverFrozen + shadow observer in parallel.

    Consumes ObsEvent stream, feeds both observers, logs everything.
    Never affects any decision.
    """

    def __init__(self, shadow_mode: str = "prob",
                 n_grid: int = 32,
                 c_bnd: float = 20.0,
                 sigma_kappa: float = 0.1,
                 use_kappa_emission: bool = False,
                 use_action_likelihood: bool = False):
        """
        Args:
            shadow_mode: "null" for NullShadowObserver, "prob" for ProbShadowObserver
            n_grid: grid points per dimension (32 for dev, 64 for final)
        """
        self.frozen = A1MtObserverFrozen()

        if shadow_mode == "null":
            self.shadow = NullShadowObserver(n_grid=n_grid)
        else:
            self.shadow = ProbShadowObserver(
                c_bnd=c_bnd,
                sigma_kappa=sigma_kappa,
                n_grid=n_grid,
                use_kappa_emission=use_kappa_emission,
                use_action_likelihood=use_action_likelihood,
            )

        self.shadow_mode = shadow_mode
        self._frozen_history: List[dict] = []
        self._shadow_history: List[ShadowSnapshot] = []
        self._m_true_history: List[dict] = []
        self._step = 0

    def reset(self):
        self.frozen.reset()
        self.shadow.reset()
        self._frozen_history = []
        self._shadow_history = []
        self._m_true_history = []
        self._step = 0

    def step(self, ev: ObsEvent) -> dict:
        """Process one event through both observers.

        Returns dict with frozen estimate, shadow estimate, and snapshot.
        """
        # Frozen observer update
        frozen_snap = self.frozen.update(ev)
        frozen_est = self.frozen.get_estimate()
        self._frozen_history.append(dict(frozen_est))

        # Shadow observer update
        if self.shadow_mode == "null":
            shadow_snap = self.shadow.step(ev, frozen_est)
        else:
            shadow_snap = self.shadow.step(ev)
        self._shadow_history.append(shadow_snap)

        # Record true state if available
        if ev.m_true is not None:
            self._m_true_history.append(dict(ev.m_true))
        else:
            self._m_true_history.append(dict(DIM_PRIORS))

        self._step += 1

        return {
            "step": self._step,
            "frozen": frozen_est,
            "shadow": shadow_snap.as_dict(),
            "shadow_entropy": {d: shadow_snap.entropy(d) for d in DIM_NAMES},
            "event_loglik": shadow_snap.event_loglik,
        }

    def get_frozen_history(self) -> List[dict]:
        return list(self._frozen_history)

    def get_shadow_history(self) -> List[ShadowSnapshot]:
        return list(self._shadow_history)

    def get_diagnostics(self) -> ShadowDiagnostics:
        """Compute aggregate diagnostics (requires m_true in events)."""
        if isinstance(self.shadow, ProbShadowObserver):
            return self.shadow.compute_diagnostics(
                self._m_true_history, self._frozen_history)
        # Null shadow — return basic comparison
        diag = ShadowDiagnostics(n_steps=self._step)
        for dim in DIM_NAMES:
            errs = []
            for t in range(self._step):
                true_val = self._m_true_history[t].get(dim, DIM_PRIORS[dim])
                frozen_val = self._frozen_history[t].get(dim, DIM_PRIORS[dim])
                errs.append((frozen_val - true_val) ** 2)
            diag.rmse_frozen[dim] = float(np.sqrt(np.mean(errs))) if errs else 0.0
            diag.rmse[dim] = diag.rmse_frozen[dim]  # null shadow = frozen
        return diag

    def directional_responses(self) -> dict:
        """Compute directional event responses from shadow history.

        Returns dict of {event_type: mean_delta} for each relevant dim.
        """
        if len(self._shadow_history) < 2:
            return {}

        responses = {}
        deltas = {dim: [] for dim in DIM_NAMES}

        for t in range(1, len(self._shadow_history)):
            snap = self._shadow_history[t]
            prev = self._shadow_history[t - 1]
            events = snap.events_used

            # tau: trust+ should increase, trust- should decrease
            if events.get("trust+", 0) > 0:
                d = snap.mean("tau") - prev.mean("tau")
                responses.setdefault("delta_tau_trust+", []).append(d)
            if events.get("trust-", 0) > 0:
                d = snap.mean("tau") - prev.mean("tau")
                responses.setdefault("delta_tau_trust-", []).append(d)

            # nu: blind should increase, selfdisc should decrease
            if events.get("blind", 0) > 0.05:
                d = snap.mean("nu") - prev.mean("nu")
                responses.setdefault("delta_nu_blind", []).append(d)
            if events.get("selfdisc", 0) > 0.05:
                d = snap.mean("nu") - prev.mean("nu")
                responses.setdefault("delta_nu_selfdisc", []).append(d)

            # gamma_gen: pressure should increase, explore+ should decrease
            if events.get("pressure", 0) > 0.1:
                d = snap.mean("gamma_gen") - prev.mean("gamma_gen")
                responses.setdefault("delta_gg_pressure", []).append(d)
            if events.get("explore+", 0) > 0:
                d = snap.mean("gamma_gen") - prev.mean("gamma_gen")
                responses.setdefault("delta_gg_explore+", []).append(d)

            # gamma_spec: resist should increase
            if events.get("lure", 0) >= 0.3:
                d = snap.mean("gamma_spec") - prev.mean("gamma_spec")
                responses.setdefault("delta_gs_lure", []).append(d)

        # Aggregate
        result = {}
        for key, vals in responses.items():
            if vals:
                result[key] = {
                    "mean": float(np.mean(vals)),
                    "n": len(vals),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                }
        return result
