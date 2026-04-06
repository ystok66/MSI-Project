"""Tests for Step 3 Shadow Probabilistic Observer.

9 directional tests covering:
  1. posterior normalization
  2. support / clipping correctness
  3. null shadow parity (Phase 0)
  4. trust+ -> tau posterior mean increases
  5. blind_obey -> nu posterior mean increases
  6. self_discovery -> nu posterior mean decreases
  7. resisted lure -> gamma_spec posterior mean increases
  8. explore+ -> gamma_gen posterior mean decreases
  9. frozen modules untouched / no import-side mutation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.teachers.internalization_observer import ObsEvent, A1MtObserverFrozen
from src.teachers.a1mt_observer_shadow_prob import (
    NullShadowObserver, ProbShadowObserver,
)
from src.teachers.a1mt_observer_shadow_bridge import ShadowObserverBridge
from src.teachers.a1mt_observer_shadow_types import DIM_NAMES, DIM_PRIORS


def _ev(**kw):
    """Convenience ObsEvent constructor."""
    return ObsEvent(**kw)


passed = 0
failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


# ─── Test 1: Posterior normalization ──────────────────────────
print("\n1. Posterior normalization")
obs = ProbShadowObserver(n_grid=32)
for _ in range(5):
    obs.step(_ev(dose=0.5, warned=True, follow_warn=True, warn_correct=True, p_self=0.3))
for dim in DIM_NAMES:
    w_sum = obs._weights[dim].sum()
    check(f"{dim} weights sum to 1", abs(w_sum - 1.0) < 1e-6,
          f"sum={w_sum:.8f}")


# ─── Test 2: Support / clipping correctness ──────────────────
print("\n2. Support / clipping")
obs = ProbShadowObserver(n_grid=32)
obs.step(_ev())
for dim in DIM_NAMES:
    from src.teachers.a1mt_observer_shadow_types import DIM_BOUNDS
    lo, hi = DIM_BOUNDS[dim]
    grid = obs._grids[dim]
    check(f"{dim} grid within bounds",
          grid.min() >= lo - 1e-5 and grid.max() <= hi + 1e-5,
          f"range=[{grid.min():.4f}, {grid.max():.4f}], bounds=[{lo}, {hi}]")


# ─── Test 3: Null shadow parity (Phase 0) ───────────────────
print("\n3. Null shadow parity")
bridge = ShadowObserverBridge(shadow_mode="null", n_grid=32)
bridge.reset()
events = [
    _ev(dose=1.0, warned=True, follow_warn=True, warn_correct=True, p_self=0.3),
    _ev(dose=0.0, self_discovery=True, p_self=0.8),
    _ev(dose=0.5, warned=True, follow_warn=True, warn_wrong=True, p_self=0.4),
]
for ev in events:
    bridge.step(ev)
frozen_est = bridge.frozen.get_estimate()
shadow_est = bridge.shadow.get_estimate()
for dim in DIM_NAMES:
    f_val = frozen_est.get(dim, 0)
    s_val = shadow_est.get(dim, 0)
    check(f"null {dim} matches frozen",
          abs(f_val - s_val) < 1e-4,
          f"frozen={f_val:.6f} shadow={s_val:.6f}")


# ─── Test 4: trust+ -> tau increases ─────────────────────────
print("\n4. trust+ -> tau increases")
obs = ProbShadowObserver(n_grid=32)
obs.step(_ev())  # neutral step
tau_before = obs.get_estimate()["tau"]
obs.step(_ev(warned=True, follow_warn=True, warn_correct=True, p_self=0.3))
tau_after = obs.get_estimate()["tau"]
check("tau increases after trust+",
      tau_after > tau_before,
      f"before={tau_before:.4f} after={tau_after:.4f}")


# ─── Test 5: blind_obey -> nu increases ──────────────────────
print("\n5. blind_obey -> nu increases")
obs = ProbShadowObserver(n_grid=32)
obs.step(_ev())
nu_before = obs.get_estimate()["nu"]
obs.step(_ev(warned=True, follow_warn=True, p_self=0.1))  # low p_self -> high blind
nu_after = obs.get_estimate()["nu"]
check("nu increases after blind obey (low p_self)",
      nu_after > nu_before,
      f"before={nu_before:.4f} after={nu_after:.4f}")


# ─── Test 6: self_discovery -> nu decreases ──────────────────
print("\n6. self_discovery -> nu decreases")
obs = ProbShadowObserver(n_grid=32)
# First push nu up a bit
for _ in range(3):
    obs.step(_ev(warned=True, follow_warn=True, p_self=0.1))
nu_before = obs.get_estimate()["nu"]
obs.step(_ev(self_discovery=True, p_self=0.8))
nu_after = obs.get_estimate()["nu"]
check("nu decreases after self_discovery (high p_self)",
      nu_after < nu_before,
      f"before={nu_before:.4f} after={nu_after:.4f}")


# ─── Test 7: resisted lure -> gamma_spec increases ───────────
print("\n7. resisted lure -> gamma_spec increases")
obs = ProbShadowObserver(n_grid=32)
# Warm up with several resist events to push gamma_spec above 0
for _ in range(5):
    obs.step(_ev(lure=0.6, agent_choice=0, oracle_safe=0))
gs_before = obs.get_estimate()["gamma_spec"]
obs.step(_ev(lure=0.6, agent_choice=0, oracle_safe=0))  # one more resist
gs_after = obs.get_estimate()["gamma_spec"]
check("gamma_spec increases after resisted lure",
      gs_after > gs_before,
      f"before={gs_before:.4f} after={gs_after:.4f}")


# ─── Test 8: explore+ -> gamma_gen decreases ────────────────
print("\n8. explore+ -> gamma_gen decreases")
obs = ProbShadowObserver(n_grid=32)
# Push gamma_gen up with sustained pressure (many steps)
for _ in range(10):
    obs.step(_ev(dose=1.0))
gg_before = obs.get_estimate()["gamma_gen"]
obs.step(_ev(beneficial_novelty=True))
gg_after = obs.get_estimate()["gamma_gen"]
check("gamma_gen decreases after explore+",
      gg_after < gg_before,
      f"before={gg_before:.4f} after={gg_after:.4f}")


# ─── Test 9: Frozen modules untouched ────────────────────────
print("\n9. Frozen modules untouched")
# Verify frozen observer params haven't changed
frozen = A1MtObserverFrozen()
check("frozen beta_tau_probe=0.0",
      frozen.beta_tau_probe == 0.0,
      f"got {frozen.beta_tau_probe}")
check("frozen beta_nu_probe=0.0",
      frozen.beta_nu_probe == 0.0,
      f"got {frozen.beta_nu_probe}")
check("frozen lambda_tau=0.005",
      frozen.lambda_tau == 0.005,
      f"got {frozen.lambda_tau}")
# Verify no side effects from importing shadow modules
try:
    frozen2 = A1MtObserverFrozen()
    frozen2.update(_ev(warned=True, follow_warn=True, warn_correct=True, p_self=0.3))
    check("frozen update still works", True)
except Exception as e:
    check("frozen update still works", False, str(e))


# ─── Summary ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
