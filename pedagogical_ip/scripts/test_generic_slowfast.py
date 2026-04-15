"""Task 3: GenericSlowFastPredictor sanity tests."""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.structured_basis_head import StructuredBasisCostRiskHead
from src.agents.slow_fast_head import GenericSlowFastPredictor, SlowFastCostRiskHead
from src.agents.predictor_protocol import (
    PredictorProtocol, snapshot_predictor, restore_predictor,
    extract_theta, extract_theta_components, predictor_summary
)

x = np.array([0.5, 0.3, 0.8, 0.6])

print("=== T1.1: Protocol compatibility ===")
for name, factory in [
    ("Linear", lambda: LatentCostRiskHead(d=4)),
    ("Basis", lambda: StructuredBasisCostRiskHead(d=4)),
]:
    for alpha in [0.1, 0.5]:
        sf = GenericSlowFastPredictor(base_factory=factory, alpha=alpha)
        assert isinstance(sf, PredictorProtocol), f"{name} a={alpha} fails protocol"
        sf.begin_episode()
        c = sf.predict_cost(x)
        r = sf.predict_risk(x)
        sf.update_from_outcome(x, 1.5, 0.3)
        sf.end_episode()
        snap = snapshot_predictor(sf)
        theta = extract_theta(sf)
        cw, cb, rw, rb = extract_theta_components(sf)
        s = predictor_summary(sf)
        print(f"  {name} a={alpha}: OK | theta={len(theta)} dims=({s['cost_w_dim']},{s['risk_w_dim']})")

print()
print("=== T1.2: alpha extreme sanity ===")
for name, factory in [
    ("Linear", lambda: LatentCostRiskHead(d=4)),
    ("Basis", lambda: StructuredBasisCostRiskHead(d=4)),
]:
    # alpha=0: slow never changes
    sf0 = GenericSlowFastPredictor(base_factory=factory, alpha=0.0)
    slow_before = sf0.slow_risk_w.copy()
    sf0.begin_episode()
    for _ in range(10):
        sf0.update_from_outcome(x, 1.5, 0.3)
    sf0.end_episode()
    slow_after = sf0.slow_risk_w
    delta0 = np.linalg.norm(slow_after - slow_before)
    assert delta0 < 1e-12, f"{name} a=0 slow changed! delta={delta0}"
    print(f"  {name} a=0: slow unchanged (delta={delta0:.2e}) PASS")

    # alpha=1: slow = fast_end
    sf1 = GenericSlowFastPredictor(base_factory=factory, alpha=1.0)
    sf1.begin_episode()
    for _ in range(10):
        sf1.update_from_outcome(x, 1.5, 0.3)
    fast_end_w = sf1.risk_head.w.copy()
    sf1.end_episode()
    slow_w = sf1.slow_risk_w
    delta1 = np.linalg.norm(slow_w - fast_end_w)
    assert delta1 < 1e-12, f"{name} a=1 slow!=fast_end! delta={delta1}"
    print(f"  {name} a=1: slow==fast_end (delta={delta1:.2e}) PASS")

print()
print("=== T1.3: Backward compat ===")
sf_old = SlowFastCostRiskHead(d=4, alpha=0.2)
assert isinstance(sf_old, GenericSlowFastPredictor)
assert isinstance(sf_old, PredictorProtocol)
sf_old.begin_episode()
sf_old.update_from_outcome(x, 1.5, 0.3)
sf_old.end_episode()
print("  SlowFastCostRiskHead compat: OK")

print()
print("=== T1.4: Multi-episode accumulation ===")
for name, factory in [
    ("Linear", lambda: LatentCostRiskHead(d=4)),
    ("Basis", lambda: StructuredBasisCostRiskHead(d=4)),
]:
    sf = GenericSlowFastPredictor(base_factory=factory, alpha=0.3)
    norms = []
    for ep in range(5):
        sf.begin_episode()
        for _ in range(5):
            sf.update_from_outcome(x, 1.5, 0.7, weight=2.0)
        norms.append(round(float(np.linalg.norm(sf.slow_risk_w)), 4))
        sf.end_episode()
    norms.append(round(float(np.linalg.norm(sf.slow_risk_w)), 4))
    print(f"  {name}: slow_risk_norms = {norms}")

print()
print("=== T1.5: Jacobian uncertainty (Basis) ===")
bh = StructuredBasisCostRiskHead(d=4)
bh.update_from_outcome(x, 1.5, 0.3, weight=2.0)
x_var = np.array([0.1, 0.1, 0.2, 0.2])
uc = bh.predict_cost_uncertainty_from_var(x_var)
ur = bh.predict_risk_uncertainty_from_var(x_var)
print(f"  cost_unc_from_var = {uc:.6f}")
print(f"  risk_unc_from_var = {ur:.6f}")
assert uc > 0 and ur > 0, "Uncertainty must be positive"
assert np.isfinite(uc) and np.isfinite(ur), "Uncertainty must be finite"
print("  Jacobian uncertainty: PASS")

print()
print("=== T1.6: Full episode integration ===")
from src.envs.lattice_v2_runner import LatticeV2Runner
runner = LatticeV2Runner()
for name, factory in [
    ("Linear", lambda: LatentCostRiskHead(d=4)),
    ("Basis", lambda: StructuredBasisCostRiskHead(d=4)),
]:
    sf = GenericSlowFastPredictor(base_factory=factory, alpha=0.3)
    sf.begin_episode()
    state = runner.reset(
        seed=42, latent_mode=True, latent_predictor=sf,
        tutor_mode="none", warning_mode="none", patch_radius=2,
        prefix_horizon=5, belief_planning_mode=True,
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=True, difficulty="medium")
    while not state.done:
        state = runner.step(state)
    sf.end_episode()
    print(f"  {name}: surv={state.survived} goal={state.reached_goal} steps={state.steps}")

print()
print("ALL SANITY TESTS PASS")
