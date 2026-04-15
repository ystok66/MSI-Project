"""Phase 2B regression check: verify GenericSlowFastPredictor doesn't regress on canonical families."""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.structured_basis_head import StructuredBasisCostRiskHead
from src.agents.slow_fast_head import GenericSlowFastPredictor, SlowFastCostRiskHead

FAMILIES = ["baseline_v2", "goal_preference_temptation_entanglement_lattice",
            "deep_tree_mixed_bottleneck_lattice"]
N_SEEDS = 5

runner = LatticeV2Runner()
print("Phase 2B Regression: canonical families x 3 head types")
print("=" * 80)

for family in FAMILIES:
    print(f"\n--- {family} ---")
    for head_name, make in [
        ("linear", lambda: LatentCostRiskHead(d=4)),
        ("basis", lambda: StructuredBasisCostRiskHead(d=4)),
        ("sf_basis_0.3", lambda: GenericSlowFastPredictor(
            base_factory=lambda: StructuredBasisCostRiskHead(d=4), alpha=0.3)),
    ]:
        survs = []
        goals = []
        for s in range(N_SEEDS):
            pred = make()
            if hasattr(pred, 'begin_episode'):
                pred.begin_episode()
            state = runner.reset(
                seed=s * 100, latent_mode=True, latent_predictor=pred,
                tutor_mode="none", warning_mode="none", patch_radius=2,
                prefix_horizon=5, belief_planning_mode=True,
                robot_belief_mode=True, intervention_family_mode=True,
                item_drop_enabled=True, difficulty="medium",
                scenario_family=family)
            while not state.done:
                state = runner.step(state)
            survs.append(state.survived)
            goals.append(state.reached_goal)
        s_rate = np.mean(survs)
        g_rate = np.mean(goals)
        print(f"  {head_name:>15s}: surv={s_rate:.3f} goal={g_rate:.3f}")

print("\nRegression check complete.")
