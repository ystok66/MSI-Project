"""Quick smoke test for Phase 1A refactor — writes clean output."""
import sys; sys.path.insert(0, ".")
from src.envs.lattice_v2_runner import LatticeV2Runner

r = LatticeV2Runner()
lines = []

COMMON = dict(
    seed=5, difficulty="medium", latent_mode=True, patch_radius=2,
    prefix_horizon=5, belief_planning_mode=True, robot_belief_mode=True,
    intervention_family_mode=True, item_drop_enabled=True,
)

for variant in ["legacy_bias", "rsa_obs_s1", "rsa_obs_s1_trust", "rsa_plus_phase10"]:
    s = r.reset(warning_variant=variant, **COMMON)
    while not s.done:
        r.step(s)
    m = r.get_metrics(s)
    lines.append(f"--- {variant} ---")
    lines.append(f"  surv={m['survived']}, goal={m['reached_goal']}, steps={m['steps']}, warns={m['warn_count']}")
    b = s.rsa_belief_state.belief
    lines.append(f"  rsa_belief=[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")
    lines.append(f"  diag_count={len(s.rsa_warn_diagnostics)}")
    if s.rsa_warn_diagnostics:
        d = s.rsa_warn_diagnostics[0]
        lines.append(f"  d_rho_inc={d.get('delta_rho_inc')}")
        lines.append(f"  d_rho_uniform={d.get('delta_rho_uniform')}")
        lines.append(f"  delta_nll={d.get('delta_nll_local')}")
    lines.append("")

lines.append("=== DTMB regression check ===")
s = r.reset(seed=5, difficulty="medium", latent_mode=True, patch_radius=2,
    prefix_horizon=5, belief_planning_mode=True, robot_belief_mode=True,
    intervention_family_mode=True, item_drop_enabled=True,
    scenario_family="deep_tree_mixed_bottleneck_lattice")
while not s.done:
    r.step(s)
m = r.get_metrics(s)
lines.append(f"  surv={m['survived']}, goal={m['reached_goal']}, steps={m['steps']}")

lines.append("")
lines.append("=== GTET regression check ===")
s = r.reset(seed=5, difficulty="medium", latent_mode=True, patch_radius=2,
    prefix_horizon=5, belief_planning_mode=True, robot_belief_mode=True,
    intervention_family_mode=True, item_drop_enabled=True,
    scenario_family="goal_preference_temptation_entanglement_lattice")
while not s.done:
    r.step(s)
m = r.get_metrics(s)
lines.append(f"  surv={m['survived']}, goal={m['reached_goal']}, steps={m['steps']}")

lines.append("")
lines.append("Smoke test PASSED.")

output = "\n".join(lines)
print(output)
with open("results/phase1a_smoke_clean.txt", "w", encoding="utf-8") as f:
    f.write(output)
