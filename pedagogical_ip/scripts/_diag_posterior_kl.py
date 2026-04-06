"""Quick diagnostic: check posterior weights and KL after updates."""
import sys
sys.path.insert(0, ".")
import numpy as np

from src.teachers.joint_goal_pref_posterior import (
    JointGoalPrefPosterior, THETA_2, DEFAULT_TEMPT_GRID, DEFAULT_TEMPT_PRIOR,
)
from src.agents.stochastic_agent_policy import BranchAttributes
from src.teachers.gtet_factor_adapter import (
    build_factor_restricted_view, compute_posterior_epistemic_modifier,
)

jgpp = JointGoalPrefPosterior(
    pref_types=THETA_2,
    tempt_grid=DEFAULT_TEMPT_GRID,
    tempt_prior=DEFAULT_TEMPT_PRIOR,
)

print("Initial weights shape:", jgpp._weights().shape)
print("Initial entropy:", jgpp.entropy())
w0 = jgpp._weights()
print("Initial weight sample (first 5 flat):", w0.ravel()[:5])
print("Is product? max diff:", np.max(np.abs(w0 - w0)))  # trivially 0

# Simulate 10 updates with strong branch contrasts
branches_strong = [
    BranchAttributes(safety_score=0.9, temptation_score=0.1,
                     texture_novelty=0.2, shortcut_bonus=0.0,
                     risk_penalty=0.1),
    BranchAttributes(safety_score=0.2, temptation_score=0.8,
                     texture_novelty=0.7, shortcut_bonus=0.2,
                     risk_penalty=0.5),
]

for t in range(10):
    obs = 0 if t < 5 else 1  # first 5: safe, last 5: risky
    jgpp.update(None, branches_strong, obs)

w = jgpp._weights()
print(f"\nAfter 10 updates:")
print(f"  Entropy: {jgpp.entropy():.4f}")
print(f"  MAP: {jgpp.map_hypothesis()}")
print(f"  Marginal goal: {jgpp.marginal_goal()}")
print(f"  Marginal pref: {jgpp.marginal_pref()}")
print(f"  Marginal tempt: {jgpp.marginal_tempt()}")

# Check if weights are factored
mg = w.sum(axis=(1,2))
mp = w.sum(axis=(0,2))
mz = w.sum(axis=(0,1))
product = np.einsum('g,p,z->gpz', mg, mp, mz)
factored_diff = np.max(np.abs(w - product))
print(f"\n  Max |w - product(marginals)|: {factored_diff:.6f}")
print(f"  Are weights factored? {'YES (no correlation)' if factored_diff < 0.01 else 'NO (correlations exist!)'}")

# Test factor restriction
for mode in ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY"]:
    q_r = build_factor_restricted_view(jgpp, mode)
    kl = compute_posterior_epistemic_modifier(w, q_r)
    print(f"  KL(full || {mode:10s}) = {kl:.6f}")
