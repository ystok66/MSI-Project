"""Phase A unit tests: V5.1 Branch Summary, V5.2 Gaussian Concepts, V5.3 Familiarity.

Tests per spec:
  V5.1 Test 1: same branch → same summary (determinism)
  V5.1 Test 2: safe/risky oracle summary separable
  V5.1 Test 3: mirror map → mirror summary (side-invariant)
  V5.2 Test 1: new concept broad prior
  V5.2 Test 2: repeated update → κ grows, variance shrinks
  V5.2 Test 3: safe/risky ranking correct
  V5.3 Test 1: mature concept → higher familiarity
  V5.3 Test 2: unseen summary → below novelty threshold
  V5.3 Test 3: side swap with same semantics → similar familiarity
"""
import sys
sys.path.insert(0, ".")
import numpy as np

from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch, SUMMARY_DIM
from src.agents.branch_concepts import (
    BranchConceptLibrary, log_inclusion_score, update_concept
)
from src.agents.familiarity import familiarity_score, is_novel, teaching_priority

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1

# ── Setup ──
lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
# Train to distinguish safe (low texture) vs risky (high texture)
rng = np.random.default_rng(42)
for _ in range(50):
    safe_x = rng.uniform([0, 0, 0.05, 0.05], [1, 1, 0.15, 0.10])
    lp.update_from_outcome(safe_x, cost_label=1.0, risk_label=0.1)
    risky_x = rng.uniform([0, 0, 0.5, 0.4], [1, 1, 0.8, 0.7])
    lp.update_from_outcome(risky_x, cost_label=1.0, risk_label=0.7)

# Mock belief maps (7x11x4)
H, W = 7, 11
belief_mean = np.full((H, W, 4), 0.3)
belief_var = np.full((H, W, 4), 0.5)

# Branch cells
safe_cells = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
risky_cells = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]

# Set safe branch features (low texture)
for r, c in safe_cells:
    belief_mean[r, c] = [0.0, 0.0, 0.08, 0.06]
# Set risky branch features (high texture)
for r, c in risky_cells:
    belief_mean[r, c] = [1.0, 0.0, 0.6, 0.5]

# ═══════════════════════════════════════════════════════════════
print("=== V5.1 Branch Summary ===")

# Test 1: Determinism
s1 = summarize_branch(safe_cells, belief_mean, belief_var, lp)
s2 = summarize_branch(safe_cells, belief_mean, belief_var, lp)
check("T1: determinism", np.allclose(s1, s2))
check("T1: correct dim", s1.shape == (SUMMARY_DIM,))

# Test 2: Safe/risky separable
s_safe = summarize_branch(safe_cells, belief_mean, belief_var, lp)
s_risky = summarize_branch(risky_cells, belief_mean, belief_var, lp)
check("T2: safe mean_risk < risky mean_risk",
      s_safe[0] < s_risky[0])
check("T2: safe max_risk < risky max_risk",
      s_safe[1] < s_risky[1])
print(f"     safe summary:  {np.round(s_safe, 3)}")
print(f"     risky summary: {np.round(s_risky, 3)}")

# Test 3: Mirror symmetry
mirror_safe = [(3, c) for _, c in safe_cells]  # row 3 instead of 1
mirror_risky = [(1, c) for _, c in risky_cells]  # row 1 instead of 3
# Swap features to match
belief_mirror = belief_mean.copy()
for (or_, oc), (mr, mc) in zip(safe_cells, mirror_safe):
    belief_mirror[mr, mc] = belief_mean[or_, oc]
for (or_, oc), (mr, mc) in zip(risky_cells, mirror_risky):
    belief_mirror[mr, mc] = belief_mean[or_, oc]

s_mirror_safe = summarize_branch(mirror_safe, belief_mirror, belief_var, lp)
check("T3: mirror safe ≈ original safe",
      np.allclose(s_safe, s_mirror_safe, atol=0.05))

# ═══════════════════════════════════════════════════════════════
print("\n=== V5.2 Gaussian Branch Concepts ===")

lib = BranchConceptLibrary()

# Test 1: Broad prior
check("T1: initial kappa=1.0", lib.concepts["safe_branch"].kappa == 1.0)
check("T1: initial var all=1.0",
      np.allclose(lib.concepts["safe_branch"].var, 1.0))

# Test 2: Update → κ grows, variance shrinks
lib2 = BranchConceptLibrary()
initial_var = lib2.concepts["safe_branch"].var.copy()
for _ in range(20):
    # Feed safe-like summaries
    fake_safe = s_safe + rng.normal(0, 0.05, SUMMARY_DIM)
    lib2.update("safe_branch", fake_safe)

check("T2: kappa grew", lib2.concepts["safe_branch"].kappa > 10)
check("T2: variance shrunk",
      float(np.mean(lib2.concepts["safe_branch"].var)) <
      float(np.mean(initial_var)))
print(f"     kappa: {lib2.concepts['safe_branch'].kappa:.1f}, "
      f"mean_var: {np.mean(lib2.concepts['safe_branch'].var):.4f}")

# Test 3: Ranking correct
score_safe_as_safe = log_inclusion_score(s_safe, lib2.concepts["safe_branch"])
score_risky_as_safe = log_inclusion_score(s_risky, lib2.concepts["safe_branch"])
check("T3: safe branch scores higher on safe concept",
      score_safe_as_safe > score_risky_as_safe)
print(f"     safe→safe: {score_safe_as_safe:.3f}, risky→safe: {score_risky_as_safe:.3f}")

# Train risky concept too
for _ in range(20):
    fake_risky = s_risky + rng.normal(0, 0.05, SUMMARY_DIM)
    lib2.update("risky_branch", fake_risky)

# Best concept should match
best_for_safe, _ = lib2.best_concept(s_safe)
best_for_risky, _ = lib2.best_concept(s_risky)
check("T3: best for safe is safe_branch", best_for_safe == "safe_branch")
check("T3: best for risky is risky_branch", best_for_risky == "risky_branch")

# ═══════════════════════════════════════════════════════════════
print("\n=== V5.3 Familiarity ===")

# Test 1: Mature concept → higher familiarity
lib_mature = BranchConceptLibrary()
for _ in range(50):
    lib_mature.update("safe_branch", s_safe + rng.normal(0, 0.03, SUMMARY_DIM))
    lib_mature.update("risky_branch", s_risky + rng.normal(0, 0.03, SUMMARY_DIM))

f_known = familiarity_score(s_safe, lib_mature)
lib_fresh = BranchConceptLibrary()
f_fresh = familiarity_score(s_safe, lib_fresh)
check("T1: mature > fresh familiarity", f_known > f_fresh)
print(f"     mature: {f_known:.3f}, fresh: {f_fresh:.3f}")

# Test 2: Completely unseen pattern → novel
weird = np.array([0.99, 0.99, 5.0, 0.01, 0.01, 0.0, 0.0, 1.0])
check("T2: unseen is novel", is_novel(weird, lib_mature))

# Known pattern should NOT be novel
check("T2: known is not novel", not is_novel(s_safe, lib_mature))

# Test 3: Side swap with same semantics → similar familiarity
f_mirror = familiarity_score(s_mirror_safe, lib_mature)
check("T3: side-swap ≈ same familiarity",
      abs(f_known - f_mirror) < 1.0)
print(f"     original: {f_known:.3f}, mirror: {f_mirror:.3f}")

# Teaching priority
tp = teaching_priority(weird, lib_mature, risk_estimate=0.2)
check("T_extra: novel+low_risk → teach", tp["teaching_mode"] == "teach")
tp2 = teaching_priority(s_risky, lib_mature, risk_estimate=0.8)
check("T_extra: known+high_risk → rescue", tp2["teaching_mode"] == "rescue")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
print("All Phase A tests passed.")
