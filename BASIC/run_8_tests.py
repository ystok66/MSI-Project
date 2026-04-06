"""Run all 8 TEST_CASES comparing RSA vs L0."""
import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable, Concept
from learner import learn_step
from rsa import infer_posterior, score_L0
from scipy.special import softmax

def make_obj(shape_key, color_key):
    occ = np.array(SHAPES[shape_key], dtype=np.float32)
    rgb = COLORS_RGB[color_key]
    return Obj(shape_name=shape_key, color_rgb=rgb, occ=occ)

def l0_posterior(X, mask, tokens, table, alpha=5.0):
    """Compute L0-only posterior (no pragmatic reasoning)."""
    from rsa import normalize_tokens
    U = normalize_tokens(tokens)
    l0_scores = score_L0(X, mask, U, table)
    valid_scores = l0_scores[mask]
    probs = np.zeros(4)
    probs[mask] = softmax(valid_scores * alpha)
    return probs

def rsa_posterior(X, mask, tokens, table, alpha=5.0):
    """Compute RSA posterior (with pragmatic reasoning)."""
    return infer_posterior(X, mask, tokens, table, alpha=alpha)

print("=" * 70)
print("RSA vs L0 COMPARISON - 8 TEST CASES")
print("=" * 70)
print()

# =========================================================================
# Test 1: Single Concept Generalization
# =========================================================================
print("TEST 1: Single Concept Generalization to Novel Shapes")
print("-" * 50)
table = ConceptTable(d=12)
# Train on diverse shapes
for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    learn_step(X, m, k=1, tokens=['g'], table=table)

# Test: green L vs blue box
X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), make_obj('box', 'blue'), None, None]))
l0 = l0_posterior(X_test, m_test, ['g'], table)
rsa = rsa_posterior(X_test, m_test, ['g'], table)
print(f"Scene: green_L (0) vs blue_box (1), Query: 'g'")
print(f"  L0:  P(green_L)={l0[0]:.4f}, P(blue_box)={l0[1]:.4f}")
print(f"  RSA: P(green_L)={rsa[0]:.4f}, P(blue_box)={rsa[1]:.4f}")
print(f"  Correct? green_L should win: L0={l0[0]>l0[1]}, RSA={rsa[0]>rsa[1]}")
print()

# =========================================================================
# Test 2: Token Independence from Templates
# =========================================================================
print("TEST 2: Token Independence from Templates")
print("-" * 50)
table = ConceptTable(d=12)
# Train 'xyz' on blue objects
for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
    learn_step(X, m, k=1, tokens=['xyz'], table=table)

# Test: blue L vs red box
X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'blue'), make_obj('box', 'red'), None, None]))
l0 = l0_posterior(X_test, m_test, ['xyz'], table)
rsa = rsa_posterior(X_test, m_test, ['xyz'], table)
print(f"Scene: blue_L (0) vs red_box (1), Query: 'xyz' (trained on blue)")
print(f"  L0:  P(blue_L)={l0[0]:.4f}, P(red_box)={l0[1]:.4f}")
print(f"  RSA: P(blue_L)={rsa[0]:.4f}, P(red_box)={rsa[1]:.4f}")
print(f"  Correct? blue_L should win: L0={l0[0]>l0[1]}, RSA={rsa[0]>rsa[1]}")
print()

# =========================================================================
# Test 3: Diverse Training Prevents Dimensional Dominance
# =========================================================================
print("TEST 3: Diverse Training Prevents Dimensional Dominance")
print("-" * 50)
# Low diversity (only box+solid)
table_low = ConceptTable(d=12)
for shape in ['box', 'solid']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    for _ in range(3):
        learn_step(X, m, k=1, tokens=['g'], table=table_low)

# High diversity (all shapes)
table_high = ConceptTable(d=12)
for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    learn_step(X, m, k=1, tokens=['g'], table=table_high)

# Test: green L vs red box
X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), make_obj('box', 'red'), None, None]))

l0_low = l0_posterior(X_test, m_test, ['g'], table_low)
rsa_low = rsa_posterior(X_test, m_test, ['g'], table_low)
l0_high = l0_posterior(X_test, m_test, ['g'], table_high)
rsa_high = rsa_posterior(X_test, m_test, ['g'], table_high)

print(f"Scene: green_L (0) vs red_box (1), Query: 'g'")
print(f"LOW diversity training (box+solid only):")
print(f"  L0:  P(green_L)={l0_low[0]:.4f}, P(red_box)={l0_low[1]:.4f}")
print(f"  RSA: P(green_L)={rsa_low[0]:.4f}, P(red_box)={rsa_low[1]:.4f}")
print(f"HIGH diversity training (all shapes):")
print(f"  L0:  P(green_L)={l0_high[0]:.4f}, P(red_box)={l0_high[1]:.4f}")
print(f"  RSA: P(green_L)={rsa_high[0]:.4f}, P(red_box)={rsa_high[1]:.4f}")
print()

# =========================================================================
# Test 4: Empty Alternative Enables Single-Concept Discrimination
# =========================================================================
print("TEST 4: Empty Alternative Enables Single-Concept Discrimination")
print("-" * 50)
table = ConceptTable(d=12)
for shape in ['box', 'solid', 'l']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    learn_step(X, m, k=1, tokens=['g'], table=table)

X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), make_obj('box', 'blue'), None, None]))
l0 = l0_posterior(X_test, m_test, ['g'], table)
rsa = rsa_posterior(X_test, m_test, ['g'], table)
print(f"Scene: green_L (0) vs blue_box (1), Query: 'g' (single concept)")
print(f"  L0:  P(green_L)={l0[0]:.4f}, P(blue_box)={l0[1]:.4f}")
print(f"  RSA: P(green_L)={rsa[0]:.4f}, P(blue_box)={rsa[1]:.4f}")
print()

# =========================================================================
# Test 5: Auto Alt From Table for Multi-Concept Discrimination
# =========================================================================
print("TEST 5: Auto Alt From Table for Multi-Concept Discrimination")
print("-" * 50)
table = ConceptTable(d=12)
# Train 'g' on green
for shape in ['box', 'solid', 'l']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    learn_step(X, m, k=1, tokens=['g'], table=table)
# Train 'b' on blue
for shape in ['box', 'solid', 'l']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
    learn_step(X, m, k=1, tokens=['b'], table=table)

X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), make_obj('t', 'blue'), None, None]))
l0 = l0_posterior(X_test, m_test, ['g'], table)
rsa = rsa_posterior(X_test, m_test, ['g'], table)
print(f"Scene: green_T (0) vs blue_T (1), Query: 'g' (with 'b' as competitor)")
print(f"  L0:  P(green_T)={l0[0]:.4f}, P(blue_T)={l0[1]:.4f}")
print(f"  RSA: P(green_T)={rsa[0]:.4f}, P(blue_T)={rsa[1]:.4f}")
print()

# =========================================================================
# Test 6: Intersection Semantics (AND)
# =========================================================================
print("TEST 6: Intersection Semantics (AND)")
print("-" * 50)
table = ConceptTable(d=12)
# Create custom concepts for testing AND
a = table.ensure('a')
a.mu = np.array([0.5] + [0.0]*11)
a.var = np.array([0.01] + [1.0]*11)
a.kappa = 10

b = table.ensure('b')
b.mu = np.array([0.0, 0.5] + [0.0]*10)
b.var = np.array([1.0, 0.01] + [1.0]*10)
b.kappa = 10

# Create test objects
X_test = np.zeros((4, 12))
X_test[0] = [0.5, 0.0] + [0.0]*10  # matches 'a' only
X_test[1] = [0.0, 0.5] + [0.0]*10  # matches 'b' only
X_test[2] = [0.4, 0.4] + [0.0]*10  # matches both!
m_test = np.array([True, True, True, False])

l0 = l0_posterior(X_test, m_test, ['a', 'b'], table)
rsa = rsa_posterior(X_test, m_test, ['a', 'b'], table)
print(f"Scene: matches_a_only (0), matches_b_only (1), matches_both (2)")
print(f"Query: 'a b' (AND semantics)")
print(f"  L0:  P(0)={l0[0]:.4f}, P(1)={l0[1]:.4f}, P(2)={l0[2]:.4f}")
print(f"  RSA: P(0)={rsa[0]:.4f}, P(1)={rsa[1]:.4f}, P(2)={rsa[2]:.4f}")
print(f"  Correct? Region 2 (matches both) should win: L0={np.argmax(l0)==2}, RSA={np.argmax(rsa)==2}")
print()

# =========================================================================
# Test 7: Multi-Intent (basic check)
# =========================================================================
print("TEST 7: Multi-Intent Overlap Detection (basic)")
print("-" * 50)
print("(Multi-intent requires separate API - see infer_joint_posterior)")
print()

# =========================================================================
# Test 8: RSA Mutual Exclusivity
# =========================================================================
print("TEST 8: RSA Mutual Exclusivity (ME)")
print("-" * 50)
table = ConceptTable(d=12)
# Train 'a' on diverse BLUE objects (so 'a' becomes a general blue concept)
for shape in ['box', 'solid', 'l', 't']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
    learn_step(X, m, k=1, tokens=['a'], table=table)

# Test with SAME shape (box) so shape doesn't dominate - color is the key difference
# blue_box (familiar, matches 'a') vs red_box (novel, doesn't match 'a')
X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'blue'), make_obj('box', 'red'), None, None]))
l0 = l0_posterior(X_test, m_test, ['b'], table)
rsa = rsa_posterior(X_test, m_test, ['b'], table)
print(f"Scene: blue_box (0, familiar, matches 'a') vs red_box (1, novel)")
print(f"Query: 'b' (novel word), 'a' exists for blue objects")
print(f"  L0:  P(blue_box)={l0[0]:.4f}, P(red_box)={l0[1]:.4f}")
print(f"  RSA: P(blue_box)={rsa[0]:.4f}, P(red_box)={rsa[1]:.4f}")
print(f"  ME Effect: novel word 'b' should prefer novel object (red_box)")
print(f"  L0 correct? {l0[1] > l0[0]}, RSA correct? {rsa[1] > rsa[0]}")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
