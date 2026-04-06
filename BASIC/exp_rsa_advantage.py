"""
Experiment to find cases where RSA outperforms L0.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import infer_posterior, score_L0
from scipy.special import softmax


def make_obj(s, c):
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], 
               occ=np.array(SHAPES[s], dtype=np.float32))


def pad_features(X, target_dim):
    if X.shape[1] >= target_dim:
        return X
    padding = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=X.dtype)
    return np.concatenate([X, padding], axis=1)


def l0_posterior(X, mask, tokens, table, alpha=5.0):
    l0_scores = score_L0(X, mask, tuple(t.lower().strip() for t in tokens), table)
    valid = mask
    valid_scores = l0_scores[valid]
    probs = np.zeros(4)
    probs[valid] = softmax(valid_scores * alpha)
    return probs


def run_tests(d=12):
    print(f"\n{'='*60}")
    print(f"DIMENSION d={d}")
    print('='*60)
    
    results = []
    
    # =========================================================================
    # Test A: Mutual Exclusivity - RSA should show ME effect
    # =========================================================================
    print("\n--- Test A: Mutual Exclusivity ---")
    print("Train 'a' on blue objects")
    print("Test: blue_box(familiar) vs red_solid(novel), query novel word 'b'")
    
    table = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['a'], table=table)
    
    # Test scene
    X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'blue'),   # familiar=a
                                                  make_obj('solid', 'red'),  # novel
                                                  None, None]))
    X_test = pad_features(X_test, d)
    
    l0 = l0_posterior(X_test, m_test, ['b'], table)
    rsa = infer_posterior(X_test, m_test, ['b'], table)
    
    print(f"  L0:  P(familiar)={l0[0]:.4f}, P(novel)={l0[1]:.4f}")
    print(f"  RSA: P(familiar)={rsa[0]:.4f}, P(novel)={rsa[1]:.4f}")
    print(f"  L0 prefers novel: {l0[1] > l0[0]}, RSA prefers novel: {rsa[1] > rsa[0]}")
    results.append(('ME Effect', l0[1], rsa[1], l0[1] > l0[0], rsa[1] > rsa[0]))
    
    # =========================================================================
    # Test B: Competition Effect - two similar objects, one better fit
    # =========================================================================
    print("\n--- Test B: Competition with close alternatives ---")
    print("Train 'green' and 'blue' concepts")
    print("Test: green_box vs cyan_box (similar colors), query 'green'")
    
    table = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['b'], table=table)
    
    # Test with green vs blue (clear difference)
    X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), 
                                                  make_obj('t', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0 = l0_posterior(X_test, m_test, ['g'], table)
    rsa = infer_posterior(X_test, m_test, ['g'], table)
    
    print(f"  L0:  P(green)={l0[0]:.4f}, P(blue)={l0[1]:.4f}")
    print(f"  RSA: P(green)={rsa[0]:.4f}, P(blue)={rsa[1]:.4f}")
    results.append(('Competition', l0[0], rsa[0], l0[0] > 0.5, rsa[0] > 0.5))
    
    # =========================================================================
    # Test C: Pragmatic precision - both match but one is better
    # =========================================================================
    print("\n--- Test C: Pragmatic Precision ---")
    print("Train 'shape' concepts (l, t, box)")
    print("Test: l vs t with concept trained only on l")
    
    table = ConceptTable(d=d)
    for color in ['green', 'blue', 'red']:
        X, m = encode_scene(Scene(regions=[make_obj('l', color), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['L'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj('t', color), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['T'], table=table)
    
    # Test: green_l vs green_t, query 'L'
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), 
                                                  make_obj('t', 'green'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0 = l0_posterior(X_test, m_test, ['L'], table)
    rsa = infer_posterior(X_test, m_test, ['L'], table)
    
    print(f"  L0:  P(l)={l0[0]:.4f}, P(t)={l0[1]:.4f}")
    print(f"  RSA: P(l)={rsa[0]:.4f}, P(t)={rsa[1]:.4f}")
    results.append(('Shape Precision', l0[0], rsa[0], l0[0] > l0[1], rsa[0] > rsa[1]))
    
    # =========================================================================
    # Test D: Ambiguous reference resolution
    # =========================================================================
    print("\n--- Test D: Ambiguous Reference ---")
    print("Scene: green_box, green_solid, blue_l")
    print("Query: 'box' - should prefer green_box clearly")
    
    table = ConceptTable(d=d)
    for color in ['green', 'blue', 'red']:
        X, m = encode_scene(Scene(regions=[make_obj('box', color), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['box'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj('solid', color), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['solid'], table=table)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'green'), 
                                                  make_obj('solid', 'green'),
                                                  make_obj('l', 'blue'), None]))
    X_test = pad_features(X_test, d)
    
    l0 = l0_posterior(X_test, m_test, ['box'], table)
    rsa = infer_posterior(X_test, m_test, ['box'], table)
    
    print(f"  L0:  P(box)={l0[0]:.4f}, P(solid)={l0[1]:.4f}, P(l)={l0[2]:.4f}")
    print(f"  RSA: P(box)={rsa[0]:.4f}, P(solid)={rsa[1]:.4f}, P(l)={rsa[2]:.4f}")
    results.append(('Ambiguous Ref', l0[0], rsa[0], l0[0] > l0[1], rsa[0] > rsa[1]))
    
    return results


print("="*60)
print("FINDING CASES WHERE RSA OUTPERFORMS L0")
print("="*60)

results_12 = run_tests(d=12)
results_15 = run_tests(d=15)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Test':<20} | {'d=12 L0':<10} {'d=12 RSA':<10} | {'d=15 L0':<10} {'d=15 RSA':<10}")
print("-"*60)
for i, (name, l0_12, rsa_12, _, _) in enumerate(results_12):
    _, l0_15, rsa_15, _, _ = results_15[i]
    diff_12 = "RSA>" if rsa_12 > l0_12 + 0.01 else ("L0>" if l0_12 > rsa_12 + 0.01 else "=")
    diff_15 = "RSA>" if rsa_15 > l0_15 + 0.01 else ("L0>" if l0_15 > rsa_15 + 0.01 else "=")
    print(f"{name:<20} | {l0_12:<10.4f} {rsa_12:<10.4f} | {l0_15:<10.4f} {rsa_15:<10.4f}   {diff_12} {diff_15}")
