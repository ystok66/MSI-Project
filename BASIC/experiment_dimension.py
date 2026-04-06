"""
Dimension Experiment: Compare RSA vs L0 at d=12 and d=15

This script runs key test cases from TEST_CASES.md and compares:
1. RSA (L1) vs L0-only performance
2. d=12 (original) vs d=15 (extended) concept dimensions
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
    """Pad feature matrix with zeros to target dimension."""
    if X.shape[1] >= target_dim:
        return X
    padding = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=X.dtype)
    return np.concatenate([X, padding], axis=1)


def run_experiment(d=12):
    """Run all test cases at given dimension."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: d={d}")
    print('='*60)
    
    results = {}
    
    # =========================================================================
    # Test 1: Single Concept Generalization
    # =========================================================================
    print("\n--- Test 1: Single Concept Generalization ---")
    print("Train: green box, green solid -> 'g'")
    print("Test: green_l vs blue_box, query 'g'")
    
    table = ConceptTable(d=d)
    
    # Training
    X, m = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
    X = pad_features(X, d)
    learn_step(X, m, k=1, tokens=['g'], table=table)
    
    X, m = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
    X = pad_features(X, d)
    learn_step(X, m, k=1, tokens=['g'], table=table)
    
    # Test
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), 
                                                  make_obj('box', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    # L0 only (softmax of L0 scores)
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_posterior = softmax(l0_scores[:2] * 5.0)  # alpha=5
    
    # RSA (L1)
    rsa_posterior = infer_posterior(X_test, m_test, ['g'], table)
    
    print(f"  L0:  green_l={l0_posterior[0]:.3f}, blue_box={l0_posterior[1]:.3f}, correct={l0_posterior[0] > l0_posterior[1]}")
    print(f"  RSA: green_l={rsa_posterior[0]:.3f}, blue_box={rsa_posterior[1]:.3f}, correct={rsa_posterior[0] > rsa_posterior[1]}")
    
    results['test1'] = {
        'l0': (l0_posterior[0], l0_posterior[1], l0_posterior[0] > l0_posterior[1]),
        'rsa': (rsa_posterior[0], rsa_posterior[1], rsa_posterior[0] > rsa_posterior[1])
    }
    
    # =========================================================================
    # Test 2: Token Independence (xyz = blue)
    # =========================================================================
    print("\n--- Test 2: Token Independence (xyz learned as blue) ---")
    print("Train: blue objects with diverse shapes -> 'xyz'")
    print("Test: blue_l vs red_box, query 'xyz'")
    
    table = ConceptTable(d=d)
    
    # Training with diverse shapes
    for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
        for _ in range(2):
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
            X = pad_features(X, d)
            learn_step(X, m, k=1, tokens=['xyz'], table=table)
    
    # Test
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'blue'), 
                                                  make_obj('box', 'red'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_scores = score_L0(X_test, m_test, ('xyz',), table)
    l0_posterior = softmax(l0_scores[:2] * 5.0)
    rsa_posterior = infer_posterior(X_test, m_test, ['xyz'], table)
    
    print(f"  L0:  blue_l={l0_posterior[0]:.3f}, red_box={l0_posterior[1]:.3f}, correct={l0_posterior[0] > l0_posterior[1]}")
    print(f"  RSA: blue_l={rsa_posterior[0]:.3f}, red_box={rsa_posterior[1]:.3f}, correct={rsa_posterior[0] > rsa_posterior[1]}")
    
    results['test2'] = {
        'l0': (l0_posterior[0], l0_posterior[1], l0_posterior[0] > l0_posterior[1]),
        'rsa': (rsa_posterior[0], rsa_posterior[1], rsa_posterior[0] > rsa_posterior[1])
    }
    
    # =========================================================================
    # Test 3: Diverse vs Limited Training
    # =========================================================================
    print("\n--- Test 3: Effect of Training Diversity ---")
    print("Compare: box+solid only vs diverse shapes")
    
    # Limited training (box + solid only)
    table_limited = ConceptTable(d=d)
    for _ in range(5):
        X, m = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table_limited)
        X, m = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table_limited)
    
    # Diverse training
    table_diverse = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
        for _ in range(2):
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
            X = pad_features(X, d)
            learn_step(X, m, k=1, tokens=['g'], table=table_diverse)
    
    # Test both
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l_90', 'green'),  # Novel shape!
                                                  make_obj('box', 'red'), None, None]))
    X_test = pad_features(X_test, d)
    
    rsa_limited = infer_posterior(X_test, m_test, ['g'], table_limited)
    rsa_diverse = infer_posterior(X_test, m_test, ['g'], table_diverse)
    
    print(f"  Limited training: green_corner={rsa_limited[0]:.3f}, red_box={rsa_limited[1]:.3f}, correct={rsa_limited[0] > rsa_limited[1]}")
    print(f"  Diverse training: green_corner={rsa_diverse[0]:.3f}, red_box={rsa_diverse[1]:.3f}, correct={rsa_diverse[0] > rsa_diverse[1]}")
    
    results['test3'] = {
        'limited': (rsa_limited[0], rsa_limited[1], rsa_limited[0] > rsa_limited[1]),
        'diverse': (rsa_diverse[0], rsa_diverse[1], rsa_diverse[0] > rsa_diverse[1])
    }
    
    # =========================================================================
    # Test 4: Multi-concept discrimination
    # =========================================================================
    print("\n--- Test 4: Multi-concept Discrimination ---")
    print("Train: green objects -> 'g', blue objects -> 'b'")
    print("Test: green_l vs blue_box, query 'g'")
    
    table = ConceptTable(d=d)
    
    # Train both concepts with diverse shapes
    for shape in ['box', 'solid', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['b'], table=table)
    
    # Test
    X_test, m_test = encode_scene(Scene(regions=[make_obj('vbar', 'green'), 
                                                  make_obj('hbar', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_posterior = softmax(l0_scores[:2] * 5.0)
    rsa_posterior = infer_posterior(X_test, m_test, ['g'], table)
    
    print(f"  L0:  green={l0_posterior[0]:.3f}, blue={l0_posterior[1]:.3f}, correct={l0_posterior[0] > l0_posterior[1]}")
    print(f"  RSA: green={rsa_posterior[0]:.3f}, blue={rsa_posterior[1]:.3f}, correct={rsa_posterior[0] > rsa_posterior[1]}")
    
    results['test4'] = {
        'l0': (l0_posterior[0], l0_posterior[1], l0_posterior[0] > l0_posterior[1]),
        'rsa': (rsa_posterior[0], rsa_posterior[1], rsa_posterior[0] > rsa_posterior[1])
    }
    
    return results


def main():
    print("="*60)
    print("DIMENSION EXPERIMENT: RSA vs L0 at d=12 and d=15")
    print("="*60)
    
    # Run at d=12
    results_12 = run_experiment(d=12)
    
    # Run at d=15
    results_15 = run_experiment(d=15)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    print("\n| Test | Metric | d=12 | d=15 |")
    print("|------|--------|------|------|")
    
    for test_name in ['test1', 'test2', 'test3', 'test4']:
        if test_name == 'test3':
            # Special handling for test3
            print(f"| {test_name} | Limited correct | {results_12[test_name]['limited'][2]} | {results_15[test_name]['limited'][2]} |")
            print(f"| {test_name} | Diverse correct | {results_12[test_name]['diverse'][2]} | {results_15[test_name]['diverse'][2]} |")
        else:
            print(f"| {test_name} | L0 correct | {results_12[test_name]['l0'][2]} | {results_15[test_name]['l0'][2]} |")
            print(f"| {test_name} | RSA correct | {results_12[test_name]['rsa'][2]} | {results_15[test_name]['rsa'][2]} |")
    
    print("\n" + "="*60)
    print("DETAILED PROBABILITIES")
    print("="*60)
    
    for test_name in ['test1', 'test2', 'test4']:
        print(f"\n{test_name}:")
        print(f"  d=12: L0 P(target)={results_12[test_name]['l0'][0]:.4f}, RSA P(target)={results_12[test_name]['rsa'][0]:.4f}")
        print(f"  d=15: L0 P(target)={results_15[test_name]['l0'][0]:.4f}, RSA P(target)={results_15[test_name]['rsa'][0]:.4f}")


if __name__ == "__main__":
    main()
