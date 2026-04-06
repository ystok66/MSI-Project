"""
Full TEST_CASES Experiment: RSA vs L0 at d=12 and d=15

This script implements all 8 test cases from TEST_CASES.md and compares:
1. RSA (L1) vs L0-only performance  
2. d=12 (original) vs d=15 (extended) concept dimensions
"""

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


def make_obj(s, c):
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], 
               occ=np.array(SHAPES[s], dtype=np.float32))


def pad_features(X, target_dim):
    """Pad feature matrix with zeros to target dimension."""
    if X.shape[1] >= target_dim:
        return X
    padding = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=X.dtype)
    return np.concatenate([X, padding], axis=1)


def l0_posterior(X, mask, tokens, table, alpha=5.0):
    """Compute L0-based posterior (no RSA)."""
    l0_scores = score_L0(X, mask, tuple(t.lower().strip() for t in tokens), table)
    valid = mask
    # Apply softmax only to valid regions
    valid_scores = l0_scores[valid]
    probs = np.zeros(4)
    probs[valid] = softmax(valid_scores * alpha)
    return probs


def run_all_tests(d=12):
    """Run all 8 test cases at given dimension."""
    results = {}
    
    # =========================================================================
    # Test 1: Single Concept Generalization
    # Train: green box, green solid -> 'g'
    # Test: green L vs blue box, query 'g'
    # =========================================================================
    table = ConceptTable(d=d)
    for shape in ['box', 'solid']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), 
                                                  make_obj('box', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_p = l0_posterior(X_test, m_test, ['g'], table)[0]
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    results['test1'] = {'name': 'Generalization', 'l0': l0_p, 'rsa': rsa_p, 
                        'l0_correct': l0_p > 0.5, 'rsa_correct': rsa_p > 0.5}
    
    # =========================================================================
    # Test 2: Token Independence (xyz = blue)
    # Train: blue objects diverse shapes -> 'xyz'
    # Test: blue L vs red box, query 'xyz'
    # =========================================================================
    table = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
        for _ in range(2):
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
            X = pad_features(X, d)
            learn_step(X, m, k=1, tokens=['xyz'], table=table)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'blue'), 
                                                  make_obj('box', 'red'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_p = l0_posterior(X_test, m_test, ['xyz'], table)[0]
    rsa_p = infer_posterior(X_test, m_test, ['xyz'], table)[0]
    results['test2'] = {'name': 'Token Independence', 'l0': l0_p, 'rsa': rsa_p,
                        'l0_correct': l0_p > 0.5, 'rsa_correct': rsa_p > 0.5}
    
    # =========================================================================
    # Test 3: Diverse Training (limited vs diverse)
    # Test: green l_90 vs red box after limited training
    # =========================================================================
    # Limited training
    table_limited = ConceptTable(d=d)
    for _ in range(5):
        for shape in ['box', 'solid']:
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
            X = pad_features(X, d)
            learn_step(X, m, k=1, tokens=['g'], table=table_limited)
    
    # Diverse training
    table_diverse = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l', 't', 'vbar', 'hbar']:
        for _ in range(2):
            X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
            X = pad_features(X, d)
            learn_step(X, m, k=1, tokens=['g'], table=table_diverse)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l_90', 'green'), 
                                                  make_obj('box', 'red'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_limited = l0_posterior(X_test, m_test, ['g'], table_limited)[0]
    rsa_limited = infer_posterior(X_test, m_test, ['g'], table_limited)[0]
    l0_diverse = l0_posterior(X_test, m_test, ['g'], table_diverse)[0]
    rsa_diverse = infer_posterior(X_test, m_test, ['g'], table_diverse)[0]
    
    results['test3_limited'] = {'name': 'Limited Train', 'l0': l0_limited, 'rsa': rsa_limited,
                                'l0_correct': l0_limited > 0.5, 'rsa_correct': rsa_limited > 0.5}
    results['test3_diverse'] = {'name': 'Diverse Train', 'l0': l0_diverse, 'rsa': rsa_diverse,
                                'l0_correct': l0_diverse > 0.5, 'rsa_correct': rsa_diverse > 0.5}
    
    # =========================================================================
    # Test 4: Empty Alternative (single concept discrimination)
    # This tests the include_empty_alt effect
    # =========================================================================
    table = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), 
                                                  make_obj('box', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_p = l0_posterior(X_test, m_test, ['g'], table)[0]
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    results['test4'] = {'name': 'Single Concept', 'l0': l0_p, 'rsa': rsa_p,
                        'l0_correct': l0_p > 0.5, 'rsa_correct': rsa_p > 0.5}
    
    # =========================================================================
    # Test 5: Multi-concept discrimination (green vs blue)
    # With auto_alt_from_table, RSA should show competition effect
    # =========================================================================
    table = ConceptTable(d=d)
    for shape in ['box', 'solid', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['b'], table=table)
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('vbar', 'green'), 
                                                  make_obj('hbar', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_p = l0_posterior(X_test, m_test, ['g'], table)[0]
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    results['test5'] = {'name': 'Multi-concept', 'l0': l0_p, 'rsa': rsa_p,
                        'l0_correct': l0_p > 0.5, 'rsa_correct': rsa_p > 0.5}
    
    # =========================================================================
    # Test 6: AND Semantics
    # Query "a b" should prefer object matching both concepts
    # =========================================================================
    table = ConceptTable(d=d)
    
    # Create concepts focused on different colors
    for _ in range(3):
        X, m = encode_scene(Scene(regions=[make_obj('box', 'red'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['a'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj('solid', 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['b'], table=table)
    
    # Test scene: obj0=red_box (a only), obj1=blue_solid (b only), obj2=purple_l (neither)
    X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'red'),      # a only
                                                  make_obj('solid', 'blue'),   # b only
                                                  make_obj('l', 'purple'),     # neither
                                                  None]))
    X_test = pad_features(X_test, d)
    
    # Query "a" - should prefer red_box
    l0_a = l0_posterior(X_test, m_test, ['a'], table)
    rsa_a = infer_posterior(X_test, m_test, ['a'], table)
    
    results['test6'] = {'name': 'AND Semantics', 
                        'l0': l0_a[0], 'rsa': rsa_a[0],  # P(red_box | a)
                        'l0_correct': l0_a[0] > l0_a[1] and l0_a[0] > l0_a[2],
                        'rsa_correct': rsa_a[0] > rsa_a[1] and rsa_a[0] > rsa_a[2]}
    
    # =========================================================================
    # Test 7: Mutual Exclusivity (novel word -> novel object)
    # Train 'a' on blue_box, then query 'b' (novel) with familiar+novel objects
    # =========================================================================
    table = ConceptTable(d=d)
    
    # Train 'a' on blue objects
    for shape in ['box', 'solid', 'l']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['a'], table=table)
    
    # Test: blue_box (familiar) vs red_solid (novel), query novel word 'b'
    X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'blue'),   # familiar
                                                  make_obj('solid', 'red'),  # novel
                                                  None, None]))
    X_test = pad_features(X_test, d)
    
    # Query 'b' - RSA with auto_alt should prefer novel object
    l0_p = l0_posterior(X_test, m_test, ['b'], table)
    rsa_p = infer_posterior(X_test, m_test, ['b'], table)
    
    # Novel word should go to novel object (index 1)
    results['test7'] = {'name': 'Mutual Exclusivity', 
                        'l0': l0_p[1], 'rsa': rsa_p[1],  # P(novel_obj | novel_word)
                        'l0_correct': l0_p[1] > l0_p[0],
                        'rsa_correct': rsa_p[1] > rsa_p[0]}
    
    return results


def print_results(results, d):
    """Print results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"RESULTS FOR d={d}")
    print('='*70)
    print(f"{'Test':<20} {'L0 P':<10} {'RSA P':<10} {'L0 OK':<8} {'RSA OK':<8} {'RSA>L0?':<8}")
    print('-'*70)
    
    for key in sorted(results.keys()):
        r = results[key]
        l0_ok = 'Yes' if r['l0_correct'] else 'No'
        rsa_ok = 'Yes' if r['rsa_correct'] else 'No'
        rsa_better = 'Yes' if r['rsa'] > r['l0'] else ('Same' if abs(r['rsa']-r['l0']) < 0.001 else 'No')
        print(f"{r['name']:<20} {r['l0']:<10.4f} {r['rsa']:<10.4f} {l0_ok:<8} {rsa_ok:<8} {rsa_better:<8}")


def main():
    print("="*70)
    print("FULL TEST_CASES EXPERIMENT: RSA vs L0 at d=12 and d=15")
    print("="*70)
    
    results_12 = run_all_tests(d=12)
    print_results(results_12, 12)
    
    results_15 = run_all_tests(d=15)
    print_results(results_15, 15)
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY: d=12 vs d=15")
    print("="*70)
    print(f"{'Test':<20} {'d=12 L0':<10} {'d=12 RSA':<10} {'d=15 L0':<10} {'d=15 RSA':<10}")
    print('-'*70)
    
    for key in sorted(results_12.keys()):
        r12 = results_12[key]
        r15 = results_15[key]
        print(f"{r12['name']:<20} {r12['l0']:<10.4f} {r12['rsa']:<10.4f} {r15['l0']:<10.4f} {r15['rsa']:<10.4f}")
    
    # Count differences
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    rsa_wins = sum(1 for k in results_12 if results_12[k]['rsa'] > results_12[k]['l0'] + 0.001)
    l0_wins = sum(1 for k in results_12 if results_12[k]['l0'] > results_12[k]['rsa'] + 0.001)
    ties = len(results_12) - rsa_wins - l0_wins
    
    print(f"At d=12: RSA better in {rsa_wins} tests, L0 better in {l0_wins} tests, ties: {ties}")
    
    diff_count = sum(1 for k in results_12 if abs(results_12[k]['rsa'] - results_15[k]['rsa']) > 0.001)
    print(f"Difference between d=12 and d=15: {diff_count} tests differ")


if __name__ == "__main__":
    main()
