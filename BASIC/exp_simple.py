"""Simple dimension experiment - cleaner output."""
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


def test_generalization(d):
    """Test 1: green generalizes to novel shape."""
    table = ConceptTable(d=d)
    
    # Training: green box + green solid
    for shape in ['box', 'solid']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
    
    # Test: green L vs blue box
    X_test, m_test = encode_scene(Scene(regions=[make_obj('l', 'green'), 
                                                  make_obj('box', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    # L0
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_p = softmax(l0_scores[:2] * 5.0)[0]
    
    # RSA
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    
    return l0_p, rsa_p


def test_multi_concept(d):
    """Test 4: discriminate green vs blue with competing concepts."""
    table = ConceptTable(d=d)
    
    # Train both concepts
    for shape in ['box', 'solid', 'l', 't']:
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['g'], table=table)
        
        X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
        X = pad_features(X, d)
        learn_step(X, m, k=1, tokens=['b'], table=table)
    
    # Test: green vbar vs blue hbar, query 'g'
    X_test, m_test = encode_scene(Scene(regions=[make_obj('vbar', 'green'), 
                                                  make_obj('hbar', 'blue'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_p = softmax(l0_scores[:2] * 5.0)[0]
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    
    return l0_p, rsa_p


def test_hard_case(d):
    """Hard case: only 2 training examples, test on very different shape."""
    table = ConceptTable(d=d)
    
    # Minimal training
    X, m = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
    X = pad_features(X, d)
    learn_step(X, m, k=1, tokens=['g'], table=table)
    
    # Test: green T vs red box (T is quite different from box)
    X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), 
                                                  make_obj('box', 'red'), None, None]))
    X_test = pad_features(X_test, d)
    
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_p = softmax(l0_scores[:2] * 5.0)[0]
    rsa_p = infer_posterior(X_test, m_test, ['g'], table)[0]
    
    return l0_p, rsa_p


print("DIMENSION EXPERIMENT: d=12 vs d=15")
print("="*50)

print("\nTest 1: Generalization (green box/solid -> green L)")
l0_12, rsa_12 = test_generalization(12)
l0_15, rsa_15 = test_generalization(15)
print(f"  d=12: L0={l0_12:.4f}, RSA={rsa_12:.4f}")
print(f"  d=15: L0={l0_15:.4f}, RSA={rsa_15:.4f}")

print("\nTest 2: Multi-concept (green vs blue)")
l0_12, rsa_12 = test_multi_concept(12)
l0_15, rsa_15 = test_multi_concept(15)
print(f"  d=12: L0={l0_12:.4f}, RSA={rsa_12:.4f}")
print(f"  d=15: L0={l0_15:.4f}, RSA={rsa_15:.4f}")

print("\nTest 3: Hard case (1 training example, very different test shape)")
l0_12, rsa_12 = test_hard_case(12)
l0_15, rsa_15 = test_hard_case(15)
print(f"  d=12: L0={l0_12:.4f}, RSA={rsa_12:.4f}, correct={l0_12 > 0.5}")
print(f"  d=15: L0={l0_15:.4f}, RSA={rsa_15:.4f}, correct={l0_15 > 0.5}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("| Test | Metric | d=12 | d=15 |")
print("|------|--------|------|------|")
