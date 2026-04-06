"""Specific test requested by user."""
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

def l0_posterior(X, mask, tokens, table, alpha=5.0):
    l0_scores = score_L0(X, mask, tuple(t.lower().strip() for t in tokens), table)
    valid = mask
    valid_scores = l0_scores[valid]
    probs = np.zeros(4)
    probs[valid] = softmax(valid_scores * alpha)
    return probs

print("Step 1: Train")
print("  - green box -> 'g'")
print("  - green solid -> 'g'")

table = ConceptTable(d=12)

X1, m1 = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
learn_step(X1, m1, k=1, tokens=['g'], table=table)

X2, m2 = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
learn_step(X2, m2, k=1, tokens=['g'], table=table)

print()
print("Step 2: Test Scene")
print("  Region 0: green T")
print("  Region 1: red L")

X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), 
                                              make_obj('l', 'red'), 
                                              None, None]))

print()
print("Step 3: Query '1 red'")
print()

l0 = l0_posterior(X_test, m_test, ['red'], table)
rsa = infer_posterior(X_test, m_test, ['red'], table)

print("Results:")
print(f"  L0:  P(green_T)={l0[0]:.4f}, P(red_L)={l0[1]:.4f}")
print(f"  RSA: P(green_T)={rsa[0]:.4f}, P(red_L)={rsa[1]:.4f}")
print()
print(f"L0 prefers red_L: {l0[1] > l0[0]}")
print(f"RSA prefers red_L: {rsa[1] > rsa[0]}")
