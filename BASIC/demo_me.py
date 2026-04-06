"""Final ME effect demonstration."""
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
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], occ=np.array(SHAPES[s], dtype=np.float32))

def l0_posterior(X, mask, tokens, table, alpha=5.0):
    l0_scores = score_L0(X, mask, tuple(t.lower().strip() for t in tokens), table)
    valid_scores = l0_scores[mask]
    probs = np.zeros(4)
    probs[mask] = softmax(valid_scores * alpha)
    return probs

print("=" * 60)
print("MUTUAL EXCLUSIVITY EFFECT DEMONSTRATION")
print("=" * 60)

# Setup
table = ConceptTable(d=12)
X1, m1 = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
learn_step(X1, m1, k=1, tokens=['g'], table=table)
X2, m2 = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
learn_step(X2, m2, k=1, tokens=['g'], table=table)

print("Training: green box and green solid labeled as 'g'")
print()

# Test scene
X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), make_obj('t', 'red'), None, None]))
print("Test scene: green T (slot 0) vs red T (slot 1)")
print("Query: 'red' (novel word)")
print()

l0 = l0_posterior(X_test, m_test, ['red'], table)
rsa = infer_posterior(X_test, m_test, ['red'], table)

print("Results:")
print(f"  L0:  P(green_T)={l0[0]:.4f}, P(red_T)={l0[1]:.4f}")
print(f"  RSA: P(green_T)={rsa[0]:.4f}, P(red_T)={rsa[1]:.4f}")
print()
print("ME Effect Analysis:")
print(f"  L0 difference:  {l0[1] - l0[0]:+.4f}")
print(f"  RSA difference: {rsa[1] - rsa[0]:+.4f}")
print()
if rsa[1] > rsa[0] + 0.3:
    print("RSA shows strong ME effect: novel word 'red' strongly prefers red T!")
elif rsa[1] > rsa[0]:
    print("RSA shows ME effect: novel word 'red' prefers red T")
else:
    print("ME effect not observed")
