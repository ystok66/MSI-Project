"""Debug why ME effect is not working."""
import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import infer_posterior

def make_obj(s, c):
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], occ=np.array(SHAPES[s], dtype=np.float32))

print("Train: green box, green solid -> 'g'")
table = ConceptTable(d=12)
X1, m1 = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
learn_step(X1, m1, k=1, tokens=['g'], table=table)
X2, m2 = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
learn_step(X2, m2, k=1, tokens=['g'], table=table)

print("Known concepts in table:", list(table._concepts.keys()))

print()
print("Test: green_T vs red_T, query 'red'")
X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), make_obj('t', 'red'), None, None]))

# Get debug info
result = infer_posterior(X_test, m_test, ['red'], table, return_debug=True)

print()
print("Debug info:")
print("  Alternatives:", result["alternatives"])
print("  L0 scores:", result["l0_scores"])
print("  S1 scores:", result["s1_scores"])
print("  Posterior:", result["posterior"])

print()
print("Expected ME reasoning:")
print("  - Speaker said 'red' instead of 'g'")
print("  - If speaker meant green_T, they would say 'g'")
print("  - Therefore 'red' -> red_T (not green_T)")
print()
print("RSA should give P(red_T) >> P(green_T)")
