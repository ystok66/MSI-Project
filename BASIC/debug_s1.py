"""Debug S1 calculation."""
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

table = ConceptTable(d=12)
X1, m1 = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
learn_step(X1, m1, k=1, tokens=['g'], table=table)
X2, m2 = encode_scene(Scene(regions=[make_obj('solid', 'green'), None, None, None]))
learn_step(X2, m2, k=1, tokens=['g'], table=table)

X_test, m_test = encode_scene(Scene(regions=[make_obj('box', 'green'), make_obj('box', 'red'), None, None]))

dbg = infer_posterior(X_test, m_test, ['red'], table, return_debug=True)
print("alts:", dbg["alts"])
print()
print("L0 scores:")
for alt, scores in dbg["L0"].items():
    alt_str = str(alt) if alt else "(empty)"
    print(f"  {alt_str:10s}: green_box={scores[0]:8.2f}, red_box={scores[1]:8.2f}")

print()
print("S1 (speaker prob):")
print(f"  S1(red|green_box) = {dbg['S1'][0]:.10f}")
print(f"  S1(red|red_box)   = {dbg['S1'][1]:.10f}")
print()
print("Posterior:")
print(f"  P(green_box|red) = {dbg['p'][0]:.6f}")
print(f"  P(red_box|red)   = {dbg['p'][1]:.6f}")
