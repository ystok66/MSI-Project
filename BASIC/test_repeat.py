"""Test: Effect of repeating same training example."""
import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import score_L0
from scipy.special import softmax


def make_obj(s, c):
    return Obj(shape_name=s, color_rgb=COLORS_RGB[c], 
               occ=np.array(SHAPES[s], dtype=np.float32))


print("Effect of repeating same training example (green box)")
print("Test: green T vs red box, query 'g'")
print("="*60)
print()

for n in [1, 2, 5, 10, 20, 50, 100]:
    table = ConceptTable(d=12)
    
    X_train, m_train = encode_scene(Scene(regions=[make_obj('box', 'green'), None, None, None]))
    for _ in range(n):
        learn_step(X_train, m_train, k=1, tokens=['g'], table=table)
    
    g = table.get('g')
    
    X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), 
                                                  make_obj('box', 'red'), None, None]))
    
    l0_scores = score_L0(X_test, m_test, ('g',), table)
    l0_p = softmax(l0_scores[:2] * 5.0)[0]
    
    print(f"n={n:3d}: kappa={g.kappa:5.1f}, var_color={g.var[:3].mean():.6f}, var_shape={g.var[3:].mean():.6f}, P(green_T)={l0_p:.2e}, correct={l0_p > 0.5}")

print()
print("Conclusion: Repeating same example makes it WORSE!")
print("Variance collapses -> KL becomes extremely sensitive to shape differences")
