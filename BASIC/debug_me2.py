"""Debug ME effect in detail."""
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

X_test, m_test = encode_scene(Scene(regions=[make_obj('t', 'green'), make_obj('t', 'red'), None, None]))
result = infer_posterior(X_test, m_test, ['red'], table, return_debug=True)

print("Alternatives:", result['alts'])
print()
print("L0 scores for each alternative (higher = better match):")
for alt, scores in result['L0'].items():
    if len(alt) > 0:
        print(f"  {str(alt):15s}: green_T={scores[0]:8.2f}, red_T={scores[1]:8.2f}")
    else:
        print(f"  {'(empty)':15s}: green_T={scores[0]:8.2f}, red_T={scores[1]:8.2f}")

print()
print("S1 scores (probability of speaker saying 'red' given target):")
print(f"  S1(red | green_T) = {result['S1'][0]:.6f}")
print(f"  S1(red | red_T)   = {result['S1'][1]:.6f}")

print()
print("Posterior:")
print(f"  P(green_T | red) = {result['p'][0]:.4f}")
print(f"  P(red_T | red)   = {result['p'][1]:.4f}")

print()
print("="*60)
print("ANALYSIS")
print("="*60)
print()
print("For ME to work:")
print("  S1(red | green_T) should be LOW because:")
print("    - L0(green_T, 'g') is high (green_T matches 'g')")
print("    - Speaker would prefer to say 'g' for green_T")
print()
print("  S1(red | red_T) should be HIGHER because:")
print("    - L0(red_T, 'g') is low (red_T does not match 'g')")
print("    - Speaker cannot use 'g' for red_T")
print()

# Let's check L0 for 'g' alternative
print("L0 for 'g' alternative:")
g_scores = result['L0'][('g',)]
print(f"  L0(green_T, 'g') = {g_scores[0]:.2f}")
print(f"  L0(red_T, 'g')   = {g_scores[1]:.2f}")
