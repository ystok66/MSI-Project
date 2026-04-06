"""Debug sequential learning ME test."""
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
obj_blue_l = make_obj('l', 'blue')
scene_train = Scene(regions=[obj_blue_l, None, None, None])
X_train, mask_train = encode_scene(scene_train)

for _ in range(10):
    learn_step(X_train, mask_train, k=1, tokens=['blue'], table=table)

obj_blue_box = make_obj('box', 'blue')
obj_red_t = make_obj('t', 'red')
scene_test = Scene(regions=[obj_blue_box, obj_red_t, None, None])
X_test, mask_test = encode_scene(scene_test)

# Original test call
p = infer_posterior(X_test, mask_test, ['t'], table, include_empty_alt=True, alt_extra_tokens=['blue'])
print(f"Original (include_empty_alt=True): P(blue_box)={p[0]:.4f}, P(red_t)={p[1]:.4f}")

# Without explicit include_empty_alt (uses dynamic)
p2 = infer_posterior(X_test, mask_test, ['t'], table, alt_extra_tokens=['blue'])
print(f"Dynamic (no explicit):             P(blue_box)={p2[0]:.4f}, P(red_t)={p2[1]:.4f}")

# Also check with default
p3 = infer_posterior(X_test, mask_test, ['t'], table)
print(f"Full default:                      P(blue_box)={p3[0]:.4f}, P(red_t)={p3[1]:.4f}")

print()
print(f"Known tokens: {list(table._concepts.keys())}")
print(f"Is 't' novel? {'t' not in table._concepts}")
