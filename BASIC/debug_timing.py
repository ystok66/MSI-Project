"""Debug the timing of novel token detection."""
import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
import rsa

# Monkey-patch speaker_S1 to add debug output
original_speaker_S1 = rsa.speaker_S1

def debug_speaker_S1(X, mask, U, table, **kwargs):
    known_tokens = list(table._concepts.keys())
    U_tokens = set(U)
    novel_tokens = U_tokens - set(known_tokens)
    print(f"[speaker_S1] U={U}, known={known_tokens}, novel={novel_tokens}")
    print(f"[speaker_S1] Condition: len(known)={len(known_tokens)} >= 1 and len(novel)={len(novel_tokens)} > 0 => {len(known_tokens) >= 1 and len(novel_tokens) > 0}")
    return original_speaker_S1(X, mask, U, table, **kwargs)

rsa.speaker_S1 = debug_speaker_S1

def make_obj(shape_key, color_key):
    occ = np.array(SHAPES[shape_key], dtype=np.float32)
    rgb = COLORS_RGB[color_key]
    return Obj(shape_name=shape_key, color_rgb=rgb, occ=occ)

table = ConceptTable(d=12)

# Train on diverse blue shapes
for shape in ['l', 'box', 'solid']:
    obj_blue = make_obj(shape, 'blue')
    scene_train = Scene(regions=[obj_blue, None, None, None])
    X_train, mask_train = encode_scene(scene_train)
    for _ in range(5):
        learn_step(X_train, mask_train, k=1, tokens=['blue'], table=table)

obj_blue_vbar = make_obj('vbar', 'blue')
obj_red_t = make_obj('t', 'red')
scene_test = Scene(regions=[obj_blue_vbar, obj_red_t, None, None])
X_test, mask_test = encode_scene(scene_test)

print('Before infer_posterior:', list(table._concepts.keys()))
p = rsa.infer_posterior(X_test, mask_test, ['t'], table)
print('After infer_posterior:', list(table._concepts.keys()))
print(f'p: [{p[0]:.6f}, {p[1]:.6f}]')
