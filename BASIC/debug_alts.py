"""Debug the alts being built."""
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

def debug_speaker_S1(X, mask, U, table, *, alpha=5.0, beta=0.1, lam=0.0,
                     include_empty_alt=True, alt_extra_tokens=None, 
                     auto_alt_from_table=True, eps_obj=1e-4, tau=1.0, min_var=1e-8):
    known_tokens = list(table._concepts.keys())
    U_tokens = set(U)
    novel_tokens = U_tokens - set(known_tokens)
    
    # Check condition
    should_disable_empty = len(known_tokens) >= 1 and len(novel_tokens) > 0
    
    print(f"[S1] U={U}, known={known_tokens}, novel={novel_tokens}")
    print(f"[S1] include_empty_alt param = {include_empty_alt}")
    print(f"[S1] auto_alt_from_table = {auto_alt_from_table}")
    print(f"[S1] Should disable empty? {should_disable_empty}")
    
    if auto_alt_from_table and should_disable_empty:
        print(f"[S1] Setting include_empty_alt = False")
    
    # Call original
    result = original_speaker_S1(X, mask, U, table,
                                  alpha=alpha, beta=beta, lam=lam,
                                  include_empty_alt=include_empty_alt,
                                  alt_extra_tokens=alt_extra_tokens,
                                  auto_alt_from_table=auto_alt_from_table,
                                  eps_obj=eps_obj, tau=tau, min_var=min_var)
    print(f"[S1] Result: {result[:2]}")
    return result

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
print('=' * 60)
p = rsa.infer_posterior(X_test, mask_test, ['t'], table)
print('=' * 60)
print('After infer_posterior:', list(table._concepts.keys()))
print(f'p: [{p[0]:.6f}, {p[1]:.6f}]')
