"""Test 7: Multi-Intent Overlap Detection

Scene:
  obj0: blue box
  obj1: red solid
  obj2: blue solid  <- Overlaps: both blue AND solid!
  obj3: green L

Utterance: "1 blue, 1 solid"
intents = [(['blue'], 1), (['solid'], 1)]

Expected:
- Non-overlapping assignments like {obj0, obj1} preferred
- Overlapping assignment using obj2 twice should be penalized
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import listener_L1_set, normalize_tokens

def make_obj(shape_key, color_key):
    occ = np.array(SHAPES[shape_key], dtype=np.float32)
    rgb = COLORS_RGB[color_key]
    return Obj(shape_name=shape_key, color_rgb=rgb, occ=occ)

print("=" * 70)
print("TEST 7: Multi-Intent Overlap Detection")
print("=" * 70)
print()

# Setup concepts
table = ConceptTable(d=12)

# Train 'blue' on diverse blue objects
for shape in ['box', 'solid', 'l', 't']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'blue'), None, None, None]))
    learn_step(X, m, k=1, tokens=['blue'], table=table)

# Train 'solid' on diverse solid objects  
for color in ['blue', 'red', 'green', 'yellow']:
    X, m = encode_scene(Scene(regions=[make_obj('solid', color), None, None, None]))
    learn_step(X, m, k=1, tokens=['solid'], table=table)

# Train 'red' and 'green' for completeness
for shape in ['box', 'solid', 'l']:
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'red'), None, None, None]))
    learn_step(X, m, k=1, tokens=['red'], table=table)
    X, m = encode_scene(Scene(regions=[make_obj(shape, 'green'), None, None, None]))
    learn_step(X, m, k=1, tokens=['green'], table=table)

# Test scene
obj0 = make_obj('box', 'blue')    # blue box
obj1 = make_obj('solid', 'red')   # red solid
obj2 = make_obj('solid', 'blue')  # blue solid (OVERLAPS!)
obj3 = make_obj('l', 'green')     # green L

scene = Scene(regions=[obj0, obj1, obj2, obj3])
X, mask = encode_scene(scene)

print("Scene:")
print("  obj0: blue box")
print("  obj1: red solid")
print("  obj2: blue solid (overlaps: both blue AND solid)")
print("  obj3: green L")
print()

# Test 1: Query "blue" with k=1
print("-" * 50)
print("Query: 'blue' (k=1)")
U_blue = normalize_tokens(['blue'])
posterior_blue = listener_L1_set(X, mask, U_blue, table, k=1)
print("P(T | 'blue'):")
for T, p in sorted(posterior_blue.items(), key=lambda x: -x[1]):
    labels = {0: "blue_box", 1: "red_solid", 2: "blue_solid", 3: "green_L"}
    print(f"  {labels[T[0]]:12s}: {p:.4f}")
print()

# Test 2: Query "solid" with k=1
print("-" * 50)
print("Query: 'solid' (k=1)")
U_solid = normalize_tokens(['solid'])
posterior_solid = listener_L1_set(X, mask, U_solid, table, k=1)
print("P(T | 'solid'):")
for T, p in sorted(posterior_solid.items(), key=lambda x: -x[1]):
    labels = {0: "blue_box", 1: "red_solid", 2: "blue_solid", 3: "green_L"}
    print(f"  {labels[T[0]]:12s}: {p:.4f}")
print()

# Test 3: Query "blue" with k=2 (want 2 blue objects)
print("-" * 50)
print("Query: 'blue' (k=2) - want 2 blue objects")
posterior_blue_2 = listener_L1_set(X, mask, U_blue, table, k=2)
print("P(T | 'blue', k=2):")
labels = {0: "blue_box", 1: "red_solid", 2: "blue_solid", 3: "green_L"}
for T, p in sorted(posterior_blue_2.items(), key=lambda x: -x[1]):
    t_names = "+".join(labels[i] for i in T)
    print(f"  {t_names:30s}: {p:.4f}")
print()
print("Expected: {blue_box, blue_solid} should have highest P")
print()

# Multi-Intent simulation (manual combination)
print("=" * 50)
print("MULTI-INTENT: '1 blue, 1 solid'")
print("=" * 50)
print()
print("This requires choosing ONE blue object and ONE solid object.")
print("Possible assignments:")
print("  Option A: blue_box(0) + red_solid(1)    - non-overlapping, GOOD")
print("  Option B: blue_box(0) + blue_solid(2)   - blue_solid counted for solid")  
print("  Option C: blue_solid(2) + red_solid(1)  - blue_solid counted for blue")
print("  Option D: blue_solid(2) + blue_solid(2) - OVERLAP, blue_solid for BOTH!")
print()

# Compute joint scores manually
# P(assignment) proportional to P(obj_blue | 'blue') * P(obj_solid | 'solid')

p_blue = {T[0]: p for T, p in posterior_blue.items()}
p_solid = {T[0]: p for T, p in posterior_solid.items()}

# Non-overlapping assignments
assignments = [
    ((0, 1), "blue_box + red_solid"),
    ((0, 2), "blue_box + blue_solid"),
    ((2, 1), "blue_solid + red_solid"), 
    ((2, 2), "blue_solid + blue_solid (OVERLAP!)")
]

print("Joint probability (naive product):")
scores = []
for (i_blue, i_solid), name in assignments:
    if i_blue == i_solid:
        # Overlap penalty: can't use same object for both
        score = p_blue[i_blue] * p_solid[i_solid] * 0.1  # 10x penalty
    else:
        score = p_blue[i_blue] * p_solid[i_solid]
    scores.append(score)
    penalty_note = " (penalized)" if i_blue == i_solid else ""
    print(f"  {name:35s}: {score:.6f}{penalty_note}")

# Normalize
total = sum(scores)
print()
print("Normalized probabilities:")
for (score, ((i_blue, i_solid), name)) in zip(scores, assignments):
    print(f"  {name:35s}: {score/total:.4f}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
best_idx = np.argmax(scores)
print(f"Best assignment: {assignments[best_idx][1]}")
if assignments[best_idx][0][0] != assignments[best_idx][0][1]:
    print("Non-overlapping assignment preferred!")
else:
    print("WARNING: Overlapping assignment chosen!")
