"""
Reproduce the Multi-Intent RSA bug:
  L0 gives correct answer (95.4% blue_vbar+red_hbar)
  RSA gives wrong answer (43.2% to both correct and incorrect)
"""
import numpy as np
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step
from rsa import infer_posterior_multi_intent, score_L0
from scoring import log_inc_single

# ============================================================
# 1. Train concepts
# ============================================================
table = ConceptTable(d=12)

def make_obj(color_name, shape_name):
    return Obj(
        shape_name=shape_name,
        color_rgb=COLORS_RGB[color_name],
        occ=np.array(SHAPES[shape_name], dtype=np.int32)
    )

def make_scene_from_descs(descs):
    """descs: list of (color, shape) or None"""
    regions = []
    for desc in descs:
        if desc is None:
            regions.append(None)
        else:
            regions.append(make_obj(desc[0], desc[1]))
    while len(regions) < 4:
        regions.append(None)
    return Scene(regions=regions)

# Train "blue"
for shape in ['vbar', 'hbar', 't', 'l']:
    scene = make_scene_from_descs([('blue', shape)])
    X, mask = encode_scene(scene)
    learn_step(X, mask, k=1, tokens=['blue'], table=table)

# Train "hbar"
for color in ['blue', 'red', 'green', 'yellow']:
    scene = make_scene_from_descs([(color, 'hbar')])
    X, mask = encode_scene(scene)
    learn_step(X, mask, k=1, tokens=['hbar'], table=table)

print("=== Trained Concepts ===")
for token in table.tokens():
    c = table.get(token)
    print(f"  {token}: kappa={c.kappa:.1f}")

# ============================================================
# 2. Test scene
# ============================================================
test_scene = Scene(regions=[
    make_obj('blue', 'vbar'),   # 0: blue_vbar
    make_obj('red', 'hbar'),    # 1: red_hbar
    make_obj('blue', 'hbar'),   # 2: blue_hbar (overlap!)
    make_obj('red', 'vbar'),    # 3: red_vbar
])
X, mask = encode_scene(test_scene)

print("\n=== Test Scene ===")
labels = ['blue_vbar', 'red_hbar', 'blue_hbar', 'red_vbar']
for i, label in enumerate(labels):
    print(f"  Region {i}: {label}")

# ============================================================
# 3. Per-object L0 scores
# ============================================================
print("\n=== Per-Object L0 Scores ===")
print(f"{'Object':<12} {'L0(blue)':>10} {'L0(hbar)':>10}")
for i, label in enumerate(labels):
    c_blue = table.get('blue')
    c_hbar = table.get('hbar')
    s_blue = log_inc_single(X[i], c_blue.mu, c_blue.var)
    s_hbar = log_inc_single(X[i], c_hbar.mu, c_hbar.var)
    print(f"  {label:<12} {s_blue:>10.2f} {s_hbar:>10.2f}")

# ============================================================
# 4. Multi-Intent: L0-only mode
# ============================================================
intents = [(['blue'], 1), (['hbar'], 1)]

print("\n=== Multi-Intent L0 Only (use_rsa=False) ===")
result_l0 = infer_posterior_multi_intent(
    X, mask, intents, table,
    alpha=5.0, use_rsa=False, return_debug=True
)
assignments = result_l0['assignments']
posterior_l0 = result_l0['posterior']

for a in sorted(posterior_l0.keys(), key=lambda x: posterior_l0[x], reverse=True):
    blue_obj = labels[a[0][0]]
    hbar_obj = labels[a[1][0]]
    p = posterior_l0[a]
    if p > 0.001:
        print(f"  {blue_obj:>12} + {hbar_obj:<12} = {p:.4f} ({p*100:.1f}%)")

# ============================================================
# 5. Multi-Intent: Full RSA (use_rsa=True)
# ============================================================
print("\n=== Multi-Intent RSA (use_rsa=True) ===")
result_rsa = infer_posterior_multi_intent(
    X, mask, intents, table,
    alpha=5.0, use_rsa=True, return_debug=True
)
posterior_rsa = result_rsa['posterior']
s1_scores = result_rsa.get('s1_scores', {})

print("  Posterior:")
for a in sorted(posterior_rsa.keys(), key=lambda x: posterior_rsa[x], reverse=True):
    blue_obj = labels[a[0][0]]
    hbar_obj = labels[a[1][0]]
    p = posterior_rsa[a]
    if p > 0.001:
        print(f"    {blue_obj:>12} + {hbar_obj:<12} = {p:.4f} ({p*100:.1f}%)")

if s1_scores:
    print("\n  S1 Scores:")
    for a in sorted(s1_scores.keys(), key=lambda x: s1_scores[x], reverse=True):
        blue_obj = labels[a[0][0]]
        hbar_obj = labels[a[1][0]]
        s1 = s1_scores[a]
        if s1 > 1e-6:
            print(f"    {blue_obj:>12} + {hbar_obj:<12} = S1={s1:.6f}")

# Also show alternatives used
if 'alternatives' in result_rsa:
    print(f"\n  Alternatives ({len(result_rsa['alternatives'])} total):")
    for alt in result_rsa['alternatives']:
        desc = ', '.join(f"{cnt} {'+'.join(toks)}" for toks, cnt in alt)
        print(f"    {desc}")

print("\n=== COMPARISON ===")
print(f"{'Assignment':<30} {'L0':>8} {'RSA':>8} {'S1':>10}")
for a in sorted(posterior_l0.keys(), key=lambda x: posterior_l0[x], reverse=True):
    blue_obj = labels[a[0][0]]
    hbar_obj = labels[a[1][0]]
    p_l0 = posterior_l0.get(a, 0)
    p_rsa = posterior_rsa.get(a, 0)
    s1 = s1_scores.get(a, 0)
    if p_l0 > 0.001 or p_rsa > 0.001:
        name = f"{blue_obj}+{hbar_obj}"
        print(f"  {name:<30} {p_l0*100:>7.1f}% {p_rsa*100:>7.1f}% {s1:>10.6f}")
