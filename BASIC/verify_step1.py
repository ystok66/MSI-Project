"""Quick verification script for Step 1."""
import numpy as np
from world import parse_utterance, sample_scene
from encoders import encode_scene

print("=== parse_utterance tests ===")
result1 = parse_utterance("1 blue box")
print(f'"1 blue box" -> {result1}')
assert result1 == (1, ["blue", "box"]), f"Expected (1, ['blue', 'box']), got {result1}"

result2 = parse_utterance("blue box")
print(f'"blue box" -> {result2}')
assert result2 == (1, ["blue", "box"]), f"Expected (1, ['blue', 'box']), got {result2}"

result3 = parse_utterance("2 TV green")
print(f'"2 TV green" -> {result3}')
assert result3 == (2, ["tv", "green"]), f"Expected (2, ['tv', 'green']), got {result3}"

print("\n=== Scene generation test ===")
rng = np.random.default_rng(42)
scene = sample_scene(rng, p_empty=0.3)
print(f"Scene regions: {[r.shape_name if r else None for r in scene.regions]}")

print("\n=== Encoding test ===")
X, mask = encode_scene(scene)
print(f"X.shape: {X.shape}")
print(f"mask: {mask}")
print(f"X[0,:3] (Lab normalized): {X[0,:3]}")
print(f"X[0,3:] (shape binary): {X[0,3:]}")

# Verify shapes
assert X.shape == (4, 12), f"Expected (4, 12), got {X.shape}"
assert mask.shape == (4,), f"Expected (4,), got {mask.shape}"

# Verify Lab values in range
for t in range(4):
    if mask[t]:
        assert np.all(X[t, :3] >= -1.5) and np.all(X[t, :3] <= 1.5), \
            f"Lab values out of range at t={t}"
        assert np.all((X[t, 3:] == 0) | (X[t, 3:] == 1)), \
            f"Shape values not binary at t={t}"

print("\n=== Reproducibility test ===")
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
scene1 = sample_scene(rng1, p_empty=0.3)
scene2 = sample_scene(rng2, p_empty=0.3)
X1, mask1 = encode_scene(scene1)
X2, mask2 = encode_scene(scene2)
assert np.allclose(X1, X2), "Reproducibility failed for X"
assert np.array_equal(mask1, mask2), "Reproducibility failed for mask"
print("Same seed produces identical results: PASS")

print("\n" + "="*50)
print("ALL VERIFICATION PASSED!")
print("="*50)
