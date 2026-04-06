# Dimension Experiment Results

Comparing RSA vs L0 performance at d=12 (original) vs d=15 (extended) concept dimensions.

---

## Experiment Setup

| Dimension | Object Embedding      | Concept Representation      |
| --------- | --------------------- | --------------------------- |
| d=12      | 3 Lab color + 9 shape | Same 12D Gaussian           |
| d=15      | 12D + 3 zeros padding | 15D Gaussian (3 extra dims) |

---

## Test Results

### Test 1: Generalization (green box/solid → green L)

Training: 2 examples (green box, green solid) with token 'g'
Test: green L vs blue box, query 'g'

| Dimension | L0 P(correct) | RSA P(correct) |
| --------- | ------------- | -------------- |
| d=12      | 0.9928        | 0.9928         |
| d=15      | 0.9928        | 0.9928         |

**Result**: ✅ Both correct, identical performance

---

### Test 2: Multi-concept (green vs blue)

Training: 4 shapes × 2 colors (green→'g', blue→'b')
Test: green vbar vs blue hbar, query 'g'

| Dimension | L0 P(correct) | RSA P(correct) |
| --------- | ------------- | -------------- |
| d=12      | 1.0000        | 1.0000         |
| d=15      | 1.0000        | 1.0000         |

**Result**: ✅ Both correct, identical performance

---

### Test 3: Hard Case (1 training example, very different test shape)

Training: 1 example (green box only) with token 'g'
Test: green T vs red box, query 'g'

| Dimension | L0 P(correct) | RSA P(correct) | Correct? |
| --------- | ------------- | -------------- | -------- |
| d=12      | 0.0003        | 0.0003         | ❌       |
| d=15      | 0.0003        | 0.0003         | ❌       |

**Result**: ❌ Both WRONG! The model prefers red box over green T!

---

## Key Findings

### 1. Dimension Extension Has No Effect

Adding 3 zero-padded dimensions (d=12 → d=15) produces **identical results**:

- The extra dimensions contribute nothing to KL divergence
- Zero features have zero difference from zero-initialized concept means
- No information gain from padding

### 2. RSA = L0 in These Cases

In all tests, RSA and L0 gave the same probabilities. This is because:

- With diverse training relative to test, L0 is already decisive
- RSA amplifies differences, but when L0 is already 0.99+, amplification doesn't change the answer

### 3. Single-Example Training Fails

The hard case reveals a fundamental limitation:

- Training only on "green box" creates a concept centered on box shape
- Green T has very different shape (6 of 9 cells differ)
- Red box has same shape as training
- Shape mismatch (6 dims) >> Color mismatch (3 dims)

**This is the "Dimensional Signal Dominance" problem documented in DEVLOG.md.**

---

## Why Padding Doesn't Help

```
Extra dimensions = [0, 0, 0]

Object embedding (padded):
  green_T: [Lab, shape, 0, 0, 0]
  red_box: [Lab, shape, 0, 0, 0]

Concept prior (padded):
  'g': [zeros(12), zeros(3)] with var = ones(15)

KL contribution from extra dims:
  = (0 - 0)² / 1.0 = 0 for all objects

No discrimination signal!
```

For higher dimensions to help, they would need to encode **meaningful features** that distinguish concepts.

---

## Conclusion

1. **Simply padding dimensions doesn't improve performance**
2. **Need more training examples to build robust concepts**
3. **RSA doesn't rescue poor L0** - it amplifies L0, so if L0 is wrong, RSA is wrong
4. **Diverse training is critical** for color concepts to generalize across shapes
