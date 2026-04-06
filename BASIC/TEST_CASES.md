# Test Cases Documentation

This document explains the key test scenarios for the RSA Pragmatic Communication system, including what each test validates, expected outcomes, and the reasoning behind those expectations.

---

## Test 1: Single Concept Generalization to Novel Shapes

### Test File

`tests/test_rsa.py::TestSingleConceptGeneralization::test_green_generalizes_to_l_shape`

### Scenario

```python
# Training
green_box  → 'g'
green_solid → 'g'

# Test Scene
Region 0: green L-shape
Region 1: blue box

# Query: "1 g"
```

### Expected Result

- `P(green_l | 'g') > 0.9`
- `P(blue_box | 'g') < 0.1`

### Why This Is Expected

1. **Color matches, shape doesn't need to**: The concept 'g' was trained on two different shapes (box, solid), so the learned concept has:
   - Low variance in color dimensions (discriminative)
   - Higher variance in shape dimensions (permissive)

2. **Blue box has wrong color**: The blue box's Lab color values differ significantly from the learned green concept mean.

3. **RSA amplifies the difference**: Through S1/L1 reasoning, even small L0 differences get amplified into decisive posteriors.

---

## Test 2: Token Independence from Templates

### Scenario

```python
# Training (using arbitrary token "xyz", NOT "blue")
blue_box   → 'xyz'
blue_solid → 'xyz'
blue_l     → 'xyz'
blue_t     → 'xyz'
blue_vbar  → 'xyz'
blue_hbar  → 'xyz'

# Test Scene
Region 0: blue L-shape
Region 1: red box

# Query: "xyz"
```

### Expected Result

- `P(blue_l | 'xyz') ≈ 1.0`
- `P(red_box | 'xyz') ≈ 0.0`

### Why This Is Expected

**Key point: There is NO text matching between tokens and COLORS_RGB!**

The code path is:

```
Token "xyz"
  → table.ensure("xyz")
  → Looks up/creates Concept in ConceptTable
  → NEVER accesses COLORS_RGB["xyz"]
```

The token "xyz" learns the blue color purely through visual features during training, not through any string matching to template colors. This proves that:

- **Human perception** (COLORS_RGB) is separated from **learned concepts** (ConceptTable)
- The system can learn arbitrary word-to-meaning mappings

---

## Test 3: Diverse Training Prevents Dimensional Dominance

### Problem (Without Fix)

```python
# Training only on box + solid (similar shapes)
green_box   → 'g'
green_solid → 'g'

# Result: shape variance = 0.09 (very small!)

# Test
green_l vs red_box, query 'g'

# Actual: red_box wins! (Wrong!)
# Why: L shape mismatch = 30, color mismatch = 2
#      Shape dominates → wrong answer
```

### Solution

```python
# Training on diverse shapes
green_box, green_solid, green_l, green_t, green_vbar, green_hbar → 'g'

# Result: shape variance = 0.27 (higher)

# Test
green_l vs red_box, query 'g'

# Now: green_l wins! (Correct!)
```

### Why This Works

| Dimension      | Low Diversity Training | High Diversity Training |
| -------------- | ---------------------- | ----------------------- |
| Color variance | 0.05 (low)             | 0.05 (low)              |
| Shape variance | 0.09 (low)             | 0.27 (higher)           |

Higher shape variance means:

- `(x_shape - μ_shape)² / σ²_shape` is smaller
- Shape mismatch contributes less to KL divergence
- Color difference becomes the dominant signal

---

## Test 4: Empty Alternative Enables Single-Concept Discrimination

### Problem (Without Fix)

```python
# Only one concept 'g' in table
Alt(U) = {('g',)}  # Only one alternative!

# S1 calculation
P_S1('g' | t) = exp(α·L0) / exp(α·L0) = 1.0  # Always 1!

# Result: S1 = [1, 1] for all objects → uniform posterior
```

### Solution

```python
include_empty_alt = True

Alt(U) = {(), ('g',)}  # Empty utterance added!

# S1 calculation for green_l
P_S1('g' | green_l) = exp(α·(-20)) / (exp(α·(-20)) + exp(0)) ≈ 1.0

# S1 calculation for blue_box
P_S1('g' | blue_box) = exp(α·(-50)) / (exp(α·(-50)) + exp(0)) ≈ 0.0

# Result: Correct discrimination!
```

### Why Empty Alternative Works

The empty utterance `()` has `L0 = 0` (saying nothing has no semantic mismatch). This provides a baseline:

- If `L0(t, 'g')` is much better than 0 → high S1
- If `L0(t, 'g')` is close to 0 or worse → low S1

---

## Test 5: Auto Alt From Table for Multi-Concept Discrimination

### Problem

```python
# Two concepts in table: 'g' (green), 'b' (blue)
# Query: 'g'

# Without auto_alt_from_table
Alt = {('g',)}  # Only query itself
# S1 = uniform (no competition)
```

### Solution

```python
auto_alt_from_table = True

Alt = {('g',), ('b',)}  # All known tokens!

# For green object:
L0('g') = -20 (good)
L0('b') = -50 (bad)
S1('g') = exp(-100) / (exp(-100) + exp(-250)) ≈ 1.0

# For blue object:
L0('g') = -50 (bad)
L0('b') = -20 (good)
S1('g') = exp(-250) / (exp(-250) + exp(-100)) ≈ 0.0
```

### Why This Works

This implements the RSA principle: "What you didn't say matters."

If the speaker said 'g' instead of 'b', they must mean an object where 'g' is a better description than 'b'. The competition between alternatives creates the pragmatic effect.

---

## Test 6: Intersection Semantics (AND)

### Test File

`tests/test_rsa.py::TestANDEffect::test_intersection`

### Scenario

```python
# Concepts
'a': focuses on dimension 0 (mu[0] = 0.5)
'b': focuses on dimension 1 (mu[1] = 0.5)

# Test Scene
Region 0: dim0=0.5, dim1=0.0  (matches 'a' only)
Region 1: dim0=0.0, dim1=0.5  (matches 'b' only)
Region 2: dim0=0.4, dim1=0.4  (matches both!)

# Query: "a b" (both tokens)
```

### Expected Result

- Region 2 has highest posterior

### Why This Is Expected

With AND semantics:

```
log_inc(t, "a b") = log_inc(t, "a") + log_inc(t, "b")
```

Only Region 2 has good scores for BOTH concepts, so it wins.

---

## Test 7: Multi-Intent Overlap Detection

### Test File

`tests/test_multi_intent.py`

### Scenario

```python
# Scene
obj0: blue box
obj1: red solid
obj2: blue solid  # Overlaps: blue AND solid!
obj3: red box

# Utterance: "1 blue, 1 solid"
intents = [(['blue'], 1), (['solid'], 1)]
```

### Expected Result

- Non-overlapping assignments like `{obj0, obj1}` preferred
- Overlapping assignment `{obj2, obj2}` penalized

### Why Overlap Is Penalized

1. **Exactness constraint**: If `obj2` is used for both intents, `soft_count('blue') ≈ 1` and `soft_count('solid') ≈ 1`, but we want exactly 1 for each.

2. **Informativeness**: The speaker could have said "1 blue solid" if they meant one overlapping object. Saying "1 blue, 1 solid" implies two distinct objects.

---

## Test 8: RSA Mutual Exclusivity

### Test File

`tests/test_me.py`

### Scenario

```python
# Training
blue_box → 'a'
(concept 'a' learns blue box features)

# Test Scene with familiar + novel object
Region 0: blue box (familiar, matches 'a')
Region 1: red solid (novel)

# Query: "b" (novel word)
```

### Expected Result

- Novel word 'b' goes to novel object (red solid)
- Familiar object is "protected" by known concept 'a'

### Why Mutual Exclusivity Emerges

Through RSA reasoning:

1. If speaker meant blue box, they would say 'a' (known word)
2. They said 'b' instead, so they must NOT mean blue box
3. Therefore 'b' → novel object (red solid)

This is the key ME effect that enables vocabulary growth.

---

## Summary of Test Coverage

| Test               | What It Validates                     | Key Mechanism               |
| ------------------ | ------------------------------------- | --------------------------- |
| Generalization     | Color concepts transfer to new shapes | Variance learning           |
| Token Independence | No text→template matching             | Pure feature learning       |
| Diverse Training   | Shape variance prevents dominance     | Multi-example learning      |
| Empty Alt          | Single-concept discrimination         | Baseline comparison         |
| Auto Alt           | Multi-concept discrimination          | Alternative competition     |
| Intersection       | AND semantics work correctly          | Sum of log-inclusions       |
| Overlap Detection  | Multi-intent penalizes overlap        | Exactness + informativeness |
| Mutual Exclusivity | Novel words go to novel objects       | RSA counterfactual          |
