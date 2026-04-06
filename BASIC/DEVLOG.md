# Development Log: RSA Pragmatic Communication Project

This document chronicles the development journey from an empty project to a fully functional RSA-based concept learning and pragmatic inference system.

---

## Phase 1: Foundation (World & Encoder Module)

### What Was Built

- **templates.py**: Defined 8 shapes (box, solid, l, t, vbar, hbar, cross, corner) as 3×3 binary occupancy grids, and 8 colors in normalized RGB.
- **world.py**: Created `Obj` and `Scene` dataclasses for representing visual scenes with up to 4 regions.
- **encoders.py**: Implemented feature encoding, converting objects to 12D vectors (Lab color + shape occupancy).

### Design Decisions

1. **Lab Color Space**: Chose CIE Lab over RGB for perceptual uniformity. sRGB→Lab conversion uses D65 illuminant.
2. **Fixed 4 Regions**: Simplified scene to exactly 4 slots for easier tensor operations.
3. **Normalized Lab**: Scaled Lab to roughly [0,1] range to match shape features.

### Edge Cases Handled

- **Empty regions**: Represented as `None`, encoded as zeros with `mask[t]=False`.
- **Color clamping**: Lab conversion can produce out-of-range values due to color space differences; clamped during conversion.

---

## Phase 2: Gaussian Math & Scoring Module

### What Was Built

- **gaussian.py**: KL divergence for diagonal Gaussians, log-determinant.
- **scoring.py**: Inclusion scoring functions using KL divergence.

### Key Formula

```
log_inc(t, u) = -KL(A_t || B_u) / τ

where:
  A_t ~ N(x_t, ε_obj · I)  # Object (point-like)
  B_u ~ N(μ_u, diag(σ²_u))  # Concept
```

### Bugs Encountered

#### Bug 1: NaN in KL Divergence

- **Symptom**: KL returned NaN for some inputs
- **Cause**: `log(0)` when variance was exactly 0
- **Solution**: Added `clamp_var()` function to enforce `var >= min_var` (1e-8)

#### Bug 2: Infinite KL Values

- **Symptom**: KL = inf when concept variance was very small
- **Cause**: Division by near-zero variance in `(μ_a - μ_b)² / σ²_b` term
- **Solution**: Same variance clamping, plus explicit finite check with informative error

### Hyperparameter Choices

| Parameter | Value | Rationale                                         |
| --------- | ----- | ------------------------------------------------- |
| `eps_obj` | 1e-4  | Small but nonzero; makes object nearly point-like |
| `tau`     | 1.0   | No temperature scaling by default                 |
| `min_var` | 1e-8  | Numerical stability floor                         |

---

## Phase 3: Concept Table & Learner Module

### What Was Built

- **concepts.py**: `Concept` (Gaussian parameters) and `ConceptTable` (lazy-init dictionary).
- **learner.py**: Incremental learning with soft-count weighted updates.

### Design Decisions

1. **Lazy Initialization**: Concepts are created on first access via `ensure()`, not upfront.
2. **Weak Prior**: `kappa0=0.5` gives minimal prior influence, letting data dominate quickly.
3. **Order-Independent Updates**: Used Welford's parallel algorithm for weighted batch merge.

### Bugs Encountered

#### Bug 3: Variance Collapse

- **Symptom**: After many updates, variance → 0, causing KL → inf
- **Cause**: Weighted variance update without floor
- **Solution**: Added `var_floor=1e-6` in update formula:
  ```python
  var_new = M2_new / kappa_new + var_floor
  ```

#### Bug 4: Division by Zero in Weighted Mean

- **Symptom**: NaN when total weight W = 0
- **Cause**: No observations with positive weight
- **Solution**: Early return if `W <= 0`

### The Welford Update Formula

```
κ' = κ + W
μ' = μ + (W/κ') · (μ_batch - μ)
M₂' = M₂ + M₂_batch + (κ·W/κ') · (μ_batch - μ)²
σ² = M₂'/κ' + var_floor
```

This is numerically stable and order-independent.

---

## Phase 4: RSA Inference Module

### What Was Built

- **rsa.py**: L0 (literal), S1 (speaker), L1 (listener) inference.

### Key Formulas

```
S_L0(t, U) = log_inc(t, U) - β · vol(U)  # Volume penalty
P_S1(U | t) = exp(α · S_L0) / Σ_{U'} exp(α · S_L0)  # Softmax
P_L1(t | U) ∝ P(t) · P_S1(U | t)  # Bayes
```

### Bugs Encountered

#### Bug 5: Uniform S1 for Single Tokens

- **Symptom**: `S1 = [1, 1, 1, 1]` for all objects when utterance is single token like "green"
- **Cause**: Alternative set `Alt = {('green',)}` has only one element, so softmax returns 1
- **Analysis**:
  ```
  Alt(U) = P(U) \ {∅}  # Power set minus empty
  |U| = 1 → Alt = {U}  # Only one alternative!
  S1 = exp(α·L0) / exp(α·L0) = 1  # Always 1
  ```
- **Solution**: Two changes:
  1. `include_empty_alt=True` (default): Add `()` to alternatives
  2. `auto_alt_from_table=True` (default): Add all known tokens as alternatives

#### Bug 6: RSA Not Discriminating

- **Symptom**: Posterior was nearly uniform even with clear concept matches
- **Cause**: Volume penalty `β=0.1` with broad prior variance dominated the signal
- **Solution**: Adjusted beta and ensured concepts learn tighter distributions

### Hyperparameter Choices

| Parameter | Value | Rationale                                         |
| --------- | ----- | ------------------------------------------------- |
| `alpha`   | 5.0   | Moderate rationality; sharp but not deterministic |
| `beta`    | 0.1   | Mild size principle; prefer specific concepts     |
| `lam`     | 0.0   | No length penalty by default                      |

---

## Phase 5: Multi-Intent RSA

### What Was Built

Extended RSA to handle utterances like "1 blue, 1 solid" (multiple cardinalities).

### Key Concepts

- **Intent**: `(tokens, k)` tuple, e.g., `(['blue'], 1)`
- **Assignment**: Mapping of object indices to intents
- **Overlap**: Object matched by multiple intents

### Bugs Encountered

#### Bug 7: RSA Boosting Overlap Objects (Paradox)

- **Symptom**: Objects matching ALL intents got higher probability than non-overlapping targets
- **Example**:

  ```
  Scene: blue_box(0), red_solid(1), blue_solid(2), green_l(3)
  Utterance: "1 blue, 1 solid"

  Correct: {(0,), (1,)} or {(0,), (2,)} (non-overlap preferred)
  Actual: {(2,), (2,)} got high score (overlap boosted)
  ```

- **Cause**: S1 was using raw L0 scores, which didn't account for ambiguity
- **Analysis**: An ambiguous utterance (applies to many targets) should get LOWER S1, but raw L0 gave same score regardless of how many other targets it matched
- **Solution**: Changed S1 to use normalized L0 posterior:
  ```
  P_L0(T | U') = exp(α·L0(T,U')) / Σ_T' exp(α·L0(T',U'))
  S1(U | T) ∝ P_L0(T | U)^α
  ```
  This penalizes ambiguous utterances naturally.

#### Bug 8: Exactness Not Enforced

- **Symptom**: "2 blue" could match 1 or 3 blue objects
- **Cause**: No constraint on cardinality matching
- **Solution**: Added exactness constraint:
  ```
  soft_count(u, A) = Σ_t sigmoid(α · L0(t, u))  # How many objects match u?
  exactness_penalty = γ · Σ_u (soft_count - k)²
  ```
  With `exactness_gamma=2.0`, this strongly penalizes count mismatches.

#### Bug 9: Alternative Generation Created Duplicates

- **Symptom**: Alternatives like `[('1 b',), ('1 b',)]` appeared
- **Cause**: Multi-intent alternative generation was including semantically identical patterns
- **Solution**: Deduplicated alternatives during generation

### Hyperparameter Choices

| Parameter         | Value | Rationale                       |
| ----------------- | ----- | ------------------------------- |
| `exactness_gamma` | 2.0   | Strong penalty for wrong counts |

---

## Phase 6: Concept Learning Generalization

### What Was Built

- Tests for zero-shot generalization to novel shapes
- Auto-alt system for single-concept cases

### Bugs Encountered

#### Bug 10: Shape Dominates Color ("Dimensional Signal Dominance")

- **Symptom**: Concept trained on "green box" didn't generalize to "green L"
- **Cause**: Shape occupancy (9 dimensions) overwhelmed color (3 dimensions) in KL calculation
- **Analysis**:

  ```
  green box → green L:
    Color difference: small (same green)
    Shape difference: large (box vs L = 6 dimensions differ)

  KL = color_term + shape_term
  Shape term >> Color term → L shape "looks" very different from box
  ```

- **Solution**: Train on diverse shapes (box AND solid) to increase variance in shape dimensions:
  ```python
  learn_step(green_box, 'g')
  learn_step(green_solid, 'g')  # Different shape, same color
  ```
  Now the concept has high shape variance (less sensitive) but low color variance (more discriminative).

#### Bug 11: Single Concept Still Gave Uniform Posterior

- **Symptom**: Even with diverse training, `P(green_l | "g") = 0.5`
- **Cause**: Only one concept in table → `Alt = {('g',)}` → S1 = 1 for all
- **Solution**:
  1. `include_empty_alt=True`: Compare to saying nothing
  2. Changed default from `False` to `True` for all RSA functions

### Key Insight: Why Empty Alt Works

```
Alt = {(), ('g',)}

For green_l:
  L0('g') = -20 (good match)
  L0(())  = 0 (empty = no penalty)
  S1      = exp(5·(-20)) / (exp(5·(-20)) + exp(0)) ≈ 1

For blue_box:
  L0('g') = -50 (poor match)
  L0(())  = 0
  S1      = exp(5·(-50)) / (exp(5·(-50)) + exp(0)) ≈ 0

Result: green_l >> blue_box ✓
```

---

## Summary of All Hyperparameters

| Module       | Parameter             | Default | Description                          |
| ------------ | --------------------- | ------- | ------------------------------------ |
| scoring      | `eps_obj`             | 1e-4    | Object observation variance          |
| scoring      | `tau`                 | 1.0     | KL temperature                       |
| scoring      | `min_var`             | 1e-8    | Minimum variance for stability       |
| concepts     | `kappa0`              | 0.5     | Initial pseudo-count                 |
| concepts     | `var0`                | ones    | Prior variance (broad)               |
| learner      | `var_floor`           | 1e-6    | Variance floor                       |
| rsa          | `alpha`               | 5.0     | Speaker rationality                  |
| rsa          | `beta`                | 0.1     | Volume penalty weight                |
| rsa          | `lam`                 | 0.0     | Length cost weight                   |
| rsa          | `include_empty_alt`   | True    | Include ∅ in alternatives            |
| rsa          | `auto_alt_from_table` | True    | Use all known tokens as alternatives |
| multi-intent | `exactness_gamma`     | 2.0     | Exactness constraint strength        |

---

## Phase 7: Scalar Implicature (v3.3)

### What Was Built

Extended dynamic `include_empty_alt` logic to handle Scalar Implicature scenarios automatically.

### The Problem

```
Scene: red_box, blue_box (no specialized name), blue_solid (has "solid" name)
Query: "blue"

L0:  blue_box=68.5%, blue_solid=31.5%  ← No pragmatic reasoning
RSA: blue_box=68.5%, blue_solid=31.5%  ← RSA = L0, no improvement!
```

**Cause**: Empty utterance `∅` (L0=0) dominated all real utterances (L0≈-40) in S1 softmax.

### Root Cause Analysis

The dynamic logic only disabled empty alt for **Mutual Exclusivity (ME)**:

```python
# Original condition
if len(known_tokens) >= 1 and len(novel_tokens) > 0:
    include_empty_alt = False
```

But Scalar Implicature uses **known tokens only** (e.g., "blue" when "solid" was available), so the condition never triggered.

### Solution: Extended Dynamic Logic

```python
# Scenario A: ME - novel word + known concepts
me_condition = len(known_tokens) >= 1 and len(novel_tokens) > 0

# Scenario B: Scalar Implicature - ≥2 known concepts compete
scalar_condition = len(known_tokens) >= 2

if me_condition or scalar_condition:
    include_empty_alt = False
```

### Results After Fix

```
Scene: red_box, blue_box, blue_solid
Query: "blue"

L0:  blue_box=68.5%, blue_solid=31.5%
RSA: blue_box=98.2%, blue_solid=1.8%  ← Scalar Implicature works!
```

**Interpretation**: Speaker said "blue" instead of "solid", so target must not have a "solid" name.

### Key Insight: Breaking Asymmetry

Scalar Implicature requires **information asymmetry**:

- Target: Only describable by query word (blue_box → "blue")
- Competitor: Has a more specific alternative (blue_solid → "solid" better than "blue")

If both objects have equal alternatives, no pragmatic advantage exists.

### Tests Added

| Test   | Scenario           | L0          | RSA        | Delta  |
| ------ | ------------------ | ----------- | ---------- | ------ |
| Case 1 | ME (novel word)    | 37.9%/62.0% | 0%/100%    | +37.9% |
| Case 9 | Scalar Implicature | 68.5%/31.5% | 98.2%/1.8% | +29.7% |

---

## Simplifications Made

1. **Fixed 4 Regions**: Instead of variable-length scenes, always 4 slots with mask.
2. **Diagonal Covariance**: No cross-dimensional correlations (12 params vs 78 for full).
3. **Uniform Prior**: `P(t) = 1/n` over non-empty regions.
4. **Greedy Assignment**: Multi-intent uses factorial enumeration, not sampling.
5. **Single-Step Learning**: Each utterance updates immediately, no batch epochs.

---

## Files Changed (Chronological)

1. `templates.py` - Shape/color definitions
2. `world.py` - Scene representation
3. `encoders.py` - Feature encoding
4. `gaussian.py` - Gaussian math
5. `scoring.py` - Inclusion scoring
6. `concepts.py` - Concept table
7. `learner.py` - Learning updates
8. `rsa.py` - RSA inference (most changes)
9. `tests/*.py` - 125+ tests

---

## Lessons Learned

1. **Test Early, Test Often**: Unit tests caught many edge cases before they propagated.
2. **Log-Space is Your Friend**: Work in log-probabilities to avoid underflow.
3. **Clamp Everything**: Variance floors, probability floors prevent NaN explosions.
4. **Theory Guides Debugging**: Understanding RSA math helped diagnose paradoxes.
5. **Diverse Training Matters**: Single-example learning can overfit to dimensions.
