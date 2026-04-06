# Scenario & Environment Report: Core Map, Corridor, and Lesson Design

> **Handoff document** — self-contained reference describing all environments, scenarios, cell types, and pedagogical corridor designs.

---

## 1. Grid World Foundation

### 1.1 Cell Types

All environments are built on an H×W grid (default 8×8 or 7×W corridor). Each cell has a **type**, **traversal cost**, **risk probability**, and a **4D feature vector**.

| CellType | Value | Cost | Risk | Meaning |
|----------|:-----:|:----:|:----:|---------|
| `NORMAL` | 0 | 1.0 | 0.0 | Standard walkable cell |
| `WALL` | 1 | ∞ | 0.0 | Impassable barrier |
| `HIGH_COST` | 2 | 3.0–7.0 | 0.0 | Expensive but safe terrain |
| `RISKY` | 3 | 1.0 | 0.15–0.50 | Cheap but dangerous |
| `LOCKED_DOOR` | 4 | ∞ → 1.0 | 0.0 | Impassable until UNLOCK action |
| `TARGET` | 5 | 1.0 | 0.0 | Episode goal cell |
| `OBJECT_SPAWN` | 6 | 1.0 | 0.0 | Item pickup location |

### 1.2 Feature Space (4D per cell)

Each cell carries a 4-dimensional feature vector used by the learner's risk/cost model:

| Dim | Name | Semantic Role | Signal Type |
|:---:|------|--------------|-------------|
| 0 | `lane_id` | Position / row identity | **IDENTITY** (zeroed in risk weights) |
| 1 | `gate_flag` / tempt | Temptation cue / gate marker | NUISANCE (weak risk signal) |
| 2 | `texture_a` | Safety-related texture | **SEMANTIC** (strong risk signal) |
| 3 | `texture_b` | Diagnostic / safety cue | **SEMANTIC** (strong risk signal) |

**Critical design**: Risk weights are **orthogonal to identity dims**. The `semantic_subspace.py` module enforces `w_risk[0] = 0` so the learner's risk model cannot cheat by using row position alone. This is verified by `identity_leakage_probe()` (ideal accuracy ≈ 50%).

### 1.3 Default 8×8 Map

```
A . . . H H O .      A = Agent start (0,0)
. W W . H H . .      O = Object spawn (0,6)
. W W . . . . R      T = Target (7,7)
. . . D . . R R      W = Wall
. . . . . . R .      D = Locked door
H H . . . . . .      H = High cost
H H . W W . . .      R = Risky
. . . W W . . T      . = Normal
```

- 4 wall clusters, 4 high-cost zones, 4 risky cells
- Locked door at (3,3) — requires UNLOCK to open
- Path options: safe detour through high-cost, or shortcut through risky zone

---

## 2. Fork-Branch Corridor Topology

All teaching scenarios (PP-MRB, TIC, TIC-v4, ACTIVE) share a common **fork-branch corridor** topology:

```
     ┌── Branch A (row 1): cells [fork+1 .. fork+blen] ──┐
     │                                                      │
Start → Fork (row 2, col 2) ──────────────────────── Merge → Goal
     │                                                      │
     └── Branch B (row 3): cells [fork+1 .. fork+blen] ──┘
```

### Structural Parameters

| Parameter | Symbol | Meaning |
|-----------|:------:|---------|
| Branch length | `blen` | Cells per branch (default 10) |
| Commit depth | `d_commit` | Steps before branch choice is irreversible |
| Reveal depth | `d_reveal` | Steps before true nature of risky branch is revealed |
| Mirror | `mirror` | 0 = safe on row 1; 1 = safe on row 3 |
| Lure strength | `lure` | Temptation intensity on risky branch [0, 1] |
| Risk level | `risk` | Danger level of risky branch [0, 1] |
| Risk gap | `risk_gap` | Risk difference between branches |

### Why Fork-Branch Works for Teaching

The fork forces a **binary decision**: safe branch (lower reward, lower risk) vs. risky branch (higher temptation, higher risk). The interplay of `d_commit` and `d_reveal` creates the **information asymmetry** that makes tutoring valuable:

- **d_commit < d_reveal**: Learner must commit before seeing the truth → tutor's warning is crucial
- **d_commit ≈ d_reveal**: Ambiguous — learner may or may not need help
- **d_commit > d_reveal**: Truth revealed early → self-discovery opportunity, warning is unnecessary

### Feature Layout Along Branch

Each branch has a 4-segment feature structure:

```
[Pre-junction] → [Pre-commit zone] → [Commit zone] → [Post-reveal zone]
   ambiguous         weak cues          commit point       true signal visible
```

- **Safe branch**: Low temptation (0.0–0.1), high safety cues after reveal
- **Risky branch**: Low temptation before reveal; high temptation (0.5–0.9 × lure) after reveal

---

## 3. Scenario Families

### 3.1 PP-MRB: Persistent-Profile Mixed-Reveal Branches

**Source**: `src/envs/persistent_profile_mixed_reveal.py`

**Purpose**: Test persistent learner modelling across multi-episode sessions.

**Key design**: θ is fixed across a session; goal varies per episode. The tutor must remember what it has already taught.

#### Subtypes

| Subtype | d_commit | d_reveal | Lure | Risk Gap | Test Focus |
|---------|:--------:|:--------:|:----:|:--------:|------------|
| `wait_clean` | 4–6 | 1–2 | 0.1–0.4 | 0.15–0.25 | Low temptation, tutor should WAIT |
| `wait_lure` | 4–6 | 1–3 | 0.6–1.0 | 0.15–0.25 | High lure but learner sees early → still WAIT |
| `boundary_obs` | 2–4 | 2–4 | 0.3–0.7 | 0.15–0.25 | Ambiguous timing, near-equal Q → boundary test |
| `warn_trap` | 1–3 | 4–6 | 0.8–1.3 | 0.20–0.35 | Late reveal, high lure → must WARN |

#### Lesson Entries

| Lesson | Family | Severity | Primary Gain |
|--------|--------|:--------:|-------------|
| `ppmrb_standard` | PP-MRB | 0.4 | RC=0.15, EP=0.12, VA=0.10 |
| `ppmrb_self_discovery` | PP-MRB | 0.3 | EP=0.25, IA=0.10 (zero dose) |

---

### 3.2 TIC: Teaching-Internalization Corridor

**Source**: `src/envs/teaching_internalization_corridor.py`

**Purpose**: Primary teaching environment — tests lesson ranking, trust/dependence dynamics, and intervention selectivity.

#### 3-Phase Session Structure

| Phase | Episodes | Tutor | Purpose |
|:-----:|:--------:|:-----:|---------|
| A | 8 | Active | Tutoring block — observe learner response to interventions |
| B | 4 | OFF | Autonomy transfer — same structure, no tutor |
| C | 4 | OFF | Shifted structure — higher lure/risk, tests transfer |

#### Core Subtypes

| Subtype | d_commit | d_reveal | Lure | Risk | Test Focus |
|---------|:--------:|:--------:|:----:|:----:|------------|
| `temptation_repeat` | 3–5 | 1–3 | 0.7–1.0 | 0.3–0.5 | Repeated lure → temptation resistance |
| `self_discovery_teach` | 5–7 | 1–2 | 0.4–0.7 | 0.2–0.4 | Long commit, early reveal → learner should self-discover |
| `warn_rescue` | 2–3 | 3–5 | 0.6–0.9 | 0.4–0.6 | Short commit, late reveal → must warn immediately |
| `boundary_obs` | 3–5 | 3–5 | 0.3–0.5 | 0.15–0.3 | Near-equal branches → ambiguous boundary |

#### P3-A Balanced Active Subtypes

| Subtype | d_commit | d_reveal | Lure | Risk | Purpose |
|---------|:--------:|:--------:|:----:|:----:|---------|
| `soft_gradual` | 3–4 | 2–4 | 0.5–0.8 | 0.35–0.5 | Gradual dose escalation |
| `blind_corridor` | 2–3 | 4–6 | 0.5–0.8 | 0.4–0.55 | Very late reveal → blind obey test |

#### Lesson Entries

| Lesson | Family | Severity | Primary Gain | ν_push | γg_push |
|--------|--------|:--------:|-------------|:------:|:-------:|
| `tic_rescue_heavy` | TIC | 0.8 | RC=0.25, VA=0.12 | +0.06 | +0.05 |
| `tic_temptation` | TIC | 0.6 | TR=0.22 | +0.03 | +0.02 |
| `tic_self_discovery` | TIC | 0.5 | EP=0.20, VA=0.10 | −0.03 | −0.03 |

---

### 3.3 TIC-v4: 5-Phase Extended Corridor

**Source**: `src/envs/teaching_internalization_corridor_v4.py`

**Purpose**: Extended TIC with advice validity and exploration preservation tests.

#### 5-Phase Session Structure

| Phase | Episodes | Focus |
|:-----:|:--------:|-------|
| A | 10 | Tutor present — mixed subtypes |
| B | 4 | Autonomy transfer (temptation, self-discovery, false suppression, novelty) |
| C | 4 | Sparse valid advice — does learner use correct advice? |
| D | 4 | Sparse invalid advice — does learner reject wrong advice? |
| E | 4 | Beneficial novelty — does learner explore when exploration is adaptive? |

#### Additional Subtypes (beyond TIC base)

| Subtype | Source | Test Focus |
|---------|--------|------------|
| `verified_warn` | TIC-v4 | Warning that is subsequently verified correct |
| `self_discovery_needed` | TIC-v4 | Tutor must NOT intervene to let self-discovery happen |
| `false_suppression_cost` | TIC-v4 | Risky branch is **actually good** → over-warning is harmful |
| `sparse_valid_advice` | TIC-v4 | Infrequent but correct advice → tests valid-advice uptake |
| `sparse_invalid_advice` | TIC-v4 | Infrequent but wrong advice → tests invalid-advice resistance |
| `beneficial_novelty` | TIC-v4 | Novel branch that appears risky but is safe → exploration preservation |

#### Critical Flags on ScenarioConfig

| Flag | Subtypes | Meaning |
|------|----------|---------|
| `risky_branch_actually_good` | false_suppression, beneficial_novelty | Risky-looking branch is oracle-safe |
| `advice_valid` | sparse_valid_advice | External advice is correct |
| `advice_invalid` | sparse_invalid_advice | External advice is incorrect |

#### Lesson Entries

| Lesson | Family | Severity | Primary Gain |
|--------|--------|:--------:|-------------|
| `sparse_valid_advice` | TIC-v4 | 0.4 | VA=0.25 |
| `sparse_invalid_advice` | TIC-v4 | 0.5 | IA=0.25 |
| `beneficial_novelty` | TIC-v4 | 0.4 | EP=0.28 |
| `verified_warn` | TIC-v4 | 0.5 | VA=0.20 |
| `false_suppression` | TIC-v4 | 0.5 | EP=0.22 |

---

### 3.4 ACTIVE: Balanced Active Coverage Suite

**Source**: Added in P3-A to break `tic_rescue_heavy`'s monopoly on active events.

**Purpose**: Ensure nontrivial WARN/blind events under natural tutor policy.

| Lesson | Subtype | Severity | Primary Gain | Risk Family? |
|--------|---------|:--------:|-------------|:------------:|
| `warn_symmetric_rescue` | warn_rescue | 0.85 | RC=0.20, TR=0.15 | ✅ |
| `soft_boundary_tradeoff` | soft_gradual | 0.60 | RC=0.12, TR=0.10 | ❌ |
| `blind_activation_corridor` | blind_corridor | 0.75 | RC=0.15, TR=0.12 | ✅ |

**Risk families** (`{tic_rescue_heavy, warn_symmetric_rescue, blind_activation_corridor}`) are the target of the κ̂ macro bonus.

---

## 4. Complete Lesson Catalog

### 4.1 All 13 Lessons

| # | Name | Family | Subtype | Sev | Dose |
|:-:|------|--------|---------|:---:|:----:|
| 1 | `ppmrb_standard` | PP-MRB | mixed | 0.4 | 0.3 |
| 2 | `ppmrb_self_discovery` | PP-MRB | self_discovery_teach | 0.3 | 0.0 |
| 3 | `tic_rescue_heavy` | TIC | warn_rescue | 0.8 | 1.0 |
| 4 | `tic_temptation` | TIC | temptation_repeat | 0.6 | 0.5 |
| 5 | `tic_self_discovery` | TIC | self_discovery_needed | 0.5 | 0.0 |
| 6 | `sparse_valid_advice` | TIC-v4 | sparse_valid_advice | 0.4 | 0.3 |
| 7 | `sparse_invalid_advice` | TIC-v4 | sparse_invalid_advice | 0.5 | 0.0 |
| 8 | `beneficial_novelty` | TIC-v4 | beneficial_novelty | 0.4 | 0.0 |
| 9 | `verified_warn` | TIC-v4 | verified_warn | 0.5 | 0.5 |
| 10 | `false_suppression` | TIC-v4 | false_suppression_cost | 0.5 | 0.0 |
| 11 | `warn_symmetric_rescue` | ACTIVE | warn_rescue | 0.85 | 0.9 |
| 12 | `soft_boundary_tradeoff` | ACTIVE | soft_gradual | 0.6 | 0.6 |
| 13 | `blind_activation_corridor` | ACTIVE | blind_corridor | 0.75 | 0.8 |

### 4.2 Mastery Gain Heatmap

Gain vectors show which mastery probe each lesson primarily develops:

| Lesson | RC | TR | EP | VA | IA |
|--------|:--:|:--:|:--:|:--:|:--:|
| ppmrb_standard | ■ | · | ■ | ■ | · |
| ppmrb_self_discovery | · | · | ██ | · | ■ |
| tic_rescue_heavy | ██ | ■ | · | ■ | · |
| tic_temptation | · | ██ | · | · | · |
| tic_self_discovery | · | · | ██ | ■ | · |
| sparse_valid_advice | · | · | · | ██ | · |
| sparse_invalid_advice | · | · | · | · | ██ |
| beneficial_novelty | · | · | ██ | · | · |
| verified_warn | · | · | · | ██ | · |
| false_suppression | · | · | ██ | · | · |
| warn_symmetric_rescue | ██ | ■ | · | · | · |
| soft_boundary_tradeoff | ■ | ■ | · | ■ | · |
| blind_activation_corridor | ■ | ■ | · | ■ | · |

Legend: ██ = primary (≥0.15), ■ = secondary (0.08–0.14), · = weak (<0.08)

### 4.3 Internalization Side Effects (ν_push, γg_push)

| Lesson | ν_push | γg_push | Net Effect |
|--------|:------:|:-------:|------------|
| tic_rescue_heavy | **+0.06** | **+0.05** | Builds trust but risks dependence |
| tic_temptation | +0.03 | +0.02 | Moderate ν/γg inflation |
| warn_symmetric_rescue | +0.05 | +0.04 | Similar to tic_rescue |
| blind_activation_corridor | +0.04 | +0.03 | Moderate |
| ppmrb_self_discovery | **−0.04** | **−0.02** | Reduces dependence |
| tic_self_discovery | **−0.03** | **−0.03** | Reduces both ν and γg |
| beneficial_novelty | −0.02 | **−0.04** | Exploration-preserving |
| false_suppression | −0.02 | −0.03 | Reduced suppression |

**Key tension**: High-severity rescue lessons (tic_rescue_heavy, warn_symmetric_rescue) are the most effective for RC/TR gains but have the largest ν/γg inflation — the explicit overteaching risk that the tutor must manage.

---

## 5. Mastery Probes

All learner mastery assessment uses 5 probes:

| Probe | Full Name | What It Tests |
|:-----:|-----------|---------------|
| RC | Risk Calibration | Can the learner judge danger accurately? |
| TR | Temptation Resistance | Can the learner resist short-term lures? |
| EP | Exploration Preservation | Does the learner still explore when appropriate? |
| VA | Valid-Advice Uptake | Does the learner follow correct guidance? |
| IA | Invalid-Advice Resistance | Does the learner reject wrong guidance? |

Mastery is tracked via Beta-Bernoulli model: $u_k = a_k / (a_k + b_k)$ with decay.

---

## 6. Learner Types (θ)

The system models two canonical learner preference types:

| Type | Behavioral Signature | Risk Profile |
|------|---------------------|--------------|
| `safe` | Risk-averse, moderate temptation susceptibility | Safe but may be over-cautious |
| `shiny` | Novelty-seeking, high temptation susceptibility | Pursues attractive options despite risk |

The tutor maintains a posterior $q_t(\theta)$ over these types and adapts its decisions accordingly. Key differences:

- **safe** typically needs fewer lessons and stops earlier
- **shiny** needs more rescue/temptation lessons and has higher ν/γg inflation risk
- Per-θ STOP coefficients, risk budgets, and family priors are calibrated differently

---

## 7. Scenario Generation Pipeline

```
LessonV2 (from catalog)
    ↓
adaptive_episode_generator_v2.generate_episode_from_lesson_v2()
    ↓
EpisodeSpec (parameterized: d_commit, d_reveal, lure, risk, mirror, seed)
    ↓
generate_tic_scenario() / generate_episode_scenario()
    ↓
(GridMap, FamilyConfig, LatticeV2Meta, ScenarioConfig)
    ↓
BCICTv4.decide() → action = {WAIT, WARN}
```

### Key Objects

| Object | Contains |
|--------|---------|
| `GridMap` | Cell types, cost map, risk map, positions |
| `FamilyConfig` | max_steps, risk_budget, search_budget |
| `LatticeV2Meta` | Segments, gate cells, features, doors, path lengths |
| `ScenarioConfig` | Family name, branches, oracle labels, flags, temptation scores |
| `LessonV2` | Name, gain/prereq/ZPD vectors, ν_push, γg_push |

---

## 8. Source File Index

| File | Lines | Role |
|------|:-----:|------|
| `src/envs/map_generator.py` | 221 | CellType, GridMap, default/random map generators |
| `src/envs/semantic_subspace.py` | 119 | Feature partition, orthogonal weights, identity leakage probe |
| `src/envs/teaching_internalization_corridor.py` | 220 | TIC base: 3-phase, 6 subtypes, fork-branch generation |
| `src/envs/teaching_internalization_corridor_v4.py` | 99 | TIC-v4: 5-phase, 10 subtypes |
| `src/envs/persistent_profile_mixed_reveal.py` | 330 | PP-MRB: persistent sessions, 4 subtypes |
| `src/envs/scenario_families.py` | 120K | Full scenario family library (legacy) |
| `src/curriculum/lesson_library_v2.py` | 178 | 13-lesson catalog with gain/prereq/ZPD |
| `src/curriculum/adaptive_episode_generator_v2.py` | ~100 | Mastery-conditioned lesson→episode generator |
