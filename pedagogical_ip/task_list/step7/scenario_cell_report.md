# Step 7 Scenario & Cell Report — Complete Environment Reference

> **Handoff document** — self-contained reference for all grid environments, scenario families, cell types, and feature semantics.
> Source of truth: `src/envs/scenario_families.py` (3,050 lines), `src/envs/map_generator.py`, `src/envs/lattice_v2.py`, `src/envs/cgc_v2_family.py`, `src/envs/teaching_internalization_corridor.py`, `src/envs/map_families.py`.

---

## 1. Cell Types

All environments share a common `CellType` enum (`src/envs/map_generator.py`):

| Value | Name | Traversal | Cost | Risk | Semantic |
|:-----:|:-----|:---------:|:----:|:----:|:---------|
| 0 | `NORMAL` | ✅ | 1.0 | 0.0 | Standard passable cell |
| 1 | `WALL` | ❌ | ∞ | 0.0 | Impassable barrier |
| 2 | `HIGH_COST` | ✅ | 3.0–7.0 | 0.0 | Expensive terrain (legacy maps) |
| 3 | `RISKY` | ✅ | 1.0 | 0.02–0.60 | Risk of failure event |
| 4 | `LOCKED_DOOR` | ❌→✅ | ∞→1.0 | 0.0 | Impassable until UNLOCK intervention |
| 5 | `TARGET` | ✅ | 1.0 | 0.0 | Goal destination |
| 6 | `OBJECT_SPAWN` | ✅ | 1.0 | 0.0 | Item pickup location (legacy maps) |

### Cell Properties

Each cell `(r, c)` carries three scalar values:

```
true_cost[r, c]  : float  — traversal cost (≥1.0 for passable, ∞ for walls)
true_risk[r, c]  : float  — probability of stochastic failure event ∈ [0, 1]
features[r, c]   : float[4]  — 4D perceptual feature vector (see §2)
```

---

## 2. Feature Vector (4D)

All V2 scenario families use a shared 4-dimensional feature vector per cell:

| Dim | Constant | Name | Range | Semantic |
|:---:|:---------|:-----|:-----:|:---------|
| 0 | `F_LANE_ID` | Lane ID | [0, 1] | 0.0 = upper branch (row 1), 1.0 = lower branch (row 3), 0.5 = corridor |
| 1 | `F_GATE_FLAG` | Gate / temptation | [0, 1] | 1.0 = gate/corridor cell, high values = temptation salience |
| 2 | `F_TEXTURE_1` | Texture 1 / risk cue | [0, 1] | Higher = more visually "dangerous" (used by WorldWeights) |
| 3 | `F_TEXTURE_2` | Texture 2 / safety cue | [0, 1] | Higher = more visually "dangerous" |

### Feature Archetypes

| Archetype | F0 (lane) | F1 (gate) | F2 (tex1) | F3 (tex2) | Generator |
|:----------|:---------:|:---------:|:---------:|:---------:|:----------|
| Safe cell | lane_id | 0.0 | 0.00–0.10 | 0.00–0.10 | `_safe_feature()` |
| Trap cell | lane_id | 0.0 | 0.80–0.95 | 0.70–0.90 | `_trap_feature()` |
| Weak cue | lane_id | 0.0 | 0.30–0.50 | 0.20–0.40 | `_weak_cue_feature()` |
| Mild risky | lane_id | 0.0 | 0.10–0.20 | 0.05–0.15 | `_lane_feature(mild=True)` |
| Corridor | 0.5 | 1.0 | 0.00 | 0.00 | Hardcoded |
| Gate | lane_id | 1.0 | 0.00 | 0.00 | Hardcoded |

### Latent Mode

When `latent_mode=True`, cost and risk are **derived from features** via learned WorldWeights:

```python
cost[r, c] = ww.true_cost(features[r, c])    # linear function of 4D features
risk[r, c] = ww.true_risk(features[r, c])     # linear function of 4D features
```

The agent must **learn** the feature→risk mapping from experience. The tutor knows the true mapping.

---

## 3. Grid Topology: V2 Lattice

The standard V2 lattice topology (shared by most families) uses a **7-row grid**:

```
Row 0: ████████████████████████  (wall)
Row 1: ░░░░ risky lane ░░░░░░░  (straight, fast)
Row 2: ═══E════╤════╤════G════  (corridor: Start, forks, Goal)
Row 3: ░░░░ safe lane ░░██░░░░  (zigzag, slower)
Row 4: ████████████░░░░████████  (detour vertical)
Row 5: ████████████░░░░████████  (detour horizontal)
Row 6: ████████████████████████  (wall)
```

### Segment Structure

Each grid contains 1–4 **segments** — forced-choice sections where the agent must pick a lane:

```
     ┌─ risky lane (row 1): straight, L cells ─────┐
S ──>│                                               │──> corridor ──> next segment / Goal
     └─ safe lane (row 3): zigzag via row 4-5 ─────┘
            (longer by ~2×detour cells)
```

**Key invariant**: Row 2 is WALLED inside each segment, forcing the agent onto row 1 or row 3. Row 2 is passable only at entry/exit columns and between segments.

### Detour Mechanics

The safe lane (row 3) has an intentional **wall gap** at the midpoint, forcing a vertical detour:

```
Row 3: →→→→→→ [WALL] →→→→→→
              ↓            ↑
Row 4:        →            ←
              ↓            ↑
Row 5:        →→→→→→→→→→→→→
```

This makes the safe path **longer** than the risky path, creating a genuine cost-safety tradeoff.

---

## 4. Scenario Families — Complete Registry

### 4.1 `baseline_v2` — Regression Anchor

| Property | Value |
|:---------|:------|
| **Purpose** | Default V2 lattice; regression baseline |
| **Segments** | 3 (random widths 5–7 cols each) |
| **Primary lever** | WARN |
| **Failure mode** | Risk |
| **Grid size** | 7 × ~25 |
| **Features** | Standard trap/safe/weak-cue archetypes |

Standard multi-segment lattice with probabilistic trap placement. Each segment independently randomizes which cells are traps (difficulty controls trap probability: easy=50%, medium=70%, hard=90%).

---

### 4.2 `fork_trap` — Ambiguous Lane Fork

| Property | Value |
|:---------|:------|
| **Purpose** | Test WARN with ambiguous pre-fork cues |
| **Segments** | 1 |
| **Topology** | Single fork with two branches; one has hidden trap |
| **Primary lever** | WARN |
| **Failure mode** | Risk (agent takes risky branch) |
| **Key parameter** | `cue_ambiguity` ∈ {0.3, 0.6, 0.9} |

Two branches enter the fork with **near-identical local cues**. The risky branch is safe for the first `trap_depth` cells, then escalates. The safe branch has a zigzag detour making it longer.

| Difficulty | Ambiguity | Trap Depth | Trap Risk | Time Ratio |
|:-----------|:---------:|:----------:|:---------:|:----------:|
| Easy | 0.3 | 1 | 0.30 | 1.50 |
| Medium | 0.6 | 2 | 0.45 | 1.35 |
| Hard | 0.9 | 3 | 0.60 | 1.20 |

**Teaching insight**: The tutor must warn BEFORE the agent commits to the risky branch, because pre-trap cells look safe.

---

### 4.3 `hazard_belt` — Unavoidable Risk Zone

| Property | Value |
|:---------|:------|
| **Purpose** | Test ITEM_DROP intervention (shield against risk) |
| **Segments** | 3 (safe → belt → safe) |
| **Topology** | Middle segment has BOTH lanes risky |
| **Primary lever** | ITEM_DROP |
| **Failure mode** | Risk (belt traversal) |
| **Belt mode** | `unavoidable` (no safe lane) or `near_unavoidable` (costly bypass) |

The belt segment is a mandatory risk zone. ITEM_DROP halves the risk, making it the natural intervention. Non-belt segments have standard safe+risky structure.

| Difficulty | Belt Width | Belt Risk | Bypass Extra | Time Ratio |
|:-----------|:---------:|:---------:|:------------:|:----------:|
| Easy | 2 | 0.25 | 6 | 1.50 |
| Medium | 2 | 0.30 | 8 | 1.35 |
| Hard | 3 | 0.35 | 10 | 1.20 |

---

### 4.4 `deadline_gate` — Tight Deadline + Gated Shortcut

| Property | Value |
|:---------|:------|
| **Purpose** | Test UNLOCK intervention |
| **Topology** | Shortcut (row 1, gated) + long safe path (row 3, multi-segment) |
| **Primary lever** | UNLOCK |
| **Failure mode** | Timeout (agent can't finish via long path in budget) |
| **Key cell type** | `LOCKED_DOOR` at shortcut entry |

The shortcut is genuinely **zero risk** once unlocked — UNLOCK is pure topology assistance. The long safe path has RISKY hazard cells scattered along it with `safe_risk` ∈ {0.15, 0.20, 0.25}.

| Difficulty | Long Segments | Safe Risk | RISKY/seg | Time Ratio |
|:-----------|:------------:|:---------:|:---------:|:----------:|
| Easy | 3 | 0.15 | 1 | 1.15 |
| Medium | 4 | 0.20 | 2 | 1.10 |
| Hard | 4 | 0.25 | 2 | 1.05 |

**Teaching insight**: Hard mode time ratio = 1.05 makes the long path nearly impossible without timeout. The tutor must decide: UNLOCK (topology help) vs. WARN (navigate risky safe-path cells).

---

### 4.5 `delayed_corridor` — Late-Revealing Risk

| Property | Value |
|:---------|:------|
| **Purpose** | Test prefix-aware (forward-looking) warning |
| **Segments** | 1 (long single fork) |
| **Topology** | Corridor A: safe prefix → deep risk zone. Corridor B: safe zigzag |
| **Primary lever** | WARN (timing-sensitive) |
| **Failure mode** | Commitment (agent too deep to backtrack) |
| **Key parameter** | `safe_prefix` — how many cells look safe before risk escalates |

A myopic tutor that only considers current-cell risk will warn **too late** — after the agent has committed past the point of no return. A prefix-aware tutor can predict the path and warn at entry.

| Difficulty | Corridor Len | Safe Prefix | Deep Risk | Time Ratio |
|:-----------|:----------:|:-----------:|:---------:|:----------:|
| Easy | 7 | 2 | 0.35 | 1.30 |
| Medium | 8 | 3 | 0.45 | 1.20 |
| Hard | 9 | 4 | 0.60 | 1.10 |

---

### 4.6 `distractor_cue` — Misleading Local Features

| Property | Value |
|:---------|:------|
| **Purpose** | Test robustness to noisy/inverted feature-risk correlation |
| **Topology** | Standard V2 lattice (3 segments) |
| **Primary lever** | WARN (provides ground truth overriding misleading cues) |
| **Failure mode** | Cue error (agent's learned model misled by features) |
| **cue_mode** | `weak` (noisy) or `misleading` (inverted) |

After generating a standard V2 lattice, features are **corrupted**:
- `weak` mode: Gaussian noise scaled by `(1 - reliability)`
- `misleading` mode: risky cells get safe-looking textures, safe cells get risky-looking textures

| Difficulty | Reliability | Distractor Frac | Time Ratio |
|:-----------|:----------:|:---------------:|:----------:|
| Easy | 0.6 | 10% | 1.50 |
| Medium | 0.3 | 25% | 1.35 |
| Hard | 0.0 | 40% | 1.20 |

---

### 4.7 `funnel_trap` — Multi-Stage Funnel

| Property | Value |
|:---------|:------|
| **Purpose** | Test timing-sensitive WARN with 2-stage commitment |
| **Topology** | Stage 1: 3-branch fork → merge M1 → Stage 2: 2-corridor commitment |
| **Primary lever** | WARN (must warn before commitment point) |
| **Failure mode** | Commitment (irrecoverable past depth 2 in trap corridor) |
| **Branches** | Stage 1: 3 (rows 1, 3, 5); Stage 2: 2 (rows 1, 3) |

Most complex topology. Stage 1 has one branch with escalating weak cues (pre-trap zone). Stage 2 has a straight tempting corridor (trap) and a zigzag safe corridor. The commitment point is 2 cells into the trap corridor — past this, backtracking exceeds the deadline.

| Difficulty | S1 Length | S2 Corridor | Ambiguity | Trap Risk | Time Ratio |
|:-----------|:--------:|:-----------:|:---------:|:---------:|:----------:|
| Easy | 5 | 3 | 0.3 | 0.30 | 1.50 |
| Medium | 6 | 4 | 0.6 | 0.45 | 1.35 |
| Hard | 7 | 5 | 0.9 | 0.60 | 1.20 |

**Structure labels** (attached to ScenarioConfig):
- `decision_points`: cells where branches diverge
- `commitment_points`: cells past which backtracking fails
- `cue_cells`: cells with weak/ambiguous features
- `safe_branches` / `trap_branches`: cell lists
- `merge_cells`: cells where branches rejoin

---

### 4.8 `elcb` — Equal-Length Competing Branches

| Property | Value |
|:---------|:------|
| **Purpose** | **Diagnostic**: remove topology confound (equal-length branches) |
| **Topology** | S → fork → branch A (row 1) / branch B (row 3) → merge → G |
| **Invariants** | |πA| = |πB|, both fully passable, symmetric visibility |
| **Primary lever** | WARN |
| **Failure mode** | Risk |

Pure risk competition: branches have equal length but different risk profiles. Tests whether the agent's predictions can flip branch choice based on features alone (not path length heuristic).

| Difficulty | Branch Len | Risk Gap | Cue Strength | Time Ratio |
|:-----------|:---------:|:--------:|:------------:|:----------:|
| Easy | 4 | 0.30 | 0.8 | 2.0 |
| Medium | 5 | 0.20 | 0.6 | 2.0 |
| Hard | 6 | 0.12 | 0.4 | 2.0 |

---

### 4.9 `elcb_po` — ELCB with Partial Observability

Same topology as ELCB but with **staged cue visibility**:
- Early cells (depth < `reveal_depth`): WEAK cues (both branches look similar)
- Late cells (depth ≥ `reveal_depth`): STRONG cues (clear discrimination)

Tests whether the tutor can warn based on the trajectory prefix, before diagnostic cues become visible.

---

### 4.10 `temptation_corridor` — Temptation Branch

| Property | Value |
|:---------|:------|
| **Purpose** | Test preference-dependent warning (θ-sensitivity) |
| **Topology** | Fork: safe branch vs tempting risky branch |
| **Primary lever** | WARN |
| **Failure mode** | Temptation (shiny agents lured to risky branch) |
| **Key feature** | Dim 1 (temptation salience) on risky branch |

The risky branch has high values on feature dim 1 (temptation), which attracts `shiny`-preference agents. The tutor doesn't know the agent's θ a priori and must infer it from behavior.

---

### 4.11 `joint_conflict_corridor` — Goal vs Preference Conflict

| Property | Value |
|:---------|:------|
| **Purpose** | Test joint (g, θ) inference under goal-preference conflict |
| **Topology** | Fork: goal-aligned branch vs temptation-aligned branch |
| **Key feature** | Staggered reveal: goal cue appears before preference cue |
| **Primary lever** | WARN |
| **Failure mode** | Joint conflict |

Branch A has high goal cues (dim 2), Branch B has high temptation cues (dim 1). Goal cues reveal earlier than preference/temptation cues, testing whether the tutor can correctly infer the simultaneous influence of goal and preference on branch choice.

---

## 5. Session-Based Environments

### 5.1 TIC — Teaching Internalization Corridor

**Source**: `src/envs/teaching_internalization_corridor.py`

3-phase session designed to test internalization dynamics:

| Phase | Episodes | Tutor Active | Structure | Purpose |
|:------|:--------:|:------------:|:----------|:--------|
| A | 8 | ✅ | Mixed subtypes | Tutoring block |
| B | 4 | ❌ | Same structures as A | No-tutor transfer (same) |
| C | 4 | ❌ | Shifted (higher lure) | No-tutor transfer (shifted) |

#### TIC Subtypes

| Subtype | d_commit | d_reveal | Lure | Risk | Purpose |
|:--------|:--------:|:--------:|:----:|:----:|:--------|
| `temptation_repeat` | 3–5 | 1–3 | 0.7–1.0 | 0.3–0.5 | Repeated temptation resistance |
| `self_discovery_teach` | 5–7 | 1–2 | 0.4–0.7 | 0.2–0.4 | Agent should discover on its own |
| `warn_rescue` | 2–3 | 3–5 | 0.6–0.9 | 0.4–0.6 | Must warn NOW (urgent rescue) |
| `boundary_obs` | 3–5 | 3–5 | 0.3–0.5 | 0.15–0.3 | Ambiguous — near-equal branches |
| `soft_gradual` | 3–4 | 2–4 | 0.5–0.8 | 0.35–0.5 | Gradual difficulty escalation |
| `blind_corridor` | 2–3 | 4–6 | 0.5–0.8 | 0.4–0.55 | Late reveal, early commitment |

**Grid**: 7-row fork structure (same as ELCB). Branch length = 10. Equal-length branches (no topology confound in TIC).

**Key parameters per episode**:
- `d_commit`: commitment depth (beyond which backtracking fails)
- `d_reveal`: reveal depth (after which cues become discriminative)
- `Δ = d_commit - d_reveal`: the "intervention window" — larger Δ = more time to warn
- `lure_strength`: temptation salience on risky branch
- `risk_level`: risk probability on risky cells

---

### 5.2 CGC-v2 — Compositional Goal Corridor

**Source**: `src/envs/cgc_v2_family.py`

Session-based environment for compositional goal inference.

#### Goal Space

| Type | Goals | 4D Weight Vector [safety, tempt, novelty, speed] |
|:-----|:------|:------------------------------------------------|
| Atomic | `collect_red` | [0.0, 2.5, 0.5, 0.0] — temptation-seeking |
| Atomic | `avoid_blue` | [2.0, −1.0, 0.0, 0.0] — risk-averse |
| Atomic | `use_safe` | [3.0, −0.5, 0.0, 0.0] — safety priority |
| Atomic | `reach_fast` | [0.0, 0.0, 0.0, 3.0] — speed priority |
| Composite | `collect_red + avoid_blue` | Sum of atomics |
| Composite | `collect_red + use_safe` | Sum of atomics |
| Composite | `avoid_blue + use_safe` | Sum of atomics |
| Composite | `reach_fast + avoid_blue` | Sum of atomics |

#### Episode Subtypes

| Subtype | Alignment | d_commit | d_reveal | Lure | Purpose |
|:--------|:----------|:--------:|:--------:|:----:|:--------|
| `goal_aligned` | g ∧ θ agree | 4–6 | 1–2 | 0.1–0.4 | Easy: natural agreement |
| `goal_conflict` | g ∧ θ disagree | 1–2 | 4–6 | 0.7–1.2 | Hard: goal overrides preference |
| `goal_boundary` | Ambiguous | 2–4 | 2–4 | 0.3–0.7 | Edge case: observation needed |

#### Grid Structure

7-row fork: S → fork → branch A (row 1) / branch B (row 3) → merge → G. Branch length = 10. Both branches equal-length. One branch carries goal-aligned cues, the other carries temptation cues.

**Feature layout**: Goal weights modulate feature dimensions — e.g., `collect_red` emphasizes dim 1 (temptation), `avoid_blue` emphasizes dim 3 (safety).

---

## 6. Legacy Map Families (v1b)

**Source**: `src/envs/map_families.py` (725 lines)

Four 10×10 grid families for earlier benchmark:

| Family | Best Intervention | Grid | Purpose |
|:-------|:-----------------|:----:|:--------|
| `semantic_trap` | WARN | 10×10 | Learner misbelieves risk |
| `planning_trap` | UNLOCK | 10×10 | Bounded planner can't find safe detour |
| `exploration_useful` | WAIT | 10×10 | Exploration improves transfer |
| `mixed` | Varies by phase | 10×10 | Mixed optimal intervention |

These are **still in use** (imported by 4 modules). They predate the V2 lattice system.

---

## 7. Default Map (8×8)

**Source**: `src/envs/map_generator.py`

Hand-designed 8×8 map for regression testing:

```
A . . . H H O .     A = agent start (0,0)
. W W . H H . .     O = object spawn (0,6)
. W W . . . . R     T = target (7,7)
. . . D . . R R     W = wall
. . . . . . R .     D = locked door
H H . . . . . .     H = high cost (5.0)
H H . W W . . .     R = risky (0.3)
. . . W W . . T     . = normal (1.0)
```

---

## 8. Scenario Config & Metadata

Every scenario generation returns 4 objects:

```python
(gm: GridMap, cfg: FamilyConfig, meta: LatticeV2Meta, sc: ScenarioConfig)
```

### FamilyConfig

| Field | Type | Meaning |
|:------|:-----|:--------|
| `max_steps` | int | Episode deadline |
| `risk_budget` | float | Total risk budget (not used in all families) |
| `prior_risk_mean` | float | Agent's prior belief about risk (0.02) |
| `prior_risk_var` | float | Agent's prior variance (0.20) |
| `search_budget` | int | A* planner search limit (30) |
| `budget_class` | int | Budget discretization (8 or 10) |

### LatticeV2Meta

| Field | Type | Meaning |
|:------|:-----|:--------|
| `segments` | list[SegmentMeta] | Per-segment metadata |
| `all_gate_cells` | list[(r,c)] | All entry/exit gate positions |
| `all_door_positions` | list[(r,c)] | Doors that can be unlocked |
| `shortest_any` | int | BFS shortest path (any route) |
| `shortest_safe` | int | BFS shortest path avoiding risky gates |
| `cell_features` | ndarray (H,W,4) | Full feature map |
| `world_weights` | WorldWeights | Latent cost/risk model (if latent_mode) |
| `latent_mode` | bool | Whether features drive cost/risk |

### SegmentMeta (per segment)

| Field | Type | Meaning |
|:------|:-----|:--------|
| `index` | int | Segment index (0-based) |
| `col_start` / `col_end` | int | Column span |
| `risky_row` / `safe_row` | int | Which row is risky vs safe |
| `risky_cells` / `safe_cells` | list[(r,c)] | Cell coordinates |
| `risky_entry_gate` / `safe_entry_gate` | (r,c) | Lane entry positions |
| `trap_cell` | (r,c) or None | Highest-risk cell (if exists) |
| `weak_cue_cells` | list[(r,c)] | Cells with ambiguous features |
| `detour_len` | int | Safe lane detour length |

### ScenarioConfig (family-specific)

| Field | Type | Meaning |
|:------|:-----|:--------|
| `family_name` | str | Scenario family identifier |
| `difficulty` | str | easy / medium / hard |
| `primary_intervention` | str | Natural lever (WARN / UNLOCK / ITEM_DROP) |
| `cue_reliability` | float | Feature-risk correlation [0, 1] |
| `expected_failure_mode` | str | risk / timeout / cue_error / commitment / temptation |
| `requires_gate` | bool | Needs UNLOCK |
| `requires_item` | bool | Needs ITEM_DROP |
| `gate_mode` | str | `block_risky` or `unlock_shortcut` |
| `commitment_cells` | list[(r,c)] | Point-of-no-return cells |
| `belt_regime` | str | `unavoidable` or `near_unavoidable` |

---

## 9. Summary: Family × Mechanism Matrix

| Family | WARN | UNLOCK | ITEM_DROP | Cue Noise | Commitment | Multi-Stage | θ-Sensitive | g-Sensitive |
|:-------|:----:|:------:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:|
| `baseline_v2` | ✅ | | | | | ✅ (3 seg) | | |
| `fork_trap` | **✅** | | | ✅ | | | | |
| `hazard_belt` | | | **✅** | | | ✅ (3 seg) | | |
| `deadline_gate` | | **✅** | | | ✅ | ✅ (4-5 seg) | | |
| `delayed_corridor` | **✅** | | | | **✅** | | | |
| `distractor_cue` | **✅** | | | **✅** | | ✅ (3 seg) | | |
| `funnel_trap` | **✅** | | | ✅ | **✅** | **✅** (2-stage) | | |
| `elcb` | ✅ | | | | | | | |
| `elcb_po` | **✅** | | | ✅ | | | | |
| `temptation_corridor` | **✅** | | | | | | **✅** | |
| `joint_conflict_corridor` | **✅** | | | | | | **✅** | **✅** |
| TIC (session) | **✅** | | | | ✅ | | **✅** | |
| CGC-v2 (session) | **✅** | | | | ✅ | | **✅** | **✅** |

**Bold** = primary design target for that family.

---

## 10. Source File Index

| File | Lines | Families | Role |
|:-----|:-----:|:---------|:-----|
| `scenario_families.py` | 3,049 | baseline_v2, fork_trap, hazard_belt, deadline_gate, delayed_corridor, distractor_cue, funnel_trap, elcb, elcb_po, temptation_corridor, joint_conflict_corridor | Registry + all V2 generators |
| `cgc_v2_family.py` | 326 | CGC-v2 | Compositional goal session |
| `teaching_internalization_corridor.py` | 219 | TIC | 3-phase internalization session |
| `teaching_internalization_corridor_v4.py` | 98 | TIC-v4 | Latest TIC variant |
| `lattice_v2.py` | 394 | (base lattice) | Core V2 lattice generator |
| `map_generator.py` | 220 | default 8×8 | CellType, GridMap, default/random maps |
| `map_families.py` | 725 | v1b (semantic/planning/exploration/mixed) | Legacy 10×10 families |
| `scenario_families.py` → `SCENARIO_REGISTRY` | — | 11 families | Unified `generate_scenario()` entry point |
