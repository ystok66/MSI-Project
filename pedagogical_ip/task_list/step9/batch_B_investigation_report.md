# Batch B Investigation Report: Scenario Generator Correctness & Latent-Mode Semantics

> Date: 2026-04-08
> Scope: B1 (scenario bugs), B2 (latent-mode semantics), B3 (post-generation assertions)
> Method: Static call-chain analysis + 100-seed empirical audit + latent risk mismatch quantification
> Principle: Investigate only, no production code changes

---

## Part A. Scenario Family & Main Experiment Mapping

### A.1 Family Registry & Usage Matrix

| Family | Registry Key | Generator Location | Main Experiments |
|--------|-------------|-------------------|-----------------|
| baseline_v2 | `baseline_v2` | `scenario_families.py` L84 | Step 2, Phase 1/2, regression anchor |
| fork_trap | `fork_trap` | `scenario_families.py` L116 | Step 2/5, necessity audit, planner shadow |
| hazard_belt | `hazard_belt` | `scenario_families.py` L379 | Step 5, ITEM_DROP lever |
| deadline_gate | `deadline_gate` | `scenario_families.py` L636+ | Step 5, UNLOCK lever |
| delayed_corridor | `delayed_corridor` | `scenario_families.py` L2198 (2nd def) | Step 2, prefix-aware WARN |
| distractor_cue | `distractor_cue` | `scenario_families.py` L2406 (2nd def) | Step 2, transfer |
| harder_baseline_v2 | `harder_baseline_v2` | `harder_baseline.py` L60 | Phase 2B, transfer benchmark |
| DTMB | `deep_tree_mixed_bottleneck_lattice` | `dtmb_lattice.py` L315 | Eval DTMB freeze, predictor/redundancy audits |
| GTET | `goal_preference_temptation_entanglement_lattice` | `gtet_lattice.py` | Eval GTET Z-repair, fair dispatch, calibration |

### A.2 Per-Family Summary

#### baseline_v2
- **In main experiments**: Yes (regression anchor for all evaluations)
- **Scientific purpose**: Default V2 lattice, no specialized mechanism
- **Latent mode**: **PURE_LATENT** — risk fully from WorldWeights, 0% override
- **Known bugs**: None new. Risk may be low for "trap" cells if WorldWeights don't align with features, but this is by design (agent must learn).
- **Enters main experiments**: Yes

#### fork_trap
- **In main experiments**: Yes (Step 2, Step 5 necessity audits)
- **Scientific purpose**: Ambiguous lane fork with WARN lever
- **Latent mode**: **PURE_LATENT** — 0% override (80/80 risky cells match WorldWeights)
- **Known bugs**: **CRITICAL** — safe_row==1 breaks detour connectivity (see B1.2)
- **Enters main experiments**: Yes — directly affects Step 2/5 results

#### hazard_belt
- **In main experiments**: Yes (ITEM_DROP lever evaluation)
- **Scientific purpose**: Unavoidable high-risk zone, shield reduces risk
- **Latent mode**: **FULL_OVERRIDE** — 100% of risky cells have post-hoc risk override (L566-584)
- **Known bugs**: Latent-mode override breaks feature->risk learning contract
- **Enters main experiments**: Yes

#### deadline_gate
- **In main experiments**: Yes (Step 5)
- **Scientific purpose**: Tight deadline + gated shortcut, UNLOCK lever
- **Latent mode**: PURE_LATENT (0% override in checked seeds)
- **Known bugs**: `shortest_safe` metadata mismatch — 0/100 seeds match BFS recomputation
- **Enters main experiments**: Yes

#### delayed_corridor
- **In main experiments**: Yes (Step 2)
- **Scientific purpose**: Late-revealing risk, prefix-aware WARN
- **Latent mode**: PURE_LATENT
- **Known bugs**: **Duplicate definition** (1st at L911, 2nd at L2198). Registry binds 2nd. 1st is dead code.
- **Enters main experiments**: Yes (via 2nd definition)

#### distractor_cue
- **In main experiments**: Limited (Step 2 only)
- **Scientific purpose**: Misleading local cues, transfer
- **Latent mode**: PURE_LATENT
- **Known bugs**: **Duplicate definition** (1st at L1117, 2nd at L2406). Registry binds 2nd. 1st is dead code.
- **Enters main experiments**: Yes (via 2nd definition)

#### harder_baseline_v2
- **In main experiments**: Yes (Phase 2B, transfer benchmark)
- **Scientific purpose**: Multi-segment transfer difficulty
- **Latent mode**: **PURE_LATENT** — 0% override
- **Known bugs**: seg_width=3 was suspected but **empirically not broken** — 100/100 seeds have reachable safe paths even with seg_width=3 (70% of seeds have at least one seg_width=3 segment)
- **Enters main experiments**: Yes

#### DTMB
- **In main experiments**: Yes (eval freeze, predictor audit, redundancy audit)
- **Scientific purpose**: 3-stage tree with mixed bottleneck
- **Latent mode**: **CONTRACTED_OVERRIDE** — belt cells use `max(ww_risk, belt_risk*0.8)` floor
- **Known bugs**: Stage 2/3 topology order-sensitivity (see B1.4)
- **Enters main experiments**: Yes

#### GTET
- **In main experiments**: Yes (Z-repair, fair dispatch, calibration)
- **Scientific purpose**: Goal-preference-temptation entanglement
- **Latent mode**: **NO_WORLDWEIGHTS** — does not use WorldWeights at all, risk is handcrafted
- **Known bugs**: None structural (separate architecture from 7-row families)
- **Enters main experiments**: Yes

---

## Part B. B1-B3 Detailed Investigation

---

### B1.1 Duplicate Function Definitions

#### Evidence

| Function | 1st definition | 2nd definition | Registry binds |
|----------|---------------|----------------|---------------|
| `generate_delayed_corridor` | L911 | L2198 | **L2198** (2nd) |
| `generate_distractor_cue` | L1117 | L2406 | **L2406** (2nd) |

#### Caller Analysis

All callers import from `scenario_families` module level:
- `tests/test_scenario_families.py` L11 — imports both names
- `SCENARIO_REGISTRY` L3042-3043 — binds both to 2nd definition

Python name binding: the 2nd `def` overwrites the 1st in module namespace. All imports get the 2nd definition. **The 1st definitions are 100% dead code.**

#### External import check
```
grep -r "generate_delayed_corridor" --include="*.py" → only test_scenario_families.py and scenario_families.py
grep -r "generate_distractor_cue" --include="*.py" → only test_scenario_families.py and scenario_families.py
```

No external script directly imports the 1st definition separately.

#### Impact Level: **Low** (dead code, not affecting results)

#### Fix Recommendation: **Delete 1st definitions** (L911-1116 for delayed_corridor, L1117-~2197 for distractor_cue). ~1300 lines of dead code removed.

---

### B1.2 fork_trap safe_row==1 Detour Disconnection

#### Empirical Evidence (100 seeds)

| Metric | Value |
|--------|-------|
| safe_row==1 frequency | **61/100 (61%)** |
| safe_row==1 with working safe path | **0/61 (0%)** |
| safe_row==3 frequency | 39/100 (39%) |
| safe_row==3 with working safe path | 39/39 (100%) |
| **Overall safe path availability** | **39/100 (39%)** |

#### Root Cause

`scenario_families.py` L220-225:
```python
if safe_row == 3:
    detour_rows = [4, 5]
else:
    # safe_row == 1, detour not needed in current architecture
    # but we still use rows 4, 5 for consistency
    detour_rows = [4, 5]  # ← BUG: rows 4,5 are NOT adjacent to row 1
```

Row 1 cannot reach row 4 without passing through rows 2 and 3. Row 2 is the corridor (walled in segment), row 3 is the risky branch. So the detour in rows 4,5 is physically unreachable from row 1.

#### Which experiments are affected

All scripts using `fork_trap`:
- `run_step2_phase2b.py` — 61% of seeds have no safe alternative
- `run_step2_warning_experiment.py` — same
- `run_step5a1_necessity_gate_audit.py` — specifically tests safe/risky choice
- `run_step5a_planner_shadow.py` — planner evaluation

**Impact**: When safe_row==1, the scenario degenerates from "choose safe or risky" to "only risky available". This means:
- WARN intervention has no effect (no safe alternative to redirect to)
- `shortest_safe` falls back to 999 → `t_max = max(int(time_ratio * 999), 1001)` → absurdly long episodes
- Fork-specific metrics (safe lane selection rate) are meaningless for 61% of seeds

#### Fix Recommendation

**Option A: Don't build detour when safe_row==1** (minimal diff)
```python
if safe_row == 3:
    detour_rows = [4, 5]
else:
    detour_rows = []  # row 1 has enough space for straight safe path
```
But this means safe_row==1 has no detour → straight-through safe path → different difficulty.

**Option B: Fix safe_row to 3** (most stable)
```python
risky_row = 1
safe_row = 3
```
Eliminates randomization of which row is safe. Fork choice is still meaningful because entry from row 2 goes to either row 1 or 3.

**Option C (RECOMMENDED): Keep randomization but mirror detour geometry**
```python
if safe_row == 3:
    detour_rows = [4, 5]
else:
    # safe_row == 1: detour goes UP through row 0
    # But row 0 is wall → NOT possible
    detour_rows = []  # no detour for row 1
    # Instead, build straight safe path without gap
```

Since row 0 is wall and can't be used for detour, the cleanest fix is **Option B** (fix safe_row=3). This eliminates the randomization but keeps the fork choice meaningful. The comment at L223 already acknowledges "detour not needed" for safe_row==1, suggesting the original author knew this was problematic.

**Actual recommended fix**: **Option B** — remove `rng.choice([1,3])` and hardcode `risky_row=1, safe_row=3`.

#### Impact Level: **CRITICAL** — 61% of main experiment seeds are broken

---

### B1.3 harder_baseline_v2 seg_width=3

#### Empirical Evidence (100 seeds)

| Metric | Value |
|--------|-------|
| Seeds with seg_width=3 | 70/100 (70%) |
| Safe path broken when seg_width=3 | **0/70 (0%)** |
| Overall safe path availability | **100/100 (100%)** |

#### Conclusion

**NOT A BUG.** The seg_width=3 concern from the audit report was theoretical. In practice, the detour logic handles seg_width=3 correctly — the detour start/end calculations produce valid positions within the segment bounds.

#### Impact Level: **None** — no fix needed

---

### B1.4 DTMB Stage 2/3 Topology Issues

#### Static Analysis

**Stage 2 (detour row overlap)**: `dtmb_lattice.py` builds branch detours sequentially. If two branches share adjacent rows, their detour cells may overlap. However, since DTMB uses a tree structure with explicit `s1_rows` assignment, row conflicts are unlikely unless `n_branches >= 4` (which would require H > 7).

**Stage 3 (`entry_rows.index`)**: The `sorted(set(...))` on entry rows can indeed break the parent-child mapping if multiple branches converge to the same row. But DTMB's architecture ensures each branch exits to a distinct row.

#### Empirical Test

DTMB generated successfully for all 10 seeds tested. Goal was reachable in all cases. Would need specific multi-branch configurations to trigger the theoretical bugs.

#### Impact Level: **Low-Medium** — theoretically possible but not triggered in standard configurations

#### Recommendation: Add Stage 2/3 assertions (see B3), don't restructure.

---

### B2. Latent-Mode Semantic Classification

#### Empirical Results (20-30 seeds per family)

| Family | Classification | Risky Cells | Pure Latent | Override | Mismatch |
|--------|---------------|------------|-------------|----------|----------|
| baseline_v2 | **PURE_LATENT** | 255 | 255 (100%) | 0 | N/A |
| fork_trap | **PURE_LATENT** | 80 | 80 (100%) | 0 | N/A |
| hazard_belt | **FULL_OVERRIDE** | 200 | 0 (0%) | 200 (100%) | mean=0.271, max=0.715 |
| harder_baseline_v2 | **PURE_LATENT** | 100 | 100 (100%) | 0 | N/A |
| DTMB | **CONTRACTED_OVERRIDE** | 240 | 240 (100%)* | 0* | *belt cells use `max()` floor |
| GTET | **NO_WORLDWEIGHTS** | N/A | N/A | N/A | Uses handcrafted risk |

*DTMB: The `max()` floor at L453-455 only activates when WorldWeights would produce risk below `belt_risk * 0.8`. In practice, belt features are designed with high texture values, so WorldWeights often produces adequate risk → empirically 0% override. But the floor exists and CAN override.

#### Design Decision Analysis

**Route A (Pure Latent)**:
- Families that naturally fit: `baseline_v2`, `fork_trap`, `harder_baseline_v2`, `delayed_corridor`, `distractor_cue`
- These families' risk structure can be fully controlled by feature design + WorldWeights sampling
- No post-hoc override needed

**Route B (Contracted Override)**:
- Families that need it: `hazard_belt`, `DTMB` (belt cells only)
- These families have a "bottleneck must be dangerous" contract that WorldWeights alone cannot guarantee
- Override is explicit and bounded

**GTET**:
- Doesn't use WorldWeights at all — fundamentally different architecture
- Should be classified separately: **topology-driven structured benchmark**

#### Recommendation

> [!IMPORTANT]
> **Do NOT force GTET or hazard_belt into pure latent mode.** Their scientific purpose requires guaranteed high-risk bottlenecks. Instead, formalize the override contract:
>
> 1. Pure latent families: metadata `latent_contract = "pure"` — `true_risk == WorldWeights(z)` always
> 2. Override families: metadata `latent_contract = "contracted_override"` — belt/bottleneck cells may have `max(ww_risk, floor)` with floor documented in metadata
> 3. GTET: metadata `latent_contract = "topology_driven"` — no WorldWeights, risk is structural

This formalizes the existing reality rather than trying to unify what shouldn't be unified.

---

### B3. Post-Generation Assertion Design

#### B3.1 Current State

| Family | Existing checks | Missing checks |
|--------|----------------|----------------|
| All families | None post-generation | goal reachable, safe path existence, metadata consistency |
| fork_trap | None | safe_row detour connectivity |
| hazard_belt | None | belt risk floor, override documentation |
| DTMB | None | stage topology, branch reachability |

#### B3.2 Metadata Consistency Issues

From the 100-seed audit:

| Family | `shortest_any` match | `shortest_safe` match |
|--------|---------------------|-----------------------|
| baseline_v2 | 100/100 | 100/100 |
| fork_trap | 100/100 | **39/100** (only safe_row==3 seeds) |
| hazard_belt | 100/100 | **0/100** |
| deadline_gate | 100/100 | **0/100** |
| delayed_corridor | 100/100 | 100/100 |
| distractor_cue | 100/100 | 100/100 |
| harder_baseline_v2 | 100/100 | 100/100 |

**Cause of mismatches**:
- `fork_trap`: `shortest_safe` uses `risky_gates` set for BFS avoidance, which is the risky entry gate. My audit used `CellType.RISKY` avoidance — different criteria. The metadata BFS avoids gate cells, not all RISKY cells. **This is a semantic difference, not a bug.** The metadata definition of "safe" means "avoid the risky branch entry", not "avoid all individually risky cells".
- `hazard_belt`/`deadline_gate`: Similar — metadata `shortest_safe` uses gate-avoidance BFS, while my audit avoided all RISKY cells.

> [!NOTE]
> The metadata `shortest_safe` is **semantically correct within its own definition** (gate-avoidance BFS). The mismatch is in my audit's BFS definition, not in the code. However, the metadata concept of "safe" should be explicitly documented.

#### B3.3 Recommended Assertion Hierarchy

**Level 1: Universal (add to all families)**
```python
def validate_scenario_contract(gm, meta, start, goal):
    assert bfs_reachable(gm, start, goal), "Goal unreachable"
    assert meta.shortest_any > 0, "shortest_any invalid"
    # Validate shortest_any matches BFS
    actual = bfs_shortest(gm, start, goal)
    assert actual == meta.shortest_any, f"shortest_any mismatch: meta={meta.shortest_any}, actual={actual}"
```

**Level 2: Family-specific**
```python
# fork_trap
assert safe_path_exists(gm, start, goal, avoiding=risky_entry_gate), \
    "fork_trap: safe path must exist"

# hazard_belt
for belt_cell in belt_cells:
    assert gm.true_risk[belt_cell] >= belt_risk_floor, \
        "hazard_belt: belt risk below floor"

# DTMB
for stage_branches in branch_topology:
    assert all(bfs_reachable(gm, branch_entry, branch_exit) for ...), \
        "DTMB: branch not reachable"
```

**Level 3: Latent contract**
```python
if meta.latent_contract == "pure":
    for r, c in risky_cells:
        assert abs(gm.true_risk[r,c] - ww.true_risk(features[r,c])) < 0.01
```

#### B3.4 Implementation Recommendation

Create `src/envs/scenario_contract.py` with shared validation helpers. Each generator calls `validate_scenario_contract()` at the end. Family-specific checks are added as needed.

**Batch B scope**: Implement Level 1 only. Level 2 and 3 are Batch C.

---

## Part C. Test Design

### C.1 Unit Tests (B1 fixes)

| Test | Target | Priority |
|------|--------|----------|
| `test_fork_trap_safe_path_always_exists` | 100 seeds, all must have safe path | P0 |
| `test_fork_trap_detour_connectivity` | Detour cells reachable from safe lane | P0 |
| `test_dead_code_removal` | Import delayed_corridor/distractor_cue works after cleanup | P0 |
| `test_dtmb_branch_reachability` | All branches reachable in 20 seeds | P1 |

### C.2 100-seed Reachability Smoke

```python
@pytest.mark.parametrize("family", MAIN_FAMILIES)
def test_goal_reachable_100_seeds(family):
    for seed in range(100):
        gm, _, meta, _ = generate_scenario(family, seed=seed, latent_mode=True)
        assert bfs_reachable(gm, start, goal)
```

### C.3 Family Contract Tests

```python
@pytest.mark.parametrize("family", PURE_LATENT_FAMILIES)
def test_pure_latent_contract(family):
    """Pure latent families: true_risk == WorldWeights(z)."""
    for seed in range(20):
        gm, _, meta, _ = generate_scenario(family, seed=seed, latent_mode=True)
        ww = meta.world_weights
        for r, c in risky_cells(gm):
            z = meta.cell_features[r, c]
            assert abs(gm.true_risk[r,c] - ww.true_risk(z)) < 0.01
```

### C.4 Metadata Consistency

```python
def test_metadata_shortest_any_consistency(family, seeds=100):
    for seed in range(seeds):
        gm, _, meta, _ = generate_scenario(family, seed=seed, latent_mode=True)
        actual = bfs_shortest(gm, start, goal)
        assert actual == meta.shortest_any
```

---

## Part D. Redundancy & Cleanup

### D.1 Must Fix Now (Batch B)

| Item | Type | Lines | Impact |
|------|------|-------|--------|
| fork_trap safe_row==1 detour bug | **Real bug** | L176-177, L220-225 | 61% of seeds broken |
| Dead code: 1st `generate_delayed_corridor` | Dead code | L911-~2197 | ~1300 lines removable |
| Dead code: 1st `generate_distractor_cue` | Dead code | L1117-~2197 | (included in above range) |

### D.2 Mark Deprecated (not in Batch B)

| Item | Reason |
|------|--------|
| `lambda_uncertainty` single-weight path | Replaced by `lambda_uc/lambda_ur` |
| `_bfs_len` duplicate implementations | Multiple BFS scattered across files |
| `observation_model.py` V0 | No callers |

### D.3 Archive/Delete Later

| Item | Reason |
|------|--------|
| 1st definitions of delayed_corridor/distractor_cue | 100% dead code, confirmed no callers |
| V0 `bounded_astar`, `plan_next_action` (non-V2) | No callers, already deprecated |
| Multiple BFS implementations | Consolidate to one shared helper |

---

## Answers to Hard Questions

### 1. Are the duplicate function definitions 100% caller-free?

**Yes.** Python module-level name binding means the 2nd definition overwrites the 1st. All imports get the 2nd. `SCENARIO_REGISTRY` binds the 2nd. No external script imports the 1st definition separately. **Safe to delete.**

### 2. fork_trap safe-row bug real trigger frequency?

**61%** of seeds have safe_row==1, and **100% of those are broken** (no safe path). Overall, only 39/100 seeds have a working fork_trap scenario. This is the highest-priority fix in Batch B.

### 3. harder_baseline_v2 seg_width=3 — does it pollute transfer benchmark?

**No.** Empirical 100-seed audit shows 0/100 broken safe paths, even though 70% of seeds have at least one seg_width=3 segment. The detour logic handles it correctly. **Not a bug, no fix needed.**

### 4. DTMB Stage 2/3 — which is worse for main experiments?

Neither triggers in standard configurations (10/10 seeds OK). Stage 3's `sorted(set(...))` is theoretically more dangerous because it could silently reorder parent-child mappings. But with typical n_branches=2-3, the sort doesn't change order. **Add assertions, don't restructure.**

### 5. Which families have metadata misalignment?

`shortest_any` is consistent across all families. `shortest_safe` has a **semantic definition difference** — metadata computes it by avoiding risky entry gates (topology-level), not by avoiding individual RISKY cells. This is **by design**, not a bug. But the definition should be documented explicitly.

`fork_trap` is the exception: when safe_row==1, `shortest_safe` falls back to 999, which directly corrupts `t_max`.

### 6. Pure latent vs contracted override?

| Classification | Families |
|---------------|----------|
| **Pure latent** | `baseline_v2`, `fork_trap`, `harder_baseline_v2`, `delayed_corridor`, `distractor_cue` |
| **Contracted override** | `hazard_belt` (full override), `DTMB` (belt floor) |
| **Topology-driven** | `GTET` (no WorldWeights) |

**Recommendation**: Don't force unification. Formalize the existing three-tier classification in metadata. Pure latent families must guarantee `true_risk == WorldWeights(z)`. Override families must document which cells are overridden and by how much.

### 7. Which assertions give maximum value for minimum diff?

1. **`validate_scenario_contract(gm, meta, start, goal)`** — universal, ~15 lines, catches reachability + metadata
2. **`fork_trap` safe-path assertion** — family-specific, ~5 lines, prevents the 61% breakage
3. **Pure latent contract check** — `assert abs(risk - ww_risk) < epsilon`, ~10 lines

Total: ~30 lines of assertions, catches 3 categories of problems.

### 8. What should be fixed vs archived?

| Action | Items |
|--------|-------|
| **Fix** | fork_trap safe_row bug |
| **Fix** | Add Level 1 assertions (reachability + metadata) |
| **Delete** | 1st definitions of delayed_corridor/distractor_cue (~1300 lines) |
| **Document** | Latent-mode classification in metadata |
| **Later** | BFS consolidation, DTMB Stage 2/3 assertions, GTET contract |

---

## Recommended Batch B Execution Order

1. `fix(fork_trap): hardcode safe_row=3 to fix 61% broken scenarios`
2. `refactor(scenario_families): remove dead code — 1st definitions of delayed_corridor and distractor_cue`
3. `feat(scenario_contract): add Level 1 post-generation validation`
4. `docs(latent_mode): formalize pure_latent / contracted_override / topology_driven classification`
5. `test(batch-b): 100-seed reachability + fork_trap + latent contract tests`
