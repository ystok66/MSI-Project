# DTMB-L Benchmark Contract v1

## Design Goal

DTMB-L (Deep Tree Mixed-Bottleneck Lattice) is a benchmark family that tests
whether a pedagogical tutor can correctly **switch between macro intervention
levers** ({WAIT, WARN, UNLOCK, ITEM_DROP}) across three sequential bottleneck
stages within a single episode:

1. **Stage 1 — Epistemic ambiguity** → WARN / WAIT
2. **Stage 2 — Structural pressure** → UNLOCK
3. **Stage 3 — Outcome bottleneck** → ITEM_DROP

## Difficulty Semantics

Each difficulty level has a **dominant lever gradient**:

| Difficulty | Primary Lever | Secondary | Tertiary | Design Purpose |
|------------|---------------|-----------|----------|----------------|
| easy       | —             | —         | —        | Proof-of-concept / entry |
| **medium** | **WARN**      | ITEM_DROP | —        | Epistemic discrimination |
| **hard**   | UNLOCK        | ITEM_DROP | WARN     | Full mixed-bottleneck |

### Medium

- **Required**: Δ_warn > 0, Δ_item > 0
- **Not required**: Δ_unlock > 0

Medium is designed so that the agent can always detour around doors
(`mid_door_fraction=0.25`, `deadline_ratio=1.15`). UNLOCK provides no
marginal survival value. This is **by design** — medium tests whether
WARN can steer the agent away from epistemic traps.

### Hard (HARD_v2)

- **Required**: Δ_unlock > 0, Δ_item > 0
- **Expected**: Δ_warn ≥ 0

Hard uses denser doors (`mid_door_fraction=0.35`) and a tighter deadline
(`deadline_ratio=1.16`), making UNLOCK structurally necessary.

Calibrated parameters (from Exp B, J_hard=0.250):
- `belt_risk = 0.40`
- `terminal_belt_fraction = 0.50`
- `deadline_ratio = 1.16`

## Frozen Components

### WARN Helper: W1

Locked as the default WARN target scoring function. Uses:
- GT risk (door presence)
- Distance weighting (agent proximity)
- Door suppression (−100 penalty)

**Rationale**: W2 (risk-only) and W3 (risk+commit) produce Δ_warn ≤ 0
because they fail to target the branch the agent is actually heading toward.
The distance term is essential, not redundant.

### Oracle: dtmb_oracle

- `always_close` must NOT be used as DTMB oracle (it blocks the goal path)
- O1: low-intervention schedule baseline (warns once, unlocks nearby, drops shield)
- O2: route-aware targeting (provides survival upper bound)

### Evaluation Policies

| Tag | Description |
|-----|-------------|
| canonical | All levers active |
| no_warn | {WAIT, UNLOCK, ITEM_DROP} |
| no_unlock | {WAIT, WARN, ITEM_DROP} |
| no_item_drop | {WAIT, WARN, UNLOCK} |
| no_tutor | No interventions |
| oracle_O1 | GT stage-aware, minimal intervention |
| oracle_O2 | GT stage-aware, route-targeting |

## What Counts as Regression

A change is a **regression** if any of these become true:

1. Medium: Δ_warn ≤ 0 (50-seed, previously +0.30–0.40)
2. Medium: Δ_total ≤ 0 (canonical vs no_tutor)
3. Hard (HARD_v2): Surv_canonical leaves [0.10, 0.35]
4. Hard (HARD_v2): Surv_no_tutor > 0.10
5. Oracle goal = 0 on any difficulty
6. Any existing invariance test fails
7. Non-DTMB regression checks fail

## What Is NOT a Bug

1. **Medium Δ_unlock = 0**: UNLOCK is optional on medium by design
2. **Oracle_O1 survival < canonical**: O1 is conservative and may under-intervene
3. **WARN count > 1 per episode for canonical**: The intervention policy may
   fire WARN multiple times (via `score_interventions`); only the oracle is
   single-WARN
4. **Hard survival < 0.30**: Hard is meant to be difficult

## Reference Results (DTMB-L v1, 50-seed)

### Medium
| Policy | Surv | Goal | Δ |
|--------|------|------|---|
| canonical | 0.56 | 0.08 | — |
| no_warn | 0.16 | 0.16 | Δ_warn = +0.40 |
| no_item | 0.48 | 0.02 | Δ_item = +0.08 |
| no_tutor | 0.02 | 0.02 | Δ_total = +0.54 |
| oracle_O2 | 0.74 | 0.20 | upper bound |

### Hard (HARD_v2)
| Policy | Surv | Goal |
|--------|------|------|
| canonical | 0.20 | varies |
| no_tutor | 0.00 | 0.00 |
| no_item | 0.10 | varies |
| oracle_O2 | 0.54 | 0.22 |
