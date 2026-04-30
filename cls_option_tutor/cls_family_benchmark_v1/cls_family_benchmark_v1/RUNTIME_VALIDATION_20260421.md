# CLS Family Benchmark v1 Runtime Validation

## Scope

This note records a first-pass runtime validation of the candidate pool in
`cls_family_benchmark_v1`.

Validation was run against the current active mainline:

```text
SIS_cf_mix_loop_v1
```

using the real repo loader, renderer, environment, learner, tutor, and
decision-time family audit.

## 1. Static Runtime Smoke

All 40 candidate tasks passed parser/render smoke:

```text
tasks: 40
support_render_ok: 40 / 40
query_render_ok:   40 / 40
```

This means the tasks are structurally usable by the current CLS loader and
renderer.

## 2. Manifest Issue Found

The original manifest used:

```text
diagnostic_quota_allow_critical_heavy
```

for the allow family rows.

The current codebase supports:

```text
diagnostic_quota_allow_heavy
```

The manifest has been normalized to the current runtime name.

## 3. Runtime Family Validation

Validation setup:

- tasks from this candidate pool
- seeds: `42, 43, 44`
- condition: `SIS_cf_mix_loop_v1`
- intended family read from manifest
- actual family measured from `build_allow_family_audit(block)`

### Observed family labels

Only the following runtime families appeared:

- `NO_PRODUCTIVE_OPPORTUNITY`
- `ROUND_BLOCKED`
- `BORING_MASTERY`

Notably absent:

- `NATIVE_LIKE_ALLOW`
- `MIXED_PROD_HARM`
- `PROTECT_CRITICAL`

### Aggregated results by intended family

#### ALLOW_CRITICAL_HEAVY

```text
total_states          = 193
BORING_MASTERY        = 3   (0.0155)
NO_PRODUCTIVE_OPPORTUNITY = 177 (0.9171)
ROUND_BLOCKED         = 13  (0.0674)
```

#### MIXED_PROD_HARM_HEAVY

```text
total_states          = 220
BORING_MASTERY        = 2   (0.0091)
NO_PRODUCTIVE_OPPORTUNITY = 178 (0.8091)
ROUND_BLOCKED         = 40  (0.1818)
```

#### PROTECT_CRITICAL_HEAVY

```text
total_states          = 289
BORING_MASTERY        = 31  (0.1073)
NO_PRODUCTIVE_OPPORTUNITY = 149 (0.5156)
ROUND_BLOCKED         = 109 (0.3772)
```

#### BORING_MASTERY_HEAVY

```text
total_states          = 227
BORING_MASTERY        = 17  (0.0749)
NO_PRODUCTIVE_OPPORTUNITY = 163 (0.7181)
ROUND_BLOCKED         = 47  (0.2070)
```

## 4. Interpretation

This candidate pool is **syntactically valid** but **not yet runtime-valid** as
a family benchmark.

The main problem is not parser failure. The main problem is family mismatch:

- intended allow tasks mostly collapse to `NO_PRODUCTIVE_OPPORTUNITY`
- intended mixed tasks do not produce `MIXED_PROD_HARM`
- intended protect tasks do not produce `PROTECT_CRITICAL`
- boring tasks do not cleanly dominate `BORING_MASTERY`

So this pool should still be treated as:

```text
candidate tasks
not a final family-shaped benchmark
```

## 5. Practical Next Step

The next useful step is not another full benchmark pass.

The next useful step is candidate refinement:

1. regenerate tasks with stronger family-specific structure
2. re-run runtime family validation
3. keep only tasks with high observed family precision

In other words:

```text
LLM generation -> parser/render validation -> runtime family validation
-> filtered family slice benchmark
```

This pool has completed the first two stages and failed the third.
