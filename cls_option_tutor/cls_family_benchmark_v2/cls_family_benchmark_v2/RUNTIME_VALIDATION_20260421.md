# CLS Family Benchmark v2 Runtime Validation

Date: `2026-04-21`

## Scope

This note records a first-pass validation of:

```text
cls_option_tutor/cls_family_benchmark_v2/cls_family_benchmark_v2
```

against the current active mainline:

```text
SIS_cf_mix_loop_v1
```

Validation used the real repository loader, renderer, environment, learner,
tutor, and decision-time family audit.

## 1. Static format validator

The bundled validator passes:

```text
validated 100 tasks
```

This only checks file structure and simple manifest-level constraints.
It does not verify that the repository parser / renderer reproduces the
declared outputs.

## 2. Real parser/render validation

Using the real `task_adapter.py` renderer:

```text
tasks total                    = 100
support/query load failures    = 0
tasks exact-render valid       = 0 / 100
tasks with at least one None render = 22 / 100
```

Per intended family:

```text
MIXED_PROD_HARM_HEAVY:
  tasks    = 50
  exact_ok = 0
  with_none = 11

PROTECT_CRITICAL_HEAVY:
  tasks    = 50
  exact_ok = 0
  with_none = 11
```

So the v2 pool is **not semantically valid under the current repository
parser/render semantics**.

### Typical failure mode

Example from `mixed_v2_001.txt`:

```text
IN: zup nolo mip kree
declared OUT: RED ORANGE RED ORANGE RED ORANGE
repo render:   RED ORANGE ORANGE ORANGE
```

The candidate file assumes a different operator composition / precedence model
than the current renderer.

This is the dominant issue in v2.

## 3. Runtime family validation under `SIS_cf_mix_loop_v1`

A runtime family audit was run with:

- seeds: `42, 43, 44`
- intended families from `family_manifest.csv`
- actual family from `build_allow_family_audit(block)`

### MIXED_PROD_HARM_HEAVY

```text
state_count            = 1189
NO_PRODUCTIVE_OPPORTUNITY = 847  (0.7124)
ROUND_BLOCKED             = 289  (0.2431)
BORING_MASTERY            = 53   (0.0446)

NativeLikeAllowRate    = 0.0000
MixedProdHarmRate      = 0.0000
ProtectCriticalRate    = 0.0000
AllowPreserveRate      = 0.0000
ProductiveRevealRate   = 0.5021
LoopCompleteRate       = 0.0000
MeanPProd              = 0.2016
MeanHarmMass           = 0.0000
MeanSafeDiagQualityGap = 0.2016
MeanPcorrectWAIT       = 0.1758
```

### PROTECT_CRITICAL_HEAVY

```text
state_count            = 1382
NO_PRODUCTIVE_OPPORTUNITY = 810  (0.5861)
ROUND_BLOCKED             = 482  (0.3488)
BORING_MASTERY            = 90   (0.0651)

NativeLikeAllowRate    = 0.0000
MixedProdHarmRate      = 0.0000
ProtectCriticalRate    = 0.0000
AllowPreserveRate      = 0.0000
ProductiveRevealRate   = 0.1013
LoopCompleteRate       = 0.0000
MeanPProd              = 0.0830
MeanHarmMass           = 0.0000
MeanSafeDiagQualityGap = 0.0830
MeanPcorrectWAIT       = 0.2686
```

## 4. Interpretation

`cls_family_benchmark_v2` does **not** currently meet the acceptance bar for a
runtime-valid mixed/protect family benchmark.

There are two independent failures:

1. Real parser/render mismatch:
   all 100 tasks fail exact semantic validation.

2. Runtime family mismatch:
   intended mixed/protect tasks collapse to:
   `NO_PRODUCTIVE_OPPORTUNITY`, `ROUND_BLOCKED`, and `BORING_MASTERY`.
   They do not produce runtime `MIXED_PROD_HARM` or `PROTECT_CRITICAL`.

So this pool should remain:

```text
candidate generation artifact
not accepted benchmark slice
```

## 5. Practical next step

Do not benchmark this pool as-is.

Next step should be:

```text
strict generation prompt / grammar alignment
-> repo parser/render validation
-> runtime family validation
-> keep only tasks with actual mixed/protect precision
```

The main fix target is generation semantics, not tutor policy.
