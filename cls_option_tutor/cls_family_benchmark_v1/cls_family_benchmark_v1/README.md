# CLS Family Benchmark v1

Generated candidate benchmark for `cls_option_tutor` family-conditioned evaluation.

## Contents

- `tasks/*.txt`: 40 CLS task candidates, 10 per intended family.
- `family_manifest.jsonl`: manifest with intended family and generator mode.
- `family_manifest.csv`: CSV version of the manifest.
- `validate_generated_tasks.py`: lightweight structure validator.

## Intended families

1. `ALLOW_CRITICAL_HEAVY`
2. `MIXED_PROD_HARM_HEAVY`
3. `PROTECT_CRITICAL_HEAVY`
4. `BORING_MASTERY_HEAVY`

## Important caveat

The `intended_family` field is only a generation intent. Final family membership must be
validated by the current runtime family classifier using decision-time features:

```text
P_prod
HarmMass
Q_safe = P_safe_diag - (P_far + P_highrisk)
P_correct_WAIT
NativeLikeAllowRate
MixedProdHarmRate
ProtectCriticalRate
BoringMasteryRate
```

Do not treat the manifest label as ground truth.

## Suggested validation pipeline

1. Copy or symlink `tasks/*.txt` into a temporary task directory used by `task_adapter.py`.
2. Run parser/render smoke tests.
3. Run family audit under the desired generator mode from the manifest.
4. Keep only candidates whose runtime family rates match the target slice.
5. Use filtered tasks for formal benchmark.

## Generation constraints

- 4--5 nouns per task.
- 3--4 operators per task.
- 13--14 support examples per task.
- 10 query examples per task.
- Grammar uses simple patterns only:
  - `noun -> COLOR`
  - `u1 op u2 -> ...`
  - `x1 op -> ...`
  - `op x1 -> ...`

