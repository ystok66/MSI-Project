# CLS Family Benchmark v1 — Generation Audit

## Generated artifacts

- 40 candidate task files under `tasks/`
- 10 candidates per intended family:
  - `ALLOW_CRITICAL_HEAVY`
  - `MIXED_PROD_HARM_HEAVY`
  - `PROTECT_CRITICAL_HEAVY`
  - `BORING_MASTERY_HEAVY`
- `family_manifest.jsonl`
- `family_manifest.csv`
- `validate_generated_tasks.py`

## Local static validation performed

The packaged validator checked:

```text
validated 40 tasks
```

It verifies:

- each file has `*SUPPORT*`, `*QUERY*`, `*GRAMMAR*`
- support count is 12–16
- query count is 8–12
- every support/query row has `IN: ... OUT: ...`
- every grammar row contains ` -> `
- all manifest rows point to existing task files

## Important limitation

The manifest field `intended_family` is **not ground-truth**.

Runtime family must be validated by the existing decision-time classifier, using quantities such as:

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

The tasks are intentionally candidate pools, not final filtered benchmark slices.

## Recommended runtime filtering

1. Copy/symlink these tasks into a temporary task directory used by `task_adapter.py`.
2. Run parser/render smoke tests through the actual repo loader.
3. Run family audit with the manifest `generator_mode`.
4. Keep only tasks whose runtime family rates match their intended slice.
5. Use filtered tasks for formal benchmark.

## Reflection / risk checks

- The generated grammar uses only noun rules, binary infix rules, and unary postfix/prefix rules.
- The task files do not assign `risk_class`, option menus, or final family labels.
- This avoids encoding family membership by assertion; family membership must be measured from runtime learner/tutor state.
- `MIXED_PROD_HARM_HEAVY` and `PROTECT_CRITICAL_HEAVY` are intentionally designed to be screened; some candidates may fail runtime family validation and should be discarded.
