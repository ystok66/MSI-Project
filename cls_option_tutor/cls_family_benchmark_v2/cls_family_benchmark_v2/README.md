# CLS Family Benchmark v2 Candidate Pool

Candidate pool for runtime family validation in `cls_option_tutor`.
This is not a final benchmark: the manifest `intended_family` is only a generation target.
Ground-truth family membership must be computed by the runtime family audit.

## Contents

- regenerated semantics-aligned CLS task files in `tasks/`
  - `MIXED_PROD_HARM_HEAVY` candidates from screened source clones
  - `PROTECT_CRITICAL_HEAVY` candidates from screened source clones
- `family_manifest.jsonl`
- `family_manifest.csv`
- `validate_generated_tasks.py`
- `generate_semantics_aligned_candidates.py`
- `GENERATION_AUDIT.md`

## Regeneration method

This pool is no longer free-form LLM-authored by declared outputs.

It is regenerated from screened real source tasks by:

```text
source task
-> lowercase vocabulary rename
-> exact render validation
-> runtime family validation
-> formal slice filtering
```

This keeps the current repository semantics as ground truth.

## Required validation pipeline

Do not use file names or intended family as truth.

The required pipeline is:

```text
1. structure sanity
2. exact render validation under current task_adapter.py semantics
3. runtime family validation on exact-valid tasks only
4. formal slice assembly from accepted tasks only
```

The old lightweight validator is no longer sufficient by itself.

### Exact render validation

Run:

```powershell
python cls_option_tutor\cls_family_benchmark_v2\cls_family_benchmark_v2\validate_generated_tasks.py
```

This now checks exact support/query render consistency, not just file shape.

### Full pipeline

Run:

```powershell
python -m cls_option_tutor.experiments.validate_family_candidate_pool `
  --pool-dir cls_option_tutor\cls_family_benchmark_v2\cls_family_benchmark_v2 `
  --condition SIS_cf_mix_loop_v1 `
  --seeds 42 43 44 `
  --rho 0.3
```

The pipeline writes:

- `validation_outputs/structure_validation.csv`
- `validation_outputs/exact_render_validation.csv`
- `validation_outputs/runtime_family_validation.csv`
- `validation_outputs/formal_slice/`
- `validation_outputs/validation_report.md`

Filter tasks by actual decision-time metrics:

- `MixedProdHarmRate`
- `ProtectCriticalRate`
- `NativeLikeAllowRate`
- `MeanPProd`
- `MeanHarmMass`
- `MeanSafeDiagQualityGap`
- no-tutor `TeachDamage` / `DeathBeforeCorrect`

Do not treat file names or intended family as final truth.

For the exact generation contract aligned to the current renderer, use:

- [docs/cls_option_tutor/FAMILY_BENCHMARK_TASK_ADAPTER_GENERATION_SPEC_20260421.md](/F:/SCAI/Learning-agent/docs/cls_option_tutor/FAMILY_BENCHMARK_TASK_ADAPTER_GENERATION_SPEC_20260421.md)
