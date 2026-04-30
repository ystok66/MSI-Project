# Family Candidate Pool Validation Report

- Pool: `F:\SCAI\Learning-agent\cls_option_tutor\cls_family_benchmark_v2\cls_family_benchmark_v2`
- Condition: `SIS_cf_mix_loop_v1`
- Seeds: `42, 43, 44`
- `min_target_rate`: `0.05`

## Stage 1: Structure

- Structure-valid tasks: `30 / 30`

## Stage 2: Exact render

- Exact-render-valid tasks: `3 / 30`

## Stage 3: Runtime family validation

| Task | Intended | TargetRate | DominantRuntimeFamily | NativeLikeAllowRate | MixedProdHarmRate | ProtectCriticalRate | BoringMasteryRate | LoopCompleteRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed_v2_004 | MIXED_PROD_HARM_HEAVY | 0.6129 | MIXED_PROD_HARM | 0.0968 | 0.6129 | 0.0323 | 0.0000 | 0.0000 |
| mixed_v2_005 | MIXED_PROD_HARM_HEAVY | 0.4815 | MIXED_PROD_HARM | 0.2222 | 0.4815 | 0.0370 | 0.0000 | 0.0000 |
| mixed_v2_006 | MIXED_PROD_HARM_HEAVY | 0.6129 | MIXED_PROD_HARM | 0.0968 | 0.6129 | 0.0323 | 0.0000 | 0.0000 |
## Stage 4: Formal slice assembly

- Accepted tasks: `3`
- Rejected tasks: `27`

A formal slice is assembled only from tasks that are exact-render valid
and whose target family is actually present and dominant at runtime.
