# Phase 3 Sanity Check (FIXED): task 000001, seed 42


## no_tutor_generated (status: ok)
- query_source: generated
- n_obs=0, n_teach=8, n_eval=8
- shadow_fidelity: none

### Teach Metrics
- TeachSuccessRate: 0.625
- TeachDeathRate: 0.375
- TeachTimeoutRate: 0.0
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 1.25
- TeachDangerSelectCount: 0.375
- TeachStuckRetryRate: 0.0
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.375
- EvalDeathRate: 0.5
- EvalTimeoutRate: 0.125
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 10.25
- EvalDangerSelectCount: 0.5
- EvalStuckRetryRate: 0.125
- EvalN: 8

## T0_generated (status: ok)
- query_source: generated
- n_obs=4, n_teach=8, n_eval=8
- shadow_fidelity: none

### Teach Metrics
- TeachSuccessRate: 0.75
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.25
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 22.625
- TeachDangerSelectCount: 1.875
- TeachStuckRetryRate: 0.25
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.5
- EvalDeathRate: 0.5
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 1.0
- EvalDangerSelectCount: 0.5
- EvalStuckRetryRate: 0.0
- EvalN: 8

## T1_generated (status: ok)
- query_source: generated
- n_obs=4, n_teach=8, n_eval=8
- shadow_fidelity: none

### Teach Metrics
- TeachSuccessRate: 0.75
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.25
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 22.625
- TeachDangerSelectCount: 1.875
- TeachStuckRetryRate: 0.25
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.5
- EvalDeathRate: 0.5
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 1.0
- EvalDangerSelectCount: 0.5
- EvalStuckRetryRate: 0.0
- EvalN: 8

## T2_generated (status: ok)
- query_source: generated
- n_obs=4, n_teach=8, n_eval=8
- shadow_fidelity: exact

### Teach Metrics
- TeachSuccessRate: 0.75
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.25
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 22.625
- TeachDangerSelectCount: 1.875
- TeachStuckRetryRate: 0.25
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.375
- EvalDeathRate: 0.625
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.875
- EvalDangerSelectCount: 0.625
- EvalStuckRetryRate: 0.0
- EvalN: 8

### Joint Debug (DIVERGENCE)
- **n_divergence_records**: 9
- **n_counterfactual_records**: 0
- **D_gram_top1_agreement**: 1.0
- **D_gram_beam_entropy_gap**: 7.270414876805175e-05
- **D_risk_l1**: 0.0
- **CF_mean_error**: 0.0
- **CF_abs_error**: 0.0
- **D_gram_JS**: 7.270414876805175e-05
- **D_param_role_l1**: 9.868649107779169e-17
- **D_param_emit_l1**: 1.3158198810372225e-16

## T2_txt_only (status: ok)
- query_source: txt_only
- n_obs=2, n_teach=4, n_eval=4
- shadow_fidelity: exact

### Teach Metrics
- TeachSuccessRate: 0.5
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.5
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 41.25
- TeachDangerSelectCount: 1.75
- TeachStuckRetryRate: 0.5
- TeachN: 4

### Eval Metrics
- EvalSuccessRate: 0.25
- EvalDeathRate: 0.5
- EvalTimeoutRate: 0.25
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 20.0
- EvalDangerSelectCount: 0.5
- EvalStuckRetryRate: 0.25
- EvalN: 4

### Joint Debug (DIVERGENCE)
- **n_divergence_records**: 5
- **n_counterfactual_records**: 0
- **D_gram_top1_agreement**: 1.0
- **D_gram_beam_entropy_gap**: 0.0009885739664601922
- **D_risk_l1**: 0.0
- **CF_mean_error**: 0.0
- **CF_abs_error**: 0.0
- **D_gram_JS**: 0.0009885739664601922
- **D_param_role_l1**: 0.0035506959409954643
- **D_param_emit_l1**: 0.00178879135916902

## T2_exact (status: ok)
- query_source: generated
- n_obs=4, n_teach=8, n_eval=8
- shadow_fidelity: exact

### Teach Metrics
- TeachSuccessRate: 0.75
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.25
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 22.625
- TeachDangerSelectCount: 1.875
- TeachStuckRetryRate: 0.25
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.375
- EvalDeathRate: 0.625
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.875
- EvalDangerSelectCount: 0.625
- EvalStuckRetryRate: 0.0
- EvalN: 8

### Joint Debug (DIVERGENCE)
- **n_divergence_records**: 9
- **n_counterfactual_records**: 0
- **D_gram_top1_agreement**: 1.0
- **D_gram_beam_entropy_gap**: 7.270414876805175e-05
- **D_risk_l1**: 0.0
- **CF_mean_error**: 0.0
- **CF_abs_error**: 0.0
- **D_gram_JS**: 7.270414876805175e-05
- **D_param_role_l1**: 9.868649107779169e-17
- **D_param_emit_l1**: 1.3158198810372225e-16

## T2_compressed (status: ok)
- query_source: generated
- n_obs=4, n_teach=8, n_eval=8
- shadow_fidelity: compressed

### Teach Metrics
- TeachSuccessRate: 0.75
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.25
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 22.625
- TeachDangerSelectCount: 1.875
- TeachStuckRetryRate: 0.25
- TeachN: 8

### Eval Metrics
- EvalSuccessRate: 0.375
- EvalDeathRate: 0.625
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.875
- EvalDangerSelectCount: 0.625
- EvalStuckRetryRate: 0.0
- EvalN: 8

### Joint Debug (DIVERGENCE)
- **n_divergence_records**: 9
- **n_counterfactual_records**: 0
- **D_gram_top1_agreement**: 1.0
- **D_gram_beam_entropy_gap**: 7.270414876805175e-05
- **D_risk_l1**: 0.0
- **CF_mean_error**: 0.0
- **CF_abs_error**: 0.0
- **D_gram_JS**: 7.270414876805175e-05
- **D_param_role_l1**: 9.868649107779169e-17
- **D_param_emit_l1**: 1.3158198810372225e-16