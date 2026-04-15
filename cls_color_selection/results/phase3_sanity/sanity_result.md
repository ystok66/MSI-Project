# Phase 3 Sanity Check: task 000001, seed 42


## no_tutor (status: ok)
- shadow_fidelity: none

### Teach Metrics
- TeachSuccessRate: 0.2
- TeachDeathRate: 0.6
- TeachTimeoutRate: 0.2
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 15.6
- TeachDangerSelectCount: 0.6
- TeachStuckRetryRate: 0.2
- TeachN: 5

### Eval Metrics
- EvalSuccessRate: 0.0
- EvalDeathRate: 0.8
- EvalTimeoutRate: 0.2
- EvalConfirmMean@Success: 0.0
- EvalRetryMean: 15.6
- EvalDangerSelectCount: 0.8
- EvalStuckRetryRate: 0.2
- EvalN: 5

### Belief Summary
- B_sem_a_probe: 0.2857142857142857
- B_sem_e_beam: 0.0
- B_risk_detect: 0.2
- B_risk_overavoid: 0.2222222222222222
- B_type_map: balanced
- B_type_posterior: [np.float64(0.3333333333333333), np.float64(0.3333333333333333), np.float64(0.3333333333333333)]
- n_warnings: 0
- n_hints: 0
- n_courage: 0

## T0_rule (status: ok)
- shadow_fidelity: none

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
- EvalDeathRate: 0.75
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.5
- EvalDangerSelectCount: 0.75
- EvalStuckRetryRate: 0.0
- EvalN: 4

### Belief Summary
- B_sem_a_probe: 0.5
- B_sem_e_beam: 2.0419179073770444
- B_risk_detect: 0.7777777777777778
- B_risk_overavoid: 0.9342105263157895
- B_type_map: balanced
- B_type_posterior: [np.float64(0.3333333333333333), np.float64(0.3333333333333333), np.float64(0.3333333333333333)]
- n_warnings: 8
- n_hints: 0
- n_courage: 0

## T1_proxy (status: ok)
- shadow_fidelity: none

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
- EvalDeathRate: 0.75
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.5
- EvalDangerSelectCount: 0.75
- EvalStuckRetryRate: 0.0
- EvalN: 4

### Belief Summary
- B_sem_a_probe: 0.5
- B_sem_e_beam: 2.0419179073770444
- B_risk_detect: 0.7777777777777778
- B_risk_overavoid: 0.9342105263157895
- B_type_map: balanced
- B_type_posterior: [np.float64(0.3333333333333333), np.float64(0.3333333333333333), np.float64(0.3333333333333333)]
- n_warnings: 8
- n_hints: 0
- n_courage: 1

## T2_exact (status: ok)
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
- EvalDeathRate: 0.75
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.5
- EvalDangerSelectCount: 0.75
- EvalStuckRetryRate: 0.0
- EvalN: 4

### Joint Debug
- n_divergence_records: 4
- n_counterfactual_records: 0
- D_gram_top1_agreement: 1.0
- D_gram_beam_entropy_gap: 0.0
- D_risk_l1: 0.0
- CF_mean_error: 0.0
- CF_abs_error: 0.0

### Belief Summary
- B_sem_a_probe: 0.5
- B_sem_e_beam: 2.0419179073770444
- B_risk_detect: 0.7777777777777778
- B_risk_overavoid: 0.9342105263157895
- B_type_map: balanced
- B_type_posterior: [np.float64(0.3333333333333333), np.float64(0.3333333333333333), np.float64(0.3333333333333333)]
- n_warnings: 8
- n_hints: 0
- n_courage: 1

## T2_compressed (status: ok)
- shadow_fidelity: compressed

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
- EvalDeathRate: 0.75
- EvalTimeoutRate: 0.0
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 0.5
- EvalDangerSelectCount: 0.75
- EvalStuckRetryRate: 0.0
- EvalN: 4

### Joint Debug
- n_divergence_records: 4
- n_counterfactual_records: 0
- D_gram_top1_agreement: 1.0
- D_gram_beam_entropy_gap: 0.0
- D_risk_l1: 0.0
- CF_mean_error: 0.0
- CF_abs_error: 0.0

### Belief Summary
- B_sem_a_probe: 0.5
- B_sem_e_beam: 2.0419179073770444
- B_risk_detect: 0.7777777777777778
- B_risk_overavoid: 0.9342105263157895
- B_type_map: balanced
- B_type_posterior: [np.float64(0.3333333333333333), np.float64(0.3333333333333333), np.float64(0.3333333333333333)]
- n_warnings: 8
- n_hints: 0
- n_courage: 1