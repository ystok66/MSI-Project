# Phase 2 Sanity Check: task 000001, seed 42


## no_tutor (status: ok)

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

### Teach Details
  - Q0: DEATH (confirms=0, retries=0)
  - Q1: SUCCESS (confirms=1, retries=1)
  - Q2: DEATH (confirms=0, retries=0)
  - Q3: DEATH (confirms=0, retries=0)
  - Q4: TIMEOUT (confirms=5, retries=77)

## tutor_rule (status: ok)

### Observation Summary
- ObsN: 2
- ObsSuccessRate: 0.5
- ObsDeathRate: 0.0
- ObsTimeoutRate: 0.5
- ObsMeanConfirms: 3.0
- ObsMeanRetries: 40.0
- ObsMeanDangerSelects: 0.5
- ObsMeanBeamEntropy: 2.0419179073770444
- ObsCounterfactualDeaths: 1
- ObsStuckRetries: 68

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

### Teach Details
  - Q2: SUCCESS (confirms=1, retries=1)
  - Q3: TIMEOUT (confirms=5, retries=77)
  - Q4: TIMEOUT (confirms=5, retries=86)
  - Q5: SUCCESS (confirms=1, retries=1)

## tutor_proxy (status: ok)

### Observation Summary
- ObsN: 2
- ObsSuccessRate: 0.5
- ObsDeathRate: 0.0
- ObsTimeoutRate: 0.5
- ObsMeanConfirms: 3.0
- ObsMeanRetries: 40.0
- ObsMeanDangerSelects: 0.5
- ObsMeanBeamEntropy: 2.0419179073770444
- ObsCounterfactualDeaths: 1
- ObsStuckRetries: 68

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

### Teach Details
  - Q2: SUCCESS (confirms=1, retries=1)
  - Q3: TIMEOUT (confirms=5, retries=77)
  - Q4: TIMEOUT (confirms=5, retries=86)
  - Q5: SUCCESS (confirms=1, retries=1)