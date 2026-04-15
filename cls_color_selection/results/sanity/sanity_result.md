# Sanity Check: Single Episode (task 000001, seed 42)

**Status**: unknown

## Teach Metrics
- TeachSuccessRate: 0.375
- TeachDeathRate: 0.0
- TeachTimeoutRate: 0.625
- TeachConfirmMean@Success: 1.0
- TeachRetryMean: 50.75
- TeachDangerSelectCount: 1.5
- TeachStuckRetryRate: 0.625
- TeachN: 8

## Eval Metrics
- EvalSuccessRate: 0.5
- EvalDeathRate: 0.0
- EvalTimeoutRate: 0.5
- EvalConfirmMean@Success: 1.0
- EvalRetryMean: 41.5
- EvalDangerSelectCount: 0.0
- EvalStuckRetryRate: 0.5
- EvalN: 2

## Teach Details
  - Q0: TIMEOUT (confirms=5, retries=78, danger_sel=1)
  - Q1: SUCCESS (confirms=1, retries=2, danger_sel=0)
  - Q2: SUCCESS (confirms=1, retries=1, danger_sel=0)
  - Q3: TIMEOUT (confirms=5, retries=77, danger_sel=0)
  - Q4: TIMEOUT (confirms=5, retries=85, danger_sel=6)
  - Q5: SUCCESS (confirms=1, retries=2, danger_sel=0)
  - Q6: TIMEOUT (confirms=5, retries=80, danger_sel=2)
  - Q7: TIMEOUT (confirms=5, retries=81, danger_sel=3)

## Eval Details
  - Q8: SUCCESS (confirms=1, retries=2, danger_sel=0)
  - Q9: TIMEOUT (confirms=5, retries=81, danger_sel=0)