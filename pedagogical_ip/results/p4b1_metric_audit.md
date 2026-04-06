# P4-B.1: Metric Integrity Audit

## Audit 1: Old vs New Active Mask

**Old active**: step where oracle warned (dose > 0)
**New active**: step where oracle OR infer is not WAIT

| Suite | θ | tempt | DivAll | Div@OldAct | Div@NewAct | n_old | n_new | R_active |
|-------|:-:|:-----:|:------:|:----------:|:----------:|:----:|:-----:|:--------:|
| Canonical | safe | none | 0.0133 | 0.0000 | 0.2667 | 11 | 15 | 0.0307 |
| Canonical | shiny | none | 0.0067 | 0.0000 | 0.2222 | 7 | 9 | 0.0219 |
| Active | safe | none | 0.0933 | 0.0000 | 0.4912 | 29 | 57 | 0.0972 |
| Active | shiny | none | 0.0767 | 0.0000 | 0.5111 | 22 | 45 | 0.0864 |
| Tempt | safe | none | 0.0133 | 0.0000 | 0.2667 | 11 | 15 | 0.0307 |
| Tempt | safe | al=0.6 | 0.0000 | 0.0000 | 0.0000 | 15 | 15 | 0.0000 |
| Tempt | safe | cf=1.0 | 0.0000 | 0.0000 | 0.0000 | 15 | 15 | 0.0000 |
| Tempt | shiny | none | 0.0067 | 0.0000 | 0.2222 | 7 | 9 | 0.0219 |
| Tempt | shiny | al=0.6 | 0.0100 | 0.0000 | 0.3333 | 6 | 9 | 0.0420 |
| Tempt | shiny | cf=1.0 | 0.0100 | 0.0000 | 0.3333 | 6 | 9 | 0.0418 |

## Audit 2: Divergence Forensics

**Total divergences: 6 across 600 steps**

| θ | Family | Step | Oracle | Infer | Active(old) | Active(new) |
|:-:|--------|:----:|:------:|:-----:|:-----------:|:-----------:|
| safe | tic_rescue_heavy | 15 | WAIT | WARN | ❌ | ✅ |
| safe | tic_rescue_heavy | 15 | WAIT | WARN | ❌ | ✅ |
| safe | blind_activation_corridor | 12 | WAIT | WARN | ❌ | ✅ |
| safe | blind_activation_corridor | 12 | WAIT | WARN | ❌ | ✅ |
| shiny | tic_rescue_heavy | 15 | WAIT | WARN | ❌ | ✅ |
| shiny | warn_symmetric_rescue | 10 | WAIT | WARN | ❌ | ✅ |

## Verdict

> **6 divergences were OLD-active=False but NEW-active=True.**
> These are steps where the infer-only tutor chose WARN but oracle chose WAIT.
> The old `active` mask (oracle-warned-only) missed them. **NEW mask is the correct definition.**
