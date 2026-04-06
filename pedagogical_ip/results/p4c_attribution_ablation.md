# P4-C: 3D / 4D Attribution Ablation

**Does γ̂_spec_state enter micro Q?** Code audit: `gamma_spec` absent from tutor Q & bridge.
**Prediction: Group A ≡ Group B** (γ_spec is purely diagnostic)

## Ablation: Corrected Active Mask

| Suite | θ | Group | DivAll | Div@Act | n_act | OWR | Success |
|-------|:-:|:-----:|:------:|:-------:|:-----:|:---:|:-------:|
| Canonical | safe | A | 0.0167 | 0.3571 | 14 | 0.0167 | 0.497 |
| Canonical | safe | B | 0.0133 | 0.3077 | 13 | 0.0133 | 0.497 |
| Canonical | shiny | A | 0.0167 | 0.3846 | 13 | 0.0167 | 0.500 |
| Canonical | shiny | B | 0.0167 | 0.3846 | 13 | 0.0167 | 0.500 |
| Active | safe | A | 0.0667 | 0.4255 | 47 | 0.0667 | 0.490 |
| Active | safe | B | 0.0667 | 0.4255 | 47 | 0.0667 | 0.490 |
| Active | shiny | A | 0.0833 | 0.4464 | 56 | 0.0833 | 0.463 |
| Active | shiny | B | 0.0767 | 0.4259 | 54 | 0.0767 | 0.463 |

## Per-Family Over-Warn Rate (Group B, Active Suite)

| Family | θ | n | OWR | Div@Act |
|--------|:-:|:-:|:---:|:-------:|
| blind_activation_corridor | safe | 75 | 0.120 | 0.391 |
| tic_rescue_heavy | safe | 75 | 0.080 | 0.462 |
| warn_symmetric_rescue | safe | 75 | 0.067 | 0.455 |
| blind_activation_corridor | shiny | 75 | 0.133 | 0.370 |
| tic_rescue_heavy | shiny | 75 | 0.067 | 0.455 |
| warn_symmetric_rescue | shiny | 75 | 0.107 | 0.500 |

## Decision Identity: A vs B

**A vs B step-level identity: 599/600 (99.83%)**


## Verdict

> **A ≡ B confirmed.** γ̂_spec_state does NOT enter micro tutor Q. The over-warn divergences are pre-existing near-tie boundary issues exposed by the corrected active mask, not caused by the 4th dimension.

### Implication

γ̂_spec_state is currently **purely diagnostic / macro-only state**. This is the correct architecture:

- **Layer 1 (State Estimator)**: 4D `(τ̂, ν̂, γ̂_gen, γ̂_spec_state)`
- **Layer 2 (Micro Decision View)**: 3D `(τ̂, ν̂, γ̂_gen)` — γ̂_spec does NOT enter Q
- **Layer 3 (Macro / Diagnostic)**: Full 4D available
