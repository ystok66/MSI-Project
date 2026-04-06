# P4-A.2: γ_spec Semantic Audit

## Audit 1: State Semantic — γ̂_spec vs Resist Rate

| θ | tempt | γ̂_spec(final) | r_resist | Corr(γ̂_spec, r_resist) | ν̂(final) | Δν̂ |
|:-:|:-----:|:-------------:|:--------:|:---------------------:|:--------:|:---:|
| safe | 0.0 | 0.1683 | 0.5067 | 0.9320 | 0.0669 | 0.0000 |
| safe | 0.3 | 0.2631 | 0.5900 | 0.9645 | 0.0582 | -0.0088 |
| safe | 0.6 | 0.3956 | 0.7367 | 0.9875 | 0.0596 | -0.0073 |
| safe | 1.0 | 0.5725 | 0.8900 | 0.9751 | 0.0589 | -0.0080 |

**θ=safe overall: Corr(γ̂_spec, r_resist) = 0.9568 (p=0.0000)**

| shiny | 0.0 | 0.1643 | 0.5100 | 0.9882 | 0.0603 | 0.0000 |
| shiny | 0.3 | 0.0962 | 0.2100 | 0.9718 | 0.0937 | 0.0334 |
| shiny | 0.6 | 0.0362 | 0.0633 | 0.9580 | 0.0997 | 0.0394 |
| shiny | 1.0 | 0.0377 | 0.0533 | 0.9160 | 0.1002 | 0.0400 |

**θ=shiny overall: Corr(γ̂_spec, r_resist) = 0.9623 (p=0.0000)**

## Audit 2: Trait Semantic — Fixed Tempt, Sweep γ_spec_init

| θ | tempt | γ_spec_init | γ̂_spec(final) | γ_spec(true,final) | r_resist |
|:-:|:-----:|:-----------:|:-------------:|:------------------:|:--------:|
| safe | 0.3 | 0.1 | 0.2631 | 0.6912 | 0.5900 |
| safe | 0.3 | 0.3 | 0.2631 | 0.7000 | 0.5900 |
| safe | 0.3 | 0.5 | 0.2631 | 0.7000 | 0.5900 |
| safe | 0.3 | 0.7 | 0.2614 | 0.7000 | 0.5867 |

**θ=safe, tempt=0.3: Corr(γ̂_spec, γ_spec_true) = -0.3680 (p=0.0038)**

| safe | 0.6 | 0.1 | 0.3966 | 0.6131 | 0.7400 |
| safe | 0.6 | 0.3 | 0.3979 | 0.6580 | 0.7433 |
| safe | 0.6 | 0.5 | 0.3990 | 0.6986 | 0.7467 |
| safe | 0.6 | 0.7 | 0.4012 | 0.7000 | 0.7533 |

**θ=safe, tempt=0.6: Corr(γ̂_spec, γ_spec_true) = -0.4404 (p=0.0004)**


## Verdict

> **γ̂_spec is best interpreted as a BEHAVIORAL STATE** (correlated with conditional resist rate) rather than a trait-like latent (weakly correlated with true γ_spec).

**Semantic label: `gamma_spec_state`**
