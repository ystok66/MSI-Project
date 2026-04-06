# PP-MRB v2: Persistent-Profile Selective Fading

**Config**: 20 seeds × 30 episodes × 2 θ

## Exp A: Main Results

| θ | Strategy | SBCR | WarnRate | WR(wait_fav) | WR(warn_nec) | SelGap | Ent(1st) | Ent(2nd) |
|---|----------|------|---------|:------------:|:------------:|:------:|:--------:|:--------:|
| shiny | always_wait | 18% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| shiny | always_warn | 18% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| shiny | oracle_theta | 18% | 43% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| shiny | persistent_original | 18% | 89% | 75% | 100% | 0.248 | 0.7409 | 0.3112 |
| shiny | autonomy_only | 18% | 88% | 73% | 100% | 0.271 | 0.7409 | 0.3112 |
| shiny | gated_tempt_only | 18% | 93% | 85% | 100% | 0.154 | 0.7409 | 0.3112 |
| shiny | autonomy+gated_tempt | 18% | 90% | 78% | 100% | 0.223 | 0.7409 | 0.3112 |
| shiny | persistent_v2.1 | 18% | 85% | 64% | 100% | 0.360 | 0.7409 | 0.3112 |
| shiny | reset_v2.1 | 18% | 98% | 96% | 100% | 0.045 | 1.6094 | 1.6094 |
| shiny | v4_reset | 18% | 88% | 72% | 100% | 0.276 | 0.0000 | 0.0000 |
| safe | always_wait | 87% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| safe | always_warn | 87% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| safe | oracle_theta | 87% | 45% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| safe | persistent_original | 87% | 93% | 84% | 100% | 0.156 | 1.2835 | 0.9998 |
| safe | autonomy_only | 87% | 90% | 78% | 100% | 0.218 | 1.2835 | 0.9998 |
| safe | gated_tempt_only | 87% | 98% | 95% | 100% | 0.051 | 1.2835 | 0.9998 |
| safe | autonomy+gated_tempt | 87% | 96% | 90% | 100% | 0.099 | 1.2835 | 0.9998 |
| safe | persistent_v2.1 | 87% | 95% | 89% | 100% | 0.108 | 1.2835 | 0.9998 |
| safe | reset_v2.1 | 87% | 99% | 98% | 100% | 0.025 | 1.6094 | 1.6094 |
| safe | v4_reset | 87% | 89% | 74% | 100% | 0.263 | 0.0000 | 0.0000 |

## Exp B: Time-series WarnRate by Episode Bins


### θ = shiny


**wait_clean**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 78% | 64% | 78% | 0.004 |
| persistent_v2.1 | 74% | 47% | 48% | 0.256 |
| reset_v2.1 | 97% | 92% | 83% | 0.141 |
| oracle_theta | 0% | 0% | 0% | 0.000 |

**wait_lure**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 75% | 75% | 87% | -0.121 |
| persistent_v2.1 | 83% | 62% | 75% | 0.083 |
| reset_v2.1 | 100% | 100% | 100% | 0.000 |
| oracle_theta | 0% | 0% | 0% | 0.000 |

**warn_trap**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 100% | 100% | 100% | 0.000 |
| persistent_v2.1 | 100% | 100% | 100% | 0.000 |
| reset_v2.1 | 100% | 100% | 100% | 0.000 |
| oracle_theta | 100% | 100% | 100% | 0.000 |

### θ = safe


**wait_clean**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 80% | 75% | 76% | 0.040 |
| persistent_v2.1 | 91% | 76% | 86% | 0.048 |
| reset_v2.1 | 96% | 95% | 96% | 0.005 |
| oracle_theta | 0% | 0% | 0% | 0.000 |

**wait_lure**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 87% | 98% | 97% | -0.096 |
| persistent_v2.1 | 100% | 98% | 94% | 0.065 |
| reset_v2.1 | 100% | 100% | 100% | 0.000 |
| oracle_theta | 0% | 0% | 0% | 0.000 |

**warn_trap**

| Strategy | ep 1-10 | ep 11-20 | ep 21-30 | Δ(first-last) |
|----------|:-------:|:--------:|:--------:|:--------------:|
| persistent_original | 100% | 100% | 100% | 0.000 |
| persistent_v2.1 | 100% | 100% | 100% | 0.000 |
| reset_v2.1 | 100% | 100% | 100% | 0.000 |
| oracle_theta | 100% | 100% | 100% | 0.000 |

## Exp C: Actionability Audit

| θ | Subtype | Term | PCR |
|---|---------|------|-----|
| shiny | wait_clean | autonomy_bonus | 14.5% |
| shiny | wait_clean | tempt_risk_gated | 11.8% |
| shiny | wait_clean | missed_window_gated | 8.2% |
| shiny | warn_trap | autonomy_bonus | 0.0% |
| shiny | warn_trap | tempt_risk_gated | 0.0% |
| shiny | warn_trap | missed_window_gated | 0.0% |
| safe | wait_clean | autonomy_bonus | 8.1% |
| safe | wait_clean | tempt_risk_gated | 35.3% |
| safe | wait_clean | missed_window_gated | 11.0% |
| safe | warn_trap | autonomy_bonus | 0.0% |
| safe | warn_trap | tempt_risk_gated | 0.0% |
| safe | warn_trap | missed_window_gated | 0.0% |
