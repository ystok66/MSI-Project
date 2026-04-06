# Stage-2: Joint Tutor on CGC-v2

**Config**: 16 seeds × 30 episodes × 4 conditions

## Exp A: Main Results

| Condition | Strategy | WarnRate | WR(aligned) | WR(conflict) | SelGap | SBCR |
|-----------|----------|---------|:-----------:|:------------:|:------:|:----:|
| shiny-aligned | always_wait | 0% | 0% | 0% | 0.000 | 38% |
| shiny-aligned | always_warn | 100% | 100% | 100% | 0.000 | 38% |
| shiny-aligned | oracle | 29% | 0% | 100% | 1.000 | 38% |
| shiny-aligned | pref_only | 82% | 66% | 100% | 0.338 | 38% |
| shiny-aligned | coupled_v1 | 95% | 91% | 100% | 0.093 | 38% |
| shiny-aligned | coupled_v2 | 90% | 82% | 100% | 0.183 | 38% |
| safe-conflict | always_wait | 0% | 0% | 0% | 0.000 | 87% |
| safe-conflict | always_warn | 100% | 100% | 100% | 0.000 | 87% |
| safe-conflict | oracle | 63% | 0% | 100% | 1.000 | 87% |
| safe-conflict | pref_only | 98% | 79% | 100% | 0.207 | 87% |
| safe-conflict | coupled_v1 | 99% | 96% | 100% | 0.036 | 87% |
| safe-conflict | coupled_v2 | 98% | 85% | 100% | 0.154 | 87% |
| shiny-composite | always_wait | 0% | 0% | 0% | 0.000 | 34% |
| shiny-composite | always_warn | 100% | 100% | 100% | 0.000 | 34% |
| shiny-composite | oracle | 32% | 0% | 100% | 1.000 | 34% |
| shiny-composite | pref_only | 78% | 57% | 100% | 0.429 | 34% |
| shiny-composite | coupled_v1 | 95% | 90% | 100% | 0.099 | 34% |
| shiny-composite | coupled_v2 | 90% | 81% | 100% | 0.194 | 34% |
| safe-aligned-comp | always_wait | 0% | 0% | 0% | 0.000 | 84% |
| safe-aligned-comp | always_warn | 100% | 100% | 100% | 0.000 | 84% |
| safe-aligned-comp | oracle | 32% | 0% | 100% | 1.000 | 84% |
| safe-aligned-comp | pref_only | 95% | 88% | 100% | 0.118 | 84% |
| safe-aligned-comp | coupled_v1 | 97% | 94% | 100% | 0.060 | 84% |
| safe-aligned-comp | coupled_v2 | 95% | 89% | 100% | 0.108 | 84% |

## Exp B: Time-series Joint Confidence


### shiny-aligned

| Strategy | c_t(1-10) | c_t(11-20) | c_t(21-30) | WR(1-10) | WR(21-30) | ΔWR |
|----------|:---------:|:----------:|:----------:|:--------:|:---------:|:---:|
| coupled_v1 | 0.000 | 0.000 | 0.000 | 94% | 95% | -0.012 |
| coupled_v2 | 0.136 | 0.244 | 0.275 | 91% | 91% | 0.006 |
| pref_only | 0.260 | 0.581 | 0.688 | 89% | 78% | 0.107 |

### safe-conflict

| Strategy | c_t(1-10) | c_t(11-20) | c_t(21-30) | WR(1-10) | WR(21-30) | ΔWR |
|----------|:---------:|:----------:|:----------:|:--------:|:---------:|:---:|
| coupled_v1 | 0.000 | 0.000 | 0.000 | 100% | 98% | 0.019 |
| coupled_v2 | 0.160 | 0.232 | 0.255 | 99% | 99% | 0.000 |
| pref_only | 0.179 | 0.331 | 0.420 | 99% | 98% | 0.013 |

## Exp C: Actionability Audit (coupled_v2 only)

| Condition | Subtype | Term | PCR |
|-----------|---------|------|-----|
| shiny-aligned | goal_aligned | autonomy_bonus | 9.2% |
| shiny-aligned | goal_aligned | v_obs_gated | 2.0% |
| shiny-aligned | goal_aligned | tempt_risk_gated | 28.5% |
| shiny-aligned | goal_aligned | missed_window_gated | 7.6% |
| shiny-aligned | goal_conflict | autonomy_bonus | 0.0% |
| shiny-aligned | goal_conflict | v_obs_gated | 0.0% |
| shiny-aligned | goal_conflict | tempt_risk_gated | 0.0% |
| shiny-aligned | goal_conflict | missed_window_gated | 0.0% |
| safe-conflict | goal_aligned | autonomy_bonus | 5.2% |
| safe-conflict | goal_aligned | v_obs_gated | 0.0% |
| safe-conflict | goal_aligned | tempt_risk_gated | 39.0% |
| safe-conflict | goal_aligned | missed_window_gated | 7.8% |
| safe-conflict | goal_conflict | autonomy_bonus | 0.0% |
| safe-conflict | goal_conflict | v_obs_gated | 0.0% |
| safe-conflict | goal_conflict | tempt_risk_gated | 0.0% |
| safe-conflict | goal_conflict | missed_window_gated | 0.0% |
| shiny-composite | goal_aligned | autonomy_bonus | 10.2% |
| shiny-composite | goal_aligned | v_obs_gated | 4.5% |
| shiny-composite | goal_aligned | tempt_risk_gated | 29.5% |
| shiny-composite | goal_aligned | missed_window_gated | 11.1% |
| shiny-composite | goal_conflict | autonomy_bonus | 0.0% |
| shiny-composite | goal_conflict | v_obs_gated | 0.0% |
| shiny-composite | goal_conflict | tempt_risk_gated | 0.0% |
| shiny-composite | goal_conflict | missed_window_gated | 0.0% |
| safe-aligned-comp | goal_aligned | autonomy_bonus | 5.5% |
| safe-aligned-comp | goal_aligned | v_obs_gated | 0.5% |
| safe-aligned-comp | goal_aligned | tempt_risk_gated | 38.4% |
| safe-aligned-comp | goal_aligned | missed_window_gated | 11.9% |
| safe-aligned-comp | goal_conflict | autonomy_bonus | 0.0% |
| safe-aligned-comp | goal_conflict | v_obs_gated | 0.0% |
| safe-aligned-comp | goal_conflict | tempt_risk_gated | 0.0% |
| safe-aligned-comp | goal_conflict | missed_window_gated | 0.0% |

## Exp D: Coupled v1 vs v2 Differential

| Condition | Metric | coupled_v1 | coupled_v2 | Δ |
|-----------|--------|:----------:|:----------:|:--:|
| shiny-aligned | SelGap | 0.093 | 0.183 | +0.090 |
| shiny-aligned | WR(aligned) | 0.907 | 0.817 | -0.090 |
| shiny-aligned | WR(conflict) | 1.000 | 1.000 | +0.000 |
| safe-conflict | SelGap | 0.036 | 0.154 | +0.118 |
| safe-conflict | WR(aligned) | 0.964 | 0.846 | -0.118 |
| safe-conflict | WR(conflict) | 1.000 | 1.000 | +0.000 |
| shiny-composite | SelGap | 0.099 | 0.194 | +0.095 |
| shiny-composite | WR(aligned) | 0.901 | 0.806 | -0.095 |
| shiny-composite | WR(conflict) | 1.000 | 1.000 | +0.000 |
| safe-aligned-comp | SelGap | 0.060 | 0.108 | +0.048 |
| safe-aligned-comp | WR(aligned) | 0.940 | 0.892 | -0.048 |
| safe-aligned-comp | WR(conflict) | 1.000 | 1.000 | +0.000 |
