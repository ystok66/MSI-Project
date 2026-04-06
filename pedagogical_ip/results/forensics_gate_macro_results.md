# Active Divergence Forensics + Tie-Aware Gate

## Exp 1: Active Divergence Forensics

**Total divergent steps found: 4**

### Divergence Type Breakdown

| Oracle → Infer | Count | % |
|:---:|:---:|:---:|
| SOFT→WARN | 2 | 50.0% |
| WAIT→WARN | 2 | 50.0% |

### Margin on Divergent Steps

- Mean margin: 0.8370
- Median margin: 0.8370
- Min margin: 0.8140
- Max margin: 0.8600

### Q Difference on Divergent Steps

- Mean |ΔQ|: 0.0840
- Max |ΔQ|: 0.0880

### m̂ Error on Divergent Steps

- Mean Δτ: 0.040637
- Mean Δν: 0.115515
- Mean Δγ: 0.183842

### Family Distribution of Divergences

| Family | Count |
|--------|:-----:|
| tic_rescue_heavy | 4 |

## Exp 2: Tie-Aware Gate Comparison

| Gate | Regime | θ | Div All | Div@Active | Div@Hard | n_div |
|------|--------|:-:|:-------:|:----------:|:--------:|:-----:|
| raw | natural | safe | 0.0000 | 0.0000 | 0.0000 | 0 |
| raw | natural | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |
| raw | active_0.5 | safe | 0.0083 | 0.5000 | 0.0333 | 2 |
| raw | active_0.5 | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |
| wait_gate | natural | safe | 0.0125 | 0.7500 | 0.0500 | 3 |
| wait_gate | natural | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |
| wait_gate | active_0.5 | safe | 0.0125 | 0.7500 | 0.0500 | 3 |
| wait_gate | active_0.5 | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |
| soft_gate | natural | safe | 0.0125 | 0.7500 | 0.0500 | 3 |
| soft_gate | natural | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |
| soft_gate | active_0.5 | safe | 0.0083 | 0.5000 | 0.0333 | 2 |
| soft_gate | active_0.5 | shiny | 0.0000 | 0.0000 | 0.0000 | 0 |

## Exp 3: Macro Lesson Ranking Replay

| α | θ | Top-1 Agree | Kendall τ | Spearman ρ |
|:-:|:-:|:-----------:|:---------:|:----------:|
| 0.0 | safe | 1.000 | 1.0000 | 1.0000 |
| 0.0 | shiny | 1.000 | 1.0000 | 1.0000 |
| 0.5 | safe | 1.000 | 0.9969 | 0.9988 |
| 0.5 | shiny | 1.000 | 0.9969 | 0.9988 |
| 1.0 | safe | 1.000 | 0.9984 | 0.9994 |
| 1.0 | shiny | 1.000 | 0.9984 | 0.9994 |

## Exp 4: Aligned vs Conflicting Temptation

| Variant | θ | Tempt | Corr_ν | MAE_ν | Div All |
|---------|:-:|:-----:|:------:|:-----:|:-------:|
| Aligned | safe | 0.0 | 0.8768 | 0.0145 | 0.0000 |
| Aligned | shiny | 0.8 | 0.9970 | 0.0012 | 0.0000 |
| Conflicting | safe | 0.8 | 0.8949 | 0.0147 | 0.0000 |
| Conflicting | shiny | 0.0 | 0.9962 | 0.0065 | 0.0000 |
