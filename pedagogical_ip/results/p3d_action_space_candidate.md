# P3-D: Action-Space Canonical Candidate

**2-act (WAIT/WARN) vs 3-act (WAIT/SOFT/WARN)**

### Suite 1: Canonical Mixed-Family

| θ | tempt | Config | Success | Dose | Warn | Active | STOP Ag | Top-1 |
|:-:|:-----:|--------|:-------:|:----:|:----:|:------:|:------:|:-----:|
| safe | none | 3-act | 0.507 | 0.020 | 0.017 | 0.020 | 0.977 | 1 |
| safe | none | 2-act | 0.507 | 0.020 | 0.020 | 0.020 | 0.977 | 1 |
| shiny | none | 3-act | 0.510 | 0.033 | 0.033 | 0.033 | 0.970 | 1 |
| shiny | none | 2-act | 0.510 | 0.033 | 0.033 | 0.033 | 0.970 | 1 |

### Suite 2: Balanced Active

| θ | tempt | Config | Success | Dose | Warn | Active | STOP Ag | Top-1 |
|:-:|:-----:|--------|:-------:|:----:|:----:|:------:|:------:|:-----:|
| safe | none | 3-act | 0.500 | 0.100 | 0.070 | 0.100 | 0.710 | 1 |
| safe | none | 2-act | 0.500 | 0.090 | 0.090 | 0.090 | 0.727 | 1 |
| shiny | none | 3-act | 0.517 | 0.117 | 0.090 | 0.117 | 0.677 | 1 |
| shiny | none | 2-act | 0.517 | 0.110 | 0.110 | 0.110 | 0.717 | 1 |

### Suite 3: Hidden Temptation

| θ | tempt | Config | Success | Dose | Warn | Active | STOP Ag | Top-1 |
|:-:|:-----:|--------|:-------:|:----:|:----:|:------:|:------:|:-----:|
| safe | none | 3-act | 0.507 | 0.020 | 0.017 | 0.020 | 0.977 | 1 |
| safe | none | 2-act | 0.507 | 0.020 | 0.020 | 0.020 | 0.977 | 1 |
| safe | aligned=0.6 | 3-act | 0.770 | 0.020 | 0.017 | 0.020 | 0.977 | 1 |
| safe | aligned=0.6 | 2-act | 0.770 | 0.020 | 0.020 | 0.020 | 0.977 | 1 |
| safe | conflict=1.0 | 3-act | 0.903 | 0.020 | 0.017 | 0.020 | 0.990 | 1 |
| safe | conflict=1.0 | 2-act | 0.903 | 0.020 | 0.020 | 0.020 | 0.990 | 1 |
| shiny | none | 3-act | 0.510 | 0.033 | 0.033 | 0.033 | 0.970 | 1 |
| shiny | none | 2-act | 0.510 | 0.033 | 0.033 | 0.033 | 0.970 | 1 |
| shiny | aligned=0.6 | 3-act | 0.050 | 0.037 | 0.033 | 0.037 | 0.913 | 1 |
| shiny | aligned=0.6 | 2-act | 0.050 | 0.033 | 0.033 | 0.033 | 0.940 | 1 |
| shiny | conflict=1.0 | 3-act | 0.037 | 0.037 | 0.033 | 0.037 | 0.913 | 1 |
| shiny | conflict=1.0 | 2-act | 0.037 | 0.033 | 0.033 | 0.033 | 0.940 | 1 |

### Suite 4: 2-act vs 3-act Decision Disagreement Summary

**Total disagreements: 18/1200 (1.50%)**

## Verdict

> **18/1200 disagreements — investigate further.**
