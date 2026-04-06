# PP-MRB: Persistent-Profile Mixed-Reveal Branches

**Config**: 8 sessions × 12 episodes × 4 θ × 6 strategies

## Main Results

| θ | Strategy | SBCR | WarnRate | WR(wait_clean) | WR(warn_trap) | SelGap | Ent(1st) | Ent(2nd) |
|---|----------|------|---------|----------------|---------------|--------|----------|----------|
| safe | always_wait | 93% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| safe | always_warn | 93% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| safe | v4_reset | 93% | 84% | 64% | 100% | 0.365 | 0.0000 | 0.0000 |
| safe | pref_v2_persistent | 93% | 97% | 92% | 100% | 0.083 | 1.3630 | 1.1200 |
| safe | oracle_theta | 93% | 30% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| safe | wrong_memory | 93% | 97% | 92% | 100% | 0.083 | 1.2040 | 1.1080 |
| shiny | always_wait | 25% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| shiny | always_warn | 25% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| shiny | v4_reset | 25% | 95% | 92% | 100% | 0.048 | 0.0000 | 0.0000 |
| shiny | pref_v2_persistent | 25% | 100% | 100% | 100% | 0.000 | 1.1510 | 0.5420 |
| shiny | oracle_theta | 25% | 29% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| shiny | wrong_memory | 25% | 100% | 100% | 100% | 0.000 | 0.8370 | 0.5870 |
| shortcut | always_wait | 79% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| shortcut | always_warn | 79% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| shortcut | v4_reset | 79% | 81% | 52% | 100% | 0.479 | 0.0000 | 0.0000 |
| shortcut | pref_v2_persistent | 79% | 97% | 88% | 100% | 0.125 | 1.3920 | 1.2850 |
| shortcut | oracle_theta | 79% | 29% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| shortcut | wrong_memory | 79% | 96% | 83% | 100% | 0.167 | 1.1150 | 1.2610 |
| neutral | always_wait | 79% | 0% | 0% | 0% | 0.000 | 0.0000 | 0.0000 |
| neutral | always_warn | 79% | 100% | 100% | 100% | 0.000 | 0.0000 | 0.0000 |
| neutral | v4_reset | 79% | 89% | 63% | 100% | 0.367 | 0.0000 | 0.0000 |
| neutral | pref_v2_persistent | 79% | 98% | 93% | 100% | 0.067 | 1.4080 | 1.3350 |
| neutral | oracle_theta | 79% | 33% | 0% | 100% | 1.000 | 0.0000 | 0.0000 |
| neutral | wrong_memory | 79% | 98% | 93% | 100% | 0.067 | 1.0680 | 1.3430 |

## Persistent vs Reset Comparison

| θ | Metric | v4_reset | persistent | oracle | wrong_mem |
|---|--------|----------|------------|--------|-----------|
| safe | sbcr | 93% | 93% | 93% | 93% |
| safe | warn_rate | 84% | 97% | 30% | 97% |
| safe | sel_gap | 0.365 | 0.083 | 1.000 | 0.083 |
| shiny | sbcr | 25% | 25% | 25% | 25% |
| shiny | warn_rate | 95% | 100% | 29% | 100% |
| shiny | sel_gap | 0.048 | 0.000 | 1.000 | 0.000 |
| shortcut | sbcr | 79% | 79% | 79% | 79% |
| shortcut | warn_rate | 81% | 97% | 29% | 96% |
| shortcut | sel_gap | 0.479 | 0.125 | 1.000 | 0.167 |
| neutral | sbcr | 79% | 79% | 79% | 79% |
| neutral | warn_rate | 89% | 98% | 33% | 98% |
| neutral | sel_gap | 0.367 | 0.067 | 1.000 | 0.067 |

## Entropy Decay (persistent conditions only)

| θ | Strategy | Ent(episodes 1-6) | Ent(episodes 7-12) | Δ |
|---|----------|-------------------|--------------------|---------|
| safe | pref_v2_persistent | 1.3630 | 1.1200 | 0.2430 |
| safe | wrong_memory | 1.2040 | 1.1080 | 0.0960 |
| shiny | pref_v2_persistent | 1.1510 | 0.5420 | 0.6090 |
| shiny | wrong_memory | 0.8370 | 0.5870 | 0.2500 |
| shortcut | pref_v2_persistent | 1.3920 | 1.2850 | 0.1070 |
| shortcut | wrong_memory | 1.1150 | 1.2610 | -0.1460 |
| neutral | pref_v2_persistent | 1.4080 | 1.3350 | 0.0730 |
| neutral | wrong_memory | 1.0680 | 1.3430 | -0.2750 |
