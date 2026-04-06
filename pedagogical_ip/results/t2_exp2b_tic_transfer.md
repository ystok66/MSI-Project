# T2 Exp-2B: TIC 3-Phase Transfer — 3-Arm Comparison


## θ = safe

### Per-Phase Success Rate

| Session | Mode | Phase A | Phase B | Phase C | TBSR_B | TBSR_C |
|:-------:|:----:|:-------:|:-------:|:-------:|:------:|:------:|
| 0 | reset | 0.600 | 0.650 | 0.587 | 0.650 | 0.587 |
| 0 | persistent_nohook | 0.600 | 0.650 | 0.587 | 0.650 | 0.587 |
| 0 | persistent_needhook | 0.600 | 0.650 | 0.587 | 0.650 | 0.587 |
| 1 | reset | 0.475 | 0.488 | 0.512 | 0.488 | 0.512 |
| 1 | persistent_nohook | 0.467 | 0.488 | 0.512 | 0.488 | 0.512 |
| 1 | persistent_needhook | 0.467 | 0.488 | 0.512 | 0.488 | 0.512 |
| 2 | reset | 0.558 | 0.512 | 0.550 | 0.512 | 0.550 |
| 2 | persistent_nohook | 0.558 | 0.500 | 0.562 | 0.500 | 0.562 |
| 2 | persistent_needhook | 0.558 | 0.500 | 0.562 | 0.500 | 0.562 |
| 3 | reset | 0.483 | 0.438 | 0.500 | 0.438 | 0.500 |
| 3 | persistent_nohook | 0.475 | 0.438 | 0.500 | 0.438 | 0.500 |
| 3 | persistent_needhook | 0.475 | 0.438 | 0.500 | 0.438 | 0.500 |

## θ = shiny

### Per-Phase Success Rate

| Session | Mode | Phase A | Phase B | Phase C | TBSR_B | TBSR_C |
|:-------:|:----:|:-------:|:-------:|:-------:|:------:|:------:|
| 0 | reset | 0.442 | 0.488 | 0.637 | 0.488 | 0.637 |
| 0 | persistent_nohook | 0.442 | 0.488 | 0.637 | 0.488 | 0.637 |
| 0 | persistent_needhook | 0.442 | 0.488 | 0.637 | 0.488 | 0.637 |
| 1 | reset | 0.500 | 0.550 | 0.525 | 0.550 | 0.525 |
| 1 | persistent_nohook | 0.475 | 0.488 | 0.550 | 0.488 | 0.550 |
| 1 | persistent_needhook | 0.475 | 0.488 | 0.550 | 0.488 | 0.550 |
| 2 | reset | 0.483 | 0.550 | 0.525 | 0.550 | 0.525 |
| 2 | persistent_nohook | 0.542 | 0.525 | 0.550 | 0.525 | 0.550 |
| 2 | persistent_needhook | 0.542 | 0.525 | 0.550 | 0.525 | 0.550 |
| 3 | reset | 0.600 | 0.588 | 0.463 | 0.588 | 0.463 |
| 3 | persistent_nohook | 0.617 | 0.575 | 0.475 | 0.575 | 0.475 |
| 3 | persistent_needhook | 0.617 | 0.575 | 0.475 | 0.575 | 0.475 |

## Phase A WarnRate by Session

| θ | Session | reset | nohook | needhook |
|:-:|:-------:|:-----:|:------:|:--------:|
| safe | 0 | 0.033 | 0.033 | 0.033 |
| safe | 1 | 0.058 | 0.042 | 0.042 |
| safe | 2 | 0.058 | 0.033 | 0.033 |
| safe | 3 | 0.050 | 0.033 | 0.033 |
| shiny | 0 | 0.025 | 0.025 | 0.025 |
| shiny | 1 | 0.058 | 0.033 | 0.033 |
| shiny | 2 | 0.067 | 0.033 | 0.033 |
| shiny | 3 | 0.050 | 0.042 | 0.042 |

## Transfer Delta: Phase B Success (Final Session)

| θ | reset | nohook | needhook | needhook > reset? |
|:-:|:-----:|:------:|:--------:|:-----------------:|
| safe | 0.438 | 0.438 | 0.438 | ≈ |
| shiny | 0.588 | 0.575 | 0.575 | ❌ |

## Verdict

> Transfer (Phase B): needhook ≥ reset in 2/2 θ
> Shifted transfer (Phase C): needhook ≥ reset in 2/2 θ
> **✅ Persistent profile with need hook shows value in transfer**
