# P4-A: γ_spec Verification

## Exp 1: γ̂_spec Trajectory by Temptation Level

| θ | Tempt | γ̂_spec(final) | γ_spec(true,final) | ν̂(final) | ν(true,final) | γ̂_gen(final) |
|:-:|:-----:|:-------------:|:------------------:|:--------:|:-------------:|:------------:|
| safe | none | 0.1657 | 0.6837 | 0.0849 | 0.1373 | 0.0186 |
| safe | 0.3 | 0.2604 | 0.6311 | 0.0726 | 0.1199 | 0.0177 |
| safe | 0.6 | 0.3969 | 0.5781 | 0.0580 | 0.0956 | 0.0161 |
| safe | 1.0 | 0.5881 | 0.2918 | 0.0608 | 0.0799 | 0.0149 |
| shiny | none | 0.1831 | 0.6906 | 0.0689 | 0.1083 | 0.0142 |
| shiny | 0.3 | 0.1209 | 0.7000 | 0.0676 | 0.1621 | 0.0164 |
| shiny | 0.6 | 0.0414 | 0.7000 | 0.0872 | 0.1939 | 0.0161 |
| shiny | 1.0 | 0.0360 | 0.7000 | 0.0908 | 0.1990 | 0.0161 |

## Exp 2: ν Contamination Check

Does ν̂ change when temptation increases? (It shouldn't.)

| θ | ν̂(tempt=0) | ν̂(tempt=0.6) | ν̂(tempt=1.0) | Δν̂(0→1.0) |
|:-:|:----------:|:------------:|:------------:|:----------:|
| safe | 0.0849 | 0.0580 | 0.0608 | -0.0241 |
| shiny | 0.0689 | 0.0872 | 0.0908 | 0.0219 |

## Exp 3: γ̂_spec Monotonicity

Does γ̂_spec increase with resistance (higher tempt + correct choices)?

| θ | Tempt | Mean γ̂_spec | Monotone? |
|:-:|:-----:|:-----------:|:---------:|
| safe | 0.0 | 0.1657 | ✅ |
| safe | 0.3 | 0.2604 | ✅ |
| safe | 0.6 | 0.3969 | ✅ |
| safe | 1.0 | 0.5881 | ✅ |
| shiny | 0.0 | 0.1831 | ✅ |
| shiny | 0.3 | 0.1209 | ❌ |
| shiny | 0.6 | 0.0414 | ❌ |
| shiny | 1.0 | 0.0360 | ❌ |
