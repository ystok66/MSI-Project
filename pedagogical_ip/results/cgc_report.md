# CGC: Compositional-Goal Corridor

**Config**: 6 sessions × 12 episodes × 2θ × 8 strategies

## Train Compositions

| θ | Strategy | SBCR | WR | WR(aln) | WR(cnf) | WR(bnd) | WR(dcy) | **SelGap** | ConflRes |
|---|----------|------|----|---------|---------|---------|---------|-----------|----------|
| safe | always_wait | 56% | 0% | 0% | 0% | 0% | 0% | **0.000** | 46% |
| safe | always_warn | 56% | 100% | 100% | 100% | 100% | 100% | **0.000** | 46% |
| safe | v4_reset | 56% | 92% | 62% | 100% | 100% | 100% | **0.375** | 46% |
| safe | v1_1_persistent | 56% | 57% | 25% | 72% | 97% | 47% | **0.475** | 46% |
| safe | joint_v1 | 56% | 96% | 78% | 100% | 100% | 100% | **0.222** | 46% |
| safe | joint_v2_coupled | 56% | 78% | 35% | 92% | 97% | 85% | **0.578** | 46% |
| safe | joint_v2_fact_abl | 56% | 94% | 75% | 100% | 100% | 94% | **0.250** | 46% |
| safe | oracle | 56% | 8% | 0% | 0% | 28% | 0% | **0.000** | 46% |
| shiny | always_wait | 69% | 0% | 0% | 0% | 0% | 0% | **0.000** | 56% |
| shiny | always_warn | 69% | 100% | 100% | 100% | 100% | 100% | **0.000** | 56% |
| shiny | v4_reset | 69% | 90% | 71% | 90% | 100% | 100% | **0.194** | 56% |
| shiny | v1_1_persistent | 69% | 62% | 22% | 71% | 94% | 57% | **0.489** | 56% |
| shiny | joint_v1 | 69% | 93% | 69% | 100% | 100% | 100% | **0.306** | 56% |
| shiny | joint_v2_coupled | 69% | 90% | 62% | 100% | 100% | 93% | **0.375** | 56% |
| shiny | joint_v2_fact_abl | 69% | 96% | 92% | 100% | 100% | 93% | **0.083** | 56% |
| shiny | oracle | 69% | 8% | 0% | 0% | 36% | 0% | **0.000** | 56% |

## SelGap Comparison (train compositions)

| θ | v4 | v1.1 | joint_v1 | **joint_v2** | fact_abl | oracle |
|---|-----|------|----------|-------------|----------|--------|
| safe | 0.375 | 0.475 | 0.222 | **0.578** | 0.250 | 0.000 |
| shiny | 0.194 | 0.489 | 0.306 | **0.375** | 0.083 | 0.000 |

## Held-Out Composition Generalization

| θ | Strategy | SBCR | WR | SelGap | ConflRes |
|---|----------|------|----|--------|----------|
| safe | joint_v2_coupled | 49% | 82% | 0.292 | 54% |
| safe | v1_1_persistent | 49% | 62% | 0.125 | 54% |
| safe | oracle | 49% | 4% | 0.000 | 54% |
| shiny | joint_v2_coupled | 83% | 90% | 0.417 | 81% |
| shiny | v1_1_persistent | 83% | 81% | 0.267 | 81% |
| shiny | oracle | 83% | 7% | 0.000 | 81% |
