# Step 5B: Continuous Reward Shadow Results

**Seeds**: 30 | **Elapsed**: 1.4s

## Headline Metrics

| Mode | Scenario | Train_NLL↓ | Test_NLL↓ | Resid_Norm | Params |
|------|----------|-----------|----------|------------|--------|
| rigid | safe_agent | 0.1862 | 0.2083 | 0.0000 | 0 |
| rigid | shiny_agent | 0.2353 | 0.2671 | 0.0000 | 0 |
| rigid | composite_safe | 0.1862 | 0.2083 | 0.0000 | 0 |
| rigid | shortcut_agent | 0.2108 | 0.2476 | 0.0000 | 0 |
| B1 | safe_agent | 0.1863 | 0.2083 | 0.0845 | 1 |
| B1 | shiny_agent | 0.2344 | 0.2652 | 0.0676 | 1 |
| B1 | composite_safe | 0.1863 | 0.2083 | 0.0754 | 1 |
| B1 | shortcut_agent | 0.2108 | 0.2474 | 0.0928 | 1 |
| B2 | safe_agent | 0.1863 | 0.2083 | 0.0429 | 2 |
| B2 | shiny_agent | 0.2345 | 0.2656 | 0.0295 | 2 |
| B2 | composite_safe | 0.1863 | 0.2083 | 0.0463 | 2 |
| B2 | shortcut_agent | 0.2108 | 0.2475 | 0.0082 | 2 |
| B3 | safe_agent | 0.1862 | 0.2083 | 0.0277 | 4 |
| B3 | shiny_agent | 0.2345 | 0.2656 | 0.0190 | 4 |
| B3 | composite_safe | 0.1863 | 0.2083 | 0.0286 | 4 |
| B3 | shortcut_agent | 0.2108 | 0.2475 | 0.0258 | 4 |

## Promotion Analysis

### safe_agent
- B1 vs rigid: Δ test NLL = -0.0000 PARITY
- B2 vs rigid: Δ test NLL = -0.0000 PARITY

### shiny_agent
- B1 vs rigid: Δ test NLL = -0.0019 BETTER
- B2 vs rigid: Δ test NLL = -0.0015 BETTER

### composite_safe
- B1 vs rigid: Δ test NLL = -0.0000 PARITY
- B2 vs rigid: Δ test NLL = -0.0000 PARITY

### shortcut_agent
- B1 vs rigid: Δ test NLL = -0.0002 PARITY
- B2 vs rigid: Δ test NLL = -0.0000 PARITY

