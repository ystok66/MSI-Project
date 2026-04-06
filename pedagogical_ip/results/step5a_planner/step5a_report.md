# Step 5A: Risk-Sensitive Planner Shadow Results

**Seeds**: 30 | **Elapsed**: 0.3s

## Headline Metrics

| Mode | Scenario | TBSR↑ | Crash↓ | Timeout↓ | Detour↓ | Mono | NecGate |
|------|----------|-------|--------|----------|---------|------|--------|
| baseline | hazard_belt | 0.200 | 0.800 | 0.000 | 0.0 | ✓ | N/A |
| baseline | hazard_belt_high | 0.000 | 1.000 | 0.000 | 0.0 | ✓ | N/A |
| baseline | deadline_gate | 0.767 | 0.000 | 0.233 | 0.0 | ✓ | N/A |
| baseline | fork_trap | 0.767 | 0.233 | 0.000 | 0.0 | ✓ | N/A |
| baseline | fork_trap_necessary | 0.800 | 0.200 | 0.000 | 0.0 | ✓ | N/A |
| baseline | elcb_po | 0.200 | 0.800 | 0.000 | 0.0 | ✓ | N/A |
| A1 | hazard_belt | 0.167 | 0.833 | 0.000 | 0.0 | ✗ | N/A |
| A1 | hazard_belt_high | 0.000 | 1.000 | 0.000 | 0.0 | ✗ | N/A |
| A1 | deadline_gate | 0.367 | 0.000 | 0.633 | 0.0 | ✓ | N/A |
| A1 | fork_trap | 0.733 | 0.267 | 0.000 | 0.0 | ✗ | N/A |
| A1 | fork_trap_necessary | 0.800 | 0.200 | 0.000 | 0.0 | ✓ | N/A |
| A1 | elcb_po | 0.233 | 0.767 | 0.000 | 0.0 | ✗ | N/A |
| A2 | hazard_belt | 0.200 | 0.800 | 0.000 | 0.0 | ✗ | N/A |
| A2 | hazard_belt_high | 0.000 | 1.000 | 0.000 | 0.0 | ✗ | N/A |
| A2 | deadline_gate | 0.533 | 0.000 | 0.467 | 0.0 | ✓ | N/A |
| A2 | fork_trap | 0.900 | 0.100 | 0.000 | 0.0 | ✗ | N/A |
| A2 | fork_trap_necessary | 0.633 | 0.367 | 0.000 | 0.0 | ✓ | N/A |
| A2 | elcb_po | 0.300 | 0.700 | 0.000 | 0.0 | ✗ | N/A |
| A3 | hazard_belt | 0.133 | 0.867 | 0.000 | 0.0 | ✗ | N/A |
| A3 | hazard_belt_high | 0.000 | 1.000 | 0.000 | 0.0 | ✗ | N/A |
| A3 | deadline_gate | 0.600 | 0.000 | 0.400 | 0.0 | ✓ | N/A |
| A3 | fork_trap | 0.833 | 0.167 | 0.000 | 0.0 | ✗ | N/A |
| A3 | fork_trap_necessary | 0.833 | 0.167 | 0.000 | 0.0 | ✓ | N/A |
| A3 | elcb_po | 0.133 | 0.867 | 0.000 | 0.0 | ✗ | N/A |

## Promotion Analysis

### hazard_belt
- TBSR: 0.200 vs 0.200 PARITY
- Crash: 0.800 vs 0.800 PARITY

### hazard_belt_high
- TBSR: 0.000 vs 0.000 PARITY
- Crash: 1.000 vs 1.000 PARITY

### deadline_gate
- TBSR: 0.533 vs 0.767 WORSE
- Crash: 0.000 vs 0.000 PARITY

### fork_trap
- TBSR: 0.900 vs 0.767 **BETTER**
- Crash: 0.100 vs 0.233 **BETTER**

### fork_trap_necessary
- TBSR: 0.633 vs 0.800 WORSE
- Crash: 0.367 vs 0.200 WORSE

### elcb_po
- TBSR: 0.300 vs 0.200 **BETTER**
- Crash: 0.700 vs 0.800 **BETTER**

