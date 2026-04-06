# Step 5A.1: Necessity Gate Hardening Audit

**Seeds**: 50 | **Elapsed**: 1.8s

## Phase A: Necessity Gate Comparison

| Gate | Scenario | TBSR↑ | Crash↓ | Timeout↓ | Mono |
|------|----------|-------|--------|----------|------|
| baseline | hazard_belt | 0.160 | 0.840 | 0.000 | ✗ |
| baseline | hazard_belt_high | 0.000 | 1.000 | 0.000 | ✗ |
| baseline | deadline_gate | 0.580 | 0.000 | 0.420 | ✓ |
| baseline | fork_trap | 0.900 | 0.100 | 0.000 | ✗ |
| baseline | fork_trap_necessary | 0.900 | 0.100 | 0.000 | ✓ |
| baseline | elcb_po | 0.220 | 0.780 | 0.000 | ✗ |
| original | hazard_belt | 0.160 | 0.840 | 0.000 | ✗ |
| original | hazard_belt_high | 0.000 | 1.000 | 0.000 | ✗ |
| original | deadline_gate | 0.580 | 0.000 | 0.420 | ✓ |
| original | fork_trap | 0.900 | 0.100 | 0.000 | ✗ |
| original | fork_trap_necessary | 0.900 | 0.100 | 0.000 | ✓ |
| original | elcb_po | 0.220 | 0.780 | 0.000 | ✗ |
| N1 | hazard_belt | 0.160 | 0.840 | 0.000 | ✗ |
| N1 | hazard_belt_high | 0.000 | 1.000 | 0.000 | ✗ |
| N1 | deadline_gate | 0.580 | 0.000 | 0.420 | ✓ |
| N1 | fork_trap | 0.900 | 0.100 | 0.000 | ✗ |
| N1 | fork_trap_necessary | 0.900 | 0.100 | 0.000 | ✓ |
| N1 | elcb_po | 0.220 | 0.780 | 0.000 | ✗ |
| N2 | hazard_belt | 0.160 | 0.840 | 0.000 | ✗ |
| N2 | hazard_belt_high | 0.000 | 1.000 | 0.000 | ✗ |
| N2 | deadline_gate | 0.580 | 0.000 | 0.420 | ✓ |
| N2 | fork_trap | 0.900 | 0.100 | 0.000 | ✗ |
| N2 | fork_trap_necessary | 0.900 | 0.100 | 0.000 | ✓ |
| N2 | elcb_po | 0.220 | 0.780 | 0.000 | ✗ |
| N3 | hazard_belt | 0.160 | 0.840 | 0.000 | ✗ |
| N3 | hazard_belt_high | 0.000 | 1.000 | 0.000 | ✗ |
| N3 | deadline_gate | 0.580 | 0.000 | 0.420 | ✓ |
| N3 | fork_trap | 0.900 | 0.100 | 0.000 | ✗ |
| N3 | fork_trap_necessary | 0.900 | 0.100 | 0.000 | ✓ |
| N3 | elcb_po | 0.220 | 0.780 | 0.000 | ✗ |

### Necessity Regression Gap

| Gate | fork_trap TBSR | fork_trap_nec TBSR | Δ_nec | Passes? |
|------|---------------|-------------------|------|--------|
| baseline | 0.900 | 0.900 | +0.000 | ✓ |
| original | 0.900 | 0.900 | +0.000 | ✓ |
| N1 | 0.900 | 0.900 | +0.000 | ✓ |
| N2 | 0.900 | 0.900 | +0.000 | ✓ |
| N3 | 0.900 | 0.900 | +0.000 | ✓ |

## Phase B: Monotonicity Fixes (on N2 gate)

| Mono | Scenario | TBSR↑ | Crash↓ | Mono_OK | Violations |
|------|----------|-------|--------|---------|------------|
| none | hazard_belt | 0.160 | 0.840 | ✗ | 50 |
| none | hazard_belt_high | 0.000 | 1.000 | ✗ | 50 |
| none | deadline_gate | 0.580 | 0.000 | ✓ | 0 |
| none | fork_trap | 0.900 | 0.100 | ✗ | 50 |
| none | fork_trap_necessary | 0.900 | 0.100 | ✓ | 0 |
| none | elcb_po | 0.220 | 0.780 | ✗ | 50 |
| M1 | hazard_belt | 0.160 | 0.840 | ✗ | 50 |
| M1 | hazard_belt_high | 0.000 | 1.000 | ✗ | 50 |
| M1 | deadline_gate | 0.580 | 0.000 | ✓ | 0 |
| M1 | fork_trap | 0.900 | 0.100 | ✗ | 50 |
| M1 | fork_trap_necessary | 0.900 | 0.100 | ✓ | 0 |
| M1 | elcb_po | 0.220 | 0.780 | ✗ | 50 |
| M2 | hazard_belt | 0.160 | 0.840 | ✗ | 50 |
| M2 | hazard_belt_high | 0.000 | 1.000 | ✗ | 50 |
| M2 | deadline_gate | 0.580 | 0.000 | ✓ | 0 |
| M2 | fork_trap | 0.900 | 0.100 | ✗ | 50 |
| M2 | fork_trap_necessary | 0.900 | 0.100 | ✓ | 0 |
| M2 | elcb_po | 0.220 | 0.780 | ✗ | 50 |

## Phase C: Promotion Summary

### Criteria Assessment

| Gate | fork_trap ↑ | fork_trap_nec ≥ baseline | elcb_po ↑ | deadline ≥ parity | Mono |
|------|------------|-------------------------|----------|-----------------|------|
| baseline | ✓ (0.900) | ✓ (0.900) | ✗ (0.220) | ✓ (0.580) | ✗ |
| original | ✓ (0.900) | ✓ (0.900) | ✗ (0.220) | ✓ (0.580) | ✗ |
| N1 | ✓ (0.900) | ✓ (0.900) | ✗ (0.220) | ✓ (0.580) | ✗ |
| N2 | ✓ (0.900) | ✓ (0.900) | ✗ (0.220) | ✓ (0.580) | ✗ |
| N3 | ✓ (0.900) | ✓ (0.900) | ✗ (0.220) | ✓ (0.580) | ✗ |
