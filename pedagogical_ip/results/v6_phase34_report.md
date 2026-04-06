# V6 Phase 3–4 Report

## P3.1: Branch Gating Audit

| Family | Maps | Triggered | Rate |
|--------|------|-----------|------|
| fork_trap | 20 | 0 | 0% |
| hazard_belt | 20 | 0 | 0% |
| deadline_gate | 20 | 0 | 0% |
| elcb | 20 | 20 | 100% |

## P3.2: Canonical Compatibility Sweep

| Family | Condition | SR | DR | TR | Steps |
|--------|-----------|----|----|-------|-------|
| fork_trap | no_tutor | 5% | 95% | 0% | 4.7 |
| fork_trap | robot_belief | 70% | 30% | 0% | 8.9 |
| hazard_belt | no_tutor | 30% | 70% | 0% | 14.3 |
| hazard_belt | robot_belief | 65% | 35% | 0% | 19.4 |
| deadline_gate | no_tutor | 70% | 30% | 0% | 22.4 |
| deadline_gate | robot_belief | 100% | 0% | 0% | 26.0 |

## P3.3: ELCB Regression Lock

| Condition | SBCR | SideBias |
|-----------|------|----------|
| old_planner | 67% | 0.500 |
| concept_only | 100% | 0.167 |
| scorer_only | 100% | 0.167 |
| hybrid | 100% | 0.167 |

## P4.1: RSA Warning + Branch Planner

| Condition | k=0 | k=1 | k=3 | k=10 | k=30 |
|-----------|-----|-----|-----|------|------|
| no_tutor+old | 60% | 70% | 30% | 30% | 70% |
| rsa_warn+old | 60% | 70% | 30% | 30% | 70% |
| no_tutor+branch | 100% | 100% | 100% | 100% | 100% |
| rsa_warn+branch | 100% | 100% | 100% | 100% | 100% |

## Summary

### Canonical Compatibility
Check that SR values are not degraded vs existing baselines.

### ELCB Regression
Check that concept/scorer/hybrid still ≈100%.

### Tutor Effect
Compare `rsa_warn+branch` vs `no_tutor+branch` at each k.
