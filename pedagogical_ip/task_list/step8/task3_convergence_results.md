# Task 3 Convergence Results — GTET z Demotion

**Date**: 2026-04-06  
**Experiment**: E3.1 — 20 seeds × 3 families × 2 factor_modes

---

## Summary

| Family | G_THETA (no-z) | FULL (with-z) | Drop |
|--------|----------------|---------------|------|
| GTET surv | 0.950 | 0.950 | 0.000 |
| GTET goal | 0.950 | 0.950 | 0.000 |
| DTMB surv | 0.450 | 0.450 | 0.000 |
| DTMB goal | 0.100 | 0.100 | 0.000 |
| BL_V2 surv | 0.300 | 0.300 | 0.000 |
| BL_V2 goal | 0.300 | 0.300 | 0.000 |

---

## Verdict

**All regression criteria PASS with zero drop.**

The temptation latent z has zero effect on tutor decision outcomes across all three families (20 seeds each). This confirms Phase 0's finding: z can be safely demoted from the default decision path.

### What changed
- `factor_mode` default: `"FULL"` → `"G_THETA"`
- Tutor now uses `q_dec(g,θ) = Σ_z q(g,θ,z)` for decisions
- Posterior still computes full `q(g,θ,z)` for diagnostics
- PRS session updated to match

### What was preserved
- Full posterior computation (z is still updated)
- GTET generator temptation cues (unchanged)
- All `factor_mode` variants remain available for ablation
- `FULL` mode accessible via explicit `factor_mode="FULL"` arg

---

## Canonical Defaults After Task 3

| Parameter | Previous | Canonical | Task |
|-----------|----------|-----------|------|
| `warning_variant` | `legacy_bias` | `rsa_obs_s1` | 1A |
| `boredom_weight` | 0.0 | 0.3 | 1B |
| `factor_mode` | `FULL` | **`G_THETA`** | 3 |
