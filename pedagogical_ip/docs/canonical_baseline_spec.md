# Canonical Baseline Specification

**Frozen: 2026-03-29 | Authority: Task 1 Closure**

---

## Default Configuration

### Observer: 5D Canonical
- **Config**: `configs/observer/a1_5d_kappa.yaml`
- **Class**: `A1MtObserverFrozen`
- **Dimensions**: τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂

### Tutor: 2-Act Canonical
- **Config**: `configs/tutor/bcictv4_2act.yaml`
- **Class**: `BCICTv4(use_dose=False)`
- **Action Space**: {WAIT, WARN}

### κ̂ Macro Bonus: **DEFAULT-ON**
- **β_κ = 0.02** (minimum effective dose)
- **Risk Lessons**: `tic_rescue_heavy`, `blind_activation_corridor`, `warn_symmetric_rescue`
- **Formula**: `S_teach^5d(ℓ) = S_teach^base(ℓ) + β_κ · 1[ℓ ∈ L_risk] · |κ̂ − κ_0|`
- **Evidence**: OOD 10/10, per-family ΔR² all positive, Top-1/STOP stable

### Active Mask (Corrected)
```
Active_t = 1[a_oracle ≠ WAIT  OR  a_infer ≠ WAIT]
```

### Dead-Zone Tie Policy (Recommended)
- **ε_Q = 0.05**: WAIT-preferred when |ΔQ| ≤ ε_Q
- **Status**: External wrapper, not in source code
- **WarnNecRecall**: 1.0000 (no degradation at ε=0.05)

---

## Three-Layer Architecture

| Layer | View | Dimensions | Purpose |
|:-----:|:----:|:----------:|---------|
| L1 | State Estimator | 5D (full) | "What we can estimate" |
| L2 | Micro Decision | 3D (τ̂,ν̂,γ̂_gen) | Q(WAIT\|WARN) |
| L3 | Macro/Diagnostic | 5D (full) | κ̂ bonus + γ̂_spec diagnostic |

**Design Principle**: Estimate first, consume later.

---

## Frozen Parameters

| Parameter | Value | Source | Modifiable? |
|-----------|:-----:|:------:|:-----------:|
| τ̂/ν̂/γ̂ init | 0.3/0.1/0.0 | A1 frozen | ❌ |
| λ_τ/ν/γ | 0.005 | A1 frozen | ❌ |
| β probes | all 0.0 | A1 frozen | ❌ |
| α_gs_resist | 0.03 | P4-A | ❌ |
| α_gs_follow | 0.025 | P4-A | ❌ |
| λ_κ | 0.02 | P5 | ❌ |
| α_κ⁺/α_κ⁻ | 0.015/0.012 | P5 | ❌ |
| κ_0 | 0.3 | P5 | ❌ |
| β_κ | 0.02 | P5-D | ❌ |
| 2-act {WAIT,WARN} | canonical | P3 | ❌ |
| Active mask | corrected | P3 | ❌ |

---

## Module Status

### Frozen (do not modify)
- A1Frozen 3D parameters
- 2-act canonical action space
- γ̂_spec exclusion from micro Q
- κ̂ exclusion from micro Q
- Micro Q formula structure

### Locked Default-On
- κ̂ macro bonus (β=0.02)
- 5D observer estimation

### Deferred (not in scope)
- Observer damping simplification
- CurriculumControllerV13 refactoring
- Legacy parameter cleanup
- Persistent learner profiles
- EPU / Belief-Horizon / EIG (shadow-mode research)
