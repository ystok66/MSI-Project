# 5D Observer Architecture — Canonical Reference

**Frozen: 2026-03-29 | Evidence: P0–P6**

## Three-Layer Architecture

```
┌───────────────────────────────────────────────┐
│  Layer 1: State Estimator (5D)                │
│  m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)     │
│  "What we can estimate"                       │
├───────────────────────────────────────────────┤
│  Layer 2: Micro Decision View (3D)            │
│  m̃_micro = (τ̂, ν̂, γ̂_gen)                    │
│  Action: {WAIT, WARN}   (2-act canonical)     │
│  γ̂_spec, κ̂ do NOT enter micro Q              │
├───────────────────────────────────────────────┤
│  Layer 3: Macro / Diagnostic View (5D)        │
│  m̃_macro = (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)  │
│  κ̂ → TEACH bonus (β=0.02, risk lessons)      │
│  γ̂_spec → diagnostic only (no score yet)     │
└───────────────────────────────────────────────┘
```

## Dimension Semantics

| Dim | Name | Semantic | Layer | Update |
|-----|------|----------|:-----:|--------|
| τ̂ | Trust | Valid-advice uptake | L2 | Warn outcome |
| ν̂ | Dependence | Blind obedience | L2 | Self-discovery / blind-follow |
| γ̂_gen | General suppression | Exploration inhibition | L2 | Pressure / exploration |
| γ̂_spec_state | Temptation resistance | Behavioral state (NOT trait) | L1+L3 | Resist/follow under lure |
| κ̂ | Risk calibration | Risk prediction accuracy | L1+L3 | Signed risk error |

## κ̂ Update Formula

```
δ_risk = risk_true − risk_hat        (signed)
κ̂(t+1) = clip(
  (1−λ_κ)·κ̂(t) + λ_κ·κ_0            // mean reversion
  + α⁺·max(δ,0)·(κ_max − κ̂)         // underestimated → cautious
  + α⁻·min(δ,0)·(κ̂ − κ_min)         // overestimated → relax
)
Gate: only when risk ≥ 0.1 AND risk_hat available
```

## κ̂ Macro Bonus

```
S_teach^5d(ℓ) = S_teach^base(ℓ) + β_κ · 1[ℓ ∈ L_risk] · |κ̂ − κ_0|
β_κ = 0.02 (minimum effective dose)
```

## Frozen Parameters

| Parameter | Value | Source |
|-----------|:-----:|--------|
| β_τ_probe | 0.0 | A1 frozen |
| β_ν_probe | 0.0 | A1 frozen |
| β_γ_probe | 0.0 | A1 frozen |
| λ_τ/ν/γ | 0.005 | A1 frozen |
| α_gs_resist | 0.03 | P4-A |
| α_gs_follow | 0.025 | P4-A |
| λ_κ | 0.02 | P5 |
| α_κ⁺ | 0.015 | P5 |
| α_κ⁻ | 0.012 | P5 |
| κ_0 | 0.3 | P5 |
| β_κ | 0.02 | P5-D |

## Evidence Summary

| Property | Value | Source |
|----------|:-----:|:------:|
| Sign accuracy | 99.6% | P5-A |
| Corr(κ̂, δ_risk) | 0.876 | P5-A |
| ΔR² | +0.116 | P5-D/E |
| Per-family ΔR² | All 13 positive | P5-E |
| Partial corr max | 0.175 | P5-E |
| Held-out wins | 8/13 | P5-E |
| β robustness | [0.02, 0.20] | P5-D |
| Top-1 / STOP | 100% / 100% | P5-D |

## Configs

| File | Role |
|------|------|
| `configs/observer/a1_5d_kappa.yaml` | 5D canonical |
| `configs/observer/a1_4d_gamma_spec_state.yaml` | 4D (backward compat) |
| `configs/tutor/bcictv4_2act.yaml` | 2-act canonical |
| `configs/tutor/bcictv4_3act_legacy.yaml` | Legacy/research only |

## Active Mask (corrected)

```
Active_t = 1[a_oracle ≠ WAIT  OR  a_infer ≠ WAIT]
```
Old `oracle-warned-only` mask is deprecated.
