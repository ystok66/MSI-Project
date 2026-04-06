# Stage 5 Summary: Shadow Observer → 5D Canonical Architecture

> **Handoff document** — self-contained reference for the next agent or collaborator.
> Covers the full research trajectory from A1 observer to 5D canonical architecture with κ̂ macro bonus.
> Does not require conversation history to understand.

---

## 1. Stage Identity & Goal

This stage addresses the **observability problem**: can a non-RL, belief-updating pedagogical tutor make correct micro (WAIT/WARN) and macro (TEACH/EVAL/STOP) decisions **without reading the learner's true internalization state** $m_t$?

The goal is NOT to make the observer "more like a regression model." It is:

> **To let the Tutor operate on estimated latent state $\hat m_t$ instead of oracle $m_t$, while maintaining micro/macro decision quality, and to expand the latent space to capture dimensions (temptation resistance, risk calibration) that the original 3D state missed.**

---

## 2. Architecture: Three-Layer 5D System

The canonical architecture established in this stage is:

```
┌───────────────────────────────────────────────────┐
│  Layer 1: State Estimator (5D)                     │
│  m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)          │
│  "What we can estimate about the learner"          │
├───────────────────────────────────────────────────┤
│  Layer 2: Micro Decision View (3D)                 │
│  m̃_micro = (τ̂, ν̂, γ̂_gen)                         │
│  Action: {WAIT, WARN}   (2-act canonical)          │
│  γ̂_spec, κ̂ do NOT enter micro Q                   │
├───────────────────────────────────────────────────┤
│  Layer 3: Macro / Diagnostic View (5D)             │
│  m̃_macro = (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)       │
│  κ̂ → TEACH bonus (β=0.02, risk-relevant lessons)  │
│  γ̂_spec_state → diagnostic only (no score yet)    │
└───────────────────────────────────────────────────┘
```

### Design Principle

**"Estimate first, consume later."** Every new dimension enters Layer 1 as a state estimator variable. It enters Layer 2 (micro Q) or Layer 3 scoring only after passing signal audit, non-redundancy proof, and OOD robustness.

---

## 3. Dimension Semantics

$$\hat m_t^{(5)} = (\hat\tau_t, \hat\nu_t, \hat\gamma_t^{gen}, \hat\gamma_t^{spec\_state}, \hat\kappa_t)$$

| Dim | Name | Semantic | Layer | Update Trigger | Direction |
|-----|------|----------|:-----:|----------------|-----------|
| $\hat\tau$ | Trust | Valid-advice uptake | L2 micro | Warn outcome | ↑ = better |
| $\hat\nu$ | Dependence | Blind obedience | L2 micro | Self-disc / blind-follow | **↑ = worse** |
| $\hat\gamma^{gen}$ | General suppression | Exploration inhibition | L2 micro | Pressure / exploration | **↑ = worse** |
| $\hat\gamma^{spec}_{state}$ | Temptation resistance | Behavioral state (NOT trait) | L1+L3 diag | Resist/follow under lure | ↑ = better |
| $\hat\kappa$ | Risk calibration | Risk prediction accuracy | L1+L3 macro | Signed risk error | ↑ = more cautious |

### Critical Semantic Distinctions

- **$\hat\gamma^{spec}_{state}$ is a behavioral state**, not a trait latent. It tracks *current* temptation resistance in the *current* context, not stable personality. Evaluated via resist-rate correlation, NOT trait correlation.
- **$\hat\kappa$ is independent of the other 4 dims**. Sign accuracy 99.6%, partial correlations all < 0.18.
- **Layer 2 (micro Q) only sees 3 dims**. The 4th and 5th dimensions were rigorously proved to not enter the tutor's Q function (P4-C attribution ablation: 599/600 step identity).

---

## 4. Observer Implementation

### 4.1 Class Hierarchy

| Class | File | Role |
|-------|------|------|
| `RuleBasedMtObserver` | `internalization_observer.py` | Base 3D observer (τ̂, ν̂, γ̂_gen) |
| `A1MtObserver` | `internalization_observer.py` | Extended: +γ̂_spec_state +κ̂ (5D), fixes probe noise, conditional reversion |
| `A1MtObserverFrozen` | `internalization_observer.py` | Frozen baseline with parameter guards |
| `A2MtObserver` | `internalization_observer.py` | Research variant (not canonical) |

### 4.2 A1 Core Updates (Frozen — DO NOT MODIFY)

**Trust** ($\hat\tau$): Increased by correct warning outcomes, decreased by incorrect warnings. Probe-corrected (gated OFF by default). Conditional mean-reversion only when no recent informative events AND confidence is low.

**Dependence** ($\hat\nu$): Increased by blind obey, decreased by self-discovery. Uses timing separation for confidence.

**General Suppression** ($\hat\gamma^{gen}$): Increased by sustained pressure, decreased by successful exploration. Conditional mean-reversion.

### 4.3 γ̂_spec Update (P4-A)

Only fires when `lure ≥ 0.3`:

$$\hat\gamma^{spec}_{t+1} = \hat\gamma^{spec}_t + \begin{cases} \alpha^+_{gs} \cdot \text{lure} \cdot (1 - \hat\gamma^{spec}_t) & \text{if resisted} \\ -\alpha^-_{gs} \cdot \text{lure} \cdot \hat\gamma^{spec}_t & \text{if followed} \end{cases}$$

| Param | Value |
|-------|:-----:|
| $\alpha^+_{gs}$ | 0.03 |
| $\alpha^-_{gs}$ | 0.025 |
| lure threshold | 0.3 |

### 4.4 κ̂ Update (P5)

Uses **signed risk error** with event gating:

$$\delta_t^{risk} = r_t^{true} - r_t^{hat}$$

$$\hat\kappa_{t+1} = \text{clip}\Big( (1-\lambda_\kappa)\hat\kappa_t + \lambda_\kappa \kappa_0 + \alpha_\kappa^+ \max(\delta_t, 0)(\kappa_{max} - \hat\kappa_t) + \alpha_\kappa^- \min(\delta_t, 0)(\hat\kappa_t - \kappa_{min}) \Big)$$

| Param | Value | Meaning |
|-------|:-----:|---------|
| $\kappa_0$ | 0.3 | Mean-reversion anchor |
| $\lambda_\kappa$ | 0.02 | Mean-reversion rate (slow) |
| $\alpha_\kappa^+$ | 0.015 | Underestimated → more cautious |
| $\alpha_\kappa^-$ | 0.012 | Overestimated → relax |
| Risk gate | 0.1 | Min risk to trigger update |

**Key properties:**
- Direction-aware (positive δ → κ↑, negative δ → κ↓)
- Event-gated (only on risk ≥ 0.1 with risk_hat available)
- Slow variable (converges gradually, σ < 0.006)

---

## 5. Micro Tutor: 2-Act Canonical

### 5.1 Action Space

$$\mathcal{A}_{micro} = \{WAIT, WARN\}$$

SOFT (dose=0.5) was **proven structurally redundant** via exact-Q audit:

$$V_{WAIT} = 0.9733, \quad V_{SOFT} = 0.0050, \quad V_{WARN} = 0.0217$$

3-act vs 2-act showed near-identical success/transfer/macro metrics. SOFT is retained as `bcictv4_3act_legacy.yaml` for research-only comparison.

### 5.2 Micro Q Function

The canonical micro tutor (`BCICTv4`) computes Q using **only 3D**:

$$Q(a | m̃^{micro}) = f(\hat\tau, \hat\nu, \hat\gamma^{gen}, \text{scene context})$$

$\hat\gamma^{spec}$ and $\hat\kappa$ are provably absent from this computation (P4-C attribution ablation: A≡B identity on 599/600 steps).

---

## 6. Macro κ Bonus

### 6.1 Formula

For risk-relevant lesson $\ell \in \mathcal{L}_{risk}$:

$$S^{5d}_{teach}(\ell | x_t) = S^{base}_{teach}(\ell | x_t) + \beta_\kappa \cdot \mathbf{1}[\ell \in \mathcal{L}_{risk}] \cdot \frac{|\hat\kappa_t - \kappa_0|}{\kappa_{max} - \kappa_{min}}$$

| Param | Value |
|-------|:-----:|
| $\beta_\kappa$ | 0.02 (minimum effective dose) |
| $\mathcal{L}_{risk}$ | {tic_rescue_heavy, blind_activation_corridor, warn_symmetric_rescue} |

### 6.2 Centered-Deviation Design

Uses $|\hat\kappa - \kappa_0|$ (absolute deviation from anchor), NOT directional split. This avoids premature assumptions about which direction of κ deviation demands more risk lessons.

---

## 7. Corrected Active Mask

**Canonical (corrected):**

$$\text{Active}_t = \mathbf{1}[a_t^{oracle} \neq WAIT \;\lor\; a_t^{infer} \neq WAIT]$$

**Deprecated (old):** `oracle-warned-only` — missed Oracle=WAIT, Infer=WARN (over-warn) cases.

**Root cause:** The old mask reported `Div@Active=0` while `DivAll>0` because all divergences were over-warns (Infer=WARN when Oracle=WAIT), which the old mask didn't flag as active.

---

## 8. Research Trajectory (Staged Conclusions)

| Stage | Core Question | Key Result | Status |
|-------|--------------|------------|:------:|
| **P0** | Can A0 observer support micro/macro? | Corr(τ̂)=1.0, Corr(ν̂)=0.99, Corr(γ̂)=0.98 | ✅ |
| **P1** | A1 vs A0: does fixing probe/reversion help? | A1 strictly ≥ A0 on all metrics | ✅ |
| **P2** | Online micro infer-only: zero diverge? | Natural + temptation: 0 diverge | ✅ |
| **P3-A** | Coverage: does expanding active families break it? | 3 active families, still 0 diverge | ✅ |
| **P3-B/C** | Is SOFT redundant? | Exact-Q: V_SOFT ≈ 0, optimal volume ≈ 0 | ✅ |
| **P3-D** | 2-act vs 3-act canonical? | Near-identical metrics → 2-act canonical | ✅ |
| **P4-A** | γ̂_spec: can observer estimate temptation resistance? | Corr=0.96, responsive under temptation | ✅ |
| **P4-B** | 4D observer eval? | Micro/macro stable, ν-consistency ✅ | ✅ |
| **P4-B.1** | Metric audit: why DivAll>0 but Div@Active=0? | Old mask bug → corrected mask | ✅ |
| **P4-C** | Does γ̂_spec enter micro Q? | **No.** Attribution: A≡B (99.83% step identity) | ✅ |
| **P4-C.3** | κ observability: is there an independent risk signal? | Mean |e_risk|=0.146, orthogonal to 4D | ✅ |
| **P5** | κ̂ implementation + 5D no-score integration? | 48/48 tests, micro/macro stable | ✅ |
| **P5-A** | κ̂ signal quality? | **Sign=99.6%, Corr=0.876** | ✅ |
| **P5-C** | κ̂ macro bonus effective? | +2.3 rank shift, Top-1 stable | ✅ |
| **P5-D** | κ̂ non-redundant? | **ΔR²=+0.116**, β robust [0.02, 0.20] | ✅ |
| **P5-E** | Per-family ΔR² + held-out generalization? | **All 13 families positive**, 8/13 held-out | ✅ |
| **P6-A** | OOD robustness (5 conditions)? | **10/10 pass**, κ̂ directionally correct | ✅ |

---

## 9. Canonical Code Paths

### 9.1 Observer Stack

| File | Role | Lines |
|------|------|:-----:|
| `src/teachers/internalization_observer.py` | **5D observer** (A0 base → A1 → A1Frozen) | ~680 |
| `src/teachers/internalization_control_tutor_v4.py` | **BC-ICT-v4 micro tutor** (2-act canonical) | ~300 |
| `src/agents/internalization_state_v3.py` | True internalization state (5D: κ,τ,ν,γ_spec,γ_gen) | ~250 |
| `src/agents/stochastic_agent_policy.py` | Learner choice model with theta-dependent biases | ~100 |
| `src/agents/cost_risk_model.py` | Latent cost/risk head (provides risk_hat for κ) | ~250 |

### 9.2 Scenario & Curriculum

| File | Role |
|------|------|
| `src/curriculum/lesson_library_v2.py` | Lesson catalog: 13 families, BALANCED_ACTIVE_LESSONS, PROBE_NAMES |
| `src/curriculum/adaptive_episode_generator_v2.py` | Mastery-conditioned lesson → episode generator |
| `src/curriculum/curriculum_controller_v13.py` | Canonical macro controller (from Stage 4) |
| `src/envs/teaching_internalization_corridor_v4.py` | TIC-v4 corridor environment |
| `src/envs/scenario_families.py` | 120KB scenario definition library |
| `src/metrics/self_discovery.py` | Self-discovery probability estimator |

### 9.3 Configs

| File | Status | Role |
|------|:------:|------|
| `configs/observer/a1_5d_kappa.yaml` | **Canonical** | 5D observer with κ̂ params + β_κ=0.02 |
| `configs/observer/a1_4d_gamma_spec_state.yaml` | Compat | 4D observer (backward-compatible) |
| `configs/observer/a1_frozen.yaml` | Reference | Frozen A1 3D baseline |
| `configs/tutor/bcictv4_2act.yaml` | **Canonical** | 2-act {WAIT, WARN} |
| `configs/tutor/bcictv4_3act_legacy.yaml` | Legacy | 3-act {WAIT, SOFT, WARN} (research only) |

### 9.4 Test Suite

| File | Tests | Coverage |
|------|:-----:|---------|
| `tests/test_internalization_observer.py` | **55** | A0 base (23), A1 frozen (4), γ_spec (6), κ (5), 2-act (1), gate (6), metrics (3), P6 protocol (7) |

### 9.5 Experiment Scripts (Stage 5)

| Script | Purpose |
|--------|---------|
| `scripts/run_p3b_soft_optimality_audit.py` | SOFT redundancy exact-Q |
| `scripts/run_p3d_action_space_candidate.py` | 2-act vs 3-act comparison |
| `scripts/run_p4a_gamma_spec.py` | γ̂_spec implementation eval |
| `scripts/run_p4b_4d_observer_eval.py` | 4D observer micro/macro |
| `scripts/run_p4b1_metric_audit.py` | Active mask correction discovery |
| `scripts/run_p4c_attribution_ablation.py` | γ̂_spec attribution (A≡B proof) |
| `scripts/run_p4c3_macro_kappa.py` | κ observability audit |
| `scripts/run_p5ac_kappa_signal_bonus.py` | κ̂ per-family signal + macro bonus |
| `scripts/run_p5b_5d_integration.py` | 5D no-score integration |
| `scripts/run_p5d_kappa_nonredundancy_sweep.py` | ΔR² + β sweep |
| `scripts/run_p5e_formalization.py` | Per-family ΔR², partials, held-out |
| `scripts/run_p6a_ood_robustness.py` | OOD robustness (5 conditions) |

### 9.6 Results (Stage 5)

| File | Content |
|------|---------|
| `results/p3b_soft_optimality_audit.md` | V_SOFT ≈ 0, SOFT redundant |
| `results/p3d_action_space_candidate.md` | 2-act ≡ 3-act |
| `results/p4a_gamma_spec_results.md` | γ̂_spec Corr=0.96 |
| `results/p4b1_metric_audit.md` | Corrected active mask |
| `results/p4c_attribution_ablation.md` | A≡B (99.83%) |
| `results/p5ac_kappa_signal_bonus.md` | Sign=99.6%, Corr=0.876 |
| `results/p5b_5d_no_score_integration.md` | 5D stable |
| `results/p5d_kappa_nonredundancy_sweep.md` | ΔR²=+0.126, β robust |
| `results/p5e_formalization.md` | All 13 families positive, held-out 8/13 |
| `results/p6a_ood_robustness.md` | 10/10 OOD pass |

### 9.7 Documentation

| File | Content |
|------|---------|
| `docs/architecture_5d_canonical.md` | Formal 5D three-layer architecture reference |

---

## 10. Evidence Summary Tables

### 10.1 κ̂ Complete Evidence Chain

| Property | Value | Source |
|----------|:-----:|:------:|
| Sign accuracy | 99.6% | P5-A |
| Corr(κ̂, δ_risk) | 0.876 | P5-A |
| ΔR² (global) | +0.116 | P5-D |
| Per-family ΔR² | All 13 positive | P5-E |
| Top ΔR² family | tic_rescue_heavy (+0.256) | P5-E |
| Partial corr max | 0.175 | P5-E |
| Held-out wins | 8/13 families | P5-E |
| β min effective dose | 0.02 | P5-D |
| β robustness range | [0.02, 0.20] | P5-D |
| Top-1 stability | 100% all conditions | P5-D, P6-A |
| STOP stability | 100% all conditions | P5-D, P6-A |
| OOD pass rate | 10/10 (5 conditions × 2θ) | P6-A |

### 10.2 γ̂_spec Evidence

| Property | Value | Source |
|----------|:-----:|:------:|
| Corr with resist rate | 0.96 | P4-A |
| Enters micro Q? | **No** (proved) | P4-C |
| 3D vs 4D macro | Identical (by construction) | P4-C.3 |
| Semantic | Behavioral state, NOT trait | P4-A2 |

### 10.3 Observer Baseline (A1 Frozen — 3D)

| Property | Value | Source |
|----------|:-----:|:------:|
| Corr(τ̂, τ) | 1.00 | P0 |
| Corr(ν̂, ν) | 0.99 | P0 |
| Corr(γ̂_gen, γ_gen) | 0.98 | P0 |
| Online infer-only diverge | 0 | P2 |
| Macro STOP agreement | 97.7–100% | P2 |
| Macro Kendall τ | ≥ 0.994 | P2 |

---

## 11. Frozen vs. Active vs. Deferred Modules

### Frozen (DO NOT MODIFY)

| Module | Reason |
|--------|--------|
| A1Frozen-3D parameters | Baseline locked, evidence sufficient |
| 2-act canonical {WAIT, WARN} | SOFT proven redundant |
| Corrected active mask | Old mask deprecated |
| 3D micro score (tutor Q) | Attribution ablation: A≡B |

### Active (Current Main Line)

| Module | Status |
|--------|--------|
| 5D Layer-1 state estimator | Canonical |
| κ̂ macro bonus (β=0.02) | Canonical candidate |
| 5D no-score + κ-bonus protocol | Formalized |
| 55/55 test suite | Green |

### Deferred (NOT current priority)

| Module | Reason |
|--------|--------|
| γ̂_spec → macro score | No clean utility design yet |
| κ̂ → micro Q | Not proven necessary |
| Posterior observer | Framework not ready |
| 3-act / SOFT reinstatement | Proven redundant |

---

## 12. Known Deficiencies & Open Questions

1. **κ̂ convergence is slow** — by design (slow variable), but means short episodes may not see much κ movement
2. **γ̂_spec partial correlation with κ̂** — moderate negative in safe θ (−0.36), near-zero in shiny. May indicate shared context, not latent overlap
3. **β plateau saturation** — β=0.02 to 0.20 produce identical reranking because rank gap is small; this is a feature (robustness) but limits fine-grained β tuning
4. **Held-out prediction: 5D loses on 4/13 families** — soft_boundary_tradeoff, tic_self_discovery, verified_warn, warn_symmetric_rescue show small MAE regressions
5. **τ̂ is near-constant** — partial correlation with κ̂ is NaN because τ̂ has near-zero variance in current slice; this is by design (trust stabilizes quickly) but limits partial correlation analysis
6. **Over-warn remains pre-existing** — concentrated in 3 active families (blind_corridor, warn_rescue, tic_rescue); caused by near-tie Q boundaries, NOT by 4D/5D expansion

---

## 13. Recommended Next Steps

### 13.1 Immediate (within current framework)

1. **Switch κ̂ macro bonus to default-on** — all evidence supports this
2. **Formalize evaluation protocol** as mandatory regression suite
3. **Address over-warn** in 3 near-tie families via Q-margin analysis

### 13.2 Next Research Phase

1. **Persistent profile** — multi-session learner tracking; 5D Layer-1 is the prerequisite
2. **Compositional goals** — goal + preference + temptation + risk calibration coexisting
3. **γ̂_spec macro utility** — only after persistent profile reveals temptation-curriculum patterns
4. **κ̂ directional split** — $g_\kappa^+$ vs $g_\kappa^-$ for different risk lesson subtypes

### 13.3 Do NOT Do

- Do not put γ̂_spec or κ̂ into micro Q without specific evidence they improve micro decisions
- Do not reopen 3-act / SOFT as default
- Do not add high-order nonlinear terms to κ̂ update
- Do not make observer updates more complex than necessary
- Do not change frozen A1 3D parameters

---

## Appendix: Canonical Configuration Summary

```
Observer: A1MtObserverFrozen (5D)
  Layer 1: (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)
  Layer 2: (τ̂, ν̂, γ̂_gen)   ← micro Q input
  Layer 3: full 5D           ← macro/diagnostic

  Frozen 3D params:
    β_τ_probe = 0.0, β_ν_probe = 0.0, β_γ_probe = 0.0
    λ_τ = λ_ν = λ_γ = 0.005

  γ̂_spec params:
    α_gs_resist = 0.03, α_gs_follow = 0.025
    lure_threshold = 0.3

  κ̂ params:
    κ_0 = 0.3, λ_κ = 0.02
    α_κ⁺ = 0.015, α_κ⁻ = 0.012
    risk_gate = 0.1

Micro Tutor: BCICTv4 (2-act)
  Action space: {WAIT, WARN}
  Score: Q(a | τ̂, ν̂, γ̂_gen, scene)

Macro κ Bonus:
  β_κ = 0.02 (minimum effective dose)
  g_κ = |κ̂ − κ_0| / (κ_max − κ_min)
  Applied to: L_risk = {tic_rescue_heavy, blind_corridor, warn_rescue}

Active Mask:
  Active_t = 1[a_oracle ≠ WAIT  OR  a_infer ≠ WAIT]

Tests: 55/55 passing
```
