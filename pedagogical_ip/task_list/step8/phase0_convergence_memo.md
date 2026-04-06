# Phase 0 Convergence Memo

**Date**: 2026-04-06  
**Status**: Q1=partial, Q2=deferred, Q3=complete, Q4=complete

---

## A. Warning Audit Verdict

### Finding
The `warning_variant` parameter (legacy_bias / rsa_obs_s1 / rsa_obs_s1_trust / rsa_plus_phase10) **only routes the legacy baseline_v2 family's segment-level warning**. The DTMB and GTET families have their own warning implementations (`apply_dtmb_warning`, `_apply_gtet_warning`) that are **completely independent** of the RSA/legacy variant selection.

### Evidence
- Ran 20 seeds × 4 warning variants on DTMB-medium
- **All 4 variants produced identical outcomes**: surv=0.450, goal=0.100, 241 warning events
- Source quantities (Δρ, lane_bias_mass, rsa_delta_rho) are all 0.0
- This is because DTMB warnings go through `apply_dtmb_warning()` (line 675 of lattice_v2_runner.py) not through `_apply_segment_warning`

### Verdict
> **The warning_variant routing question is moot for the two primary families (DTMB, GTET).** The RSA channel exists only for the original `baseline_v2` lattice family but is never exercised by the canonical evaluation families.

**Recommendation**: 
- RSA channel can be **retained as infrastructure** but is not actively differentiating decisions
- If future families want RSA semantics, they need explicit integration (DTMB/GTET styles)
- The legacy/RSA convergence question does not block Phase 1

---

## B. Transfer Audit Verdict

### Status: Deferred
The Q2 script (`phase0_transfer_capacity_audit.py`) with the shadow structured basis head is ready but requires the `PRSSession` infrastructure which has significant runtime. This audit should be run separately.

### Existing strong evidence (from PRS-2 audit)
- stateful ≈ stateless (APD 95% CI crosses 0)
- LR audit: StateGain ≈ 0 across 30× lr range
- Root cause identified: 10-parameter linear model converges within ~50 samples (1 episode)

### Provisional Verdict (pending Q2 confirmation)
> **"继续调 lr/线性头" 正式结束。** Whether structured basis helps is the one remaining sub-question.
>
> If basis also shows zero StateGain → accept negative result  
> If basis shows positive StateGain → proceed to slow-fast / dual-timescale

---

## C. GTET Temptation Verdict

### Finding
20 seeds × 4 factor_modes (FULL, G_THETA, G_Z, THETA_Z) produced **identical outcomes**:

| Metric | Value |
|--------|-------|
| Survival | 0.950 |
| Goal rate | 0.950 |
| Δg (goal update mass) | 0.0378 |
| Δθ (pref update mass) | 0.0583 |
| Δz (tempt update mass) | 0.0373 |
| Δz/Δg ratio | 0.988 |

### Key insight
The posterior **always** updates all three marginals regardless of `factor_mode`. The `factor_mode` only controls which marginals the **tutor** uses for decision-making. z has nearly identical update mass to g (Δz/Δg ≈ 0.99).

**But**: removing z from the tutor's decision weights (G_THETA mode) produces **zero performance loss** (surv=0.950 identical to FULL). z is being maintained by the posterior but contributes nothing to the tutor's decisions.

### Verdict
> **z 从 canonical posterior 中降级为 optional plugin。**
>
> - 默认 canonical: `q(g,θ)` only (P4 no-z, which is already the default predictor)
> - z: retained in posterior computation but excluded from tutor weights
> - z: only activated in specialized temptation/hidden-preference families
>
> **This question is closed.**

---

## D. Time-Learning Closure Verdict

### Finding
20 seeds × 2 families × 3 tutor modes with per-step uncertainty/stall tracing:

| Family | TutorMode | BoreRatio | FP_wait | StallCost |
|--------|-----------|-----------|---------|-----------|
| DTMB | selective | 0.468 | 0.000 | 22.4 |
| DTMB | always_warn | 0.468 | 0.000 | 22.4 |
| DTMB | no_tutor | 0.480 | 0.000 | 19.1 |
| **GTET** | **selective** | **0.530** | **0.800** | **26.4** |
| **GTET** | **always_warn** | **0.530** | **0.800** | **26.4** |
| GTET | no_tutor | 0.529 | 0.000 | 25.9 |

### Key findings
1. **BoreRatio ≈ 50%** across all conditions: Half of all movement cost produces zero information gain. This is a structural property of the grid, not a tutor failure.
2. **GTET selective FP_wait = 0.800**: When the tutor chooses WAIT, 80% of the time the agent's next step produces no uncertainty reduction. The tutor is **excessively conservative** about intervening.
3. **DTMB FP_wait = 0.000**: DTMB agents die quickly (surv=0.45), so WAIT decisions are very rare.
4. **BoreRatio is identical across tutor modes**: This suggests the stall pattern is a property of the **agent's exploration dynamics**, not the tutor. The tutor's WAIT decision exacerbates it (high FP_wait) but doesn't cause it.

### Verdict
> **VERDICT A: Gap 真实存在。当前 canonical utility 还未闭环。**
>
> - `FP_wait = 0.80` means WAIT is almost always a false positive in GTET
> - `BoreRatio = 0.53` means over half of exploration effort produces no learning
> - Phase 1 should add a **boredom/frustration penalty** to `Q_WAIT`:
>
>   `Q_WAIT += w_bore · 1[IG ≤ 0] · FC`
>
> This would penalize WAIT when the agent is in a low-information-gain region.

---

## Phase 0 Completion Status

| Question | Status | Verdict |
|----------|--------|---------|
| Q1: Warning path | **Resolved** | Moot for primary families; RSA/legacy distinction irrelevant |
| Q2: Transfer capacity | **Deferred** | Provisionally: lr/linear head tuning ended; basis test pending |
| Q3: GTET temptation | **Closed** | z demoted to optional plugin |
| Q4: Time-learning | **Closed** | Gap is real; boredom item needed in Phase 1 Q_WAIT |
