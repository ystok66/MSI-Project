# Step 5 Contract Closure: micro_bayes_shadow_v2.1

## 1. Final Specification

### Module Identity

| Field | Value |
|:------|:------|
| **Name** | `micro_bayes_shadow_v2_1` |
| **File** | `src/teachers/micro_bayes_shadow_v2_1.py` |
| **Class** | `MicroBayesShadowV2_1` |
| **Layer** | Micro shadow (sits on top, does not reach into frozen modules) |
| **Action space** | `{WAIT, WARN}` — canonical 2-act, same as BCICTv4 |
| **State consumption** | 3D only: (τ̂, ν̂, γ̂_gen). **No** κ̂, **no** γ̂_spec |
| **Default mode** | `micro_policy_mode="micro_bayes_shadow_v2_1"` + `p_self_mode="posterior_C"` |

### Utility Formula

```
Q_v2.1(a) = β_T·TaskGain(a) + β_L·LearnGain_cc(a) − β_D·DepCost_lite(a) − β_C·C(a)
```

#### TaskGain

```
TaskGain(WARN) = Δs + λ_voi·dVOI + λ_risk·p_fail + λ_tempt·tempt − λ_u·p_undecided
TaskGain(WAIT) = λ_sd·p_self·Δs − λ_fail·p_fail + λ_u·p_undecided·0.5
```

#### LearnGain_cc (credit-corrected)

```
LearnGain_cc(a) = max(L_now − L_next, 0) · ρ_self(a)
ρ_self(WAIT) = 1.0
ρ_self(WARN) = 1 − λ_cc · p_directed
```

#### DepCost_lite

```
DepCost_lite(a) = λ_blind · p_blind(a)
p_blind = 0.7 (no self-evidence) or 0.2 (has self-evidence), × dose
```

#### Cost

```
C(WARN) = 1.0, C(WAIT) = 0.0
```

### Decision Rule

```
ΔQ = Q(WARN) − Q(WAIT)
N_t = w_f·p_fail − w_s·p_self − w_u·p_undecided + w_Δ·Δs + w_v·dVOI
WARN iff ΔQ > δ AND N > τ_N
```

### Default Parameters

| Parameter | Default | Provenance |
|:---------:|:-------:|:-----------|
| β_task | 1.5 | Step 2 rebalancing |
| β_learn | 2.5 | Step 2 rebalancing |
| β_dep | 2.0 | Step 2 compound constraint |
| β_cost | 1.5 | Step 2 (↑ from v1's 0.5) |
| λ_voi | 1.2 | Step 2 (↓ from v1's 2.0) |
| λ_risk | 0.8 | Step 2 (↓ from v1's 1.5) |
| λ_tempt | 0.8 | Step 2 |
| λ_sd | 2.0 | Carried from v1 |
| λ_fail | 1.5 | Carried from v1 |
| λ_undecided | 0.5 | Step 2 (new for three-outcome) |
| λ_blind | 1.0 | Step 2 (matched to lower β_dep) |
| λ_cc | 0.6 | Step 3 credit correction |
| δ_threshold | 0.5 | Step 2 gate |
| τ_necessity | 0.2 | Step 2 gate |
| w_fail | 2.0 | Step 2 necessity |
| w_self | 1.5 | Step 2 necessity |
| w_undecided | 0.8 | Step 2 necessity |
| w_delta_s | 1.0 | Step 2 necessity |
| w_dvoi | 0.8 | Step 2 necessity |

---

## 2. p_self_posterior_shadow Contract

### Unified API

All variants (`BASELINE`, `OLD_BLEND`, `POSTERIOR_A`, `POSTERIOR_B`, `POSTERIOR_C`)
now return the same schema:

```python
{
    "p_self": float,       # ∈ [0, 1]
    "p_fail": float,       # ∈ [0, 1]
    "p_undecided": float,  # ∈ [0, 1]
    "variant": str,
    "fallback": bool,
    # ... variant-specific diagnostics
}
# Invariant: p_self + p_fail + p_undecided = 1
```

### Event Definitions

| Event | Definition |
|:------|:-----------|
| **self** | Pr(learner discovers correct branch under do(WAIT), before commit/failure window closes) |
| **fail** | Pr(learner commits to wrong branch or times out under do(WAIT), irreversible error) |
| **undecided** | Pr(learner remains uncommitted at window end, neither discovered nor failed — still observing) |

### Recommended Default

`POSTERIOR_C` — the only variant that models p_undecided as an independent quantity
rather than computing it as a remainder.

---

## 3. Module Status

| Module | Status | Include by Default | Reason |
|:-------|:------:|:------------------:|:-------|
| micro_bayes_shadow_v2_1 | **PROMOTE** | ✅ | Converged best from Steps 1-4 |
| p_self_posterior_C | **DEFAULT** | ✅ | Binary rollback: ΔSelGap=−0.34 |
| credit_correction | **INCLUDED** | ✅ | Improves causal correctness |
| p_self_calibration | **DIAGNOSTICS** | ❌ | Zero policy impact |
| effort_latent_shadow | **DIAGNOSTICS** | ❌ | Causes WR regression in policy |
| micro_bayes_shadow (v1) | **ABLATION** | ❌ | Superseded |
| micro_bayes_shadow_v2 | **ABLATION** | ❌ | Superseded by v2.1 |
| micro_bayes_shadow_v3 | **ABLATION** | ❌ | Effort component regresses |

---

## 4. What This Module Does NOT Do

- Does NOT modify frozen 5D observer
- Does NOT modify canonical BCICTv4 default behavior (flag-off = unchanged)
- Does NOT expand micro action space beyond {WAIT, WARN}
- Does NOT consume κ̂ or γ̂_spec at micro layer
- Does NOT mix UNLOCK/ITEM_DROP into micro (those stay in option layer)
- Does NOT use effort_latent as policy input
- Does NOT use calibration as default policy input
- Does NOT do multi-step POMDP planning (single-point evaluation)

---

## 5. Known Failure Modes

1. **TBSR saturation**: TBSR = 0.500 across all arms in 30-step sessions.
   Longer episodes (100+) needed to see tutor impact on correctness.
2. **Credit leakage underreported**: The info pathway through BCICTv4 adapter
   doesn't surface v2.1 credit_info in all logging paths.
3. **Variant B fallback**: When branches aren't available, posterior_B
   falls back to variant A (documented, tracked via fallback field).

---

## 6. Recommended Usage

```python
from src.teachers.micro_bayes_shadow_v2_1 import MicroBayesShadowV2_1
from src.teachers.p_self_posterior_shadow import compute_p_self_posterior, PSelfMode

# Get three-outcome p_self
ps = compute_p_self_posterior(
    PSelfMode.POSTERIOR_C, d_commit, d_reveal,
    tau_hat=m.tau, nu_hat=m.nu, gamma_gen_hat=m.gamma_gen)

# Score
scorer = MicroBayesShadowV2_1()
action, dose, info = scorer.score(
    m, delta_s, dvoi, tempt, risk,
    ps["p_self"], ps["p_fail"], ps["p_undecided"],
    subtype, has_self_ev, zones)
```

Or through BCICTv4:
```python
tutor = BCICTv4(
    micro_policy_mode="micro_bayes_shadow_v2_1",
    p_self_mode="posterior_C")
```
