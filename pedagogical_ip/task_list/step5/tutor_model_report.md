# Tutor Model Report: Micro Tutor + Observer Mechanism

> **Handoff document** — self-contained reference covering the micro tutor (BCICTv4), observer (5D), behavior bridge, and pedagogical decision system.

---

## 1. System Overview

The tutor system has two interacting subsystems:

```
┌─────────────────────────────────────────────────────┐
│  Observer: estimates learner state from visible cues │
│  m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec_state, κ̂)             │
└────────────────────┬────────────────────────────────┘
                     │ m̂_t (3D view)
┌────────────────────▼────────────────────────────────┐
│  Micro Tutor (BCICTv4): decides WAIT or WARN        │
│  Q(WAIT | m̃_micro, scene) vs Q(WARN | m̃_micro, sc) │
└─────────────────────────────────────────────────────┘
```

**Key invariant**: The micro tutor **never reads true $m_t$**. In oracle mode it reads $m_t$ directly; in infer mode it reads $\hat m_t$ from the observer. The observer **never reads true $m_t$** — it only sees tutor-visible signals (`ObsEvent`).

---

## 2. Micro Tutor: BCICTv4

**Source**: `src/teachers/internalization_control_tutor_v4.py`

### 2.1 Action Space

| Config | Actions | Status |
|--------|---------|:------:|
| `bcictv4_2act.yaml` | {WAIT, WARN} | **Canonical** |
| `bcictv4_3act_legacy.yaml` | {WAIT, SOFT, WARN} | Legacy |

SOFT (dose=0.5) was proven redundant (P3-B exact-Q: $V_{SOFT}=0.005$, optimal volume ≈ 0). The canonical system uses **2-act only**.

### 2.2 Q Function Architecture

For each candidate dose $\omega \in \{0, 1\}$ (or $\{0, 0.5, 1\}$ in legacy):

$$Q(\omega) = Q_{online}(\omega) + \lambda_{teach} \cdot V_{full}(\omega) - \lambda_{over} \cdot R_{over}(\omega)$$

| Component | Formula | Role |
|-----------|---------|------|
| $Q_{online}^{warn}$ | $1.0 \cdot \Delta_s + 2.0 \cdot \Delta_{VOI} + 1.5(1-p_{self}) + 1.0 \cdot tempt - 0.05$ | Raw WARN value |
| $Q_{online}^{wait}$ | $2.0 \cdot p_{self} \cdot \Delta_s - 1.5 \cdot p_{fail} + 2.0$ | Raw WAIT value |
| $V_{full}$ | $V + \lambda_{sd} \cdot p_{sd} - \lambda_{dep} \cdot p_{blind}$ | Teaching value |
| $R_{over}$ | Overteach penalty from bridge | Overteaching cost |

### 2.3 Teaching Value Decomposition

$$V = L_{beh}(m_t) - L_{beh}(m_{t+1}^{predicted})$$

"Does this action move the learner's predicted behavior closer to the target zone?"

$$V_{full} = V + \lambda_{sd} \cdot p_{sd} - \lambda_{dep} \cdot p_{blind}$$

| Term | Weight | Meaning |
|------|:------:|---------|
| $V$ | (direct) | Behavior loss reduction |
| $p_{sd}$ | $\lambda_{sd}=1.5$ | Self-discovery opportunity bonus |
| $p_{blind}$ | $\lambda_{dep}=2.0$ | Blind obedience penalty |
| $\lambda_{teach}$ | 3.5 | Overall teaching weight |
| $\lambda_{over}$ | 4.0 | Overteaching penalty weight |

### 2.4 Hyper-Parameters

| Parameter | Value | Role |
|-----------|:-----:|------|
| $\lambda_{teach}$ | 3.5 | Weight of teaching value |
| $\lambda_{over}$ | 4.0 | Weight of overteach penalty |
| $\lambda_{sd}$ | 1.5 | Self-discovery bonus |
| $\lambda_{dep}$ | 2.0 | Blind-obey penalty |

---

## 3. Key Decision Signals

### 3.1 Self-Discovery Probability

**Source**: `src/metrics/self_discovery.py`

$$p_{self} = \sigma\Big(\frac{d_{commit} - d_{reveal} - m}{\tau_v}\Big)$$

| Condition | $p_{self}$ | Interpretation |
|-----------|:----------:|---------------|
| $d_c \gg d_r$ | → 1.0 | Learner sees truth before committing → WAIT |
| $d_c \ll d_r$ | → 0.0 | Learner commits blind → must WARN |
| $d_c \approx d_r$ | ≈ 0.5 | Ambiguous boundary |

$p_{self}$ is the **central signal** for the WAIT/WARN decision. High $p_{self}$ favors WAIT (let learner self-discover); low $p_{self}$ favors WARN (prevent blind commitment).

### 3.2 Failure-If-Wait Probability

$$p_{fail} = 1 - p_{self}(d_c, d_r; \tau_f=1.5)$$

Uses a slightly softer temperature ($\tau_f=1.5$ vs $\tau_v=1.0$) to bias toward safety.

### 3.3 Value-of-Information Delta

$$\Delta_s = \max\Big(|s_A^{full} - s_B^{full}| - |s_A^{vis} - s_B^{vis}|, 0\Big)$$

$$\Delta_{VOI} = \max\Big(\sigma(|s_A^{full} - s_B^{full}|) - \sigma(|s_A^{vis} - s_B^{vis}|), 0\Big)$$

How much more information the full branch summary reveals vs. what the learner can currently see. Higher $\Delta_{VOI}$ → warning is more informative.

---

## 4. Behavior Bridge

**Source**: `src/agents/behavior_bridge.py`

### 4.1 Purpose

Maps internalization state $m$ to predicted behavioral probes $\hat{z}$:

$$\hat{z}_p(m, c) = \sigma(w_p \cdot \phi(m, c))$$

where $\phi$ is a 13-dimensional feature vector:

| Feature | Formula | Captures |
|---------|---------|----------|
| bias | 1.0 | Intercept |
| κ, τ, ν, γ_spec, γ_gen | raw state | Direct state influence |
| κ × risk | $\kappa \cdot r$ | Risk-calibrated interaction |
| γ_spec × lure | $\gamma^{spec} \cdot l$ | Temptation interaction |
| γ_gen × novelty | $\gamma^{gen} \cdot n$ | Exploration suppression interaction |
| τ × (1−self_ev) | $\tau \cdot (1-s)$ | Trust under low self-evidence |
| ν × self_ev | $\nu \cdot s$ | Dependence under high self-evidence |
| κ², γ_gen² | quadratic | Nonlinear saturation |

### 4.2 Pre-Calibrated Weights

| Probe | Key Positive Drivers | Key Negative Drivers |
|:-----:|---------------------|---------------------|
| RC | κ (+0.9), κ×risk (+1.2), bias (+1.8) | γ_gen (−0.3) |
| TR | γ_spec (+1.2), γ_spec×lure (+1.5), bias (+1.5) | γ_gen (−0.2) |
| EP | bias (+0.3) | γ_gen (−2.5), γ_gen×novelty (−1.8), γ_gen² (−1.0), ν (−0.3) |
| VA | τ (+1.8), τ×(1−self_ev) (+2.0) | ν (−1.5), ν×self_ev (−1.5) |
| IA | ν (+2.5), ν×self_ev (+2.0), γ_gen (+0.5) | bias (−1.5), τ (−0.3), τ×(1−self_ev) (−0.5) |

### 4.3 Behavior Loss

$$L_{beh}(m) = \sum_p w_p \cdot \text{band\_loss}(\hat{z}_p, lo_p, hi_p)$$

where `band_loss(x, lo, hi) = max(lo−x, 0)² + max(x−hi, 0)²`

| Probe | Weight | Focus |
|:-----:|:------:|-------|
| RC | 1.0 | Risk calibration |
| TR | 1.2 | Temptation resistance |
| EP | **2.5** | Exploration preservation (highest priority) |
| VA | 1.5 | Valid advice uptake |
| IA | **2.5** | Invalid advice resistance (highest priority) |

EP and IA have the **highest weights** — the system prioritizes preventing over-suppression and maintaining learner independence.

### 4.4 Overteach Penalty

$$R_{over} = 2.5 \cdot [\hat{z}_{IA} - hi_{IA}]_+^2 + 2.5 \cdot [lo_{EP} - \hat{z}_{EP}]_+^2 + 1.5 \cdot [\hat{z}_{TR} - hi_{TR}]_+^2$$

Penalizes: invalid-advice suppression too high (learner rejects everything), exploration too low (over-suppressed), temptation resistance too high (learner avoids all novelty).

### 4.5 Empirical Zone Calibration

Behavior zones are **not** hard-coded — they are calibrated from baseline rollout quantiles:

$$lo_p = Q_\alpha(\hat{z}_p), \quad hi_p = Q_{1-\alpha}(\hat{z}_p), \quad \alpha = 0.15$$

This ensures the target zone adapts to the actual distribution of bridge predictions in the current episode.

---

## 5. Observer: 5D State Estimator

**Source**: `src/teachers/internalization_observer.py`

### 5.1 Class Hierarchy

| Class | Dims | Role |
|-------|:----:|------|
| `RuleBasedMtObserver` | 3 (τ̂,ν̂,γ̂_gen) | Base observer (A0) |
| `A1MtObserver` | 5 (+γ̂_spec,κ̂) | Extended with probe fixes + conditional reversion |
| `A1MtObserverFrozen` | 5 | **Canonical** — parameter guards, immutable 3D core |
| `A2MtObserver` | 5 | Research variant (not canonical) |

### 5.2 Input: ObsEvent

The observer receives a structured event after each episode step:

| Field | Type | Source |
|-------|------|--------|
| `dose` | float | Tutor action (0/0.5/1.0) |
| `warned` / `follow_warn` | bool | Did tutor warn? Did agent follow? |
| `d_commit` / `d_reveal` | int | Corridor structure |
| `p_self` | float | Tutor-computed self-discovery probability |
| `lure` / `risk` | float | Scene parameters |
| `risk_hat` | float | Agent's predicted risk (for κ̂) |
| `agent_choice` / `oracle_safe` | int | Branch chosen vs. oracle answer |
| `self_discovery` | bool | Agent found safe path without warning |
| `probe_VA/IA/EP` | float? | Behavioral probes (when available) |

**Critical rule**: `m_true` is recorded for logging ONLY. The observer **never reads it**.

### 5.3 Base 3D Update (A0: RuleBasedMtObserver)

#### Soft Events (from tutor-visible signals)

| Event | Formula | Drives |
|-------|---------|--------|
| $e_{trust+}$ | `warned ∧ follow_warn ∧ warn_correct` | τ̂ ↑ |
| $e_{trust-}$ | `warned ∧ follow_warn ∧ warn_wrong` | τ̂ ↓ |
| $e_{blind}$ | `warned ∧ follow_warn × (1 − p_self)` | ν̂ ↑ |
| $e_{selfdisc}$ | `self_discovery × p_self` | ν̂ ↓ |
| $e_{pressure}$ | EMA of dose | γ̂_gen ↑ |
| $e_{explore+}$ | `novelty ∨ self_discovery` | γ̂_gen ↓ |

#### τ̂ Update

$$\hat\tau_{t+1} = (1-\lambda_\tau)\hat\tau_t + \lambda_\tau \tau_0 + \alpha_\tau^+ e_{trust+}(1-\hat\tau_t) - \alpha_\tau^- e_{trust-}\hat\tau_t + \beta_\tau^{probe} r_{VA}$$

| Param | Value | vs Agent |
|-------|:-----:|:--------:|
| $\alpha_\tau^+$ | 0.22 | 0.25 (conservative) |
| $\alpha_\tau^-$ | 0.10 | 0.12 (conservative) |
| $\beta_\tau^{probe}$ | 0.15 → **0.0** (A1) | Probes OFF |
| $\lambda_\tau$ | 0.02 → **0.005** (A1) | Slow reversion |

#### ν̂ Update

$$\hat\nu_{t+1} = (1-\lambda_\nu)\hat\nu_t + \lambda_\nu \nu_0 + \alpha_\nu^+ e_{blind}(\nu_{max}-\hat\nu_t) - \alpha_\nu^- e_{selfdisc}\hat\nu_t + \beta_\nu^{probe} r_{IA}$$

| Param | Value | vs Agent |
|-------|:-----:|:--------:|
| $\alpha_\nu^+$ | 0.18 | 0.20 (conservative) |
| $\alpha_\nu^-$ | 0.13 | 0.15 (conservative) |
| $\beta_\nu^{probe}$ | 0.10 → **0.0** (A1) | Probes OFF |

#### γ̂_gen Update

$$\hat\gamma^{gen}_{t+1} = (1-\lambda_\gamma)\hat\gamma^{gen}_t + \lambda_\gamma \gamma_0 + \alpha_\gamma^+ e_{pressure}(\gamma_{max}-\hat\gamma^{gen}_t) - \alpha_\gamma^- e_{explore+}\hat\gamma^{gen}_t + \beta_\gamma^{probe} r_{EP}$$

| Param | Value | vs Agent |
|-------|:-----:|:--------:|
| $\alpha_\gamma^+$ | 0.07 | 0.08 (conservative) |
| $\alpha_\gamma^-$ | 0.10 | 0.12 (conservative) |
| $\beta_\gamma^{probe}$ | 0.10 → **0.0** (A1) | Probes OFF |

### 5.4 Per-Dimension Confidence

Each dim has a confidence $c \in [0,1]$, updated via EMA:

$$c_{t+1} = (1-\rho) c_t + \rho \cdot eq$$

where $eq$ is evidence quality (max of event amplitude, probe availability, p_self discriminability).

### 5.5 A1 Improvements Over A0

| Feature | A0 | A1 |
|---------|:--:|:--:|
| Probe correction weights | 0.15/0.10/0.10 | **0.0** (OFF) |
| Mean-reversion λ | 0.02 | **0.005** (gentler) |
| Conditional reversion | Always-on | **Confidence-gated** |
| γ̂_spec | ✗ | ✅ (temptation resistance) |
| κ̂ | ✗ | ✅ (risk calibration) |
| Parameter freeze | ✗ | **`__setattr__` guard** |

### 5.6 γ̂_spec Update (Dim 4)

See Stage Summary for formula. Only fires when `lure ≥ 0.3`. Tracks behavioral state, not trait.

### 5.7 κ̂ Update (Dim 5)

See Stage Summary for formula. Uses signed risk error with event gating. Slow variable (λ_κ=0.02).

---

## 6. State Prediction Model

The tutor uses `_predict_m()` to simulate forward the internalization state under each candidate action:

```python
for dose in [0.0, 1.0]:
    mc = _predict_m(m, dose, tempt, risk, subtype, has_self_ev)
    # Evaluate V = L_beh(m) - L_beh(mc)
```

| Action | Predicted Effects |
|--------|-------------------|
| WARN (dose=1.0) | risk↓ (safe update), trust↑ (if warranted), dependence↑ (if blind), γ_gen↑ (pressure) |
| WAIT (dose=0.0) | risk stays, self-discovery possible → dependence↓, γ_gen↓ (if exploration) |
| SOFT (dose=0.5) | Interpolated: half the ν/γ push, half the online Q | Legacy |

### Subtype-Aware Prediction

| Subtype | Special Handling |
|---------|-----------------|
| `self_discovery_needed` / `self_discovery_teach` | High p_sd when dose=0 |
| `false_suppression_cost` / `beneficial_novelty` | Successful exploration under WAIT |
| Others | Standard prediction |

---

## 7. Shadow Refactors (Research Only)

Three research-only refactors exist, all **default OFF**:

| Refactor | Flag | Purpose |
|----------|------|---------|
| EPU Shadow | `use_epu_shadow` | One-step rollout expected pedagogical utility |
| Belief-Horizon p_self | `use_belief_horizon_pself` | Hybrid geometric + belief-based self-discovery |
| EIG Observation | `use_eig_observation` | Mutual information $I(A; \theta)$ for wait value |

These are shadow-mode computations logged alongside the canonical decision for comparison. They do not affect the canonical action.

---

## 8. Macro Score: κ̂ Bonus

While the micro tutor uses only 3D, the **macro TEACH scorer** can use κ̂:

$$S^{5d}_{teach}(\ell | x_t) = S^{base}_{teach}(\ell | x_t) + 0.02 \cdot \mathbf{1}[\ell \in \mathcal{L}_{risk}] \cdot \frac{|\hat\kappa_t - 0.3|}{1.0}$$

This is the **only** point where dims 4–5 currently influence any score. It is a controlled, minimal, macro-only integration.

---

## 9. Decision Pipeline Summary

```
1. Read scene: risk, tempt, d_commit, d_reveal, subtype, branches
2. Compute p_self = σ((d_commit − d_reveal) / τ)
3. Compute p_fail = 1 − p_self(τ_f=1.5)
4. Summarize visible vs. full branches → Δ_s, Δ_VOI
5. For each dose ω ∈ {0, 1}:
   a. Predict m̃(t+1) = _predict_m(m, ω, ...)
   b. Compute V = L_beh(m) − L_beh(m̃)
   c. Compute V_full = V + λ_sd·p_sd − λ_dep·p_blind
   d. Compute R_over = bridge_overteach_penalty(m̃)
   e. Q(ω) = Q_online(ω) + λ_teach·V_full − λ_over·R_over
6. best_action = argmax Q(ω)
7. Return action ∈ {WAIT, WARN}
```

---

## 10. Source File Index

| File | Lines | Role |
|------|:-----:|------|
| `src/teachers/internalization_control_tutor_v4.py` | 297 | BCICTv4 micro tutor: Q function, dose control, shadow refactors |
| `src/teachers/internalization_observer.py` | 685 | 5D observer: A0→A1→A1Frozen class hierarchy, all update rules |
| `src/agents/behavior_bridge.py` | 115 | Bridge: m→ẑ mapping, zone calibration, loss, overteach penalty |
| `src/agents/behavior_probes.py` | ~150 | Raw probe definitions and default zones |
| `src/agents/internalization_state_v3.py` | ~250 | True 5D state (κ,τ,ν,γ_spec,γ_gen), update methods |
| `src/metrics/self_discovery.py` | 73 | p_self and p_fail computation |
| `src/envs/observation_mask.py` | ~100 | Branch visibility mask from observation radius |
| `src/agents/branch_summary.py` | ~100 | Summarize visible branch features |
| `configs/tutor/bcictv4_2act.yaml` | ~20 | 2-act canonical config |
| `configs/observer/a1_5d_kappa.yaml` | ~40 | 5D observer canonical config |

---

## 11. Key Design Principles

1. **Don't read true state**: Observer and tutor operate on estimated $\hat m_t$, never $m_t$
2. **Conservative estimation**: All observer α terms are 85–90% of agent's true values
3. **Probe corrections OFF**: A1 turns off all β_probe terms (they introduced estimation noise)
4. **Slow reversion**: Mean-reversion λ reduced from 0.02 to 0.005 (prevent washing out learned signal)
5. **Conditional reversion**: Only reverts toward prior when confidence is low AND evidence is sparse
6. **Estimate first, consume later**: New dims enter Layer 1 but don't enter scores until proven
7. **Minimal Q coupling**: micro Q uses exactly 3 dims + scene context; no high-order interactions
8. **Overteaching as first-class concern**: EP and IA have 2.5× weight in behavior loss
