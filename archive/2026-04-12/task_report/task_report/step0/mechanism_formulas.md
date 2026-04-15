# CLS Option Tutor — Mechanism & Formula Reference

This document provides a complete mathematical specification of every learner and tutor computation, referenced directly from the source code.

---

## 1. Notation

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| Y* = (y*₁, ..., y*_L) | Target output sequence | e.g. (GREEN, BLUE, GREEN, BLUE) |
| ν_j | Program (word sequence) of option j | e.g. (emit, kiki, emit, blicket) |
| Ŷ_j = F_G(ν_j) | Rendered output of program ν_j under grammar G | (GREEN, BLUE, GREEN, RED) |
| K | Number of options per menu | 10 |
| L | Length of target output | varies (1–8) |
| T_max | Maximum rounds per query | 5 |
| H_0 | Initial HP per query | 5 |
| v_j ∈ ℝ^m | Danger feature vector for option j | m = 16 |
| d_j ∈ {0,1,2,3,4} | Damage dealt by option j (0 = safe) | — |
| w = (w₁, ..., w_L) | Attention weights over target cells | sum to 1 |

---

## 2. Learner: Semantic Scoring

### 2.1 Oracle Scorer (DeterministicSemanticScorer)

The oracle has access to grammar G and computes exact renders.

**Weighted Mismatch:**

```
M_t(j) = Σ_{ℓ=1}^{L} w_ℓ · 𝟙[Ŷ_{j,ℓ} ≠ y*_ℓ]
```

Where w_ℓ are the current attention weights (see §4).

**Semantic Score:**

```
S_sem(j) = − M_t(j) / τ_sem
```

With τ_sem = 1.0 (default).

**Length mismatch**: If |Ŷ_j| ≠ L, missing cells count as mismatches:
```
M_t(j) = Σ_{ℓ=1}^{min(L,|Ŷ_j|)} w_ℓ · 𝟙[Ŷ_{j,ℓ} ≠ y*_ℓ]  +  Σ_{ℓ=min+1}^{L} w_ℓ
```

**Posterior Probabilities:**

```
P_L(j | Y*, G) = softmax(S_sem)_j = exp(S_sem(j)) / Σ_k exp(S_sem(k))
```

### 2.2 CLS Scorer (CLSSemanticPosterior)

The CLS learner does NOT have access to G. It learns rules from support examples via EM.

**Study Phase:**
```
support[:n_sup] → CLSAgent.reset_episode() → CLSAgent.study(support)
                                            → n_em=2 EM iterations
                                            → cortex: probabilistic grammar rules
                                            → hippocampus: exact example storage
```

**Prediction:**
```
ν_j → CLSAgent.predict(ν_j) → Ŷ_j (may be incorrect!)
```

**Score**: Same formula as §2.1 but using CLS predictions instead of oracle renders.

**Incremental Study** (Phase 3 only):
```
After wrong-pick reveal of option j:
  new_example = (ν_j, render(ν_j))
  CLS.incremental_study([new_example])  → re-runs EM with expanded dataset
```

**Freeze** (Phase 4):
```
CLS.freeze()  → stops accepting new study data
```

---

## 3. Learner: Risk Model (DangerHead)

Two-layer architecture: HazardHead (binary classifier) + SeverityHead (regression).

### 3.1 Feature Expansion

```
φ(v) = [v; v ⊙ v; 1] ∈ ℝ^{2m+1}
```

Where v ∈ ℝ^m is the danger feature vector and ⊙ denotes element-wise product. For m = 16, φ(v) has dimension 33.

### 3.2 HazardHead: P(risky | v)

Online logistic regression:

```
p_h(v) = σ(wₕᵀ φ(v))     where σ(z) = 1/(1+e^{-z})
```

**Initialization** (safe-biased prior):
```
wₕ = [0, 0, ..., 0, -1.0]
→ p_h(v) = σ(-1) ≈ 0.27  for all v (assumes mostly safe)
```

**Update** (online cross-entropy gradient descent):
```
Given observation (v, y_h) where y_h ∈ [0, 1]:
  p = p_h(v)
  grad = (p - y_h) · φ(v)          # cross-entropy gradient
  reg = wₕ / σ²_prior              # L2 regularization (decaying)
  wₕ ← wₕ - lr · (grad + reg / (n_updates + 1))
```

Learning signals:
- Wrong-pick reveal with d > 0: y_h = 1.0
- Wrong-pick reveal with d = 0: y_h = 0.0
- RISK_HINT: y_h = η = 0.8 (weak label)

### 3.3 SeverityHead: E[d | v, d > 0]

Online linear regression, only updated on risky observations:

```
μ_s(v) = clamp(wₛᵀ φ(v), 1, 4)
u_s(v) = 1 / (1 + 0.1 · n_updates)  # decreasing uncertainty
```

**Update** (online MSE gradient descent):
```
Given observation (v, d) where d > 0:
  grad = (μ_s(v) - d) · φ(v)
  reg = wₛ / σ²_prior
  wₛ ← wₛ - lr · (grad + reg / (n_updates + 1))
```

### 3.4 Composite Prediction

```
μ_d(v) = p_h(v) · μ_s(v)           # expected damage
u_d(v) = p_h(v) · u_s(v)           # composite uncertainty
```

### 3.5 KO Probability

```
P_KO(v, HP) = p_h(v) · max(0, μ_s(v) / HP)
```

If μ_s(v) ≥ HP, then P_KO = p_h(v) (full KO risk).

---

## 4. Learner: Attention Model

### 4.1 Default

```
w_ℓ = 1/L   for all ℓ ∈ {1, ..., L}   (uniform)
```

Attention is **reset per query** (`init_for_query(L)`).

### 4.2 HIGHLIGHT Boost

When tutor highlights cell set H ⊆ {1, ..., L}:

```
w_ℓ^(H) ∝ w_ℓ · exp(ρ_H · 𝟙[ℓ ∈ H])
```

With ρ_H = 2.0:
- Highlighted cell weight: w_ℓ × e^2.0 ≈ w_ℓ × 7.39
- Non-highlighted: unchanged
- Then re-normalize so Σ w_ℓ = 1

**Example** (L=4, highlight cell 2):
```
Before: [0.25, 0.25, 0.25, 0.25]
After:  [0.10, 0.10, 0.71, 0.10]   (cell 2 gets 71% of attention)
```

HIGHLIGHT **persists through refresh** (text options don't change in V2).

---

## 5. Learner: Episodic Memory

Tracks previously-tried wrong options. Applies elimination penalty:

```
penalty(j) = -|{past wrong picks matching j}| × α_elim
```

This discourages re-picking known-wrong options (within the same query).

---

## 6. Learner: Policy

### 6.1 Pick Utility

For each active option j (not banned):

```
U_pick(j) = α_sem · S_sem(j)
           - α_risk · μ_d(j)
           - α_unc · u_d(j)
           + penalty_memory(j)
```

| Weight | Value | Role |
|--------|-------|------|
| α_sem | 1.0 | Semantic match importance |
| α_risk | 0.5 | Danger avoidance |
| α_unc | 0.2 | Uncertainty avoidance |

### 6.2 Refresh Decision (Deterministic Threshold)

```
j* = argmax_j S_sem(j)          # best semantic option

IF μ_d(j*) ≥ HP  AND  rounds_used < T_max - 1:
    action = REFRESH              # re-roll risk, costs 1 round
ELSE:
    action = sample from π(a)     # pick from softmax
```

**Intuition**: "If my best option will probably kill me, let me re-roll the risk assignments. Otherwise, just pick."

Refresh has **no count limit** — each refresh costs 1 round (same as a pick attempt).

### 6.3 Action Probabilities (Softmax with Lapse)

When not refreshing:

```
U = [U_pick(1), ..., U_pick(K), U_refresh]

π'(a) = (1 - ε) · softmax(β_L · U) + ε · Uniform(K+1)
```

With β_L = 4.0 (temperature) and ε = 0.05 (lapse rate).

The ε-lapse ensures the learner occasionally explores non-optimal options — modeling realistic human behavior (attention lapses, curiosity).

---

## 7. Tutor: Profile Inference

At the transition from Observation → Teaching (after N_obs queries):

### 7.1 Data Collection

During observation, the system records `PolicyStateSnapshots`:
```
For each learner action:
  - semantic_scores[j]           # learner's option rankings
  - danger_preds[j]              # learner's risk estimates
  - attention_weights[ℓ]         # current attention
  - hazard_posterior_mean         # HazardHead weights
  - learner_action               # "pick" or "refresh"
  - learner_pick_index            # which option chosen
```

### 7.2 Profile State

```python
@dataclass
class ProfileState:
    semantic_competence: float = 0.5    # estimated grammar knowledge
    risk_awareness: float = 0.5         # estimated danger sensitivity
    g_highlight: float = 1.0            # highlight responsiveness
```

The profile inference module estimates these from the observation traces using RSA-style reasoning.

---

## 8. Tutor: Counterfactual Intervention Scoring

The tutor computes Q-values for all legal interventions and selects the best.

### 8.0 Learner Pick Probability Estimate

The tutor estimates what the learner will do:

```
P_L(j) = softmax(β · (sc · S_sem(j) - μ_d(j)))
```

Where:
- β = 4.0 (learner's softmax temperature)
- sc = profile.semantic_competence (scales semantic vs. random)
- S_sem(j) = tutor's oracle semantic scores
- μ_d(j) = tutor's danger predictions

**Expected damage under current learner policy:**
```
E_damage = Σ_j P_L(j) · μ_d(j)
```

### 8.1 Q(WAIT) — Baseline

```
Q(WAIT) = 0
```

All other actions are evaluated relative to WAIT.

### 8.2 Q(BAN, j) — Remove Option

```
Q(BAN, j) = β_safe · danger_j · P_L(j)  -  0.5 · sem_penalty(j)  -  c_ban
```

Where:
```
sem_penalty(j) = max(0, S_sem(j) / max(|min(S_sem)|, 10⁻⁵))
```

This penalizes banning the semantically best option (which is likely correct).

| Param | Value |
|-------|-------|
| β_safe | 1.5 |
| c_ban | 0.0 |

### 8.3 Q(RISK_HINT, j) — Warn About Risk

```
Q(RISK_HINT, j) = β_safe · p_h_tutor(j) · P_L(j)  -  c_hint
```

Where `p_h_tutor(j) = min(1, μ_d(j) / 2)` (rough hazard proxy).

| Param | Value |
|-------|-------|
| β_safe | 1.5 |
| c_hint | 0.3 |

**Current issue**: With safe-biased hazard prior (p_h ≈ 0.27), `Q(RISK_HINT) < 0 < Q(WAIT)` always. RISK_HINT is never selected.

### 8.4 Q(HIGHLIGHT, H) — Highlight Cells

```
Q(HL, H) = β_IG · Discrimination(H)  -  β_over · |H|/L  -  c_hl
```

**Discrimination** measures how well highlighting cell set H separates correct from incorrect options:

```
disc(ℓ) = 𝟙[correct matches at ℓ] · (# incorrect mismatching at ℓ) / (K-1)
Discrimination(H) = Σ_{ℓ ∈ H} disc(ℓ)
```

Detailed per-cell computation:
```
For each cell ℓ:
  1. Render correct option (best oracle score): Ŷ_correct
  2. Check: does Ŷ_{correct,ℓ} = y*_ℓ?
     - No → disc(ℓ) = 0 (cell doesn't help if correct is wrong here)
     - Yes → count how many incorrect options j have Ŷ_{j,ℓ} ≠ y*_ℓ
             disc(ℓ) = count / (K-1)
```

**Candidate generation**: All single cells (ℓ,) and adjacent pairs (ℓ, ℓ+1).

| Param | Value |
|-------|-------|
| β_IG | 1.0 |
| β_over | 0.2 |
| c_hl | 0.0 |
| max_highlight_cells | 2 |

### 8.5 Q(SKIP) — Skip Query

```
Q(SKIP) = β_mastery · P_corr
         + β_certainty · (1 - H_sem / H_max)
         + β_time · r
         + β_hp · h
         - β_learn · LG
         - c_skip
```

Where:
```
P_corr = max_j P_L(j)                           # learner's confidence
H_sem = entropy of softmax(S_sem)                # semantic entropy
H_max = ln(K)                                    # maximum entropy
r = rounds_used / T_max                          # time pressure
h = 1 - HP / (β_hp · H_0)                       # HP pressure
LG = H_sem / H_max                              # potential learning gain
```

| Param | Value |
|-------|-------|
| β_mastery | 0.8 |
| β_certainty | 0.4 |
| β_time | 0.3 |
| β_hp | 1.0 |
| β_learn | 0.5 |
| c_skip | 1.5 |

**Current issue**: c_skip = 1.5 is too high. Tutor's mastery estimate never reaches the threshold. SKIP is never selected.

---

## 9. Tutor: Action Selection

```python
candidates = scorer.score_all(qs, profile, oracle_scorer, danger_head)
# candidates sorted by total_q descending
best = candidates[0]

# Phase-gated execution:
if block.in_observation_phase or block.in_evaluation_phase:
    action = "WAIT"         # forced
elif block.in_teaching_phase:
    action = best.action    # best Q-value
```

---

## 10. Environment: Risk Model

### 10.1 Danger Vector Generation

Each block generates a danger model with risk_classes clusters:

```
For each risk class r ∈ {1, 2, 3, 4}:
  center_r ~ N(0, I_m)                       # cluster center
  
For each risky option with risk_class r:
  v_j ~ N(center_r, σ²_cluster · I_m)        # σ_cluster = 0.5
  damage_j = r

For safe options (risk_class = 0):
  v_j ~ N(0, I_m)                            # no cluster structure
  damage_j = 0
```

### 10.2 Menu Composition

Per query: 6 safe + 4 risky (risk classes sampled from {1,2,3,4}).

### 10.3 Refresh

Refresh re-samples risk assignments (new danger vectors, new risk classes) but keeps the same text options. Clears BAN and RISK_HINT interventions. Preserves HIGHLIGHT.

---

## 11. Environment: V2 Option Generation (ProgramPool)

### 11.1 Pool Construction

```
For grammar G, enumerate all valid programs:
  programs = {ν : |ν| ≤ 5 and |F_G(ν)| ≤ 8}
```

All programs are guaranteed renderable (no NONE outputs).

### 11.2 Menu Generation

```
1. Place correct option: (ν_correct, Y*)
2. For remaining K-1 slots:
   a. Compute per-cell diff: diff(ν_j) = {ℓ : F_G(ν_j)_ℓ ≠ y*_ℓ}
   b. Sample distractors maximizing cell diversity
      (prefer options that differ at different cells)
3. Assign risk classes: 6 safe + 4 risky
4. Shuffle and assign indices
```

---

## 12. Complete Decision Flow: One Round

```
┌─────────────────────────────────────────────────────────────────┐
│ TUTOR TURN                                                       │
│                                                                   │
│  1. Read current QueryState (target, menu, HP, round, bans...)   │
│  2. If first teaching query → infer profile from obs traces      │
│  3. Compute P_L(j) = softmax(β·(sc·S_sem - μ_d))               │
│  4. Score all candidates:                                        │
│     - Q(WAIT) = 0                                                │
│     - Q(BAN, j) = β_safe·danger_j·P_L(j) - sem_penalty - c_ban  │
│     - Q(HINT, j) = β_safe·p_h·P_L(j) - c_hint                  │
│     - Q(HL, H) = β_IG·disc(H) - β_over·|H|/L - c_hl            │
│     - Q(SKIP) = mastery + certainty + time + hp - learn - c_skip │
│  5. Execute best action (or WAIT if obs/eval phase)              │
├─────────────────────────────────────────────────────────────────┤
│ LEARNER TURN                                                     │
│                                                                   │
│  1. Read highlight → update attention: w_ℓ ∝ w_ℓ·exp(ρ_H·𝟙[ℓ∈H])│
│  2. Read risk_hint → update hazard head (weak label y=0.8)       │
│  3. Compute semantic scores: S_sem(j) = -M(j)/τ_sem             │
│  4. Compute danger predictions: μ_d(j) = p_h(v_j)·μ_s(v_j)     │
│  5. Compute utilities:                                           │
│     U_pick(j) = α_sem·S_sem(j) - α_risk·μ_d(j) - α_unc·u_d(j) │
│  6. Refresh check: if μ_d(j*) ≥ HP → REFRESH                   │
│  7. Otherwise: sample action from softmax(β_L·U) + ε-lapse      │
│  8. If wrong pick:                                               │
│     a. Reveal (ν_j, F_G(ν_j), damage)                           │
│     b. Update hazard + severity heads                            │
│     c. If teaching phase: CLS.incremental_study(reveal)          │
│  9. If refresh: re-roll risk (keep text), clear BAN/HINT         │
├─────────────────────────────────────────────────────────────────┤
│ QUERY END CONDITIONS                                             │
│  - Correct pick → success = True                                 │
│  - HP = 0 → KO                                                  │
│  - rounds_used = T_max → timeout                                │
│  - SKIP → skipped = True                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Information Flow Diagram

```
                    ┌──────────────────────────┐
                    │       GRAMMAR G          │
                    │  (hidden from learner)    │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              │ F_G renders    │                 │
              ▼                ▼                 ▼
     ┌────────────┐   ┌────────────┐    ┌──────────────┐
     │  Learner   │   │   Tutor    │    │  Environment │
     │  CLS Agent │   │   Agent    │    │  (OptionEnv) │
     └──────┬─────┘   └──────┬─────┘    └──────┬───────┘
            │                │                  │
   study()  │                │ oracle scorer    │ generates queries
   predict()│                │ (sees G)         │ assigns risk
            │                │                  │
            ▼                ▼                  ▼
     ┌──────────┐    ┌──────────────┐   ┌──────────────┐
     │ S_sem(j) │    │ disc(ℓ)      │   │ menu + risk  │
     │ μ_d(j)   │    │ P_L(j) est.  │   │ damage model │
     │ U_pick(j)│    │ Q(action)    │   │ HP tracking  │
     └──────┬───┘    └──────┬───────┘   └──────┬───────┘
            │               │                   │
            │  pick/refresh │  BAN/HL/HINT/SKIP │ reveal/damage
            └───────────────┴───────────────────┘
                        Interaction Loop
```

**Key asymmetry**: Tutor sees grammar G (can compute exact renders), but learner only has CLS-learned approximation. This is the "knowledge gap" that tutoring should bridge.

**Anti-oracle constraint**: Despite having oracle access, the tutor NEVER reads `option.is_correct`. It must infer correctness from semantic scores and infer risk from its own danger head.
