# Lattice V2: Pedagogical Risk Grid — Technical Report

> **Date:** 2026-03-11 | **Project:** pedagogical_ip | **Status:** L2C-minimal complete

---

## 1. Problem Statement

We study a **pedagogical tutoring** problem: a bounded-rational agent navigates a risky grid environment, and a tutor intervenes to reduce harm while preserving the agent's learning opportunities. The central question is:

> How should a tutor balance **direct intervention** (closing dangerous doors) versus **communication** (issuing feature-level warnings) to maximize both the agent's immediate survival and its long-term ability to independently identify risk?

### Design Constraints

1. **No risk oracle:** The agent does NOT observe risk scalars directly. It observes noisy *feature vectors* and must learn a feature→risk mapping.
2. **Doors start open:** The tutor can only *close* doors (irreversible within an episode), never open them.
3. **Bounded rationality:** The agent plans with bounded A* (max 30 node expansions) and updates beliefs online from noisy observations.

---

## 2. Environment: Lattice V2

### 2.1 Grid Structure

A 7-row × W-column grid with K=3 sequential **segments**, each containing two parallel lanes:

```
Row 0: ████████████████████████████   (wall)
Row 1: ██.!!!!.█.!!!!.█.!!!!.██████   (risky lane — straight)
Row 2: █S.████...████...████.G█████   (corridor — walled inside segments)
Row 3: ██..█....█..█...█..█...████   (safe lane — has gap)
Row 4: ███....████....███....██████   (detour connector)
Row 5: ███....████....███....██████   (detour horizontal)
Row 6: ████████████████████████████   (wall)
```

- **Start** S = (2, 1), **Goal** G = (2, W−2)
- **Corridor** (row 2): passable between segments, walled *inside* segments → forces lane choice
- **Risky lane** (row 1): straight path, L_r cells
- **Safe lane** (rows 3→4→5→4→3): zigzag with detour through row 5, L_s = L_r + 2 × d_detour

### 2.2 Segment Parameters

Per segment k:
- Width: w_k ∈ {5, 6, 7} (random)
- Detour length: d_k ∈ {1, 2} (random)
- Path length delta: Δ_k = L_s − L_r = 2 × (d_k + 2) steps

Across 50 seeds: **Δ ∈ [6, 12], mean ≈ 8.8**

### 2.3 Feature Vectors

Each cell carries a 4-dimensional feature vector x ∈ [0,1]⁴:

| Index | Name | Semantics |
|-------|------|-----------|
| 0 | `lane_id` | 0.0 = risky lane, 1.0 = safe lane |
| 1 | `gate_flag` | 1.0 at entry/exit gates, 0.0 elsewhere |
| 2 | `texture_cue_1` | Primary danger texture (high = trap-like) |
| 3 | `texture_cue_2` | Secondary danger texture |

Feature generation by cell type:

| Cell Type | x₂ (texture_1) | x₃ (texture_2) | ρ (true risk) |
|-----------|-----------------|-----------------|---------------|
| Safe lane | U(0, 0.1) | U(0, 0.1) | 0 |
| Risky (no cue) | U(0.1, 0.2) | U(0.05, 0.15) | U(0.08, 0.15) |
| Weak cue | U(0.3, 0.5) | U(0.2, 0.4) | U(0.15, 0.25) |
| **Trap** | **U(0.8, 0.95)** | **U(0.7, 0.9)** | **U(0.3, 0.5)** |

### 2.4 Trap Placement

Per segment, trap existence is stochastic:

| Difficulty | P(trap in segment) | T_max |
|------------|-------------------|-------|
| Easy | 0.50 | 1.5 × L_safe |
| Medium | 0.70 | 1.4 × L_safe |
| Hard | 0.90 | 1.3 × L_safe |

If a trap exists, exactly one cell in the risky lane is designated as the trap cell. Adjacent cells become weak-cue cells.

### 2.5 Death Model

Death is **probabilistic**: stepping on a cell with risk ρ causes death with probability ρ.

---

## 3. Agent Architecture

### 3.1 Feature Belief Map

The agent maintains a Gaussian belief (μ, σ²) for each cell's feature vector. Prior: μ₀ = 0.5, σ₀² = 0.25 (uninformative).

**Observation model** (per step):
- Self cell: σ²_obs = 0.01 (near-exact)
- 1-hop neighbors: σ²_obs = 0.08 (informative but noisy)

**Kalman update** for cell (r,c) observing feature z with noise σ²_obs:

$$K = \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{prior}} + \sigma^2_{\text{obs}}}$$

$$\mu_{\text{post}} = \mu_{\text{prior}} + K \cdot (z - \mu_{\text{prior}})$$

$$\sigma^2_{\text{post}} = \sigma^2_{\text{prior}} \cdot (1 - K)$$

### 3.2 Bayesian Risk Head

A shared linear model predicts risk from features:

$$\hat{\rho}(x) = \sigma(\mathbf{w}^\top x + b)$$

where σ is the sigmoid function and w ∈ ℝ⁴, b ∈ ℝ are learned online.

**Online MAP update** from (feature x, label y, weight w_e):

$$\nabla_\mathbf{w} = -(y - \hat\rho) \cdot x \cdot w_e + \frac{\mathbf{w}}{\sigma^2_{\text{prior}}}$$

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_\mathbf{w}$$

Learning rate η = 0.3. Prior variance σ²_prior = 1.0.

**Differentiated outcome weights:**

| Event | Label y | Weight w_e |
|-------|---------|-----------|
| Death on trap | 1.0 | 4.0 |
| Survived risky cell | ρ_true | 1.5 |
| Safe cell | 0.0 | 0.1 |

**Predictive uncertainty** (Laplace approximation):

$$\hat{u}(x) = \hat\rho(1-\hat\rho)(1 + x^\top H^{-1} x)$$

where H = (1/n) X⊤X + I/σ²_prior is the empirical Hessian.

### 3.3 Planner: Bounded A*

The planner uses feature-based risk prediction for cell costs:

$$c_{\text{plan}}(s) = c(s) + \lambda_r \cdot \phi(\hat\rho(x_s)) + \lambda_u \cdot \hat{u}(x_s)$$

where:
- c(s) = base movement cost (1.0 for normal cells, ∞ for walls)
- φ(ρ) = −log(1 − ρ) is the survival-form risk penalty
- λ_r = 5.0 (risk weight)
- λ_u = 0.1 (uncertainty weight)

**Key property:** The planner uses the agent's *believed* feature means from the FeatureBeliefMap, never the true features. At prior (μ = 0.5), the risk prediction is ≈ 0.5 for all cells, so the planner initially has no preference between lanes.

**Budget:** 30 node expansions. If goal not reached within budget, returns partial path toward best frontier node.

---

## 4. Tutor: Time-Aware Door Tutor

### 4.1 Action Space

The tutor's available actions:
- **CLOSE_RISKY_GATE:** Permanently close the risky-lane entry gate of a segment
- **WARN:** Issue a feature-level utterance warning (L2C extension)
- **WAIT:** No action

### 4.2 Trigger Condition

The tutor acts when the agent is in the corridor (row 2) within 1 column of a segment entry.

### 4.3 Slack Computation

$$\text{slack} = \frac{T_{\text{left}} - L_{\text{safe\_remaining}}}{T_{\text{left}}}$$

where L_safe_remaining is the BFS distance from the agent's position to the goal through safe lanes only.

### 4.4 Decision Logic

| Slack Range | Mode | Action |
|-------------|------|--------|
| slack < 0.3 | Tight | Always close risky gate |
| 0.3 ≤ slack < 0.7 | Medium | Close if segment has trap |
| slack ≥ 0.7 | Loose | Close if segment has trap |

### 4.5 Closure Budget

In experiments, the tutor is limited to at most B gate closures per episode (budget=0,1,2,3).

---

## 5. Warning System (L2C-minimal)

### 5.1 Utterance Vocabulary

| Utterance | Prototype x_u | Pseudo-label y_u |
|-----------|-------------|-----------------|
| RISKY_TEXTURE_AHEAD | [0.5, 0.0, 0.85, 0.80] | 0.8 |
| UPPER_LANE_RISKY | [0.0, 0.0, 0.70, 0.60] | 0.7 |
| SAFE_DETOUR_OPEN | [1.0, 0.0, 0.05, 0.05] | 0.0 |

### 5.2 Warning Application

For each upcoming cell j in the warned segment, compute feature-concept matching:

$$\alpha_j(u) = \exp\!\left(-\frac{\|\hat{x}_j - x_u\|^2}{\tau}\right)$$

Then inject pseudo-label evidence into the shared risk model:

$$\text{risk\_head.update}(\hat{x}_j,\; y_u,\; w_{\text{warn}} \cdot \alpha_j)$$

Parameters: w_warn = 5.0, τ = 0.3.

**Critical design:** Warnings do NOT directly set risk scalars. They go through the shared feature→risk model, meaning their effect depends on how well the warned cell's features match the utterance prototype.

### 5.3 Utterance Selection

Utility-based selection (MVP, not full RSA):

$$u^* = \arg\max_u \sum_j \alpha_j(u) \cdot |y_u - \hat\rho(\hat{x}_j)|$$

This selects the utterance that would maximally change the risk model's predictions for the upcoming cells.

---

## 6. Experimental Results

All experiments at working point: **medium difficulty, time_ratio=1.3, N=100 seeds.**

### 6.1 Main 6-Condition Matrix

| Condition | Survival | Goal | Closures | Warnings | Risky Entered |
|-----------|----------|------|----------|----------|---------------|
| no_tutor | **9%** | 9% | 0 | 0 | 5.1 |
| door_budget_2 | **48%** | 48% | 1.9 | 0 | 3.2 |
| warning_only | **9%** | 9% | 0 | 1.6 | 5.1 |
| door_2 + fixed_warn | **48%** | 48% | 1.9 | 1.9 | 3.1 |
| door_2 + select_warn | **48%** | 48% | 1.9 | 1.9 | 3.1 |
| door_budget_3 | **87%** | 87% | 2.5 | 0 | 1.6 |
| always_close_3 | **100%** | 100% | 3.0 | 0 | 0.0 |

### 6.2 Closure Budget Sweep

| Budget | Survival | Marginal Gain |
|--------|----------|---------------|
| 0 | 9% | — |
| 1 | 18% | +9pp |
| 2 | 48% | +30pp |
| 3 | 87% | +39pp |
| all (always) | 100% | +13pp |

### 6.3 Persistent vs Reset Learning (10 episodes × 30 seeds)

| Condition | Persistent | Reset | Δ |
|-----------|------------|-------|---|
| no_tutor | 7% | 7% | 0pp |
| door_2 | **46%** | 43% | **+3pp** |
| door_2 + fixed_warn | **52%** | 47% | **+5pp** |

### 6.4 Time Ratio Sweep (tutor with unlimited budget)

| Ratio | no_tutor | time_aware |
|-------|----------|------------|
| 1.05 | 9% | 100% |
| 1.10 | 9% | 100% |
| 1.20 | 9% | 94% |
| 1.30 | 9% | 87% |
| 1.40 | 9% | 87% |
| 1.60 | 9% | 78% |

---

## 7. L2C.1: Planner-Relevant Warnings (Breakthrough)

The initial L2C implementation (§6) showed warning had zero standalone effect. Four fixes were applied:

1. **Action-gap utterance selection**: optimize for lane-choice shift, not prediction change
2. **Lane-level warning bias**: temporary cost penalty on ALL risky cells in warned segment
3. **Warning-first loose mode**: tutor tries communication before intervention
4. **Smaller detour delta**: dt fixed at 1, Δ=6 constant

### 7.1 Main Results (dt=1, lambda_lane_warn=5.0)

| Condition | Survival | Closures | Warnings | Risky Entered |
|-----------|----------|----------|----------|---------------|
| no_tutor | **9%** | 0 | 0 | 5.8 |
| **warning_only** | **80%** | **0** | **2.8** | **1.7** |
| door_2 | 68% | 1.9 | 0.5 | 2.4 |
| door_3 | 99% | 2.3 | 0.7 | 0.3 |
| always_close_3 | 100% | 3.0 | 0 | 0 |

**`warning_only (80%) > door_2 (68%)`** — Communication alone outperforms 2-door intervention.

### 7.2 Lambda Lane-Warn Sweep

| λ_lw | Survival | Interpretation |
|------|----------|----------------|
| 1 | 9% | Below threshold, no planner shift |
| 3 | **46%** | Crossing point begins |
| 5 | **80%** | Strong effect |
| 7 | **100%** | Saturates |
| 10 | 100% | Stays saturated |

The crossing point at λ_lw ≈ 3 corresponds to the lane bias exceeding the safe detour delta (≈6 steps).

### 7.3 Transfer Evaluation

| Train Condition | Learning | Test Survival |
|-----------------|----------|---------------|
| no_tutor | persistent/reset | 16% |
| warn_first | persistent/reset | 16% |

Transfer still flat. The linear risk_head overfits to training-seed feature patterns and does not generalize.

### 7.4 Why L2C Failed and L2C.1 Succeeded

**L2C failure**: Warning updated risk_head predictions per-cell, but the total risk penalty on the risky path (φ(0.63) × λ_r = 4.96) never exceeded the safe lane detour cost (Δ=6). Additionally, the A* heuristic naturally favors row 1 (risky) over row 3 (safe), so even a high gate-only bias was insufficient.

**L2C.1 fix**: Apply the full lane bias (λ_lw × Σ_j α_j y_u) to **every cell** in the warned risky lane. With 5 cells × 4.0 bias = 20.0 total extra cost on the risky path, the planner decisively shifts to the safe lane.

---

## 8. Code Architecture

```
pedagogical_ip/
├── src/
│   ├── envs/
│   │   └── lattice_v2.py          # 7-row grid, segment generation, features
│   ├── agents/
│   │   ├── feature_belief.py      # FeatureBeliefMap (Kalman d=4)
│   │   ├── risk_model.py          # BayesianRiskHead (sigmoid linear, lr=0.3)
│   │   ├── observation_model.py   # observe_features()
│   │   ├── planner_astar.py       # cell_cost_v2 + warned_cell_extra_cost
│   │   └── warning_update.py      # 3 utterances, lane bias, action-gap select
│   └── teachers/
│       └── time_aware_door_tutor.py  # 3-mode: tight/medium/loose(warn-first)
├── scripts/
│   ├── _diag_l2c1_sweep.py        # L2C.1 full sweep + transfer eval
│   └── _diag_l2b5_sweep.py        # non-saturation sweep
└── results/
    ├── l2c1_sweep3.txt             # L2C.1 breakthrough results
    ├── l2c_sweep2.txt              # L2C baseline results
    └── l2b5_sweep3.txt             # B.5 results
```

---

## 9. Current Status & Next Steps

### Established
- ✅ Non-cheating perception (feature belief + shared risk head)
- ✅ Warning can substitute for intervention (warning_only 80% > door_2 68%)
- ✅ Clean lambda sweep from 9% to 100%

### Open Issues
- ❌ Transfer learning doesn't work (risk_head too simple to generalize)
- ❌ Persistent vs reset still indistinguishable at test time
- ⚠️ Lambda_lane_warn must be tuned per geometry (delta-dependent)

### Recommended Next Steps
1. **Full RSA speaker**: Replace hand-crafted prototypes with L0/S1/L1 pragmatic inference
2. **Non-linear risk head**: MLP or kernel-based model for transfer
3. **Joint door+warning optimization**: Rather than separate mechanisms, unified policy

