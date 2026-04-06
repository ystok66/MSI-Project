# Pedagogical Inverse Planning — Research Prototype

> **When should a teacher intervene to help a learning agent, and how?**

A Gymnasium-based research prototype studying pedagogical intervention under partial observability. A bounded-rational agent navigates a grid world while an external teacher observes and decides whether to **wait**, **warn**, **unlock a door**, or **drop a shield**. Two teacher modes: an **Oracle** teacher (v0) with full belief access, and a **Particle Inference** teacher (v1a) that infers the agent's beliefs from observed actions via SIPS-lite.

**Created**: 2025-03-10  
**Environment**: Python 3.10+, Gymnasium  
**Location**: `f:/SCAI/Learning-agent/pedagogical_ip/`

---

## Table of Contents

1. [Version History](#version-history)  
2. [Architecture](#architecture)  
3. [Environment: PedagogicalGridEnv](#environment-pedagogicalgridenv)  
4. [Agent: BoundedRationalAgent](#agent-boundedrationalagent)  
5. [Teacher: Oracle (v0)](#teacher-oracle-v0)  
6. [Teacher: Particle Inference (v1a)](#teacher-particle-inference-v1a)  
7. [Symbolic RSA Warnings (v1a)](#symbolic-rsa-warnings-v1a)  
8. [Logging & Data Formats](#logging--data-formats)  
9. [Metrics](#metrics)  
10. [Experiments & Results](#experiments--results)  
11. [How to Run](#how-to-run)  
12. [File Structure](#file-structure)

---

## Version History

| Version | Date | Contents |
|---------|------|----------|
| **v0** | 2025-03-10 | Gymnasium env + bounded-rational agent + Oracle teacher + JSONL/NPZ logging + 25 tests |
| **v1a** | 2025-03-10 | SIPS-lite particle teacher + symbolic RSA warnings + 4 evaluation metrics + 4-baseline comparison. Total: 43 tests |
| **v1b** | 2025-03-10 | Benchmark suite: 4 parameterized map families (SemanticTrap, PlanningTrap, ExplorationUseful, Mixed), interaction + transfer protocol, difficulty scaling, diagnostic plots. Total: 69 tests |
| **v1c** | 2025-03-10 | Causal scoring fix: structural UNLOCK simulation, calibration-based IG, intervention margin, false-warning penalty, plan-change gate. Fixes over-intervention on C/D families |
| **v1d** | 2025-03-10 | Cause-aware teacher: latent failure cause z ∈ {explore, belief, plan, hazard}, two-stage decision (safety gate → modality selection), per-family modality ablation. Particle beats oracle on 3 conditions |

---

## Architecture

```
External controller (teacher)
     │
     ├── select_action() → InterventionType ∈ {WAIT, WARN, UNLOCK_DOOR, DROP_SHIELD}
     │
     ▼
┌─────────────────── PedagogicalGridEnv ───────────────────┐
│  env.step(robot_action)                                   │
│    1. Apply intervention to agent/world                    │
│    2. Agent observes surroundings (noisy cues)             │
│    3. Agent updates Gaussian beliefs (Kalman fusion)       │
│    4. Agent plans with bounded-budget A*                   │
│    5. Agent executes one move                              │
│    6. Environment evaluates outcome (cost/risk/pickup)     │
│  → return (obs, reward, terminated, truncated, info)       │
└───────────────────────────────────────────────────────────┘
```

**Key design**: The learner is **internal** to the environment. The external action space belongs to the teacher/robot. This makes teacher policy development (including future RL teacher) a standard Gymnasium problem.

---

## Environment: PedagogicalGridEnv

### Grid World

- **Size**: 8×8 (configurable)
- **Cell types** (enum `CellType`):

| Type | Value | true_cost | true_risk | Notes |
|------|-------|-----------|-----------|-------|
| NORMAL | 0 | 1.0 | 0.0 | Free movement |
| WALL | 1 | ∞ | 0.0 | Impassable |
| HIGH_COST | 2 | 5.0 | 0.0 | Expensive to traverse |
| RISKY | 3 | 1.0 | 0.3 | Probabilistic damage |
| LOCKED_DOOR | 4 | ∞ → 1.0 | 0.0 | Impassable until teacher unlocks |
| TARGET | 5 | 1.0 | 0.0 | Delivery destination |
| OBJECT_SPAWN | 6 | 1.0 | 0.0 | Where object appears |

### Default Map Layout (v0/v1a)

```
A . . . H H O .       A = agent start (0,0)
. # # . H H . .       O = object spawn (0,6)
. # # . . . . !       T = target (7,7)
. . . D . . ! !       # = wall
. . . . . . ! .       H = high cost (5.0)
H H . . . . . .       ! = risky (risk=0.3)
H H . # # . . .       D = locked door
. . . # # . . T       . = normal (cost=1.0)
```

### Task

Two-phase pickup-delivery:
1. Navigate to object at `(0,6)`, pick it up
2. Deliver to target at `(7,7)`

### Episode Dynamics

| Parameter | Default | Config key |
|-----------|---------|------------|
| `max_steps` | 60 | `env.episode.max_steps` |
| `initial_risk_budget` | 1.0 | `env.episode.initial_risk_budget` |
| `shield_duration` | 5 | `env.episode.shield_duration` |
| `risk_trigger_prob` | 0.3 | `env.terrain.risk_trigger_prob` |
| `risk_trigger_prob_shield` | 0.02 | `env.terrain.risk_trigger_prob_shield` |

**Risk event**: When agent enters a RISKY cell, with probability `true_risk[r,c] × p / 0.3`, a risk event fires. If shielded, `p = 0.02`; otherwise `p = 0.3`. Damage = `true_risk[r,c]`, subtracted from `risk_budget_left`. Death if budget ≤ 0.

**Rewards**:
- Step cost: `-true_cost × 0.01`
- Risk damage: `-damage × 0.5`
- Object pickup: `+1.0`
- Delivery: `+10.0`
- Death: `-5.0`
- Timeout: `-2.0`

### Action Space (Teacher)

```python
Discrete(4)
# 0 = WAIT, 1 = WARN, 2 = UNLOCK_DOOR, 3 = DROP_SHIELD
```

### Observation Space (Teacher sees)

```python
Dict({
    "agent_pos":        Box(shape=(2,), int32),
    "object_pos":       Box(shape=(2,), int32),
    "goal_pos":         Box(shape=(2,), int32),
    "has_object":       Discrete(2),
    "has_shield":       Discrete(2),
    "time_left":        Box(shape=(1,), int32),
    "risk_budget_left": Box(shape=(1,), float32),
    "belief_cost_mean": Box(shape=(8,8), float32),  # oracle mode only
    "belief_risk_mean": Box(shape=(8,8), float32),
    "belief_risk_var":  Box(shape=(8,8), float32),
})
```

---

## Agent: BoundedRationalAgent

### Belief Map

Per-cell Gaussian beliefs: `BeliefMap` stores `cost_mean`, `cost_var`, `risk_mean`, `risk_var` ∈ ℝ^(H×W) and `visited_mask` ∈ {0,1}^(H×W).

**Prior** (at episode start):
```
cost_mean = 1.5,  cost_var = 4.0
risk_mean = 0.1,  risk_var = 0.25
```

### Observation Model

Agent sees current cell near-exactly and neighbors with Gaussian noise:

```
For current cell (r,c):     obs_var = 0.001
For neighbor cells (±1):    obs_var = 1.0
```

Each observation is `N(true_value, obs_var)`, clipped to [0, ∞) for cost and [0, 1] for risk.

### Bayesian Update (Kalman fusion)

For each observed cell, scalar precision-weighted fusion:

```
σ²_post = 1 / (1/σ²_prior + 1/σ²_obs)
μ_post  = σ²_post × (μ_prior/σ²_prior + y_obs/σ²_obs)
```

### Planner: Bounded-Budget A*

Cell planning cost:

```
score(r,c) = E[cost](r,c) + λ_risk × E[risk](r,c) × 10.0 + λ_unc × 2 × Var[cost](r,c)
```

Default: `λ_risk = 3.0`, `λ_unc = 0.5`.

**v0**: Agent replans every step with fixed `budget = 30`.

**v1a**: Explicit partial plan tracking. Agent replans **only** when:
1. Plan is exhausted (all steps executed)
2. Agent deviated from planned trajectory
3. Teacher intervention invalidated the plan

Budget sampled from discrete NegBin approximation:

| budget_class | Candidate budgets | Probabilities |
|---|---|---|
| 4 (short) | {3, 4, 5} | [0.25, 0.50, 0.25] |
| 8 (medium) | {6, 8, 12} | [0.25, 0.50, 0.25] |
| 16 (long) | {14, 16, 20} | [0.25, 0.50, 0.25] |

**Action selection**: Deterministic from plan (`a_t = plan[step_idx]`) with ε-greedy fallback (ε = 0.05).

### Key Parameters

| Parameter | Config key | Default |
|-----------|-----------|---------|
| `prior_cost_mean` | `agent.belief.prior_cost_mean` | 1.5 |
| `prior_cost_var` | `agent.belief.prior_cost_var` | 4.0 |
| `prior_risk_mean` | `agent.belief.prior_risk_mean` | 0.1 |
| `prior_risk_var` | `agent.belief.prior_risk_var` | 0.25 |
| `self_noise_var` | `agent.observation.self_cell_noise_var` | 0.001 |
| `neighbor_noise_var` | `agent.observation.neighbor_noise_var` | 1.0 |
| `search_budget` | `agent.planner.search_budget` | 30 |
| `lambda_risk` | `agent.planner.lambda_risk` | 3.0 |
| `lambda_uncertainty` | `agent.planner.lambda_uncertainty` | 0.5 |

---

## Teacher: Oracle (v0)

**File**: `src/teachers/oracle_teacher.py`

Has full access to agent's `belief_cost_mean`, `belief_risk_mean`, `belief_cost_var`.

### Decision Process

1. Score each candidate intervention:
   - WAIT, WARN (best from vocabulary), UNLOCK_DOOR (if locked), DROP_SHIELD
2. For each candidate, estimate:
   - `P_success`: A* reachability from agent's belief + path time/risk feasibility
   - `learning_gain`: variance reduction from intervention
3. Pedagogical utility:

```
U(a_R) = w_s × P_succ + w_learn × learning_gain - w_cost × C_int - w_take × takeover
```

| Weight | Default |
|--------|---------|
| `w_success` | 1.0 |
| `w_learn` | 0.3 |
| `w_cost` | 0.2 |
| `w_takeover` | 0.1 |

---

## Teacher: Particle Inference (v1a)

**File**: `src/teachers/particle_teacher.py`

Does **NOT** access the learner's belief maps during decision time. Infers learner state from observed actions.

### Particle State

Each particle `z^(i)_t` is one hypothesis about the learner:

```python
@dataclass
class Particle:
    belief: BeliefMap            # (H,W) × 4 arrays: cost/risk mean/var
    current_plan: list[tuple]    # partial plan positions
    plan_step_idx: int           # execution pointer into plan
    budget_class: int            # ∈ {4, 8, 16}
    warn_sensitivity: float      # ∈ {0.25, 0.5, 1.0}
    risk_aversion: float         # ∈ {0.5, 1.0, 2.0}
    weight: float                # normalized importance weight
```

### Initialization

`N = 16` particles (default). All share the same prior belief maps. Discrete traits sampled uniformly from the 3×3×3 = 27 trait combinations (budget_class × warn_sensitivity × risk_aversion), 16 drawn without replacement.

### Per-Step Update

```
For each particle i:
  1. Apply last robot action to particle's belief
  2. Predict learner's action: â^(i) = Planner(particle.belief, particle.budget_class, ...)
  3. Weight update:
       w^(i) *= exp(-λ_a × 𝟙[â^(i) ≠ a_observed])    (λ_a = 2.0)
  4. Advance particle's plan step index

Normalize: w^(i) /= Σ w^(j)

Propagate: update each particle's belief with observation model at agent's new position

Resample when ESS < N/2:
  ESS = 1 / Σ(w^(i))²
  Multinomial resampling → uniform weights
```

### Action Selection (Counterfactual Rollout)

For each candidate robot action `a_R`:

1. Compute weighted-average belief from particles (teacher's estimate)
2. Simulate effect on estimated belief
3. Score via:

```
U(a_R) = w_s × P_succ + w_ig × IG - w_c × C_int - w_f × F

Where:
  P_succ = A* reachability heuristic (path found within time/risk budget)
  IG     = log|Σ_risk_before| - log|Σ_risk_after|    (log-det reduction)
  C_int  = intervention cost from table
  F      = E[Δcost] / (ε + IG)                        (frustration)
```

| Weight | Default |
|--------|---------|
| `w_success` | 4.0 |
| `w_ig` | 1.0 |
| `w_cost` | 0.5 |
| `w_frustration` | 0.5 |

| Intervention | Cost |
|---|---|
| WAIT | 0.0 |
| WARN | 0.1 |
| UNLOCK_DOOR | 0.3 |
| DROP_SHIELD | 0.5 |

---

## Symbolic RSA Warnings (v1a)

**File**: `src/teachers/rsa_warning.py`

Fixed utterance inventory with diagonal-Gaussian RSA scoring.

### Utterance Vocabulary (6 utterances)

| Utterance | Semantics | Region | Signal |
|---|---|---|---|
| `LEFT_RISKY` | Left half has high risk | cols [0, W/2) | risky |
| `RIGHT_RISKY` | Right half has high risk | cols [W/2, W) | risky |
| `UPPER_RISKY` | Upper half has high risk | rows [0, H/2) | risky |
| `LOWER_RISKY` | Lower half has high risk | rows [H/2, H) | risky |
| `DOOR_PATH_SAFE` | Center corridor is safe | 3×3 around center | safe |
| `CURRENT_PATH_RISKY` | Agent's current path prefix is dangerous | 8-step plan + 1-ring | risky |

### RSA Scoring

Each region `r` has a Gaussian risk concept `B_r = N(μ_r, σ²_r)` computed from `mean(true_risk[mask])` and `var(true_risk[mask])`.

Learner's belief over the region: `A_t = N(μ_a, σ²_a)` where `μ_a = mean(belief_risk_mean[mask])`.

**Information Need** (KL divergence — how much the learner's belief differs from truth):

```
KL(B || A) = 0.5 × [log(σ²_a/σ²_b) + σ²_b/σ²_a + (μ_b - μ_a)²/σ²_a - 1]
```

**S1 Utility** (speaker selects the most informative utterance):

```
utility(u) = KL_need(u) / τ - β × log(σ²_b)
score(u)   = α × utility(u)
```

| RSA Parameter | Default |
|---|---|
| α (rationality) | 5.0 |
| β (volume penalty) | 0.1 |
| τ (temperature) | 1.0 |

### Learner Warning Update (Precision Fusion)

On receiving a warning `u` for region mask `M`:

```
y_pseudo = 1.0 (risky utterance) or 0.0 (safe utterance)
σ²_eff   = σ²_pseudo / ω_warn        (ω_warn = particle-specific sensitivity)

For each cell c ∈ M:
  prec_old = 1 / σ²_risk(c)
  prec_obs = ω_warn / σ²_eff
  prec_new = prec_old + prec_obs
  σ²_new(c) = 1 / prec_new
  μ_new(c)  = σ²_new(c) × (μ_old(c) × prec_old + y_pseudo × prec_obs)
```

---

## Logging & Data Formats

### JSONL Records (per step)

File: `output/logs/episode_XXXX.jsonl`, one JSON object per line.

```json
{
  "episode_id": 0,
  "step": 1,
  "robot_action": {"type": "WARN", "param": "RIGHT_RISKY"},
  "agent_action": "DOWN",
  "agent_pos_before": [0, 0],
  "agent_pos_after": [1, 0],
  "true_cost": 1.0,
  "true_risk": 0.0,
  "time_left": 59,
  "risk_budget_left": 1.0,
  "has_object": false,
  "has_shield": false,
  "epistemic_gain": 0.032451,
  "frustration_score": 0.0167,
  "reward": -0.01,
  "terminated": false,
  "truncated": false,
  "teacher_scores": {"WAIT": 0.5123, "WARN": 0.7841, ...}
}
```

### NPZ Snapshots (per step)

File: `output/logs/episode_XXXX_npz/step_YYYY.npz`

```python
data = np.load("step_0001.npz")
data["belief_cost_mean"]  # (8, 8) float64
data["belief_cost_var"]   # (8, 8) float64
data["belief_risk_mean"]  # (8, 8) float64
data["belief_risk_var"]   # (8, 8) float64
data["true_cost_map"]     # (8, 8) float64  (optional)
data["true_risk_map"]     # (8, 8) float64  (optional)
```

### v1a Comparison Results

File: `output/v1a_comparison/{baseline}_results.json`

```json
[
  {
    "outcome": "SUCCESS",
    "steps": 14,
    "reward": 10.86,
    "interventions": {"WAIT": 10, "WARN": 4},
    "ece": 102.514,
    "tom_mse": 0.0148
  }
]
```

---

## Metrics

### v0 Online Metrics

| Metric | Formula | File |
|---|---|---|
| **Epistemic Gain** | `max(0, Σ_var_before - Σ_var_after)` | `metrics/online_metrics.py` |
| **Frustration Score** | `0.4 × time_pressure + 0.3 × risk_pressure + 0.3 × replan_pressure` ∈ [0,1] | `metrics/online_metrics.py` |

### v1a Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Zero-Shot Transfer** | `(1/N) Σ 𝟙[success_i]` | Freeze learner, remove teacher, test new map |
| **Epistemic-Cost Efficiency (ECE)** | `(log\|Σ₀\| - log\|Σ_T\|) / (Σcost + λ × Σint_cost)` | Learning per unit cost; higher = better |
| **Counterfactual Frustration Avoidance (CFA)** | `(1/\|K\|) Σ_k [F(WAIT) - F(actual)]` | Frustration reduction vs doing nothing; positive = intervention helped |
| **ToM Estimation MSE** | `(1/\|S\|) Σ_c (μ̂_risk(c) - μ_risk_true(c))²` | Particle teacher's inference accuracy; lower = better |

---

## Experiments & Results

### v0 Oracle Demo (20 episodes, default map, seed=42)

```
20/20 SUCCESS, avg steps: 14.2, avg reward: 10.78
20 JSONL logs + 20 NPZ snapshot dirs + 1 belief heatmap PNG
```

### v1a 4-Baseline Comparison (20 episodes each, default map)

| Baseline | Success | Avg Steps | Avg Reward | Avg ECE | ToM-MSE |
|---|---|---|---|---|---|
| no_teacher | 20/20 (100%) | 14.2 | 10.70 | 352.8 | — |
| always_help | 20/20 (100%) | 14.9 | 10.68 | 104.4 | — |
| oracle | 20/20 (100%) | 14.6 | 10.68 | 76.5 | — |
| **particle** | **20/20 (100%)** | **14.6** | **10.67** | **102.5** | **0.0148** |

### v1a Analysis

- **Default map is trivially easy** — all baselines succeed 100%, so no differentiation on success rate.
- **ToM-MSE = 0.0148**: The particle filter accurately infers the learner's risk beliefs from action observations alone. This validates the core SIPS-lite inference mechanism.
- **ECE ordering**: `no_teacher (352.8) > always_help (104.4) ≈ particle (102.5) > oracle (76.5)`. No-teacher has highest ECE because the agent explores freely without intervention; oracle has lowest because interventions reduce exploration.

### v1b Benchmark Suite (2400 episodes: 4 baselines × 4 families × 3 difficulties × 5 seeds × 10 eps)

#### Map Families

| Family | Size | Structural Property | Expected Best Intervention |
|---|---|---|---|
| **A. SemanticTrap** | 10×10 | Two routes: short-risky vs long-safe; learner underestimates right-side risk | WARN |
| **B. PlanningTrap** | 10×10 | Safe shortcut behind locked door; bounded planner can't discover it | UNLOCK |
| **C. ExplorationUseful** | 10×10 | Low-risk grid with high-uncertainty region; safe to explore | WAIT |
| **D. Mixed** | 10×10 | Three phases: safe-explore → risky-corridor → door-bottleneck | Phase-dependent |

Each family has 3 difficulty levels (easy/medium/hard) controlling: risk cell density, time budget, risk budget, search budget class.

#### Results: Constrained Success Rate (CSR%)

**Family A — Semantic Trap** (particle inference + RSA warning)

| Diff | no_teacher | always_help | oracle | particle | PMA |
|------|-----------|-------------|--------|----------|-----|
| easy | 16% | 20% | 22% | 20% | 1.00 |
| medium | 2% | 6% | 6% | **14%** | 1.00 |
| hard | 2% | 6% | 6% | 6% | 1.00 |

**Family B — Planning Trap** (door unlock advantage)

| Diff | no_teacher | always_help | oracle | particle | PMA |
|------|-----------|-------------|--------|----------|-----|
| easy | 4% | 8% | 0% | 2% | 0.01 |
| medium | 4% | 8% | **16%** | 8% | 0.00 |
| hard | 2% | 8% | 6% | 4% | 0.00 |

**Family C — Exploration Useful** (WAIT preserves learning)

| Diff | no_teacher | always_help | oracle | particle | PMA |
|------|-----------|-------------|--------|----------|-----|
| easy | **100%** | 20% | **100%** | 30% | 0.75 |
| medium | **100%** | 18% | **94%** | 6% | 0.84 |
| hard | **92%** | 6% | **88%** | 6% | 0.88 |

**Family D — Mixed** (multi-phase)

| Diff | no_teacher | always_help | oracle | particle | PMA |
|------|-----------|-------------|--------|----------|-----|
| easy | **100%** | 56% | **98%** | 64% | 0.25 |
| medium | **100%** | 58% | **100%** | 54% | 0.25 |
| hard | **98%** | 36% | **100%** | 38% | 0.26 |

#### v1b Analysis

**What succeeded:**
- Map families create **real differentiation** — difficulty scaling works (100% → 2% on SemanticTrap)
- Oracle makes **correct decisions** — matches no_teacher on exploration families (correctly WAITs)
- always_help is **demonstrably harmful** on C/D families (20% vs 100%)
- PMA = 1.00 on SemanticTrap — particle inference accurately tracks oracle
- Particle **beats all baselines** on SemanticTrap medium (14% vs 6%)

**What failed — and why:**
- **Particle ≈ always_help** on families C/D. Root cause: teacher utility rewards "belief sharpened" not "pedagogically useful". WARN reduces estimated variance → always looks positive → over-intervention.
- **PMA ≈ 0 on PlanningTrap**. Root cause: counterfactual rollout doesn't simulate door-opening as structural map change → UNLOCK is never valued.
- The scoring function optimizes "make my belief estimate look better" not "actually help the learner succeed in the real environment".

**Diagnosis: causal confusion in teacher objective** → fixed in v1c

### v1c Causal Scoring Fix — Results

**Changes**: Structural UNLOCK (real door→passable), calibration-based IG on critical cells, intervention margin (θ=0.3), false-warning penalty (λ_fp=0.5), plan-change gate (λ_plan=0.3), risk-aware path cost L(τ) = Σc + λ_r·Σ[-log(1-ρ)].

#### v1a → v1c Comparison (particle teacher CSR%)

| Family | Diff | v1a | v1c | Δ | Oracle |
|---|---|---|---|---|---|
| **Exploration** | easy | 30% | **100%** | **+70pp** | 100% |
| **Exploration** | medium | 6% | **96%** | **+90pp** | 94% |
| **Exploration** | hard | 6% | **78%** | **+72pp** | 92% |
| **Mixed** | easy | 64% | **96%** | **+32pp** | 100% |
| **Mixed** | medium | 54% | **100%** | **+46pp** | 96% |
| **Mixed** | hard | 38% | **98%** | **+60pp** | 94% |
| SemanticTrap | medium | **14%** | 6% | -8pp | 2% |
| PlanningTrap | medium | 8% | 8% | 0pp | 16% |

#### v1c Analysis

**Fixed**: Over-intervention eliminated — particle no longer ≈ always_help. On C/D families, particle now matches oracle.

**New issue**: Global threshold θ=0.3 causes **under-intervention** on families A/B where WARN/UNLOCK genuinely help. Calibration gain from WARN is too small (~0.04) to exceed margin. A single scalar utility + single global threshold cannot distinguish epistemic teaching from structural empowerment.

**Diagnosis**: Need cause-aware teacher (v1d) — first infer WHY the learner fails, then select intervention modality.

### v1d Cause-Aware Teacher — Results

**Approach**: Teacher infers a latent failure cause z_t ∈ {safe_exploration, belief_error, planning_bottleneck, immediate_hazard} via softmax posterior q(z | h_t), then selects intervention modality by dominant cause. No global θ_intervene; modality-specific margins θ_warn = θ_unlock = 0.05.

**Cause scores**:
- S_explore = safe(WAIT) × IG_wait (uncertainty on critical cells)
- S_belief = ΔP_succ(WARN) + λ_cal·IG_cal + λ_plan·Δplan - C_false
- S_plan = ΔP_succ(UNLOCK) + λ_reach·ΔReach + λ_dist·Δcost_to_go
- S_hazard = λ_r·P_fatal(WAIT) + λ_t·P_timeout(WAIT)

**Per-family modality restrictions**: SemanticTrap = WAIT/WARN, PlanningTrap = WAIT/UNLOCK, ExplorationUseful = WAIT/WARN, Mixed = all.

#### v1a → v1c → v1d Comparison (particle CSR%)

| Family | Diff | v1a | v1c | **v1d** | Oracle |
|---|---|---|---|---|---|
| **SemanticTrap** | easy | 20% | 12% | **36%** | 42% |
| **SemanticTrap** | medium | 14% | 6% | **16%** | 8% |
| **SemanticTrap** | hard | 6% | 2% | **8%** | 12% |
| PlanningTrap | easy | 2% | 4% | 4% | 0% |
| PlanningTrap | medium | 8% | 8% | 8% | 16% |
| PlanningTrap | hard | 4% | 6% | 6% | 6% |
| **Exploration** | easy | 30% | 100% | **100%** | 100% |
| **Exploration** | medium | 6% | 96% | **98%** | 96% |
| **Exploration** | hard | 6% | 78% | **82%** | 92% |
| **Mixed** | easy | 64% | 96% | **98%** | 100% |
| **Mixed** | medium | 54% | 100% | **100%** | 98% |
| **Mixed** | hard | 38% | 98% | **~98%** | 98% |

#### v1d Analysis

**Particle beats oracle on 3 conditions**: SemanticTrap medium (16% vs 8%), ExplorationUseful medium (98% vs 96%), Mixed medium (100% vs 98%). This validates the cause-aware approach: the particle teacher's cause inference selects WARN precisely when the learner has a genuine belief error, without over-intervening when the learner should explore.

**Cause-awareness resolves the v1c tradeoff**: v1c used a global margin that traded off A (under-intervention) against C/D (over-intervention). v1d eliminates this tradeoff by routing decisions through the dominant latent cause, not a single scalar utility.

**Remaining issue**: PlanningTrap still does not improve (UNLOCK not yet selected effectively). The S_plan score relies on bounded A* finding a path through the unlocked door, but the learner's budget-constrained planner may not discover it even after unlock.

---

## How to Run

```bash
conda activate base310
cd f:/SCAI/Learning-agent/pedagogical_ip

# Run all 69 tests (v0 + v1a + v1b)
python -m pytest tests/ -v

# v0: Oracle teacher demo (20 episodes)
python scripts/run_oracle_teacher.py

# v1a: 4-baseline comparison (default map)
python scripts/run_v1a_comparison.py

# v1b: Full benchmark suite (2400 episodes)
python scripts/run_benchmark_suite.py

# v1b: Transfer evaluation (robot-free, unseen maps)
python scripts/run_transfer_suite.py

# v1b: Plots + acceptance check
python scripts/plot_benchmark_results.py
```

---

## File Structure

```
pedagogical_ip/
├── pyproject.toml
├── README.md
├── configs/
│   ├── env.yaml              # Grid, terrain, episode params
│   ├── agent.yaml            # Belief priors, observation, planner
│   ├── teacher.yaml          # Teacher mode, weights, RSA, particles
│   ├── experiment.yaml       # Episodes, seed, output dirs
│   └── benchmark.yaml        # v1b: family params, protocol settings
├── src/
│   ├── envs/
│   │   ├── pedagogical_grid.py    # Gymnasium env (PedagogicalGridEnv)
│   │   ├── map_generator.py       # CellType, GridMap, default/random maps
│   │   ├── map_families.py        # v1b: 4 parameterized map families
│   │   └── benchmark_generator.py # v1b: unified family + seed + difficulty API
│   ├── agents/
│   │   ├── bounded_agent.py       # BoundedRationalAgent (v1a: partial plan)
│   │   ├── belief.py              # BeliefMap, Bayesian update, RSA fusion
│   │   ├── planner_astar.py       # Bounded A*, NegBin budget sampling
│   │   └── observation_model.py   # Noisy observation generation
│   ├── teachers/
│   │   ├── oracle_teacher.py      # v0 Oracle teacher
│   │   ├── particle_teacher.py    # v1d Cause-aware particle teacher
│   │   ├── cause_scoring.py       # v1d: 4 latent cause scores + helpers
│   │   ├── rsa_warning.py         # v1a symbolic RSA warning module
│   │   ├── utilities.py           # Pedagogical utility functions
│   │   └── interventions.py       # InterventionType, Intervention, vocab
│   ├── logging/
│   │   ├── episode_logger.py      # JSONL + NPZ per-step logging
│   │   └── visualize.py           # Belief heatmap visualization
│   └── metrics/
│       ├── online_metrics.py      # Epistemic gain, frustration score
│       └── eval_v1.py             # ZS transfer, ECE, CFA, ToM-MSE
├── scripts/
│   ├── run_oracle_teacher.py      # v0 demo
│   ├── run_v1a_comparison.py      # v1a: 4-baseline comparison
│   ├── run_benchmark_suite.py     # v1b: interaction phase (2400 eps)
│   ├── run_transfer_suite.py      # v1b: transfer phase (robot-free)
│   └── plot_benchmark_results.py  # v1b: plots + acceptance check
├── tests/
│   ├── test_env.py                # 7 tests
│   ├── test_belief_update.py      # 8 tests
│   ├── test_planner.py            # 6 tests
│   ├── test_teacher.py            # 4 tests
│   ├── test_particle_teacher.py   # 10 tests
│   ├── test_rsa_warning.py        # 8 tests
│   └── test_benchmark.py          # 26 tests
└── output/
    ├── logs/                      # JSONL + NPZ per episode
    ├── viz/                       # Belief heatmap PNGs
    ├── v1a_comparison/            # Per-baseline JSON results
    └── v1b_benchmark/             # Per-family CSVs + aggregate + plots
```
