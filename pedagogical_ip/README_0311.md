# Pedagogical Intervention in Grid Navigation — Technical README

**Date: 2026-03-11**

---

## 1. Problem Overview

A **bounded-rational learner agent** navigates a grid world from start $S$ to goal $G$. The agent has imperfect beliefs about cell costs and risks, and makes decisions using a budget-limited A* planner. A **teacher (robot)** observes the agent externally and can intervene with pedagogical actions (WARN, UNLOCK, BLOCK) to improve the learner's chances of reaching the goal safely.

The core research question: **Can a teacher, by choosing the right intervention at the right time, significantly improve the learner's success rate beyond what the learner achieves alone?**

---

## 2. Grid Environment

### 2.1 Cell Properties

Each cell $(r, c)$ in the $H \times W$ grid has:

| Property | Symbol | Description |
|---|---|---|
| Cell type | $\tau_{r,c}$ | NORMAL, WALL, RISKY, HIGH_COST, LOCKED_DOOR, OBJECT_SPAWN, TARGET |
| True cost | $c^*_{r,c}$ | Movement cost ($1.0$ for normal, $\infty$ for walls/locked doors) |
| True risk | $\rho^*_{r,c}$ | Risk probability $\in [0, 1]$; agent doesn't know this initially |

### 2.2 Deceptive Fork (MVP Environment)

**Fixed geometry, 6×8 grid:**

```
  0 1 2 3 4 5 6 7
0 W W W W W W W W
1 W . . a T . . W     Path A (bait): trap at T=(1,4), rho*=1.0
2 S . F . . G . W     S=(2,0), Fork=(2,2), Goal=(2,5)
3 W . . . . . . W
4 W . . D b . . W     Path B (safe): door at D=(4,3)
5 W W W W W W W W
```

**Two paths:**

| Property | Path A (bait) | Path B (safe) |
|---|---|---|
| Total length (S→G) | 6 steps | 8 steps |
| Fork→Goal | 4 steps | 6 steps |
| Risk | Trap at (1,4), $\rho^* = 1.0$ | All $\rho^* = 0$ |
| Door | None | Locked at (4,3) (Env-B only) |

**Path A route:** `S(2,0) → (2,1) → (2,2=fork) → (1,2) → (1,3) → (1,4=TRAP) → (1,5) → (2,5=G)`

**Path B route:** `fork(2,2) → (3,2) → (4,2) → (4,3=DOOR) → (4,4) → (3,4) → (3,5) → (2,5=G)`

**Geometric constraints:**
- $d_{\text{Manhattan}}(\text{fork}, \text{trap}) = |1-2| + |4-2| = 3 \geq 3$ (trap invisible at fork decision)
- Observation radius = 1, so agent cannot see trap when at fork
- $T_{\max} \approx 9\text{--}12$ (B-path reachable, backtrack from A impossible)

**Two sub-environments:**
- **Env-A** (`with_door=False`): Tests WARN in isolation
- **Env-B** (`with_door=True`): Tests WARN + UNLOCK complementarity

**Implementation:** [`src/envs/map_families.py`](src/envs/map_families.py) → `generate_deceptive_fork(seed, difficulty, with_door)`

---

## 3. Agent (Bounded-Rational Learner)

### 3.1 Belief Representation

The agent maintains **per-cell Gaussian beliefs** over cost and risk:

$$
b_t(r,c) = \big(\mu^c_{r,c},\;\sigma^{c,2}_{r,c},\;\mu^\rho_{r,c},\;\sigma^{\rho,2}_{r,c}\big)
$$

**Prior (at $t = 0$):**

| Parameter | Symbol | Value |
|---|---|---|
| Cost mean | $\mu_0^c$ | 1.0 |
| Cost variance | $\sigma_0^{c,2}$ | 0.1 |
| Risk mean | $\mu_0^\rho$ | **0.02** (optimistic) |
| Risk variance | $\sigma_0^{\rho,2}$ | 0.20 |

The deliberate optimistic risk prior ($\mu_0^\rho = 0.02$) makes the agent underestimate risk, causing it to prefer the shorter but dangerous Path A. This creates the opportunity for teacher intervention.

**Implementation:** [`src/agents/belief.py`](src/agents/belief.py) → `BeliefMap`

### 3.2 Observation Model

At each step, the agent observes its current cell and 1-hop neighbors:

$$
\tilde{c}_{r,c} = c^*_{r,c} + \epsilon_c, \quad \tilde{\rho}_{r,c} = \rho^*_{r,c} + \epsilon_\rho
$$

| Position | Noise variance $\sigma^2_{\text{obs}}$ |
|---|---|
| Current cell $(r_0, c_0)$ | **0.001** (near-exact) |
| Neighbors ($\|d\| \leq 1$) | **1.0** (very noisy) |

This means the agent quickly converges on the truth for visited cells, but neighboring cells remain uncertain until visited.

**Implementation:** [`src/agents/observation_model.py`](src/agents/observation_model.py) → `generate_observations()`

### 3.3 Belief Update (Kalman Filter)

For each observed cell, the belief is updated via scalar Kalman:

$$
\sigma^{2}_{\text{post}} = \left(\frac{1}{\sigma^{2}_{\text{prior}}} + \frac{1}{\sigma^2_{\text{obs}}}\right)^{-1}
$$

$$
\mu_{\text{post}} = \sigma^{2}_{\text{post}} \left(\frac{\mu_{\text{prior}}}{\sigma^{2}_{\text{prior}}} + \frac{\tilde{y}}{\sigma^2_{\text{obs}}}\right)
$$

Applied independently to cost and risk for each cell.

**Implementation:** [`src/agents/belief.py`](src/agents/belief.py) → `bayesian_update()`, `update_belief_cell()`

### 3.4 Planner (Bounded A*)

The agent plans using **bounded A*** — expanding at most $\eta$ nodes (sampled from a discrete distribution).

**Planning cost for cell $s = (r,c)$:**

$$
c_{\text{plan}}(s) = \mu^{c}_s + \lambda_r \cdot \varphi(\mu^{\rho}_s) + \lambda_u \cdot \sigma^{c,2}_s
$$

where the risk penalty uses a **survival form**:

$$
\varphi(\mu^{\rho}) = -\log\big(1 - \text{clip}(\mu^{\rho},\; \epsilon,\; 1-\epsilon)\big), \quad \epsilon = 10^{-4}
$$

**Properties of the survival-form risk penalty:**

| $\mu^\rho$ | $\varphi(\mu^\rho)$ | $\lambda_r \cdot \varphi$ (at $\lambda_r = 0.8$) |
|---|---|---|
| 0.02 (prior) | 0.020 | 0.016 |
| 0.20 | 0.223 | 0.179 |
| 0.55 | 0.799 | 0.639 |
| 0.85 | 1.897 | 1.517 |
| 0.99 | 4.605 | 3.684 |

At the prior risk ($\mu^\rho = 0.02$), the penalty is negligible (~0.016). This is critical: it means the agent treats unknown cells as "probably safe" and prefers shorter paths.

**Parameters:**

| Parameter | Symbol | Value |
|---|---|---|
| Risk weight | $\lambda_r$ | **0.8** |
| Uncertainty weight | $\lambda_u$ | **0.02** |
| Budget class | — | 8 → support {6, 8, 12} |
| Budget probs | — | [0.25, 0.50, 0.25] |

**Heuristic:** Manhattan distance $h(s, g) = |r_s - r_g| + |c_s - c_g|$

**Search:** Standard A* with $f = g + h$, terminated after $\eta$ expansions. If goal not reached, returns partial path toward best frontier node.

**Implementation:** [`src/agents/planner_astar.py`](src/agents/planner_astar.py) → `bounded_astar()`, `cell_cost()`

### 3.5 Replanning Triggers

The agent does **NOT** replan every step. It replans only when:

1. Current plan exhausted
2. Agent deviated from planned trajectory
3. Teacher intervention occurred (plan invalidated)

Between replans, the agent follows its existing plan. Action selection is $\epsilon$-greedy ($\epsilon = 0.05$): with probability $\epsilon$, take a random valid move.

**Implementation:** [`src/agents/bounded_agent.py`](src/agents/bounded_agent.py) → `_needs_replan()`, `plan_and_act()`

---

## 4. Environment Step Loop

Each `env.step(teacher_action)` executes:

```
1. Apply teacher action (WAIT / WARN / UNLOCK / BLOCK)
2. Tick blocked-cell TTLs
3. Agent observes current cell + neighbors (→ belief update)
4. If needs_replan: sample budget η, run bounded A*
5. Agent executes one move (or STAY if plan says so)
6. Risk check: if rho* > 0 on entered cell → sample damage
7. Check objective completion / timeout
8. Return (obs, reward, terminated, truncated, info)
```

**Risk trigger mechanism:**
When agent enters a cell with $\rho^*_{r,c} > 0$:
- Sample $u \sim \text{Uniform}(0,1)$
- If $u < \rho^*_{r,c} \cdot p_{\text{trigger}} / 0.3$: apply damage $= \rho^*_{r,c}$ to risk budget
- If risk_budget $\leq 0$: episode terminates (fatal)

With the default $p_{\text{trigger}} = 0.3$ and $\rho^* = 1.0$, the trap cell is **deterministic death** (trigger probability = 100%).

**Implementation:** [`src/envs/pedagogical_grid.py`](src/envs/pedagogical_grid.py) → `PedagogicalGridEnv.step()`

---

## 5. Teacher (Robot) Actions

### 5.1 Action Space

| Action | Index | Effect |
|---|---|---|
| **WAIT** | 0 | No intervention |
| **WARN** | 1 | Update agent's risk belief on target cells |
| **UNLOCK** | 2 | Open a locked door (cost → 1.0, inform agent) |
| **BLOCK** | 4 | Temporarily block a cell (TTL-based) |

(DROP_SHIELD at index 3 exists in code but is disabled for fork experiments.)

### 5.2 WARN Mechanism (Deterministic Oracle, Current Implementation)

The teacher applies a **pseudo-observation** to a set of target cells $S_{\text{warn}}$, using the same Kalman update as normal observations but with very low variance (high certainty):

For each cell $j \in S_{\text{warn}}$:

$$
\sigma^{\rho,2}_{j,\text{new}} = \left(\frac{1}{\sigma^{\rho,2}_{j,\text{old}}} + \frac{1}{\sigma^2_{\text{warn}}}\right)^{-1}
$$

$$
\mu^{\rho}_{j,\text{new}} = \sigma^{\rho,2}_{j,\text{new}} \left(\frac{\mu^{\rho}_{j,\text{old}}}{\sigma^{\rho,2}_{j,\text{old}}} + \frac{y_j}{\sigma^2_{\text{warn}}}\right)
$$

where $\sigma^2_{\text{warn}} = 0.005$ (very precise), and $y_j$ is the warning's pseudo-observation value.

**Warning profile (front-loaded):**

The warning targets Path A's post-fork segment with a **graduated risk profile**:

| Cell | Position | $y_j$ (pseudo-obs) |
|---|---|---|
| Fork+1 | (1,2) | 0.55 |
| Fork+2 | (1,3) | 0.75 |
| Trap | (1,4) | 0.90 |
| Post-trap | (1,5) | 0.30 |

This profile is **front-loaded**: high values near the fork ensure the bounded planner sees the risk immediately during search, not just at the distant trap cell.

**Effect:** After warning, the agent's $\mu^\rho$ on Path A cells jumps from prior 0.02 to approximately $y_j$ (due to very low $\sigma^2_{\text{warn}}$), making the survival-form penalty $\varphi$ large enough to redirect the planner to Path B.

### 5.3 UNLOCK Mechanism

When the teacher performs UNLOCK:
1. Door cell cost in the dynamic cost map is set to 1.0 (from $\infty$)
2. Agent's belief cost for the door cell is set to 1.0 with low variance
3. Agent's plan is invalidated → triggers replan

### 5.4 BLOCK Mechanism

When the teacher performs BLOCK on a cell:
1. Cell becomes impassable for TTL steps (default 2-3)
2. Agent's plan is invalidated → triggers replan
3. After TTL expires, cell returns to passable

Used as emergency override when warning is insufficient (e.g., agent ignores linguistic signals).

---

## 6. Gymnasium Interface

The environment follows standard Gymnasium API:

```python
env = PedagogicalGridEnv(
    grid_map=gm,
    max_steps=12,
    initial_risk_budget=1.0,
    prior_cost_mean=1.0,  prior_cost_var=0.1,
    prior_risk_mean=0.02, prior_risk_var=0.20,
    search_budget=30,
    lambda_risk=0.8, lambda_uncertainty=0.02,
    seed=42,
)
obs, info = env.reset()

for t in range(max_steps):
    teacher_action = policy(obs)  # 0=WAIT, 1=WARN, 2=UNLOCK
    obs, reward, terminated, truncated, info = env.step(teacher_action)
    if terminated or truncated:
        break
```

**Architecture:**
- External policy controls **teacher only**
- Learner agent runs **inside** `env.step()` (not externally controllable)
- This allows `copy.deepcopy(env)` for counterfactual rollouts

---

## 7. Experimental Results (2026-03-11)

### 7.1 Phase 1: No-Teacher Baseline (Env-A, no door)

| Metric | Value |
|---|---|
| Branch A (bait) | **98%** |
| Branch B (safe) | 2% |
| Hit trap | 100% |
| CSR | **0%** |
| Avg steps | 5.3 |

**Conclusion:** Agent almost always takes the shorter bait path and dies at the trap. Teacher has maximum intervention opportunity.

### 7.2 Phase 2A: Warning Profile Comparison

**Env-A (no door), 200 seeds, T_max=9:**

| Warning type | Profile | P(first_B\|warn) | CSR |
|---|---|---|---|
| None | — | 2% | 0% |
| Trap-only | [—, —, 0.85, —] | 18% | 4% |
| Original | [0.20, 0.40, 0.85, 0.15] | 18% | 4% |
| **Front-loaded** | **[0.55, 0.75, 0.90, 0.30]** | **80%** | **12%** |
| Front-loaded (eps=0) | [0.55, 0.75, 0.90, 0.30] | **86%** | **15%** |

**Key findings:**
1. **Trap-only warning is insufficient** — bounded planner doesn't propagate distant risk to fork decision
2. **Front-loaded profile dramatically improves fork redirection** — 86% vs 18% first_B rate
3. **Reversion is rare** (<1%) — once agent commits to B, it stays on B
4. **Low CSR despite high first_B** — dominated by **timeout**, not fatal

### 7.3 T_max Sweep (Front-loaded + eps=0)

| T_max | First B | Commit B | **CSR** | **P(s\|commit_B)** | Timeout | Fatal |
|---|---|---|---|---|---|---|
| 9 | 86% | 140 | 15% | 0.21 | 170 | 0 |
| 10 | 89% | 155 | 27% | 0.35 | 146 | 0 |
| 11 | 92% | 165 | **40%** | **0.49** | 119 | 0 |
| 12 | 94% | 173 | **52%** | **0.60** | 96 | 0 |

**No-teacher baseline at T_max=12: CSR = 0%**

**Key findings:**
1. **Fatal = 0 across all T_max** — front-loaded warning completely eliminates trap death
2. **CSR scales linearly with T_max** — purely a budget constraint issue
3. **WARN creates 52% absolute CSR improvement** (0% → 52% at T_max=12)
4. P(success|commit_B) = 0.60 at T_max=12 — B-path execution still somewhat tight

---

## 8. Identified Problems & Next Steps

### Problem 1: B-path execution overhead
Bounded planner on B-path takes more steps than optimal (replan overhead, non-optimal path segments). At T_max=12, P(success|commit_B)=0.60 instead of ~1.0. Need either more generous T_max or smoother B-path execution.

### Problem 2: Warning is deterministic oracle
Current WARN directly injects risk values — not realistic. Next step (Phase 3) is semantic warning with path embeddings and RSA-style utterance selection.

### Problem 3: Teacher decision logic not yet integrated
Current experiments use hand-triggered warning at the fork. Need counterfactual Q-function for teacher action selection that compares WAIT/WARN/UNLOCK/BLOCK.

### Planned Phases

| Phase | Goal | Status |
|---|---|---|
| 1. MVP env + agent fix | Agent takes bait path | **Done** ✅ |
| 2A. Deterministic WARN | Warning redirects at fork | **Done** ✅ |
| 2B. T_max calibration | B-path feasible | **In progress** |
| 3. Semantic WARN | Path embeddings + RSA | Planned |
| 4. UNLOCK + Env-B | Modality complementarity | Planned |
| 5. Particle teacher | Online inference | Planned |

---

## 9. Code Structure

```
pedagogical_ip/
  src/
    agents/
      belief.py              # BeliefMap, Kalman update
      observation_model.py   # Noisy observations
      planner_astar.py       # Bounded A* with survival-form risk
      bounded_agent.py       # Online replanning agent
    envs/
      map_generator.py       # GridMap, CellType
      map_families.py        # generate_deceptive_fork(), etc.
      pedagogical_grid.py    # Gymnasium PedagogicalGridEnv
    teachers/
      interventions.py       # InterventionType enum
      cause_scoring.py       # 4-cause scoring (legacy, to be replaced)
      block_scoring.py       # BLOCK trigger conditions
      oracle_cause_teacher.py
      particle_teacher.py    # 27-particle SIPS-lite
  scripts/
    _diag_fork.py            # Phase 1 diagnostic
    _diag_fork_warn.py       # Phase 2A warning comparison
    _diag_reversion.py       # Reversion diagnosis
    _diag_tmax_sweep.py      # T_max calibration
  tests/
    test_benchmark.py        # 77 tests, all passing
  results/
    fork_diag.txt
    fork_warn_diag.txt
    reversion_diag_v2.txt
    tmax_sweep.txt
```
