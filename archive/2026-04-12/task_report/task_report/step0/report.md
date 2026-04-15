# CLS Option Tutor — Step 0 Status Report

## 1. System Overview

The CLS Option Tutor is a pedagogical framework where a **tutor agent** observes and intervenes to help a **learner agent** solve grammar-based multiple-choice tasks. The system models a realistic tutoring interaction: the tutor cannot directly tell the learner the answer (anti-oracle constraint) but can provide indirect guidance through interventions.

### Task Domain

Each task is based on a **formal grammar** (e.g., `"emit kiki" → RED RED`). The learner must:
1. Study a few support examples (program → output mappings)
2. Given a target output (e.g., `GREEN BLUE GREEN BLUE`), identify which program among K=10 options produces it
3. Survive risk: some options carry hidden damage that reduces HP

### Core Challenge
The learner uses a **CLS (Complementary Learning Systems)** model to learn grammar rules from few examples. The tutor observes the learner's behavior and decides whether/how to intervene.

---

## 2. Architecture

### 2.1 Four-Phase Block Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                         ONE BLOCK                                │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Phase 1     │  Phase 2     │  Phase 3     │  Phase 4           │
│  PRE-TRAIN   │  OBSERVATION │  TEACHING    │  EVALUATION        │
│              │              │              │                    │
│  CLS studies │  Tutor       │  Tutor can   │  CLS FROZEN        │
│  n_sup       │  watches     │  intervene:  │  No tutor          │
│  support     │  frozen      │  BAN, HL,    │  3 queries         │
│  examples    │  learner     │  HINT, SKIP  │  → EVAL_SR metric  │
│              │  N_obs qs    │  N_teach qs  │                    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

**Phase 1 — Pre-training**: CLS agent calls `study(support[:n_sup])` to learn grammar rules via EM.

**Phase 2 — Observation**: Tutor watches the frozen learner answer queries. Tutor is forced to WAIT. Collects `PolicyStateSnapshots` for profile inference.

**Phase 3 — Teaching**: Tutor selects interventions. Learner picks options; wrong picks trigger `reveal` events. CLS calls `incremental_study()` on revealed (program, output) pairs.

**Phase 4 — Evaluation**: CLS is frozen (`freeze()`). Tutor forced to WAIT. 3 queries measure persistent learning gain → **EVAL_SR**.

### 2.2 Per-Query Interaction Loop

```
For each query (target output Y*):
  Menu: K=10 options (programs), 1 correct + 9 distractors
  Risk: 6 safe (dmg=0) + 4 risky (dmg ∈ {1,2,3,4})
  HP: starts at H_0=5, reduced by damage on wrong picks
  T_max: 5 rounds maximum

  Each round:
    1. Tutor acts:  WAIT | BAN(j) | HIGHLIGHT(cells) | RISK_HINT(j) | SKIP
    2. Learner acts: pick(j) | refresh
    3. If wrong pick: reveal (program, rendered_output, damage)
    4. Query ends when: correct pick | HP=0 | T_max reached | SKIP
```

---

## 3. Component Details

### 3.1 Learner Agent

| Component | Description | File |
|-----------|-------------|------|
| **CLS Adapter** | Wraps CLSAgent (3-layer: cortex EM + hippocampus + sleep) for grammar learning | `learner/cls_adapter.py` |
| **Semantic Scorer** | Deterministic scorer (oracle) or CLS posterior. Scores options by mismatch with target | `learner/semantic_scorer.py` |
| **Danger Head** | Two-layer risk model: HazardHead (P(risky\|v)) + SeverityHead (E[dmg\|risky,v]) | `learner/danger_head.py` |
| **Attention Model** | Per-cell attention weights over target output. HIGHLIGHT boosts specific cells (ρ_H=2.0) | `learner/attention_model.py` |
| **Episodic Memory** | Elimination penalty for previously-tried wrong options | `learner/episodic_memory.py` |
| **Policy** | Risk-gated pick/refresh decision (see §3.3) | `learner/policy.py` |

#### CLS Learning Pipeline
```
support[:n_sup] → CLSAgent.study() → EM iterations (n_em=2)
                                    → cortex learns grammar rules
                                    → hippocampus stores exact examples
                                    
For scoring:  program ν_j → CLSAgent.predict(ν_j) → Ŷ_j
              mismatch(Ŷ_j, Y*) → semantic score S_sem(j)
```

#### Incremental Learning (Phase 3)
When learner picks wrong option j → reveal shows (ν_j, render(ν_j)):
```python
new_example = Example(words=ν_j, output=render(ν_j))
cls_scorer.incremental_study([new_example])  # re-runs EM with expanded dataset
```

### 3.2 Tutor Agent

| Component | Description | File |
|-----------|-------------|------|
| **Profile Inference** | RSA-style inverse planning from observation traces | `tutor/profile_inference.py` |
| **Tutor Policy** | Selects best intervention via counterfactual scoring | `tutor/tutor_policy.py` |
| **Counterfactual Scorer** | Scores each candidate action (see §3.4) | `tutor/counterfactual.py` |

**Anti-Oracle Constraint (§12)**: Tutor NEVER accesses `option.is_correct`.  
Uses its own `DeterministicSemanticScorer` (oracle-level) + `DangerHead` to estimate which options are correct/dangerous.

### 3.3 Learner Policy: Risk-Gated Refresh

```
Decision logic:
  1. Compute semantic scores for all K active options (attention-weighted)
  2. Compute pick utilities:
     U_pick(j) = α_sem·S_sem(j) - α_risk·μ_d(j) - α_unc·u_d(j) + memory_penalty(j)
  3. Find best semantic option: j* = argmax(S_sem)
  4. Check risk threshold:
     IF predicted_damage(j*) ≥ current_HP AND rounds_remaining > 1:
       → REFRESH (re-roll risk assignments, costs 1 round)
     ELSE:
       → Sample from softmax(β_L · U_pick) with ε-lapse
```

**Key design choices**:
- Refresh has NO max count limit — each refresh costs 1 round
- Refresh only when best semantic option would likely KO the learner
- HazardHead initializes with safe-biased prior: σ(-1) ≈ 0.27 (not 0.50)
- β_L = 4.0, ε = 0.05

### 3.4 Counterfactual Intervention Scoring

```
Q(WAIT)         = 0                                      (baseline)
Q(BAN, j)       = β_safe · danger_j · P_L(j) - c_ban    (c_ban = 0.0)
Q(RISK_HINT, j) = β_safe · p_h(v_j) · P_L(j) - c_hint  (c_hint = 0.3)
Q(HIGHLIGHT, H) = β_IG · IG(H) - β_over · |H|/L - c_hl  (c_hl = 0.0)
Q(SKIP)         = β_mastery · P_corr + β_certainty · (1 - H_sem) - c_skip  (c_skip = 1.5)
```

Where:
- `P_L(j)` = learner's estimated pick probability of option j
- `IG(H)` = information gain of highlighting cell set H (oracle-based)
- `danger_j` = tutor's predicted danger for option j

### 3.5 Intervention Mechanics

| Action | Effect | Persists through refresh? |
|--------|--------|---------------------------|
| **WAIT** | No action | — |
| **BAN(j)** | Remove option j from active menu | No (cleared on refresh) |
| **HIGHLIGHT(cells)** | Boost attention weights on specified output cells (ρ_H=2.0) | **Yes** (text unchanged) |
| **RISK_HINT(j)** | Weak hazard label for learner's HazardHead (η=0.8) | No (cleared on refresh) |
| **SKIP** | End query immediately, no damage | — |

### 3.6 Option Generation (V2)

Uses `ProgramPool`: enumerates all valid programs up to length 5 and output length 8 from the grammar. Guarantees:
- 100% renderable (no NONE outputs)
- Cell-level diversity (distractors differ from correct answer in specific cells)
- 1 correct + 9 diverse distractors per menu

---

## 4. File Structure

```
cls_option_tutor/
├── config.py                    # All hyperparameters (EnvConfig, LearnerConfig, TutorConfig)
├── interfaces.py                # Data classes (Option, TutorStep, LearnerStep, etc.)
├── __init__.py
│
├── env/                         # Environment
│   ├── state.py                 # BlockState, QueryState, ProfileState
│   ├── option_env.py            # OptionEnv: reset_block, tutor_act, learner_act
│   ├── interventions.py         # apply_wait/ban/highlight/risk_hint/skip
│   ├── danger_model.py          # Risk vector generation
│   └── option_gen_v2.py         # V2 ProgramPool + diverse menu generation
│
├── learner/                     # Learner agent
│   ├── learner_agent.py         # LearnerAgent: autonomous block execution
│   ├── policy.py                # LearnerPolicy: risk-gated pick/refresh
│   ├── semantic_scorer.py       # DeterministicSemanticScorer (oracle)
│   ├── cls_adapter.py           # CLSSemanticPosterior (CLS-backed scorer)
│   ├── semantic_protocol.py     # SemanticPosteriorProtocol interface
│   ├── danger_head.py           # HazardHead + SeverityHead (2-layer risk)
│   ├── attention_model.py       # Attention weights + HIGHLIGHT boost
│   └── episodic_memory.py       # Elimination memory for wrong picks
│
├── tutor/                       # Tutor agent
│   ├── tutor_agent.py           # TutorAgent: observe → infer → teach
│   ├── tutor_policy.py          # TutorPolicy: action selection
│   ├── counterfactual.py        # CounterfactualScorer: Q-value computation
│   └── profile_inference.py     # RSA-style learner profiling
│
├── grammar/                     # Grammar processing
│   ├── task_adapter.py          # Load grammar + support + queries from data
│   └── query_synthesizer.py     # Generate novel queries from grammar
│
├── eval/                        # Evaluation harness
│   ├── benchmark.py             # Block-level metrics
│   ├── experiment_harness.py    # Multi-condition sweep
│   └── pre_post_eval.py         # Pre/post teaching comparison
│
├── exp_4phase.py                # 4-phase experiment script
├── exp_focused.py               # Focused n_sup={0,2,4,6} experiment
├── exp_nsup_sweep.py            # Full n_sup=1..10 sweep
├── exp_obs_depth.py             # Observation depth experiment (N_obs=10)
├── run_diagnostic.py            # Per-step diagnostic logging
├── diagnostic_logger.py         # JSON diagnostic output
│
├── tests/                       # 83 passing tests
│   ├── test_env.py
│   ├── test_learner.py
│   ├── test_tutor.py
│   └── ...
│
├── results/                     # Current experiment results
│   ├── focused_results.txt      # n_sup={0,2,4,6}, 600 jobs
│   └── obs_depth_results.txt    # N_obs=10, 2400 jobs (latest)
│
└── task_report/
    └── step0/                   # This report
```

---

## 5. Current Experiment Results

### 5.1 Latest Experiment: Observation Depth (2400 jobs)

**Design**: 5 conditions × 4 n_sup × 2 N_teach × 20 seeds × 3 grammars = 2400 jobs

```
Phases: Pre-train(n_sup) → Observe(10) → Teach(1 or 2) → Eval(3)
```

#### Eval-Phase SR (N_teach=1)

| n_sup | baseline | ban_only | hl_only | full_tutor | oracle |
|-------|---------|----------|---------|------------|--------|
| 2 | 0.422±0.038 | 0.428±0.041 | 0.433±0.040 | 0.433±0.040 | 0.783±0.036 |
| 4 | 0.489±0.038 | 0.517±0.034 | 0.500±0.038 | 0.500±0.038 | 0.783±0.036 |
| 6 | 0.650±0.038 | 0.617±0.038 | 0.644±0.037 | 0.644±0.037 | 0.783±0.036 |
| 8 | 0.594±0.042 | 0.600±0.042 | 0.606±0.040 | 0.606±0.040 | 0.783±0.036 |

#### Eval-Phase SR (N_teach=2)

| n_sup | baseline | ban_only | hl_only | full_tutor | oracle |
|-------|---------|----------|---------|------------|--------|
| 2 | 0.522±0.033 | 0.500±0.036 | 0.489±0.035 | 0.494±0.035 | 0.839±0.029 |
| 4 | 0.506±0.037 | 0.522±0.035 | 0.500±0.036 | 0.500±0.036 | 0.839±0.029 |
| 6 | 0.650±0.032 | 0.550±0.035 | 0.644±0.031 | 0.644±0.031 | 0.839±0.029 |
| 8 | 0.628±0.038 | 0.628±0.034 | 0.583±0.036 | 0.583±0.036 | 0.839±0.029 |

#### Delta vs Baseline

| n_sup | N_teach | ban_only | hl_only | full_tutor | oracle |
|-------|---------|----------|---------|------------|--------|
| 2 | 1 | +0.006 | +0.011 | +0.011 | +0.361 |
| 4 | 1 | +0.028 | +0.011 | +0.011 | +0.294 |
| 6 | 1 | -0.033 | -0.006 | -0.006 | +0.133 |
| 8 | 1 | +0.006 | +0.011 | +0.011 | +0.189 |
| 2 | 2 | -0.022 | -0.033 | -0.028 | +0.317 |
| 4 | 2 | +0.017 | -0.006 | -0.006 | +0.333 |
| 6 | 2 | **-0.100** | -0.006 | -0.006 | +0.189 |
| 8 | 2 | +0.000 | -0.044 | -0.044 | +0.211 |

#### Learning Gain: OBS_SR vs EVAL_SR

| n_sup | N_teach | Condition | OBS_SR | EVAL_SR | dEval | #TeachEx |
|-------|---------|-----------|--------|---------|-------|----------|
| 2 | 1 | baseline | 0.452 | 0.422 | -0.029 | 3.2 |
| 2 | 1 | full_tutor | 0.452 | 0.433 | -0.018 | 3.1 |
| 6 | 1 | baseline | 0.668 | 0.650 | -0.018 | 2.9 |
| 6 | 1 | full_tutor | 0.668 | 0.644 | -0.024 | 2.9 |
| 8 | 2 | baseline | 0.722 | 0.628 | -0.094 | 4.8 |
| 8 | 2 | full_tutor | 0.722 | 0.583 | -0.138 | 4.9 |

#### Tutor Actions in Teaching Phase (full_tutor)

| n_sup | N_teach | BAN | HIGHLIGHT | RISK_HINT | SKIP | WAIT |
|-------|---------|-----|-----------|-----------|------|------|
| 2 | 1 | 0.00 | 3.62 | 0.00 | 0.00 | 0.00 |
| 2 | 2 | 0.02 | 7.68 | 0.00 | 0.00 | 0.00 |
| 6 | 1 | 0.02 | 3.58 | 0.00 | 0.00 | 0.00 |
| 8 | 2 | 0.02 | 6.33 | 0.00 | 0.00 | 0.00 |

### 5.2 Key Metrics Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Oracle SR | 0.783–0.839 | Upper bound (deterministic scorer) |
| Baseline SR (n_sup=6) | 0.650 | CLS learner without tutor |
| Full tutor delta | -0.044 to +0.011 | **All within 1 SE of zero** |
| BAN delta (n_sup=6, nt=2) | **-0.100** | Actively harmful |
| Tutor fills oracle gap | 0–25% | Effectively zero |
| Avg refresh/block | < 1 | Risk-gated policy working correctly |
| EVAL_SR < OBS_SR | Yes, in most cells | **Negative learning transfer** |

---

## 6. Diagnosed Problems

### 6.1 Tutor is Inert (Primary Failure)

**Symptom**: full_tutor delta ∈ [-0.044, +0.011] across all conditions — indistinguishable from noise.

**Root cause 1: No Theory of Mind**

The tutor uses an oracle-level `DeterministicSemanticScorer` to compute which cells are discriminative. It never models what the CLS learner actually knows or doesn't know.

```
Tutor's perspective (oracle):  "Cell 3 differs between option A and B"
Learner's CLS state:           "Cell 3? I already know that one" OR "Cell 3? I'm wrong there"
Result:                         HIGHLIGHT is either redundant or amplifies errors
```

**Root cause 2: HIGHLIGHT doesn't persist into eval**

Attention weights are reset per-query (`init_for_query(L)`). HIGHLIGHT only affects the current teaching query. The eval phase starts with uniform attention — no transfer.

**Root cause 3: incremental_study is harmful**

Wrong-pick reveals add (wrong_program, rendered_output) to CLS training data. This contaminates the CLS posterior — the EM re-convergence shifts away from correct rules. Evidence: EVAL_SR < OBS_SR in most conditions.

### 6.2 BAN is Counterproductive

BAN removes options from the menu. This has two harmful effects:
1. Reduces learner's chance of stumbling onto the correct answer
2. Removes potential wrong-pick reveals (learning signals)

At n_sup=6, N_teach=2: BAN delta = **-10.0%**.

### 6.3 RISK_HINT and SKIP Never Selected

- **RISK_HINT**: c_hint=0.3 creates a fixed cost barrier. With safe-biased hazard prior, Q(RISK_HINT) < Q(WAIT)=0 always.
- **SKIP**: c_skip=1.5 is too high. Tutor's mastery estimate never exceeds this threshold.

---

## 7. Hyperparameter Reference

### Environment
| Param | Value | Description |
|-------|-------|-------------|
| K | 10 | Options per menu |
| T_max | 5 | Rounds per query |
| H_0 | 5 | Initial HP |
| n_safe | 6 | Safe options per menu |
| n_risky | 4 | Risky options |
| risk_classes | (1,2,3,4) | Damage values |
| danger_dim | 16 | Danger vector dimension |

### Learner
| Param | Value | Description |
|-------|-------|-------------|
| α_sem | 1.0 | Semantic weight |
| α_risk | 0.5 | Danger weight |
| α_unc | 0.2 | Uncertainty weight |
| β_L | 4.0 | Softmax temperature |
| ε | 0.05 | Lapse rate |
| n_em | 2 | EM iterations |
| ρ_H | 2.0 | Highlight boost |
| Hazard bias | -1.0 | Safe prior: σ(-1)≈0.27 |

### Tutor
| Param | Value | Description |
|-------|-------|-------------|
| β_safe | 1.5 | Safety weight |
| β_IG | 1.0 | Info gain weight |
| β_over | 0.2 | Over-reveal penalty |
| c_ban | 0.0 | BAN cost |
| c_hl | 0.0 | HIGHLIGHT cost |
| c_hint | 0.3 | RISK_HINT cost |
| c_skip | 1.5 | SKIP cost |
| max_highlight_cells | 2 | Max cells per HIGHLIGHT |

---

## 8. What Needs to Change (Next Steps)

### Priority 1: Theory of Mind for Tutor

The tutor needs to model the learner's CLS state, not use oracle discrimination. Options:

- **MCMC/SMC particle filter**: Maintain N particles, each = a possible CLS posterior. Update likelihoods based on observed learner picks. Compute HIGHLIGHT IG from the **learner's estimated perspective**.
- **Amortized inference**: Train a small network to predict CLS confusion patterns from observation traces.

### Priority 2: Fix incremental_study

Wrong-pick reveals contaminate CLS. Options:
- **Negative examples**: Teach CLS "this program does NOT produce Y*" instead of "this program produces Z"
- **Confidence gating**: Only add reveals where CLS confidence is low
- **Separate memory**: Store reveals in episodic memory, not cortex EM

### Priority 3: Cross-query HIGHLIGHT transfer

Current attention resets per query. Options:
- **Persistent attention bias**: Carry forward a prior from teaching to eval
- **Meta-attention**: Learn which cell types are generally important across queries

---

## 9. Test Coverage

83 tests, all passing:
- `test_env.py`: Environment mechanics, phase transitions, risk model
- `test_learner.py`: Policy, semantic scorer, danger head, attention
- `test_tutor.py`: Counterfactual scoring, intervention scoring, profile inference
