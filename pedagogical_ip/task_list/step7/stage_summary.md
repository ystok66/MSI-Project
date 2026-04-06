# Step 7 Stage Summary: Codebase Audit, Cleanup & Canonical Mainline Report

> **Handoff document** — self-contained reference for the next agent or collaborator.
> Covers: Post-Step-5 codebase audit, import graph analysis, orphan detection, archive migration, and canonical mainline documentation.
> Does not require conversation history to understand.

---

## 1. Stage Identity & Goal

Step 7 is **not a new mechanism or experiment**. It is a **systematic codebase audit and cleanup** performed after Steps 1–6 / Step 5A.2 convergence. The goals:

> 1. Build a complete import dependency graph of all `src/` modules
> 2. Classify every module by status tier (Frozen / Canonical / Shadow / Paper / Deprecated / Orphan)
> 3. Fix any silent regressions or default-value mismatches
> 4. Archive dead code, legacy scripts, and debug artifacts without breaking any import chain
> 5. Produce a definitive inventory document for future sessions

---

## 2. Critical Finding: prior_mode Default Bug

During audit, a **silent regression** was discovered:

```python
# src/teachers/joint_goal_pref_posterior.py, line 77
# BEFORE (wrong):
prior_mode: str = "legacy_bonus"
# AFTER (fixed):
prior_mode: str = "structural"
```

**Impact**: Any code instantiating `JointGoalPrefPosterior` without explicitly passing `prior_mode` was silently using the deprecated compatibility bonus path instead of the Step 4-promoted structural prior. This made all default-config runs equivalent to the pre-Step-4 architecture.

**Root Cause**: The default was never updated when Step 4 promoted `structural` to canonical. Step 4's experiment scripts all passed `prior_mode` explicitly, so the bug was invisible in test results.

---

## 3. Current Canonical Mainline

### 3.1 Architectural Layers

```
┌───────────────────────────────────────────────────────────────────┐
│  Layer 5 (Step 4): Compositional Goal Hypothesis Space            │
│  G = {4 atomic + 4 composite} goals from CGC-v2                  │
│  compositional_goal_hypotheses.py                                 │
├───────────────────────────────────────────────────────────────────┤
│  Layer 4 (Step 4): Joint Posterior q(g, θ, z)                     │
│  prior_mode = "structural" (CANONICAL DEFAULT)                    │
│  8 goals × 2 prefs × 4 temptation = 64 cells, exact discrete     │
│  joint_goal_pref_posterior.py + compositional_goal_prior.py       │
├───────────────────────────────────────────────────────────────────┤
│  Layer 3 (Stage 6): POMDP Interface + Options                    │
│  WorldState / AgentBelief / RobotBeliefOverAgent / ActionPredictor│
│  {WARN, UNLOCK, ITEM_DROP} via OptionInterventionController       │
├───────────────────────────────────────────────────────────────────┤
│  Layer 2 (Step 2): RSA Warning Channel                            │
│  warning_mode = "rsa_obs_s1" (canonical)                          │
│  rsa_warning_channel.py + warning_utterance_policy.py             │
├───────────────────────────────────────────────────────────────────┤
│  Layer 1 (FROZEN): 5D Observer                                    │
│  m̂_t = (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)                              │
│  internalization_observer.py (RuleBasedMtObserver)                │
├───────────────────────────────────────────────────────────────────┤
│  Layer 0 (FROZEN): Micro Tutor                                    │
│  A_micro = {WAIT, WARN}   (2-act canonical)                      │
│  internalization_control_tutor_v4.py (BCICTv4)                    │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 Canonical Default Configuration

| Parameter | Value | Source |
|:----------|:------|:-------|
| Observer | RuleBasedMtObserver (5D) | Frozen since Stage 5 |
| Micro Tutor | BCICTv4 {WAIT, WARN} | Frozen since Stage 5 |
| Posterior | `JointGoalPrefPosterior` | Step 4 canonical |
| Prior Mode | `structural` | Step 4 promoted (bug fixed in Step 7) |
| Goal Space | 8-goal (4 atomic + 4 composite) | Step 4 / Stage 6 |
| Preference | Θ₂ = {safe, shiny} | Canonical |
| Temptation | z ∈ {0.0, 0.3, 0.6, 0.9} | Stage 6 |
| Warning | `rsa_obs_s1` | Step 2 |
| Planner | canonical `belief_planning` + `planner_astar` | Frozen |
| Reward | Rigid discrete table (4D weights) | Frozen |
| Macro | Hand-crafted curriculum hook | Canonical |
| κ̂ | Additive macro state, NOT posterior latent | Design rule |

### 3.3 Red Lines (DO NOT VIOLATE)

1. **DO NOT** modify frozen observer or micro tutor
2. **DO NOT** put κ̂ into any posterior latent
3. **DO NOT** change the 2-act micro action space
4. **DO NOT** use `prior_mode="legacy_bonus"` in new code
5. **DO NOT** optimize for exact composite label top-1 (observational equivalence)
6. **DO NOT** explode the latent grid beyond tractable exact inference without evidence

---

## 4. Complete Module Inventory (122 active modules)

### 4.1 `src/agents/` — 32 modules, 7,303 lines

#### Frozen Core (DO NOT MODIFY)

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `stochastic_agent_policy.py` | 120 | **41** | Agent softmax policy, θ-family reward table, BranchAttributes. **Most-imported module** |
| `belief.py` | 290 | **21** | Core belief state. Foundation for all tutor/observer modules |
| `internalization_state_v3.py` | 216 | 11 | FactoredInternalizationState: (τ, ν, γ_gen, γ_spec, κ) |
| `agent_belief_state.py` | 202 | 11 | AgentBelief wrapper |
| `branch_summary.py` | 124 | 11 | Branch feature summarization |
| `planner_astar.py` | 524 | 9 | A* path planner |
| `branch_concepts.py` | 146 | 9 | Branch concept features |
| `branch_scorer_probe.py` | 116 | 8 | Probe-based branch scoring |
| `behavior_bridge.py` | 114 | 5 | Behavior prediction bridge (m̂ → ẑ) |
| `behavior_probes.py` | 145 | 5 | Behavior zone probes |
| `belief_planning.py` | 290 | 3 | Canonical planner wrapper |
| `prefix_prediction.py` | 100 | 2 | Prefix path prediction |
| `internalization_agent.py` | 163 | 1 | Agent-side internalization |

#### Canonical Active

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `risk_model.py` | 126 | 11 | Risk model |
| `cost_risk_model.py` | 236 | 7 | Bayesian cost/risk heads |
| `rsa_warning_channel.py` | 498 | 3 | RSA warning channel (Step 2) |
| `world_state.py` | 87 | 5 | World state representation |
| `route_necessity.py` | 137 | 5 | Route necessity computation |
| `observation_model.py` | 228 | 3 | Grid observation model |
| `warning_update.py` | 395 | 1 | Warning belief update |
| `trainable_bridge.py` | 221 | 2 | Trainable behavior bridge |
| `bounded_agent.py` | 315 | 1 | Bounded rationality agent |
| `feature_belief.py` | 201 | 2 | Feature belief |
| `familiarity.py` | 116 | 2 | Familiarity scoring |

#### Shadow Active

| File | Lines | Refs | Status |
|:-----|:-----:|:----:|:------:|
| `planner_risk_shadow.py` | 344 | 0 | **SHADOW-READY** (A2 mode, +44pp SafeTop1) |
| `necessity_gate_variants.py` | 222 | 0 | NOT PROMOTING (GateDiffRate=0 on CGC-v2) |
| `continuous_reward_shadow.py` | 214 | 0 | FROZEN (Step 5B: Δ NLL < 0.002) |

#### Research / Paper Baseline

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `goal_posterior_v1.py` | 143 | 4 | v1 goal posterior (pre Step-4) |
| `preference_posterior.py` | 118 | 2 | v1 θ posterior |
| `preference_posterior_v2.py` | 121 | 2 | v2 θ posterior |
| `joint_posterior_v2.py` | 178 | 3 | Joint posterior v2 (pre Step-4) |
| `goal_factor_posterior.py` | 205 | 1 | Factor goal posterior |

---

### 4.2 `src/teachers/` — 44 modules, 10,690 lines

#### Frozen Core

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `internalization_control_tutor_v4.py` | 455 | 0 | **Frozen micro tutor** BCICTv4 {WAIT, WARN} |
| `internalization_observer.py` | 719 | 2 | **Frozen 5D observer** RuleBasedMtObserver |

#### Canonical Active — Step 4 Posterior

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `compositional_goal_hypotheses.py` | 185 | **7** | 8-goal hypothesis space (4A+4C) |
| `action_predictor.py` | 122 | **7** | ActionPredictor — inverse planning core |
| `joint_goal_pref_posterior.py` | 328 | 3 | q(g,θ,z) joint posterior. **DEFAULT=structural** |
| `compositional_goal_prior.py` | 331 | 1 | Structural/PCFG prior (Step 4) |
| `compositional_goal_bridge.py` | 132 | 0 | CGC-v2 → POMDP adapter |
| `goal_conditional_curriculum_hook.py` | 167 | 0 | Macro scoring with posterior + κ̂ |

#### Canonical Active — POMDP / Intervention

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `robot_belief.py` | 160 | 5 | Robot belief |
| `robot_belief_over_agent.py` | 190 | 2 | Robot belief over agent |
| `intervention_semantics.py` | 221 | 2 | WARN/UNLOCK/ITEM_DROP definitions |
| `consequence_grounded_option_rollout.py` | 180 | 3 | Intervention → BranchAttributes |
| `interventions.py` | 167 | 4 | Intervention types |
| `intervention_policy.py` | 429 | 1 | Intervention policy |
| `option_intervention_controller.py` | 260 | 0 | Family-selective option controller |
| `perceptual_model.py` | 142 | 3 | Perceptual model |
| `warning_utterance_policy.py` | 147 | 0 | RSA warning rsa_obs_s1 |
| `utilities.py` | 95 | 0 | Teacher utilities |

#### Canonical Active — Other

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `agent_predictor.py` | 208 | 1 | Agent predictor |
| `cause_scoring.py` | 364 | 1 | Cause scoring |
| `time_aware_door_tutor.py` | 171 | 1 | Door tutor |
| `preference_aware_policy_v2.py` | 220 | 1 | Preference-aware policy |
| `goal_temptation_posterior.py` | 252 | 0 | q(g,z) posterior |
| `intervention_risk_head.py` | 135 | 1 | Intervention risk head |
| `bottleneck_diagnosis.py` | 145 | 1 | Bottleneck diagnosis |
| `profile_bootstrap.py` | 197 | 0 | Profile bootstrap |
| `profile_state.py` | 103 | 2 | Profile state |
| `profile_manager.py` | 165 | 0 | Profile manager |
| `macro_predictive_hook.py` | 162 | 0 | Macro predictive hook |

#### Shadow Active (Steps 1/3)

| File | Lines | Refs | Status |
|:-----|:-----:|:----:|:------:|
| `micro_bayes_shadow.py` | 270 | 1 | Step 1 v1 micro shadow |
| `micro_bayes_shadow_v2.py` | 263 | 1 | Step 1 v2 |
| `micro_bayes_shadow_v2_1.py` | 247 | 1 | Step 1 v2.1 (latest stable) |
| `micro_bayes_shadow_v3.py` | 260 | 1 | Step 1 v3 |
| `p_self_posterior_shadow.py` | 365 | 1 | P_self posterior shadow |
| `p_self_calibration.py` | 168 | 2 | P_self calibration |
| `shadow_bridge.py` | 205 | 0 | Shadow bridge adapter |
| `a1mt_observer_shadow_prob.py` | 508 | 1 | Step 3 probabilistic shadow observer |
| `a1mt_observer_shadow_bridge.py` | 172 | 0 | Step 3 shadow observer bridge |
| `a1mt_observer_shadow_types.py` | 134 | 2 | Step 3 shadow observer types |
| `effort_latent_shadow.py` | 132 | 1 | Step 3 effort latent |
| `credit_correction.py` | 128 | 2 | Credit correction |

#### Paper / Deprecated

| File | Lines | Refs | Status |
|:-----|:-----:|:----:|:------:|
| `composite_goal_compatibility.py` | 205 | 0 | **DEPRECATED** (no_bonus ≡ legacy_bonus) |
| `bayesian_macro_objective_shadow.py` | 274 | 0 | NARRATIVE-ONLY (Step 5C: 100% baseline match) |

---

### 4.3 `src/envs/` — 13 modules, 6,934 lines

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `scenario_families.py` | 3,049 | 7 | **Largest module.** Scenario generation, family configs |
| `lattice_v2_runner.py` | 826 | 2 | Episode runner |
| `map_families.py` | 725 | 4 | Map family definitions |
| `lattice_v2.py` | 394 | 6 | Lattice v2 grid generation |
| `pedagogical_grid.py` | 396 | 0 | Pedagogical grid env |
| `cgc_v2_family.py` | 326 | 0 | CGC-v2 episode generation |
| `lattice_v2_env.py` | 274 | 0 | Lattice v2 env wrapper |
| `map_generator.py` | 220 | **11** | GridMap dataclass (**core**) |
| `teaching_internalization_corridor.py` | 219 | 5 | TIC base |
| `observation_mask.py` | 153 | 7 | Observation masking |
| `teaching_internalization_corridor_v3.py` | 136 | 1 | TIC v3 |
| `semantic_subspace.py` | 118 | 1 | Semantic subspace |
| `teaching_internalization_corridor_v4.py` | 98 | 2 | TIC v4 (latest) |

---

### 4.4 `src/curriculum/` — 13 modules, 2,229 lines

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `curriculum_controller_v13.py` | 655 | 1 | Curriculum controller v13 |
| `pedagogical_framework.py` | 256 | 0 | Main framework entry point |
| `pairwise_response_model.py` | 203 | 2 | Pairwise response model |
| `adaptive_episode_generator.py` | 192 | 1 | v1 episode generator |
| `lesson_library_v2.py` | 177 | 5 | v2 lesson library |
| `lesson_response_model_v3.py` | 126 | 1 | v3 response model (hierarchical EB) |
| `lesson_library.py` | 116 | 6 | v1 lesson library |
| `adaptive_episode_generator_v2.py` | 109 | 0 | v2 episode generator |
| `family_prior.py` | 106 | 1 | Family prior |
| `lesson_response_model.py` | 101 | 1 | v1 response model |
| `risk_budget_calibration.py` | 83 | 2 | Risk budget calibration |
| `mastery_model.py` | 73 | 2 | Beta-Bernoulli mastery |
| `dose_budget.py` | 32 | 0 | Dose budget |

---

### 4.5 `src/metrics/` — 13 modules, 1,710 lines

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `phase9_metrics.py` | 446 | 1 | Phase 9 metrics suite |
| `step_logger.py` | 393 | 0 | Step logger |
| `teaching_zone_v2.py` | 127 | 1 | v2 teaching zone |
| `transfer_eval.py` | 122 | 0 | Transfer evaluation |
| `calibration.py` | 113 | 4 | Calibration |
| `eval_v1.py` | 111 | 0 | v1 evaluation |
| `self_discovery.py` | 72 | **7** | P_self estimation (**consumed by 7 modules**) |
| `teaching_zone.py` | 67 | 1 | v1 teaching zone |
| `overteaching.py` | 55 | 0 | Overteaching detection |
| `online_metrics.py` | 52 | 0 | Online metrics |
| `calibrated_confidence.py` | 52 | 1 | Calibrated confidence |
| `change_detection.py` | 64 | 1 | Change detection |
| `actionability.py` | 36 | 0 | Actionability |

---

### 4.6 `src/planner/` — 3 modules, 321 lines

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `branch_reranker.py` | 141 | 1 | Branch reranking |
| `branch_semantic_score.py` | 96 | 1 | Branch semantic scoring |
| `branch_candidates.py` | 84 | 2 | Candidate generation |

### 4.7 `src/core/` — 2 modules, 395 lines

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `state_types.py` | 202 | 1 | State type definitions |
| `adapters.py` | 193 | 0 | Module adapters |

### 4.8 `src/evals/` — 1 module + `src/logging/` — 2 modules

| File | Lines | Refs | Role |
|:-----|:-----:|:----:|:-----|
| `evals/pipeline.py` | 253 | 0 | Eval pipeline entry |
| `logging/episode_logger.py` | 113 | 0 | Episode logger |
| `logging/visualize.py` | 130 | 0 | Visualization |

---

## 5. Import Dependency Structure

### 5.1 Top-10 Dependency Hubs

| Rank | Module | Refs | Package |
|:----:|:-------|:----:|:--------|
| 1 | `stochastic_agent_policy` | 41 | agents |
| 2 | `belief` | 21 | agents |
| 3 | `risk_model` | 11 | agents |
| 4 | `map_generator` | 11 | envs |
| 5 | `internalization_state_v3` | 11 | agents |
| 6 | `branch_summary` | 11 | agents |
| 7 | `agent_belief_state` | 11 | agents |
| 8 | `planner_astar` | 9 | agents |
| 9 | `branch_concepts` | 9 | agents |
| 10 | `branch_scorer_probe` | 8 | agents |

> **Implication**: Any change to `stochastic_agent_policy.py` affects 41 modules. This file must be treated as frozen infrastructure.

### 5.2 Leaf Modules (Refs=0, entry points)

32 modules have zero intra-`src/` importers. They are end-points called from scripts, tests, or runners:

- **Tutors**: `internalization_control_tutor_v4`, `calibrated_adaptive_joint_tutor_v3` (archived)
- **Entry points**: `pedagogical_framework`, `pipeline`, `lattice_v2_env`
- **Shadow leaves**: `planner_risk_shadow`, `shadow_bridge`, `a1mt_observer_shadow_bridge`
- **Metrics leaves**: `transfer_eval`, `step_logger`, `eval_v1`, `online_metrics`

---

## 6. Research Steps 1–5: Final Status

### Step 1: Micro Bayes Shadow

| Item | Status |
|:-----|:------:|
| shadow micro tutor (4 versions: v1→v3) | ✅ Completed |
| P_self posterior shadow | ✅ Completed |
| Promotion decision | **Not promoting** — canonical micro sufficient |

### Step 2: RSA Warning

| Item | Status |
|:-----|:------:|
| RSA warning channel | ✅ Canonical (`rsa_obs_s1`) |
| Warning utterance policy | ✅ Canonical |
| Trust-hybrid warning | Deferred |

### Step 3: Probabilistic Shadow Observer

| Item | Status |
|:-----|:------:|
| Shadow observer (Beta/Gaussian) | ✅ Completed |
| Effort latent shadow | ✅ Completed |
| Promotion decision | **Not promoting** — frozen 5D observer sufficient |

### Step 4: Generative Prior

| Item | Status |
|:-----|:------:|
| Structural prior | ✅ **CANONICAL** (default promoted) |
| PCFG prior | ✅ Paper baseline |
| Legacy compatibility bonus | **DEPRECATED** (no_bonus ≡ legacy_bonus) |
| subgoal_marginals() | ✅ Canonical metric |
| prior_mode default fix | ✅ **Fixed in Step 7** (was silently wrong) |

### Step 5A: Risk-Sensitive Planner

| Item | Status |
|:-----|:------:|
| A2 mode (+44pp SafeTop1) | ✅ Shadow-ready |
| Necessity gates (N1/N2/N3) | NOT PROMOTING (GateDiffRate=0) |
| CGC-v2 multi-path validation | Completed, gate not differentiating |

### Step 5B: Continuous Reward

| Item | Status |
|:-----|:------:|
| Reward residuals (B1/B2/B3) | **FROZEN** — Δ NLL < 0.002, no benefit |

### Step 5C: Bayesian Macro Objective

| Item | Status |
|:-----|:------:|
| Unified Bayes objective | **NARRATIVE-ONLY** — 100% agreement with baseline |

---

## 7. Cleanup Actions Performed

### 7.1 Phase 1: Bug Fix + Status Headers

| Action | Count |
|:-------|:-----:|
| prior_mode default `"legacy_bonus"` → `"structural"` | 1 |
| `[STATUS: DEPRECATED]` header on composite_goal_compatibility | 1 |
| `[STATUS: FROZEN]` header on continuous_reward_shadow | 1 |
| `[STATUS: NARRATIVE-ONLY]` header on bayesian_macro_objective_shadow | 1 |
| `[STATUS: NOT PROMOTING]` header on necessity_gate_variants | 1 |
| Docstring reorder (structural first as CANONICAL DEFAULT) | 1 |

### 7.2 Phase 2: Legacy Scripts & Results (141 files)

| Source | Destination | Count |
|:-------|:-----------|:-----:|
| `scripts/run_stage*`, `run_t*`, `run_p*`, etc. | `archive/legacy_runners/` | 64 |
| `scripts/_analyze*`, `_debug*`, `_diagnose*` | `archive/legacy_runners/` | 6 |
| `results/*.txt` (debug output) | `archive/old_reports/` | 52 |
| `results/*.csv`, `*.json` (raw data) | `archive/old_reports/` | 19 |

### 7.3 Phase 3: Orphaned src/ Modules (22 files)

Modules with **zero references** from any active code:

| Package | Files Moved | Examples |
|:--------|:----------:|:---------|
| `agents/` | 7 | `pragmatic_warning`, `hierarchical_goal_posterior`, `joint_latent_belief`, `mixed_effects_risk_head`, `factor_action_bridge`, `internalization_dynamics_v2`, `belief_protocol` |
| `teachers/` | 4 | `calibrated_adaptive_joint_tutor_v3`, `joint_latent_tutor_v2`, `joint_tutor_v2`, `block_scoring` |
| `envs/` | 5 | `compositional_goal_corridor` v1/v2, `teaching_internalization_corridor_v2`, `persistent_profile_mixed_reveal`, `benchmark_generator` |
| `curriculum/` | 1 | `lesson_response_model_v2` |
| `metrics/` | 5 | `decision_info`, `decision_aware_metrics`, `actionability_v2`, `curriculum_metrics`, `pedagogical_metrics` |

All moved to `archive/deprecated/orphaned_src/` preserving package structure.

---

## 8. Final Repository Structure

```
pedagogical_ip/
├── src/                          122 active Python modules
│   ├── agents/                    32 modules, 7,303 lines
│   ├── teachers/                  44 modules, 10,690 lines
│   ├── envs/                      13 modules, 6,934 lines
│   ├── curriculum/                13 modules, 2,229 lines
│   ├── metrics/                   13 modules, 1,710 lines
│   ├── planner/                    3 modules,   321 lines
│   ├── core/                       2 modules,   395 lines
│   ├── evals/                      1 module,    253 lines
│   └── logging/                    2 modules,   243 lines
│                            ──────────────────────────────
│                            TOTAL: 122 modules, ~30,078 lines
│
├── scripts/                       17 files
│   ├── _regression_check.py       Regression tool (10 checks)
│   └── run_step*.py               16 current step scripts
│
├── tests/                         58 test files
│
├── results/                      219 files (reports + step subdirs)
│   ├── *.md                       Report files
│   └── step*/                     Per-step structured results
│
├── archive/                      310 files total
│   ├── deprecated/                27 files (compat + 22 orphans)
│   ├── ablations/                  1 file  (necessity gates)
│   ├── paper_baselines/            2 files (5B reward, 5C macro)
│   ├── legacy_runners/            71 files (old scripts)
│   ├── legacy_scripts/           104 files (pre-existing)
│   ├── legacy_controllers/        12 files
│   ├── legacy_teachers/           16 files
│   ├── legacy_response_models/     6 files
│   └── old_reports/               71 files (debug txt/csv/json)
│
├── docs/cleanup/                   6 files
│   ├── code_status_inventory.md    Complete 144-module inventory
│   ├── archive_manifest.md         All archive moves with rationale
│   ├── migration_notes.md          Next-session guide
│   ├── missing_items_audit.md      Residual audit
│   ├── regression_report.md        Regression results
│   └── import_graph_raw.txt        Full dependency data
│
└── task_list/step7/                This report
```

---

## 9. Regression Verification

10/10 checks passed after every cleanup phase:

| # | Check | What It Validates |
|:-:|:------|:------------------|
| 1 | posterior default = structural | Step 4 promotion is actually active |
| 2 | structural prior normalized | Prior sums to 1.0 |
| 3 | A2 shadow imports | Shadow planner is importable |
| 4 | warning policy imports | RSA warning is importable |
| 5 | frozen observer imports | RuleBasedMtObserver class exists |
| 6 | frozen micro tutor imports | BCICTv4 class exists |
| 7 | 8-goal hypothesis space | DEFAULT_GOAL_SPACE has 8 goals |
| 8 | subgoal_marginals() | Method exists and returns dict |
| 9 | CGC-v2 generation | Episode scenario generator works |
| 10 | deprecated compat importable | Backward compatibility preserved |

---

## 10. Pending / Deferred Decisions

### 10.1 Shadow → Promote Candidates

| Module | Evidence | Next Step |
|:-------|:---------|:----------|
| A2 planner shadow | +44pp SafeTop1 on synthetic | Needs real CGC-v2 multi-path validation |
| Θ_K (5-type pref) | +11-18% NLL on hard cases | Needs systematic held-out eval |

### 10.2 Modules Kept But Low-Activity

These have src/ references (not orphaned) but low recent activity:

| File | Refs | Assessment |
|:-----|:----:|:-----------|
| `goal_posterior_v1.py` | 4 | Pre Step-4, still imported by `joint_posterior_v2` |
| `preference_posterior.py` / `_v2.py` | 2 each | Pre Step-4, still imported by `joint_latent_belief` (now archived) → may become orphan |
| `joint_posterior_v2.py` | 3 | Pre Step-4, imported by legacy tutors |

> After archiving the 22 orphans, these may have lost their importers. A follow-up check is recommended.

### 10.3 Large Files Deserving Review

| File | Lines | Note |
|:-----|:-----:|:-----|
| `scenario_families.py` | 3,049 | Largest single module; may benefit from splitting |
| `lattice_v2_runner.py` | 826 | Episode runner; complex but functional |
| `map_families.py` | 725 | Map definitions; stable |
| `internalization_observer.py` | 719 | Frozen; DO NOT modify |

---

## 11. For the Next Agent: Quick Reference

### "Where is the main entry point?"

- **Episode runner**: `src/envs/lattice_v2_runner.py`
- **Curriculum framework**: `src/curriculum/pedagogical_framework.py`
- **Eval pipeline**: `src/evals/pipeline.py`

### "Where is the posterior?"

`src/teachers/joint_goal_pref_posterior.py` — `JointGoalPrefPosterior` class, `prior_mode="structural"` default.

### "Where is the observer?"

`src/teachers/internalization_observer.py` — `RuleBasedMtObserver` class. **FROZEN.**

### "Where is the micro tutor?"

`src/teachers/internalization_control_tutor_v4.py` — `BCICTv4` class. **FROZEN.**

### "Where is the goal space?"

`src/teachers/compositional_goal_hypotheses.py` — `DEFAULT_GOAL_SPACE` with 8 goals (4 atomic + 4 composite).

### "Where is the structural prior?"

`src/teachers/compositional_goal_prior.py` — `compute_normalized_goal_prior(goal_space, context, cfg)`.

### "How do I run regression?"

```bash
conda activate pedip310
cd pedagogical_ip
python scripts/_regression_check.py
```

### "What got archived?"

See `docs/cleanup/archive_manifest.md` for full list. Key: 22 orphaned src/ modules, 70 legacy scripts, 71 debug files.

---

## 12. Codebase Statistics Summary

| Metric | Value |
|:-------|:------|
| Active src/ modules | 122 |
| Total active lines | ~30,078 |
| Archived files | 310 |
| Active scripts | 17 |
| Test files | 58 |
| Regression checks | 10/10 ✅ |
| Bug fixes (Step 7) | 1 (prior_mode default) |
| Status headers added | 4 |
| Files moved to archive | 163 (scripts + results + orphans) |
