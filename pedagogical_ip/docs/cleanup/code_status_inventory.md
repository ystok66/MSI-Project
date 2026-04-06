# Code Status Inventory — COMPLETE (Post Step 5 Cleanup)

Generated: 2026-04-01 | Covers all 144 source modules

> **Legend**: Refs = number of other `src/` files that import this module.
> Modules with Refs=0 are "leaf" modules — called from scripts/tests/runners only.

---

## A. `src/agents/` — 39 modules

### A1. Frozen Core (DO NOT MODIFY)

| File | Refs | Role |
|:-----|:----:|:-----|
| `stochastic_agent_policy.py` | **41** | Agent softmax policy, θ-family reward table, BranchAttributes. **Most-imported module in codebase** |
| `belief.py` | **21** | Core belief state. Foundation for all tutor/observer modules |
| `internalization_state_v3.py` | 11 | FactoredInternalizationState: (τ, ν, γ_gen, γ_spec, κ) |
| `agent_belief_state.py` | 11 | AgentBelief wrapper, consumed by posterior/bridge/curriculum |
| `branch_summary.py` | 11 | Summarize branch features for scoring |
| `planner_astar.py` | 9 | A* path planner, consumed by belief_planning + tutors |
| `branch_concepts.py` | 9 | Branch concept features for scorer |
| `branch_scorer_probe.py` | 8 | Probe-based branch scoring |
| `behavior_bridge.py` | 5 | Behavior prediction bridge (m̂ → ẑ) |
| `behavior_probes.py` | 5 | Behavior zone probes |
| `belief_planning.py` | 3 | Canonical planner wrapper |
| `prefix_prediction.py` | 2 | Prefix prediction for paths |
| `internalization_agent.py` | 1 | Agent-side internalization dynamics |

### A2. Canonical Active

| File | Refs | Role |
|:-----|:----:|:-----|
| `risk_model.py` | 11 | Risk model, consumed by cost_risk + envs |
| `cost_risk_model.py` | 7 | Bayesian cost/risk heads |
| `rsa_warning_channel.py` | 3 | RSA warning channel (Step 2) |
| `warning_update.py` | 1 | Warning belief update |
| `observation_model.py` | 3 | Observation model for grid |
| `world_state.py` | 5 | World state representation |
| `route_necessity.py` | 5 | Route necessity computation (consumed by gate + planner) |
| `trainable_bridge.py` | 2 | Trainable behavior bridge |

### A3. Shadow Active

| File | Refs | Role | Status |
|:-----|:----:|:-----|:------:|
| `planner_risk_shadow.py` | 0 | A2 risk-sensitive planner | **SHADOW-READY** |
| `necessity_gate_variants.py` | 0 | N1/N2/N3 gates | NOT PROMOTING |
| `continuous_reward_shadow.py` | 0 | Step 5B reward residuals | FROZEN |

### A4. Research / Paper Baseline

| File | Refs | Role |
|:-----|:----:|:-----|
| `preference_posterior.py` | 2 | v1 θ posterior (pre Step-4, imported by joint_latent_belief + pref_aware_v2) |
| `preference_posterior_v2.py` | 2 | v2 θ posterior (pre Step-4, same importers) |
| `goal_posterior_v1.py` | 4 | v1 goal posterior (pre Step-4, imported by joint_posterior_v2 + tutors) |
| `hierarchical_goal_posterior.py` | 0 | Hierarchical goal posterior (leaf) |
| `goal_factor_posterior.py` | 1 | Factor goal posterior |
| `joint_posterior_v2.py` | 3 | Joint posterior v2 (pre Step-4, imported by CAJT-v3 + JLT-v2 + JT-v2) |
| `joint_latent_belief.py` | 0 | Joint latent belief (leaf, uses pref_posterior v1+v2) |
| `mixed_effects_risk_head.py` | 0 | Mixed effects risk (leaf) |
| `bounded_agent.py` | 1 | Bounded rationality agent |
| `feature_belief.py` | 2 | Feature belief |
| `familiarity.py` | 2 | Familiarity scoring |
| `factor_action_bridge.py` | 0 | Factor action bridge (leaf) |
| `internalization_dynamics_v2.py` | 0 | Older dynamics version (leaf) |

### A5. Protocol / Interface

| File | Refs | Role |
|:-----|:----:|:-----|
| `pragmatic_warning.py` | 0 | PragmaticWarner protocol (leaf, not imported by any src/ module) |
| `belief_protocol.py` | 0 | Belief protocol interface (leaf) |

---

## B. `src/teachers/` — 47 modules

### B1. Frozen Core (DO NOT MODIFY)

| File | Refs | Role |
|:-----|:----:|:-----|
| `internalization_control_tutor_v4.py` | 0 | **Frozen micro tutor** BCICTv4: {WAIT, WARN}. Leaf — called from runners |
| `internalization_observer.py` | 2 | **Frozen 5D observer** RuleBasedMtObserver. Consumed by shadow bridge |

### B2. Canonical Active

| File | Refs | Role |
|:-----|:----:|:-----|
| `action_predictor.py` | **7** | ActionPredictor — inverse planning. Core for posterior + curriculum |
| `compositional_goal_hypotheses.py` | **7** | 8-goal hypothesis space. Step 4 core |
| `joint_goal_pref_posterior.py` | 3 | Joint posterior q(g,θ). **CANONICAL** default=structural |
| `compositional_goal_prior.py` | 1 | Structural/PCFG prior. Step 4 |
| `compositional_goal_bridge.py` | 0 | Goal-conditioned option scoring (leaf) |
| `goal_conditional_curriculum_hook.py` | 0 | Macro curriculum hook (leaf) |
| `warning_utterance_policy.py` | 0 | RSA warning rsa_obs_s1 (leaf) |
| `robot_belief.py` | 5 | Robot belief |
| `robot_belief_over_agent.py` | 2 | Robot belief over agent |
| `intervention_semantics.py` | 2 | Intervention semantics |
| `intervention_policy.py` | 1 | Intervention policy |
| `interventions.py` | 4 | Intervention types |
| `perceptual_model.py` | 3 | Perceptual model |
| `consequence_grounded_option_rollout.py` | 3 | CGOR |
| `option_intervention_controller.py` | 0 | Option controller (leaf) |
| `utilities.py` | 0 | Teacher utilities (leaf) |

### B3. Shadow Active

| File | Refs | Role | Status |
|:-----|:----:|:-----|:------:|
| `micro_bayes_shadow.py` | 1 | Step 1 v1 shadow micro | SHADOW |
| `micro_bayes_shadow_v2.py` | 1 | Step 1 v2 | SHADOW |
| `micro_bayes_shadow_v2_1.py` | 1 | Step 1 v2.1 (latest stable) | SHADOW |
| `micro_bayes_shadow_v3.py` | 1 | Step 1 v3 | SHADOW |
| `p_self_posterior_shadow.py` | 1 | P_self posterior shadow | SHADOW |
| `p_self_calibration.py` | 2 | P_self calibration | SHADOW |
| `shadow_bridge.py` | 0 | Shadow bridge (leaf) | SHADOW |
| `a1mt_observer_shadow_bridge.py` | 0 | Step 3 shadow observer bridge (leaf) | SHADOW |
| `a1mt_observer_shadow_prob.py` | 1 | Step 3 probabilistic shadow observer | SHADOW |
| `a1mt_observer_shadow_types.py` | 2 | Step 3 shadow observer types | SHADOW |
| `effort_latent_shadow.py` | 1 | Step 3 effort latent | SHADOW |
| `macro_predictive_hook.py` | 0 | Macro predictive hook (leaf) | SHADOW |
| `credit_correction.py` | 2 | Credit correction (consumed by v2.1 + v3) | SHADOW |

### B4. Paper / Ablation (keep for comparison)

| File | Refs | Role | Status |
|:-----|:----:|:-----|:------:|
| `bayesian_macro_objective_shadow.py` | 0 | Step 5C Bayes macro (leaf) | NARRATIVE-ONLY |
| `composite_goal_compatibility.py` | 0 | Compatibility bonus (leaf) | DEPRECATED |
| `goal_temptation_posterior.py` | 0 | Temptation posterior (leaf) | PAPER |
| `intervention_risk_head.py` | 1 | Intervention risk head | PAPER |
| `bottleneck_diagnosis.py` | 1 | Bottleneck diagnosis | PAPER |
| `profile_bootstrap.py` | 0 | Profile bootstrap (leaf) | PAPER |
| `profile_state.py` | 2 | Profile state | PAPER |
| `profile_manager.py` | 0 | Profile manager (leaf) | PAPER |

### B5. Legacy Tutors (pre-Step-4, superseded)

| File | Refs | Role |
|:-----|:----:|:-----|
| `calibrated_adaptive_joint_tutor_v3.py` | 0 | CAJT-v3 (leaf, pre-Step-4 tutor) |
| `joint_latent_tutor_v2.py` | 0 | Joint latent tutor v2 (leaf) |
| `joint_tutor_v2.py` | 0 | Joint tutor v2 (leaf) |
| `preference_aware_policy_v2.py` | 1 | Preference-aware policy v2 |
| `agent_predictor.py` | 1 | Agent predictor (vs action_predictor) |
| `time_aware_door_tutor.py` | 1 | Door tutor |
| `block_scoring.py` | 0 | Block scoring (leaf) |
| `cause_scoring.py` | 1 | Cause scoring |

---

## C. `src/envs/` — 18 modules

### C1. Canonical

| File | Refs | Role |
|:-----|:----:|:-----|
| `map_generator.py` | **11** | GridMap dataclass. **Core grid representation** |
| `scenario_families.py` | 7 | Scenario generation (build_gridmap, family configs) |
| `observation_mask.py` | 7 | Observation masking |
| `map_families.py` | 4 | Map family definitions |
| `lattice_v2.py` | 6 | Lattice v2 grid generation |
| `lattice_v2_runner.py` | 2 | Lattice v2 episode runner |
| `lattice_v2_env.py` | 0 | Lattice v2 env wrapper (leaf) |
| `cgc_v2_family.py` | 0 | CGC-v2 episode generation (leaf) |
| `teaching_internalization_corridor.py` | 5 | TIC base |
| `teaching_internalization_corridor_v4.py` | 2 | TIC v4 (latest) |
| `pedagogical_grid.py` | 0 | Pedagogical grid (leaf) |

### C2. Older / Supporting

| File | Refs | Role |
|:-----|:----:|:-----|
| `teaching_internalization_corridor_v2.py` | 0 | TIC v2 (leaf) |
| `teaching_internalization_corridor_v3.py` | 1 | TIC v3 |
| `compositional_goal_corridor.py` | 0 | CGC v1 (leaf) |
| `compositional_goal_corridor_v2.py` | 0 | CGC v2 (leaf, separate from cgc_v2_family) |
| `benchmark_generator.py` | 0 | Benchmark generation (leaf) |
| `semantic_subspace.py` | 1 | Semantic subspace |
| `persistent_profile_mixed_reveal.py` | 0 | Mixed reveal env (leaf) |

---

## D. `src/curriculum/` — 13 modules

| File | Refs | Role |
|:-----|:----:|:-----|
| `lesson_library.py` | 6 | v1 lesson library |
| `lesson_library_v2.py` | 5 | v2 lesson library |
| `curriculum_controller_v13.py` | 1 | Curriculum controller v13 |
| `pedagogical_framework.py` | 0 | Main framework (leaf — entry point) |
| `mastery_model.py` | 2 | Mastery model |
| `pairwise_response_model.py` | 2 | Pairwise response model |
| `risk_budget_calibration.py` | 2 | Risk budget calibration |
| `lesson_response_model.py` | 1 | v1 response model |
| `lesson_response_model_v2.py` | 0 | v2 response model (leaf) |
| `lesson_response_model_v3.py` | 1 | v3 response model |
| `family_prior.py` | 1 | Family prior |
| `adaptive_episode_generator.py` | 1 | v1 episode generator |
| `adaptive_episode_generator_v2.py` | 0 | v2 episode generator (leaf) |
| `dose_budget.py` | 0 | Dose budget (leaf) |

---

## E. `src/metrics/` — 16 modules

| File | Refs | Role |
|:-----|:----:|:-----|
| `self_discovery.py` | **7** | P_self estimation. Core for tutor decision |
| `calibration.py` | 4 | Calibration metrics |
| `teaching_zone.py` | 1 | v1 teaching zone |
| `teaching_zone_v2.py` | 1 | v2 teaching zone |
| `phase9_metrics.py` | 1 | Phase 9 metrics |
| `change_detection.py` | 1 | Change detection |
| `calibrated_confidence.py` | 1 | Calibrated confidence |
| `eval_v1.py` | 0 | v1 evaluation (leaf) |
| `online_metrics.py` | 0 | Online metrics (leaf) |
| `pedagogical_metrics.py` | 0 | Pedagogical metrics (leaf) |
| `transfer_eval.py` | 0 | Transfer eval (leaf) |
| `step_logger.py` | 0 | Step logger (leaf) |
| `overteaching.py` | 0 | Overteaching detection (leaf) |
| `curriculum_metrics.py` | 0 | Curriculum metrics (leaf) |
| `decision_aware_metrics.py` | 0 | Decision-aware metrics (leaf) |
| `decision_info.py` | 0 | Decision info metrics (leaf) |
| `actionability.py` | 0 | Actionability (leaf) |
| `actionability_v2.py` | 0 | Actionability v2 (leaf) |

---

## F. `src/planner/` — 3 modules

| File | Refs | Role |
|:-----|:----:|:-----|
| `branch_candidates.py` | 2 | Branch candidate generation |
| `branch_reranker.py` | 1 | Branch reranking |
| `branch_semantic_score.py` | 1 | Branch semantic scoring |

---

## G. `src/core/` — 2 modules

| File | Refs | Role |
|:-----|:----:|:-----|
| `state_types.py` | 1 | State type definitions |
| `adapters.py` | 0 | Module adapters (leaf) |

---

## H. `src/evals/` — 1 module

| File | Refs | Role |
|:-----|:----:|:-----|
| `pipeline.py` | 0 | Eval pipeline (leaf — entry point) |

---

## I. `src/logging/` — 2 modules

| File | Refs | Role |
|:-----|:----:|:-----|
| `episode_logger.py` | 0 | Episode logger (leaf) |
| `visualize.py` | 0 | Visualization (leaf) |

---

## Summary Statistics

| Tier | Count | Description |
|:-----|:-----:|:------------|
| **Frozen Core** | 15 | Never modify — foundations |
| **Canonical Active** | 32 | Current mainline — maintain carefully |
| **Shadow Active** | 16 | Shadow experiments — don't default-enable |
| **Paper / Ablation** | 20 | Keep for regression / paper comparison |
| **Legacy Superseded** | 8 | Pre-Step-4 tutors, may eventually archive |
| **Supporting / Infra** | 53 | Envs, curriculum, metrics, logging, evals |
| **Total** | **144** | |

### Top-10 Most-Imported Modules (dependency hubs)

| # | Module | Refs | Package |
|:-:|:-------|:----:|:--------|
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
