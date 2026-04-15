# cls_option_tutor — J-Based Option-Level CLS Tutor

**Current architecture**: J = ΔEvalAcc − β·Deaths − γ·Timeouts

This package implements an adaptive, safety-aware tutor for option-level
curriculum learning (CLS). The tutor selects shortlist interventions to
maximize a safety-adjusted evaluation objective.

---

## Architecture Overview

```
exp_option_level.py          ← Experiment runner + CLI
config.py                    ← All hyperparameters (FullConfig)
interfaces.py                ← Shared data types (Option, LearnerStep, etc.)

env/
  option_env.py              ← Environment: observe → teach → evaluate loop
  state.py                   ← BlockState, QueryState, ProfileState
  interventions.py           ← SHORTLIST / BAN / HIGHLIGHT / WAIT actions
  danger_model.py            ← Danger generation for risky options

grammar/
  task_adapter.py            ← Load + render task templates
  option_generator_v2.py     ← V2 menu generator (ProgramPool)
  option_generator.py        ← V1 fallback
  query_synthesizer.py       ← Query synthesis
  query_families.py          ← Query family classification

learner/
  learner_agent.py           ← LearnerAgent: act + CLS integration
  policy.py                  ← LearnerPolicy: softmax pick (sem + risk + unc)
  cls_adapter.py             ← CLSAdapter: study / incremental_study
  danger_head.py             ← DangerHead: Bayesian risk prediction
  attention_model.py         ← Attention weights for CLS scoring
  semantic_scorer.py         ← SemanticScorer: score_option
  episodic_memory.py         ← Episodic memory
  semantic_protocol.py       ← Semantic protocol definitions
  rsa_listener.py            ← [LEGACY] RSA L1 listener (use_rsa=False default)

tutor/
  option_level_tutor.py      ← J-based tutor: Q_T decision + shortlist selection
  g_learn.py                 ← G_learn estimator: ProbeEvaluator + OracleDistanceSurrogate

tests/
  test_g_learn.py            ← 18 tests for g_learn.py
  test_option_level.py       ← 9 tests for option_level_tutor.py
  test_env_smoke.py          ← 24 smoke tests for env infrastructure
  test_learner.py            ← 40 tests for learner modules
```

## Running Experiments

```bash
# Smoke test (fast, 24 jobs)
python cls_option_tutor/exp_option_level.py --smoke --scenario A B --cond no_tutor new_probe

# Full experiment (3200 jobs, 6 workers)
python cls_option_tutor/exp_option_level.py --workers 6

# Budget mode (step-count-based teach phase)
python cls_option_tutor/exp_option_level.py --workers 6 --teach_budget 10
```

## Running Tests

```bash
python -m pytest cls_option_tutor/tests/ -v
```

## Objective Function

```
J = ΔEvalAcc − β·Deaths − γ·Timeouts
  = (EVAL_SR − OBS_SR) − 0.5·DeathRate − 0.2·TimeoutRate
```

The tutor selects `SHORTLIST` vs `WAIT` by estimating expected ΔJ
from probe evaluation (`ProbeEvaluator`) or oracle surrogate
(`OracleDistanceSurrogate`).

## Scenarios

| Scenario | Description |
|----------|-------------|
| A | K=10 >> tau_t=3: shortlist forces sub-menu |
| B | H_0=3 + 7 risky options: safety filter critical |
| C | Deadline + safety simultaneously |
| D | T_max >> K, no risk: no intervention expected |

## Tutor Conditions

| Condition | Strategy |
|-----------|----------|
| `no_tutor` | Baseline: learner always sees full menu |
| `old_tutor` | Legacy BAN/HIGHLIGHT tutor |
| `new_baseline` | J-tutor with random shortlist selection |
| `new_probe` | J-tutor with ProbeEvaluator shortlist ranking |
| `new_oracle_surrogate` | J-tutor with OracleDistanceSurrogate |

## Archived Code

Legacy experiments, RSA implementations, and old eval harnesses are in:
```
archive/2026-04-12/
  experiments/    ← legacy exp_*.py scripts
  tutor/          ← tutor_agent, counterfactual, shadow_learner, etc.
  eval/           ← legacy eval harness
  tests/          ← legacy test files
  results/        ← all historical experiment results
  task_report/    ← historical analysis reports
```
