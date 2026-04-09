# CLS Option Tutor

A block-structured, discrete option-selection pedagogical environment for studying tutor interventions in compositional learning.

## Overview

Given a CLS (Compositional Learning System) grammar with nouns and rules, this system:
1. **Generates queries** — novel (program, output) compositions from the grammar
2. **A learner** selects from a menu of K candidate options, learning through feedback
3. **A tutor** observes the learner and intervenes (BAN, HIGHLIGHT, SKIP) to improve outcomes

## Quick Start

```python
from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.tutor_agent import TutorAgent

# Setup
env = OptionEnv(data_dir="BASIC/cls_learner/data")
learner = LearnerAgent(seed=42)
tutor = TutorAgent()

# Run a block with grammar-synthesized queries
block = tutor.run_block(env, learner, "000001", seed=42, synthesize=True)
metrics = OptionEnv.get_block_metrics(block)
print(f"Solve rate: {metrics['solve_rate']:.3f}")
print(f"Damage: {metrics['total_damage']}")
```

## Architecture

```
cls_option_tutor/
  config.py          — FullConfig = EnvConfig + LearnerConfig + TutorConfig
  interfaces.py      — Option, RevealEvent, LearnerStep, TutorStep

  grammar/
    task_adapter.py       — CLS data parser + memoized recursive renderer
    option_generator.py   — Menu gen: exactly-one-correct, 5 distractor strategies
    query_synthesizer.py  — Within-grammar query composition from nouns + rules
    query_families.py     — Family A/B/C/D specs

  env/
    state.py              — QueryState, BlockState, ProfileState
    danger_model.py       — Quadratic feature expansion + sigmoid damage
    interventions.py      — BAN / HIGHLIGHT / SKIP / WAIT pure functions
    option_env.py         — Block/query stepping engine

  learner/
    semantic_scorer.py    — Cell-by-cell mismatch with attention weights
    danger_head.py        — Bayesian linear ridge regression on danger vectors
    attention_model.py    — Uniform baseline + highlight boost
    episodic_memory.py    — Block-scoped reveal history + elimination penalties
    policy.py             — softmax(beta * U) + epsilon-lapse
    cls_adapter.py        — CLSAgent wrapper with graceful fallback
    learner_agent.py      — Autonomous learner orchestrator

  tutor/
    profile_inference.py  — Grid-based MAP learner profile from observation trace
    counterfactual.py     — Q-value scoring for interventions (anti-oracle verified)
    tutor_policy.py       — argmax Q action selection
    tutor_agent.py        — Full obs -> infer -> teach lifecycle

  eval/
    benchmark.py          — Within-grammar multi-task evaluation harness
```

## Key Design Principles

### Anti-Oracle Constraint (Section 12)
The tutor **never** accesses `option.is_correct`. This is enforced by:
- AST-level static analysis tests verifying no `.is_correct` attribute access
- All tutor decisions based on semantic scores + predicted danger only

### Within-Grammar Multi-Task
`synthesize=True` generates novel query programs by composing the grammar's
nouns and rules, enabling evaluation on **unseen compositions** from the same
production system. This tests generalization within a grammar, not just
memorization of fixed query sets.

### Conservative Interventions
The tutor defaults to WAIT (Q=0 baseline) and only intervenes when
counterfactual Q-value exceeds the wait threshold:
- **BAN**: Only when option danger > mean danger (excess-risk criterion)
- **HIGHLIGHT**: Only when information gain > over-reveal penalty + cost
- **SKIP**: Only when HP critically low AND learner genuinely confused

## Running Tests

```bash
# All phases (74 tests)
python -m pytest cls_option_tutor/tests/ -v

# Individual phases
python -m pytest cls_option_tutor/tests/test_env_smoke.py -v  # Phase A: 24 tests
python -m pytest cls_option_tutor/tests/test_learner.py -v    # Phase B: 23 tests
python -m pytest cls_option_tutor/tests/test_tutor.py -v      # Phase C: 16 tests
python -m pytest cls_option_tutor/tests/test_phase_d.py -v    # Phase D: 11 tests
```

## Running Benchmarks

```bash
# Within-grammar benchmark (grammar 000001, 5 blocks, 3 seeds)
python -m cls_option_tutor.eval.benchmark --task 000001 --blocks 5 --seeds 3
```

## Performance Summary

| Condition | Solve Rate | Damage | Notes |
|-----------|-----------|--------|-------|
| Baseline (file queries) | ~0.93 | ~6 | Learner-only, no tutor |
| Tutor (file queries) | ~0.78 | ~10 | Tutor intervenes ~36% |
| Baseline (synth queries) | ~0.90 | ~7 | Novel compositions |
| Tutor (synth queries) | ~0.69 | ~14 | Harder queries |

Generalisation gap (file vs synth): ~0.04 SR — minimal, confirming
the learner generalizes within grammar.

## Dependencies

- NumPy
- Python 3.10+
- CLS data files in `BASIC/cls_learner/data/`
