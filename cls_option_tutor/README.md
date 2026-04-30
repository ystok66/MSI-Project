# cls_option_tutor

`cls_option_tutor` is the current option-world mechanism sandbox for studying:

- learning increment under tutor intervention
- assist leakage from direct answers
- safe diagnostic exploration
- online self-correction
- inverse-prediction-based sparse tutoring

This package is no longer best described as only an option-level shortlist tutor.

The active code path is now:

```text
SparseTutorAgent
+ inverse predictor
+ diagnostic / self_correct tutoring modes
+ global/local probe evaluation
+ causal action-effect audit
```

The frozen active sparse-tutor condition alias is:

```text
SIS_cf_mix_loop_v1
```

which currently resolves to:

```text
SIS_horizon_self_correct_cf_mix_netharm_direct_allow_budgeted_allowctl2_consolidate_tmax5
```

Use the short alias in scripts, tests, benchmark tables, and reports unless
you are explicitly auditing runtime suffix tags.

## Current entrypoints

If you are reading code for current work, start here:

- Main runner:
  [experiments/run_learning_increment_micro.py](/F:/SCAI/Learning-agent/cls_option_tutor/experiments/run_learning_increment_micro.py)
- Main tutor:
  [tutor/sparse_tutor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/sparse_tutor.py)
- Main inverse stack:
  [tutor/inverse_predictor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/inverse_predictor.py),
  [tutor/predictor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/predictor.py),
  [tutor/oracle_predictor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/oracle_predictor.py),
  [tutor/learner_model.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/learner_model.py)
- Main learner:
  [learner/learner_agent.py](/F:/SCAI/Learning-agent/cls_option_tutor/learner/learner_agent.py)
- Main environment:
  [env/option_env.py](/F:/SCAI/Learning-agent/cls_option_tutor/env/option_env.py)
- Main evaluation:
  [eval/autonomous_probe.py](/F:/SCAI/Learning-agent/cls_option_tutor/eval/autonomous_probe.py),
  [eval/local_probe.py](/F:/SCAI/Learning-agent/cls_option_tutor/eval/local_probe.py)
- Main debugging tools:
  [tutor/causal_audit.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/causal_audit.py),
  [tutor/highlight_selection.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/highlight_selection.py)

## Baselines That Still Matter

These are not the mainline, but they are still part of the active research
comparison set:

- [tutor/direct_answer_tutor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/direct_answer_tutor.py)
- [tutor/scripted_protocols.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/scripted_protocols.py)
- [tutor/option_level_tutor.py](/F:/SCAI/Learning-agent/cls_option_tutor/tutor/option_level_tutor.py)
- [exp_option_level.py](/F:/SCAI/Learning-agent/cls_option_tutor/exp_option_level.py)

These remain comparison baselines or archival research paths. They are not the
current sparse-tutor mainline.

## Current bottleneck

The current main research bottleneck is:

```text
Can the tutor reliably close the learning loop
productive reveal -> cue/grace -> correct consolidation
under the frozen loop-v1 mainline?
```

Current family-level reading:

- `SIS_cf_mix_loop_v1` already behaves like a native-like allow gate on the
  default diagnostic distribution.
- The remaining unresolved issue is family-dependent performance:
  `ALLOW_CRITICAL_HEAVY` is viable, while `MIXED_PROD_HARM_HEAVY` and
  `PROTECT_CRITICAL_HEAVY` remain weak.
- `phasecalib_v1` remains diagnostic-only and should not be promoted.

For the integrated family-shaped benchmark interpretation, use:

- [docs/cls_option_tutor/PHASE6I13_GENERATOR_SHAPED_BENCHMARK_REPORT_20260421.md](/F:/SCAI/Learning-agent/docs/cls_option_tutor/PHASE6I13_GENERATOR_SHAPED_BENCHMARK_REPORT_20260421.md)
- [docs/cls_option_tutor/FAMILY_BENCHMARK_TASK_ADAPTER_GENERATION_SPEC_20260421.md](/F:/SCAI/Learning-agent/docs/cls_option_tutor/FAMILY_BENCHMARK_TASK_ADAPTER_GENERATION_SPEC_20260421.md)

This means current work should prioritize:

- learning-loop ledger quality
- allow-family / phase calibration
- productive-reveal opportunity surfacing
- same-wrong / far-wrong failure decomposition once loop entry is fixed

Not current priorities:

- grammar expansion
- synonym-heavy task complexity
- adding `DIRECT_ANSWER` to the sparse tutor action set
- migrating to grid-world before online self-correct is stable

## Documentation

Repository-level organizing docs:

- [docs/CURRENT_MAINLINE.md](/F:/SCAI/Learning-agent/docs/CURRENT_MAINLINE.md)
- [docs/CODE_ROLE_INVENTORY.md](/F:/SCAI/Learning-agent/docs/CODE_ROLE_INVENTORY.md)
- [docs/ARCHIVE_PLAN_20260419.md](/F:/SCAI/Learning-agent/docs/ARCHIVE_PLAN_20260419.md)
- [docs/CURRENT_RESULTS_INDEX.md](/F:/SCAI/Learning-agent/docs/CURRENT_RESULTS_INDEX.md)
- [docs/CLEANUP_BASELINE_20260419.md](/F:/SCAI/Learning-agent/docs/CLEANUP_BASELINE_20260419.md)

Longer phase-history and code-report docs were moved out of the package root:

- [docs/cls_option_tutor/README.md](/F:/SCAI/Learning-agent/docs/cls_option_tutor/README.md)

## Package layout

The package root is intentionally split into four categories:

- Active runtime code:
  `env/`, `learner/`, `tutor/`, `eval/`, `experiments/`, `config.py`
- Active package docs:
  this README plus high-level repo docs under `docs/`
- Benchmark artifacts:
  [results/README.md](/F:/SCAI/Learning-agent/cls_option_tutor/results/README.md)
- Scratch / temporary outputs:
  [tmp/README.md](/F:/SCAI/Learning-agent/cls_option_tutor/tmp/README.md)

Historical text-result bundles that used to live under the accidental nested
path `cls_option_tutor/cls_option_tutor/results/` should live under
`results/legacy_text_runs/` instead.

## Cleanup boundary

The package root should stay focused on:

- active Python package files
- current experiment entrypoints
- minimal package-facing documentation

Long-form reports and phase guides should live under `docs/cls_option_tutor/`,
not in the package root.
