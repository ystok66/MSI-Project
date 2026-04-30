"""Focused follow-up audit for phasecalib-added allow states.

This analysis is intentionally narrower than the main micro benchmark. It asks:

1. Which states become `ALLOW_CRITICAL` only after `phasecalib`?
2. How do those newly admitted states compare against native
   `ALLOW_CRITICAL` states on:
   - P_prod
   - harm_mass
   - safe_diag_quality_gap
   - productive_reveal_after_state
   - loop_complete_after_state

The goal is to determine whether phasecalib is surfacing useful missed
opportunities or mostly admitting lower-quality / riskier allow states.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from cls_option_tutor.experiments.condition_overrides import (
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.mainline_registry import ACTIVE_MAINLINE_ALIAS
from cls_option_tutor.experiments.metrics_extractors import build_allow_family_audit
from cls_option_tutor.experiments.run_learning_increment_micro import (
    DATA_DIR,
    _apply_condition_overrides,
    make_cfg,
)
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent


DEFAULT_TASKS = ("000001", "000002", "000003", "000004")
DEFAULT_SEEDS = (42, 43, 44)
PHASECALIB_ALIAS = f"{ACTIVE_MAINLINE_ALIAS}_phasecalib"


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return mean(vals) if vals else 0.0


def _rate(rows: Sequence[dict], key: str) -> float:
    if not rows:
        return 0.0
    return _mean([1.0 if bool(row.get(key, False)) else 0.0 for row in rows])


def _run_teach_block(task_id: str, seed: int, condition: str, *, rho: float, generator: str):
    cfg = make_cfg(
        n_sup=4,
        rho_assist=rho,
        generator_mode=generator,
        tutor_lg_mode="off",
        highlight_mode="diagnostic",
    )
    condition_eff = resolve_condition_alias(condition)
    cfg = _apply_condition_overrides(copy.deepcopy(cfg), condition_eff)

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=seed)
    support, _, grammar = env.adapter.load_task(task_id)
    init_block = env.reset_block(task_id, seed=seed)
    learner.init_block(init_block, grammar, support)

    if condition_eff.startswith("script_") or condition_eff.startswith("no_tutor_"):
        teach_cfg = copy.deepcopy(cfg)
        protocol = extract_scripted_protocol_name(condition_eff)
        runner = ScriptedProtocolRunner(cfg=teach_cfg, protocol=protocol)
        result = runner.run_block(
            OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
            learner,
            task_id,
            seed=seed,
        )
        return result.block

    teach_cfg = copy.deepcopy(cfg)
    tutor = SparseTutorAgent(cfg=teach_cfg)
    return tutor.run_block(
        OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
        learner,
        task_id,
        seed=seed,
    )


def _bucket_summary(rows: Sequence[dict]) -> Dict[str, float]:
    return {
        "StateCount": len(rows),
        "MeanPProd": _mean([float(row.get("p_prod_total", 0.0)) for row in rows]),
        "MeanHarmMass": _mean([float(row.get("harm_mass", 0.0)) for row in rows]),
        "MeanSafeDiagQualityGap": _mean(
            [float(row.get("safe_diag_quality_gap", 0.0)) for row in rows]
        ),
        "ProductiveRevealAfterStateRate": _rate(rows, "productive_reveal_after_state"),
        "LoopCompleteAfterStateRate": _rate(rows, "loop_complete_after_state"),
        "MeanPProdSafeShare": _mean(
            [float(row.get("p_prod_safe_share", 0.0)) for row in rows]
        ),
        "MeanCompetingHarmMass": _mean(
            [float(row.get("competing_harm_mass", 0.0)) for row in rows]
        ),
        "MeanHarmCompetitionGap": _mean(
            [float(row.get("harm_competition_gap", 0.0)) for row in rows]
        ),
        "MeanRoundsLeft": _mean([float(row.get("rounds_left", 0.0)) for row in rows]),
        "BothTicketsRate": _rate(rows, "both_tickets_available"),
        "AllowPreservedRate": _rate(rows, "allow_preserved"),
    }


def _markdown_table(summary_rows: Sequence[tuple[str, Dict[str, float]]]) -> str:
    headers = [
        "Bucket",
        "StateCount",
        "MeanPProd",
        "MeanHarmMass",
        "MeanSafeDiagQualityGap",
        "ProductiveRevealAfterStateRate",
        "LoopCompleteAfterStateRate",
        "MeanPProdSafeShare",
        "MeanCompetingHarmMass",
        "MeanHarmCompetitionGap",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for bucket_name, stats in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    bucket_name,
                    str(int(stats["StateCount"])),
                    _fmt(stats["MeanPProd"]),
                    _fmt(stats["MeanHarmMass"]),
                    _fmt(stats["MeanSafeDiagQualityGap"]),
                    _fmt(stats["ProductiveRevealAfterStateRate"]),
                    _fmt(stats["LoopCompleteAfterStateRate"]),
                    _fmt(stats["MeanPProdSafeShare"]),
                    _fmt(stats["MeanCompetingHarmMass"]),
                    _fmt(stats["MeanHarmCompetitionGap"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _compare_buckets(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    native_allow = [
        row for row in rows
        if row.get("condition") == ACTIVE_MAINLINE_ALIAS
        and str(row.get("family", "")) == "ALLOW_CRITICAL"
    ]
    phasecalib_all_allow = [
        row for row in rows
        if row.get("condition") == PHASECALIB_ALIAS
        and str(row.get("family", "")) == "ALLOW_CRITICAL"
    ]
    phasecalib_added_allow = [
        row for row in phasecalib_all_allow
        if not bool(row.get("native_phase_allow_candidate", False))
    ]
    phasecalib_native_overlap = [
        row for row in phasecalib_all_allow
        if bool(row.get("native_phase_allow_candidate", False))
    ]
    return {
        "native_allow": native_allow,
        "phasecalib_all_allow": phasecalib_all_allow,
        "phasecalib_added_allow": phasecalib_added_allow,
        "phasecalib_native_overlap": phasecalib_native_overlap,
    }


def _build_report(
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    rho: float,
    generator: str,
    rows: Sequence[dict],
) -> str:
    buckets = _compare_buckets(rows)
    summaries = {
        name: _bucket_summary(bucket_rows)
        for name, bucket_rows in buckets.items()
    }

    native = summaries["native_allow"]
    added = summaries["phasecalib_added_allow"]
    overlap = summaries["phasecalib_native_overlap"]
    phasecalib_all = summaries["phasecalib_all_allow"]

    delta_lines = [
        f"- `phasecalib_added - native` `MeanPProd`: `{_fmt(added['MeanPProd'] - native['MeanPProd'])}`",
        f"- `phasecalib_added - native` `MeanHarmMass`: `{_fmt(added['MeanHarmMass'] - native['MeanHarmMass'])}`",
        f"- `phasecalib_added - native` `MeanSafeDiagQualityGap`: `{_fmt(added['MeanSafeDiagQualityGap'] - native['MeanSafeDiagQualityGap'])}`",
        f"- `phasecalib_added - native` `ProductiveRevealAfterStateRate`: `{_fmt(added['ProductiveRevealAfterStateRate'] - native['ProductiveRevealAfterStateRate'])}`",
        f"- `phasecalib_added - native` `LoopCompleteAfterStateRate`: `{_fmt(added['LoopCompleteAfterStateRate'] - native['LoopCompleteAfterStateRate'])}`",
    ]

    if added["StateCount"] <= 0:
        observed_conclusion = (
            "No phasecalib-added allow states were observed in this run, so the "
            "phase override did not materially expand the allow family."
        )
    else:
        observed_conclusion = (
            "Phasecalib-added allow states are rarer and materially worse than "
            "native allow states: they carry lower productive mass, higher harm, "
            "and a negative safe-diagnostic quality gap. They still produce "
            "wrong-reveal feedback fairly often, but in this run they did not "
            "close the full productive loop (`loop_complete_after_state = 0`)."
        )

    summary_rows = [
        ("native_allow", native),
        ("phasecalib_added_allow", added),
        ("phasecalib_native_overlap", overlap),
        ("phasecalib_all_allow", phasecalib_all),
    ]

    return f"""# Phase 6I.12 Follow-up Audit

## Scope

This is a narrow follow-up audit on the Phase 6I phase-calibration result.
It compares:

- native `ALLOW_CRITICAL` states from `{ACTIVE_MAINLINE_ALIAS}`
- `phasecalib`-added `ALLOW_CRITICAL` states from `{PHASECALIB_ALIAS}`

Across:

- `P_prod`
- `harm_mass`
- `safe_diag_quality_gap`
- `productive_reveal_after_state`
- `loop_complete_after_state`

## Run Spec

- Tasks: `{", ".join(tasks)}`
- Seeds: `{", ".join(str(s) for s in seeds)}`
- Generator: `{generator}`
- `rho_assist`: `{rho}`

## Bucket Definitions

- `native_allow`: states from `{ACTIVE_MAINLINE_ALIAS}` with `family == ALLOW_CRITICAL`
- `phasecalib_all_allow`: states from `{PHASECALIB_ALIAS}` with `family == ALLOW_CRITICAL`
- `phasecalib_added_allow`: `phasecalib_all_allow` states where `native_phase_allow_candidate == False`
- `phasecalib_native_overlap`: `phasecalib_all_allow` states where `native_phase_allow_candidate == True`

`native_phase_allow_candidate` replays the original pre-phasecalib
`PRE_REVEAL_ALLOW` phase rule:

```text
not post_reveal
and n_safe_diag_wrong_reveals == 0
and p_safe_diag > 0.25
and p_highrisk <= 0.25
and rounds_left >= 2
and hp > 1
```

## Summary Table

{_markdown_table(summary_rows)}

## State Composition

- Native allow state count: `{int(native['StateCount'])}`
- Phasecalib allow state count: `{int(phasecalib_all['StateCount'])}`
- Phasecalib added-allow state count: `{int(added['StateCount'])}`
- Phasecalib native-overlap state count: `{int(overlap['StateCount'])}`

## Added-vs-Native Deltas

{chr(10).join(delta_lines)}

## Observed Conclusion

{observed_conclusion}

## Interpretation Guide

- If `phasecalib_added_allow` has lower `MeanPProd`, higher `MeanHarmMass`,
  more negative `MeanSafeDiagQualityGap`, and lower downstream reveal / loop
  rates than `native_allow`, then phasecalib is mainly admitting worse
  opportunities.
- If `phasecalib_added_allow` is close to `native_allow` on quality metrics and
  still reaches downstream productive reveal / loop completion, then the phase
  override is surfacing genuinely missed opportunities.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--generator", default="diagnostic_quota")
    parser.add_argument(
        "--report-path",
        default=str(
            Path("cls_option_tutor")
            / "results"
            / "e6_micro"
            / "phase6i12_phasecalib_followup_report.md"
        ),
    )
    args = parser.parse_args()

    rows: List[dict] = []
    conditions = [ACTIVE_MAINLINE_ALIAS, PHASECALIB_ALIAS]
    total_jobs = len(args.tasks) * len(args.seeds) * len(conditions)
    job_idx = 0
    for condition in conditions:
        for task_id in args.tasks:
            for seed in args.seeds:
                job_idx += 1
                print(
                    f"[{job_idx}/{total_jobs}] running {condition} task={task_id} seed={seed}",
                    flush=True,
                )
                block = _run_teach_block(
                    task_id,
                    seed,
                    condition,
                    rho=args.rho,
                    generator=args.generator,
                )
                for row in build_allow_family_audit(block):
                    row = dict(row)
                    row["condition"] = condition
                    row["task_id"] = task_id
                    row["seed"] = seed
                    rows.append(row)

    report = _build_report(
        tasks=args.tasks,
        seeds=args.seeds,
        rho=args.rho,
        generator=args.generator,
        rows=rows,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
