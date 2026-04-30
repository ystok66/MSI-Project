"""Family-tagged trace-slice benchmark report for prereveal state families.

This is not generator shaping. It is a decision-time trace slice over the
current benchmark distribution, intended to answer:

- how often each prereveal family appears
- which families close the learning loop
- which families carry damage / death
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cls_option_tutor.experiments.analysis_runtime import (
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
    fmt,
    mean_or_zero,
    run_teach_block,
)
from cls_option_tutor.experiments.mainline_registry import (
    ACTIVE_MAINLINE_ALIAS,
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS,
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)
from cls_option_tutor.experiments.metrics_extractors import build_allow_family_audit

PHASECALIB_ALIAS = f"{ACTIVE_MAINLINE_ALIAS}_phasecalib"

SLICE_FAMILIES = (
    "NATIVE_LIKE_ALLOW",
    "MIXED_PROD_HARM",
    "PROTECT_CRITICAL",
    "BORING_MASTERY",
)

SLICE_DISPLAY = {
    "NATIVE_LIKE_ALLOW": "ALLOW_CRITICAL_HEAVY",
    "MIXED_PROD_HARM": "MIXED_PROD_HARM_HEAVY",
    "PROTECT_CRITICAL": "PROTECT_CRITICAL_HEAVY",
    "BORING_MASTERY": "BORING_MASTERY_HEAVY",
}

CONDITIONS = (
    ACTIVE_MAINLINE_ALIAS,
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS,
    PHASECALIB_ALIAS,
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)


def _summarize(rows: Sequence[dict]) -> Dict[str, float]:
    return {
        "StateCount": float(len(rows)),
        "Rate": mean_or_zero(1.0 for _ in rows),
        "MeanPProd": mean_or_zero(float(row.get("p_prod_total", 0.0)) for row in rows),
        "MeanHarmMass": mean_or_zero(float(row.get("harm_mass", 0.0)) for row in rows),
        "MeanSafeDiagQualityGap": mean_or_zero(
            float(row.get("safe_diag_quality_gap", 0.0)) for row in rows
        ),
        "AllowPreserveRate": mean_or_zero(
            1.0 if bool(row.get("allow_preserved", False)) else 0.0 for row in rows
        ),
        "ProductiveRevealRate": mean_or_zero(
            1.0 if bool(row.get("productive_reveal_after_state", False)) else 0.0
            for row in rows
        ),
        "LoopCompleteRate": mean_or_zero(
            1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0
            for row in rows
        ),
        "TeachDamageMean": mean_or_zero(
            float(row.get("damage_after_state", 0.0)) for row in rows
        ),
        "DeathBeforeCorrectRate": mean_or_zero(
            1.0 if bool(row.get("death_before_correct_after_state", False)) else 0.0
            for row in rows
        ),
    }


def _render_family_table(family: str, condition_rows: Sequence[tuple[str, Dict[str, float]]]) -> str:
    headers = [
        "Condition",
        "StateCount",
        "MeanPProd",
        "MeanHarmMass",
        "MeanSafeDiagQualityGap",
        "AllowPreserveRate",
        "ProductiveRevealRate",
        "LoopCompleteRate",
        "TeachDamageMean",
        "DeathBeforeCorrectRate",
    ]
    lines = [
        f"### {SLICE_DISPLAY.get(family, family)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for cond, stats in condition_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    cond,
                    str(int(stats["StateCount"])),
                    fmt(stats["MeanPProd"]),
                    fmt(stats["MeanHarmMass"]),
                    fmt(stats["MeanSafeDiagQualityGap"]),
                    fmt(stats["AllowPreserveRate"]),
                    fmt(stats["ProductiveRevealRate"]),
                    fmt(stats["LoopCompleteRate"]),
                    fmt(stats["TeachDamageMean"]),
                    fmt(stats["DeathBeforeCorrectRate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_report(
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    rho: float,
    generator: str,
    rows_by_condition: Dict[str, List[dict]],
) -> str:
    family_sections = []
    for family in SLICE_FAMILIES:
        condition_rows = []
        for condition in CONDITIONS:
            family_rows = [
                row
                for row in rows_by_condition.get(condition, [])
                if str(row.get("family_split", "")) == family
            ]
            condition_rows.append((condition, _summarize(family_rows)))
        family_sections.append(_render_family_table(family, condition_rows))

    return f"""# Phase 6I.13 Family-Tagged Trace Slices

## Scope

This report slices the current micro benchmark by decision-time prereveal
families. It is a trace-slice audit, not generator-shaped family sampling.

Families included:

- `ALLOW_CRITICAL_HEAVY` -> `NATIVE_LIKE_ALLOW`
- `MIXED_PROD_HARM_HEAVY` -> `MIXED_PROD_HARM`
- `PROTECT_CRITICAL_HEAVY` -> `PROTECT_CRITICAL`
- `BORING_MASTERY_HEAVY` -> `BORING_MASTERY`

## Run Spec

- Tasks: `{", ".join(tasks)}`
- Seeds: `{", ".join(str(s) for s in seeds)}`
- Generator: `{generator}`
- `rho_assist`: `{rho}`

## Conditions

- `{ACTIVE_MAINLINE_ALIAS}`
- `{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}`
- `{PHASECALIB_ALIAS}`
- `{NO_TUTOR_BUDGETED_CONDITION}`
- `{SCRIPTED_SAFE_GOLD_CONDITION}`

## Slice Tables

{chr(10).join(family_sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--generator", default="diagnostic_quota")
    parser.add_argument(
        "--out",
        default="cls_option_tutor/results/e6_micro/phase6i13_family_slice_report.md",
    )
    args = parser.parse_args()

    rows_by_condition: Dict[str, List[dict]] = {cond: [] for cond in CONDITIONS}
    for condition in CONDITIONS:
        for task_id in args.tasks:
            for seed in args.seeds:
                block = run_teach_block(
                    task_id,
                    seed,
                    condition,
                    rho=args.rho,
                    generator=args.generator,
                )
                for row in build_allow_family_audit(block):
                    row["condition"] = condition
                    row["task_id"] = task_id
                    row["seed"] = seed
                    rows_by_condition[condition].append(row)

    report = _build_report(
        tasks=args.tasks,
        seeds=args.seeds,
        rho=args.rho,
        generator=args.generator,
        rows_by_condition=rows_by_condition,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
