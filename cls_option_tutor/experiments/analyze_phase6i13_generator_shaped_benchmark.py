"""Generator-shaped family benchmark for Phase 6I.13+.

This script benchmarks the frozen mainline and key baselines under four
generator-shaped opportunity priors:

- ALLOW_CRITICAL_HEAVY
- MIXED_PROD_HARM_HEAVY
- PROTECT_CRITICAL_HEAVY
- BORING_MASTERY_HEAVY

It reports family distribution plus core learning-loop / safety metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from cls_option_tutor.experiments.analysis_runtime import (
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
    fmt,
    mean_or_zero,
    run_teach_block,
)
from cls_option_tutor.experiments.mainline_registry import (
    ACTIVE_MAINLINE_ALIAS,
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)
from cls_option_tutor.experiments.metrics_extractors import build_allow_family_audit

GENERATOR_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("ALLOW_CRITICAL_HEAVY", "diagnostic_quota_allow_heavy"),
    ("MIXED_PROD_HARM_HEAVY", "diagnostic_quota_mixed_prod_harm_heavy"),
    ("PROTECT_CRITICAL_HEAVY", "diagnostic_quota_protect_critical_heavy"),
    ("BORING_MASTERY_HEAVY", "diagnostic_quota_boring_mastery_heavy"),
)

CONDITIONS = (
    ACTIVE_MAINLINE_ALIAS,
    NO_TUTOR_BUDGETED_CONDITION,
    SCRIPTED_SAFE_GOLD_CONDITION,
)


def _rate(rows: Sequence[dict], key: str) -> float:
    return mean_or_zero(1.0 if bool(row.get(key, False)) else 0.0 for row in rows)


def _family_rate(rows: Sequence[dict], family: str) -> float:
    return mean_or_zero(
        1.0 if str(row.get("family_split", "")) == family else 0.0
        for row in rows
    )


def _summarize(rows: Sequence[dict]) -> Dict[str, float]:
    return {
        "StateCount": float(len(rows)),
        "NativeLikeAllowRate": _family_rate(rows, "NATIVE_LIKE_ALLOW"),
        "MixedProdHarmRate": _family_rate(rows, "MIXED_PROD_HARM"),
        "ProtectCriticalRate": _family_rate(rows, "PROTECT_CRITICAL"),
        "BoringMasteryRate": _family_rate(rows, "BORING_MASTERY"),
        "LoopCompleteRate": _rate(rows, "loop_complete_after_state"),
        "ProductiveRevealRate": _rate(rows, "productive_reveal_after_state"),
        "AllowPreserveRate": _rate(rows, "allow_preserved"),
        "DeathBeforeCorrectRate": _rate(rows, "death_before_correct_after_state"),
        "MeanDamageAfterState": mean_or_zero(
            float(row.get("damage_after_state", 0.0)) for row in rows
        ),
        "MeanPProd": mean_or_zero(float(row.get("p_prod_total", 0.0)) for row in rows),
        "MeanHarmMass": mean_or_zero(float(row.get("harm_mass", 0.0)) for row in rows),
    }


def _render_generator_table(
    generator_label: str,
    rows: Sequence[tuple[str, Dict[str, float]]],
) -> str:
    headers = [
        "Condition",
        "StateCount",
        "NativeLikeAllowRate",
        "MixedProdHarmRate",
        "ProtectCriticalRate",
        "BoringMasteryRate",
        "AllowPreserveRate",
        "ProductiveRevealRate",
        "LoopCompleteRate",
        "MeanDamageAfterState",
        "DeathBeforeCorrectRate",
    ]
    lines = [
        f"### {generator_label}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for condition, stats in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(int(stats["StateCount"])),
                    fmt(stats["NativeLikeAllowRate"]),
                    fmt(stats["MixedProdHarmRate"]),
                    fmt(stats["ProtectCriticalRate"]),
                    fmt(stats["BoringMasteryRate"]),
                    fmt(stats["AllowPreserveRate"]),
                    fmt(stats["ProductiveRevealRate"]),
                    fmt(stats["LoopCompleteRate"]),
                    fmt(stats["MeanDamageAfterState"]),
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
    rows_by_generator_condition: Dict[Tuple[str, str], List[dict]],
) -> str:
    sections = []
    for generator_label, generator_mode in GENERATOR_FAMILIES:
        condition_rows = []
        for condition in CONDITIONS:
            rows = rows_by_generator_condition.get((generator_mode, condition), [])
            condition_rows.append((condition, _summarize(rows)))
        sections.append(_render_generator_table(generator_label, condition_rows))

    return f"""# Phase 6I.13 Generator-Shaped Family Benchmark

## Scope

This report runs the active mainline and key baselines under four
generator-shaped opportunity priors.

These are generator priors, not guarantees of downstream decision-time family.

## Run Spec

- Tasks: `{", ".join(tasks)}`
- Seeds: `{", ".join(str(s) for s in seeds)}`
- `rho_assist`: `{rho}`

## Conditions

- `{ACTIVE_MAINLINE_ALIAS}`
- `{NO_TUTOR_BUDGETED_CONDITION}`
- `{SCRIPTED_SAFE_GOLD_CONDITION}`

## Generator-Shaped Benchmark Tables

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument(
        "--out",
        default="cls_option_tutor/results/e6_micro/phase6i13_generator_shaped_benchmark_report.md",
    )
    args = parser.parse_args()

    rows_by_generator_condition: Dict[Tuple[str, str], List[dict]] = {}
    for _, generator_mode in GENERATOR_FAMILIES:
        for condition in CONDITIONS:
            all_rows: List[dict] = []
            for task_id in args.tasks:
                for seed in args.seeds:
                    block = run_teach_block(
                        task_id,
                        seed,
                        condition,
                        rho=args.rho,
                        generator=generator_mode,
                    )
                    all_rows.extend(build_allow_family_audit(block))
            rows_by_generator_condition[(generator_mode, condition)] = all_rows

    report = _build_report(
        tasks=args.tasks,
        seeds=args.seeds,
        rho=args.rho,
        rows_by_generator_condition=rows_by_generator_condition,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
