"""Audit why loop_v1 and nativeallow behave almost identically.

This script answers a narrow question:

How much of the current `loop_v1` productive-allow behavior is already
native-like in practice?

It compares decision-time allow-preserved states between:

- SIS_cf_mix_loop_v1
- SIS_cf_mix_loop_v1_nativeallow
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
    ACTIVE_MAINLINE_NATIVEALLOW_ALIAS,
)
from cls_option_tutor.experiments.metrics_extractors import build_allow_family_audit
from cls_option_tutor.tutor.allow_family import FAMILY_NATIVE_LIKE_ALLOW


def _state_key(task_id: str, seed: int, row: dict) -> Tuple[str, int, int, int]:
    return (
        str(task_id),
        int(seed),
        int(row.get("query_id", -1)),
        int(row.get("round_t", -1)),
    )


def _summarize_preserved(rows: Sequence[dict]) -> Dict[str, float]:
    preserved = [row for row in rows if bool(row.get("allow_preserved", False))]
    native = [
        row for row in preserved
        if str(row.get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
    ]
    mixed = [
        row for row in preserved
        if str(row.get("family_split", "")) == "MIXED_PROD_HARM"
    ]
    return {
        "PreservedStateCount": float(len(preserved)),
        "NativeLikePreservedCount": float(len(native)),
        "MixedProdHarmPreservedCount": float(len(mixed)),
        "NativeLikePrecision": (
            len(native) / max(len(preserved), 1)
        ),
        "MeanPProd_Preserved": mean_or_zero(
            float(row.get("p_prod_total", 0.0)) for row in preserved
        ),
        "MeanHarmMass_Preserved": mean_or_zero(
            float(row.get("harm_mass", 0.0)) for row in preserved
        ),
        "MeanSafeDiagQualityGap_Preserved": mean_or_zero(
            float(row.get("safe_diag_quality_gap", 0.0)) for row in preserved
        ),
        "LoopCompleteRate_Preserved": mean_or_zero(
            1.0 if bool(row.get("loop_complete_after_state", False)) else 0.0
            for row in preserved
        ),
    }


def _render_summary_table(rows: Sequence[tuple[str, Dict[str, float]]]) -> str:
    headers = [
        "Condition",
        "PreservedStateCount",
        "NativeLikePreservedCount",
        "MixedProdHarmPreservedCount",
        "NativeLikePrecision",
        "MeanPProd_Preserved",
        "MeanHarmMass_Preserved",
        "MeanSafeDiagQualityGap_Preserved",
        "LoopCompleteRate_Preserved",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, stats in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(int(stats["PreservedStateCount"])),
                    str(int(stats["NativeLikePreservedCount"])),
                    str(int(stats["MixedProdHarmPreservedCount"])),
                    fmt(stats["NativeLikePrecision"]),
                    fmt(stats["MeanPProd_Preserved"]),
                    fmt(stats["MeanHarmMass_Preserved"]),
                    fmt(stats["MeanSafeDiagQualityGap_Preserved"]),
                    fmt(stats["LoopCompleteRate_Preserved"]),
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
    loop_rows: Sequence[dict],
    native_rows: Sequence[dict],
) -> str:
    loop_preserved = [row for row in loop_rows if bool(row.get("allow_preserved", False))]
    native_preserved = [row for row in native_rows if bool(row.get("allow_preserved", False))]

    loop_by_key = {_state_key(row["task_id"], row["seed"], row): row for row in loop_preserved}
    native_by_key = {_state_key(row["task_id"], row["seed"], row): row for row in native_preserved}

    loop_keys = set(loop_by_key)
    native_keys = set(native_by_key)
    shared_keys = sorted(loop_keys & native_keys)
    loop_only_keys = sorted(loop_keys - native_keys)
    native_only_keys = sorted(native_keys - loop_keys)

    loop_summary = _summarize_preserved(loop_rows)
    native_summary = _summarize_preserved(native_rows)

    shared_same_family = mean_or_zero(
        1.0
        if str(loop_by_key[key].get("family_split", "")) == str(native_by_key[key].get("family_split", ""))
        else 0.0
        for key in shared_keys
    )
    loop_only_native_like = mean_or_zero(
        1.0
        if str(loop_by_key[key].get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        else 0.0
        for key in loop_only_keys
    )
    native_only_native_like = mean_or_zero(
        1.0
        if str(native_by_key[key].get("family_split", "")) == FAMILY_NATIVE_LIKE_ALLOW
        else 0.0
        for key in native_only_keys
    )

    if loop_summary["PreservedStateCount"] > 0 and loop_summary["NativeLikePrecision"] >= 0.9:
        conclusion = (
            f"`{ACTIVE_MAINLINE_ALIAS}` already preserves mostly native-like states in practice. "
            f"`{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}` is therefore close to behaviorally redundant on this distribution."
        )
    else:
        conclusion = (
            f"`{ACTIVE_MAINLINE_ALIAS}` still preserves a non-trivial amount of non-native-like states. "
            f"`{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}` should still be treated as a meaningful precision intervention."
        )

    return f"""# Phase 6I.13 Nativeallow Equivalence Audit

## Scope

This audit asks why `{ACTIVE_MAINLINE_ALIAS}` and
`{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}` behave almost identically in the latest
48-run trace audit.

The key quantity is:

```text
NativeLikePrecision
= P(family_split == NATIVE_LIKE_ALLOW | allow_preserved)
```

If current `loop_v1` already preserves almost only native-like states, then the
native-like gate is behaviorally redundant and the next bottleneck is upstream
phase/family exposure, not allow precision.

## Run Spec

- Tasks: `{", ".join(tasks)}`
- Seeds: `{", ".join(str(s) for s in seeds)}`
- Generator: `{generator}`
- `rho_assist`: `{rho}`

## Preserved-State Summary

{_render_summary_table([
    (ACTIVE_MAINLINE_ALIAS, loop_summary),
    (ACTIVE_MAINLINE_NATIVEALLOW_ALIAS, native_summary),
])}

## State Overlap

- Shared preserved state count: `{len(shared_keys)}`
- `{ACTIVE_MAINLINE_ALIAS}`-only preserved state count: `{len(loop_only_keys)}`
- `{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}`-only preserved state count: `{len(native_only_keys)}`
- Shared preserved states with same `family_split`: `{fmt(shared_same_family)}`
- `{ACTIVE_MAINLINE_ALIAS}`-only states that are native-like: `{fmt(loop_only_native_like)}`
- `{ACTIVE_MAINLINE_NATIVEALLOW_ALIAS}`-only states that are native-like: `{fmt(native_only_native_like)}`

## Interpretation

{conclusion}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--generator", default="diagnostic_quota")
    parser.add_argument(
        "--out",
        default="cls_option_tutor/results/e6_micro/phase6i13_nativeallow_equivalence_report.md",
    )
    args = parser.parse_args()

    loop_rows: List[dict] = []
    native_rows: List[dict] = []

    for task_id in args.tasks:
        for seed in args.seeds:
            loop_block = run_teach_block(
                task_id,
                seed,
                ACTIVE_MAINLINE_ALIAS,
                rho=args.rho,
                generator=args.generator,
            )
            for row in build_allow_family_audit(loop_block):
                row["condition"] = ACTIVE_MAINLINE_ALIAS
                row["task_id"] = task_id
                row["seed"] = seed
                loop_rows.append(row)

            native_block = run_teach_block(
                task_id,
                seed,
                ACTIVE_MAINLINE_NATIVEALLOW_ALIAS,
                rho=args.rho,
                generator=args.generator,
            )
            for row in build_allow_family_audit(native_block):
                row["condition"] = ACTIVE_MAINLINE_NATIVEALLOW_ALIAS
                row["task_id"] = task_id
                row["seed"] = seed
                native_rows.append(row)

    report = _build_report(
        tasks=args.tasks,
        seeds=args.seeds,
        rho=args.rho,
        generator=args.generator,
        loop_rows=loop_rows,
        native_rows=native_rows,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
