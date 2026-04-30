"""End-to-end validation pipeline for family-benchmark candidate pools.

Pipeline:
1. structure sanity
2. exact render consistency under current task_adapter semantics
3. runtime family validation on exact-valid tasks only
4. formal benchmark slice assembly from accepted tasks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cls_option_tutor.experiments.family_candidate_pool_tools import (
    assemble_formal_slice,
    build_validation_report,
    discover_candidate_pool,
    load_manifest_rows,
    validate_exact_render_for_row,
    validate_runtime_family_for_row,
    validate_structure_for_row,
    write_csv,
)
from cls_option_tutor.experiments.mainline_registry import ACTIVE_MAINLINE_ALIAS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--condition", default=ACTIVE_MAINLINE_ALIAS)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--min-target-rate", type=float, default=0.05)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    pool = discover_candidate_pool(args.pool_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pool.root / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_rows(pool)

    structure_rows = [
        validate_structure_for_row(pool, row)
        for row in manifest_rows
    ]
    write_csv(out_dir / "structure_validation.csv", structure_rows)

    exact_rows = [
        validate_exact_render_for_row(pool, row, rho_assist=args.rho)
        for row in manifest_rows
    ]
    write_csv(out_dir / "exact_render_validation.csv", exact_rows)

    exact_valid_ids = {
        str(row["task_id"])
        for row in exact_rows
        if bool(row.get("exact_render_valid", False))
    }
    runtime_rows = [
        validate_runtime_family_for_row(
            pool,
            row,
            seeds=args.seeds,
            condition=args.condition,
            rho_assist=args.rho,
        )
        for row in manifest_rows
        if str(row["task_id"]) in exact_valid_ids
    ]
    write_csv(out_dir / "runtime_family_validation.csv", runtime_rows)

    formal_summary = assemble_formal_slice(
        pool,
        manifest_rows,
        exact_rows,
        runtime_rows,
        min_target_rate=args.min_target_rate,
        out_dir=out_dir / "formal_slice",
    )

    report = build_validation_report(
        pool=pool,
        structure_rows=structure_rows,
        exact_rows=exact_rows,
        runtime_rows=runtime_rows,
        formal_summary=formal_summary,
        condition=args.condition,
        seeds=args.seeds,
        min_target_rate=args.min_target_rate,
    )
    report_path = out_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")

    structure_valid = sum(
        1 for row in structure_rows if bool(row.get("structure_valid", False))
    )
    exact_valid = sum(
        1 for row in exact_rows if bool(row.get("exact_render_valid", False))
    )
    print(f"[ok] pool={pool.root}")
    print(f"[ok] structure_valid={structure_valid}/{len(structure_rows)}")
    print(f"[ok] exact_render_valid={exact_valid}/{len(exact_rows)}")
    print(f"[ok] runtime_validated={len(runtime_rows)}")
    print(f"[ok] formal_slice_accepted={formal_summary['accepted_count']}")
    print(f"[ok] wrote {report_path}")


if __name__ == "__main__":
    main()

