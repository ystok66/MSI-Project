#!/usr/bin/env python3
"""Structure + exact-render validator for this candidate pool.

This replaces the older structure-only validator.

By default it validates:
1. file/section structure
2. exact support/query render consistency under the current repository
   `task_adapter.py` semantics

It does not run runtime family validation; use:

```text
python -m cls_option_tutor.experiments.validate_family_candidate_pool \
  --pool-dir cls_option_tutor/cls_family_benchmark_v2/cls_family_benchmark_v2
```

for the full pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = Path(__file__).resolve().parent
REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cls_option_tutor.experiments.family_candidate_pool_tools import (  # noqa: E402
    discover_candidate_pool,
    load_manifest_rows,
    validate_exact_render_for_row,
    validate_structure_for_row,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--out-dir", default=str(ROOT / "validation_outputs"))
    args = parser.parse_args()

    pool = discover_candidate_pool(ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_rows(pool)
    structure_rows = [validate_structure_for_row(pool, row) for row in manifest_rows]
    write_csv(out_dir / "structure_validation.csv", structure_rows)

    structure_valid = sum(
        1 for row in structure_rows if bool(row.get("structure_valid", False))
    )
    if args.structure_only:
        print(f"structure_valid={structure_valid}/{len(structure_rows)}")
        print(f"[ok] wrote {out_dir / 'structure_validation.csv'}")
        return

    exact_rows = [
        validate_exact_render_for_row(pool, row, rho_assist=args.rho)
        for row in manifest_rows
    ]
    write_csv(out_dir / "exact_render_validation.csv", exact_rows)

    exact_valid = sum(
        1 for row in exact_rows if bool(row.get("exact_render_valid", False))
    )
    print(f"structure_valid={structure_valid}/{len(structure_rows)}")
    print(f"exact_render_valid={exact_valid}/{len(exact_rows)}")
    print(f"[ok] wrote {out_dir / 'structure_validation.csv'}")
    print(f"[ok] wrote {out_dir / 'exact_render_validation.csv'}")


if __name__ == "__main__":
    main()

