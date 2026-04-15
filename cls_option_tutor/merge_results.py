"""
merge_results.py — Merge two exp_option_level result files (different nsup sweeps).

Usage:
    python cls_option_tutor/merge_results.py \
        --files cls_option_tutor/results/exp_option_level.txt \
                cls_option_tutor/results/exp_option_level_nsup02.txt \
        --output cls_option_tutor/results/exp_option_level_merged.txt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict


def _mean(vals):
    return round(float(np.mean(vals)), 4) if vals else 0.0

def _se(vals):
    n = len(vals)
    return round(float(np.std(vals) / np.sqrt(max(n, 1))), 4) if vals else 0.0


def load_rows(filepath: str) -> List[dict]:
    """Load per-job raw rows from a result file (the --- JSON --- section)."""
    content = open(filepath, encoding='utf-8').read()
    if '--- JSON ---' not in content:
        raise ValueError(f"No JSON section found in {filepath}")
    json_part = content.split('--- JSON ---')[1].strip()
    # The JSON section is an aggregated nested dict (sc->cond->row), not raw rows.
    # We need to reconstruct raw rows from it. But since the file only stores
    # aggregated data, we actually re-aggregate from the two files below.
    return json.loads(json_part)


def merge_agg(agg_list: List[dict]) -> dict:
    """
    Merge multiple aggregated result dicts (sc->cond->metrics) into one,
    re-weighting by n (sample count).

    For each (sc, cond), we compute a weighted average across files.
    Since each metric is already a mean over n samples, we combine them as:
        merged_mean = sum(mean_i * n_i) / sum(n_i)
    For SE: use pooled SE formula  sqrt(sum(se_i^2 * n_i) / sum(n_i))  (approx).
    """
    # Collect all sc/cond keys
    all_sc = set()
    all_cond = {}
    for agg in agg_list:
        for sc, sc_data in agg.items():
            all_sc.add(sc)
            all_cond.setdefault(sc, set()).update(sc_data.keys())

    merged = {}
    for sc in sorted(all_sc):
        merged[sc] = {}
        for cond in sorted(all_cond.get(sc, [])):
            rows_by_file = []
            for agg in agg_list:
                row = agg.get(sc, {}).get(cond)
                if row is not None:
                    rows_by_file.append(row)

            if not rows_by_file:
                continue

            # Weighted merge on all numeric fields
            total_n = sum(r['n'] for r in rows_by_file)
            result = {'n': total_n}

            # Get all numeric keys from first row
            numeric_keys = [
                k for k, v in rows_by_file[0].items()
                if isinstance(v, (int, float)) and k not in ('n',)
            ]

            for key in numeric_keys:
                if key.endswith('_SE') or key.endswith('_se'):
                    # Pooled SE (approximate): sqrt(sum(se^2 * n) / total_n)
                    pooled = np.sqrt(
                        sum(r.get(key, 0.0)**2 * r['n'] for r in rows_by_file) / max(total_n, 1)
                    )
                    result[key] = round(float(pooled), 4)
                else:
                    # Weighted mean
                    wmean = sum(r.get(key, 0.0) * r['n'] for r in rows_by_file) / max(total_n, 1)
                    result[key] = round(float(wmean), 4)

            merged[sc][cond] = result

    return merged


def format_merged_output(merged: dict, file_list: List[str]) -> str:
    """Format merged dict as a result file (text + JSON section)."""
    lines = []
    lines.append("Option-Level Tutor Experiment — Merged Results")
    lines.append(f"Source files: {', '.join(os.path.basename(f) for f in file_list)}")
    total_n = sum(
        row['n']
        for sc_data in merged.values()
        for row in sc_data.values()
    )
    lines.append(f"Total samples: {total_n}")
    lines.append("")

    for sc_name, sc_data in sorted(merged.items()):
        lines.append(f"\n== Scenario {sc_name} ==")
        for cond, row in sc_data.items():
            lines.append(
                f"  {cond:<32s}  n={row['n']:4d}"
                f"  EVAL-N={row.get('EVAL_SR', 0):.4f}"
                f"  EVAL-Z={row.get('eval_z_sr', 0):.4f}"
                f"  Z-1stOK={row.get('eval_z_1st_ok', 0):.3f}"
                f"  Z-AvgAtt={row.get('eval_z_avg_attempts', 0):.2f}"
                f"  J={row.get('J', 0):+.4f}"
                f"  J_SE={row.get('J_SE', 0):.4f}"
                f"  toutR={row.get('timeout_rate', 0):.4f}"
                f"  ShJS={row.get('choice_shift_js', 0):.4f}"
                f"  NWwin={row.get('nonwait_beats_wait_rate', 0):.3f}"
            )

    lines.append("\n--- JSON ---")
    lines.append(json.dumps(merged, indent=2))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Merge exp_option_level result files")
    parser.add_argument(
        '--files', nargs='+', required=True,
        help='Result files to merge (space separated)'
    )
    parser.add_argument(
        '--output',
        default='cls_option_tutor/results/exp_option_level_merged.txt',
        help='Output merged file'
    )
    args = parser.parse_args()

    print(f"Loading {len(args.files)} result files...")
    agg_list = []
    for filepath in args.files:
        print(f"  {filepath}")
        agg = load_rows(filepath)
        # Count total n
        total = sum(r['n'] for sc in agg.values() for r in sc.values())
        print(f"    → {total} condition-samples loaded")
        agg_list.append(agg)

    print(f"\nMerging...")
    merged = merge_agg(agg_list)

    # Summary
    for sc in sorted(merged):
        for cond, row in merged[sc].items():
            pass  # just to verify
    total_merged = sum(r['n'] for sc in merged.values() for r in sc.values())
    print(f"  Total merged n: {total_merged}")

    output_str = format_merged_output(merged, args.files)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output_str)

    print(f"\nMerged results saved to: {args.output}")

    # Quick per-scenario summary
    print("\n=== Merged Summary ===")
    for sc_name, sc_data in sorted(merged.items()):
        print(f"\nScenario {sc_name}:")
        for cond, row in sc_data.items():
            print(
                f"  {cond:<32s}  n={row['n']:4d}"
                f"  EVAL-N={row.get('EVAL_SR', 0):.4f}"
                f"  EVAL-Z={row.get('eval_z_sr', 0):.4f}"
                f"  J={row.get('J', 0):+.4f}"
                f"  J_SE={row.get('J_SE', 0):.4f}"
                f"  toutR={row.get('timeout_rate', 0):.4f}"
            )


if __name__ == '__main__':
    main()
