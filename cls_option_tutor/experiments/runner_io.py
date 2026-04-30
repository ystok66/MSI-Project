"""Job-building and CSV-writing helpers for micro benchmarks."""

from __future__ import annotations

import csv


def build_jobs(task_ids, seeds, conditions, n_sup_values, rho_values, generator_mode, lg_mode, highlight_mode):
    """Build executor job tuples with stable job indices."""
    jobs = []
    idx = 0
    for n_sup in n_sup_values:
        for rho in rho_values:
            for task_id in task_ids:
                for seed in seeds:
                    for cond in conditions:
                        jobs.append(
                            (task_id, seed, cond, n_sup, rho, idx, 0, generator_mode, lg_mode, highlight_mode)
                        )
                        idx += 1

    total = len(jobs)
    return [(j[0], j[1], j[2], j[3], j[4], j[5], total, j[7], j[8], j[9]) for j in jobs]


def write_rows_csv(rows, out_path: str) -> list[str] | None:
    """Write benchmark rows to CSV and return chosen fieldnames."""
    fieldnames = None
    for row in rows:
        if "ERROR" not in row:
            fieldnames = [k for k in row.keys() if k != "_job_idx"]
            break
    if fieldnames is None:
        return None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "_job_idx"})

    return fieldnames
