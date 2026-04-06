"""Quick script to read and print v1e benchmark results."""
import csv, numpy as np
from pathlib import Path

rows = []
# Try v1e first, fall back to v1b
for p in ["output/v1e_benchmark/aggregate_results.csv", "output/v1b_benchmark/aggregate_results.csv"]:
    if Path(p).exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in r:
                    try:
                        if "." in str(r[k]): r[k] = float(r[k])
                        elif r[k].lstrip("-").isdigit(): r[k] = int(r[k])
                    except: pass
                rows.append(r)
        break

families = ["semantic_trap", "planning_trap", "exploration_useful"]
baselines = ["no_teacher", "always_help", "oracle", "oracle_cause", "particle"]
diffs = ["easy", "medium", "hard"]

print(f"Total rows: {len(rows)}\n")

print(f"{'Family':<22s} {'Diff':<7s} {'Baseline':<16s} {'CSR%':>6s} {'Steps':>6s} "
      f"{'CauseAcc':>8s} {'WarnP':>6s} {'WaitS':>6s}")
print("-" * 93)
for fam in families:
    for diff in diffs:
        for mode in baselines:
            fr = [r for r in rows if r["family"]==fam and r["baseline"]==mode and r["difficulty"]==diff]
            if not fr: continue
            n = len(fr)
            csr = sum(r["constrained_success"] for r in fr) / n * 100
            steps = np.mean([r["steps"] for r in fr])
            ca = [r["cause_acc"] for r in fr if r.get("cause_acc") not in (None, "")]
            avg_ca = np.mean([float(x) for x in ca]) if ca else float("nan")
            wp = [r["warn_precision"] for r in fr if r.get("warn_precision") not in (None, "")]
            avg_wp = np.mean([float(x) for x in wp]) if wp else float("nan")
            ws = [r["wait_safety"] for r in fr if r.get("wait_safety") not in (None, "")]
            avg_ws = np.mean([float(x) for x in ws]) if ws else float("nan")
            print(f"{fam:<22s} {diff:<7s} {mode:<16s} {csr:5.1f}% {steps:6.1f} "
                  f"{avg_ca:8.2f} {avg_wp:6.2f} {avg_ws:6.2f}")
        print()
