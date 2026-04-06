"""
v1b Benchmark Plots + Acceptance Check.

Reads CSV outputs from run_benchmark_suite.py and run_transfer_suite.py.
Generates:
  1. Difficulty-vs-success rate (per family)
  2. Transfer comparison bar chart
  3. Action frequency breakdown
  4. Acceptance check printout
"""

from __future__ import annotations

import csv
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping plots")


BASELINES = ["no_teacher", "always_help", "oracle", "particle"]
FAMILIES = ["semantic_trap", "planning_trap", "exploration_useful", "mixed"]
DIFFICULTIES = ["easy", "medium", "hard"]
BASELINE_COLORS = {
    "no_teacher": "#888888",
    "always_help": "#e67e22",
    "oracle": "#2ecc71",
    "particle": "#3498db",
}


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                try:
                    if "." in str(row[key]):
                        row[key] = float(row[key])
                    elif row[key].isdigit() or (row[key].startswith("-") and row[key][1:].isdigit()):
                        row[key] = int(row[key])
                except (ValueError, AttributeError):
                    pass
            rows.append(row)
    return rows


def plot_difficulty_vs_success(rows: list[dict], out_dir: Path):
    """Plot constrained success rate vs difficulty for each family."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    fig.suptitle("v1b Benchmark: Constrained Success Rate by Difficulty", fontsize=14)

    for ax, family in zip(axes, FAMILIES):
        for mode in BASELINES:
            srs = []
            for diff in DIFFICULTIES:
                frows = [r for r in rows
                         if r["family"] == family
                         and r["baseline"] == mode
                         and r["difficulty"] == diff]
                if frows:
                    sr = sum(r["constrained_success"] for r in frows) / len(frows) * 100
                else:
                    sr = 0
                srs.append(sr)
            ax.plot(DIFFICULTIES, srs, "o-", label=mode,
                    color=BASELINE_COLORS[mode], linewidth=2, markersize=6)
        ax.set_title(family.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Difficulty")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Constrained Success Rate (%)")

    axes[-1].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_vs_success.png", dpi=150)
    plt.close()
    print(f"  Saved: difficulty_vs_success.png")


def plot_transfer_comparison(rows: list[dict], out_dir: Path):
    """Bar chart: transfer success rate per baseline per family."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(FAMILIES))
    width = 0.2

    for i, mode in enumerate(BASELINES):
        tsrs = []
        for family in FAMILIES:
            frows = [r for r in rows
                     if r["family"] == family and r["baseline"] == mode]
            if frows:
                tsr = sum(r["transfer_success"] for r in frows) / len(frows) * 100
            else:
                tsr = 0
            tsrs.append(tsr)
        ax.bar(x + i * width, tsrs, width,
               label=mode, color=BASELINE_COLORS[mode], alpha=0.85)

    ax.set_xlabel("Map Family")
    ax.set_ylabel("Transfer Success Rate (%)")
    ax.set_title("v1b Transfer: Robot-Free Evaluation on Unseen Maps")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f.replace("_", "\n") for f in FAMILIES])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "transfer_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: transfer_comparison.png")


def plot_action_frequency(rows: list[dict], out_dir: Path):
    """Stacked bar: intervention frequency breakdown per baseline per family."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    fig.suptitle("v1b Intervention Frequency Breakdown", fontsize=14)
    action_types = ["n_wait", "n_warn", "n_unlock", "n_shield"]
    action_labels = ["WAIT", "WARN", "UNLOCK", "SHIELD"]
    action_colors = ["#95a5a6", "#e74c3c", "#2980b9", "#27ae60"]

    for ax, family in zip(axes, FAMILIES):
        x = np.arange(len(BASELINES))
        bottoms = np.zeros(len(BASELINES))

        for at, label, color in zip(action_types, action_labels, action_colors):
            vals = []
            for mode in BASELINES:
                frows = [r for r in rows
                         if r["family"] == family and r["baseline"] == mode]
                if frows:
                    total_actions = sum(
                        r.get("n_wait", 0) + r.get("n_warn", 0)
                        + r.get("n_unlock", 0) + r.get("n_shield", 0)
                        for r in frows
                    )
                    at_total = sum(r.get(at, 0) for r in frows)
                    vals.append(at_total / max(total_actions, 1) * 100)
                else:
                    vals.append(0)
            ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += np.array(vals)

        ax.set_title(family.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(BASELINES, rotation=30, fontsize=8)
        ax.set_ylim(0, 105)
        if ax == axes[0]:
            ax.set_ylabel("Action Frequency (%)")

    axes[-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "action_frequency.png", dpi=150)
    plt.close()
    print(f"  Saved: action_frequency.png")


def acceptance_check(
    interaction_rows: list[dict],
    transfer_rows: list[dict],
):
    """Check if the target baseline ordering emerges."""
    print(f"\n{'=' * 70}")
    print(f"  ACCEPTANCE CHECK")
    print(f"{'=' * 70}")

    checks_passed = 0
    checks_total = 0

    # Check 1: On at least one hard family, oracle+particle > no_teacher on CSR
    for family in FAMILIES:
        checks_total += 1
        no_t = [r for r in interaction_rows
                if r["family"] == family and r["difficulty"] == "hard"
                and r["baseline"] == "no_teacher"]
        oracle = [r for r in interaction_rows
                  if r["family"] == family and r["difficulty"] == "hard"
                  and r["baseline"] == "oracle"]
        particle = [r for r in interaction_rows
                    if r["family"] == family and r["difficulty"] == "hard"
                    and r["baseline"] == "particle"]

        sr_no = sum(r["constrained_success"] for r in no_t) / max(len(no_t), 1)
        sr_or = sum(r["constrained_success"] for r in oracle) / max(len(oracle), 1)
        sr_pt = sum(r["constrained_success"] for r in particle) / max(len(particle), 1)

        teacher_better = (sr_or > sr_no + 0.05) or (sr_pt > sr_no + 0.05)
        status = "✓" if teacher_better else "✗"
        print(f"  [{status}] {family} hard: teacher>no_teacher? "
              f"oracle={sr_or:.1%} particle={sr_pt:.1%} vs no_teacher={sr_no:.1%}")
        if teacher_better:
            checks_passed += 1

    # Check 2: On at least one family, particle > always_help on transfer
    checks_total += 1
    any_particle_better = False
    for family in FAMILIES:
        ah = [r for r in transfer_rows
              if r["family"] == family and r["baseline"] == "always_help"]
        pt = [r for r in transfer_rows
              if r["family"] == family and r["baseline"] == "particle"]
        tsr_ah = sum(r["transfer_success"] for r in ah) / max(len(ah), 1)
        tsr_pt = sum(r["transfer_success"] for r in pt) / max(len(pt), 1)
        if tsr_pt > tsr_ah + 0.02:
            any_particle_better = True
            print(f"  [✓] {family}: particle transfer ({tsr_pt:.1%}) > "
                  f"always_help ({tsr_ah:.1%})")

    if any_particle_better:
        checks_passed += 1
    else:
        print(f"  [✗] No family: particle > always_help on transfer")

    # Check 3: Policy match above chance (>25%)
    checks_total += 1
    pma_rows = [r for r in interaction_rows
                if r["baseline"] == "particle" and r.get("policy_total", 0) > 0]
    if pma_rows:
        avg_pma = np.mean([r["policy_match"] for r in pma_rows])
        pma_ok = avg_pma > 0.25
        status = "✓" if pma_ok else "✗"
        print(f"  [{status}] Policy match to oracle: {avg_pma:.1%} (threshold: 25%)")
        if pma_ok:
            checks_passed += 1
    else:
        print(f"  [✗] No policy match data available")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 70}")
    return checks_passed, checks_total


def main():
    out_dir = Path(PROJECT_ROOT) / "output" / "v1b_benchmark"

    # Load interaction results
    agg_path = out_dir / "aggregate_results.csv"
    if not agg_path.exists():
        print(f"  ERROR: {agg_path} not found. Run run_benchmark_suite.py first.")
        return

    interaction_rows = load_csv(agg_path)
    print(f"  Loaded {len(interaction_rows)} interaction rows")

    # Load transfer results
    transfer_path = out_dir / "transfer_results.csv"
    transfer_rows = []
    if transfer_path.exists():
        transfer_rows = load_csv(transfer_path)
        print(f"  Loaded {len(transfer_rows)} transfer rows")
    else:
        print(f"  Warning: {transfer_path} not found. Skipping transfer plots.")

    # Generate plots
    plot_difficulty_vs_success(interaction_rows, out_dir)
    if transfer_rows:
        plot_transfer_comparison(transfer_rows, out_dir)
    plot_action_frequency(interaction_rows, out_dir)

    # Acceptance check
    acceptance_check(interaction_rows, transfer_rows)


if __name__ == "__main__":
    main()
