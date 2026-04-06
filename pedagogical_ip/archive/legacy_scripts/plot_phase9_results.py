"""
Phase 9 Results Reporting and Plot Data Export.

Primary purpose: export CSV / plotting-ready JSON from results.
NOT about making beautiful plots — focus on data export.

Usage:
  python scripts/plot_phase9_results.py               # full export
  python scripts/plot_phase9_results.py --smoke        # smoke subset
"""

import sys
sys.path.insert(0, ".")

import argparse
import csv
import json
from pathlib import Path


def load_results(output_dir: str = "output/phase9"):
    """Load online and transfer results."""
    d = Path(output_dir)
    online = []
    transfer = []
    online_path = d / "online_results.json"
    transfer_path = d / "transfer_results.json"
    if online_path.exists():
        with open(online_path) as f:
            online = json.load(f)
    if transfer_path.exists():
        with open(transfer_path) as f:
            transfer = json.load(f)
    return online, transfer


def export_online_table(results: list, output_path: str):
    """Export online task metrics as CSV."""
    if not results:
        return
    keys = ["agent_level", "teacher_condition", "env_condition", "n",
            "success_rate", "success_rate_sem", "death_rate", "timeout_rate",
            "cost_mean", "cost_std", "risk_mean", "intervention_count_mean",
            "cost_error_mean", "calibration_gap_mean", "uncertainty_reduction_mean",
            "info_gain_mean", "boredom_mean", "frustration_mean", "timing_quality_mean"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Online table: {output_path}")


def export_transfer_table(results: list, output_path: str):
    """Export transfer metrics as CSV."""
    if not results:
        return
    keys = ["agent_level", "teacher_condition", "env_condition", "n",
            "success_rate", "success_rate_sem", "death_rate", "timeout_rate",
            "cost_mean", "cost_std", "cost_error_mean", "calibration_gap_mean"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Transfer table: {output_path}")


def export_tradeoff_data(online: list, transfer: list, output_path: str):
    """Export help-vs-learning tradeoff data as JSON.

    For each teacher condition: online success_rate vs transfer success_rate.
    """
    tradeoff = []
    transfer_map = {
        (t["agent_level"], t["teacher_condition"], t["env_condition"]): t
        for t in transfer
    }
    for o in online:
        key = (o["agent_level"], o["teacher_condition"], o["env_condition"])
        t = transfer_map.get(key, {})
        tradeoff.append({
            "agent_level": o["agent_level"],
            "teacher_condition": o["teacher_condition"],
            "env_condition": o["env_condition"],
            "online_success_rate": o.get("success_rate", 0.0),
            "transfer_success_rate": t.get("success_rate", 0.0),
            "intervention_count": o.get("intervention_count_mean", 0.0),
            "calibration_gap": o.get("calibration_gap_mean", 0.0),
        })
    with open(output_path, "w") as f:
        json.dump(tradeoff, f, indent=2)
    print(f"  Tradeoff data: {output_path}")


def export_intervention_usage(online: list, output_path: str):
    """Export intervention usage per teacher condition."""
    usage = []
    for o in online:
        usage.append({
            "agent_level": o["agent_level"],
            "teacher_condition": o["teacher_condition"],
            "env_condition": o["env_condition"],
            "intervention_count_mean": o.get("intervention_count_mean", 0.0),
            "success_rate": o.get("success_rate", 0.0),
        })
    with open(output_path, "w") as f:
        json.dump(usage, f, indent=2)
    print(f"  Intervention usage: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Report Export")
    parser.add_argument("--output-dir", type=str, default="output/phase9")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    online, transfer = load_results(args.output_dir)
    if not online and not transfer:
        print("No results found. Run run_phase9_matrix.py first.")
        return

    d = Path(args.output_dir)
    export_online_table(online, str(d / "online_table.csv"))
    export_transfer_table(transfer, str(d / "transfer_table.csv"))
    export_tradeoff_data(online, transfer, str(d / "tradeoff_data.json"))
    export_intervention_usage(online, str(d / "intervention_usage.json"))
    print("\nDone.")


if __name__ == "__main__":
    main()
