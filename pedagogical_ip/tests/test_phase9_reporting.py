"""
Tests for Phase 9 reporting pipeline.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, ".")


def _make_sample_results(tmpdir):
    """Create minimal results files for testing."""
    online = [
        {"agent_level": "medium", "teacher_condition": "no_tutor",
         "env_condition": "medium", "n": 3,
         "success_rate": 0.1, "success_rate_sem": 0.05,
         "death_rate": 0.8, "timeout_rate": 0.1,
         "cost_mean": 25.0, "cost_std": 5.0,
         "risk_mean": 3.0, "risk_std": 1.0,
         "intervention_count_mean": 0.0,
         "cost_error_mean": 0.3, "calibration_gap_mean": 0.2,
         "uncertainty_reduction_mean": 0.1,
         "info_gain_mean": 0.5, "boredom_mean": 0.4,
         "frustration_mean": 0.6, "timing_quality_mean": 0.3},
        {"agent_level": "medium", "teacher_condition": "robot_belief",
         "env_condition": "medium", "n": 3,
         "success_rate": 0.7, "success_rate_sem": 0.08,
         "death_rate": 0.2, "timeout_rate": 0.1,
         "cost_mean": 20.0, "cost_std": 4.0,
         "risk_mean": 2.0, "risk_std": 0.8,
         "intervention_count_mean": 3.0,
         "cost_error_mean": 0.15, "calibration_gap_mean": 0.1,
         "uncertainty_reduction_mean": 0.3,
         "info_gain_mean": 1.0, "boredom_mean": 0.2,
         "frustration_mean": 0.3, "timing_quality_mean": 0.7},
    ]
    transfer = [
        {"agent_level": "medium", "teacher_condition": "no_tutor",
         "env_condition": "medium", "n": 2,
         "success_rate": 0.1, "success_rate_sem": 0.05,
         "death_rate": 0.8, "timeout_rate": 0.1,
         "cost_mean": 25.0, "cost_std": 5.0,
         "cost_error_mean": 0.4, "calibration_gap_mean": 0.3},
        {"agent_level": "medium", "teacher_condition": "robot_belief",
         "env_condition": "medium", "n": 2,
         "success_rate": 0.4, "success_rate_sem": 0.1,
         "death_rate": 0.4, "timeout_rate": 0.2,
         "cost_mean": 22.0, "cost_std": 4.5,
         "cost_error_mean": 0.2, "calibration_gap_mean": 0.15},
    ]
    d = Path(tmpdir)
    with open(d / "online_results.json", "w") as f:
        json.dump(online, f)
    with open(d / "transfer_results.json", "w") as f:
        json.dump(transfer, f)
    return online, transfer


def test_results_table_export_runs():
    """CSV export runs without error."""
    from scripts.plot_phase9_results import export_online_table
    with tempfile.TemporaryDirectory() as tmpdir:
        online, _ = _make_sample_results(tmpdir)
        csv_path = os.path.join(tmpdir, "online.csv")
        export_online_table(online, csv_path)
        assert os.path.exists(csv_path)


def test_tradeoff_plot_data_preparable():
    """Help-vs-learning tradeoff data is produced."""
    from scripts.plot_phase9_results import export_tradeoff_data
    with tempfile.TemporaryDirectory() as tmpdir:
        online, transfer = _make_sample_results(tmpdir)
        out = os.path.join(tmpdir, "tradeoff.json")
        export_tradeoff_data(online, transfer, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "online_success_rate" in data[0]
        assert "transfer_success_rate" in data[0]


def test_intervention_usage_plot_data_preparable():
    """Intervention usage data is produced."""
    from scripts.plot_phase9_results import export_intervention_usage
    with tempfile.TemporaryDirectory() as tmpdir:
        online, _ = _make_sample_results(tmpdir)
        out = os.path.join(tmpdir, "usage.json")
        export_intervention_usage(online, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 2


def test_transfer_plot_data_preparable():
    """Transfer table export runs."""
    from scripts.plot_phase9_results import export_transfer_table
    with tempfile.TemporaryDirectory() as tmpdir:
        _, transfer = _make_sample_results(tmpdir)
        csv_path = os.path.join(tmpdir, "transfer.csv")
        export_transfer_table(transfer, csv_path)
        assert os.path.exists(csv_path)


def test_report_schema_contains_required_columns():
    """Online CSV has required columns."""
    from scripts.plot_phase9_results import export_online_table
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as tmpdir:
        online, _ = _make_sample_results(tmpdir)
        csv_path = os.path.join(tmpdir, "online.csv")
        export_online_table(online, csv_path)
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            cols = set(reader.fieldnames)
        required = {"success_rate", "death_rate", "cost_mean",
                    "calibration_gap_mean", "info_gain_mean"}
        assert required.issubset(cols)


def test_plotting_handles_empty_subset():
    """Empty results don't crash export."""
    from scripts.plot_phase9_results import export_online_table
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "empty.csv")
        export_online_table([], csv_path)
        # No file created for empty results is acceptable


def test_reporting_is_read_only():
    """Report export does not modify result files."""
    from scripts.plot_phase9_results import load_results, export_tradeoff_data
    with tempfile.TemporaryDirectory() as tmpdir:
        online, transfer = _make_sample_results(tmpdir)
        # Read original
        with open(os.path.join(tmpdir, "online_results.json")) as f:
            orig = f.read()
        # Export
        export_tradeoff_data(online, transfer,
                            os.path.join(tmpdir, "tradeoff.json"))
        # Check original unchanged
        with open(os.path.join(tmpdir, "online_results.json")) as f:
            after = f.read()
        assert orig == after
