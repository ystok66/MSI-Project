"""
Tests for experiment matrix system — Phase 9.
"""

import pytest
import yaml
import json
import os
from pathlib import Path

# Make sure we can import project modules
import sys
sys.path.insert(0, ".")


def _load_config():
    with open("configs/phase9_eval.yaml") as f:
        return yaml.safe_load(f)


def test_phase9_matrix_config_loads():
    """phase9_eval.yaml is valid YAML."""
    cfg = _load_config()
    assert "agent_levels" in cfg
    assert "teacher_conditions" in cfg
    assert "env_conditions" in cfg
    assert "evaluation" in cfg


def test_matrix_expands_agent_teacher_env_grid():
    """Matrix expansion produces correct number of jobs."""
    from scripts.run_phase9_matrix import expand_matrix
    cfg = _load_config()
    jobs = expand_matrix(cfg)
    n_agents = len(cfg["agent_levels"])
    n_teachers = len(cfg["teacher_conditions"])
    n_envs = len(cfg["env_conditions"])
    n_families = len(cfg.get("scenario_families", ["baseline_v2"]))
    assert len(jobs) == n_agents * n_teachers * n_envs * n_families


def test_matrix_can_filter_subset():
    """Filtering reduces matrix size."""
    from scripts.run_phase9_matrix import expand_matrix
    cfg = _load_config()
    full = expand_matrix(cfg)
    filtered = expand_matrix(cfg, filters={"agent": "medium"})
    assert len(filtered) < len(full)
    assert all(j["agent_level"] == "medium" for j in filtered)


def test_matrix_job_schema_stable():
    """Each job has required fields."""
    from scripts.run_phase9_matrix import expand_matrix
    cfg = _load_config()
    jobs = expand_matrix(cfg)
    for j in jobs:
        assert "agent_level" in j
        assert "teacher_condition" in j
        assert "env_condition" in j
        assert "agent_cfg" in j
        assert "teacher_cfg" in j


def test_matrix_runs_small_smoke_subset():
    """Smoke subset runs end-to-end."""
    from scripts.run_phase9_matrix import expand_matrix, run_job, load_config
    from src.envs.lattice_v2_runner import LatticeV2Runner
    cfg = _load_config()
    jobs = expand_matrix(cfg, smoke=True)
    assert len(jobs) > 0
    runner = LatticeV2Runner()
    # Run first job only
    online, transfer = run_job(runner, jobs[0], cfg, smoke=True)
    assert len(online) > 0
    assert len(transfer) > 0


def test_matrix_outputs_episode_summaries():
    """Each job produces EpisodeSummary objects."""
    from scripts.run_phase9_matrix import expand_matrix, run_job
    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.metrics.phase9_metrics import EpisodeSummary
    cfg = _load_config()
    jobs = expand_matrix(cfg, smoke=True)
    runner = LatticeV2Runner()
    online, _ = run_job(runner, jobs[0], cfg, smoke=True)
    assert all(isinstance(s, EpisodeSummary) for s in online)


def test_matrix_keeps_online_and_transfer_distinct():
    """Online and transfer outputs use different types."""
    from scripts.run_phase9_matrix import expand_matrix, run_job
    from src.envs.lattice_v2_runner import LatticeV2Runner
    from src.metrics.phase9_metrics import EpisodeSummary, TransferSummary
    cfg = _load_config()
    jobs = expand_matrix(cfg, smoke=True)
    runner = LatticeV2Runner()
    online, transfer = run_job(runner, jobs[0], cfg, smoke=True)
    assert all(isinstance(s, EpisodeSummary) for s in online)
    assert all(isinstance(s, TransferSummary) for s in transfer)


def test_teacher_condition_allowed_actions_declared():
    """All teacher conditions have allowed_interventions field."""
    cfg = _load_config()
    for name, tc in cfg["teacher_conditions"].items():
        assert "allowed_interventions" in tc, \
            f"teacher '{name}' missing allowed_interventions"


def test_matrix_legacy_path_still_reproducible():
    """Legacy no_tutor path still runs via matrix infrastructure."""
    from scripts.run_phase9_matrix import expand_matrix, run_job
    from src.envs.lattice_v2_runner import LatticeV2Runner
    cfg = _load_config()
    # Find no_tutor/medium/baseline_v2 job
    jobs = expand_matrix(cfg, filters={"teacher": "no_tutor", "env": "medium",
                                        "agent": "medium",
                                        "family": "baseline_v2"})
    assert len(jobs) == 1
    runner = LatticeV2Runner()
    online, transfer = run_job(runner, jobs[0], cfg, smoke=True)
    assert len(online) > 0
