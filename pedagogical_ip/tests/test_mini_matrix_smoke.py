"""
Mini-matrix smoke test for scenario families.

Runs a small grid: 3 families × 3 teacher conditions × 3 seeds.
Verifies each cell produces valid EpisodeSummary and completes
without error. This is the integration test before adding
scenario_family to the full phase9 eval matrix.
"""
import pytest
import sys
sys.path.insert(0, ".")

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.metrics.phase9_metrics import (
    compute_episode_summary, EpisodeSummary, aggregate_summaries,
)

# ── Mini-matrix dimensions ───────────────────────────────────────────

FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
TEACHERS = [
    ("no_tutor",      dict(tutor_mode="none", warning_mode="none")),
    ("warning_only",  dict(tutor_mode="none", warning_mode="fixed",
                           lambda_lane_warn=5.0)),
    ("robot_belief",  dict(tutor_mode="none", robot_belief_mode=True,
                           intervention_family_mode=True,
                           item_drop_enabled=True, prefix_horizon=5)),
]
SEEDS = [0, 1, 42]

runner = LatticeV2Runner()


# ── Helpers ──────────────────────────────────────────────────────────

def _run_episode(family, teacher_name, teacher_kw, seed):
    """Run one episode and return (EpisodeSummary, metrics_dict)."""
    s = runner.reset(
        seed=seed,
        scenario_family=family,
        latent_mode=True,
        difficulty="medium",
        **teacher_kw,
    )
    while not s.done:
        runner.step(s)
    summary = compute_episode_summary(
        s, seed=seed,
        agent_level="medium",
        teacher_condition=teacher_name,
        env_condition="medium",
    )
    metrics = runner.get_metrics(s)
    return summary, metrics


# ══════════════════════════════════════════════════════════════════════
# 1. Every cell completes without error
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("teacher_name,teacher_kw", TEACHERS,
                         ids=[t[0] for t in TEACHERS])
@pytest.mark.parametrize("seed", SEEDS)
def test_mini_matrix_cell_completes(family, teacher_name, teacher_kw, seed):
    """Each family × teacher × seed cell runs to completion."""
    summary, metrics = _run_episode(family, teacher_name, teacher_kw, seed)
    assert isinstance(summary, EpisodeSummary)
    # Exactly one terminal condition must be true
    assert summary.success or summary.death or summary.timeout
    assert summary.steps > 0
    assert "survived" in metrics
    assert "reached_goal" in metrics


# ══════════════════════════════════════════════════════════════════════
# 2. Per-family aggregation produces valid AggregateMetrics
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", FAMILIES)
def test_mini_matrix_aggregation(family):
    """3 seeds × no_tutor produces valid AggregateMetrics."""
    summaries = []
    for seed in SEEDS:
        s, _ = _run_episode(family, "no_tutor",
                            dict(tutor_mode="none", warning_mode="none"), seed)
        summaries.append(s)
    agg = aggregate_summaries(summaries,
                              agent_level="medium",
                              teacher_condition="no_tutor",
                              env_condition="medium")
    assert agg.n == len(SEEDS)
    assert 0.0 <= agg.success_rate <= 1.0
    assert 0.0 <= agg.death_rate <= 1.0
    assert agg.success_rate + agg.death_rate + agg.timeout_rate == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# 3. Matrix runner expand_matrix includes scenario_family
# ══════════════════════════════════════════════════════════════════════

def test_expand_matrix_includes_families():
    """expand_matrix with smoke=True includes scenario_family in jobs."""
    from scripts.run_phase9_matrix import expand_matrix, load_config
    cfg = load_config()
    jobs = expand_matrix(cfg, smoke=True)
    assert all("scenario_family" in j for j in jobs)
    families_seen = {j["scenario_family"] for j in jobs}
    # Smoke subset should include at least baseline + the 3 priority families
    assert "baseline_v2" in families_seen
    assert "fork_trap" in families_seen


def test_expand_matrix_filter_by_family():
    """Filtering by family reduces matrix size."""
    from scripts.run_phase9_matrix import expand_matrix, load_config
    cfg = load_config()
    full = expand_matrix(cfg, smoke=True)
    filtered = expand_matrix(cfg, smoke=True,
                             filters={"family": "fork_trap"})
    assert len(filtered) < len(full)
    assert all(j["scenario_family"] == "fork_trap" for j in filtered)


def test_expand_matrix_full_family_count():
    """Full matrix has n_families × n_agents × n_teachers × n_envs jobs."""
    from scripts.run_phase9_matrix import expand_matrix, load_config
    cfg = load_config()
    jobs = expand_matrix(cfg)
    n_agents = len(cfg["agent_levels"])
    n_teachers = len(cfg["teacher_conditions"])
    n_envs = len(cfg["env_conditions"])
    n_families = len(cfg.get("scenario_families", ["baseline_v2"]))
    assert len(jobs) == n_agents * n_teachers * n_envs * n_families


# ══════════════════════════════════════════════════════════════════════
# 4. Matrix run_job works with scenario_family
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("family", FAMILIES)
def test_matrix_run_job_with_family(family):
    """run_job produces EpisodeSummary objects for each scenario family."""
    from scripts.run_phase9_matrix import expand_matrix, run_job, load_config
    cfg = load_config()
    jobs = expand_matrix(cfg, smoke=True,
                         filters={"family": family, "teacher": "no_tutor"})
    assert len(jobs) >= 1
    online, transfer = run_job(runner, jobs[0], cfg, smoke=True)
    assert len(online) > 0
    assert all(isinstance(s, EpisodeSummary) for s in online)
