"""Diagnostic episode runner with step-level logging.

Runs 5 seeds × 3 families × {no_tutor, strongest_lever, pre_tpm, post_tpm}
with full 4-phase per-step logging.

Outputs:
  results/step_logs/*.jsonl              — one file per episode
  results/diagnostic_summary.csv         — episode-level aggregates
  results/diagnostic_report.md           — human-readable report
"""
import sys, csv, json
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner, V2EpisodeState
from src.envs.lattice_v2 import CellType
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod
from src.metrics.step_logger import StepLogger

runner = LatticeV2Runner()
SEEDS = [0, 1, 2, 3, 4]
FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTY = "medium"

# ── Condition definitions ────────────────────────────────────────
def _rb_kw():
    return dict(tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5)

FAMILY_CONDITIONS = {
    "fork_trap": {
        "no_tutor":          dict(tutor_mode="none", warning_mode="none"),
        "warning_only":      dict(tutor_mode="none", warning_mode="lane",
                                  intervention_family_mode=True, item_drop_enabled=False),
        "robot_belief_pre":  _rb_kw(),
        "robot_belief_post": _rb_kw(),
    },
    "hazard_belt": {
        "no_tutor":          dict(tutor_mode="none", warning_mode="none"),
        "item_only":         {**_rb_kw(), "allowed_interventions": frozenset({"ITEM_DROP"})},
        "robot_belief_pre":  _rb_kw(),
        "robot_belief_post": _rb_kw(),
    },
    "deadline_gate": {
        "no_tutor":          dict(tutor_mode="none", warning_mode="none"),
        "unlock_only":       {**_rb_kw(), "item_drop_enabled": False,
                              "allowed_interventions": frozenset({"UNLOCK"})},
        "robot_belief_pre":  _rb_kw(),
        "robot_belief_post": _rb_kw(),
    },
}

TPM_FLAGS = {
    "robot_belief_pre":  dict(use_bottleneck_matching=False, use_warn_damping=False,
                              use_unlock_memory=False, use_perceptual_access=False),
    "robot_belief_post": dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True),
}

# ── Monkeypatch ──────────────────────────────────────────────────
_orig_score = ip.score_interventions
_current_flags = {}

def _patched_score(*args, **kwargs):
    cfg = kwargs.get("config")
    if cfg is not None and _current_flags:
        for k, v in _current_flags.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return _orig_score(*args, **kwargs)

ip.score_interventions = _patched_score
runner_mod.score_interventions = _patched_score


def run_diagnostic_episode(family, cond_name, seed):
    """Run one episode with step-level logging."""
    global _current_flags
    kw = dict(FAMILY_CONDITIONS[family][cond_name])
    _current_flags = TPM_FLAGS.get(cond_name, {})

    s = runner.reset(seed=seed, scenario_family=family,
                     latent_mode=True, difficulty=DIFFICULTY, **kw)

    logger = StepLogger(family, DIFFICULTY, cond_name, seed)

    while not s.done:
        # Phase 1: observe
        runner.observe(s)
        logger.record_pre_decision(s)

        # Phase 2: tutor acts
        runner.apply_tutor(s)
        logger.record_post_intervention(s)

        # Phase 3+4: plan, move, outcome, learning
        # We need to capture pre-move position and post-outcome event
        pos_before = tuple(s.agent_pos)
        runner.plan_and_move(s)

        # Determine event type
        if not s.survived:
            event = "death"
        elif s.reached_goal:
            event = "goal"
        elif s.t >= s.t_max and s.done:
            event = "timeout"
        else:
            event = "safe"

        r, c = s.agent_pos
        true_cost = float(s.gridmap.true_cost[r, c]) if hasattr(s.gridmap, 'true_cost') else 1.0
        true_risk = float(s.gridmap.true_risk[r, c])

        logger.record_post_transition(s, event=event,
                                      true_cost=true_cost, true_risk=true_risk)
        logger.record_post_learning(s)

    # Flush logs
    logger.flush()

    # Get metrics
    m = runner.get_metrics(s)
    success = m["reached_goal"] and m["survived"]

    summary = logger.summary
    summary["success"] = int(success)
    summary["death"] = int(not m["survived"])
    summary["steps"] = m["steps"]
    summary["unlocks"] = m.get("unlock_count", 0)
    summary["warns"] = m.get("warn_count", 0)

    return summary


if __name__ == "__main__":
    all_summaries = []
    total = sum(len(FAMILY_CONDITIONS[f]) for f in FAMILIES) * len(SEEDS)
    i = 0

    for fam in FAMILIES:
        for cond in FAMILY_CONDITIONS[fam]:
            for seed in SEEDS:
                i += 1
                print(f"  [{i}/{total}] {fam}/{cond}/s{seed}", file=sys.stderr)
                s = run_diagnostic_episode(fam, cond, seed)
                all_summaries.append(s)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── Write CSV ─────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "diagnostic_summary.csv"
    keys = all_summaries[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    # ── Write markdown report ─────────────────────────────────────
    md_path = out_dir / "diagnostic_report.md"
    from collections import defaultdict

    agg = defaultdict(lambda: {"n": 0, "dt": [], "dt_c": [], "dt_r": [],
                               "dB_lat": [], "dB_pred": [], "bar": [],
                               "sr": []})
    for s in all_summaries:
        k = (s["family"], s["condition"])
        a = agg[k]
        a["n"] += 1
        a["dt"].append(s["mean_delta_theta"])
        a["dt_c"].append(s["mean_delta_theta_c"])
        a["dt_r"].append(s["mean_delta_theta_r"])
        a["dB_lat"].append(s["mean_delta_B_latent"])
        a["dB_pred"].append(s["mean_delta_B_pred"])
        a["bar"].append(s["bar"])
        a["sr"].append(s["success"])

    with open(md_path, "w") as f:
        f.write("# Step-Level Diagnostic Report\n\n")
        f.write("5 seeds × 3 families × 3-4 conditions (medium difficulty)\n\n")

        for fam in FAMILIES:
            f.write(f"## {fam}\n\n")
            f.write("| Condition | SR | mean Δθ | Δθ_c | Δθ_r | ΔB_lat | ΔB_pred | BAR |\n")
            f.write("|-----------|----|---------|----|------|--------|---------|-----|\n")
            for cond in FAMILY_CONDITIONS[fam]:
                a = agg[(fam, cond)]
                n = a["n"]
                sr = sum(a["sr"]) / n
                f.write(
                    f"| {cond} "
                    f"| {sr:.0%} "
                    f"| {np.mean(a['dt']):.6f} "
                    f"| {np.mean(a['dt_c']):.6f} "
                    f"| {np.mean(a['dt_r']):.6f} "
                    f"| {np.mean(a['dB_lat']):.6f} "
                    f"| {np.mean(a['dB_pred']):.6f} "
                    f"| {np.mean(a['bar']):.2f} |\n")
            f.write("\n")

        f.write("## Key Questions This Should Answer\n\n")
        f.write("1. Is Δθ near zero? → learner barely updates → null transfer explained\n")
        f.write("2. Is Δθ_r >> Δθ_c or vice versa? → which head is updating\n")
        f.write("3. Is ΔB_pred > 0 but Δθ ≈ 0? → belief changes but weights don't learn\n")
        f.write("4. Is BAR higher for post_tpm vs pre_tpm? → TPM improves intervention targeting\n")

    print(f"Report -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
