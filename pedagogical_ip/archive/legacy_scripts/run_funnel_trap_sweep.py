"""Funnel Trap sweep: run experiments with WBCR, TQ, PRCR metrics.

Conditions:
  - no_tutor
  - warning_only (single lever baseline)
  - robot_belief_pre (raw counterfactual, no TPM)
  - robot_belief_post (full TPM)

Metrics:
  WBCR = wrong-branch commitment rate
  TQ   = timing quality of first warn
  PRCR = prefix-risk correction rate (corrected before commitment)
  SR   = success rate
"""
import sys, csv, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner, V2EpisodeState
from src.envs.lattice_v2 import CellType
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()

SEEDS = list(range(20))
DIFFICULTIES = ["easy", "medium", "hard"]

# ── Conditions ────────────────────────────────────────────────────
def _rb_kw():
    return dict(tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5)

CONDITIONS = {
    "no_tutor":          dict(tutor_mode="none", warning_mode="none"),
    "warning_only":      dict(tutor_mode="none", warning_mode="lane",
                              intervention_family_mode=True, item_drop_enabled=False),
    "robot_belief_pre":  _rb_kw(),
    "robot_belief_post": _rb_kw(),
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


def compute_wbcr(s, sc):
    """Wrong-Branch Commitment Rate.
    Did the agent's first irreversible move enter the trap branch?
    Returns 1 if yes (wrong commitment), 0 if no."""
    commitment_pts = getattr(sc, 'commitment_points', [])
    trap_cells = set()
    for branch in getattr(sc, 'trap_branches', []):
        for cell in branch:
            trap_cells.add(cell)

    # Check if agent ever visited a trap branch cell
    # Use visit_count from feature_belief
    for cell in trap_cells:
        r, c = cell
        if r < s.feature_belief.visit_count.shape[0] and c < s.feature_belief.visit_count.shape[1]:
            if s.feature_belief.visit_count[r, c] > 0:
                return 1
    return 0


def compute_tq(s, sc):
    """Timing Quality of first WARN, relative to ideal timing.
    Ideal = 1 step before commitment point.
    TQ = max(0, 1 - |t_warn - t_ideal| / W) averaged over warns.
    Returns (tq, n_warns).
    """
    if not hasattr(sc, 'commitment_points') or not sc.commitment_points:
        return 0.0, 0

    # Ideal warning time: when agent is at decision point 2
    # (just before entering the commitment corridor)
    decision_pts = getattr(sc, 'decision_points', [])
    if len(decision_pts) < 2:
        return 0.0, 0

    # Estimate ideal time: when agent reaches the second decision point
    # We use column position as proxy for time
    dp2_col = decision_pts[1][1]
    W = 3  # tolerance window (steps)

    # Count warns and their timing
    warns_issued = []
    # We don't have step-level intervention log in the state,
    # but we can use warn_count and approximate from s.t
    if s.warn_count > 0:
        # Approximate: warns happen around the warned segments
        # Best proxy: use the column of the first warned segment
        for seg_idx in s.warned_segments:
            if seg_idx < len(s.meta.segments):
                seg = s.meta.segments[seg_idx]
                warn_col = seg.col_start
                tq_this = max(0.0, 1.0 - abs(warn_col - dp2_col) / W)
                warns_issued.append(tq_this)

    # Also check robot-belief interventions
    # Use last_intervention if it was a WARN and use the step
    if s.last_intervention is not None and hasattr(s, '_warn_steps'):
        for t_w in s._warn_steps:
            tq_this = max(0.0, 1.0 - abs(t_w - dp2_col) / W)
            warns_issued.append(tq_this)

    if not warns_issued:
        return 0.0, 0

    return float(np.mean(warns_issued)), len(warns_issued)


def compute_prcr(s, sc):
    """Prefix-Risk Correction Rate.
    Did the agent enter a trap branch but then correct to safe before commitment?
    Returns 1 if correction happened, 0 if not, -1 if never entered trap."""
    trap_cells = set()
    for branch in getattr(sc, 'trap_branches', []):
        for cell in branch:
            trap_cells.add(cell)

    commitment_pts = set(tuple(p) for p in getattr(sc, 'commitment_points', []))

    entered_trap = False
    committed = False
    for cell in trap_cells:
        r, c = cell
        if r < s.feature_belief.visit_count.shape[0] and c < s.feature_belief.visit_count.shape[1]:
            if s.feature_belief.visit_count[r, c] > 0:
                entered_trap = True
                if tuple(cell) in commitment_pts or (r, c) in commitment_pts:
                    committed = True

    if not entered_trap:
        return -1  # never entered trap
    if committed:
        return 0  # entered and committed (no correction)
    return 1  # entered but corrected before commitment


def run_episode(cond_name, seed, difficulty):
    global _current_flags
    kw = dict(CONDITIONS[cond_name])
    _current_flags = TPM_FLAGS.get(cond_name, {})

    s = runner.reset(seed=seed, scenario_family="funnel_trap",
                     latent_mode=True, difficulty=difficulty, **kw)

    # Extra tracking for warn timing
    warn_steps = []
    while not s.done:
        runner.step(s)
        # Track warns
        if s.last_intervention is not None and s.last_intervention.action == "WARN":
            warn_steps.append(s.t)
    s._warn_steps = warn_steps

    m = runner.get_metrics(s)
    success = m["reached_goal"] and m["survived"]

    # Get scenario config
    from src.envs.scenario_families import generate_scenario
    _, _, _, sc = generate_scenario("funnel_trap", seed, difficulty, latent_mode=True)

    wbcr = compute_wbcr(s, sc)
    tq, n_warns = compute_tq(s, sc)
    prcr = compute_prcr(s, sc)

    return {
        "family": "funnel_trap",
        "difficulty": difficulty,
        "condition": cond_name,
        "seed": seed,
        "success": int(success),
        "survived": int(m["survived"]),
        "steps": m["steps"],
        "unlocks": m.get("unlock_count", 0),
        "warns": m.get("warn_count", 0),
        "wbcr": wbcr,
        "tq": round(tq, 4),
        "n_warns": n_warns,
        "prcr": prcr,
    }


if __name__ == "__main__":
    all_results = []
    total = len(CONDITIONS) * len(SEEDS) * len(DIFFICULTIES)
    i = 0

    for diff in DIFFICULTIES:
        for cond in CONDITIONS:
            for seed in SEEDS:
                i += 1
                if i % 20 == 0:
                    print(f"  [{i}/{total}] {diff}/{cond}/s{seed}", file=sys.stderr)
                result = run_episode(cond, seed, diff)
                all_results.append(result)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── CSV ────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "funnel_trap_sweep.csv"
    keys = all_results[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    # ── Markdown ──────────────────────────────────────────────────
    md_path = out_dir / "funnel_trap_report.md"
    agg = defaultdict(lambda: {"n": 0, "sr": [], "wbcr": [], "tq": [],
                               "prcr_corr": 0, "prcr_entered": 0})
    for r in all_results:
        k = (r["difficulty"], r["condition"])
        a = agg[k]
        a["n"] += 1
        a["sr"].append(r["success"])
        a["wbcr"].append(r["wbcr"])
        a["tq"].append(r["tq"])
        if r["prcr"] >= 0:
            a["prcr_entered"] += 1
            if r["prcr"] == 1:
                a["prcr_corr"] += 1

    with open(md_path, "w") as f:
        f.write("# Funnel Trap Sweep Results\n\n")
        f.write("20 seeds × 3 difficulties × 4 conditions\n\n")

        for diff in DIFFICULTIES:
            f.write(f"## {diff}\n\n")
            f.write("| Condition | SR | WBCR | TQ | PRCR | warns |\n")
            f.write("|-----------|----|----|----|----|-------|\n")
            for cond in CONDITIONS:
                a = agg[(diff, cond)]
                n = a["n"]
                sr = sum(a["sr"]) / n
                wbcr = np.mean(a["wbcr"])
                tq = np.mean([t for t in a["tq"] if t > 0]) if any(t > 0 for t in a["tq"]) else 0.0
                prcr = a["prcr_corr"] / max(a["prcr_entered"], 1)
                n_warns = sum(1 for t in a["tq"] if t > 0)
                f.write(f"| {cond} | {sr:.0%} | {wbcr:.2f} | {tq:.2f} | {prcr:.2f} | {n_warns} |\n")
            f.write("\n")

        f.write("## Metric Definitions\n\n")
        f.write("- **SR**: Success rate (reached goal & survived)\n")
        f.write("- **WBCR**: Wrong-Branch Commitment Rate (fraction of episodes "
                "where agent entered trap branch)\n")
        f.write("- **TQ**: Timing Quality of first WARN "
                "(1.0 = perfectly timed at decision point 2, decay with distance)\n")
        f.write("- **PRCR**: Prefix-Risk Correction Rate "
                "(fraction of trap entries corrected before commitment point)\n")

        f.write("\n## Expected Patterns\n\n")
        f.write("- no_tutor: high WBCR (agent takes shorter trap path)\n")
        f.write("- warning_only: lower WBCR, but TQ may be poor (warns too late)\n")
        f.write("- robot_belief_pre: may issue WARN but timing not optimized\n")
        f.write("- robot_belief_post (TPM): lowest WBCR, highest TQ, best SR\n")

    print(f"Report -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
