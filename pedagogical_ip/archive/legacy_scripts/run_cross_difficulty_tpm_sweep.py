"""Cross-difficulty TPM sweep.

Runs: 3 families x 3 difficulties x 4-5 conditions x 20 seeds.

Conditions:
  no_tutor            - baseline, no intervention
  warning_only        - heuristic lane warning, no items
  unlock_only         - robot_belief with only UNLOCK allowed
  item_only           - robot_belief with only ITEM_DROP allowed
  robot_belief_pre    - robot_belief with all TPM flags OFF (Phase 8 equiv)
  robot_belief_post   - robot_belief with full TPM (Phase 10)

Outputs:
  results/tpm_sweep_cross_difficulty.csv
  results/tpm_sweep_cross_difficulty.json
  results/tpm_sweep_summary.md
"""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod  # needed for patching

runner = LatticeV2Runner()
SEEDS = list(range(20))
DIFFICULTIES = ["easy", "medium", "hard"]
FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]

def _rb_base():
    return dict(tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5)

CONDITIONS = {
    "no_tutor":  dict(tutor_mode="none", warning_mode="none"),
    "warning_only": dict(tutor_mode="none", warning_mode="lane",
                         intervention_family_mode=True, item_drop_enabled=False),
    "unlock_only": {**_rb_base(), "item_drop_enabled": False,
                    "allowed_interventions": frozenset({"UNLOCK"})},
    "item_only":   {**_rb_base(),
                    "allowed_interventions": frozenset({"ITEM_DROP"})},
    "robot_belief_pre":  _rb_base(),
    "robot_belief_post": _rb_base(),
}

TPM_FLAGS = {
    "robot_belief_pre":  dict(use_bottleneck_matching=False, use_warn_damping=False,
                              use_unlock_memory=False, use_perceptual_access=False),
    "robot_belief_post": dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True),
}

FAMILY_CONDITIONS = {
    "fork_trap":      ["no_tutor", "warning_only", "robot_belief_pre", "robot_belief_post"],
    "hazard_belt":    ["no_tutor", "warning_only", "item_only", "robot_belief_pre", "robot_belief_post"],
    "deadline_gate":  ["no_tutor", "unlock_only", "robot_belief_pre", "robot_belief_post"],
}

LEVER_MATCH = {"WARN": "epistemic", "UNLOCK": "structural", "ITEM_DROP": "outcome"}

# Monkeypatch: must patch BOTH the source module AND the runner module's reference
_orig_score = ip.score_interventions
_current_flags = {}

def _patched_score(*args, **kwargs):
    cfg = kwargs.get("config")
    if cfg is not None and _current_flags:
        for k, v in _current_flags.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return _orig_score(*args, **kwargs)

# Patch both references so the runner uses the patched version
ip.score_interventions = _patched_score
runner_mod.score_interventions = _patched_score


def run_episode(family, difficulty, cond_name, seed):
    global _current_flags
    kw = dict(CONDITIONS[cond_name])
    _current_flags = TPM_FLAGS.get(cond_name, {})

    s = runner.reset(seed=seed, scenario_family=family,
                     latent_mode=True, difficulty=difficulty, **kw)

    bn_epi = bn_str = bn_out = 0
    total_non_wait = 0
    lever_matched = 0

    while not s.done:
        runner.step(s)
        if s.last_intervention is not None and s.last_intervention.bottleneck is not None:
            bn = s.last_intervention.bottleneck
            dom = bn.dominant
            if dom == "epistemic": bn_epi += 1
            elif dom == "structural": bn_str += 1
            elif dom == "outcome": bn_out += 1
            action = s.last_intervention.action
            if action != "WAIT":
                total_non_wait += 1
                if LEVER_MATCH.get(action) == dom:
                    lever_matched += 1

    m = runner.get_metrics(s)
    if m["reached_goal"] and m["survived"]:
        outcome = "success"
    elif not m["survived"]:
        outcome = "death"
    else:
        outcome = "timeout"

    unlocks = m.get("unlock_count", 0)
    redundant_unlocks = max(0, unlocks - 1) if unlocks > 0 else 0
    rur = redundant_unlocks / max(unlocks, 1)
    lmr = lever_matched / max(total_non_wait, 1)

    return {
        "family": family, "difficulty": difficulty, "condition": cond_name,
        "seed": seed, "outcome": outcome,
        "success": int(outcome == "success"),
        "death": int(outcome == "death"),
        "timeout": int(outcome == "timeout"),
        "steps": m["steps"],
        "unlocks": unlocks,
        "warns": m.get("warn_count", 0),
        "item_drops": 1 if (s.inventory and s.inventory.has_shield()) else 0,
        "redundant_unlock_rate": round(rur, 3),
        "bottleneck_epistemic_count": bn_epi,
        "bottleneck_structural_count": bn_str,
        "bottleneck_outcome_count": bn_out,
        "lever_match_rate": round(lmr, 3),
    }


if __name__ == "__main__":
    results = []
    total = sum(len(FAMILY_CONDITIONS[f]) * len(DIFFICULTIES) * len(SEEDS)
                for f in FAMILIES)
    i = 0

    for fam in FAMILIES:
        for diff in DIFFICULTIES:
            for cond in FAMILY_CONDITIONS[fam]:
                for seed in SEEDS:
                    i += 1
                    r = run_episode(fam, diff, cond, seed)
                    results.append(r)
                    if i % 60 == 0:
                        print(f"  [{i}/{total}]", file=sys.stderr)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "tpm_sweep_cross_difficulty.csv"
    keys = results[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    json_path = out_dir / "tpm_sweep_cross_difficulty.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON -> {json_path}", file=sys.stderr)

    md_path = out_dir / "tpm_sweep_summary.md"
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "s": 0, "d": 0, "t": 0, "steps": [],
                               "unlocks": [], "warns": [], "item_drops": [],
                               "rur": [], "lmr": []})
    for r in results:
        k = (r["family"], r["difficulty"], r["condition"])
        a = agg[k]
        a["n"] += 1
        a["s"] += r["success"]
        a["d"] += r["death"]
        a["t"] += r["timeout"]
        a["steps"].append(r["steps"])
        a["unlocks"].append(r["unlocks"])
        a["warns"].append(r["warns"])
        a["item_drops"].append(r["item_drops"])
        a["rur"].append(r["redundant_unlock_rate"])
        a["lmr"].append(r["lever_match_rate"])

    with open(md_path, "w") as f:
        f.write("# TPM Cross-Difficulty Sweep Results\n\n")
        f.write("| Family | Diff | Condition | SR | DR | TR | Steps | "
                "Unlocks | Warns | Items | RUR | LMR |\n")
        f.write("|--------|------|-----------|----|----|----|----|"
                "---------|-------|-------|-----|-----|\n")
        for fam in FAMILIES:
            for diff in DIFFICULTIES:
                for cond in FAMILY_CONDITIONS[fam]:
                    k = (fam, diff, cond)
                    a = agg[k]
                    n = a["n"]
                    f.write(
                        f"| {fam} | {diff} | {cond} "
                        f"| {a['s']/n:.0%} | {a['d']/n:.0%} | {a['t']/n:.0%} "
                        f"| {np.mean(a['steps']):.1f} "
                        f"| {np.mean(a['unlocks']):.1f} "
                        f"| {np.mean(a['warns']):.1f} "
                        f"| {np.mean(a['item_drops']):.1f} "
                        f"| {np.mean(a['rur']):.2f} "
                        f"| {np.mean(a['lmr']):.2f} |\n")
                f.write("| | | | | | | | | | | | |\n")
            f.write("\n")
    print(f"Summary -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
