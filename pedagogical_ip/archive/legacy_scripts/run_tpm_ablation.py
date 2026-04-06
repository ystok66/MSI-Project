"""TPM Ablation Study.

Runs 6 ablation conditions x 3 families x 20 seeds on medium difficulty.

Conditions:
  full_tpm              - all TPM features ON (baseline)
  no_bottleneck_match   - use_bottleneck_matching=False
  no_warn_damping       - use_warn_damping=False
  no_unlock_memory      - use_unlock_memory=False
  no_perceptual_access  - use_perceptual_access=False
  cf_only               - all TPM OFF (counterfactual-only Phase 8 equiv)

Outputs:
  results/tpm_ablation.csv
  results/tpm_ablation_summary.md
"""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()
SEEDS = list(range(20))
FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTY = "medium"

LEVER_MATCH = {"WARN": "epistemic", "UNLOCK": "structural", "ITEM_DROP": "outcome"}

ABLATIONS = {
    "full_tpm": dict(
        use_bottleneck_matching=True, use_warn_damping=True,
        use_unlock_memory=True, use_perceptual_access=True),
    "no_bottleneck_match": dict(
        use_bottleneck_matching=False, use_warn_damping=True,
        use_unlock_memory=True, use_perceptual_access=True),
    "no_warn_damping": dict(
        use_bottleneck_matching=True, use_warn_damping=False,
        use_unlock_memory=True, use_perceptual_access=True),
    "no_unlock_memory": dict(
        use_bottleneck_matching=True, use_warn_damping=True,
        use_unlock_memory=False, use_perceptual_access=True),
    "no_perceptual_access": dict(
        use_bottleneck_matching=True, use_warn_damping=True,
        use_unlock_memory=True, use_perceptual_access=False),
    "cf_only": dict(
        use_bottleneck_matching=False, use_warn_damping=False,
        use_unlock_memory=False, use_perceptual_access=False),
}

# Monkeypatch: patch BOTH module references
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


def run_ablation_episode(family, ablation_name, seed):
    global _current_flags
    _current_flags = ABLATIONS[ablation_name]

    s = runner.reset(
        seed=seed, scenario_family=family, latent_mode=True,
        difficulty=DIFFICULTY, tutor_mode="none", robot_belief_mode=True,
        intervention_family_mode=True, item_drop_enabled=True, prefix_horizon=5)

    bn_epi = bn_str = bn_out = 0
    total_non_wait = 0
    lever_matched = 0
    false_warn_outcome = 0
    total_warns = 0

    while not s.done:
        runner.step(s)

        if s.last_intervention is not None:
            action = s.last_intervention.action
            bn = s.last_intervention.bottleneck
            if bn is not None:
                dom = bn.dominant
                if dom == "epistemic": bn_epi += 1
                elif dom == "structural": bn_str += 1
                elif dom == "outcome": bn_out += 1
                if action != "WAIT":
                    total_non_wait += 1
                    if LEVER_MATCH.get(action) == dom:
                        lever_matched += 1
                if action == "WARN" and bn.outcome > bn.epistemic:
                    false_warn_outcome += 1
            if action == "WARN":
                total_warns += 1

    m = runner.get_metrics(s)
    if m["reached_goal"] and m["survived"]:
        outcome = "success"
    elif not m["survived"]:
        outcome = "death"
    else:
        outcome = "timeout"

    unlocks = m.get("unlock_count", 0)
    repeat_unlocks = max(0, unlocks - 1) if unlocks > 0 else 0

    return {
        "family": family, "ablation": ablation_name, "seed": seed,
        "outcome": outcome,
        "success": int(outcome == "success"),
        "death": int(outcome == "death"),
        "timeout": int(outcome == "timeout"),
        "steps": m["steps"],
        "unlocks": unlocks,
        "warns": m.get("warn_count", 0),
        "repeat_unlocks": repeat_unlocks,
        "lever_match_rate": round(lever_matched / max(total_non_wait, 1), 3),
        "false_warn_outcome": false_warn_outcome,
        "bn_epistemic": bn_epi,
        "bn_structural": bn_str,
        "bn_outcome": bn_out,
    }


if __name__ == "__main__":
    results = []
    total = len(FAMILIES) * len(ABLATIONS) * len(SEEDS)
    i = 0

    for fam in FAMILIES:
        for abl_name in ABLATIONS:
            _current_flags = ABLATIONS[abl_name]
            for seed in SEEDS:
                i += 1
                r = run_ablation_episode(fam, abl_name, seed)
                results.append(r)
                if i % 60 == 0:
                    print(f"  [{i}/{total}]", file=sys.stderr)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "tpm_ablation.csv"
    keys = results[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    md_path = out_dir / "tpm_ablation_summary.md"
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "s": 0, "d": 0, "steps": [],
                               "repeat_unlocks": [], "lmr": [],
                               "false_warn": []})
    for r in results:
        k = (r["family"], r["ablation"])
        a = agg[k]
        a["n"] += 1
        a["s"] += r["success"]
        a["d"] += r["death"]
        a["steps"].append(r["steps"])
        a["repeat_unlocks"].append(r["repeat_unlocks"])
        a["lmr"].append(r["lever_match_rate"])
        a["false_warn"].append(r["false_warn_outcome"])

    with open(md_path, "w") as f:
        f.write("# TPM Ablation Results (medium difficulty, 20 seeds)\n\n")
        for fam in FAMILIES:
            f.write(f"## {fam}\n\n")
            f.write("| Ablation | SR | DR | Steps | RepeatUL | LMR | FalseWarnOC |\n")
            f.write("|----------|----|----|-------|----------|-----|-------------|\n")
            for abl_name in ABLATIONS:
                k = (fam, abl_name)
                a = agg[k]
                n = a["n"]
                f.write(
                    f"| {abl_name} "
                    f"| {a['s']/n:.0%} | {a['d']/n:.0%} "
                    f"| {np.mean(a['steps']):.1f} "
                    f"| {np.mean(a['repeat_unlocks']):.1f} "
                    f"| {np.mean(a['lmr']):.2f} "
                    f"| {np.mean(a['false_warn']):.1f} |\n")
            f.write("\n")

        f.write("## Delta SR from full_tpm\n\n")
        f.write("| Ablation | fork_trap | hazard_belt | deadline_gate |\n")
        f.write("|----------|-----------|-------------|---------------|\n")
        for abl_name in ABLATIONS:
            row = f"| {abl_name} "
            for fam in FAMILIES:
                full_sr = agg[(fam, "full_tpm")]["s"] / agg[(fam, "full_tpm")]["n"]
                abl_sr = agg[(fam, abl_name)]["s"] / agg[(fam, abl_name)]["n"]
                delta = abl_sr - full_sr
                sign = "+" if delta >= 0 else ""
                row += f"| {sign}{delta:.0%} "
            row += "|\n"
            f.write(row)

    print(f"Summary -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
