"""Transfer Evaluation: tutor-assisted training -> no-tutor probe.

Measures whether tutor-assisted episodes improve the agent's autonomous
performance on held-out maps (no tutor).

Protocol:
  For each (family, condition):
    1. Create fresh LatentCostRiskHead
    2. Run k=0 probe (baseline, no prior training)
    3. Run 1 training episode (with tutor, seed=0), then probe
    4. Run 1 more training (seed=1), then probe  (k=2 cumulative)
    5. Run 1 more training (seed=2), then probe  (k=3 cumulative)

  Probe = 10 no-tutor episodes on held-out seeds (100-109)

Conditions:
  no_tutor            - training also has no tutor
  warning_only        - training with heuristic lane warning
  item_only           - training with ITEM_DROP only
  robot_belief_pre    - training with Phase 8 tutor (no TPM)
  robot_belief_post   - training with full TPM tutor

Outputs:
  results/transfer_eval.csv
  results/transfer_eval_summary.md
"""
import sys, json, csv, copy
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.cost_risk_model import LatentCostRiskHead
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()

FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTY = "medium"
TRAINING_SEEDS = [0, 1, 2]     # 3 training episodes
PROBE_SEEDS = list(range(100, 110))  # 10 held-out probe episodes
EXPOSURE_LEVELS = [0, 1, 2, 3]  # k = number of training episodes completed

# ── Condition definitions ─────────────────────────────────────────
def _rb_kw():
    return dict(tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5)

TRAIN_CONDITIONS = {
    "no_tutor":       dict(tutor_mode="none", warning_mode="none"),
    "warning_only":   dict(tutor_mode="none", warning_mode="lane",
                           intervention_family_mode=True, item_drop_enabled=False),
    "item_only":      {**_rb_kw(), "allowed_interventions": frozenset({"ITEM_DROP"})},
    "robot_belief_pre":  _rb_kw(),
    "robot_belief_post": _rb_kw(),
}

# Ablation flags for pre vs post TPM
TPM_FLAGS = {
    "robot_belief_pre":  dict(use_bottleneck_matching=False, use_warn_damping=False,
                              use_unlock_memory=False, use_perceptual_access=False),
    "robot_belief_post": dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True),
}

# Family-specific applicable conditions
FAMILY_CONDITIONS = {
    "fork_trap":      ["no_tutor", "warning_only", "robot_belief_pre", "robot_belief_post"],
    "hazard_belt":    ["no_tutor", "warning_only", "item_only", "robot_belief_pre", "robot_belief_post"],
    "deadline_gate":  ["no_tutor", "warning_only", "robot_belief_pre", "robot_belief_post"],
}

# ── Monkeypatch for TPM ablation ──────────────────────────────────
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


def run_single_episode(family, seed, kw, latent_pred):
    """Run one episode. Returns (metrics_dict, final_state)."""
    s = runner.reset(
        seed=seed, scenario_family=family,
        latent_mode=True, difficulty=DIFFICULTY,
        latent_predictor=latent_pred,
        **kw)
    while not s.done:
        runner.step(s)
    m = runner.get_metrics(s)
    return m, s


def run_probe_batch(family, latent_pred, probe_seeds):
    """Run no-tutor probe episodes. Returns list of result dicts."""
    results = []
    for seed in probe_seeds:
        # Make a COPY of the predictor so probes don't change weights
        probe_pred = copy.deepcopy(latent_pred)
        m, s = run_single_episode(
            family, seed,
            dict(tutor_mode="none", warning_mode="none"),
            probe_pred)

        success = m["reached_goal"] and m["survived"]
        # Autonomous efficiency: shortest_safe / actual_steps
        shortest = s.meta.shortest_safe
        ae = shortest / max(m["steps"], 1) if success else 0.0

        results.append({
            "seed": seed,
            "success": int(success),
            "death": int(not m["survived"]),
            "steps": m["steps"],
            "risky_entered": m["risky_entered"],
            "ae": round(ae, 3),
        })
    return results


def evaluate_transfer(family, cond_name):
    """Run full exposure schedule for one (family, condition).

    Returns list of {exposure_k, ...} result dicts.
    """
    global _current_flags
    _current_flags = TPM_FLAGS.get(cond_name, {})

    # Fresh learner
    latent_pred = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")

    all_results = []

    for k in EXPOSURE_LEVELS:
        if k > 0:
            # Run one training episode with tutor
            train_seed = TRAINING_SEEDS[k - 1]
            train_kw = dict(TRAIN_CONDITIONS[cond_name])
            _, _ = run_single_episode(family, train_seed, train_kw, latent_pred)

        # Probe (no tutor, held-out seeds)
        _current_flags = {}  # disable TPM flags during probe (no tutor anyway)
        probes = run_probe_batch(family, latent_pred, PROBE_SEEDS)
        _current_flags = TPM_FLAGS.get(cond_name, {})  # restore for next training

        # Aggregate probe results
        n = len(probes)
        sr = sum(p["success"] for p in probes) / n
        dr = sum(p["death"] for p in probes) / n
        mean_steps = np.mean([p["steps"] for p in probes])
        mean_ae = np.mean([p["ae"] for p in probes])
        mean_risky = np.mean([p["risky_entered"] for p in probes])

        all_results.append({
            "family": family,
            "condition": cond_name,
            "exposure_k": k,
            "n_updates_learner": latent_pred.n_updates,
            "probe_sr": round(sr, 3),
            "probe_dr": round(dr, 3),
            "probe_steps": round(mean_steps, 1),
            "probe_ae": round(mean_ae, 3),
            "probe_risky": round(mean_risky, 1),
        })

    return all_results


if __name__ == "__main__":
    all_results = []
    total_combos = sum(len(FAMILY_CONDITIONS[f]) for f in FAMILIES)
    i = 0

    for fam in FAMILIES:
        for cond in FAMILY_CONDITIONS[fam]:
            i += 1
            print(f"  [{i}/{total_combos}] {fam} / {cond}", file=sys.stderr)
            results = evaluate_transfer(fam, cond)
            all_results.extend(results)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── Write CSV ─────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "transfer_eval.csv"
    keys = all_results[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV -> {csv_path}", file=sys.stderr)

    # ── Write summary markdown ────────────────────────────────────
    md_path = out_dir / "transfer_eval_summary.md"
    with open(md_path, "w") as f:
        f.write("# Transfer Evaluation Results\n\n")
        f.write("Protocol: k training episodes (with tutor) -> 10 no-tutor probes "
                "(held-out seeds 100-109)\n\n")

        for fam in FAMILIES:
            f.write(f"## {fam}\n\n")
            f.write("| Condition | k=0 SR | k=1 SR | k=2 SR | k=3 SR | "
                    "k=0 AE | k=3 AE | LG(3) |\n")
            f.write("|-----------|--------|--------|--------|--------|"
                    "--------|--------|-------|\n")

            # Get no_tutor k=0 as baseline
            baseline_sr = 0.0
            for r in all_results:
                if r["family"] == fam and r["condition"] == "no_tutor" and r["exposure_k"] == 0:
                    baseline_sr = r["probe_sr"]

            for cond in FAMILY_CONDITIONS[fam]:
                srs = {}
                aes = {}
                for r in all_results:
                    if r["family"] == fam and r["condition"] == cond:
                        srs[r["exposure_k"]] = r["probe_sr"]
                        aes[r["exposure_k"]] = r["probe_ae"]

                lg3 = srs.get(3, 0) - baseline_sr
                sign = "+" if lg3 >= 0 else ""
                f.write(
                    f"| {cond} "
                    f"| {srs.get(0, 0):.0%} | {srs.get(1, 0):.0%} "
                    f"| {srs.get(2, 0):.0%} | {srs.get(3, 0):.0%} "
                    f"| {aes.get(0, 0):.2f} | {aes.get(3, 0):.2f} "
                    f"| {sign}{lg3:.0%} |\n")
            f.write("\n")

        # ── Pedagogical Efficiency table ──
        f.write("## Pedagogical Efficiency\n\n")
        f.write("PE = LG / OHG where OHG = assisted_SR - no_tutor_SR, "
                "LG = probe_SR(k=3) - baseline\n\n")
        f.write("| Family | Condition | OHG | LG(3) | PE |\n")
        f.write("|--------|-----------|-----|-------|----|\n")

        # Load assisted SR from existing sweep data (if available)
        sweep_path = out_dir / "tpm_sweep_cross_difficulty.csv"
        assisted_sr = {}
        if sweep_path.exists():
            import csv as csv_mod
            with open(sweep_path) as sf:
                reader = csv_mod.DictReader(sf)
                for row in reader:
                    if row["difficulty"] == DIFFICULTY:
                        k2 = (row["family"], row["condition"])
                        if k2 not in assisted_sr:
                            assisted_sr[k2] = {"s": 0, "n": 0}
                        assisted_sr[k2]["s"] += int(row["success"])
                        assisted_sr[k2]["n"] += 1

        # Map transfer condition names to sweep condition names
        SWEEP_MAP = {
            "no_tutor": "no_tutor",
            "warning_only": "warning_only",
            "item_only": "item_only",
            "robot_belief_pre": "robot_belief_pre",
            "robot_belief_post": "robot_belief_post",
        }

        for fam in FAMILIES:
            baseline_sr = 0.0
            for r in all_results:
                if r["family"] == fam and r["condition"] == "no_tutor" and r["exposure_k"] == 0:
                    baseline_sr = r["probe_sr"]

            no_tutor_asr_key = (fam, "no_tutor")
            no_tutor_asr = (assisted_sr[no_tutor_asr_key]["s"] / assisted_sr[no_tutor_asr_key]["n"]
                            if no_tutor_asr_key in assisted_sr else baseline_sr)

            for cond in FAMILY_CONDITIONS[fam]:
                # LG
                lg = 0.0
                for r in all_results:
                    if r["family"] == fam and r["condition"] == cond and r["exposure_k"] == 3:
                        lg = r["probe_sr"] - baseline_sr

                # OHG from sweep data
                sweep_key = (fam, SWEEP_MAP.get(cond, cond))
                if sweep_key in assisted_sr:
                    asr = assisted_sr[sweep_key]["s"] / assisted_sr[sweep_key]["n"]
                    ohg = asr - no_tutor_asr
                else:
                    ohg = 0.0

                pe = lg / max(ohg, 0.01) if ohg > 0.01 else (lg if lg > 0 else 0.0)
                f.write(f"| {fam} | {cond} | {ohg:.0%} | {lg:+.0%} | {pe:.2f} |\n")
            f.write("\n")

    print(f"Summary -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
