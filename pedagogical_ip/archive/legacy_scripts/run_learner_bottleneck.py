"""Step D: Learner Bottleneck Micro-Suite.

D1 — Exposure scaling:  k ∈ {1, 3, 10, 30} training episodes → probe
D2 — Transfer gradient: same-map, same-family, held-out-family
D3 — Oracle upper bound: direct supervised training (no episode needed)

Outputs:
  results/d1_exposure_scaling.csv
  results/d2_transfer_gradient.csv
  results/d3_oracle_upperbound.csv
  results/learner_bottleneck_report.md
"""
import sys, csv, copy
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.scenario_families import generate_scenario
from src.agents.cost_risk_model import LatentCostRiskHead
from src.teachers.intervention_policy import InterventionConfig
from src.teachers import intervention_policy as ip
from src.envs import lattice_v2_runner as runner_mod

runner = LatticeV2Runner()

# ── Constants ─────────────────────────────────────────────────────
FAMILIES = ["fork_trap", "hazard_belt", "deadline_gate"]
DIFFICULTY = "medium"
PROBE_SEEDS = list(range(100, 110))  # 10 held-out seeds for probing

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

# ── Helpers ──────────────────────────────────────────────────────
def run_episode(family, seed, kw, latent_pred, difficulty=DIFFICULTY):
    s = runner.reset(seed=seed, scenario_family=family,
                     latent_mode=True, difficulty=difficulty,
                     latent_predictor=latent_pred, **kw)
    while not s.done:
        runner.step(s)
    m = runner.get_metrics(s)
    return m, s


def probe_sr(family, latent_pred, seeds=PROBE_SEEDS, difficulty=DIFFICULTY):
    """Run no-tutor probes, return success rate."""
    successes = 0
    for seed in seeds:
        pred_copy = copy.deepcopy(latent_pred)
        m, _ = run_episode(family, seed,
                           dict(tutor_mode="none", warning_mode="none"),
                           pred_copy, difficulty=difficulty)
        if m["reached_goal"] and m["survived"]:
            successes += 1
    return successes / len(seeds)


def train_kw_post():
    return dict(tutor_mode="none", robot_belief_mode=True,
                intervention_family_mode=True, item_drop_enabled=True,
                prefix_horizon=5)


# ═══════════════════════════════════════════════════════════════════
# D1: Exposure Scaling
# ═══════════════════════════════════════════════════════════════════
def run_d1():
    """k ∈ {1, 3, 10, 30}: how much training until transfer appears?"""
    global _current_flags
    _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                          use_unlock_memory=True, use_perceptual_access=True)

    K_VALUES = [0, 1, 3, 10, 30]
    results = []

    for fam in FAMILIES:
        print(f"  D1: {fam}", file=sys.stderr)
        for cond in ["no_tutor", "robot_belief_post"]:
            # Fresh learner per condition
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")

            kw = (dict(tutor_mode="none", warning_mode="none")
                  if cond == "no_tutor" else train_kw_post())

            k_done = 0
            for k_target in K_VALUES:
                # Train from k_done to k_target
                while k_done < k_target:
                    seed = k_done  # unique seed per episode
                    run_episode(fam, seed, kw, lp)
                    k_done += 1

                # Probe
                _current_flags = {}  # disable TPM for probe
                sr = probe_sr(fam, lp)
                _current_flags = dict(use_bottleneck_matching=True,
                                      use_warn_damping=True,
                                      use_unlock_memory=True,
                                      use_perceptual_access=True)

                results.append({
                    "family": fam, "condition": cond,
                    "exposure_k": k_target,
                    "n_updates": lp.n_updates,
                    "probe_sr": round(sr, 3),
                    "w_norm": round(float(np.linalg.norm(
                        list(lp.cost_head.w) + [lp.cost_head.b]
                        + list(lp.risk_head.w) + [lp.risk_head.b])), 4),
                })

    _current_flags = {}
    return results


# ═══════════════════════════════════════════════════════════════════
# D2: Transfer Gradient
# ═══════════════════════════════════════════════════════════════════
def run_d2():
    """3-tier generalization radius:
      T1: same-map new-seed (train s=0, probe s=0 with fresh latent)
      T2: same-family new-map (train s∈{0..4}, probe s∈{100..109})
      T3: held-out-family (train on fork_trap, probe on hazard_belt etc.)
    """
    global _current_flags
    _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                          use_unlock_memory=True, use_perceptual_access=True)
    results = []
    K_TRAIN = 5  # training episodes

    for fam in FAMILIES:
        print(f"  D2: {fam}", file=sys.stderr)

        # ── T1: Same-map ──
        # Train on seed=0, probe on seed=0 (different run)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        # Baseline probe
        _current_flags = {}
        sr_before = probe_sr(fam, lp, seeds=[0, 1, 2, 3, 4])
        _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True)

        # Train on seed 0 only, repeat K_TRAIN times
        for _ in range(K_TRAIN):
            run_episode(fam, 0, train_kw_post(), lp)

        _current_flags = {}
        sr_same_map = probe_sr(fam, lp, seeds=[0, 1, 2, 3, 4])

        results.append({"family": fam, "tier": "T1_same_map",
                        "sr_before": round(sr_before, 3),
                        "sr_after": round(sr_same_map, 3),
                        "delta": round(sr_same_map - sr_before, 3)})

        # ── T2: Same-family new-map ──
        lp2 = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        _current_flags = dict(use_bottleneck_matching=True, use_warn_damping=True,
                              use_unlock_memory=True, use_perceptual_access=True)
        for s in range(K_TRAIN):
            run_episode(fam, s, train_kw_post(), lp2)

        _current_flags = {}
        sr_new_map = probe_sr(fam, lp2, seeds=PROBE_SEEDS)

        results.append({"family": fam, "tier": "T2_same_family",
                        "sr_before": round(sr_before, 3),
                        "sr_after": round(sr_new_map, 3),
                        "delta": round(sr_new_map - sr_before, 3)})

        # ── T3: Held-out family ──
        held_out = [f for f in FAMILIES if f != fam]
        for hf in held_out:
            lp3 = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            _current_flags = dict(use_bottleneck_matching=True,
                                  use_warn_damping=True,
                                  use_unlock_memory=True,
                                  use_perceptual_access=True)
            for s in range(K_TRAIN):
                run_episode(fam, s, train_kw_post(), lp3)

            _current_flags = {}
            sr_held = probe_sr(hf, lp3, seeds=PROBE_SEEDS)

            results.append({"family": f"{fam}→{hf}", "tier": "T3_held_out",
                            "sr_before": round(sr_before, 3),
                            "sr_after": round(sr_held, 3),
                            "delta": round(sr_held - sr_before, 3)})

    _current_flags = {}
    return results


# ═══════════════════════════════════════════════════════════════════
# D3: Oracle Supervision Upper Bound
# ═══════════════════════════════════════════════════════════════════
def run_d3():
    """Direct supervised training: give learner perfect labels.
    Question: can the linear head THEORETICALLY learn the mapping?
    """
    results = []
    N_SAMPLES = [10, 50, 200, 1000]

    for fam in FAMILIES:
        print(f"  D3: {fam}", file=sys.stderr)
        for n in N_SAMPLES:
            # Generate training data from the ground truth
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")

            # Sample cells from multiple seeds
            for seed in range(min(n, 20)):
                gm, cfg, meta, sc = generate_scenario(
                    fam, seed, DIFFICULTY, latent_mode=True)
                ww = meta.world_weights
                if ww is None:
                    continue
                feats = meta.cell_features
                H, W = gm.height, gm.width

                cells_per_seed = max(n // 20, 1)
                rng = np.random.default_rng(seed + 9999)
                for _ in range(cells_per_seed):
                    r = rng.integers(0, H)
                    c = rng.integers(0, W)
                    if gm.cell_types[r, c] == 0:  # WALL
                        continue
                    z = feats[r, c]
                    true_c = ww.true_cost(z)
                    true_r = ww.true_risk(z)
                    lp.update_from_outcome(z, true_c, true_r)

            # Probe on held-out maps
            sr = probe_sr(fam, lp)

            # Also compute prediction accuracy on a held-out map
            gm_test, _, meta_test, _ = generate_scenario(
                fam, 999, DIFFICULTY, latent_mode=True)
            ww_test = meta_test.world_weights
            feats_test = meta_test.cell_features
            mse_c, mse_r = 0.0, 0.0
            n_test = 0
            if ww_test is not None:
                for r in range(gm_test.height):
                    for c in range(gm_test.width):
                        if gm_test.cell_types[r, c] == 0:
                            continue
                        z = feats_test[r, c]
                        pred_c = lp.predict_cost(z)
                        pred_r = lp.predict_risk(z)
                        true_c = ww_test.true_cost(z)
                        true_r = ww_test.true_risk(z)
                        mse_c += (pred_c - true_c) ** 2
                        mse_r += (pred_r - true_r) ** 2
                        n_test += 1
                if n_test > 0:
                    mse_c /= n_test
                    mse_r /= n_test

            results.append({
                "family": fam,
                "n_train_labels": n,
                "n_updates": lp.n_updates,
                "probe_sr": round(sr, 3),
                "mse_cost": round(mse_c, 6),
                "mse_risk": round(mse_r, 6),
                "w_norm": round(float(np.linalg.norm(
                    list(lp.cost_head.w) + [lp.cost_head.b]
                    + list(lp.risk_head.w) + [lp.risk_head.b])), 4),
            })

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # D1
    print("=== D1: Exposure Scaling ===", file=sys.stderr)
    d1 = run_d1()
    with open(out_dir / "d1_exposure_scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d1[0].keys())
        w.writeheader()
        w.writerows(d1)

    # D2
    print("=== D2: Transfer Gradient ===", file=sys.stderr)
    d2 = run_d2()
    with open(out_dir / "d2_transfer_gradient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d2[0].keys())
        w.writeheader()
        w.writerows(d2)

    # D3
    print("=== D3: Oracle Upper Bound ===", file=sys.stderr)
    d3 = run_d3()
    with open(out_dir / "d3_oracle_upperbound.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d3[0].keys())
        w.writeheader()
        w.writerows(d3)

    # Restore
    ip.score_interventions = _orig_score
    runner_mod.score_interventions = _orig_score

    # ── Report ────────────────────────────────────────────────────
    md_path = out_dir / "learner_bottleneck_report.md"
    with open(md_path, "w") as f:
        f.write("# Learner Bottleneck Micro-Suite Report\n\n")

        # D1 table
        f.write("## D1: Exposure Scaling\n\n")
        f.write("Does more training produce transfer?\n\n")
        f.write("| Family | Condition | k=0 | k=1 | k=3 | k=10 | k=30 | Trend |\n")
        f.write("|--------|-----------|-----|-----|-----|------|------|-------|\n")
        d1_map = {}
        for r in d1:
            key = (r["family"], r["condition"])
            if key not in d1_map:
                d1_map[key] = {}
            d1_map[key][r["exposure_k"]] = r["probe_sr"]

        for fam in FAMILIES:
            for cond in ["no_tutor", "robot_belief_post"]:
                srs = d1_map.get((fam, cond), {})
                vals = [srs.get(k, 0) for k in [0, 1, 3, 10, 30]]
                trend = "↑" if vals[-1] > vals[0] + 0.05 else ("→" if abs(vals[-1] - vals[0]) <= 0.05 else "↓")
                f.write(f"| {fam} | {cond} | "
                        + " | ".join(f"{v:.0%}" for v in vals)
                        + f" | {trend} |\n")
            f.write("| | | | | | | | |\n")

        # D2 table
        f.write("\n## D2: Transfer Gradient\n\n")
        f.write("Generalization radius: same-map → same-family → held-out\n\n")
        f.write("| Source Family | Tier | SR Before | SR After | Δ |\n")
        f.write("|-------------|------|-----------|----------|---|\n")
        for r in d2:
            sign = "+" if r["delta"] >= 0 else ""
            f.write(f"| {r['family']} | {r['tier']} | {r['sr_before']:.0%} "
                    f"| {r['sr_after']:.0%} | {sign}{r['delta']:.0%} |\n")

        # D3 table
        f.write("\n## D3: Oracle Supervision Upper Bound\n\n")
        f.write("Can the linear head learn the mapping with perfect labels?\n\n")
        f.write("| Family | n_labels | Probe SR | MSE_cost | MSE_risk |\n")
        f.write("|--------|----------|----------|----------|----------|\n")
        for r in d3:
            f.write(f"| {r['family']} | {r['n_train_labels']} "
                    f"| {r['probe_sr']:.0%} | {r['mse_cost']:.6f} "
                    f"| {r['mse_risk']:.6f} |\n")

        # Interpretation
        f.write("\n## Interpretation\n\n")
        f.write("### Key Questions Answered\n\n")
        f.write("1. **Does more exposure help?** (D1)\n")
        f.write("   - If SR flat at all k → learner capacity issue\n")
        f.write("   - If SR rises slowly → learning rate / sample efficiency issue\n\n")
        f.write("2. **Where does generalization break?** (D2)\n")
        f.write("   - T1 positive, T2/T3 zero → map-specific learning\n")
        f.write("   - T1+T2 positive, T3 zero → family-specific learning\n")
        f.write("   - All zero → no generalization at all\n\n")
        f.write("3. **Is the capacity sufficient?** (D3)\n")
        f.write("   - High SR with oracle → capacity OK, supervision is the bottleneck\n")
        f.write("   - Low SR even with oracle → linear head can't represent the mapping\n")

    print(f"Report -> {md_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)
