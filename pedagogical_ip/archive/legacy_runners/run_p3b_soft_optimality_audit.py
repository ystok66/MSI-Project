"""P3-B: SOFT-Optimality Geometry Audit.

Is dose=0.5 (SOFT) ever the best action in BCICTv4's Q structure?

Line 1: Synthetic grid sweep over (p_self, risk, tempt, Δs, m) → Q(d=0), Q(d=0.5), Q(d=1)
Line 2: Real trajectory empirical optimality volumes
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from itertools import product

from src.agents.stochastic_agent_policy import AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.behavior_probes import all_probes
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4, _sigmoid
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from src.agents.behavior_bridge import (
    bridge_behavior_loss, bridge_overteach_penalty,
    EmpiricalZoneCalibrator,
)

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)


def compute_Q_per_dose(tutor, m, risk, tempt, p_self, delta_s, dvoi,
                       subtype="warn_rescue"):
    """Compute Q for each dose given state summary z."""
    has_self_ev = p_self > 0.5
    self_ev = 0.7 if has_self_ev else 0.3
    novelty = 0.0

    Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05
    Q_online_wait = 2.0 * p_self * delta_s - 1.5 * estimate_failure_if_wait(3, 2) + 2.0

    results = {}
    for dose in [0.0, 0.5, 1.0]:
        mc = tutor._predict_m(m, dose, tempt, risk, subtype, has_self_ev)
        L_now = bridge_behavior_loss(m, tutor.zones, risk, tempt, novelty, self_ev)
        L_next = bridge_behavior_loss(mc, tutor.zones, risk, tempt, novelty, self_ev)
        R = bridge_overteach_penalty(mc, tutor.zones, risk, tempt, novelty, self_ev)
        V = L_now - L_next

        p_blind = (0.7 if not has_self_ev else 0.2) * dose
        p_sd = p_self * (0.8 if subtype in ("self_discovery_needed",
                         "self_discovery_teach") else 0.4) * (1.0 - dose)
        V_full = V + tutor.lambda_sd * p_sd - tutor.lambda_dep * p_blind

        if dose == 0:
            Q = Q_online_wait + tutor.lambda_teach * V_full - tutor.lambda_over * R
        elif dose == 0.5:
            Q_soft = 0.5 * Q_online_warn + 0.5 * Q_online_wait
            Q = Q_soft + tutor.lambda_teach * V_full - tutor.lambda_over * R
        else:
            Q = Q_online_warn + tutor.lambda_teach * V_full - tutor.lambda_over * R
        results[dose] = Q

    best_dose = max(results, key=results.get)
    action = "WAIT" if best_dose == 0 else ("SOFT" if best_dose == 0.5 else "WARN")
    return results, action, best_dose


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def main():
    print("═══ P3-B: SOFT Optimality Audit ═══\n", file=sys.stderr)
    L = ["# P3-B: SOFT-Optimality Geometry Audit\n\n"]

    tutor = BCICTv4(agent_params=AP)

    # ─── Line 1: Synthetic Grid Sweep ────────────────────
    L.append("## Line 1: Synthetic Grid Sweep\n\n")
    print("Line 1: Synthetic sweep...", file=sys.stderr)

    p_self_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    risk_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tempt_grid = [0.0, 0.3, 0.5, 0.7, 1.0]
    delta_s_grid = [0.0, 0.2, 0.5, 0.8]
    dvoi_grid = [0.0, 0.1, 0.3, 0.5]
    m_configs = [
        ("low", 0.3, 0.1, 0.1),
        ("mid", 0.5, 0.3, 0.3),
        ("high", 0.7, 0.5, 0.5),
    ]

    total = 0; wait_count = 0; soft_count = 0; warn_count = 0
    soft_cases = []  # Store SOFT-optimal cases for analysis
    margins_soft = []

    for m_label, tau, nu, gg in m_configs:
        m = FactoredInternalizationState()
        m.tau = tau; m.nu = nu; m.gamma_gen = gg; m.snapshot()
        for p_self, risk, tempt, ds, dv in product(
            p_self_grid, risk_grid, tempt_grid, delta_s_grid, dvoi_grid
        ):
            Qs, action, best_dose = compute_Q_per_dose(
                tutor, m, risk, tempt, p_self, ds, dv)
            total += 1
            if action == "WAIT": wait_count += 1
            elif action == "SOFT":
                soft_count += 1
                margin = Qs[0.5] - max(Qs[0.0], Qs[1.0])
                margins_soft.append(margin)
                soft_cases.append({
                    "p_self": p_self, "risk": risk, "tempt": tempt,
                    "delta_s": ds, "dvoi": dv, "m": m_label,
                    "Q_wait": Qs[0.0], "Q_soft": Qs[0.5], "Q_warn": Qs[1.0],
                    "margin": margin,
                })
            else: warn_count += 1

    V_wait = wait_count / total
    V_soft = soft_count / total
    V_warn = warn_count / total
    L.append(f"**Total grid points: {total}**\n\n")
    L.append("| Action | Optimality Volume V_d | Count |\n")
    L.append("|--------|:---------------------:|:-----:|\n")
    L.append(f"| WAIT | {V_wait:.4f} | {wait_count} |\n")
    L.append(f"| **SOFT** | **{V_soft:.4f}** | **{soft_count}** |\n")
    L.append(f"| WARN | {V_warn:.4f} | {warn_count} |\n")
    print(f"  V_wait={V_wait:.4f} V_soft={V_soft:.4f} V_warn={V_warn:.4f}",
          file=sys.stderr)

    if soft_cases:
        L.append(f"\n### SOFT-Optimal Cases (n={len(soft_cases)})\n\n")
        L.append(f"- Mean margin: {np.mean(margins_soft):.4f}\n")
        L.append(f"- Min margin: {np.min(margins_soft):.4f}\n")
        L.append(f"- Max margin: {np.max(margins_soft):.4f}\n")

        # Characterize where SOFT wins
        L.append("\n### SOFT-Optimal Zone Characterization\n\n")
        L.append("| Feature | Mean | Med | Min | Max |\n")
        L.append("|---------|:----:|:---:|:---:|:---:|\n")
        for feat in ["p_self", "risk", "tempt", "delta_s", "dvoi"]:
            vals = [c[feat] for c in soft_cases]
            L.append("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |\n".format(
                feat, np.mean(vals), np.median(vals), np.min(vals), np.max(vals)))

        # m-config distribution
        L.append("\n### By m-config\n\n")
        for ml in ["low", "mid", "high"]:
            n_ml = sum(1 for c in soft_cases if c["m"] == ml)
            L.append(f"- {ml}: {n_ml} ({100*n_ml/max(len(soft_cases),1):.1f}%)\n")

        # Show top-5 by margin
        L.append("\n### Top-5 SOFT-Optimal by Margin\n\n")
        L.append("| p_self | risk | tempt | Δs | dvoi | m | Q_w | Q_s | Q_W | margin |\n")
        L.append("|:------:|:----:|:-----:|:--:|:----:|:-:|:---:|:---:|:---:|:------:|\n")
        for c in sorted(soft_cases, key=lambda x: -x["margin"])[:5]:
            L.append("| {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} | {:.2f} | {:.2f} | {:.2f} | {:.4f} |\n".format(
                c["p_self"], c["risk"], c["tempt"], c["delta_s"], c["dvoi"],
                c["m"], c["Q_wait"], c["Q_soft"], c["Q_warn"], c["margin"]))
    else:
        L.append("\n**SOFT is NEVER optimal in the synthetic grid.**\n")

    # ─── Line 2: Real Trajectory Empirical ───────────────
    L.append("\n## Line 2: Real Trajectory Empirical Volumes\n\n")
    print("\nLine 2: Real trajectories...", file=sys.stderr)
    L.append("| Family | n | WAIT | SOFT | WARN |\n")
    L.append("|--------|:-:|:----:|:----:|:----:|\n")
    real_total = 0; real_wait = 0; real_soft = 0; real_warn = 0
    fam_results = {}
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            rng = np.random.default_rng(sid * 1000 + abs(hash(th)) % 1000)
            m = FactoredInternalizationState(); m.snapshot()
            tutor_r = BCICTv4(agent_params=AP)
            for step in range(20):
                les = ALL_LESSONS[step % len(ALL_LESSONS)]
                ub = {p: 0.4 + 0.1 * step / 20 for p in PROBE_NAMES}
                et = generate_episode_from_lesson_v2(les, step + sid*100, th, ub, rng)
                ep, spec, gm, cfg_e, meta, sc = et
                fb, ww = apply_fix(meta, sc)
                fv = np.full_like(fb, 0.3)
                lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                for _ in range(3):
                    for r in range(gm.height):
                        for c in range(gm.width):
                            if gm.cell_types[r, c] == CellType.WALL: continue
                            z = fb[r, c]
                            lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
                lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
                ss = summarize_branch(sc.safe_cells, fb, fv, lp)
                sr = summarize_branch(sc.risky_cells, fb, fv, lp)
                lib.update("safe", ss); lib.update("risky", sr)
                scr.update(build_scorer_input(ss, lib), 1.0)
                scr.update(build_scorer_input(sr, lib), 0.0)
                action, dose, info = tutor_r.decide(sc, fb, lp, lib, scr, 2, m)
                real_total += 1
                fam_results.setdefault(les.name, {"WAIT": 0, "SOFT": 0, "WARN": 0, "n": 0})
                fam_results[les.name]["n"] += 1
                fam_results[les.name][action] += 1
                if action == "WAIT": real_wait += 1
                elif action == "SOFT": real_soft += 1
                else: real_warn += 1

                # Update m
                risk = getattr(sc, 'risk_level', 0.3)
                if dose > 0:
                    m.update_trust(warn_helpful=(risk > 0.25))
                m.update_risk(risk, 0.15); m.snapshot()

    for fam in sorted(fam_results.keys()):
        d = fam_results[fam]
        n = d["n"]
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |\n".format(
            fam, n, d["WAIT"]/n, d["SOFT"]/n, d["WARN"]/n))

    L.append(f"\n**Total: WAIT={real_wait} SOFT={real_soft} WARN={real_warn} "
             f"(n={real_total})**\n")
    L.append(f"- V_WAIT = {real_wait/real_total:.4f}\n")
    L.append(f"- V_SOFT = {real_soft/real_total:.4f}\n")
    L.append(f"- V_WARN = {real_warn/real_total:.4f}\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")
    if V_soft < 0.001 and real_soft == 0:
        L.append("> **SOFT is structurally redundant in current Q design.** "
                 "Neither synthetic sweep nor real trajectories produce any "
                 "SOFT-optimal decisions.\n")
    elif V_soft < 0.01:
        L.append("> **SOFT has marginal optimality region.** "
                 f"V_soft={V_soft:.4f} in synthetic, "
                 f"{real_soft}/{real_total} in real. "
                 "Consider simplifying to WAIT/WARN.\n")
    else:
        L.append("> **SOFT has nontrivial optimality region.** "
                 f"V_soft={V_soft:.4f}. "
                 "Design families to exploit it.\n")

    rpt = out / "p3b_soft_optimality_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
