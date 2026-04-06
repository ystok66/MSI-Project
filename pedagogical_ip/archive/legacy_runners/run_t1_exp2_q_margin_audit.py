"""T1 Exp-2: Q-Margin Audit — Near-Tie Over-Warn Diagnosis.

For each step, records full Q decomposition (raw + weighted) and checks:
  1. |ΔQ| binning to see where over-warns concentrate
  2. NearTieCoverage(ε_Q) — what fraction of over-warns are near-tie
  3. Which Q component (online / V_full / R_over) drives near-tie flips

Focus families: blind_activation_corridor, warn_symmetric_rescue, tic_rescue_heavy
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)
FOCUS_FAMILIES = {"blind_activation_corridor", "warn_symmetric_rescue",
                  "tic_rescue_heavy"}


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_audit_session(theta, seed, n_teach=20):
    """Run one session, return per-step Q-margin records."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    records = []
    for step in range(n_teach):
        les = ALL_LESSONS[step % len(ALL_LESSONS)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed*100, theta, ub, rng)
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

        # Oracle tutor (reads true m)
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, info_o = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer tutor (reads observer estimate)
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, info_i = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

        # Agent simulation
        action, dose = act_o, dose_o
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b)
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a)
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0
        self_disc = correct and not warned and p_self > 0.5
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if p_self < 0.5: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()
        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        # Collect Q-margin data from INFER tutor
        qd = info_i.get("q_detail", {})
        active = (act_o != "WAIT") or (act_i != "WAIT")
        is_owr = (act_o == "WAIT" and act_i == "WARN")
        is_under = (act_o == "WARN" and act_i == "WAIT")
        records.append({
            "step": step, "family": les.name, "subtype": ep.subtype,
            "theta": theta,
            "act_oracle": act_o, "act_infer": act_i,
            "active": active, "is_owr": is_owr, "is_under": is_under,
            "diverge": (act_o != act_i),
            "delta_Q": qd.get("delta_Q", 0),
            "delta_Q_online": qd.get("delta_Q_online", 0),
            "delta_V_full_raw": qd.get("delta_V_full_raw", 0),
            "delta_V_full_weighted": qd.get("delta_V_full_weighted", 0),
            "delta_R_over_raw": qd.get("delta_R_over_raw", 0),
            "delta_R_over_weighted": qd.get("delta_R_over_weighted", 0),
            "p_self": qd.get("p_self", 0),
            "p_fail": qd.get("p_fail", 0),
            "delta_s": qd.get("delta_s", 0),
            "dvoi": qd.get("dvoi", 0),
            "tempt": tempt, "risk": risk,
            "dc_minus_dr": dc - dr,
        })
    return records


def main():
    print("═══ T1 Exp-2: Q-Margin Audit ═══\n", file=sys.stderr)
    L = ["# T1 Exp-2: Q-Margin Audit\n\n"]

    # Collect all data
    print("Collecting data...", file=sys.stderr)
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_audit_session(th, sid))
            if sid % 5 == 4:
                print(f"  {th} seed {sid+1}/{NS}", file=sys.stderr)

    n_total = len(all_recs)
    n_owr = sum(1 for r in all_recs if r["is_owr"])
    n_active = sum(1 for r in all_recs if r["active"])
    n_diverge = sum(1 for r in all_recs if r["diverge"])

    L.append(f"> Total steps: {n_total} | Active: {n_active} | "
             f"Diverge: {n_diverge} | Over-warn: {n_owr}\n\n")
    print(f"  Total={n_total} Active={n_active} Diverge={n_diverge} OWR={n_owr}",
          file=sys.stderr)

    # ═══ Table 1: |ΔQ| Binning ═══
    L.append("## Table 1: |ΔQ| Distribution — All Steps\n\n")
    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, float("inf"))]
    L.append("| |ΔQ| Range | Steps | Over-Warn | OWR% | "
             "Oracle=WARN | Disagree% |\n")
    L.append("|:----------:|:-----:|:---------:|:----:|"
             ":-----------:|:---------:|\n")
    for lo, hi in bins:
        in_bin = [r for r in all_recs
                  if lo <= abs(r["delta_Q"]) < hi]
        n_bin = len(in_bin)
        owr_bin = sum(1 for r in in_bin if r["is_owr"])
        oracle_warn_bin = sum(1 for r in in_bin if r["act_oracle"] == "WARN")
        disagree_bin = sum(1 for r in in_bin if r["diverge"])
        hi_str = f"{hi:.2f}" if hi < 100 else "∞"
        L.append("| [{:.2f}, {}) | {} | {} | {:.1f}% | {} | {:.1f}% |\n".format(
            lo, hi_str, n_bin, owr_bin,
            100 * owr_bin / max(n_bin, 1),
            oracle_warn_bin,
            100 * disagree_bin / max(n_bin, 1)))

    # ═══ Table 2: NearTieCoverage ═══
    L.append("\n## Table 2: NearTieCoverage(ε_Q)\n\n")
    L.append("| ε_Q | OWR in near-tie | Total OWR | Coverage |\n")
    L.append("|:---:|:---------------:|:---------:|:--------:|\n")
    eps_list = [0.02, 0.05, 0.10, 0.15, 0.20, 0.50]
    for eps in eps_list:
        owr_near = sum(1 for r in all_recs
                       if r["is_owr"] and abs(r["delta_Q"]) < eps)
        cov = owr_near / max(n_owr, 1)
        L.append(f"| {eps:.2f} | {owr_near} | {n_owr} | {cov:.1%} |\n")

    # ═══ Table 3: Per-Family Over-Warn ═══
    L.append("\n## Table 3: Per-Family Over-Warn Distribution\n\n")
    L.append("| Family | Steps | OWR Count | OWR% | Mean |ΔQ| | "
             "Median |ΔQ| |\n")
    L.append("|--------|:-----:|:---------:|:----:|:----------:|"
             ":-----------:|\n")

    fam_groups = defaultdict(list)
    for r in all_recs:
        fam_groups[r["family"]].append(r)

    for fam in sorted(fam_groups.keys()):
        fr = fam_groups[fam]
        n_f = len(fr)
        owr_f = sum(1 for r in fr if r["is_owr"])
        dqs = [abs(r["delta_Q"]) for r in fr]
        mark = " **←**" if fam in FOCUS_FAMILIES else ""
        L.append("| {}{} | {} | {} | {:.1f}% | {:.4f} | {:.4f} |\n".format(
            fam, mark, n_f, owr_f,
            100 * owr_f / max(n_f, 1),
            np.mean(dqs), np.median(dqs)))

    # ═══ Table 4: Q Component Decomposition at Over-Warn Points ═══
    L.append("\n## Table 4: Q Decomposition at Over-Warn Points\n\n")
    owr_recs = [r for r in all_recs if r["is_owr"]]
    if owr_recs:
        L.append("| Metric | Mean | Median | Min | Max |\n")
        L.append("|--------|:----:|:------:|:---:|:---:|\n")
        for key in ["delta_Q", "delta_Q_online",
                     "delta_V_full_raw", "delta_V_full_weighted",
                     "delta_R_over_raw", "delta_R_over_weighted",
                     "p_self", "p_fail", "delta_s", "dvoi",
                     "tempt", "risk", "dc_minus_dr"]:
            vals = [r[key] for r in owr_recs]
            L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                key, np.mean(vals), np.median(vals),
                np.min(vals), np.max(vals)))

        # Which component contributes most to positive delta_Q at OWR points?
        L.append("\n### Dominant Flip Component at Over-Warn Points\n\n")
        L.append("For each over-warn step, which component makes ΔQ positive?\n\n")
        dominant = {"online": 0, "V_full": 0, "R_over": 0}
        for r in owr_recs:
            parts = {
                "online": r["delta_Q_online"],
                "V_full": r["delta_V_full_weighted"],
                "R_over": -r["delta_R_over_weighted"],  # negative sign in Q formula
            }
            dom = max(parts, key=lambda k: parts[k])
            dominant[dom] += 1
        L.append("| Component | Count | Fraction |\n")
        L.append("|-----------|:-----:|:--------:|\n")
        for comp, cnt in dominant.items():
            L.append(f"| {comp} | {cnt} | {cnt/max(len(owr_recs),1):.1%} |\n")
    else:
        L.append("> No over-warn events detected.\n")

    # ═══ Table 5: Per-OWR-Step Detail (if few) ═══
    if 0 < len(owr_recs) <= 30:
        L.append("\n## Table 5: Per-Step Over-Warn Detail\n\n")
        L.append("| Family | θ | ΔQ | ΔQ_online | ΔV_full_w | ΔR_over_w | "
                 "p_self | tempt | risk | dc-dr |\n")
        L.append("|--------|:-:|:--:|:---------:|:---------:|:----------:|"
                 ":-----:|:-----:|:----:|:----:|\n")
        for r in owr_recs:
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
                     "{:.2f} | {:.2f} | {:.2f} | {} |\n".format(
                r["family"], r["theta"],
                r["delta_Q"], r["delta_Q_online"],
                r["delta_V_full_weighted"], r["delta_R_over_weighted"],
                r["p_self"], r["tempt"], r["risk"], r["dc_minus_dr"]))

    # ═══ Summary ═══
    L.append("\n## Summary\n\n")
    if n_owr > 0:
        cov_010 = sum(1 for r in all_recs
                      if r["is_owr"] and abs(r["delta_Q"]) < 0.10) / n_owr
        L.append(f"> Over-warn count: {n_owr}/{n_total} ({100*n_owr/n_total:.2f}%)\n")
        L.append(f"> NearTieCoverage(ε=0.10): {cov_010:.1%}\n")
        if cov_010 > 0.7:
            L.append("> **Conclusion: Over-warn is predominantly a near-tie "
                     "Q-margin phenomenon. Dead-zone fix is justified.**\n")
        else:
            L.append("> **Conclusion: Over-warn is NOT purely near-tie. "
                     "Dead-zone alone may be insufficient.**\n")
    else:
        L.append("> No over-warn events. Canonical is clean.\n")

    rpt = out / "t1_exp2_q_margin_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
