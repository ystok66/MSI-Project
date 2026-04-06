"""P3-C: Exact-Q Geometry Audit + Action-Space Ablation.

Line 1: Exact-Q per-dose from real scenarios (full tutor path)
Line 2: Action-space ablation (3-action vs 2-action)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from scipy import stats as sp_stats

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
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
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob
from src.agents.behavior_bridge import bridge_behavior_loss, bridge_overteach_penalty

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def extract_per_dose_Q(tutor, sc, fb, lp, lib, scr, obs, m):
    """Extract Q for each dose by running tutor internals.

    We run decide() three times with use_dose=True to get the full Q path,
    but we need per-dose Q values. Since decide() only returns best Q,
    we'll modify state to force each dose and extract Q values.
    """
    from src.envs.observation_mask import make_observation_mask
    from src.metrics.self_discovery import estimate_failure_if_wait
    from src.teachers.internalization_control_tutor_v4 import _sigmoid

    fv = np.full_like(fb, 0.3)
    dc = getattr(sc, 'commit_depth', obs + 1)
    dr = getattr(sc, 'reveal_depth', 3)
    p_self = estimate_self_discovery_prob(dc, dr)
    p_fail = estimate_failure_if_wait(dc, dr)

    fork = sc.fork_cell
    mask_a = make_observation_mask(sc.branch_a_cells, fork, obs)
    mask_b = make_observation_mask(sc.branch_b_cells, fork, obs)
    vis_a = [c for c, mm in zip(sc.branch_a_cells, mask_a) if mm > 0.5]
    vis_b = [c for c, mm in zip(sc.branch_b_cells, mask_b) if mm > 0.5]

    sa = summarize_branch(vis_a, fb, fv, lp)
    sb = summarize_branch(vis_b, fb, fv, lp)
    sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
    sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
    delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
    dvoi = max(_sigmoid(abs(sa2[0] - sb2[0])) - _sigmoid(abs(sa[0] - sb[0])), 0)

    tempt = getattr(sc, 'temptation_strength', 0.0)
    risk = getattr(sc, 'risk_level', 0.3)
    subtype = getattr(sc, 'episode_subtype', '')
    novelty = 0.3 if subtype in ("beneficial_novelty",) else 0.0
    has_self_ev = (obs >= dc - 1) or p_self > 0.5
    self_ev = 0.7 if has_self_ev else 0.3
    z = tutor.zones

    Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05
    Q_online_wait = 2.0 * p_self * delta_s - 1.5 * p_fail + 2.0

    Q_per_dose = {}
    for dose in [0.0, 0.5, 1.0]:
        mc = tutor._predict_m(m, dose, tempt, risk, subtype, has_self_ev)
        L_now = bridge_behavior_loss(m, z, risk, tempt, novelty, self_ev)
        L_next = bridge_behavior_loss(mc, z, risk, tempt, novelty, self_ev)
        R = bridge_overteach_penalty(mc, z, risk, tempt, novelty, self_ev)
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
        Q_per_dose[dose] = round(Q, 6)

    best_dose = max(Q_per_dose, key=Q_per_dose.get)
    action = "WAIT" if best_dose == 0 else ("SOFT" if best_dose == 0.5 else "WARN")
    margins = {
        "margin_best": Q_per_dose[best_dose] - sorted(Q_per_dose.values())[-2],
        "soft_vs_best": Q_per_dose[0.5] - Q_per_dose[best_dose],
    }
    return Q_per_dose, action, best_dose, margins, {
        "p_self": p_self, "risk": risk, "tempt": tempt,
        "delta_s": delta_s, "dvoi": dvoi, "subtype": subtype,
    }


def run_session(lessons, theta, seed, n_teach=20, use_dose=True):
    """Full session returning per-step Q decomposition + decisions."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP, use_dose=use_dose)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []

    for step in range(n_teach):
        les = lessons[step % len(lessons)]
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

        # Full tutor decision
        action, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        # Extract per-dose Q
        Qs, q_action, q_best_dose, margins, state_z = extract_per_dose_Q(
            tutor, sc, fb, lp, lib, scr, 2, m)

        # Agent acts
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
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)

        if dose > 0:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if p_self < 0.5: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if correct and p_self > 0.5:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=(dose > 0), follow_warn=(dose > 0 and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=(correct and dose == 0 and p_self > 0.5),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "action": action, "dose": dose,
            "Q_wait": Qs[0.0], "Q_soft": Qs[0.5], "Q_warn": Qs[1.0],
            "best_dose_3act": q_best_dose,
            "margin_best": margins["margin_best"],
            "soft_vs_best": margins["soft_vs_best"],
            "correct": correct,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            **state_z,
        })
    return records


def main():
    print("═══ P3-C: Exact-Q Geometry + Ablation ═══\n", file=sys.stderr)
    L = ["# P3-C: Exact-Q Geometry Audit + Action-Space Ablation\n\n"]

    # ─── Line 1: Exact-Q Geometry from Real Scenarios ────
    L.append("## Line 1: Exact-Q Per-Dose (Real Scenarios)\n\n")
    print("Line 1: Exact-Q...", file=sys.stderr)
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid))
    total = len(all_recs)

    # Optimality volumes
    wait_n = sum(1 for r in all_recs if r["best_dose_3act"] == 0.0)
    soft_n = sum(1 for r in all_recs if r["best_dose_3act"] == 0.5)
    warn_n = sum(1 for r in all_recs if r["best_dose_3act"] == 1.0)
    L.append(f"**Total decision points: {total}**\n\n")
    L.append("| Action | V_d | Count |\n|--------|:---:|:-----:|\n")
    L.append(f"| WAIT | {wait_n/total:.4f} | {wait_n} |\n")
    L.append(f"| **SOFT** | **{soft_n/total:.4f}** | **{soft_n}** |\n")
    L.append(f"| WARN | {warn_n/total:.4f} | {warn_n} |\n")
    print(f"  V_wait={wait_n/total:.4f} V_soft={soft_n/total:.4f} "
          f"V_warn={warn_n/total:.4f}", file=sys.stderr)

    # Q margin analysis
    L.append("\n### Q Margins\n\n")
    all_diffs_sw = [r["Q_soft"] - r["Q_wait"] for r in all_recs]
    all_diffs_ww = [r["Q_warn"] - r["Q_wait"] for r in all_recs]
    all_diffs_sW = [r["Q_soft"] - r["Q_warn"] for r in all_recs]
    L.append("| Comparison | Mean | Med | Min | Max | Frac>0 |\n")
    L.append("|------------|:----:|:---:|:---:|:---:|:------:|\n")
    for label, vals in [("Q_soft − Q_wait", all_diffs_sw),
                        ("Q_warn − Q_wait", all_diffs_ww),
                        ("Q_soft − Q_warn", all_diffs_sW)]:
        frac = sum(1 for v in vals if v > 0) / len(vals) if vals else 0
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            label, np.mean(vals), np.median(vals), np.min(vals), np.max(vals), frac))

    # By family
    L.append("\n### By Family: Exact-Q Optimal Action\n\n")
    L.append("| Family | n | WAIT | SOFT | WARN | Q_soft−Q_wait(mean) |\n")
    L.append("|--------|:-:|:----:|:----:|:----:|:-------------------:|\n")
    fam_groups = {}
    for r in all_recs:
        fam_groups.setdefault(r["family"], []).append(r)
    for fam in sorted(fam_groups.keys()):
        recs = fam_groups[fam]; n = len(recs)
        w = sum(1 for r in recs if r["best_dose_3act"] == 0.0) / n
        s = sum(1 for r in recs if r["best_dose_3act"] == 0.5) / n
        W = sum(1 for r in recs if r["best_dose_3act"] == 1.0) / n
        diff = np.mean([r["Q_soft"] - r["Q_wait"] for r in recs])
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.4f} |\n".format(
            fam, n, w, s, W, diff))

    # ─── Line 2: Action-Space Ablation ───────────────────
    L.append("\n## Line 2: Action-Space Ablation (3-act vs 2-act)\n\n")
    print("\nLine 2: Ablation...", file=sys.stderr)
    L.append("| θ | Config | Success | Dose Rate | Warn Rate | "
             "STOP Agree | Top-1 |\n")
    L.append("|:-:|--------|:-------:|:---------:|:---------:|"
             ":----------:|:-----:|\n")

    for th in ["safe", "shiny"]:
        recs_3 = []; recs_2 = []
        for sid in range(NS):
            recs_3.extend(run_session(ALL_LESSONS, th, sid, use_dose=True))
            recs_2.extend(run_session(ALL_LESSONS, th, sid, use_dose=False))

        for label, recs in [("3-act", recs_3), ("2-act", recs_2)]:
            n = len(recs)
            succ = sum(r["correct"] for r in recs) / n
            dose_r = sum(1 for r in recs if r["dose"] > 0) / n
            warn_r = sum(1 for r in recs if r["dose"] >= 1.0) / n

            # STOP
            agree = 0
            for r in recs:
                mt = r["m_true"]; mh = r["m_hat"]
                eo = EPS_0 + A_S * mt["nu"] + B_S * mt["gamma_gen"]
                ei = EPS_0 + A_S * mh["nu"] + B_S * mh["gamma_gen"]
                if (eo > STOP_THRESH) == (ei > STOP_THRESH): agree += 1
            stop = agree / n

            # Top-1
            from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2 as CAT
            r_last = recs[-1]
            mt = r_last["m_true"]; mh = r_last["m_hat"]
            scores_o = []; scores_h = []
            for l in CAT:
                g = np.mean(l.gain)
                scores_o.append(g * (1-mt["nu"]) * (1-mt["gamma_gen"]) * mt["tau"])
                scores_h.append(g * (1-mh["nu"]) * (1-mh["gamma_gen"]) * mh["tau"])
            top1 = 1.0 if np.argsort(scores_o)[-1] == np.argsort(scores_h)[-1] else 0.0

            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.1f} |\n".format(
                th, label, succ, dose_r, warn_r, stop, top1))

    # Compare decisions: when does 3-act differ from 2-act?
    L.append("\n### 3-act vs 2-act Decision Comparison\n\n")
    diff_count = 0; total_cmp = 0
    diff_by_fam = {}
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            r3 = run_session(ALL_LESSONS, th, sid, use_dose=True)
            r2 = run_session(ALL_LESSONS, th, sid, use_dose=False)
            for a, b in zip(r3, r2):
                total_cmp += 1
                if a["action"] != b["action"]:
                    diff_count += 1
                    diff_by_fam.setdefault(a["family"], [0, 0])
                    diff_by_fam[a["family"]][0] += 1
                diff_by_fam.setdefault(a["family"], [0, 0])
                diff_by_fam[a["family"]][1] += 1
    L.append(f"**Decision disagreements: {diff_count}/{total_cmp} "
             f"({100*diff_count/max(total_cmp,1):.2f}%)**\n\n")
    if diff_count > 0:
        L.append("| Family | Disagree | Total | Rate |\n")
        L.append("|--------|:--------:|:-----:|:----:|\n")
        for fam in sorted(diff_by_fam.keys()):
            d, t = diff_by_fam[fam]
            if d > 0:
                L.append(f"| {fam} | {d} | {t} | {100*d/t:.1f}% |\n")
    else:
        L.append("**3-act and 2-act produce identical decisions on all steps.**\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")
    if soft_n == 0 and diff_count == 0:
        verdict = ("**SOFT is confirmed structurally redundant.** "
                   "Exact-Q from real scenarios: V_soft=0. "
                   "3-act vs 2-act: zero decision disagreements. "
                   "SOFT can be safely removed from the action space.")
    elif soft_n <= 3 and diff_count <= 3:
        verdict = ("**SOFT is near-redundant.** "
                   f"V_soft={soft_n}/{total}, "
                   f"3v2 disagreements={diff_count}/{total_cmp}. "
                   "Removing SOFT would have negligible impact.")
    else:
        verdict = ("**SOFT has nontrivial presence.** "
                   f"V_soft={soft_n}/{total}, "
                   f"3v2 disagreements={diff_count}/{total_cmp}. "
                   "Further investigation needed.")
    L.append(f"> {verdict}\n")

    rpt = out / "p3c_exact_q_ablation_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
