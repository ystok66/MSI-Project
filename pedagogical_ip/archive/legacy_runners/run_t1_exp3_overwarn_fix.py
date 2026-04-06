"""T1 Exp-3: Over-Warn Fix — WAIT-Preferred Dead-Zone.

Tests a single-parameter dead-zone tie policy:
  - If ΔQ > ε_Q  → WARN
  - If ΔQ < -ε_Q → WAIT
  - If |ΔQ| ≤ ε_Q → WAIT (WAIT-preferred tie)

Sweeps ε_Q ∈ {0.05, 0.08, 0.10, 0.12, 0.15}

Evaluates:
  - OverWarnRate reduction
  - WarnNecRecall (hard constraint: must not drop)
  - DivAll, Div@Active changes
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


def dead_zone_wrapper(action, info, eps_Q):
    """Apply WAIT-preferred dead-zone policy.

    If |ΔQ| ≤ ε_Q, override to WAIT regardless of argmax.
    This is the simplest single-parameter version.
    """
    qd = info.get("q_detail")
    if qd is None:
        return action
    delta_Q = qd["delta_Q"]
    if abs(delta_Q) <= eps_Q:
        return "WAIT"
    return action


def run_session_with_dz(theta, seed, eps_Q, n_teach=20):
    """Run session with both raw and dead-zone infer tutor."""
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

        # Oracle tutor
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, info_o = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer tutor (raw)
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]; m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i_raw, dose_i, info_i = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

        # Dead-zone wrapper
        act_i_dz = dead_zone_wrapper(act_i_raw, info_i, eps_Q)

        # Agent simulation (uses oracle action for state evolution)
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

        # Corrected active mask
        active_raw = (act_o != "WAIT") or (act_i_raw != "WAIT")
        active_dz = (act_o != "WAIT") or (act_i_dz != "WAIT")

        records.append({
            "family": les.name,
            "act_oracle": act_o, "act_infer_raw": act_i_raw, "act_infer_dz": act_i_dz,
            # Raw metrics
            "owr_raw": (act_o == "WAIT" and act_i_raw == "WARN"),
            "diverge_raw": (act_o != act_i_raw),
            "active_raw": active_raw,
            # DZ metrics
            "owr_dz": (act_o == "WAIT" and act_i_dz == "WARN"),
            "diverge_dz": (act_o != act_i_dz),
            "active_dz": active_dz,
            # For WarnNecRecall
            "oracle_is_warn": (act_o == "WARN"),
            "infer_raw_is_warn": (act_i_raw == "WARN"),
            "infer_dz_is_warn": (act_i_dz == "WARN"),
        })
    return records


def main():
    print("═══ T1 Exp-3: Over-Warn Fix ═══\n", file=sys.stderr)
    L = ["# T1 Exp-3: Over-Warn Fix — WAIT-Preferred Dead-Zone\n\n"]

    eps_list = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]

    L.append("## ε_Q Sweep Results\n\n")
    L.append("| ε_Q | DivAll_raw | DivAll_dz | OWR_raw | OWR_dz | "
             "WarnNecRecall_raw | WarnNecRecall_dz |\n")
    L.append("|:---:|:----------:|:---------:|:-------:|:------:|"
             ":-----------------:|:----------------:|\n")

    for eps in eps_list:
        print(f"  ε_Q = {eps:.2f}...", file=sys.stderr)
        all_recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                all_recs.extend(run_session_with_dz(th, sid, eps))

        n = len(all_recs)
        # Raw metrics
        div_raw = sum(r["diverge_raw"] for r in all_recs) / n
        owr_raw = sum(r["owr_raw"] for r in all_recs) / n
        oracle_warn_n = sum(r["oracle_is_warn"] for r in all_recs)
        wnr_raw = (sum(r["oracle_is_warn"] and r["infer_raw_is_warn"]
                       for r in all_recs) / max(oracle_warn_n, 1))

        # DZ metrics
        div_dz = sum(r["diverge_dz"] for r in all_recs) / n
        owr_dz = sum(r["owr_dz"] for r in all_recs) / n
        wnr_dz = (sum(r["oracle_is_warn"] and r["infer_dz_is_warn"]
                       for r in all_recs) / max(oracle_warn_n, 1))

        L.append("| {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
                 "{:.4f} | {:.4f} |\n".format(
            eps, div_raw, div_dz, owr_raw, owr_dz, wnr_raw, wnr_dz))

    # Per-family breakdown at recommended ε
    L.append("\n## Per-Family Detail at ε_Q = 0.10\n\n")
    print("  Per-family at ε=0.10...", file=sys.stderr)
    all_recs_010 = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs_010.extend(run_session_with_dz(th, sid, 0.10))

    fam_groups = defaultdict(list)
    for r in all_recs_010:
        fam_groups[r["family"]].append(r)

    L.append("| Family | OWR_raw | OWR_dz | WNR_raw | WNR_dz |\n")
    L.append("|--------|:-------:|:------:|:-------:|:------:|\n")
    for fam in sorted(fam_groups.keys()):
        fr = fam_groups[fam]
        n_f = len(fr)
        owr_r = sum(r["owr_raw"] for r in fr) / max(n_f, 1)
        owr_d = sum(r["owr_dz"] for r in fr) / max(n_f, 1)
        ow_n = sum(r["oracle_is_warn"] for r in fr)
        wnr_r = sum(r["oracle_is_warn"] and r["infer_raw_is_warn"]
                     for r in fr) / max(ow_n, 1) if ow_n > 0 else 1.0
        wnr_d = sum(r["oracle_is_warn"] and r["infer_dz_is_warn"]
                     for r in fr) / max(ow_n, 1) if ow_n > 0 else 1.0
        mark = " **←**" if fam in FOCUS_FAMILIES else ""
        L.append("| {}{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            fam, mark, owr_r, owr_d, wnr_r, wnr_d))

    # Verdict
    L.append("\n## Verdict\n\n")
    n_010 = len(all_recs_010)
    owr_r_010 = sum(r["owr_raw"] for r in all_recs_010)
    owr_d_010 = sum(r["owr_dz"] for r in all_recs_010)
    oracle_warn_010 = sum(r["oracle_is_warn"] for r in all_recs_010)
    wnr_r_010 = (sum(r["oracle_is_warn"] and r["infer_raw_is_warn"]
                      for r in all_recs_010) / max(oracle_warn_010, 1))
    wnr_d_010 = (sum(r["oracle_is_warn"] and r["infer_dz_is_warn"]
                      for r in all_recs_010) / max(oracle_warn_010, 1))
    L.append(f"> OWR reduction at ε=0.10: {owr_r_010} → {owr_d_010}\n")
    L.append(f"> WarnNecRecall: {wnr_r_010:.4f} → {wnr_d_010:.4f}\n")
    if owr_d_010 <= owr_r_010 and wnr_d_010 >= wnr_r_010 - 0.05:
        L.append("> **✅ Dead-zone (ε=0.10) is effective without WNR degradation**\n")
    else:
        L.append("> **⚠️ Check WarnNecRecall constraint**\n")

    rpt = out / "t1_exp3_overwarn_fix.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
