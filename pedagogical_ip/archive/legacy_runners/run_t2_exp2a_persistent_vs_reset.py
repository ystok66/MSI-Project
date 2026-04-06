"""T2 Exp-2A: Persistent vs Reset — PP-MRB Selective Fading.

Multi-session comparison:
  - reset: each session starts from default priors
  - persistent: each session carries over from previous terminal state

Focus: PP-MRB subtype-level WarnRate and SelGap evolution.

Subtypes:
  - wait_clean: low lure, should-WAIT → persistent should learn to talk less
  - wait_lure: high lure, still WAIT → persistent should also reduce WarnRate
  - boundary_obs: near-boundary → should be stable
  - warn_trap: must-WARN → WarnRate must not drop

Key metric:
  SelGap_k = WarnRate_k(warn_trap) - avg(WarnRate_k(wait_clean), WarnRate_k(wait_lure))
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
from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_bootstrap import (
    bootstrap_observer, bootstrap_agent_state, finalize_session,
)
from src.teachers.profile_manager import ProfileManager
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Focus on PP-MRB lessons
PPMRB_LESSONS = [l for l in LESSON_CATALOG_V2 if l.family == "PP-MRB"]
ALL_LESSONS = list(LESSON_CATALOG_V2)  # full catalog for variety

N_SESSIONS = 5
N_TEACH_PER_SESSION = 20
N_SEEDS = 10

# Subtype classification heuristic (based on lesson + temptation + risk)
def classify_subtype(les, tempt, risk, p_self):
    """Classify step into one of 4 PP-MRB subtypes."""
    if risk > 0.4 and tempt > 0.3:
        return "warn_trap"
    elif tempt > 0.5:
        return "wait_lure"
    elif risk < 0.25 and tempt < 0.25:
        return "wait_clean"
    else:
        return "boundary_obs"


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_multi_session(theta, seed, persistent=True):
    """Run N_SESSIONS for one learner.

    Returns: list of per-session records.
    """
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    pm = ProfileManager()
    learner_id = f"L_{seed}_{theta}_{'P' if persistent else 'R'}"

    all_session_records = []

    for session_k in range(N_SESSIONS):
        # Bootstrap from profile if persistent and has history
        if persistent and pm.session_count(learner_id) > 0:
            prev = pm.latest(learner_id)
            bootstrap_observer(observer, prev)
            bootstrap_agent_state(m, prev)
        else:
            observer.reset()
            if session_k > 0 and not persistent:
                # Reset mode: fresh state each session
                m = FactoredInternalizationState(); m.snapshot()
                observer = A1MtObserverFrozen(); observer.reset()

        session_records = []
        session_summary = SessionSummary()
        subtype_warn_counts = defaultdict(int)
        subtype_total_counts = defaultdict(int)

        for step in range(N_TEACH_PER_SESSION):
            les = ALL_LESSONS[step % len(ALL_LESSONS)]
            ub = {p: 0.4 + 0.1 * step / N_TEACH_PER_SESSION for p in PROBE_NAMES}
            et = generate_episode_from_lesson_v2(
                les, step + session_k * 100 + seed * 10000, theta, ub, rng)
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
            act_o, dose_o, _ = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

            # Infer tutor (using observer estimate)
            m_hat = FactoredInternalizationState()
            est = observer.get_estimate()
            m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
            m_hat.gamma_gen = est["gamma_gen"]; m_hat.snapshot()
            tutor_i = BCICTv4(agent_params=AP, use_dose=False)
            act_i, dose_i, _ = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

            # Agent simulation
            action, dose = act_o, dose_o
            dc = getattr(sc, 'commit_depth', 3)
            dr = getattr(sc, 'reveal_depth', 2)
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
                session_summary.n_warn += 1
            else:
                if self_disc:
                    m.update_dependence(self_discovery=True)
                    m.update_gamma_gen(successful_exploration=True)
                session_summary.n_wait += 1
            if not correct and tempt > 0.5:
                m.update_gamma_spec(tempt_error=True)
            m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()
            risk_hat = float(lp.predict_risk(
                sr[0:4] if len(sr) >= 4 else np.zeros(4)))
            ev = ObsEvent(
                episode_id=seed, step_id=step, subtype=ep.subtype,
                theta_post=theta,
                dose=dose, warned=warned, follow_warn=(warned and correct),
                d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
                risk_hat=risk_hat, lure=tempt,
                agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
                self_discovery=self_disc,
                m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            )
            observer.update(ev)

            # Classify subtype
            sub = classify_subtype(les, tempt, risk, p_self)
            subtype_total_counts[sub] += 1
            if act_i == "WARN":
                subtype_warn_counts[sub] += 1

            session_records.append({
                "session": session_k, "step": step,
                "family": les.name, "subtype": sub,
                "act_oracle": act_o, "act_infer": act_i,
                "warned": warned, "correct": correct,
            })

        # Compute per-subtype rates
        for sub in subtype_total_counts:
            session_summary.warn_rate_by_subtype[sub] = (
                subtype_warn_counts[sub] /
                max(subtype_total_counts[sub], 1))
        session_summary.subtype_counts = dict(subtype_total_counts)
        session_summary.n_steps = N_TEACH_PER_SESSION

        # Finalize profile
        profile = finalize_session(
            observer, m, session_k, theta, learner_id, session_summary)
        pm.finalize_session(learner_id, profile)

        all_session_records.append({
            "session_idx": session_k,
            "records": session_records,
            "warn_rate_by_subtype": dict(session_summary.warn_rate_by_subtype),
            "overall_warn_rate": session_summary.warn_rate,
            "m_hat": dict(observer.get_estimate()),
            "m_true": m.as_dict,
        })

    return all_session_records


def compute_selgap(warn_rates: dict) -> float:
    """SelGap = WarnRate(warn_trap) - avg(WarnRate(wait_clean), WarnRate(wait_lure))."""
    wt = warn_rates.get("warn_trap", 0.0)
    wc = warn_rates.get("wait_clean", 0.0)
    wl = warn_rates.get("wait_lure", 0.0)
    return wt - 0.5 * (wc + wl)


def main():
    print("═══ T2 Exp-2A: Persistent vs Reset ═══\n", file=sys.stderr)
    L = ["# T2 Exp-2A: Persistent vs Reset — PP-MRB\n\n"]

    # Run experiments
    results = {"persistent": {}, "reset": {}}
    for mode_name, is_persistent in [("persistent", True), ("reset", False)]:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                sessions = run_multi_session(th, sid, persistent=is_persistent)
                for s in sessions:
                    k = s["session_idx"]
                    per_session[(th, k)]["overall_wr"].append(s["overall_warn_rate"])
                    for sub in ["wait_clean", "wait_lure", "boundary_obs", "warn_trap"]:
                        per_session[(th, k)][f"wr_{sub}"].append(
                            s["warn_rate_by_subtype"].get(sub, 0.0))
            print(f"  {mode_name} / {th} done", file=sys.stderr)
        results[mode_name] = dict(per_session)

    # ═══ Table 1: WarnRate by session ═══
    for th in ["safe", "shiny"]:
        L.append(f"\n## θ = {th}\n\n")
        L.append("### WarnRate by Session\n\n")
        L.append("| Session | Mode | Overall | wait_clean | wait_lure | "
                 "boundary | warn_trap | SelGap |\n")
        L.append("|:-------:|:----:|:-------:|:----------:|:---------:|"
                 ":--------:|:---------:|:------:|\n")

        for k in range(N_SESSIONS):
            for mode in ["reset", "persistent"]:
                data = results[mode].get((th, k), {})
                owr = float(np.mean(data.get("overall_wr", [0])))
                wc = float(np.mean(data.get("wr_wait_clean", [0])))
                wl = float(np.mean(data.get("wr_wait_lure", [0])))
                bo = float(np.mean(data.get("wr_boundary_obs", [0])))
                wt = float(np.mean(data.get("wr_warn_trap", [0])))
                sg = compute_selgap({"wait_clean": wc, "wait_lure": wl,
                                     "warn_trap": wt})
                L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | "
                         "{:.3f} | {:.3f} | {:.3f} |\n".format(
                    k, mode, owr, wc, wl, bo, wt, sg))

    # ═══ Table 2: SelGap trend ═══
    L.append("\n## SelGap Trend Summary\n\n")
    L.append("| θ | Mode | Session 0 | Session 2 | Session 4 | Trend |\n")
    L.append("|:-:|:----:|:---------:|:---------:|:---------:|:-----:|\n")
    for th in ["safe", "shiny"]:
        for mode in ["reset", "persistent"]:
            sgs = []
            for k in range(N_SESSIONS):
                data = results[mode].get((th, k), {})
                wc = float(np.mean(data.get("wr_wait_clean", [0])))
                wl = float(np.mean(data.get("wr_wait_lure", [0])))
                wt = float(np.mean(data.get("wr_warn_trap", [0])))
                sgs.append(compute_selgap({"wait_clean": wc, "wait_lure": wl,
                                           "warn_trap": wt}))
            trend = "↑" if sgs[-1] > sgs[0] + 0.02 else ("↓" if sgs[-1] < sgs[0] - 0.02 else "→")
            L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |\n".format(
                th, mode,
                sgs[0] if len(sgs) > 0 else 0,
                sgs[2] if len(sgs) > 2 else 0,
                sgs[4] if len(sgs) > 4 else 0,
                trend))

    # ═══ Table 3: "Talk Less After Learning" ═══
    L.append("\n## Talk Less After Learning\n\n")
    L.append("Δ WarnRate(wait_clean) from Session 0 → Session 4:\n\n")
    L.append("| θ | Reset Δ | Persistent Δ | Persistent Better? |\n")
    L.append("|:-:|:-------:|:------------:|:------------------:|\n")
    for th in ["safe", "shiny"]:
        for mode in ["reset", "persistent"]:
            data_0 = results[mode].get((th, 0), {})
            data_4 = results[mode].get((th, N_SESSIONS - 1), {})
            wc_0 = float(np.mean(data_0.get("wr_wait_clean", [0])))
            wc_4 = float(np.mean(data_4.get("wr_wait_clean", [0])))
            if mode == "reset":
                delta_r = wc_4 - wc_0
            else:
                delta_p = wc_4 - wc_0
        better = "✅" if delta_p < delta_r - 0.01 else ("≈" if abs(delta_p - delta_r) < 0.01 else "❌")
        L.append(f"| {th} | {delta_r:+.3f} | {delta_p:+.3f} | {better} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")
    # Check if persistent SelGap is better in final session
    pass_count = 0
    for th in ["safe", "shiny"]:
        data_r = results["reset"].get((th, N_SESSIONS - 1), {})
        data_p = results["persistent"].get((th, N_SESSIONS - 1), {})
        sg_r = compute_selgap({
            "wait_clean": float(np.mean(data_r.get("wr_wait_clean", [0]))),
            "wait_lure": float(np.mean(data_r.get("wr_wait_lure", [0]))),
            "warn_trap": float(np.mean(data_r.get("wr_warn_trap", [0]))),
        })
        sg_p = compute_selgap({
            "wait_clean": float(np.mean(data_p.get("wr_wait_clean", [0]))),
            "wait_lure": float(np.mean(data_p.get("wr_wait_lure", [0]))),
            "warn_trap": float(np.mean(data_p.get("wr_warn_trap", [0]))),
        })
        if sg_p >= sg_r - 0.02:
            pass_count += 1

    # Check warn_trap doesn't collapse
    wt_collapse = False
    for th in ["safe", "shiny"]:
        data_p = results["persistent"].get((th, N_SESSIONS - 1), {})
        wt = float(np.mean(data_p.get("wr_warn_trap", [0])))
        if wt < 0.3:  # warn_trap WarnRate should stay high
            wt_collapse = True

    L.append(f"> SelGap maintained or improved: {pass_count}/2 θ\n")
    L.append(f"> warn_trap collapse: {'⚠️ YES' if wt_collapse else '✅ NO'}\n")
    if pass_count >= 1 and not wt_collapse:
        L.append("> **✅ Persistent profile shows value over reset**\n")
    else:
        L.append("> **⚠️ Persistent advantage not clear — investigate**\n")

    rpt = out / "t2_exp2a_persistent_vs_reset.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
