"""T3 Exp-3B: needhook λ_need sensitivity sweep.

Arms:
  - reset (baseline)
  - persistent_nohook (λ_need=0.0 equivalent)
  - persistent_needhook with λ_need ∈ {0.3, 0.6, 1.0}

Key metrics:
  - SelGap, WarnRate (necessary vs unnecessary)
  - NeedMatch: fraction of selected lessons in profile's Top-K weakness
  - Top-1 shift: how often λ_need changes the top-ranked lesson
  - WR_unnecessary must not spike
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
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_bootstrap import (
    bootstrap_observer, bootstrap_agent_state, finalize_session, make_need_hook,
)
from src.teachers.profile_manager import ProfileManager
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

TICV4_LESSONS = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC-v4", "TIC")]
N_SESSIONS = 5
N_STEPS = 25
N_SEEDS = 8
LAMBDA_VALUES = [0.0, 0.3, 0.6, 1.0]

WARN_NECESSARY = {"verified_warn", "warn_rescue"}
WARN_UNNECESSARY = {"beneficial_novelty", "false_suppression_cost",
                    "self_discovery_needed", "sparse_invalid_advice"}


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def top_k_need_lessons(z_bar, k=3):
    """Return lesson names whose gain best matches profile weakness."""
    z_star = {"RC": 0.70, "TR": 0.65, "EP": 0.55, "VA": 0.70, "IA": 0.70}
    deficits = {p: max(z_star.get(p, 0.5) - z_bar.get(p, 0.5), 0.0)
                for p in PROBE_NAMES}
    scores = {}
    for les in LESSON_CATALOG_V2:
        s = sum(float(les.gain[i]) * deficits.get(p, 0)
                for i, p in enumerate(PROBE_NAMES))
        scores[les.name] = s
    ranked = sorted(scores, key=scores.get, reverse=True)
    return set(ranked[:k])


def sim_step(m, observer, theta, lesson, step_idx, seed, rng):
    ub = {p: 0.4 + 0.1 * (step_idx / N_STEPS) for p in PROBE_NAMES}
    et = generate_episode_from_lesson_v2(
        lesson, step_idx + seed * 10000, theta, ub, rng)
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

    tutor = BCICTv4(agent_params=AP, use_dose=False)
    act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)

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
    else:
        if self_disc:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
    if not correct and tempt > 0.5:
        m.update_gamma_spec(tempt_error=True)
    m.update_risk(risk if not correct else 0.05, 0.15)
    m.snapshot()

    risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
    ev = ObsEvent(
        episode_id=seed, step_id=step_idx, subtype=ep.subtype,
        theta_post=theta, dose=dose, warned=warned,
        follow_warn=(warned and correct),
        d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
        risk_hat=risk_hat, lure=tempt,
        agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
        self_discovery=self_disc,
    )
    observer.update(ev)
    return {"correct": correct, "warned": warned, "subtype": lesson.subtype}


def run_sweep(theta, seed, lambda_need):
    """Run N_SESSIONS with given λ_need. λ_need=None means nohook."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    pm = ProfileManager()
    mode = f"lam{lambda_need}" if lambda_need is not None else "nohook"
    lid = f"L_{seed}_{theta}_{mode}"
    all_sessions = []

    for sess_k in range(N_SESSIONS):
        if lambda_need is not None and pm.session_count(lid) > 0:
            prev = pm.latest(lid)
            bootstrap_observer(observer, prev)
            bootstrap_agent_state(m, prev)
        elif sess_k > 0 and lambda_need is None:
            # nohook still uses carry-over
            if pm.session_count(lid) > 0:
                prev = pm.latest(lid)
                bootstrap_observer(observer, prev)
                bootstrap_agent_state(m, prev)

        ctrl = CurriculumControllerV13(theta=theta)
        ctrl.reset_session(budget=15.0)

        # Install hook with given λ_need
        z_bar = pm.probe_weakness_summary(lid) if pm.session_count(lid) > 0 \
                else {p: 0.5 for p in PROBE_NAMES}
        top_k_need = top_k_need_lessons(z_bar, k=3)

        if lambda_need is not None and lambda_need > 0:
            hook = make_need_hook(z_bar, lambda_need=lambda_need)
            ctrl.install_profile_hook(hook)

        ss = SessionSummary()
        step_results = []
        need_matches = 0
        top1_lessons = []

        for step_i in range(N_STEPS):
            # Use controller to pick lesson
            action, les_selected, J, info = ctrl.select_action(m)
            les = les_selected if les_selected else TICV4_LESSONS[step_i % len(TICV4_LESSONS)]

            # Track NeedMatch
            if hasattr(les, 'name') and les.name in top_k_need:
                need_matches += 1
            top1_lessons.append(les.name if hasattr(les, 'name') else "?")

            # Simulate step with selected lesson
            r = sim_step(m, observer, theta, les, step_i + sess_k * 100, seed, rng)
            step_results.append(r)
            if r["warned"]: ss.n_warn += 1
            else: ss.n_wait += 1

        ss.n_steps = N_STEPS
        est = observer.get_estimate()
        ss.probe_means = {
            "RC": min(est.get("tau", 0.3) * 1.5, 1.0),
            "TR": min(est.get("tau", 0.3), 1.0),
            "EP": max(1.0 - est.get("nu", 0.1) * 2, 0.0),
            "VA": min(est.get("tau", 0.3) * 1.2, 1.0),
            "IA": max(1.0 - est.get("gamma_gen", 0.0) * 2, 0.0),
        }

        profile = finalize_session(observer, m, sess_k, theta, lid, ss)
        pm.finalize_session(lid, profile)

        n = len(step_results)
        success = sum(1 for r in step_results if r["correct"]) / max(n, 1)
        wr_nec = sum(1 for r in step_results
                     if r["warned"] and r["subtype"] in WARN_NECESSARY
                     ) / max(sum(1 for r in step_results
                                if r["subtype"] in WARN_NECESSARY), 1)
        wr_unnec = sum(1 for r in step_results
                       if r["warned"] and r["subtype"] in WARN_UNNECESSARY
                       ) / max(sum(1 for r in step_results
                                  if r["subtype"] in WARN_UNNECESSARY), 1)

        # Count unique top-1 lessons
        from collections import Counter
        top1_dist = Counter(top1_lessons)

        all_sessions.append({
            "session_idx": sess_k,
            "success": round(success, 4),
            "selgap": round(wr_nec - wr_unnec, 4),
            "wr_nec": round(wr_nec, 4),
            "wr_unnec": round(wr_unnec, 4),
            "warn_rate": ss.warn_rate,
            "need_match": need_matches / max(N_STEPS, 1),
            "top1_diversity": len(top1_dist),
        })
    return all_sessions


def main():
    print("═══ T3 Exp-3B: λ_need Sensitivity Sweep ═══\n", file=sys.stderr)
    L = ["# T3 Exp-3B: λ_need Sensitivity Sweep\n\n"]

    # Arms: reset, nohook, λ∈{0.3, 0.6, 1.0}
    arm_configs = [
        ("reset", None),
        ("nohook", None),
        ("λ=0.3", 0.3),
        ("λ=0.6", 0.6),
        ("λ=1.0", 1.0),
    ]

    results = {}
    for arm_name, lam in arm_configs:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                if arm_name == "reset":
                    # Reset mode: no carry-over at all
                    rng = np.random.default_rng(sid * 10000)
                    m = FactoredInternalizationState(); m.snapshot()
                    observer = A1MtObserverFrozen(); observer.reset()
                    sessions_data = []
                    for sess_k in range(N_SESSIONS):
                        if sess_k > 0:
                            m = FactoredInternalizationState(); m.snapshot()
                            observer = A1MtObserverFrozen(); observer.reset()
                        ctrl = CurriculumControllerV13(theta=th)
                        ctrl.reset_session(budget=15.0)
                        ss = SessionSummary()
                        step_res = []
                        for step_i in range(N_STEPS):
                            action, les, J, info = ctrl.select_action(m)
                            if not les:
                                les = TICV4_LESSONS[step_i % len(TICV4_LESSONS)]
                            r = sim_step(m, observer, th, les,
                                        step_i + sess_k*100, sid, rng)
                            step_res.append(r)
                            if r["warned"]: ss.n_warn += 1
                            else: ss.n_wait += 1
                        n = len(step_res)
                        su = sum(1 for r in step_res if r["correct"])/max(n,1)
                        wn = sum(1 for r in step_res
                                 if r["warned"] and r["subtype"] in WARN_NECESSARY
                                 )/max(sum(1 for r in step_res
                                          if r["subtype"] in WARN_NECESSARY),1)
                        wu = sum(1 for r in step_res
                                 if r["warned"] and r["subtype"] in WARN_UNNECESSARY
                                 )/max(sum(1 for r in step_res
                                          if r["subtype"] in WARN_UNNECESSARY),1)
                        sessions_data.append({
                            "session_idx": sess_k,
                            "success": round(su, 4),
                            "selgap": round(wn - wu, 4),
                            "wr_nec": round(wn, 4),
                            "wr_unnec": round(wu, 4),
                            "warn_rate": ss.warn_rate,
                            "need_match": 0.0,
                            "top1_diversity": 0,
                        })
                    sessions = sessions_data
                else:
                    sessions = run_sweep(th, sid, lam)
                for s in sessions:
                    k = s["session_idx"]
                    per_session[(th, k)]["success"].append(s["success"])
                    per_session[(th, k)]["selgap"].append(s["selgap"])
                    per_session[(th, k)]["wr_nec"].append(s["wr_nec"])
                    per_session[(th, k)]["wr_unnec"].append(s["wr_unnec"])
                    per_session[(th, k)]["wr"].append(s["warn_rate"])
                    per_session[(th, k)]["need_match"].append(s["need_match"])
                    per_session[(th, k)]["top1_div"].append(s["top1_diversity"])
            print(f"  {arm_name} / {th} done", file=sys.stderr)
        results[arm_name] = dict(per_session)

    # Table 1: Final session metrics
    L.append("## Final Session Metrics\n\n")
    L.append("| θ | Arm | Success | SelGap | WR_nec | WR_unnec | NeedMatch | Top1Div |\n")
    L.append("|:-:|:---:|:-------:|:------:|:------:|:--------:|:---------:|:-------:|\n")
    for th in ["safe", "shiny"]:
        for arm_name, _ in arm_configs:
            d = results[arm_name].get((th, N_SESSIONS-1), {})
            su = float(np.mean(d.get("success", [0])))
            sg = float(np.mean(d.get("selgap", [0])))
            wn = float(np.mean(d.get("wr_nec", [0])))
            wu = float(np.mean(d.get("wr_unnec", [0])))
            nm = float(np.mean(d.get("need_match", [0])))
            td = float(np.mean(d.get("top1_div", [0])))
            L.append(f"| {th} | {arm_name} | {su:.3f} | {sg:.3f} | "
                     f"{wn:.3f} | {wu:.3f} | {nm:.3f} | {td:.1f} |\n")

    # Table 2: NeedMatch by session (λ sweep)
    L.append("\n## NeedMatch by Session\n\n")
    L.append("| θ | Sess | nohook | λ=0.3 | λ=0.6 | λ=1.0 |\n")
    L.append("|:-:|:----:|:------:|:-----:|:-----:|:-----:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            vals = []
            for arm in ["nohook", "λ=0.3", "λ=0.6", "λ=1.0"]:
                d = results[arm].get((th, k), {})
                vals.append(float(np.mean(d.get("need_match", [0]))))
            L.append(f"| {th} | {k} | {vals[0]:.3f} | {vals[1]:.3f} | "
                     f"{vals[2]:.3f} | {vals[3]:.3f} |\n")

    # Table 3: WarnRate trend
    L.append("\n## WarnRate by Session\n\n")
    L.append("| θ | Sess | reset | nohook | λ=0.3 | λ=0.6 | λ=1.0 |\n")
    L.append("|:-:|:----:|:-----:|:------:|:-----:|:-----:|:-----:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            vals = []
            for arm, _ in arm_configs:
                d = results[arm].get((th, k), {})
                vals.append(float(np.mean(d.get("wr", [0]))))
            L.append(f"| {th} | {k} | {vals[0]:.3f} | {vals[1]:.3f} | "
                     f"{vals[2]:.3f} | {vals[3]:.3f} | {vals[4]:.3f} |\n")

    # Verdict
    L.append("\n## Verdict\n\n")

    # Check: does NeedMatch increase with λ?
    nm_increasing = 0
    for th in ["safe", "shiny"]:
        d03 = results["λ=0.3"].get((th, N_SESSIONS-1), {})
        d10 = results["λ=1.0"].get((th, N_SESSIONS-1), {})
        nm03 = float(np.mean(d03.get("need_match", [0])))
        nm10 = float(np.mean(d10.get("need_match", [0])))
        if nm10 > nm03 + 0.02:
            nm_increasing += 1
    L.append(f"> NeedMatch increases with λ: {nm_increasing}/2 θ\n")

    # Check: WR_unnecessary stays low
    wu_ok = 0
    for th in ["safe", "shiny"]:
        d = results["λ=1.0"].get((th, N_SESSIONS-1), {})
        wu = float(np.mean(d.get("wr_unnec", [0])))
        if wu <= 0.05:
            wu_ok += 1
    L.append(f"> WR_unnecessary ≤ 0.05 at λ=1.0: {wu_ok}/2 θ\n")

    # Check: any differentiation nohook vs λ=1.0?
    diff = 0
    for th in ["safe", "shiny"]:
        dn = results["nohook"].get((th, N_SESSIONS-1), {})
        d1 = results["λ=1.0"].get((th, N_SESSIONS-1), {})
        sn = float(np.mean(dn.get("success", [0])))
        s1 = float(np.mean(d1.get("success", [0])))
        nmn = float(np.mean(dn.get("need_match", [0])))
        nm1 = float(np.mean(d1.get("need_match", [0])))
        if s1 > sn + 0.01 or nm1 > nmn + 0.03:
            diff += 1
    L.append(f"> λ=1.0 differentiates from nohook: {diff}/2 θ\n")

    if nm_increasing >= 1 and wu_ok >= 1:
        if diff >= 1:
            L.append("> **✅ λ_need works: NeedMatch rises, differentiation observed**\n")
        else:
            L.append("> **⚠️ NeedMatch rises but no downstream differentiation — signal enrichment needed (3B-2)**\n")
    else:
        L.append("> **⚠️ λ_need sweep inconclusive — investigate signal quality**\n")

    rpt = out / "t3_exp3b_need_sweep.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
