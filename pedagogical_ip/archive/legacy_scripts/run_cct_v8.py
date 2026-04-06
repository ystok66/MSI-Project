"""CCT-v8: Hierarchical Risk-Calibrated Bayesian — Full Experiment.

Exp A: v8 vs v7 vs v6 vs fixed (5 strats × 2θ × 8 seeds, 4 sessions)
Exp B: Posterior ablation (hierarchical vs local vs no_harm, 2θ × 8 seeds)
Exp C: Risk constraint ablation (constraint vs fixed penalty vs none, 2θ × 8 seeds)
Exp D: Budget sweep (2,4,8 × 2θ × 8 seeds)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal, neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.behavior_probes import all_probes
from src.agents.trainable_bridge import TrainableBridge
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, LESSON_V2_BY_NAME
from src.curriculum.curriculum_controller_v6 import CurriculumControllerV6
from src.curriculum.curriculum_controller_v7 import CurriculumControllerV7
from src.curriculum.curriculum_controller_v8 import CurriculumControllerV8
from src.curriculum.lesson_response_model import LessonResponseModel
from src.curriculum.lesson_response_model_v2 import LessonResponseModelV2
from src.curriculum.lesson_response_model_v3 import LessonResponseModelV3
from src.curriculum.curriculum_controller_v2 import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.curriculum_metrics import mastery_progress_gain, stop_efficiency

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww

def fixed_lesson(strategy, rng):
    ls = {
        "ppmrb_only": ["ppmrb_standard", "ppmrb_self_discovery"],
        "tic_heavy": ["tic_rescue_heavy", "tic_temptation"],
    }
    name = rng.choice(ls.get(strategy, [l.name for l in LESSON_CATALOG_V2]))
    return LESSON_V2_BY_NAME.get(name, LESSON_CATALOG_V2[0])


def run_one_session(cct, strategy, theta, seed, max_teach=12, budget=4.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    micro = BCICTv4(agent_params=AP)
    m = FactoredInternalizationState(); m.snapshot()
    budget_tracker = DoseBudgetTracker()
    if cct and hasattr(cct, 'reset_session'):
        cct.reset_session(budget)
    bridge = cct.bridge if cct else TrainableBridge()

    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    lf_scores = []; n_warns = 0; n_teach = 0; n_eval = 0
    stopped_at = None; idx = 0; fv = None

    for step in range(max_teach + 4):
        if cct:
            action_type, lesson, q_val, info = cct.select_action(m)
        else:
            action_type, lesson, q_val = "TEACH", fixed_lesson(strategy, rng), 0

        if action_type == "STOP":
            stopped_at = step; break
        if action_type == "EVAL":
            n_eval += 1
            probes = all_probes(m, AP, theta)
            cct.update_mastery(probes)
            continue
        n_teach += 1
        if n_teach > max_teach: break
        u_before = cct.mastery.mastery() if cct else {p: 0.5 for p in ["RC","TR","EP","VA","IA"]}
        nu_before = m.nu; gg_before = m.gamma_gen
        otr_before = overteach_rate_v2(m)["total"]

        ep_tuple = generate_episode_from_lesson_v2(lesson, idx + seed * 100, theta, u_before, rng)
        ep_params = ep_tuple[0]
        if cct and hasattr(cct, 'record_realization'): cct.record_realization(ep_params)
        lf_scores.append(ep_params.fidelity_to(type('L', (), {
            'subtype': lesson.subtype, 'severity': lesson.severity,
            'dose_profile': lesson.dose_profile, 'family': lesson.family})()))

        _, spec, gm, cfg, meta, sc = ep_tuple
        fb, ww = apply_fix(meta, sc)
        if fv is None: fv = np.full_like(fb, 0.3)
        rng_ep = np.random.default_rng(spec.cue_layout_seed + 9999)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib = BranchConceptLibrary(); scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0); scorer.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

        budget_tracker.reset(ep_params)
        _, raw_dose, _ = micro.decide(sc, fb, lp, lib, scorer, 2, m)
        feasible = budget_tracker.feasible_doses()
        dose = raw_dose if raw_dose in feasible else max(d for d in feasible if d <= raw_dose)
        budget_tracker.consume(dose)
        if cct and hasattr(cct, 'consume_dose'): cct.consume_dose(dose)
        if dose > 0: n_warns += 1
        wb = [0.3 * dose, -0.3 * dose]; nf = [False, False]
        if ep_params.subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng_ep, wb, nf)
        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
        has_self_ev = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not has_self_ev: old = m.nu; m.update_dependence(blind_obey=True); m.nu = old + dose * (m.nu - old)
            old = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True); m.gamma_gen = old + dose * (m.gamma_gen - old)
        elif not chose_risky:
            m.update_dependence(self_discovery=True); m.update_gamma_gen(successful_exploration=True)
        if chose_risky and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
        if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky: m.update_gamma_spec(false_suppression=True)
        m.snapshot()
        probes = all_probes(m, AP, theta)
        bridge.update(m, probes, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                      bar.temptation_score, ep_params.novelty, 0.7 if has_self_ev else 0.3)
        if cct and hasattr(cct, 'update_response'):
            otr_after = overteach_rate_v2(m)["total"]
            try:
                cct.update_response(lesson.name, dict(u_before), cct.mastery.mastery(),
                                    nu_before, m.nu, gg_before, m.gamma_gen, otr_before, otr_after)
            except TypeError:
                cct.update_response(lesson.name, dict(u_before), cct.mastery.mastery())

        correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        mca = correct
        if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
        traces["A"].append({"correct": correct, "mca": mca, "subtype": ep_params.subtype}); idx += 1

    for phase, n_ep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(n_ep):
            ep_p, spec, gm, cfg, meta, sc = generate_transfer_episode(phase, idx + seed * 100, theta, rng)
            fb, ww = apply_fix(meta, sc)
            if fv is None: fv = np.full_like(fb, 0.3)
            rng_ep = np.random.default_rng(spec.cue_layout_seed + 9999)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL: continue
                        z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp); sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b, risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
            ga = False; ac_ok = True
            if phase == "C" and rng_ep.random() < 0.5: ga = True
            elif phase == "D" and rng_ep.random() < 0.5: ga = True; ac_ok = False
            wb = ([0.3, -0.3] if ac_ok == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3]) if ga else [0.0, 0.0]
            nf = [False, False]
            if ep_p.subtype == "beneficial_novelty": nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
            ac = sample_factored_choice([bas, bar], theta, m, AP, rng_ep, wb, nf)
            chose_risky = (ac != sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
            if phase in ("C", "D") and ga:
                hs = (spec.d_commit > spec.d_reveal + 1)
                if phase == "C" and not chose_risky: m.update_trust(warn_helpful=True)
                if phase == "D" and chose_risky: m.update_dependence(blind_obey=True)
                elif phase == "D" and not chose_risky and hs: m.update_dependence(self_discovery=True)
            if chose_risky and bar.temptation_score > 0.5: m.update_gamma_spec(tempt_error=True)
            m.snapshot()
            correct = chose_risky if ep_p.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
            mca = correct
            if ep_p.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
            traces[phase].append({"correct": correct, "mca": mca}); idx += 1

    def rate(ph, key="correct"):
        t = traces.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    otr = overteach_rate_v2(m)
    avg_lf = round(np.mean(lf_scores), 3) if lf_scores else None
    mpg = mastery_progress_gain(cct.mastery.history) if cct and hasattr(cct, 'mastery') and cct.mastery.history else 0.0
    transfer_avg = sum(rate(p) or 0 for p in ["B","C","D","E"]) / 4
    se = stop_efficiency(transfer_avg, n_teach) if n_teach > 0 else 0.0
    post_up = cct.posterior_stats().get("n_updated", 0) if cct and hasattr(cct, 'posterior_stats') else 0

    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"), "LF": avg_lf,
        "n_teach": n_teach, "n_eval": n_eval,
        "stopped": stopped_at is not None,
        "n_blocked": cct.budget_blocked_count if cct and hasattr(cct, 'budget_blocked_count') else 0,
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gg": round(m.gamma_gen, 3), "otr": otr["total"],
        "mpg": mpg, "se": round(se, 4), "post_up": post_up,
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None
def avg_int(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 1) if vs else None


def make_cct(strategy, theta, budget, ablation=None):
    if strategy == "cct_v8" or (ablation and "local" not in (ablation or "")):
        shared = LessonResponseModelV3()
        cct = CurriculumControllerV8(theta=theta, total_budget=budget, response=shared)
        if ablation == "no_harm": cct.use_harm = False
        if ablation == "no_unc": cct.use_uncertainty = False
        if ablation == "no_stop": cct.use_stop = False
        if ablation == "no_hier": cct.use_hierarchical = False
        if ablation == "fixed_penalty":
            cct.use_harm = True; cct.lambda_viol = 2.5; cct.eta_0 = 0.0  # always violate → acts as fixed penalty
        return cct
    elif strategy == "cct_v7":
        shared = LessonResponseModelV2()
        return CurriculumControllerV7(theta=theta, total_budget=budget, response=shared)
    elif strategy == "cct_v6":
        shared = LessonResponseModel()
        return CurriculumControllerV6(theta=theta, total_budget=budget, response=shared,
                                      _rng=np.random.default_rng(42))
    elif ablation == "local_only":
        shared = LessonResponseModelV3()
        cct = CurriculumControllerV8(theta=theta, total_budget=budget, response=shared)
        cct.use_hierarchical = False
        return cct
    return None


def run_multi(strategy, theta, seed, n_sessions=4, budget=4.0, ablation=None):
    cct = make_cct(strategy, theta, budget, ablation)
    results = []
    for sess in range(n_sessions):
        r = run_one_session(cct, strategy, theta, seed * 10 + sess, budget=budget)
        results.append(r)
    last = results[-1]
    first = results[0]
    last["dC"] = round((last["C"] or 0) - (first["C"] or 0), 3)
    last["dE"] = round((last["E"] or 0) - (first["E"] or 0), 3)
    return last


def main():
    print("═══ CCT-v8: Hierarchical Risk-Calibrated ═══\n", file=sys.stderr)
    lines = ["# CCT-v8: Hierarchical Risk-Calibrated Bayesian Planner\n\n"]

    # ─── Exp A ───
    strats = ["ppmrb_only", "tic_heavy", "cct_v6", "cct_v7", "cct_v8"]
    lines.append("## Exp A: CCT-v8 vs v7 vs v6 vs Fixed (4 sessions)\n\n")
    lines.append("| θ | Strat | #T | Stop | **C** | **E** | SE | OTR | ν | PostUp |\n")
    lines.append("|---|------|---|---|---|---|---|---|---|---|\n")
    for theta in ["safe", "shiny"]:
        for s in strats:
            rs = [run_multi(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","se","otr","nu","gg","mpg"]}
            a["n_teach"] = avg_int(rs, "n_teach")
            a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"]) / len(rs), 2)
            a["post_up"] = avg_int(rs, "post_up")
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["n_teach"],"{:.0f}"), sf(a["stopped_frac"]),
                sf(a["C"]), sf(a["E"]), sf(a["se"],"{:.4f}"),
                sf(a["otr"],"{:.3f}"), sf(a["nu"],"{:.3f}"), sf(a["post_up"],"{:.0f}")))
            print(f"  {theta}×{s}: C={sf(a['C'])} E={sf(a['E'])} SE={sf(a['se'],'{:.3f}')} OTR={sf(a['otr'],'{:.2f}')} ν={sf(a['nu'],'{:.2f}')} PostUp={sf(a['post_up'],'{:.0f}')}", file=sys.stderr)

    # ─── Exp B: posterior ───
    print("\nExp B: Posterior ablation...", file=sys.stderr)
    lines.append("\n## Exp B: Posterior Ablation\n\n")
    lines.append("| θ | Condition | **C** | **E** | SE | OTR | PostUp |\n")
    lines.append("|---|----------|---|---|---|---|---|\n")
    for theta in ["safe", "shiny"]:
        for abl in [None, "no_hier", "no_harm", "no_unc"]:
            label = abl if abl else "full (hierarchical)"
            rs = [run_multi("cct_v8", theta, sid, ablation=abl) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","se","otr"]}
            a["post_up"] = avg_int(rs, "post_up")
            lines.append("| {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                theta, label, sf(a["C"]), sf(a["E"]),
                sf(a["se"],"{:.4f}"), sf(a["otr"],"{:.3f}"), sf(a["post_up"],"{:.0f}")))
            print(f"  {theta}×{label}: C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')}", file=sys.stderr)

    # ─── Exp C: risk constraint ───
    print("\nExp C: Risk constraint...", file=sys.stderr)
    lines.append("\n## Exp C: Risk Constraint Ablation\n\n")
    lines.append("| θ | Condition | **C** | **E** | SE | OTR | ν |\n")
    lines.append("|---|----------|---|---|---|---|---|\n")
    for theta in ["safe", "shiny"]:
        for abl in [None, "fixed_penalty", "no_harm", "no_stop"]:
            label = abl if abl else "adaptive_budget"
            rs = [run_multi("cct_v8", theta, sid, ablation=abl) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","se","otr","nu"]}
            lines.append("| {} | {} | **{}** | **{}** | {} | {} | {} |\n".format(
                theta, label, sf(a["C"]), sf(a["E"]),
                sf(a["se"],"{:.4f}"), sf(a["otr"],"{:.3f}"), sf(a["nu"],"{:.3f}")))
            print(f"  {theta}×{label}: C={sf(a['C'])} E={sf(a['E'])} OTR={sf(a['otr'],'{:.2f}')} ν={sf(a['nu'],'{:.2f}')}", file=sys.stderr)

    # ─── Exp D: budget ───
    print("\nExp D: Budget...", file=sys.stderr)
    lines.append("\n## Exp D: Budget Sweep\n\n")
    lines.append("| θ | Budget | #T | **C** | **E** | ν | OTR | BdgBlk | SE |\n")
    lines.append("|---|-------|---|---|---|---|---|----|---|\n")
    for theta in ["safe", "shiny"]:
        for bud in [2.0, 4.0, 8.0]:
            rs = [run_multi("cct_v8", theta, sid, budget=bud) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","nu","otr","se"]}
            a["n_teach"] = avg_int(rs, "n_teach"); a["n_blocked"] = avg_int(rs, "n_blocked")
            lines.append("| {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                theta, bud, sf(a["n_teach"],"{:.0f}"), sf(a["C"]), sf(a["E"]),
                sf(a["nu"],"{:.3f}"), sf(a["otr"],"{:.3f}"),
                sf(a["n_blocked"],"{:.0f}"), sf(a["se"],"{:.4f}")))
            print(f"  {theta}×bud={bud}: C={sf(a['C'])} E={sf(a['E'])} BdgBlk={sf(a['n_blocked'],'{:.0f}')}", file=sys.stderr)

    with open(out / "cct_v8_report.md", "w") as f:
        f.writelines(lines)
    print(f"\nReport -> results/cct_v8_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
