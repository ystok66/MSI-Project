"""CCT-v4: Bayesian Mastery-Aware Curriculum — Full Experiment.

Exp A: CCT-v4 vs CCT-v3 vs fixed (6 strategies × 2θ × 8 seeds)
Exp B: Budget sweep (low/med/high × 2θ × 8 seeds)
Exp C: Ablation (5 conditions × 2θ × 8 seeds)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from collections import Counter

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
from src.curriculum.curriculum_controller_v3 import CurriculumControllerV3
from src.curriculum.curriculum_controller_v4 import CurriculumControllerV4
from src.curriculum.curriculum_controller_v2 import DoseBudgetTracker
from src.curriculum.adaptive_episode_generator import generate_transfer_episode
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.actionability_v2 import (
    lesson_fidelity, episode_realization_change_rate, micro_policy_change_rate,
)
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
        "mixed_random": [l.name for l in LESSON_CATALOG_V2],
        "self_disc_heavy": ["ppmrb_self_discovery", "tic_self_discovery",
                            "beneficial_novelty", "false_suppression"],
    }
    name = rng.choice(ls.get(strategy, ls["mixed_random"]))
    return LESSON_V2_BY_NAME.get(name, LESSON_CATALOG_V2[0])


def run_session(strategy, theta, seed=0, budget_override=None,
                ablation=None):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    micro = BCICTv4(agent_params=AP)
    bridge = TrainableBridge()
    m = FactoredInternalizationState(); m.snapshot()
    budget_tracker = DoseBudgetTracker()

    cct = None
    if strategy in ("cct_v3",):
        cct = CurriculumControllerV3(bridge=bridge, theta=theta, total_budget=8.0)
    elif strategy in ("cct_v4",) or ablation:
        bud = budget_override if budget_override else 4.0
        cct = CurriculumControllerV4(bridge=bridge, theta=theta, total_budget=bud)
        if ablation == "no_prereq": cct.use_prereq = False
        if ablation == "no_rep": cct.use_rep_penalty = False
        if ablation == "no_stop": cct.use_stop = False
        if ablation == "no_fid": cct.use_fidelity = False
        if ablation == "no_budget": cct.use_budget = False

    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    micro_decisions = []; ep_params_list = []; lf_scores = []
    n_warns = 0; n_blocked = 0; n_eval = 0; n_teach = 0
    stopped_at = None; idx = 0

    max_steps = 12
    for step in range(max_steps):
        if cct and hasattr(cct, 'select_action'):
            action_type, lesson, _, info = cct.select_action(m)
        else:
            action_type = "TEACH"
            lesson = fixed_lesson(strategy, rng)
            info = {}

        if action_type == "STOP":
            stopped_at = step
            break
        if action_type == "EVAL":
            n_eval += 1
            probes = all_probes(m, AP, theta)
            cct.update_mastery(probes)
            continue
        n_teach += 1
        mastery_dict = cct.mastery.mastery() if cct and hasattr(cct, 'mastery') else {p: 0.5 for p in ["RC", "TR", "EP", "VA", "IA"]}

        ep_tuple = generate_episode_from_lesson_v2(lesson, idx, theta, mastery_dict, rng)
        ep_params = ep_tuple[0]
        if cct and hasattr(cct, 'record_realization'):
            cct.record_realization(ep_params)
        ep_params_list.append(ep_params)

        # Lesson fidelity
        lf_scores.append(ep_params.fidelity_to(type('L', (), {
            'subtype': lesson.subtype, 'severity': lesson.severity,
            'dose_profile': lesson.dose_profile, 'family': lesson.family})()))

        # Run episode
        _, spec, gm, cfg, meta, sc = ep_tuple
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng_ep = np.random.default_rng(spec.cue_layout_seed + 9999)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

        budget_tracker.reset(ep_params)
        action, raw_dose, _ = micro.decide(sc, fb, lp, lib, scorer, 2, m)
        feasible = budget_tracker.feasible_doses()
        dose = raw_dose if raw_dose in feasible else max(d for d in feasible if d <= raw_dose)
        if raw_dose > dose: n_blocked += 1
        budget_tracker.consume(dose)
        if cct and hasattr(cct, 'consume_dose'): cct.consume_dose(dose)

        wb = [0.3 * dose, -0.3 * dose]
        nf = [False, False]
        if ep_params.subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng_ep, wb, nf)

        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
        has_self_ev = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            n_warns += 1
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not has_self_ev:
                old_nu = m.nu; m.update_dependence(blind_obey=True)
                m.nu = old_nu + dose * (m.nu - old_nu)
            old_gg = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
        elif not chose_risky:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
        if chose_risky and bar.temptation_score > 0.5:
            m.update_gamma_spec(tempt_error=True)
        if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
            m.update_gamma_spec(false_suppression=True)
        m.snapshot()

        probes = all_probes(m, AP, theta)
        bridge.update(m, probes, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                      bar.temptation_score, ep_params.novelty, 0.7 if has_self_ev else 0.3)
        if cct and hasattr(cct, 'update_mastery'):
            cct.update_mastery(probes)

        correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        mca = correct
        if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
        traces["A"].append({"correct": correct, "mca": mca, "subtype": ep_params.subtype})
        micro_decisions.append(f"{'WARN' if dose >= 1 else 'SOFT' if dose > 0 else 'WAIT'}_{ep_params.subtype}")
        idx += 1

    # Transfer phases
    for phase, n_ep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(n_ep):
            ep_p, spec, gm, cfg, meta, sc = generate_transfer_episode(phase, idx, theta, rng)
            fb, ww = apply_fix(meta, sc)
            fv = np.full_like(fb, 0.3)
            rng_ep = np.random.default_rng(spec.cue_layout_seed + 9999)
            lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL: continue
                        z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)
            ga = False; ac_ok = True
            if phase == "C" and rng_ep.random() < 0.5: ga = True; ac_ok = True
            elif phase == "D" and rng_ep.random() < 0.5: ga = True; ac_ok = False
            wb = ([0.3, -0.3] if ac_ok == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3]) if ga else [0.0, 0.0]
            nf = [False, False]
            if ep_p.subtype == "beneficial_novelty":
                nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
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
            traces[phase].append({"correct": correct, "mca": mca})
            idx += 1

    def rate(ph, key="correct"):
        t = traces.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    otr = overteach_rate_v2(m)
    avg_lf = round(np.mean(lf_scores), 3) if lf_scores else None
    mpg = mastery_progress_gain(cct.mastery.history) if cct and hasattr(cct, 'mastery') and cct.mastery.history else 0.0
    # SE
    transfer_sum = sum(rate(p) or 0 for p in ["B", "C", "D", "E"]) / 4
    se = stop_efficiency(transfer_sum, n_teach) if n_teach > 0 else 0.0
    bdg_blk = cct.budget_blocked_count if cct and hasattr(cct, 'budget_blocked_count') else n_blocked
    bdg_rem = cct.remaining_budget if cct and hasattr(cct, 'remaining_budget') else None

    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"), "mca_C": rate("C", "mca"),
        "wr": round(n_warns / max(n_teach, 1), 3),
        "LF": avg_lf, "n_teach": n_teach, "n_eval": n_eval,
        "stopped": stopped_at is not None, "stopped_at": stopped_at,
        "n_blocked": bdg_blk, "bdg_rem": bdg_rem,
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gg": round(m.gamma_gen, 3), "otr": otr["total"],
        "mpg": mpg, "se": round(se, 4),
        "micro_decisions": micro_decisions,
        "ep_params": ep_params_list[:12],
        "subtypes_a": [t["subtype"] for t in traces["A"]],
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None

def avg_int(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 1) if vs else None


def main():
    print("═══ CCT-v4: Bayesian Curriculum ═══\n", file=sys.stderr)
    lines = ["# CCT-v4: Bayesian Mastery-Aware Curriculum\n\n"]

    # ─── Exp A: v4 vs v3 vs fixed ───
    strategies_a = ["ppmrb_only", "tic_heavy", "self_disc_heavy", "cct_v3", "cct_v4"]
    lines.append("## Exp A: CCT-v4 vs CCT-v3 vs Fixed\n\n")
    lines.append("| θ | Curriculum | #T | #Ev | Stop | LF | **B** | **C** | **D** | **E** | MCA_E | SE |\n")
    lines.append("|---|-----------|---|---|---|---|---|---|---|---|---|---|\n")

    all_a = []
    for theta in ["safe", "shiny"]:
        for s in strategies_a:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["B","C","D","E","mca_E","mca_C","wr","LF",
                 "tau","nu","gg","otr","mpg","se"]}
            a["n_teach"] = avg_int(rs, "n_teach")
            a["n_eval"] = avg_int(rs, "n_eval")
            a["n_blocked"] = avg_int(rs, "n_blocked")
            a["bdg_rem"] = avg(rs, "bdg_rem")
            a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"]) / len(rs), 2)
            a["theta"] = theta; a["strategy"] = s
            a["_subtypes_a"] = [r.get("subtypes_a", []) for r in rs]
            all_a.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | **{}** | **{}** | **{}** | **{}** | {} | {} |\n".format(
                theta, s, sf(a["n_teach"],"{:.0f}"), sf(a["n_eval"],"{:.0f}"),
                sf(a["stopped_frac"]), sf(a["LF"],"{:.3f}"),
                sf(a["B"]), sf(a["C"]), sf(a["D"]), sf(a["E"]),
                sf(a["mca_E"]), sf(a["se"],"{:.4f}")))
            print(f"  {theta}×{s}: B={sf(a['B'])} C={sf(a['C'])} E={sf(a['E'])} "
                  f"#T={sf(a['n_teach'],'{:.0f}')} BdgBlk={sf(a['n_blocked'],'{:.0f}')} "
                  f"SE={sf(a['se'],'{:.3f}')} ν={sf(a['nu'],'{:.2f}')}",
                  file=sys.stderr)

    # State table
    lines.append("\n### State\n\n")
    lines.append("| θ | Curriculum | τ-ν | γg | OTR | MPG | BdgBlk | BdgRem |\n")
    lines.append("|---|-----------|-----|----|----|-----|--------|--------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies_a:
            r = [x for x in all_a if x["theta"] == theta and x["strategy"] == s][0]
            gap = round(r["tau"] - r["nu"], 3) if r["tau"] and r["nu"] else None
            lines.append("| {} | {} | **{}** | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(gap, "{:+.3f}"), sf(r["gg"],"{:.3f}"),
                sf(r["otr"],"{:.3f}"), sf(r["mpg"],"{:.4f}"),
                sf(r["n_blocked"],"{:.0f}"), sf(r["bdg_rem"],"{:.2f}")))

    # ─── Exp B: Budget sweep ───
    print("\nExp B: Budget sweep...", file=sys.stderr)
    lines.append("\n## Exp B: Budget Sweep (CCT-v4)\n\n")
    lines.append("| θ | Budget | #T | Stop | **C** | **E** | ν | γg | OTR | BdgBlk | SE |\n")
    lines.append("|---|-------|---|---|---|---|---|---|----|--------|---|\n")
    for theta in ["safe", "shiny"]:
        for bud in [2.0, 4.0, 8.0]:
            rs = [run_session("cct_v4", theta, sid, budget_override=bud) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","mca_E","nu","gg","otr","se"]}
            a["n_teach"] = avg_int(rs, "n_teach")
            a["n_blocked"] = avg_int(rs, "n_blocked")
            a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"]) / len(rs), 2)
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} | {} |\n".format(
                theta, bud, sf(a["n_teach"],"{:.0f}"), sf(a["stopped_frac"]),
                sf(a["C"]), sf(a["E"]),
                sf(a["nu"],"{:.3f}"), sf(a["gg"],"{:.3f}"),
                sf(a["otr"],"{:.3f}"), sf(a["n_blocked"],"{:.0f}"),
                sf(a["se"],"{:.4f}")))
            print(f"  {theta}×bud={bud}: C={sf(a['C'])} E={sf(a['E'])} "
                  f"#T={sf(a['n_teach'],'{:.0f}')} BdgBlk={sf(a['n_blocked'],'{:.0f}')}",
                  file=sys.stderr)

    # ─── Exp C: Ablation ───
    print("\nExp C: Ablation...", file=sys.stderr)
    lines.append("\n## Exp C: Ablation (CCT-v4)\n\n")
    lines.append("| θ | Condition | #T | Stop | LF | **C** | **E** | ν | OTR | MPG | #Unique |\n")
    lines.append("|---|----------|---|---|---|---|---|---|----|-----|--------|\n")
    for theta in ["safe", "shiny"]:
        for abl in [None, "no_prereq", "no_rep", "no_stop", "no_fid", "no_budget"]:
            label = abl if abl else "full"
            rs = [run_session("cct_v4", theta, sid, ablation=abl) for sid in range(8)]
            a = {k: avg(rs, k) for k in ["C","E","mca_E","nu","otr","mpg","LF"]}
            a["n_teach"] = avg_int(rs, "n_teach")
            a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"]) / len(rs), 2)
            all_subs = [st for seeds in [r.get("subtypes_a", []) for r in rs] for st in seeds]
            n_unique = len(set(all_subs)) if all_subs else 0
            lines.append("| {} | {} | {} | {} | {} | **{}** | **{}** | {} | {} | {} | {} |\n".format(
                theta, label, sf(a["n_teach"],"{:.0f}"), sf(a["stopped_frac"]),
                sf(a["LF"],"{:.3f}"), sf(a["C"]), sf(a["E"]),
                sf(a["nu"],"{:.3f}"), sf(a["otr"],"{:.3f}"),
                sf(a["mpg"],"{:.4f}"), n_unique))
            print(f"  {theta}×{label}: C={sf(a['C'])} E={sf(a['E'])} "
                  f"#T={sf(a['n_teach'],'{:.0f}')} #U={n_unique}",
                  file=sys.stderr)

    with open(out / "cct_v4_report.md", "w") as f:
        f.writelines(lines)
    print(f"\nReport -> results/cct_v4_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
