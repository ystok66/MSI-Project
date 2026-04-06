"""CCT-v3 Experiment: Mastery-Aware Closed-Loop Curriculum.

6 strategies × 2θ × 8 seeds.
Macro: TEACH / EVAL / STOP with mastery+prereq+budget.
Micro: BCICTv4 with dose budget constraint.
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
        "cct_v2_style": [l.name for l in LESSON_CATALOG_V2],
    }
    name = rng.choice(ls.get(strategy, ls["mixed_random"]))
    return LESSON_V2_BY_NAME.get(name, LESSON_CATALOG_V2[0])


def run_episode(spec_tuple, m, bridge, micro_tutor, budget, theta, AP, rng):
    """Run one Phase A episode. Returns (traces_entry, micro_decision)."""
    ep_params, spec, gm, cfg, meta, sc = spec_tuple
    fb, ww = apply_fix(meta, sc)
    fv = np.full_like(fb, 0.3)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    rng_ep = np.random.default_rng(spec.cue_layout_seed + 9999)
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

    budget.reset(ep_params)
    action, raw_dose, _ = micro_tutor.decide(sc, fb, lp, lib, scorer, 2, m)
    feasible = budget.feasible_doses()
    dose = raw_dose if raw_dose in feasible else max(d for d in feasible if d <= raw_dose)
    budget_blocked = (raw_dose > dose)
    budget.consume(dose)

    wb = [0.3 * dose, -0.3 * dose]
    nf = [False, False]
    if ep_params.subtype == "beneficial_novelty":
        nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
    ac = sample_factored_choice([bas, bar], theta, m, AP, rng_ep, wb, nf)

    chose_risky = (ac != sc.oracle_safe_branch_id)
    m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
    has_self_ev = (spec.d_commit > spec.d_reveal + 1)
    if dose > 0:
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
                  bar.temptation_score, ep_params.novelty,
                  0.7 if has_self_ev else 0.3)

    correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
    mca = correct
    if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False

    dec = f"{'WARN' if dose >= 1 else 'SOFT' if dose > 0 else 'WAIT'}_{ep_params.subtype}"
    return {
        "correct": correct, "mca": mca, "subtype": ep_params.subtype,
        "budget_blocked": budget_blocked, "dose": dose,
    }, dec, probes


def run_transfer(phase, idx, theta, m, AP, bridge):
    ep_params, spec, gm, cfg, meta, sc = generate_transfer_episode(
        phase, idx, theta, np.random.default_rng(idx))
    fb, ww = apply_fix(meta, sc)
    fv = np.full_like(fb, 0.3)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    rng = np.random.default_rng(spec.cue_layout_seed + 9999)
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

    give_advice = False; advice_correct = True
    if phase == "C" and rng.random() < 0.5: give_advice = True; advice_correct = True
    elif phase == "D" and rng.random() < 0.5: give_advice = True; advice_correct = False
    if give_advice:
        wb = ([0.3, -0.3] if advice_correct == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3])
    else:
        wb = [0.0, 0.0]
    nf = [False, False]
    if ep_params.subtype == "beneficial_novelty":
        nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
    ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, nf)

    chose_risky = (ac != sc.oracle_safe_branch_id)
    m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
    has_self_ev = (spec.d_commit > spec.d_reveal + 1)
    if phase in ("C", "D") and give_advice:
        if phase == "C" and not chose_risky: m.update_trust(warn_helpful=True)
        if phase == "D" and chose_risky: m.update_dependence(blind_obey=True)
        elif phase == "D" and not chose_risky and has_self_ev:
            m.update_dependence(self_discovery=True)
    if chose_risky and bar.temptation_score > 0.5:
        m.update_gamma_spec(tempt_error=True)
    m.snapshot()

    correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
    mca = correct
    if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
    return {"correct": correct, "mca": mca, "subtype": ep_params.subtype}, ep_params


def run_session(strategy, theta, seed=0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    micro = BCICTv4(agent_params=AP)
    bridge = TrainableBridge()
    m = FactoredInternalizationState(); m.snapshot()
    budget = DoseBudgetTracker()

    cct = None
    if strategy == "cct_v3":
        cct = CurriculumControllerV3(bridge=bridge, theta=theta, total_budget=8.0)

    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    micro_decisions = []; ep_params_list = []; lf_scores = []
    n_warns = 0; n_blocked = 0; n_eval = 0; n_teach = 0
    stopped_at = None; idx = 0

    # Phase A: up to 12 macro steps (TEACH/EVAL/STOP)
    max_macro_steps = 12
    for step in range(max_macro_steps):
        if strategy == "cct_v3":
            action_type, lesson, _, info = cct.select_action(m)
            if action_type == "STOP":
                stopped_at = step
                break
            if action_type == "EVAL":
                n_eval += 1
                probes = all_probes(m, AP, theta)
                cct.update_mastery(probes)
                continue
            # TEACH
            n_teach += 1
            mastery_dict = cct.mastery.mastery()
        else:
            lesson = fixed_lesson(strategy, rng)
            mastery_dict = {p: 0.5 for p in ["RC", "TR", "EP", "VA", "IA"]}
            n_teach += 1

        ep_tuple = generate_episode_from_lesson_v2(
            lesson, idx, theta, mastery_dict, rng)
        ep_params = ep_tuple[0]
        if cct:
            cct.record_realization(ep_params)
        ep_params_list.append(ep_params)
        lf_scores.append(lesson_fidelity(ep_params, type('L', (), {
            'subtype': lesson.subtype, 'severity': lesson.severity,
            'dose_profile': lesson.dose_profile, 'family': lesson.family,
            'fidelity_to': ep_params.fidelity_to})()))

        result, dec, probes = run_episode(ep_tuple, m, bridge, micro, budget, theta, AP, rng)
        if cct:
            cct.update_mastery(probes)
        traces["A"].append(result)
        micro_decisions.append(dec)
        if result["dose"] > 0: n_warns += 1
        if result["budget_blocked"]: n_blocked += 1
        idx += 1

    # Phase B-E
    for phase, n_ep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(n_ep):
            result, ep_params = run_transfer(phase, idx, theta, m, AP, bridge)
            traces[phase].append(result)
            ep_params_list.append(ep_params)
            idx += 1

    def rate(ph, key="correct"):
        t = traces.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    otr = overteach_rate_v2(m)
    avg_lf = round(np.mean(lf_scores), 3) if lf_scores else None

    mpg = 0.0
    if cct and cct.mastery.history:
        mpg = mastery_progress_gain(cct.mastery.history)

    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"), "mca_C": rate("C", "mca"),
        "wr": round(n_warns / max(n_teach, 1), 3),
        "LF": avg_lf, "n_teach": n_teach, "n_eval": n_eval,
        "stopped": stopped_at is not None, "stopped_at": stopped_at,
        "n_blocked": n_blocked,
        "EP": probes["EP"], "VA": probes["VA"], "IA": probes["IA"],
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gg": round(m.gamma_gen, 3), "otr": otr["total"],
        "mpg": mpg,
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
    print("═══ CCT-v3: Mastery-Aware Curriculum ═══\n", file=sys.stderr)
    strategies = ["ppmrb_only", "tic_heavy", "mixed_random",
                  "self_disc_heavy", "cct_v2_style", "cct_v3"]
    lines = ["# CCT-v3: Mastery-Aware Closed-Loop Curriculum\n\n"]

    lines.append("## 5-Phase Transfer\n\n")
    lines.append("| θ | Curriculum | WR | LF | #Teach | #Eval | Stopped | **B** | **C** | MCA_C | **D** | **E** | MCA_E |\n")
    lines.append("|---|-----------|---|---|--------|-------|---------|----|----|----|----|----|----|----|\n")

    all_r = []
    for theta in ["safe", "shiny"]:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in [
                "B", "C", "D", "E", "mca_E", "mca_C", "wr", "LF",
                "EP", "VA", "IA", "tau", "nu", "gg", "otr", "mpg"]}
            a["n_teach"] = avg_int(rs, "n_teach")
            a["n_eval"] = avg_int(rs, "n_eval")
            a["n_blocked"] = avg_int(rs, "n_blocked")
            a["stopped_frac"] = round(sum(1 for r in rs if r["stopped"]) / len(rs), 2)
            a["theta"] = theta; a["strategy"] = s
            a["_micro"] = [r.get("micro_decisions", []) for r in rs]
            a["_ep_params"] = [r.get("ep_params", []) for r in rs]
            a["_subtypes_a"] = [r.get("subtypes_a", []) for r in rs]
            all_r.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | **{}** | **{}** | {} | **{}** | **{}** | {} |\n".format(
                theta, s, sf(a["wr"]), sf(a["LF"], "{:.3f}"),
                sf(a["n_teach"], "{:.0f}"), sf(a["n_eval"], "{:.0f}"),
                sf(a["stopped_frac"]),
                sf(a["B"]), sf(a["C"]), sf(a["mca_C"]),
                sf(a["D"]), sf(a["E"]), sf(a["mca_E"])))
            print(f"  {theta}×{s}: B={sf(a['B'])} C={sf(a['C'])} E={sf(a['E'])} "
                  f"MCA_E={sf(a['mca_E'])} ν={sf(a['nu'],'{:.2f}')} "
                  f"#T={sf(a['n_teach'],'{:.0f}')} stop={a['stopped_frac']}",
                  file=sys.stderr)

    # State + Mastery
    lines.append("\n## State + Mastery\n\n")
    lines.append("| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | MPG | BdgBlk |\n")
    lines.append("|---|-----------|---|---|---------|----|----|-----|-------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            gap = round(r["tau"] - r["nu"], 3) if r["tau"] and r["nu"] else None
            lines.append("| {} | {} | {} | {} | **{}** | {} | {} | {} | {} |\n".format(
                theta, s,
                sf(r["tau"], "{:.3f}"), sf(r["nu"], "{:.3f}"),
                sf(gap, "{:+.3f}"), sf(r["gg"], "{:.3f}"),
                sf(r["otr"], "{:.3f}"), sf(r["mpg"], "{:.4f}"),
                sf(r["n_blocked"], "{:.0f}")))

    # Actionability
    lines.append("\n## Closed-Loop Actionability\n\n")
    lines.append("| θ | vs | ERCR | micro_PCR |\n")
    lines.append("|---|---|------|----------|\n")
    for theta in ["safe", "shiny"]:
        mixed = [x for x in all_r if x["theta"] == theta and x["strategy"] == "mixed_random"][0]
        cct = [x for x in all_r if x["theta"] == theta and x["strategy"] == "cct_v3"][0]
        ercrs, pcrs = [], []
        for i in range(min(len(mixed["_ep_params"]), len(cct["_ep_params"]))):
            if mixed["_ep_params"][i] and cct["_ep_params"][i]:
                ercrs.append(episode_realization_change_rate(
                    cct["_ep_params"][i], mixed["_ep_params"][i]))
            pcrs.append(micro_policy_change_rate(
                cct["_micro"][i], mixed["_micro"][i]))
        lines.append("| {} | cct_v3 vs mixed | {} | {} |\n".format(
            theta,
            sf(round(np.mean(ercrs), 3) if ercrs else None),
            sf(round(np.mean(pcrs), 3) if pcrs else None)))

        tic = [x for x in all_r if x["theta"] == theta and x["strategy"] == "tic_heavy"][0]
        ercrs2, pcrs2 = [], []
        for i in range(min(len(tic["_ep_params"]), len(cct["_ep_params"]))):
            if tic["_ep_params"][i] and cct["_ep_params"][i]:
                ercrs2.append(episode_realization_change_rate(
                    cct["_ep_params"][i], tic["_ep_params"][i]))
            pcrs2.append(micro_policy_change_rate(
                cct["_micro"][i], tic["_micro"][i]))
        lines.append("| {} | cct_v3 vs tic_heavy | {} | {} |\n".format(
            theta,
            sf(round(np.mean(ercrs2), 3) if ercrs2 else None),
            sf(round(np.mean(pcrs2), 3) if pcrs2 else None)))

    # Subtype distribution
    lines.append("\n## Phase A Subtype Distribution\n\n")
    lines.append("| θ | Curriculum | Subtypes (top 3) |\n")
    lines.append("|---|-----------|------------------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            all_subs = [st for seeds in r["_subtypes_a"] for st in seeds]
            if all_subs:
                top3 = Counter(all_subs).most_common(3)
                top_str = ", ".join(f"{k}({v})" for k, v in top3)
            else:
                top_str = "—"
            lines.append(f"| {theta} | {s} | {top_str} |\n")

    with open(out / "cct_v3_report.md", "w") as f:
        f.writelines(lines)
    print(f"\nReport -> results/cct_v3_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
