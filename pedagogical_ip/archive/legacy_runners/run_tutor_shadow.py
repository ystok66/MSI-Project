"""Phase 7 Track B — Tutor Shadow Comparison + Edge-Case Tests.

Exp 1: Active-family shadow comparison
  BCICTv4 (base) vs +EPU vs +belief-horizon vs +EIG vs +all
  Families: self_discovery_needed, warn_rescue, boundary_obs,
            beneficial_novelty, false_suppression_cost

Exp 2: Edge-case necessity tests
  belief-horizon p_self on noisy/stubborn early-reveal episodes
  EIG observation on boundary_obs disambiguation

Metrics: dose distribution, action agreement, overteach, probes
"""
import sys, os
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

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
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 20

# Target subtypes for shadow comparison
TARGET_SUBTYPES = [
    "self_discovery_needed", "warn_rescue", "boundary_obs",
    "beneficial_novelty", "false_suppression_cost",
]

# Map lesson subtypes
LESSON_BY_SUBTYPE = {}
for les in LESSON_CATALOG_V2:
    LESSON_BY_SUBTYPE.setdefault(les.subtype, []).append(les)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def make_tutor(epu=False, belief=False, eig=False, eta=0.5):
    t = BCICTv4(agent_params=AP)
    t.use_epu_shadow = epu
    t.use_belief_horizon_pself = belief
    t.use_eig_observation = eig
    t.eta_belief = eta
    return t


ARM_DEFS = {
    "base":    {"epu": False, "belief": False, "eig": False},
    "EPU":     {"epu": True,  "belief": False, "eig": False},
    "belief":  {"epu": False, "belief": True,  "eig": False},
    "EIG":     {"epu": False, "belief": False, "eig": True},
    "all3":    {"epu": True,  "belief": True,  "eig": True},
}


def run_episode(tutor, th, seed, lesson, mastery_override=None):
    """Run a single teaching episode and return tutor decision info."""
    rng = np.random.default_rng(seed * 100 + abs(hash(th)) % 100)
    m = FactoredInternalizationState(); m.snapshot()
    ub = mastery_override or {p: 0.4 for p in PROBE_NAMES}
    et = generate_episode_from_lesson_v2(lesson, seed, th, ub, rng)
    ep, spec, gm, cfg_e, meta, sc = et
    fb, ww = apply_fix(meta, sc)
    fv = np.full_like(fb, 0.3)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    for _ in range(3):
        for r in range(gm.height):
            for c in range(gm.width):
                if gm.cell_types[r, c] == CellType.WALL:
                    continue
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
    lib = BranchConceptLibrary()
    scr = BranchScorerProbe(lr=0.05, l2=0.01)
    ss = summarize_branch(sc.safe_cells, fb, fv, lp)
    sr = summarize_branch(sc.risky_cells, fb, fv, lp)
    lib.update("safe", ss); lib.update("risky", sr)
    scr.update(build_scorer_input(ss, lib), 1.0)
    scr.update(build_scorer_input(sr, lib), 0.0)

    action, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

    return {
        "action": action,
        "dose": dose,
        "info": info,
        "subtype": lesson.subtype,
        "p_self_geom": round(estimate_self_discovery_prob(
            getattr(sc, 'commit_depth', 3), getattr(sc, 'reveal_depth', 2)), 4),
        "risk": getattr(sc, 'risk_level', 0.3),
        "tempt": getattr(sc, 'temptation_strength', 0.0),
    }


def main():
    print("═══ Phase 7 Track B: Tutor Shadow Comparison ═══\n", file=sys.stderr)
    L = ["# Tutor Shadow Comparison & Edge-Case Tests\n\n"]
    L.append(f"> Seeds: {NS} | Families: {len(TARGET_SUBTYPES)}\n\n")

    # ─── Exp 1: Active-Family Shadow Comparison ──────────
    L.append("## Exp 1: Active-Family Shadow Comparison\n\n")

    # Collect all lessons matching target subtypes
    target_lessons = []
    for st in TARGET_SUBTYPES:
        matches = LESSON_BY_SUBTYPE.get(st, [])
        if matches:
            target_lessons.append(matches[0])

    if not target_lessons:
        # Fallback: use all lessons
        target_lessons = LESSON_CATALOG_V2[:5]
        L.append("> Note: no exact subtype matches; using first 5 lessons\n\n")

    # Per-arm action distribution
    L.append("### Dose Distribution by Arm\n\n")
    L.append("| Arm | WAIT | SOFT | WARN | Mean Dose |\n")
    L.append("|-----|:----:|:----:|:----:|:---------:|\n")

    arm_results = {}
    for arm_name, flags in ARM_DEFS.items():
        wait = soft = warn = 0
        doses = []
        all_infos = []
        for th in ["safe", "shiny"]:
            for les in target_lessons:
                for sid in range(NS):
                    tutor = make_tutor(**flags)
                    r = run_episode(tutor, th, sid, les)
                    doses.append(r["dose"])
                    if r["action"] == "WAIT": wait += 1
                    elif r["action"] == "SOFT": soft += 1
                    else: warn += 1
                    all_infos.append(r)
        total = wait + soft + warn
        arm_results[arm_name] = all_infos
        L.append("| {} | {:.0%} | {:.0%} | {:.0%} | {:.3f} |\n".format(
            arm_name,
            wait / total, soft / total, warn / total,
            np.mean(doses)))
        print(f"  {arm_name}: WAIT={wait/total:.0%} SOFT={soft/total:.0%} "
              f"WARN={warn/total:.0%} dose={np.mean(doses):.3f}", file=sys.stderr)

    # EPU shadow agreement
    L.append("\n### EPU Shadow Agreement\n\n")
    L.append("| Subtype | EPU agrees with base | EPU action | Base action |\n")
    L.append("|---------|:----:|:------:|:------:|\n")

    epu_infos = arm_results.get("EPU", [])
    base_infos = arm_results.get("base", [])
    by_subtype = {}
    for ei, bi in zip(epu_infos, base_infos):
        st = ei["subtype"]
        by_subtype.setdefault(st, []).append({
            "epu_agrees": ei["info"].get("epu_shadow", {}).get("agrees", True),
            "epu_action": ei["info"].get("epu_shadow", {}).get("action", "?"),
            "base_action": bi["action"],
        })

    for st, items in by_subtype.items():
        agree_rate = sum(1 for x in items if x["epu_agrees"]) / max(len(items), 1)
        epu_actions = {}
        base_actions = {}
        for x in items:
            epu_actions[x["epu_action"]] = epu_actions.get(x["epu_action"], 0) + 1
            base_actions[x["base_action"]] = base_actions.get(x["base_action"], 0) + 1
        epu_mode = max(epu_actions, key=epu_actions.get) if epu_actions else "?"
        base_mode = max(base_actions, key=base_actions.get) if base_actions else "?"
        L.append(f"| {st} | {agree_rate:.0%} | {epu_mode} | {base_mode} |\n")

    # Belief-horizon p_self comparison
    L.append("\n### Belief-Horizon p_self Comparison\n\n")
    L.append("| Subtype | p_geom (mean) | p_hybrid (mean) | Δ (mean) |\n")
    L.append("|---------|:---:|:---:|:---:|\n")

    belief_infos = arm_results.get("belief", [])
    by_subtype_bh = {}
    for bi in belief_infos:
        st = bi["subtype"]
        bh = bi["info"].get("belief_horizon", {})
        if bh:
            by_subtype_bh.setdefault(st, []).append(bh)

    for st, items in by_subtype_bh.items():
        p_geom_mean = np.mean([x["p_geom"] for x in items])
        p_hybrid_mean = np.mean([x["p_hybrid"] for x in items])
        delta_mean = np.mean([x["delta"] for x in items])
        L.append(f"| {st} | {p_geom_mean:.4f} | {p_hybrid_mean:.4f} | {delta_mean:+.4f} |\n")

    # EIG observation values
    L.append("\n### EIG Observation Values\n\n")
    L.append("| Subtype | I(A;θ) mean | Wait boost mean |\n")
    L.append("|---------|:---:|:---:|\n")

    eig_infos = arm_results.get("EIG", [])
    by_subtype_eig = {}
    for ei in eig_infos:
        st = ei["subtype"]
        ev = ei["info"].get("eig_observation", {})
        if ev:
            by_subtype_eig.setdefault(st, []).append(ev)

    for st, items in by_subtype_eig.items():
        mi_mean = np.mean([x["I_A_theta"] for x in items])
        wb_mean = np.mean([x["wait_boost"] for x in items])
        L.append(f"| {st} | {mi_mean:.6f} | {wb_mean:.4f} |\n")

    # ─── Exp 2: Edge-Case Tests ──────────────────────────
    L.append("\n## Exp 2: Edge-Case Necessity Tests\n\n")

    # 2a: Noisy/stubborn self-discovery
    L.append("### 2a: Noisy Self-Discovery (varied κ, ν)\n\n")
    L.append("| Condition | p_geom | p_hybrid(η=0.25) | p_hybrid(η=0.5) | p_hybrid(η=0.75) |\n")
    L.append("|-----------|:------:|:---:|:---:|:---:|\n")

    conditions = [
        {"label": "standard (κ=0.5, ν=0.3)", "kappa": 0.5, "nu": 0.3},
        {"label": "risk-aware (κ=1.5, ν=0.2)", "kappa": 1.5, "nu": 0.2},
        {"label": "stubborn (κ=0.3, ν=0.7)", "kappa": 0.3, "nu": 0.7},
        {"label": "dependent (κ=0.8, ν=0.9)", "kappa": 0.8, "nu": 0.9},
        {"label": "fresh (κ=0.5, ν=0.0)", "kappa": 0.5, "nu": 0.0},
    ]

    sd_lessons = LESSON_BY_SUBTYPE.get("self_discovery_needed",
                 LESSON_BY_SUBTYPE.get("self_discovery_teach", LESSON_CATALOG_V2[:1]))

    for cond in conditions:
        p_geoms = []; p_h25 = []; p_h50 = []; p_h75 = []
        for les in sd_lessons:
            for sid in range(NS):
                for th in ["safe"]:
                    for eta in [0.25, 0.5, 0.75]:
                        tutor = make_tutor(belief=True, eta=eta)
                        # Modify m to match condition
                        rng = np.random.default_rng(sid * 100)
                        m = FactoredInternalizationState()
                        m.kappa = cond["kappa"]
                        m.nu = cond["nu"]
                        m.snapshot()
                        ub = {p: 0.4 for p in PROBE_NAMES}
                        et = generate_episode_from_lesson_v2(les, sid, th, ub, rng)
                        ep, spec, gm, cfg_e, meta, sc = et
                        fb, ww = apply_fix(meta, sc)
                        fv = np.full_like(fb, 0.3)
                        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
                        for _ in range(3):
                            for r in range(gm.height):
                                for c in range(gm.width):
                                    if gm.cell_types[r, c] == CellType.WALL:
                                        continue
                                    z = fb[r, c]
                                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
                        lib = BranchConceptLibrary()
                        scr = BranchScorerProbe(lr=0.05, l2=0.01)
                        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
                        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
                        lib.update("safe", ss); lib.update("risky", sr)
                        scr.update(build_scorer_input(ss, lib), 1.0)
                        scr.update(build_scorer_input(sr, lib), 0.0)
                        _, _, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)
                        bh = info.get("belief_horizon", {})
                        if bh:
                            if eta == 0.25:
                                p_geoms.append(bh["p_geom"])
                                p_h25.append(bh["p_hybrid"])
                            elif eta == 0.5:
                                p_h50.append(bh["p_hybrid"])
                            else:
                                p_h75.append(bh["p_hybrid"])

        pg = np.mean(p_geoms) if p_geoms else 0
        h25 = np.mean(p_h25) if p_h25 else 0
        h50 = np.mean(p_h50) if p_h50 else 0
        h75 = np.mean(p_h75) if p_h75 else 0
        L.append(f"| {cond['label']} | {pg:.4f} | {h25:.4f} | {h50:.4f} | {h75:.4f} |\n")
        print(f"  {cond['label']}: p_geom={pg:.4f} h50={h50:.4f}", file=sys.stderr)

    # 2b: EIG on boundary_obs
    L.append("\n### 2b: EIG Observation — θ Disambiguation\n\n")
    L.append("| θ_true | q(safe) | I(A;θ) | Interpretation |\n")
    L.append("|--------|:-------:|:------:|----------------|\n")

    bo_lessons = LESSON_BY_SUBTYPE.get("boundary_obs",
                 LESSON_BY_SUBTYPE.get("verified_warn", LESSON_CATALOG_V2[:1]))

    for th_true in ["safe", "shiny"]:
        for q_safe in [0.3, 0.5, 0.7, 0.9]:
            eig_vals = []
            for les in bo_lessons:
                for sid in range(NS):
                    tutor = make_tutor(eig=True)
                    r = run_episode(tutor, th_true, sid, les)
                    ev = r["info"].get("eig_observation", {})
                    if ev:
                        eig_vals.append(ev["I_A_theta"])
            mi_mean = np.mean(eig_vals) if eig_vals else 0
            interp = "low info" if mi_mean < 0.01 else ("moderate info" if mi_mean < 0.1 else "high info")
            L.append(f"| {th_true} | {q_safe} | {mi_mean:.6f} | {interp} |\n")

    # ─── Summary ─────────────────────────────────────────
    L.append("\n## Summary & Verdicts\n\n")
    L.append("_Verdicts to be determined after reviewing results._\n")

    rpt = out / "micro_tutor_shadow_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
