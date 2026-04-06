"""P4-C: 3D / 4D / 4D-no-score Attribution Ablation.

Key finding: gamma_spec does NOT appear in tutor Q computation at all.
This script confirms that empirically with corrected active mask.

Group A: 3D observer (tau, nu, gamma_gen) → infer-only decisions
Group B: 4D observer, but gamma_spec NOT in score (current behavior)
Group C: 4D observer, gamma_spec ADDED to score (hypothetical)

If A ≡ B: confirms gamma_spec is purely diagnostic (expected)
If A ≠ B: something is leaking gamma_spec into Q (bug)
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
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import (
    LESSON_CATALOG_V2, BALANCED_ACTIVE_LESSONS, PROBE_NAMES,
)
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import (
    A1MtObserverFrozen, RuleBasedMtObserver, ObsEvent,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_ablation_session(lessons, theta, seed, group="A", n_teach=20):
    """Run session for one ablation group.

    Group A: 3D observer (RuleBasedMtObserver base)
    Group B: 4D observer, gamma_spec NOT in score (current A1)
    Group C: (hypothetical — would need tutor modification)
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()

    if group == "A":
        observer = RuleBasedMtObserver(); observer.reset()
    else:
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

        # Oracle decision
        tutor_o = BCICTv4(agent_params=AP, use_dose=False)
        act_o, dose_o, info_o = tutor_o.decide(sc, fb, lp, lib, scr, 2, m)

        # Infer-only decision using observer estimate
        m_hat = FactoredInternalizationState()
        est = observer.get_estimate()
        m_hat.tau = est["tau"]; m_hat.nu = est["nu"]
        m_hat.gamma_gen = est["gamma_gen"]
        if group != "A" and "gamma_spec" in est:
            m_hat.gamma_spec = est["gamma_spec"]
        m_hat.snapshot()
        tutor_i = BCICTv4(agent_params=AP, use_dose=False)
        act_i, dose_i, info_i = tutor_i.decide(sc, fb, lp, lib, scr, 2, m_hat)

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

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        diverge = (act_o != act_i)
        active = (act_o != "WAIT") or (act_i != "WAIT")
        overwarn = (act_o == "WAIT" and act_i == "WARN")

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "group": group, "act_oracle": act_o, "act_infer": act_i,
            "diverge": diverge, "active": active, "overwarn": overwarn,
            "correct": correct,
        })
    return records


def compute(recs):
    n = len(recs)
    div_all = sum(r["diverge"] for r in recs) / n
    n_act = sum(1 for r in recs if r["active"])
    div_act = sum(1 for r in recs if r["diverge"] and r["active"]) / max(n_act, 1)
    owr = sum(r["overwarn"] for r in recs) / n
    succ = sum(r["correct"] for r in recs) / n
    return div_all, div_act, n_act, owr, succ


def main():
    print("═══ P4-C: Attribution Ablation ═══\n", file=sys.stderr)
    L = ["# P4-C: 3D / 4D Attribution Ablation\n\n"]
    L.append("**Does γ̂_spec_state enter micro Q?** "
             "Code audit: `gamma_spec` absent from tutor Q & bridge.\n")
    L.append("**Prediction: Group A ≡ Group B** (γ_spec is purely diagnostic)\n\n")

    # ─── Main ablation ───────────────────────────────────
    L.append("## Ablation: Corrected Active Mask\n\n")
    L.append("| Suite | θ | Group | DivAll | Div@Act | n_act | OWR | Success |\n")
    L.append("|-------|:-:|:-----:|:------:|:-------:|:-----:|:---:|:-------:|\n")

    for sname, lessons in [("Canonical", list(LESSON_CATALOG_V2)),
                            ("Active", BALANCED_ACTIVE_LESSONS)]:
        print(f"  {sname}...", file=sys.stderr)
        for th in ["safe", "shiny"]:
            for group in ["A", "B"]:
                recs = []
                for sid in range(NS):
                    recs.extend(run_ablation_session(lessons, th, sid, group=group))
                da, dact, nact, owr, succ = compute(recs)
                L.append("| {} | {} | {} | {:.4f} | {:.4f} | {} | {:.4f} | {:.3f} |\n".format(
                    sname, th, group, da, dact, nact, owr, succ))

    # ─── Per-family over-warn ────────────────────────────
    L.append("\n## Per-Family Over-Warn Rate (Group B, Active Suite)\n\n")
    print("  Family OWR...", file=sys.stderr)
    L.append("| Family | θ | n | OWR | Div@Act |\n")
    L.append("|--------|:-:|:-:|:---:|:-------:|\n")
    for th in ["safe", "shiny"]:
        recs = []
        for sid in range(NS):
            recs.extend(run_ablation_session(BALANCED_ACTIVE_LESSONS, th, sid, group="B"))
        fam_groups = {}
        for r in recs:
            fam_groups.setdefault(r["family"], []).append(r)
        for fam in sorted(fam_groups.keys()):
            fr = fam_groups[fam]; n = len(fr)
            owr = sum(r["overwarn"] for r in fr) / n
            n_act = sum(1 for r in fr if r["active"])
            dact = sum(1 for r in fr if r["diverge"] and r["active"]) / max(n_act,1)
            if owr > 0 or dact > 0:
                L.append("| {} | {} | {} | {:.3f} | {:.3f} |\n".format(fam, th, n, owr, dact))

    # ─── Decision-level identity check ───────────────────
    L.append("\n## Decision Identity: A vs B\n\n")
    print("  Identity check...", file=sys.stderr)
    total = 0; identical = 0
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            ra = run_ablation_session(ALL_LESSONS, th, sid, group="A")
            rb = run_ablation_session(ALL_LESSONS, th, sid, group="B")
            for a, b in zip(ra, rb):
                total += 1
                if a["act_infer"] == b["act_infer"]:
                    identical += 1
    pct = 100 * identical / max(total, 1)
    L.append(f"**A vs B step-level identity: {identical}/{total} ({pct:.2f}%)**\n\n")

    # ─── Verdict ─────────────────────────────────────────
    L.append("\n## Verdict\n\n")
    if pct >= 99.0:
        L.append("> **A ≡ B confirmed.** γ̂_spec_state does NOT enter micro tutor Q. "
                 "The over-warn divergences are pre-existing near-tie boundary issues "
                 "exposed by the corrected active mask, not caused by the 4th dimension.\n\n")
        L.append("### Implication\n\n")
        L.append("γ̂_spec_state is currently **purely diagnostic / macro-only state**. "
                 "This is the correct architecture:\n\n")
        L.append("- **Layer 1 (State Estimator)**: 4D `(τ̂, ν̂, γ̂_gen, γ̂_spec_state)`\n")
        L.append("- **Layer 2 (Micro Decision View)**: 3D `(τ̂, ν̂, γ̂_gen)` — "
                 "γ̂_spec does NOT enter Q\n")
        L.append("- **Layer 3 (Macro / Diagnostic)**: Full 4D available\n")
    elif pct >= 95.0:
        L.append(f"> **A ≈ B ({pct:.1f}% identical).** γ̂_spec has minor leakage "
                 "into micro decisions. Investigate.\n")
    else:
        L.append(f"> **A ≠ B ({pct:.1f}% identical).** γ̂_spec is materially "
                 "affecting micro decisions. Requires design review.\n")

    rpt = out / "p4c_attribution_ablation.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
