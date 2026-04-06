"""Online Micro-Hybrid: tutor uses m_hybrid for REAL decisions.

This is the first time the observer's estimates actually affect teaching.
Previous experiments were shadow-only (observer watched but didn't drive).
Now: tutor.decide(... m_hybrid) → agent experiences hybrid-driven actions.

Exp 1: α sweep on canonical families (success, OTR, intervention, divergence)
Exp 2: Hidden temptation (tempt=0/0.6/1.0 × α)
Exp 3: No-tutor transfer (last 4 steps tutor-free)
"""
import sys
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
from src.teachers.internalization_observer import A1MtObserver, ObsEvent
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


def run_online_session(lessons, theta, seed, n_teach=16, alpha=0.0,
                       hidden_tempt=0.0, n_transfer=4):
    """Run ONLINE session: tutor decides using m_hybrid.

    Key difference from shadow experiments:
      - tutor.decide() receives m_hybrid, not m_true
      - agent experiences consequences of hybrid-driven tutoring
      - last n_transfer steps run without tutor (dose=0)
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserver(); observer.reset()
    records = []
    total_steps = n_teach + n_transfer

    for step in range(total_steps):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / max(total_steps, 1) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
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

        # ─── ORACLE tutor decides with true m (for comparison) ───
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        # ─── HYBRID tutor decides with m_hybrid ─────────────────
        is_transfer = (step >= n_teach)
        if is_transfer:
            # Transfer phase: no tutor intervention
            dose_actual = 0.0
            a_actual = a_oracle  # action doesn't matter, dose=0
        else:
            # Build m_hybrid
            m_hybrid = m.copy()
            m_hybrid.tau = (1 - alpha) * m.tau + alpha * observer.tau_hat
            m_hybrid.nu = (1 - alpha) * m.nu + alpha * observer.nu_hat
            m_hybrid.gamma_gen = (1 - alpha) * m.gamma_gen + alpha * observer.gamma_gen_hat
            a_actual, dose_actual, info_actual = tutor.decide(sc, fb, lp, lib, scr, 2, m_hybrid)

        # ─── Agent acts with whatever dose the HYBRID tutor chose ───
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        risky_branch = 1 - sc.oracle_safe_branch_id
        tempt_scores = [0.0, 0.0]
        tempt_scores[risky_branch] = hidden_tempt

        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b) + tempt_scores[0])
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a) + tempt_scores[1])
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose_actual > 0; follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev
        bn = ep.subtype == "beneficial_novelty" and correct

        # ─── Agent internalizes based on ACTUAL dose experienced ───
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if not has_self_ev: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()

        # ─── Observer updates (always using actual events) ───
        probes = all_probes(m, AP, theta) if step % 2 == 0 else {}
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose_actual, warned=warned, follow_warn=follow_warn,
            warn_correct=(warned and risk > 0.25), warn_wrong=(warned and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc, beneficial_novelty=bn,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        records.append({
            "step": step, "phase": "transfer" if is_transfer else "teach",
            "alpha": alpha, "theta": theta,
            "correct": correct, "dose": dose_actual,
            "dose_oracle": dose_oracle,
            "action_diverge": (a_actual != a_oracle) if not is_transfer else False,
            "warned": warned, "self_disc": self_disc,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            "hidden_tempt": hidden_tempt,
        })
    return records


def summarize(records, phase="teach"):
    sub = [r for r in records if r["phase"] == phase]
    if not sub: return {}
    success = sum(r["correct"] for r in sub) / len(sub)
    dose_rate = sum(1 for r in sub if r["dose"] > 0) / len(sub)
    diverge = sum(r.get("action_diverge", False) for r in sub) / len(sub)
    return {
        "success": round(success, 4),
        "dose_rate": round(dose_rate, 4),
        "divergence": round(diverge, 4),
        "n": len(sub),
    }


def main():
    print("═══ Online Micro-Hybrid ═══\n", file=sys.stderr)
    L = ["# Online Micro-Hybrid Results\n\n"]

    # ─── Exp 1: α sweep on canonical ────────────────────
    L.append("## Exp 1: α Sweep — Canonical Mixed-Family\n\n")
    L.append("| α | θ | Success | Dose Rate | Action Diverge | Transfer Success |\n")
    L.append("|:-:|:-:|:-------:|:---------:|:--------------:|:----------------:|\n")
    print("Exp 1: α sweep...", file=sys.stderr)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for th in ["safe", "shiny"]:
            all_recs = []
            for sid in range(NS):
                all_recs.extend(run_online_session(ALL_LESSONS, th, sid, alpha=alpha))
            teach = summarize(all_recs, "teach")
            trans = summarize(all_recs, "transfer")
            L.append("| {} | {} | {} | {} | {} | {} |\n".format(
                alpha, th, teach.get("success", 0), teach.get("dose_rate", 0),
                teach.get("divergence", 0), trans.get("success", 0)))
            print(f"  α={alpha} θ={th}: success={teach.get('success',0)} "
                  f"diverge={teach.get('divergence',0)} transfer={trans.get('success',0)}",
                  file=sys.stderr)

    # ─── Exp 2: Hidden temptation × α ───────────────────
    L.append("\n## Exp 2: Hidden Temptation × α\n\n")
    L.append("| Tempt | α | θ | Success | Diverge | Transfer |\n")
    L.append("|:-----:|:-:|:-:|:-------:|:-------:|:--------:|\n")
    print("\nExp 2: Hidden temptation...", file=sys.stderr)
    for ht in [0.0, 0.6, 1.0]:
        for alpha in [0.0, 1.0]:
            for th in ["safe", "shiny"]:
                recs = []
                for sid in range(NS):
                    recs.extend(run_online_session(ALL_LESSONS, th, sid,
                                                   alpha=alpha, hidden_tempt=ht))
                teach = summarize(recs, "teach")
                trans = summarize(recs, "transfer")
                L.append("| {} | {} | {} | {} | {} | {} |\n".format(
                    ht, alpha, th, teach.get("success", 0),
                    teach.get("divergence", 0), trans.get("success", 0)))

    # ─── Exp 3: Per-family breakdown at α=1 vs α=0 ─────
    L.append("\n## Exp 3: Per-Family Comparison (α=0 vs α=1)\n\n")
    L.append("| Family | α=0 Succ | α=1 Succ | α=0 Transfer | α=1 Transfer |\n")
    L.append("|--------|:--------:|:--------:|:------------:|:------------:|\n")
    print("\nExp 3: Per-family...", file=sys.stderr)
    fam_results = {}
    for alpha in [0.0, 1.0]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_online_session(ALL_LESSONS, th, sid, alpha=alpha))
            key = f"{th}_a{alpha}"
            fam_results[key] = {
                "teach": summarize(recs, "teach"),
                "transfer": summarize(recs, "transfer"),
            }
    for th in ["safe", "shiny"]:
        a0 = fam_results[f"{th}_a0.0"]; a1 = fam_results[f"{th}_a1.0"]
        L.append("| {} | {} | {} | {} | {} |\n".format(
            th,
            a0["teach"].get("success", 0), a1["teach"].get("success", 0),
            a0["transfer"].get("success", 0), a1["transfer"].get("success", 0)))

    rpt = out / "online_micro_hybrid_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
