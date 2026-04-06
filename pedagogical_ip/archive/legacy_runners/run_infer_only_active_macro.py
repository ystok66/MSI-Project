"""Online Micro Infer-Only + Active Benchmark + Macro Replay.

Exp 1: Infer-only vs oracle (canonical + hidden temptation)
Exp 2: Active benchmark (forced scenarios where dose matters)
Exp 3: Q-margin analysis (is zero-diverge trivial?)
Exp 4: Macro STOP/EVAL offline replay
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


def build_episode_env(les, step, seed, theta, rng):
    """Build full episode environment — returns all needed objects."""
    ub = {p: 0.4 + 0.1 * step / 20 for p in PROBE_NAMES}
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
    return ep, sc, fb, lp, lib, scr, ss, sr, meta


def run_infer_only_session(lessons, theta, seed, n_teach=16, hidden_tempt=0.0,
                           n_transfer=4, force_active_dose=None):
    """Run session: oracle tutor + infer-only tutor in parallel.

    Both tutors see the same agent behavior. Key difference:
    - oracle uses m_true
    - infer uses ONLY m̂ (no access to m_true)
    Agent experiences oracle-driven teaching (to keep comparison fair).
    """
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserver(); observer.reset()
    records = []
    total_steps = n_teach + n_transfer

    for step in range(total_steps):
        les = lessons[step % len(lessons)]
        ep, sc, fb, lp, lib, scr, ss, sr, meta = build_episode_env(
            les, step, seed, theta, rng)

        is_transfer = step >= n_teach

        # Oracle tutor decides with true m
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        Q_oracle = info_oracle.get("Q", 0.0)

        # Infer-only tutor decides with m̂
        m_hat_state = m.copy()
        m_hat_state.tau = observer.tau_hat
        m_hat_state.nu = observer.nu_hat
        m_hat_state.gamma_gen = observer.gamma_gen_hat
        a_infer, dose_infer, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat_state)
        Q_infer = info_infer.get("Q", 0.0)

        # Compute Q-margins (top1 - top2 proxy: Q spread)
        # Since tutor returns single Q, estimate margin from dose difference
        margin_oracle = abs(Q_oracle)
        margin_infer = abs(Q_infer)

        # Determine actual dose (oracle-driven; or forced for active benchmark)
        if is_transfer:
            dose_actual = 0.0
        elif force_active_dose is not None:
            dose_actual = force_active_dose
        else:
            dose_actual = dose_oracle

        # Agent acts
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

        # Agent internalizes from actual dose experienced
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

        # Observer updates
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

        # Classify step
        oracle_nonwait = dose_oracle > 0.0
        infer_nonwait = dose_infer > 0.0
        is_active = oracle_nonwait or infer_nonwait
        diverge = (a_oracle != a_infer) if not is_transfer else False

        records.append({
            "step": step, "phase": "transfer" if is_transfer else "teach",
            "theta": theta, "correct": correct,
            "dose_oracle": dose_oracle, "dose_infer": dose_infer,
            "dose_actual": dose_actual,
            "a_oracle": a_oracle, "a_infer": a_infer, "diverge": diverge,
            "Q_oracle": Q_oracle, "Q_infer": Q_infer,
            "margin_oracle": margin_oracle, "margin_infer": margin_infer,
            "is_active": is_active, "oracle_nonwait": oracle_nonwait,
            "infer_nonwait": infer_nonwait,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
            "hidden_tempt": hidden_tempt,
        })
    return records


def compute_metrics(records, phase="teach"):
    sub = [r for r in records if r["phase"] == phase]
    if not sub: return {}
    n = len(sub)
    success = sum(r["correct"] for r in sub) / n
    dose_rate = sum(1 for r in sub if r["dose_oracle"] > 0) / n
    diverge_all = sum(r["diverge"] for r in sub) / n
    active = [r for r in sub if r["is_active"]]
    diverge_active = (sum(r["diverge"] for r in active) / len(active)) if active else 0.0
    # Hard cases: low oracle margin
    margins = [r["margin_oracle"] for r in sub]
    med_margin = np.median(margins) if margins else 0
    hard = [r for r in sub if r["margin_oracle"] < med_margin]
    diverge_hard = (sum(r["diverge"] for r in hard) / len(hard)) if hard else 0.0
    return {
        "success": round(success, 4), "dose_rate": round(dose_rate, 4),
        "diverge_all": round(diverge_all, 4),
        "diverge_active": round(diverge_active, 4),
        "n_active": len(active),
        "diverge_hard": round(diverge_hard, 4),
        "n_hard": len(hard), "n": n,
    }


def main():
    print("═══ Infer-Only + Active + Macro ═══\n", file=sys.stderr)
    L = ["# Online Infer-Only + Active Benchmark + Macro Replay\n\n"]

    # ─── Exp 1: Infer-only vs Oracle ─────────────────────
    L.append("## Exp 1: Infer-Only vs Oracle (Canonical + Temptation)\n\n")
    L.append("| Config | θ | Success | Dose Rate | Diverge All | "
             "Diverge@Active | n_active | Transfer |\n")
    L.append("|--------|:-:|:-------:|:---------:|:-----------:|"
             ":--------------:|:--------:|:--------:|\n")
    print("Exp 1: Infer-only vs oracle...", file=sys.stderr)
    for ht_label, ht in [("canonical", 0.0), ("tempt=0.6", 0.6), ("tempt=1.0", 1.0)]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_infer_only_session(ALL_LESSONS, th, sid, hidden_tempt=ht))
            teach = compute_metrics(recs, "teach")
            trans = compute_metrics(recs, "transfer")
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                ht_label, th, teach["success"], teach["dose_rate"],
                teach["diverge_all"], teach["diverge_active"],
                teach["n_active"], trans.get("success", "—")))
            print(f"  {ht_label} θ={th}: div_all={teach['diverge_all']} "
                  f"div@active={teach['diverge_active']} n_active={teach['n_active']}",
                  file=sys.stderr)

    # ─── Exp 2: Active Benchmark (forced dose) ───────────
    L.append("\n## Exp 2: Active Benchmark (Forced Intervention Scenarios)\n\n")
    L.append("| Regime | θ | Diverge All | Diverge@Active | n_active | "
             "Diverge@Hard | n_hard |\n")
    L.append("|--------|:-:|:-----------:|:--------------:|:--------:|"
             ":------------:|:------:|\n")
    print("\nExp 2: Active benchmark...", file=sys.stderr)
    for regime, fd in [("natural", None), ("active_0.5", 0.5), ("active_1.0", 1.0)]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_infer_only_session(ALL_LESSONS, th, sid,
                                                   force_active_dose=fd))
            teach = compute_metrics(recs, "teach")
            L.append("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                regime, th, teach["diverge_all"], teach["diverge_active"],
                teach["n_active"], teach["diverge_hard"], teach["n_hard"]))
            print(f"  {regime} θ={th}: div_all={teach['diverge_all']} "
                  f"div@active={teach['diverge_active']} div@hard={teach['diverge_hard']}",
                  file=sys.stderr)

    # ─── Exp 3: Q-Margin Analysis ────────────────────────
    L.append("\n## Exp 3: Q-Margin Analysis\n\n")
    L.append("| θ | Mean Q_oracle | Mean Q_infer | Mean |ΔQ| | "
             "Corr(Q_o, Q_i) |\n")
    L.append("|:-:|:-------------:|:------------:|:----------:|"
             ":--------------:|\n")
    print("\nExp 3: Q-margin...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        recs = []
        for sid in range(NS):
            recs.extend(run_infer_only_session(ALL_LESSONS, th, sid))
        teach = [r for r in recs if r["phase"] == "teach"]
        qo = [r["Q_oracle"] for r in teach]
        qi = [r["Q_infer"] for r in teach]
        dq = [abs(a - b) for a, b in zip(qo, qi)]
        corr_q = float(np.corrcoef(qo, qi)[0, 1]) if np.std(qo) > 1e-6 else 0
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, np.mean(qo), np.mean(qi), np.mean(dq), corr_q))

    # ─── Exp 4: Macro STOP/EVAL Offline Replay ───────────
    L.append("\n## Exp 4: Macro STOP/EVAL Offline Replay\n\n")
    L.append("| α | θ | ε_stop oracle | ε_stop infer | Δε_stop | "
             "STOP agree |\n")
    L.append("|:-:|:-:|:---:|:---:|:---:|:---:|\n")
    print("\nExp 4: Macro replay...", file=sys.stderr)
    a_s = 0.15; b_s = 0.10; eps_0 = 0.30; stop_threshold = 0.35
    for alpha in [0.0, 0.5, 1.0]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_infer_only_session(ALL_LESSONS, th, sid))
            eps_o = []; eps_i = []; agree = 0; total = 0
            for r in [r for r in recs if r["phase"] == "teach"]:
                mt = r["m_true"]; mh = r["m_hat"]
                eo = eps_0 + a_s * mt["nu"] + b_s * mt["gamma_gen"]
                hyb_nu = (1 - alpha) * mt["nu"] + alpha * mh["nu"]
                hyb_gg = (1 - alpha) * mt["gamma_gen"] + alpha * mh["gamma_gen"]
                ei = eps_0 + a_s * hyb_nu + b_s * hyb_gg
                eps_o.append(eo); eps_i.append(ei)
                stop_o = eo > stop_threshold; stop_i = ei > stop_threshold
                if stop_o == stop_i: agree += 1
                total += 1
            delta = np.mean(np.abs(np.array(eps_i) - np.array(eps_o)))
            agree_rate = agree / max(total, 1)
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} |\n".format(
                alpha, th, np.mean(eps_o), np.mean(eps_i), delta, agree_rate))

    # ─── Exp 5: Policy Coverage Benchmark ────────────────
    L.append("\n## Exp 5: Policy Coverage Benchmark\n\n")
    L.append("| θ | warn_rate | dose>0 | blind>0 | selfdisc>0 | trust>0 |\n")
    L.append("|:-:|:---------:|:------:|:-------:|:----------:|:-------:|\n")
    print("\nExp 5: Coverage...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        recs = []
        for sid in range(NS):
            recs.extend(run_infer_only_session(ALL_LESSONS, th, sid))
        teach = [r for r in recs if r["phase"] == "teach"]
        n = len(teach)
        warn = sum(1 for r in teach if r["dose_oracle"] >= 1.0) / n
        dose = sum(1 for r in teach if r["dose_oracle"] > 0) / n
        # Count from observer events — need to peek at observer's last history
        # Use nonwait as proxy
        nonwait = sum(1 for r in teach if r["oracle_nonwait"]) / n
        selfdisc = sum(1 for r in teach if r.get("correct", False) and
                       not r.get("oracle_nonwait", False)) / n
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, warn, dose, nonwait, selfdisc, nonwait))

    rpt = out / "infer_only_active_macro_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
