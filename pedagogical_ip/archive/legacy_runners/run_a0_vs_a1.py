"""A0 vs A1 Shadow Observer comparison + p_self activation test.

Exp A: A0 vs A1 same matrix (longer-session, event ablation, STOP sensitivity, confidence)
Exp B: p_self activation test (gap=0/2/4+)
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
from src.teachers.internalization_observer import (
    RuleBasedMtObserver, A1MtObserver, ObsEvent,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 12
ALL_LESSONS = list(LESSON_CATALOG_V2)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach=6, observer=None, zero_pself=False):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    if observer is None:
        observer = RuleBasedMtObserver()
    observer.reset()
    records = []
    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / max(n_teach, 1) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
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
        lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe", ss); lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b)
        bar = BranchAttributes(safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a)
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose_oracle > 0; follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev
        bn = ep.subtype == "beneficial_novelty" and correct
        fs = ep.subtype == "false_suppression_cost" and not correct
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
        probes = all_probes(m, AP, theta) if step % 2 == 0 else {}
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose_oracle, warned=warned, follow_warn=follow_warn,
            warn_correct=(warned and risk > 0.25), warn_wrong=(warned and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=0.5 if zero_pself else p_self,
            risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc, false_suppression=fs, beneficial_novelty=bn,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        snap = observer.update(ev)
        m_hat = m.copy(); m_hat.tau = observer.tau_hat
        m_hat.nu = observer.nu_hat; m_hat.gamma_gen = observer.gamma_gen_hat
        a_infer, _, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat)
        rec = observer.to_log_record(step, ev, snap, a_oracle, a_infer,
            info_oracle.get("Q", 0), info_infer.get("Q", 0))
        rec["conf"] = observer.get_confidence()
        rec["p_self"] = 0.5 if zero_pself else p_self
        rec["dc_minus_dr"] = dc - dr
        records.append(rec)
    return records


def analyze(records):
    if not records:
        return {"tau": {"mae": 0, "corr": 0}, "nu": {"mae": 0, "corr": 0},
                "gamma_gen": {"mae": 0, "corr": 0}, "ADR": 0, "n": 0}
    def _m(tv, hv):
        if len(tv) < 2: return {"mae": 0, "corr": 0}
        mae = float(np.mean(np.abs(np.array(tv) - np.array(hv))))
        corr = float(np.corrcoef(tv, hv)[0, 1]) if np.std(tv) > 1e-6 else 0.0
        return {"mae": round(mae, 4), "corr": round(corr, 4)}
    tt = [r["m_true"]["tau"] for r in records if r.get("m_true")]
    th = [r["m_hat"]["tau"] for r in records if r.get("m_true")]
    nt = [r["m_true"]["nu"] for r in records if r.get("m_true")]
    nh = [r["m_hat"]["nu"] for r in records if r.get("m_true")]
    gt = [r["m_true"]["gamma_gen"] for r in records if r.get("m_true")]
    gh = [r["m_hat"]["gamma_gen"] for r in records if r.get("m_true")]
    adr = sum(1 for r in records if r["disagree"]) / max(len(records), 1)
    return {"tau": _m(tt, th), "nu": _m(nt, nh), "gamma_gen": _m(gt, gh),
            "ADR": round(adr, 3), "n": len(records)}


def main():
    print("═══ A0 vs A1 Shadow Observer ═══\n", file=sys.stderr)
    L = ["# A0 vs A1 Shadow Observer Comparison\n\n"]

    # ─── Exp A: Longer-session comparison ────────────────
    L.append("## Exp A: Longer-Session (A0 vs A1)\n\n")
    L.append("| Observer | Steps | MAE_τ | Corr_τ | MAE_ν | Corr_ν | MAE_γ | Corr_γ | ADR |\n")
    L.append("|----------|:-----:|:-----:|:------:|:-----:|:------:|:-----:|:------:|:---:|\n")
    print("Exp A: Longer sessions...", file=sys.stderr)
    for obs_name, obs_cls in [("A0", RuleBasedMtObserver), ("A1", A1MtObserver)]:
        for n_steps in [6, 12, 16, 20]:
            recs = []
            for th in ["safe", "shiny"]:
                for sid in range(NS):
                    recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=n_steps,
                                            observer=obs_cls()))
            m = analyze(recs)
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                obs_name, n_steps, m["tau"]["mae"], m["tau"]["corr"],
                m["nu"]["mae"], m["nu"]["corr"],
                m["gamma_gen"]["mae"], m["gamma_gen"]["corr"], m["ADR"]))
            print(f"  {obs_name} T={n_steps}: Corr_tau={m['tau']['corr']} "
                  f"Corr_nu={m['nu']['corr']} Corr_gamma={m['gamma_gen']['corr']}", file=sys.stderr)

    # ─── Exp B: p_self activation test ───────────────────
    L.append("\n## Exp B: p_self Activation Test (A1)\n\n")
    L.append("| p_self mode | MAE_ν | Corr_ν | Mean blind | Mean selfdisc |\n")
    L.append("|-------------|:-----:|:------:|:----------:|:-------------:|\n")
    print("\nExp B: p_self activation...", file=sys.stderr)
    for pself_mode, zero_flag in [("real", False), ("zeroed (0.5)", True)]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=12,
                                        observer=A1MtObserver(), zero_pself=zero_flag))
        m = analyze(recs)
        blind_vals = [r["events"]["blind"] for r in recs if "blind" in r.get("events", {})]
        sd_vals = [r["events"]["selfdisc"] for r in recs if "selfdisc" in r.get("events", {})]
        L.append("| {} | {} | {} | {:.4f} | {:.4f} |\n".format(
            pself_mode, m["nu"]["mae"], m["nu"]["corr"],
            np.mean(blind_vals) if blind_vals else 0,
            np.mean(sd_vals) if sd_vals else 0))
        print(f"  {pself_mode}: Corr_nu={m['nu']['corr']} blind={np.mean(blind_vals):.4f}", file=sys.stderr)

    # ─── Exp D: Macro STOP sensitivity (A0 vs A1) ───────
    L.append("\n## Exp D: Offline STOP Sensitivity\n\n")
    L.append("| Observer | ε_stop oracle | ε_stop infer | Δε_stop |\n")
    L.append("|----------|:---:|:---:|:---:|\n")
    print("\nExp D: STOP sensitivity...", file=sys.stderr)
    a_s = 0.15; b_s = 0.10; eps_0 = 0.30
    for obs_name, obs_cls in [("A0", RuleBasedMtObserver), ("A1", A1MtObserver)]:
        eo = []; ei = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs = run_session(ALL_LESSONS, th, sid, n_teach=12, observer=obs_cls())
                for r in recs:
                    mt = r.get("m_true", {}); mh = r.get("m_hat", {})
                    if mt:
                        eo.append(eps_0 + a_s*mt.get("nu",0) + b_s*mt.get("gamma_gen",0))
                        ei.append(eps_0 + a_s*mh.get("nu",0) + b_s*mh.get("gamma_gen",0))
        delta = np.mean(np.abs(np.array(ei) - np.array(eo)))
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            obs_name, np.mean(eo), np.mean(ei), delta))

    # ─── Exp E: Confidence validation (A0 vs A1) ────────
    L.append("\n## Exp E: Confidence vs Error Correlation\n\n")
    L.append("| Observer | Dim | Corr(conf, −|err|) |\n")
    L.append("|----------|-----|:---:|\n")
    print("\nExp E: Confidence...", file=sys.stderr)
    for obs_name, obs_cls in [("A0", RuleBasedMtObserver), ("A1", A1MtObserver)]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=16, observer=obs_cls()))
        for dim in ["tau", "nu", "gamma_gen"]:
            confs = [r["conf"].get(dim, 0.5) for r in recs if r.get("conf")]
            errs = [abs(r["m_true"][dim]-r["m_hat"][dim]) for r in recs if r.get("m_true")]
            if len(confs)==len(errs) and len(confs)>2:
                corr = float(np.corrcoef(confs, [-e for e in errs])[0,1]) if np.std(confs)>1e-6 else 0
                L.append(f"| {obs_name} | {dim} | {corr:.4f} |\n")

    rpt = out / "a0_vs_a1_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
