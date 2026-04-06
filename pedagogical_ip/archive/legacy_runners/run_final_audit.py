"""Final audit: hidden temptation stress-test, selfdisc scaling,
intervention-rich blind, confidence ECE.

Exp 1: Hidden temptation confusion (does observer misattribute temptation as dependence?)
Exp 2: selfdisc timing scaling (dc−dr bins)
Exp 3: Intervention-rich blind (forced dose + real vs zeroed p_self)
Exp 4: A1 T-sweep replication
Exp 5: Confidence ECE
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
    RuleBasedMtObserver, A1MtObserver, A2MtObserver, ObsEvent,
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


def run_session(lessons, theta, seed, n_teach=12, observer=None,
                force_dose=None, zero_pself=False, hidden_tempt=0.0):
    """Run session. hidden_tempt adds latent preference bias toward risky branch."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    if observer is None: observer = A1MtObserver()
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
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe", ss); lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        if force_dose is not None: dose_oracle = force_dose
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        if zero_pself: p_self = 0.5
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)

        # Hidden temptation: add latent bias toward risky branch
        # Agent doesn't "know" risk differently, but has hidden preference
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
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc, false_suppression=fs, beneficial_novelty=bn,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        snap = observer.update(ev)
        if hasattr(observer, 'record_action_agreement'):
            m_hat = m.copy(); m_hat.tau = observer.tau_hat
            m_hat.nu = observer.nu_hat; m_hat.gamma_gen = observer.gamma_gen_hat
            a_infer, _, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat)
            observer.record_action_agreement(a_oracle == a_infer)
        else:
            m_hat = m.copy(); m_hat.tau = observer.tau_hat
            m_hat.nu = observer.nu_hat; m_hat.gamma_gen = observer.gamma_gen_hat
            a_infer, _, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat)
        rec = observer.to_log_record(step, ev, snap, a_oracle, a_infer,
            info_oracle.get("Q", 0), 0)
        rec["conf"] = observer.get_confidence()
        rec["p_self"] = p_self; rec["dc_minus_dr"] = dc - dr
        rec["dose"] = dose_oracle; rec["warned"] = warned
        rec["hidden_tempt"] = hidden_tempt
        rec["correct"] = correct
        records.append(rec)
    return records


def analyze(records):
    if not records:
        return {"tau": {"mae": 0, "corr": 0}, "nu": {"mae": 0, "corr": 0},
                "gamma_gen": {"mae": 0, "corr": 0}, "ADR": 0}
    def _m(tv, hv):
        if len(tv) < 2: return {"mae": 0, "corr": 0}
        mae = float(np.mean(np.abs(np.array(tv) - np.array(hv))))
        corr = float(np.corrcoef(tv, hv)[0, 1]) if np.std(tv) > 1e-6 else 0.0
        return {"mae": round(mae, 4), "corr": round(corr, 4)}
    tt=[r["m_true"]["tau"] for r in records if r.get("m_true")]
    th=[r["m_hat"]["tau"] for r in records if r.get("m_true")]
    nt=[r["m_true"]["nu"] for r in records if r.get("m_true")]
    nh=[r["m_hat"]["nu"] for r in records if r.get("m_true")]
    gt=[r["m_true"]["gamma_gen"] for r in records if r.get("m_true")]
    gh=[r["m_hat"]["gamma_gen"] for r in records if r.get("m_true")]
    adr = sum(1 for r in records if r["disagree"]) / max(len(records), 1)
    return {"tau": _m(tt, th), "nu": _m(nt, nh), "gamma_gen": _m(gt, gh),
            "ADR": round(adr, 3)}


def main():
    print("═══ Final Audit ═══\n", file=sys.stderr)
    L = ["# Shadow Observer — Final Audit\n\n"]

    # ─── Exp 1: Hidden Temptation Confusion ──────────────
    L.append("## Exp 1: Hidden Temptation Stress Test\n\n")
    L.append("| Tempt | θ | Corr_τ | Corr_ν | Corr_γ | MAE_ν | ADR | err_rate |\n")
    L.append("|:-----:|:-:|:------:|:------:|:------:|:-----:|:---:|:--------:|\n")
    print("Exp 1: Hidden temptation...", file=sys.stderr)
    for ht in [0.0, 0.3, 0.6, 1.0]:
        for th in ["safe", "shiny"]:
            recs = []
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=16,
                                        observer=A1MtObserver(), hidden_tempt=ht))
            m = analyze(recs)
            err_rate = sum(1 for r in recs if not r.get("correct", True)) / max(len(recs), 1)
            L.append("| {} | {} | {} | {} | {} | {} | {} | {:.3f} |\n".format(
                ht, th, m["tau"]["corr"], m["nu"]["corr"], m["gamma_gen"]["corr"],
                m["nu"]["mae"], m["ADR"], err_rate))
            print(f"  ht={ht} θ={th}: Corr_nu={m['nu']['corr']} MAE_nu={m['nu']['mae']} "
                  f"err={err_rate:.3f}", file=sys.stderr)

    # ─── Exp 2: selfdisc timing scaling ──────────────────
    L.append("\n## Exp 2: selfdisc Timing Scaling\n\n")
    L.append("| dc−dr | n | mean_pself | selfdisc>0 | mean_selfdisc | blind>0 |\n")
    L.append("|:-----:|:-:|:----------:|:----------:|:-------------:|:-------:|\n")
    print("\nExp 2: selfdisc scaling...", file=sys.stderr)
    all_recs = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            all_recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=16,
                                        observer=A1MtObserver()))
    for gap_label, lo, hi in [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3+", 3, 99)]:
        sub = [r for r in all_recs if lo <= r.get("dc_minus_dr", 1) <= hi]
        if not sub: continue
        bv = [r["events"]["blind"] for r in sub if r.get("events")]
        sv = [r["events"]["selfdisc"] for r in sub if r.get("events")]
        mean_ps = np.mean([r.get("p_self", 0.5) for r in sub])
        L.append("| {} | {} | {:.3f} | {} / {} | {:.4f} | {} / {} |\n".format(
            gap_label, len(sub), mean_ps,
            sum(1 for v in sv if v>0), len(sv), np.mean(sv) if sv else 0,
            sum(1 for v in bv if v>0), len(bv)))

    # ─── Exp 3: Intervention-rich with p_self real/zeroed ─
    L.append("\n## Exp 3: Intervention-Rich Blind Audit\n\n")
    L.append("| Regime | p_self | Blind>0 | Mean blind | Corr_ν | MAE_ν |\n")
    L.append("|--------|:------:|:-------:|:----------:|:------:|:-----:|\n")
    print("\nExp 3: Intervention-rich...", file=sys.stderr)
    for regime, fd in [("natural", None), ("med", 0.5), ("high", 1.0)]:
        for ps_label, zp in [("real", False), ("zero", True)]:
            recs = []
            for th in ["safe", "shiny"]:
                for sid in range(NS):
                    recs.extend(run_session(ALL_LESSONS, th, sid, observer=A1MtObserver(),
                                            force_dose=fd, zero_pself=zp))
            m = analyze(recs)
            bv = [r["events"]["blind"] for r in recs if r.get("events")]
            n_pos = sum(1 for v in bv if v > 0)
            L.append("| {} | {} | {} / {} | {:.4f} | {} | {} |\n".format(
                regime, ps_label, n_pos, len(bv), np.mean(bv),
                m["nu"]["corr"], m["nu"]["mae"]))

    # ─── Exp 4: A1 T-sweep replication ───────────────────
    L.append("\n## Exp 4: A1 Stability Replication\n\n")
    L.append("| T | Corr_τ | Corr_ν | Corr_γ | MAE_τ | MAE_ν | ADR |\n")
    L.append("|:-:|:------:|:------:|:------:|:-----:|:-----:|:---:|\n")
    print("\nExp 4: A1 replication...", file=sys.stderr)
    for n_t in [6, 12, 16, 20]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=n_t,
                                        observer=A1MtObserver()))
        m = analyze(recs)
        L.append("| {} | {} | {} | {} | {} | {} | {} |\n".format(
            n_t, m["tau"]["corr"], m["nu"]["corr"], m["gamma_gen"]["corr"],
            m["tau"]["mae"], m["nu"]["mae"], m["ADR"]))
        print(f"  T={n_t}: τ={m['tau']['corr']} ν={m['nu']['corr']} γ={m['gamma_gen']['corr']}", file=sys.stderr)

    # ─── Exp 5: Confidence ECE ───────────────────────────
    L.append("\n## Exp 5: Confidence ECE (A1)\n\n")
    L.append("| Dim | Bin | Mean Conf | Mean |err| | n | gap |\n")
    L.append("|-----|:---:|:---------:|:---------:|:-:|:---:|\n")
    print("\nExp 5: Confidence ECE...", file=sys.stderr)
    recs_cb = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            recs_cb.extend(run_session(ALL_LESSONS, th, sid, n_teach=16,
                                       observer=A1MtObserver()))
    for dim in ["tau", "nu", "gamma_gen"]:
        confs = np.array([r["conf"].get(dim, 0.5) for r in recs_cb if r.get("conf")])
        errs = np.array([abs(r["m_true"][dim]-r["m_hat"][dim])
                         for r in recs_cb if r.get("m_true")])
        if len(confs) != len(errs) or len(confs) < 10: continue
        ece_total = 0.0; n_total = 0
        for lo, hi, label in [(0.0, 0.25, "0-25"), (0.25, 0.4, "25-40"),
                               (0.4, 0.6, "40-60"), (0.6, 1.0, "60+")]:
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() < 2: continue
            mc = confs[mask].mean(); me = errs[mask].mean()
            # "calibrated" = high conf → low err, so gap = conf - (1-err)
            gap = mc - (1.0 - me)
            ece_total += abs(gap) * mask.sum()
            n_total += mask.sum()
            L.append("| {} | {} | {:.3f} | {:.4f} | {} | {:.3f} |\n".format(
                dim, label, mc, me, int(mask.sum()), gap))
        if n_total > 0:
            ece = ece_total / n_total
            L.append(f"| {dim} | **ECE** | — | — | {n_total} | **{ece:.4f}** |\n")

    rpt = out / "final_audit_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
