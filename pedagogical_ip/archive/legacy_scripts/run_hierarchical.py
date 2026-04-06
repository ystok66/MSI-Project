"""Hierarchical vs Exact posterior comparison on CGC-v2.

Conditions:
  1. v1_1           (preference-only baseline)
  2. cajt_v3        (discrete joint, calibrated)
  3. factor_exact   (GoalFactorPosterior, 405-cell exact)
  4. factor_hier    (HierarchicalGoalPosterior, factorized)
  5. hier_cajt      (hierarchical + calibrated confidence + adaptive)
  6. oracle

Reports: SelGap, FactorAcc, AvgFactorConf, ExactGoalAcc, held-out generalization
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.compositional_goal_corridor_v2 import (
    generate_cgc2_session, generate_cgc2_scenario, CGC2_SUBTYPES,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal, neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.agents.goal_factor_posterior import GoalFactorPosterior, compute_factor_likelihood
from src.agents.hierarchical_goal_posterior import HierarchicalGoalPosterior
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.metrics.calibrated_confidence import calibrate_posterior, calibrated_confidence
from src.metrics.change_detection import (
    compute_surprisal, compute_drift_score, compute_adaptive_rho, apply_adaptive_diffusion,
)
from src.metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from src.envs.observation_mask import make_observation_mask

out = Path("results")
out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)


def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)


def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


class FactorTutorBase:
    """Base class for factor-posterior tutors."""

    def __init__(self, ap, posterior, calibrate=False, T=0.6, adaptive=False):
        self.ap = ap
        self.post = posterior
        self.calibrate = calibrate
        self.T = T
        self.adaptive = adaptive
        self.warn_count = 0
        self.wait_count = 0
        self.last_drift = 0.0

    def _cal_table(self):
        if self.calibrate:
            # Apply temperature scaling to the joint table directly
            t = self.post.table
            log_t = np.log(t + 1e-15) / self.T
            log_t -= np.max(log_t)
            ct = np.exp(log_t)
            return ct / (ct.sum() + 1e-10)
        return self.post.table

    def _confidence(self):
        ct = self._cal_table() if self.calibrate else self.post.table
        return calibrated_confidence(ct)

    def decide(self, sc, fb, lp, lib, scorer, obs=2):
        fv = np.full_like(fb, 0.3)
        dc = getattr(sc, 'commit_depth', obs + 1)
        dr = getattr(sc, 'reveal_depth', 3)
        delta = dc - dr
        p_self = estimate_self_discovery_prob(dc, dr)
        p_fail = estimate_failure_if_wait(dc, dr)

        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        sa = summarize_branch(vis_a, fb, fv, lp)
        sb = summarize_branch(vis_b, fb, fv, lp)
        sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
        dvoi = max(_sigmoid(abs(sa2[0] - sb2[0])) - _sigmoid(abs(sa[0] - sb[0])), 0)

        C_t = self._confidence()
        O_wait = p_self * _sigmoid((delta - 1.0) / 1.5)
        S_obs = _sigmoid((dc - 2.0) / 1.5)

        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        ent = self.post.entropy
        max_ent = self.post.max_entropy
        unc = ent / max(max_ent, 1e-6)

        # Warmup
        n_obs = self.post.observation_count
        warmup = _sigmoid((n_obs - 3) / 1.5)
        autonomy_bonus = max(1.0 - warmup, 0.0) * 2.0

        # Factor confidence bonus (hierarchical has per-factor confidence)
        fc_bonus = 0.0
        if hasattr(self.post, 'avg_factor_confidence'):
            fc = self.post.avg_factor_confidence()
            fc_bonus = 1.5 * fc * warmup  # reward when factor structure learned

        Q_warn = (1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self)
                  + 1.5 * warmup * (1 - C_t) * unc
                  + 1.5 * warmup * tempt_str * unc
                  + 1.0 * self.last_drift * max(delta_s, dvoi)
                  - 0.05)
        Q_wait = (2.0 * p_self * delta_s - 1.5 * p_fail
                  + 3.5 * C_t * O_wait
                  + 3.0 * S_obs * (1.0 - warmup)
                  + autonomy_bonus
                  + fc_bonus)

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1
        return action, {"C_t": round(C_t, 4), "warmup": round(warmup, 4)}

    def observe_agent_choice(self, chosen_idx, branches):
        if self.adaptive:
            t = self.post.table
            from src.agents.stochastic_agent_policy import PREFERENCE_TYPES as PT
            all_y = self.post._all_y
            n_y = len(all_y)
            lik_arr = np.array([
                compute_factor_likelihood(chosen_idx, branches, y, p, self.ap)
                for y in all_y for p in PT
            ])
            surp = compute_surprisal(t.ravel(), lik_arr)
            self.last_drift = compute_drift_score(surp)
        self.post.update_from_choice(chosen_idx, branches, self.ap)


def make_tutor(name):
    if name == "v1_1": return PersistentTutorV1_1(agent_params=AP)
    if name == "cajt_v3": return CAJTv3(agent_params=AP)
    if name == "factor_exact":
        return FactorTutorBase(AP, GoalFactorPosterior(K=4), calibrate=False)
    if name == "factor_hier":
        return FactorTutorBase(AP, HierarchicalGoalPosterior(K=4), calibrate=False)
    if name == "hier_cajt":
        return FactorTutorBase(AP, HierarchicalGoalPosterior(K=4),
                               calibrate=True, adaptive=True)
    return None


def run_session(strategy, theta, session):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = make_tutor(strategy)
    traces = []

    for ep in session.episodes:
        gm, cfg, meta, sc = generate_cgc2_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.seed + 9999)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=0.4)
        branches = [bas, bar]
        ac = sample_branch_choice(branches, theta, AP, rng)

        do_warn = False
        if strategy == "oracle":
            do_warn = (ep.d_commit < ep.d_reveal)
        elif strategy == "v1_1":
            tutor.reset_stats()
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
        elif tutor is not None:
            action, _ = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
            if hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(ac, branches)

        fact_acc = None
        exact_acc = None
        fc = None
        if isinstance(tutor, FactorTutorBase):
            fact_acc = tutor.post.factor_accuracy(ep.goal_vec)
            pred_y = tutor.post.predicted_goal_vec
            exact_acc = 1.0 if pred_y == ep.goal_vec else 0.0
            if hasattr(tutor.post, 'avg_factor_confidence'):
                fc = tutor.post.avg_factor_confidence()

        traces.append({
            "subtype": ep.subtype, "warned": do_warn,
            "agent_safe": (ac == 0), "goal_vec": ep.goal_vec,
            "fact_acc": fact_acc, "exact_acc": exact_acc, "fc": fc,
        })

    n = len(traces)
    if n == 0:
        return {}
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n
    sw = {}
    for st in CGC2_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
    sg = None
    if sw.get("conflict") is not None and sw.get("aligned") is not None:
        sg = sw["conflict"] - sw["aligned"]

    fa = [t["fact_acc"] for t in traces if t["fact_acc"] is not None]
    ea = [t["exact_acc"] for t in traces if t["exact_acc"] is not None]
    fcs = [t["fc"] for t in traces if t["fc"] is not None]

    return {
        "sbcr": round(sbcr, 3), "wr": round(wr, 3),
        "sg": round(sg, 3) if sg is not None else None,
        "fact_acc": round(np.mean(fa), 3) if fa else None,
        "exact_acc": round(np.mean(ea), 3) if ea else None,
        "avg_fc": round(np.mean(fcs), 3) if fcs else None,
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ Hierarchical vs Exact Posterior ═══\n", file=sys.stderr)
    strategies = ["v1_1", "cajt_v3", "factor_exact", "factor_hier", "hier_cajt", "oracle"]
    thetas = ["safe", "shiny"]
    lines = ["# Hierarchical vs Exact Goal-Factor Posterior\n\n"]

    # Train
    print("Train...", file=sys.stderr)
    lines.append("## Train Compositions\n\n")
    lines.append("| θ | Strategy | SBCR | WR | **SelGap** | FactorAcc | ExactAcc | AvgFC |\n")
    lines.append("|---|----------|------|----|-----------|-----------|----------|-------|\n")
    train_results = []
    for theta in thetas:
        for s in strategies:
            rs = []
            for sid in range(8):
                sess = generate_cgc2_session(sid * 1000 + abs(hash(theta)) % 1000, 12, theta, False)
                r = run_session(s, theta, sess)
                rs.append(r)
            a = {k: avg(rs, k) for k in ["sbcr", "wr", "sg", "fact_acc", "exact_acc", "avg_fc"]}
            a["theta"] = theta; a["strategy"] = s
            train_results.append(a)
            lines.append("| {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr"]), sf(a["wr"]),
                sf(a["sg"], "{:.3f}"),
                sf(a["fact_acc"], "{:.3f}"), sf(a["exact_acc"], "{:.3f}"),
                sf(a["avg_fc"], "{:.3f}")))
            print(f"  {theta} × {s}: SG={sf(a['sg'], '{:.3f}')} FA={sf(a['fact_acc'], '{:.3f}')} "
                  f"FC={sf(a['avg_fc'], '{:.3f}')}", file=sys.stderr)

    # Comparison table
    lines.append("\n### SelGap + FactorAcc Comparison\n\n")
    lines.append("| θ | Metric | exact | **hier** | hier_cajt |\n")
    lines.append("|---|--------|-------|---------|--------|\n")
    for theta in thetas:
        for metric in ["sg", "fact_acc", "avg_fc"]:
            vals = {}
            for s in ["factor_exact", "factor_hier", "hier_cajt"]:
                r = [x for x in train_results if x["theta"] == theta and x["strategy"] == s]
                vals[s] = r[0][metric] if r else None
            label = {"sg": "SelGap", "fact_acc": "FactorAcc", "avg_fc": "AvgFC"}[metric]
            lines.append("| {} | {} | {} | **{}** | {} |\n".format(
                theta, label,
                sf(vals["factor_exact"], "{:.3f}"),
                sf(vals["factor_hier"], "{:.3f}"),
                sf(vals["hier_cajt"], "{:.3f}")))

    # Held-out
    print("\nHeld-out...", file=sys.stderr)
    lines.append("\n## Held-Out Compositions\n\n")
    lines.append("| θ | Strategy | SelGap | FactorAcc | ExactAcc | AvgFC |\n")
    lines.append("|---|----------|--------|-----------|----------|-------|\n")
    for theta in thetas:
        for s in ["factor_exact", "factor_hier", "hier_cajt", "cajt_v3", "oracle"]:
            rs = []
            for sid in range(8):
                sess = generate_cgc2_session(sid * 1000 + abs(hash(theta)) % 1000 + 500,
                                              12, theta, True)
                r = run_session(s, theta, sess)
                rs.append(r)
            a = {k: avg(rs, k) for k in ["sg", "fact_acc", "exact_acc", "avg_fc"]}
            lines.append("| {} | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["sg"], "{:.3f}"),
                sf(a["fact_acc"], "{:.3f}"), sf(a["exact_acc"], "{:.3f}"),
                sf(a["avg_fc"], "{:.3f}")))
            print(f"  HELD {theta} × {s}: SG={sf(a['sg'], '{:.3f}')} FA={sf(a['fact_acc'], '{:.3f}')}",
                  file=sys.stderr)

    with open(out / "hierarchical_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/hierarchical_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
