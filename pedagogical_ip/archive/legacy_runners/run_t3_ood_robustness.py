"""T3 Exp-T3-6: OOD / Robustness Audit for POMDP Interface.

Varies:
  - observation noise (σ_obs ∈ {0.01, 0.1, 0.3})
  - risk offset (Δ_risk ∈ {-0.1, 0.0, +0.2})
  - lure strength (tempt ∈ {0.0, 0.5, 1.0})

Checks that predictor calibration degrades smoothly, not catastrophically.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.agent_belief_state import AgentBelief
from src.teachers.shadow_bridge import ShadowBridge

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
N_STEPS = 20
N_SEEDS = 8


def run_ood_condition(theta, seed, obs_noise=0.01, risk_offset=0.0,
                      tempt_strength=0.0):
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    bridge = ShadowBridge(theta=theta, params=AP)

    for step_i in range(N_STEPS):
        # Generate branches with OOD perturbations
        safe_score = 0.7 + rng.normal(0, obs_noise)
        risky_score = 0.3 + rng.normal(0, obs_noise)
        risk = max(0.0, min(1.0, 0.3 + risk_offset + rng.normal(0, 0.05)))
        tempt = max(0.0, min(1.0, tempt_strength + rng.normal(0, 0.05)))

        branches = [
            BranchAttributes(safety_score=float(np.clip(safe_score, 0, 1)),
                           risk_penalty=0.1),
            BranchAttributes(safety_score=float(np.clip(risky_score, 0, 1)),
                           risk_penalty=risk,
                           temptation_score=tempt),
        ]
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)

        # Agent chooses
        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        bridge.observe_step(None, ab, branches, ac)

        # Lightweight state update
        if ac == 0:
            m.update_trust(warn_helpful=False)
        m.update_risk(risk if ac == 1 else 0.05, 0.15)
        m.snapshot()

    return bridge.get_report()


def main():
    print("═══ T3 Exp-T3-6: OOD Robustness Audit ═══\n", file=sys.stderr)
    L = ["# T3 Exp-T3-6: OOD Robustness Audit\n\n"]

    conditions = [
        ("baseline", {"obs_noise": 0.01, "risk_offset": 0.0, "tempt_strength": 0.0}),
        ("noisy_obs", {"obs_noise": 0.3, "risk_offset": 0.0, "tempt_strength": 0.0}),
        ("high_risk", {"obs_noise": 0.01, "risk_offset": 0.2, "tempt_strength": 0.0}),
        ("low_risk", {"obs_noise": 0.01, "risk_offset": -0.1, "tempt_strength": 0.0}),
        ("high_tempt", {"obs_noise": 0.01, "risk_offset": 0.0, "tempt_strength": 1.0}),
        ("combined", {"obs_noise": 0.1, "risk_offset": 0.1, "tempt_strength": 0.5}),
    ]

    L.append("## Prediction Quality Under OOD Conditions\n\n")
    L.append("| θ | Condition | NLL | Brier | ECE | |Δ NLL| | Top1Agree | Entropy |\n")
    L.append("|:-:|:---------:|:---:|:-----:|:---:|:------:|:---------:|:-------:|\n")

    for th in ["safe", "shiny"]:
        for cond_name, kwargs in conditions:
            nlls, briers, eces, dnlls, top1s, ents = [], [], [], [], [], []
            for sid in range(N_SEEDS):
                r = run_ood_condition(th, sid, **kwargs)
                nlls.append(r.mean_new_nll)
                briers.append(r.brier_new)
                eces.append(r.ece_new)
                dnlls.append(r.nll_parity)
                top1s.append(r.top1_agreement)
                ents.append(r.mean_entropy)
            L.append(f"| {th} | {cond_name} | {np.mean(nlls):.4f} | "
                     f"{np.mean(briers):.4f} | {np.mean(eces):.4f} | "
                     f"{np.mean(dnlls):.6f} | {np.mean(top1s):.3f} | "
                     f"{np.mean(ents):.3f} |\n")
        print(f"  {th} done", file=sys.stderr)

    # Verdict: check smooth degradation
    L.append("\n## Verdict\n\n")

    # All conditions must maintain exact parity (interface doesn't change)
    all_parity = True
    for th in ["safe", "shiny"]:
        for cond_name, kwargs in conditions:
            dnlls = []
            for sid in range(N_SEEDS):
                r = run_ood_condition(th, sid, **kwargs)
                dnlls.append(r.nll_parity)
            if np.mean(dnlls) > 0.001:
                all_parity = False
    L.append(f"> Interface parity maintained across all OOD conditions: "
             f"{'✅' if all_parity else '❌'}\n")

    # NLL should increase smoothly with noise, not jump
    baseline_nlls = {}
    for th in ["safe", "shiny"]:
        bls = []
        for sid in range(N_SEEDS):
            r = run_ood_condition(th, sid, obs_noise=0.01)
            bls.append(r.mean_new_nll)
        baseline_nlls[th] = np.mean(bls)

    smooth = True
    for th in ["safe", "shiny"]:
        noisy_nlls = []
        for sid in range(N_SEEDS):
            r = run_ood_condition(th, sid, obs_noise=0.3)
            noisy_nlls.append(r.mean_new_nll)
        ratio = np.mean(noisy_nlls) / max(baseline_nlls[th], 0.01)
        if ratio > 5.0:  # catastrophic = 5x worse
            smooth = False
    L.append(f"> Smooth degradation (no catastrophic NLL spike): "
             f"{'✅' if smooth else '❌'}\n")

    if all_parity and smooth:
        L.append("> **✅ POMDP interface is robust under OOD perturbations**\n")
    else:
        L.append("> **⚠️ Robustness issues detected**\n")

    rpt = out / "t3_ood_robustness_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
