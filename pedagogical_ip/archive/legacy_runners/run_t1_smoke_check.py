"""T1 Smoke Check: verify BCICTv4 q_detail exposure does not change decisions.

Runs a few representative sessions and checks:
1. q_detail is present in info
2. delta_Q sign matches action (positive → WARN, negative → WAIT)
3. Q_WAIT + Q_WARN decomposition is consistent
"""
import sys
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import AgentPolicyParams
from src.agents.internalization_state_v3 import FactoredInternalizationState
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features

AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
ALL_LESSONS = list(LESSON_CATALOG_V2)


def run_check():
    rng = np.random.default_rng(42)
    m = FactoredInternalizationState()
    m.snapshot()
    checks_total = 0
    checks_pass = 0

    for step in range(min(26, len(ALL_LESSONS) * 2)):  # 2 rounds of all 13
        les = ALL_LESSONS[step % len(ALL_LESSONS)]
        ub = {p: 0.5 for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step, "safe", ub, rng)
        ep, spec, gm, cfg_e, meta, sc = et

        rng_w = np.random.default_rng(42)
        ww = generate_world_weights_orthogonal(rng_w, d=4)
        allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
        fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
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
        lib.update("safe", ss)
        lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)

        tutor = BCICTv4(agent_params=AP, use_dose=False)
        action, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        checks_total += 1
        # Check 1: q_detail exists
        assert "q_detail" in info, f"Step {step}: q_detail missing"
        qd = info["q_detail"]

        # Check 2: delta_Q sign matches action
        delta_Q = qd["delta_Q"]
        if action == "WARN":
            assert delta_Q > 0, f"Step {step}: WARN but delta_Q={delta_Q:.6f}"
        elif action == "WAIT":
            assert delta_Q <= 0, f"Step {step}: WAIT but delta_Q={delta_Q:.6f}"

        # Check 3: Q decomposition consistency
        q_wait_check = qd["Q_online_wait"] + qd["V_full_wait_raw"] * 3.5 - qd["R_over_wait_raw"] * 4.0
        q_warn_check = qd["Q_online_warn"] + qd["V_full_warn_raw"] * 3.5 - qd["R_over_warn_raw"] * 4.0
        assert abs(q_wait_check - qd["Q_WAIT"]) < 1e-9, \
            f"Step {step}: Q_WAIT decomposition mismatch {q_wait_check:.8f} vs {qd['Q_WAIT']:.8f}"
        assert abs(q_warn_check - qd["Q_WARN"]) < 1e-9, \
            f"Step {step}: Q_WARN decomposition mismatch"

        checks_pass += 1

    print(f"✅ Smoke check passed: {checks_pass}/{checks_total} steps verified")
    print("   - q_detail present in all steps")
    print("   - delta_Q sign consistent with action")
    print("   - Q decomposition (online + V_full*λ_teach - R_over*λ_over) matches")


if __name__ == "__main__":
    run_check()
