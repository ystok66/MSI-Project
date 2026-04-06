"""Debug consequence grounded rollout."""
import sys
sys.path.insert(0, ".")
from src.teachers.consequence_grounded_option_rollout import *
from src.teachers.action_predictor import ActionPredictor
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.agent_belief_state import AgentBelief

ap = ActionPredictor(AgentPolicyParams(beta=4.0, epsilon=0.1))
branches = [
    BranchAttributes(safety_score=0.8, risk_penalty=0.1),
    BranchAttributes(safety_score=0.3, risk_penalty=0.4, temptation_score=0.3),
]
ab = AgentBelief(theta="shiny")

# Test WARN
cgr = ConsequenceGroundedRollout(ap)
r = cgr.evaluate_option("WARN", branches, ab, safe_branch_idx=0)
print(f"WARN: p_safe_orig={r.p_safe_original:.4f} p_safe_cf={r.p_safe_counterfactual:.4f} lift={r.success_lift:.4f}")

# Test parametric
for alpha in [0.10, 0.15, 0.20]:
    cgr2 = ConsequenceGroundedRollout(ap, config=ConsequenceConfig(alpha_warn=alpha))
    r2 = cgr2.evaluate_option("WARN", branches, ab, safe_branch_idx=0)
    print(f"  alpha={alpha:.2f} lift={r2.success_lift:.4f} p_cf={r2.p_safe_counterfactual:.4f}")

# Modified branches
mod = cgr.apply_consequence("WARN", branches)
for i, b in enumerate(mod):
    print(f"  branch[{i}]: safety={b.safety_score:.3f} risk={b.risk_penalty:.3f}")
