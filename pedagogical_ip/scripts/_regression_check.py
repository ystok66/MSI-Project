"""Minimal regression check for post-cleanup sanity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = 0
failed = 0

def check(label, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {label}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        failed += 1

# 1. Posterior default = structural
def t1():
    from src.teachers.joint_goal_pref_posterior import JointGoalPrefPosterior
    from src.agents.stochastic_agent_policy import AgentPolicyParams
    p = JointGoalPrefPosterior(params=AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0))
    assert p._prior_mode == "structural", f"default is {p._prior_mode}"
check("posterior default = structural", t1)

# 2. Structural prior normalized
def t2():
    from src.teachers.compositional_goal_prior import compute_normalized_goal_prior, GoalPriorContext, GoalPriorConfig
    from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
    pr = compute_normalized_goal_prior(DEFAULT_GOAL_SPACE, GoalPriorContext(), GoalPriorConfig())
    assert abs(float(pr.sum()) - 1.0) < 1e-6
check("structural prior normalized", t2)

# 3. A2 shadow imports
def t3():
    from src.agents.planner_risk_shadow import PlannerRiskShadow
    s = PlannerRiskShadow(mode="A2")
check("A2 shadow imports", t3)

# 4. Warning policy imports
def t4():
    from src.teachers.warning_utterance_policy import WarningUtterancePolicy
check("warning policy imports", t4)

# 5. Frozen observer imports
def t5():
    from src.teachers.internalization_observer import RuleBasedMtObserver
check("frozen observer (RuleBasedMtObserver) imports", t5)

# 6. Frozen micro tutor imports
def t6():
    from src.teachers.internalization_control_tutor_v4 import BCICTv4
check("frozen micro tutor (BCICTv4) imports", t6)

# 7. 8-goal hypothesis space
def t7():
    from src.teachers.compositional_goal_hypotheses import DEFAULT_GOAL_SPACE
    assert len(DEFAULT_GOAL_SPACE.hypotheses) == 8
check("8-goal hypothesis space", t7)

# 8. Subgoal marginals work
def t8():
    from src.teachers.joint_goal_pref_posterior import JointGoalPrefPosterior
    from src.agents.stochastic_agent_policy import AgentPolicyParams
    p = JointGoalPrefPosterior(params=AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0))
    m = p.subgoal_marginals()
    assert isinstance(m, dict) and len(m) > 0
check("subgoal_marginals()", t8)

# 9. CGC-v2 episode generation
def t9():
    from src.envs.cgc_v2_family import generate_cgc_session, generate_cgc_episode_scenario
    sess = generate_cgc_session(0, n_episodes=3, theta_true="safe", goal_obj="collect_red")
    gm, cfg, meta, sc = generate_cgc_episode_scenario(sess.episodes[0], theta_true="safe")
    assert gm.true_cost.shape[0] > 0
check("CGC-v2 episode generation", t9)

# 10. Deprecated module still importable (backward compat)
def t10():
    from src.teachers.composite_goal_compatibility import CompositeGoalCompatibility
check("deprecated compat module importable", t10)

print(f"\n{'='*40}")
print(f"PASSED: {passed}  FAILED: {failed}")
if failed == 0:
    print("ALL REGRESSION CHECKS PASSED")
else:
    print("REGRESSION FAILURES DETECTED")
    sys.exit(1)
