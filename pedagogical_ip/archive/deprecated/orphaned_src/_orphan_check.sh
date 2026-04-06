#!/bin/bash
# Check which "leaf" modules are truly orphaned (no refs from scripts/ or tests/ either)
cd /mnt/f/SCAI/Learning-agent/pedagogical_ip

SUSPECTS=(
    pragmatic_warning
    preference_posterior
    preference_posterior_v2
    goal_posterior_v1
    hierarchical_goal_posterior
    goal_factor_posterior
    joint_posterior_v2
    joint_latent_belief
    mixed_effects_risk_head
    bounded_agent
    feature_belief
    familiarity
    factor_action_bridge
    internalization_dynamics_v2
    belief_protocol
    calibrated_adaptive_joint_tutor_v3
    joint_latent_tutor_v2
    joint_tutor_v2
    preference_aware_policy_v2
    agent_predictor
    time_aware_door_tutor
    block_scoring
    cause_scoring
    lesson_response_model_v2
    dose_budget
    decision_info
    decision_aware_metrics
    actionability
    actionability_v2
    overteaching
    curriculum_metrics
    online_metrics
    pedagogical_metrics
    compositional_goal_corridor
    compositional_goal_corridor_v2
    teaching_internalization_corridor_v2
    persistent_profile_mixed_reveal
    benchmark_generator
)

echo "module | src_refs | script_refs | test_refs | archive_refs | VERDICT"
for mod in "${SUSPECTS[@]}"; do
    src_refs=$(grep -rl "$mod" src/ 2>/dev/null | grep -v __pycache__ | grep -v "$mod.py" | wc -l)
    script_refs=$(grep -rl "$mod" scripts/ 2>/dev/null | wc -l)
    test_refs=$(grep -rl "$mod" tests/ 2>/dev/null | wc -l)
    archive_refs=$(grep -rl "$mod" archive/ 2>/dev/null | wc -l)
    
    total=$((src_refs + script_refs + test_refs))
    if [ "$total" -eq 0 ]; then
        verdict="ORPHAN"
    elif [ "$src_refs" -eq 0 ] && [ "$script_refs" -eq 0 ]; then
        verdict="TEST-ONLY"
    elif [ "$src_refs" -eq 0 ]; then
        verdict="SCRIPT-ONLY"
    else
        verdict="ACTIVE"
    fi
    
    echo "$mod | $src_refs | $script_refs | $test_refs | $archive_refs | $verdict"
done
