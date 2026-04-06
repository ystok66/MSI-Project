#!/bin/bash
# Move orphaned src/ modules to archive/deprecated/orphaned_src/
cd /mnt/f/SCAI/Learning-agent/pedagogical_ip

mkdir -p archive/deprecated/orphaned_src/agents
mkdir -p archive/deprecated/orphaned_src/teachers
mkdir -p archive/deprecated/orphaned_src/envs
mkdir -p archive/deprecated/orphaned_src/curriculum
mkdir -p archive/deprecated/orphaned_src/metrics

COUNT=0

# agents/ orphans
for f in agents/pragmatic_warning.py agents/hierarchical_goal_posterior.py \
         agents/joint_latent_belief.py agents/mixed_effects_risk_head.py \
         agents/factor_action_bridge.py agents/internalization_dynamics_v2.py \
         agents/belief_protocol.py; do
    if [ -f "src/$f" ]; then
        mv "src/$f" "archive/deprecated/orphaned_src/$f"
        echo "MOVED src/$f"
        COUNT=$((COUNT+1))
    fi
done

# teachers/ orphans
for f in teachers/calibrated_adaptive_joint_tutor_v3.py \
         teachers/joint_latent_tutor_v2.py \
         teachers/joint_tutor_v2.py \
         teachers/block_scoring.py; do
    if [ -f "src/$f" ]; then
        mv "src/$f" "archive/deprecated/orphaned_src/$f"
        echo "MOVED src/$f"
        COUNT=$((COUNT+1))
    fi
done

# envs/ orphans
for f in envs/compositional_goal_corridor.py \
         envs/compositional_goal_corridor_v2.py \
         envs/teaching_internalization_corridor_v2.py \
         envs/persistent_profile_mixed_reveal.py \
         envs/benchmark_generator.py; do
    if [ -f "src/$f" ]; then
        mv "src/$f" "archive/deprecated/orphaned_src/$f"
        echo "MOVED src/$f"
        COUNT=$((COUNT+1))
    fi
done

# curriculum/ orphans
for f in curriculum/lesson_response_model_v2.py; do
    if [ -f "src/$f" ]; then
        mv "src/$f" "archive/deprecated/orphaned_src/$f"
        echo "MOVED src/$f"
        COUNT=$((COUNT+1))
    fi
done

# metrics/ orphans
for f in metrics/decision_info.py metrics/decision_aware_metrics.py \
         metrics/actionability_v2.py metrics/curriculum_metrics.py \
         metrics/pedagogical_metrics.py; do
    if [ -f "src/$f" ]; then
        mv "src/$f" "archive/deprecated/orphaned_src/$f"
        echo "MOVED src/$f"
        COUNT=$((COUNT+1))
    fi
done

echo "=== Total orphaned modules moved: $COUNT ==="
