#!/bin/bash
# Move legacy scripts to archive/legacy_runners/
cd /mnt/f/SCAI/Learning-agent/pedagogical_ip

COUNT=0

# Debug/analysis scripts
for f in scripts/_analyze_compact.py scripts/_analyze_final.py scripts/_analyze_phase2a.py \
         scripts/_debug_option_scores.py scripts/_diagnose_rsa.py scripts/_diag_output.txt; do
    if [ -f "$f" ]; then
        mv "$f" archive/legacy_runners/
        echo "MOVED $f"
        COUNT=$((COUNT+1))
    fi
done

# Legacy run scripts (pre-Step patterns)
for f in scripts/run_a0_vs_a1.py scripts/run_a1_stabilization.py \
         scripts/run_ablation_enhancements.py scripts/run_canonical_final.py \
         scripts/run_cct_v12.py scripts/run_confirm_20seed.py \
         scripts/run_final_audit.py scripts/run_final_classification.py \
         scripts/run_final_stabilization.py scripts/run_final_verification.py \
         scripts/run_forensics_gate_macro.py scripts/run_gated_stop.py \
         scripts/run_gated_stop_bc.py scripts/run_infer_only_active_macro.py \
         scripts/run_observer_phase2.py scripts/run_online_micro_hybrid.py \
         scripts/run_p3a_balanced_coverage.py scripts/run_p3b_soft_optimality_audit.py \
         scripts/run_p3c_exact_q_ablation.py scripts/run_p3d_action_space_candidate.py \
         scripts/run_p4a2_gamma_spec_audit.py scripts/run_p4a_gamma_spec.py \
         scripts/run_p4b1_metric_audit.py scripts/run_p4b_4d_observer_eval.py \
         scripts/run_p4c3_macro_kappa.py scripts/run_p4c_attribution_ablation.py \
         scripts/run_p5ac_kappa_signal_bonus.py scripts/run_p5b_5d_integration.py \
         scripts/run_p5d_kappa_nonredundancy_sweep.py scripts/run_p5e_formalization.py \
         scripts/run_p6a_ood_robustness.py scripts/run_pp_mrb_robustness.py \
         scripts/run_stage2_joint.py scripts/run_stage4_v13.py \
         scripts/run_stage5_calibration.py scripts/run_stage6_5_ablation.py \
         scripts/run_stage6_6_v13_1.py scripts/run_stage6_7_v13_2.py \
         scripts/run_stage6_8_v13_3.py scripts/run_stage6_credibility.py \
         scripts/run_stage_n1_macro_coverage.py \
         scripts/run_t1_exp1_kappa_regression.py scripts/run_t1_exp2_q_margin_audit.py \
         scripts/run_t1_exp3_overwarn_fix.py scripts/run_t1_exp4_stability_retest.py \
         scripts/run_t1_smoke_check.py \
         scripts/run_t2_exp2a_persistent_vs_reset.py scripts/run_t2_exp2b_tic_transfer.py \
         scripts/run_t2_exp2c_ticv4_longitudinal.py \
         scripts/run_t3_exp3a_calibrated.py scripts/run_t3_exp3b_need_sweep.py \
         scripts/run_t3_exp3c_saturation_audit.py scripts/run_t3_followup_audit.py \
         scripts/run_t3_ood_robustness.py scripts/run_t3_shadow_pomdp_audit.py \
         scripts/run_t4_family_selective_audit.py scripts/run_t4_inflation_transfer_audit.py \
         scripts/run_t5_hidden_temptation_audit.py scripts/run_t5_mixed_grounded_audit.py \
         scripts/run_t7_compositional_posterior_audit.py \
         scripts/run_t7b2_compositional_temptation_audit.py \
         scripts/run_t7b_composite_recovery_audit.py \
         scripts/run_tutor_shadow.py scripts/run_mt_observer_shadow.py; do
    if [ -f "$f" ]; then
        mv "$f" archive/legacy_runners/
        echo "MOVED $f"
        COUNT=$((COUNT+1))
    fi
done

echo "=== Total scripts moved: $COUNT ==="

# Move debug txt/csv files from results/ to archive/old_reports/
RCOUNT=0
for f in results/*.txt results/debug_*.txt results/fork_*.txt results/lattice_*.txt \
         results/l2*.txt results/v2_*.txt results/tmax_*.txt results/reversion_*.txt; do
    if [ -f "$f" ]; then
        mv "$f" archive/old_reports/
        echo "MOVED $f"
        RCOUNT=$((RCOUNT+1))
    fi
done

# Move old CSV data dumps (keep .md reports)
for f in results/block_experiment.csv results/elcb_sweep.csv results/elcb_transfer.csv \
         results/d1_exposure_scaling.csv results/d2_transfer_gradient.csv \
         results/d3_oracle_upperbound.csv results/diagnostic_summary.csv \
         results/funnel_trap_sweep.csv results/planning_sensitivity.csv \
         results/planning_sensitivity_v2.csv results/tpm_ablation.csv \
         results/tpm_sweep_cross_difficulty.csv results/tpm_sweep_cross_difficulty.json \
         results/transfer_eval.csv results/lattice_sweep.json results/block_summary.json; do
    if [ -f "$f" ]; then
        mv "$f" archive/old_reports/
        echo "MOVED $f"
        RCOUNT=$((RCOUNT+1))
    fi
done

echo "=== Total results moved: $RCOUNT ==="
echo "=== DONE ==="
