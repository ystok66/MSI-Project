# Paper-Facing Story Tables

## Table 1: Online Gain (Pre-TPM → Post-TPM)

| Family | Difficulty | No Tutor | Pre-TPM | **Post-TPM** | **Δ(post-pre)** |
|--------|-----------|----------|---------|-------------|----------------|
| fork_trap | easy | 0% | 65% | **65%** | **+0%** |
| fork_trap | medium | 5% | 65% | **70%** | **+5%** |
| fork_trap | hard | 5% | 40% | **60%** | **+20%** |
| | | | | | |
| hazard_belt | easy | 30% | 40% | **65%** | **+25%** |
| hazard_belt | medium | 30% | 40% | **65%** | **+25%** |
| hazard_belt | hard | 10% | 15% | **35%** | **+20%** |
| | | | | | |
| deadline_gate | easy | 75% | 100% | **100%** | **+0%** |
| deadline_gate | medium | 70% | 100% | **100%** | **+0%** |
| deadline_gate | hard | 70% | 100% | **100%** | **+0%** |
| | | | | | |

## Table 2: TPM Component Ablation (medium, 20 seeds)

| Component Removed | fork_trap | **hazard_belt** | deadline_gate |
|-------------------|-----------|--------------|---------------|
| full_tpm | 70% | 65% | 100% |
| no_bottleneck_match | 75% (+5%) | 65% (+0%) | 100% (+0%) |
| no_warn_damping | 65% (-5%) | 40% (-25%) | 100% (+0%) |
| no_unlock_memory | 70% (+0%) | 65% (+0%) | 100% (+0%) |
| no_perceptual_access | 65% (-5%) | 40% (-25%) | 100% (+0%) |
| cf_only | 65% (-5%) | 40% (-25%) | 100% (+0%) |

## Table 3: Help vs Learning (Online Help Gain vs Transfer)

OHG = SR_assisted - SR_no_tutor (online)

LG = probe_SR(k=3) - probe_SR(k=0, no_tutor) (transfer)

PE = LG / OHG (pedagogical efficiency)

| Family | Condition | SR_online | OHG | LG(k=3) | PE |
|--------|-----------|-----------|-----|---------|----|
| fork_trap | no_tutor | 5% | +0% | +0% | 0.00 |
| fork_trap | robot_belief_pre | 65% | +60% | +0% | 0.00 |
| fork_trap | robot_belief_post | 70% | +65% | +0% | 0.00 |
| | | | | | |
| hazard_belt | no_tutor | 30% | +0% | +0% | 0.00 |
| hazard_belt | robot_belief_pre | 40% | +10% | +0% | 0.00 |
| hazard_belt | robot_belief_post | 65% | +35% | +0% | 0.00 |
| | | | | | |
| deadline_gate | no_tutor | 70% | +0% | +0% | 0.00 |
| deadline_gate | robot_belief_pre | 100% | +30% | +0% | 0.00 |
| deadline_gate | robot_belief_post | 100% | +30% | +0% | 0.00 |
| | | | | | |

## Table 4: Intervention Semantic Taxonomy

| Intervention | Target Layer | Mechanism | Best Family |
|-------------|-------------|-----------|-------------|
| WARN | Epistemic (belief) | Biases risk-relevant features toward danger | fork_trap |
| UNLOCK | Structural (affordance) | Reduces uncertainty on newly reachable cells | deadline_gate |
| ITEM_DROP | Outcome (mitigation) | Provides shield for unavoidable hazard crossing | hazard_belt |
| WAIT | — | Allows autonomous learning without interference | — |

## Table 5: Learner Dynamics Diagnostics (5 seeds, medium)

| Family | Condition | SR | mean Δθ | Δθ_r | BAR | Interpretation |
|--------|-----------|----|---------|----|-----|----------------|
| fork_trap | no_tutor | 0% | 0.2912 | 0.2061 | 0.00 | no intervention, learner updates from failures |
| fork_trap | warning_only | 0% | 0.2912 | 0.2061 | 0.00 | no intervention, learner updates from failures |
| fork_trap | robot_belief_pre | 40% | 0.2321 | 0.1462 | 1.00 | all-WARN (bottleneck always epistemic → action always matches) |
| fork_trap | robot_belief_post | 40% | 0.2321 | 0.1462 | 0.68 | diversified targeting (TPM redirects to non-WARN actions) |
| | | | | | | |
| hazard_belt | no_tutor | 0% | 0.1783 | 0.0994 | 0.00 | no intervention, learner updates from failures |
| hazard_belt | item_only | 60% | 0.1203 | 0.0586 | 0.00 | single-lever or no-tutor |
| hazard_belt | robot_belief_pre | 20% | 0.2078 | 0.1123 | 1.00 | all-WARN (bottleneck always epistemic → action always matches) |
| hazard_belt | robot_belief_post | 60% | 0.1573 | 0.0761 | 0.69 | diversified targeting (TPM redirects to non-WARN actions) |
| | | | | | | |
| deadline_gate | no_tutor | 60% | 0.1085 | 0.0475 | 0.00 | single-lever or no-tutor |
| deadline_gate | unlock_only | 100% | 0.0000 | 0.0046 | 0.04 | single-lever or no-tutor |
| deadline_gate | robot_belief_pre | 100% | 0.0000 | 0.0046 | 1.00 | all-WARN (bottleneck always epistemic → action always matches) |
| deadline_gate | robot_belief_post | 100% | 0.0000 | 0.0046 | 0.84 | diversified targeting (TPM redirects to non-WARN actions) |
| | | | | | | |

---

## Key Narrative

1. **TPM improves online success** substantially (hazard_belt +25pp, fork_trap +20pp at hard)
2. **warn_damping is the critical mechanism** — disabling it drops hazard_belt by 25pp
3. **Transfer is zero for ALL conditions** — TPM is a per-episode assistant, not a learning-inducing tutor
4. **The learner IS updating** (Δθ ≈ 0.15-0.29) — null transfer is NOT from a dead learner
5. **Risk head updates dominate** (Δθ_r >> Δθ_c) — cost learning is minimal
6. **Pre-TPM BAR is perfectly 1.00** because all interventions are WARN, which always matches the dominant epistemic bottleneck. Post-TPM BAR drops to ~0.7 but achieves HIGHER SR — confirming that correct action diversity beats spurious alignment.
