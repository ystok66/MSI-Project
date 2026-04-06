# Migration Notes — For Next Conversation

Generated: 2026-04-01 (post Step 5 cleanup)

## 1. Current Mainline ("What Is Canonical")

### Posterior
- **`joint_goal_pref_posterior.py`** with `prior_mode="structural"` (default)
- 8-goal exact discrete posterior (4 atomic + 4 composite)
- `subgoal_marginals()` is the primary diagnostic metric
- κ̂ is additive macro state only, NOT inside q(g,θ,z)

### Observer
- **Frozen 5D observer** (`internalization_observer.py`): (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)
- Shadow probabilistic observer exists (Step 3) but is NOT canonical

### Micro Tutor
- **Frozen canonical** (`internalization_control_tutor_v4.py`): 2-action {WAIT, WARN}
- Shadow Bayes micro exists (Step 1, v2.1 is latest stable) but is NOT canonical

### Warning
- **RSA observation channel** (`warning_utterance_policy.py`, rsa_obs_s1)
- Legacy bias path preserved for regression only

### Planner
- **Canonical**: `belief_planning.py` + `planner_astar.py`
- **Shadow-ready**: `planner_risk_shadow.py` (A2 mode) — 44pp SafeTop1 improvement
- Not promoted yet — needs formal promotion decision

### Reward Table
- **Rigid discrete table** in `stochastic_agent_policy.py`
- Continuous residuals (Step 5B) proven unnecessary

### Macro Curriculum
- **Hand-crafted score** in `goal_conditional_curriculum_hook.py`
- Bayesian decomposition (Step 5C) has 100% agreement — narrative value only

## 2. What NOT to Touch

| Module | Why |
|:-------|:----|
| Frozen 5D observer | Proven more stable than shadow alternatives |
| Frozen micro tutor | 2-action space {WAIT, WARN} is sufficient |
| Structural prior default | Step 4 promoted, validated |
| Rigid reward table | Step 5B proved continuous residuals add nothing |
| Macro curriculum hook | Step 5C proved 100% agreement with Bayes |

## 3. What Is Worth Continued Research

| Direction | Priority | Current Status |
|:----------|:--------:|:---------------|
| A2 planner → canonical promotion | HIGH | Shadow-ready, needs formal decision |
| Probabilistic observer diagnostics | LOW | Step 3: useful for γ_spec, not for policy |
| Bayesian macro decomposition logging | LOW | 5C: diagnostic use only |
| Θ_K expanded preference types | LOW | Shadow only, not default |

## 4. Paper Comparison Lines (DO NOT DELETE)

| Line | Location | Purpose |
|:-----|:---------|:--------|
| PCFG prior | `compositional_goal_prior.py` | Comparison: stronger but over-compresses composite |
| Legacy bonus | `prior_mode="legacy_bonus"` | Comparison: proven redundant vs no-bonus |
| rsa_obs_s1_trust | warning variants | Ablation |
| Step 5B residuals | `continuous_reward_shadow.py` | Negative result documentation |
| Step 5C Bayes macro | `bayesian_macro_objective_shadow.py` | Narrative decomposition |

## 5. Key Experimental Results (preserved in `results/`)

| Report | Step | Finding |
|:-------|:-----|:--------|
| `step4_prior/step4_report.md` | 4 | Structural > PCFG > legacy |
| `step5a_planner/step5a_report.md` | 5A | A2 +17% TBSR fork_trap |
| `step5a2_cgc_promotion/step5a2_report.md` | 5A.2 | A2 SafeTop1 0.56→1.00 |
| `step5b_reward/step5b_report.md` | 5B | Rigid table sufficient |
| `step5c_macro/step5c_report.md` | 5C | 100% agreement |
