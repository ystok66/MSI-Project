# Codebase Status Report — Post-Reorganization

## 1. Canonical Path

```
lesson_library_v2 → adaptive_episode_generator_v2
    → preference_aware_policy_v2 (micro tutor)
    → joint_latent_tutor_v2 (joint tutor)
    → curriculum_controller_v13 (macro controller)
        → risk_budget_calibration (θ-adaptive)
        → pairwise_response_model (ranking engine)
        → mastery_model (Beta-Bernoulli tracker)
    → pedagogical_framework (unified runtime API)
    → behavior_probes + overteach_rate_v2 (metrics)
```

### Files actively called by main experiments

| File | Role | Lines |
|------|------|:-----:|
| `src/curriculum/pedagogical_framework.py` | Unified API | 230 |
| `src/curriculum/curriculum_controller_v13.py` | Canonical controller | 460 |
| `src/curriculum/risk_budget_calibration.py` | θ-adaptive budgets | 95 |
| `src/curriculum/pairwise_response_model.py` | Ranking engine | 205 |
| `src/curriculum/mastery_model.py` | Mastery tracker | 74 |
| `src/curriculum/lesson_library_v2.py` | Lesson catalog | 131 |
| `src/curriculum/adaptive_episode_generator_v2.py` | Episode generator | 115 |
| `src/curriculum/adaptive_episode_generator.py` | Transfer episodes | 190 |
| `src/teachers/preference_aware_policy_v2.py` | Stage 1 tutor | 240 |
| `src/teachers/joint_latent_tutor_v2.py` | Stage 2 tutor | 230 |
| `src/teachers/internalization_control_tutor_v4.py` | Micro decision | 190 |

---

## 2. File Role Table

### Core (must keep in src/)

| File | Current Role | Main Path? | Comparison Only? |
|------|-------------|:----------:|:----------------:|
| `curriculum_controller_v13.py` | Canonical controller | **YES** | No |
| `risk_budget_calibration.py` | θ-adaptive budgets | **YES** | No |
| `pairwise_response_model.py` | Pairwise ranking | **YES** | No |
| `mastery_model.py` | Mastery tracker | **YES** | No |
| `pedagogical_framework.py` | Unified API | **YES** | No |
| `lesson_library_v2.py` | Lesson catalog | **YES** | No |
| `preference_aware_policy_v2.py` | Stage 1 tutor | **YES** | No |
| `joint_latent_tutor_v2.py` | Stage 2 tutor | **YES** | No |
| `internalization_control_tutor_v4.py` | Micro tutor | **YES** | No |

### Archived (in archive/, available for regression)

| Directory | Contents | Count |
|-----------|----------|:-----:|
| `archive/legacy_controllers/` | v1–v12 controllers | 12 |
| `archive/legacy_response_models/` | v1–v3 response models + hybrid | 6 |
| `archive/legacy_scripts/` | All non-canonical experiments | ~100 |
| `archive/legacy_teachers/` | v1/v3/v5 tutors, oracle, RSA, etc | 16 |

---

## 3. Mechanism Actionability (from Stage 6 results)

| Mechanism | PCR (safe) | PCR (shiny) | Status |
|-----------|:----------:|:-----------:|:------:|
| **G_pw** (pairwise gain) | **66.7%** | **60.7%** | ✅ Primary driver |
| **G** (total gain) | 66.7% | 60.7% | ✅ Active |
| **U** (uncertainty) | 16.7% | 15.2% | ⚠️ Weak but nonzero |
| **G_hier** (hierarchical) | 0.0% | 0.0% | ❌ Zero actionability |
| **G_res** (residual) | 0.0% | 0.0% | ❌ Zero actionability |
| **H** (harm penalty) | 0.0% | 0.0% | ❌ Zero (correct: goes through constraint) |

### Interpretation

- **G_pw is THE ranking engine.** It's the only term consistently changing lesson argmax.
- **U has slight influence** (16%) — worth keeping as exploration but not dominant.
- **G_hier and G_res are dead weight** in the scoring formula. They contribute AM but never change argmax. **Candidate for removal.**
- **H goes through constraint filter**, so PCR=0% is expected/correct.

---

## 4. Family Coverage

| Family | Lessons | Primary Mechanism Tested | Controller Sensitivity |
|--------|:-------:|--------------------------|:----------------------:|
| PP-MRB | 2 | Persistent selective fading, self-discovery | **HIGH** (held-out: −28pp C on shiny) |
| TIC | 3 | Warn-rescue, temptation, self-discovery | Medium (held-out: −13pp C on shiny) |
| TIC-v4 | 5 | Advice validity, novelty, suppression cost | **LOW** (held-out: 0pp C change) |

### Key Finding

Controller is **disproportionately dependent on PP-MRB** for shiny performance.
TIC-v4 can be fully held out with no C impact, suggesting controller doesn't rely on it for ranking.
This is the most important family coverage gap to address.

---

## 5. Suspicious Complexity

### A. G_hier and G_res — likely removable

Both have PCR=0% in Stage 6. They add code and computation but never change which lesson is ranked #1. **Ablation candidate.**

### B. OTR as single metric — measurement artifact

60% of OTR comes from EVAL overhead. Reporting OTR_total without decomposition is misleading. **Should split into OTR_teach and OTR_eval.**

### C. EVAL rank-change = 0% — suspicious

EVAL provides +32pp C but never changes which lesson is top-1. Its value is entirely through mastery estimation → better STOP/constraint decisions. This suggests **EVAL's mechanism is misunderstood in the current framing** — it's a mastery calibrator, not a lesson reranker.

### D. STOP margin = 0.40 on shiny — too aggressive

The threshold sits well above the best available lesson value, forcing early STOP. But no_stop×theta gets the best shiny results (C=69% E=59%). **STOP threshold needs per-θ calibration.**

### E. close-gap EVAL trigger never fires

All 24/24 EVAL triggers are "uncertainty", none are "close_gap". The close-race bonus (λ_close) may be dead code. **Ablation candidate.**

---

## 6. Recommended Ablation List (Priority Order)

### Ablation 1: G_hier / G_res removal
**Question:** Does removing these zero-PCR terms change anything?
**Compare:** full v13 vs v13−G_hier vs v13−G_res vs v13−both
**Expected:** No change → confirms safe removal.

### Ablation 2: STOP threshold sweep (per-θ)
**Question:** Is STOP too aggressive, esp. on shiny?
**Compare:** eps_0 ∈ {−0.10, −0.05, 0.00, +0.05}
**Expected:** Lower threshold → more teaching → better shiny C/E.

### Ablation 3: EVAL mechanism isolation
**Question:** Is EVAL's value through mastery update or something else?
**Compare:** full EVAL vs mastery-only-update (no EVAL action, just probe) vs no EVAL
**Expected:** Mastery-only ≈ full EVAL → confirms mastery calibrator role.

### Ablation 4: close-gap bonus removal
**Question:** Does the λ_close term ever fire?
**Compare:** full vs no_close_gap
**Expected:** Identical → confirms dead code, safe to remove.

### Ablation 5: θ-adaptive vs uniform-relaxed
**Question:** Is θ-adaptive better than simply widening budgets for all θ?
**Compare:** theta-adaptive vs fixed-wide (shiny values for all) vs fixed-tight
**Expected:** theta-adaptive wins on safe, fixed-wide might match on shiny.

### Ablation 6: PP-MRB dependency test
**Question:** Does controller collapse without PP-MRB or can other families compensate?
**Compare:** all-families vs no-PP-MRB vs PP-MRB-only vs TIC-only
**Expected:** Confirms family coverage gap; may motivate new families.
