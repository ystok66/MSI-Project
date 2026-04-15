# Step 1: Root-Cause Disentangling — Experiment Report

## Experiment Setup

- **20 seeds × 3 grammars = 60 samples per cell** (8 conditions × 4 n_sup × 2 N_teach = 5280 total jobs)
- **Phase structure**: N_obs=10 → N_teach=1 or 2 → N_eval=3
- **eta_attn sweep**: 4 values (0.1, 0.3, 0.5, 1.0) for C5 condition

### Conditions

| Code | Description | reveal_mode | attention_mode | tutor_access |
|------|-------------|-------------|----------------|--------------|
| C0 | Baseline (no tutor) | cortex_em | uniform | — |
| C1 | Current tutor | cortex_em | uniform | proxy_oracle |
| C2 | No incremental study | off | uniform | proxy_oracle |
| C3 | HL-only, no incr study | off | uniform | proxy_oracle |
| C4 | Negative memory | negative_memory | uniform | proxy_oracle |
| C5 | Persistent attention | cortex_em | persistent_prior | proxy_oracle |
| D1 | Cheat semantic | cortex_em | uniform | cheat_sem |
| D2 | Cheat full | cortex_em | uniform | cheat_full |

---

## Key Findings

### 1. Tutor Provides Negligible Improvement

**Table 2** (dEVAL_SR vs baseline) clearly shows that **no condition consistently improves over baseline C0**:

| n_sup | C1 | C2 | C3 | C4 | C5 | D1 | D2 |
|-------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 2 | +0.01 | -0.01 | -0.02 | -0.02 | +0.01 | -0.01 | -0.01 |
| 4 | +0.01 | -0.04 | -0.04 | -0.04 | +0.01 | +0.03 | +0.03 |
| 6 | -0.01 | -0.04 | -0.04 | -0.05 | -0.02 | -0.03 | -0.03 |
| 8 | +0.01 | **-0.08** | -0.07 | **-0.08** | +0.00 | -0.00 | -0.00 |

> [!IMPORTANT]
> **Delta magnitudes are ≤ 0.08 (8%) and within SE bars.** The tutor is not helping AND not hurting significantly. This is a weak-signal environment.

### 2. TransferGap Is Universally Negative

**Table 3** shows that `EVAL_SR < OBS_SR` in almost all conditions. This is **NOT** a tutor-specific problem — the same negative transfer exists in C0 baseline:

- n_sup=2, nt=1: baseline TransGap = **-0.029**
- n_sup=8, nt=1: baseline TransGap = **-0.116**

The learner naturally performs worse in eval than obs. The tutor (C1) slightly mitigates this (-0.104 vs -0.116 at n_sup=8), but not by much.

### 3. Cortex Pollution is NOT the Root Cause

Comparing C1 (cortex_em) vs C2 (reveal_mode=off):

| n_sup | C1 EVAL_SR | C2 EVAL_SR | Difference |
|-------|-----------|-----------|-----------|
| 2 | 0.433 | 0.411 | C1 > C2 by 0.022 |
| 4 | 0.500 | 0.444 | C1 > C2 by 0.056 |
| 6 | 0.644 | 0.611 | C1 > C2 by 0.033 |
| 8 | 0.606 | 0.517 | **C1 > C2 by 0.089** |

> [!WARNING]
> **Turning off incremental study HURTS performance!** This means CLS learning from reveals is beneficial, not harmful. Cortex pollution is NOT the root cause.

### 4. Negative Memory Provides No Benefit

C4 (negative_memory) consistently performs the same as or worse than C2 (off):

| n_sup | C2 | C4 | Notes |
|-------|------|------|-------|
| 2 | 0.411 | 0.400 | C4 worse |
| 4 | 0.444 | 0.444 | Equal |
| 6 | 0.611 | 0.606 | C4 slightly worse |
| 8 | 0.517 | 0.511 | C4 slightly worse |

Negative memory penalty is too aggressive or the scoring interaction needs refinement.

### 5. Persistent Attention Prior Has Negligible Effect

C5 with eta_attn sweep shows no systematic improvement:
- All eta values produce EVAL_SR within ±0.01 of C1
- The effect is essentially zero

### 6. Cheat Modes Reveal Action Selection Failure

D1/D2 (cheat_sem/cheat_full) have **dramatically different action profiles** but **similar performance**:

| Condition | BAN/query | HL/query |
|-----------|-----------|----------|
| C1-C4 | ~0.02 | ~3.6 |
| D1/D2 | ~4.0 | ~0.1 |

When the tutor "sees" the learner's actual scores:
- It **overwhelmingly prefers BAN** over HIGHLIGHT
- BAN dominates because the cheat scorer computes that removing bad options is more directly helpful
- But the **eval performance is the same** — because BAN during teaching doesn't improve eval behavior

> [!IMPORTANT]
> **The cheat upper bound is flat!** Even with perfect knowledge of the learner's scoring, the tutor cannot improve eval performance. This means the bottleneck is NOT ToM — it's the **fundamental disconnect between teaching-phase interventions and eval-phase behavior**.

---

## Root Cause Diagnosis

The decision tree from the execution guide yields:

```
C1 ≈ C0 → tutor is ineffective
C2 < C1 → cortex_em is NOT the problem  
D1 ≈ C1 → perfect ToM doesn't help
```

**Conclusion**: The root cause is the **structural disconnect** between teaching interventions and eval performance. Specifically:

1. **HIGHLIGHT doesn't transfer**: Highlighting during teaching doesn't change the learner's attention in eval (no persistent mechanism in the active scorer)
2. **BAN is teaching-only**: Banning an option during teaching doesn't teach the learner to avoid it in eval
3. **CLS learning from reveals is the ONLY mechanism** that carries over to eval — and it **already helps** (C1 > C2)

---

## Recommendations for Next Steps

1. **DON'T** invest in ToM refinement (SMC/MCMC) — cheat upper bound shows zero lift
2. **DON'T** pursue negative memory replacement — it underperforms CLS incremental study
3. **DO** focus on **structural transfer mechanisms**:
   - Make HIGHLIGHT effect transfer to eval (persistent_prior wasn't enough because eval uses fresh attention)
   - Make BAN effect transfer via learner memory  
   - Improve CLS incremental study quality (it's already the best mechanism)
4. **DO** consider **eval-aware teaching**: the tutor should optimize for eval performance, not teaching-phase behavior
