# Root-Cause Disentangling — Execution Guide

> This document is the user-authored specification that drives the step1 implementation.

## 0. 当前系统的关键事实

当前 CLS tutor 是一个四阶段 block：Pre-train → Observation → Teaching → Evaluation。Teaching 阶段 tutor 可以做 `WAIT | BAN | HIGHLIGHT | RISK_HINT | SKIP`；Evaluation 阶段 tutor 被强制 WAIT，最终看的是 frozen learner 的 **EVAL_SR**。

当前 learner 的 pick utility 为：
```
U_pick(j) = α_sem·S_sem(j) - α_risk·μ_d(j) - α_unc·u_d(j) + memory_penalty(j)
```

当前 tutor 核心近似：用 oracle semantic score 估计 learner 会选谁。

**已知结果**：full_tutor delta 基本落在噪声内；BAN 部分有害；EVAL_SR < OBS_SR（负迁移）。

---

## 1. 不要先做 full SMC over CLS-EM state

当前 root cause 还没拆干净。如果 incremental_study 真的在污染 cortex，更强的 ToM 只会"更高效地把 learner 往错误更新里送"。

优先级：
1. 判断 learner update 污染是不是主因
2. 判断 HIGHLIGHT 不迁移是不是主因  
3. 最后判断 ToM misspecification 有多大

---

## 2. 统一目标函数（设计北极星）

```
Q_CLS(ω) = Q_online(ω) + λ_teach·V_transfer(ω) - λ_over·R_over(ω) - λ_poison·R_poison(ω)
```

CLS 特殊之处：必须加 R_poison（cortex 污染代价）。

---

## 3. 实验 1：Root-Cause Disentangling

### 六个条件

| 条件 | reveal_mode | attn_mode | access_mode | 目的 |
|------|-------------|-----------|-------------|------|
| C0. baseline | cortex_em | uniform | — | 无 tutor 基线 |
| C1. current_tutor | cortex_em | uniform | proxy_oracle | 当前系统原样 |
| C2. no_incremental_study | **off** | uniform | proxy_oracle | 切断 cortex 污染 |
| C3. hl_no_incr | **off** | uniform | proxy_oracle (HL only) | 纯 HIGHLIGHT 效应 |
| C4. neg_memory_only | **negative_memory** | uniform | proxy_oracle | 保留负证据，不污染 |
| C5. persistent_attn | cortex_em | **persistent_prior** | proxy_oracle | HIGHLIGHT 跨 query 迁移 |

---

## 4. 实验 2：Cheating Upper Bounds

| 条件 | 描述 |
|------|------|
| D1. cheat_sem | tutor 只读访问 learner 真实 semantic_scores + attention |
| D2. cheat_full | tutor 只读访问 learner 完整 U_pick 分解 + 一步前向模拟 |

---

## 5. 三个统一开关

1. `reveal_learning_mode ∈ {"cortex_em", "off", "negative_memory"}`
2. `attention_init_mode ∈ {"uniform", "persistent_prior"}`
3. `tutor_access_mode ∈ {"proxy_oracle", "cheat_sem", "cheat_full"}`

---

## 6. 关键新指标

| 指标 | 公式 | 用途 |
|------|------|------|
| TransferGap | EVAL_SR - OBS_SR | 迁移学习效果 |
| NLL_ToM | -log P_tutor(actual_learner_action) | ToM 准确性 |
| ProbeDelta | ProbeScore_after - ProbeScore_before (per reveal) | cortex 污染检测 |
| HighlightUsefulness | ΔP_corr per HIGHLIGHT (same-query + eval-start) | HIGHLIGHT 有效性 |
| CortexPoisonRate | fraction of reveals with ProbeDelta < 0 | R_poison proxy |

---

## 7. 结果解读决策树

- **情况 A**: C2 >> C1 → cortex 污染主导 → 废弃 reveal→cortex EM
- **情况 B**: C5 >> others → transfer 缺失主导 → 做 persistent/meta-attention
- **情况 C**: D1 >> C5 → ToM 缺口主导 → 做 compressed learner posterior
- **情况 D**: D2 >> D1 → 需要 full policy surrogate
- **情况 E**: 都不提升 → 更深层结构问题
