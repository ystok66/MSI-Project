# Phase 3 修正后主实验结果 (500 runs, 1367s)

query_source=generated, n_obs=4, n_teach=8, n_eval=8

## 1. 教学指标 (Teach)

| Condition | TeachSuccess | TeachDeath | TeachTimeout | TeachRetry | TeachStuck |
|-----------|-------------|------------|-------------|-----------|-----------|
| no_tutor  | 0.286±0.188 | 0.562±0.199 | 0.151±0.177 | 13.6±13.6 | 0.161     |
| T0_rule   | 0.564±0.248 | 0.000±0.000 | 0.436±0.248 | 43.5±32.7 | 0.400     |
| T1_proxy  | 0.564±0.248 | 0.000±0.000 | 0.436±0.248 | 43.5±32.7 | 0.400     |
| **T2_exact** | **0.665±0.269** | **0.000** | **0.335±0.269** | **32.7±32.5** | **0.266** |
| T2_compr  | 0.665±0.269 | 0.000±0.000 | 0.335±0.269 | 32.7±32.5 | 0.266     |

## 2. 评估指标 (Eval — 无tutor)

| Condition | EvalSuccess | EvalDeath | EvalTimeout | EvalRetry |
|-----------|------------|----------|------------|----------|
| no_tutor  | 0.233±0.171 | 0.623±0.183 | 0.145±0.138 | 13.3±10.9 |
| T0_rule   | 0.210±0.167 | 0.603±0.193 | 0.188±0.174 | 16.9±13.0 |
| T1_proxy  | 0.210±0.167 | 0.603±0.193 | 0.188±0.174 | 16.9±13.0 |
| T2_exact  | 0.230±0.174 | 0.580±0.178 | 0.190±0.184 | 17.5±14.8 |
| T2_compr  | 0.230±0.174 | 0.580±0.178 | 0.190±0.184 | 17.5±14.8 |

## 3. Divergence 指标 (仅 T2)

| Metric | T2_exact | T2_compressed |
|--------|----------|---------------|
| D_gram_top1_agreement | 0.9851±0.069 | 0.9851±0.069 |
| **D_gram_JS** | **0.0035±0.015** | **0.0035±0.015** |
| D_risk_l1 | 0.0000 | 0.0000 |
| **D_param_role_l1** | **0.0176±0.087** | **0.0176±0.087** |

## 4. 关键发现

### 4.1 Divergence 不再是 0！
- D_gram_JS = 0.0035 — 微小但非零的JS散度
- D_param_role_l1 = 0.0176 — role count参数级别的漂移
- D_gram_top1_agreement = 0.985 — 有1.5%的query shadow和real预测不同
- 比修复前(全部=0)有质的改变，证实shadow grammar sync修复生效

### 4.2 T2 仍然优于 T0/T1 (TeachSuccess)
- T2: 0.665 vs T0/T1: 0.564 — 差值 +10.1%, 相对提升 +17.9%
- T2 timeout更低: 0.335 vs 0.436
- T2 retry更少: 32.7 vs 43.5
- 这个结论在修复后仍然成立

### 4.3 Eval transfer 仍然是瓶颈
- 所有tutor的EvalSuccess都在0.21-0.23之间，无显著差异
- no_tutor的EvalSuccess(0.233)甚至略高于T0/T1(0.210)
- T2的EvalSuccess(0.230)与no_tutor无显著差别
- 结论：tutor改善teaching但不改善eval

### 4.4 Compressed = Exact 仍然成立
- 两者在所有指标上完全相同（包括divergence指标）
- 但现在divergence ≠ 0，所以这个结论不再是平凡成立
- D_param_role_l1 = 0.0176 说明grammar确实有微小漂移
- exact和compressed对这个漂移的捕捉能力相同
- 结论升级：compressed shadow确实足够，可以安全使用

### 4.5 T0 = T1（Proxy tutor无效）
- T0和T1在所有指标上完全相同
- Proxy tutor的belief-based决策没有产生任何差异
- eval模式下hint/courage也没有实际影响

### 4.6 Generated queries 更难
- no_tutor TeachSuccess: 0.286 (generated) vs 0.348 (旧txt)
- EvalSuccess: 0.233 (generated) vs 0.372 (旧txt)
- generated queries的更高死亡率(0.562)证明这些题目不是过于简单