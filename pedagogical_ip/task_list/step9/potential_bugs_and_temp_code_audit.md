# 潜在 Bug / 暂时性代码 / 设计隐患 — 审计报告

> 审计范围：`pedagogical_ip/src/` 全部 Python 模块  
> 审计日期：2026-04-07

---

## 目录

| 分类 | 数量 | 含义 |
|------|------|------|
| 🔴 潜在 Bug | 8 | 会影响实际结果的逻辑错误 |
| 🟠 暂时性 / Placeholder | 5 | 标注为临时或有遗留占位代码 |
| 🟡 缺失 Safety Guard | 3 | 缺少 NaN/Inf 保护 |
| ⚪ Dead Code / 冗余 | 4 | 永远不执行或已被覆盖的代码 |

---

## 1. 🔴 潜在 Bug

### BUG-1: `begin_episode()` / `end_episode()` 从未被 Runner 调用

| 字段 | 内容 |
|------|------|
| 位置 | [slow_fast_head.py:106,124](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/slow_fast_head.py#L106-L154) |
| 问题 | `GenericSlowFastPredictor` 定义了 `begin_episode()` 和 `end_episode()` 来管理 slow/fast 权重生命周期，但 `lattice_v2_runner.py` 中 **没有任何代码调用它们**。 |
| 影响 | 如果将 `GenericSlowFastPredictor` 直接传入 runner 的 `init_episode()` 作为 `latent_predictor`，slow→fast 复制和 fast→slow EMA 更新 **永远不会发生**。slow 权重永远停留在初始值，fast 权重只在单 episode 内学习。跨 episode transfer 机制 **完全无效**。 |
| 现状 | 目前只有 `scripts/` 下的实验脚本手动调用。这意味着 **主 runner 管线不支持 SlowFast**。 |

```python
# lattice_v2_runner.py 中完全搜索不到：
#   .begin_episode()
#   .end_episode()
# 但 scripts/phase2b_harder_eval.py:109-110 正确调用了：
#   if persist and hasattr(predictor, 'begin_episode'):
#       predictor.begin_episode()
```

> [!CAUTION]
> 这是架构级别的 gap：runner 不知道 predictor 有生命周期方法。修复方案是在 runner 的 episode 循环中增加 `if hasattr(lp, 'begin_episode'): lp.begin_episode()` 和对应的 `end_episode()`。

---

### BUG-2: `end_episode()` 不更新 slow 的 `xx_sum` / `n_updates`

| 字段 | 内容 |
|------|------|
| 位置 | [slow_fast_head.py:124-154](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/slow_fast_head.py#L124-L154) |
| 问题 | `end_episode()` 只 EMA-更新 slow 的 `w` 和 `b`，**不更新** `xx_sum`、`xy_sum` 或 `n_updates`。 |
| 影响 | slow predictor 的 `n_updates` 永远为 0。当 `begin_episode()` 把 slow 复制到 fast 后再把 `n_updates` 清零，结果是 fast 每 episode 开头 `n_updates=0`，意味着 uncertainty 估计始终返回最大值 (`1.0` for cost, `0.25` for risk)。**不随 episode 积累而降低**。 |
| 性质 | 如果 BUG-1 被修复，这个 bug 会被暴露。目前因为 BUG-1，它被掩盖。 |

---

### BUG-3: `BasisRiskHead` 缺少 NaN 梯度保护

| 字段 | 内容 |
|------|------|
| 位置 | [structured_basis_head.py:230-233](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py#L230-L233) |
| 对比 | [cost_risk_model.py:90-91](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py#L90-L91) (`BayesianCostHead`), [structured_basis_head.py:155](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py#L155) (`BasisCostHead`) |
| 问题 | 三个 cost head（`BayesianCostHead`, `BasisCostHead`）都有 `if not np.isfinite(grad_norm): return`，但 **两个 risk head** `BayesianRiskHead` 和 `BasisRiskHead` **缺少这个保护**。 |
| 影响 | 如果 risk head 的梯度因数值溢出变为 NaN/Inf，更新不会被跳过，会写入 NaN 权重。后续所有 `predict_risk()` 全部返回 NaN，导致 planner 崩溃或静默产生错误路径。 |

```diff
 # BasisRiskHead.update_from_label (L230-233) — 修复：
 grad_norm = float(np.linalg.norm(grad_w))
 max_grad_norm = 5.0
+if not np.isfinite(grad_norm):
+    return  # skip update on NaN/Inf gradient
 if grad_norm > max_grad_norm:
     grad_w *= max_grad_norm / grad_norm
```

同一修复也应该加到 `BayesianRiskHead`（[risk_model.py:94-96](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/risk_model.py#L94-L96)）。

---

### BUG-4: `active_duration` 幽灵属性

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_helpers.py:293](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_helpers.py#L293) |
| 问题 | `getattr(s.inventory, 'active_duration', 0) > 0` 检查一个 **不存在的属性** `active_duration`。`InventoryState` 类只有 `shield: int`，没有 `active_duration`。 |
| 影响 | `getattr` 会永远返回默认值 `0`，所以条件永远为 `False`。这意味着 oracle 的 "已经有 shield" 检查 **永远失败**，oracle 可能在同一 episode 中重复 drop shield（虽然 `add_shield()` 有 stacking 保护）。 |
| 修复 | 应该用 `s.inventory.has_shield()` 代替。 |

---

### BUG-5: `dtmb_helpers.py` 断路 import

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_helpers.py:375](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_helpers.py#L375) |
| 问题 | `from src.agents.pragmatic_warning import InterventionDecision` 使用绝对路径 `src.agents.pragmatic_warning`，但该模块 **不存在**（经运行验证 `ModuleNotFoundError`）。 |
| 影响 | 整个 `apply_dtmb_oracle_action()` 的最后记录步骤在 `try/except` 中，会静默吞掉这个 ImportError。`s.last_intervention` 不会被设置为 `InterventionDecision`，下游 metric 收集字段缺失。 |
| 修复 | 正确路径应该是 `from ..teachers.intervention_policy import InterventionDecision` 或定义一个 lightweight dataclass。 |

---

### BUG-6: `dtmb_helpers.py` oracle 直接修改 `gridmap.true_risk`

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_helpers.py:370](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_helpers.py#L370) |
| 问题 | `s.gridmap.true_risk[br, bc] *= 0.15` 直接修改了 **真实 risk 值**（ground truth）。 |
| 影响 | 这违反了 "tutor 不修改真值" 的设计原则。oracle ITEM_DROP 在 `inventory=None` 路径下会永久降低 belt 细胞的 true risk，影响后续所有 episode（如果 gridmap 被共享）。 |
| 设计意图 | 看起来是一个 "没有 inventory 系统时的临时替代方案"，但在有 inventory 的主线上不应被触发。 |

---

### BUG-7: tutor `warn_count` 双重计数

| 字段 | 内容 |
|------|------|
| 位置 | [internalization_control_tutor_v4.py:273-278, 385-388](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/internalization_control_tutor_v4.py#L273-L278) |
| 问题 | 当 `micro_policy_mode != "canonical"` 时，canonical 路径先在 L273 增加 `warn_count`，然后 micro_bayes 覆盖 `best_action` 后，L388 试图用 `+= 1 - (1 if ...)` 来纠正。但这个纠正逻辑 **假设 canonical 路径已经 +1**，而如果 canonical 选了 WAIT 但 micro 选了 WARN，纠正就出错了。 |
| 影响 | `warn_count` / `wait_count` / `soft_count` 不准确，影响统计报告但不影响决策。 |

---

### BUG-8: Jacobian uncertainty 使用固定操作点 `z = [0.5, 0.5, 0.5, 0.5]`

| 字段 | 内容 |
|------|------|
| 位置 | [structured_basis_head.py:306-307, 345](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py#L306-L307) |
| 问题 | `predict_cost_uncertainty_from_var()` 和 `predict_risk_uncertainty_from_var()` 在计算 Jacobian 时使用固定操作点 `z = np.full(4, 0.5)`，而不是实际的 belief mean。 |
| 影响 | 对于 feature 值远离 0.5 的细胞（如边界值 0.1 或 0.9），Jacobian 近似偏差变大。交叉项 `z₀z₁` 的 Jacobian `∂/∂z₀ = z₁ = 0.5`，但实际如果 `z₁ = 0.1`，Jacobian 应该是 0.1。 |
| 性质 | 已知的设计妥协，但没有文档说明误差边界。在 `WorldWeights` 驱动的 `baseline_v2` 场景中（feature 通常在 0.3-0.7 范围），影响不大；但在 GTET/DTMB 中 feature 可能取极端值，影响 uncertainty 驱动的规划。 |

---

## 2. 🟠 暂时性 / Placeholder 代码

### TEMP-1: `predictor_mode = "P4"` 标注为 "temp default"

| 字段 | 内容 |
|------|------|
| 位置 | [lattice_v2_runner.py:173](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py#L173) |
| 内容 | `predictor_mode: str = "P4"  # "P1"=E[z], "P2"=MAP, "P3"=route_mix, "P4"=z_masked (temp default)` |
| 说明 | 注释明确标注 `(temp default)`。P4 模式 (`z_masked`) 已经在实验中成为事实标准，但注释没有更新。应确认 P4 是否是最终选择。 |

### TEMP-2: `TODO: Phase 2 ablation` 空 pass

| 字段 | 内容 |
|------|------|
| 位置 | [a1mt_observer_shadow_prob.py:392](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/a1mt_observer_shadow_prob.py#L392) |
| 内容 | `pass  # TODO: Phase 2 ablation` |
| 说明 | `kappa` 的 emission 似然被标注为 "Phase 2 ablation"，目前 `use_kappa_emission=True` 时什么也不做，直接返回 `np.ones_like(grid)`（和 `False` 时一样）。 |

### TEMP-3: `_dtmb_oracle_warned` / `_dtmb_oracle_item_dropped` 猴子补丁

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_helpers.py:250,294,350,371](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_helpers.py#L250) |
| 问题 | 这些状态变量通过 `setattr` 动态添加到 `V2EpisodeState` 上，没有在 dataclass 定义中声明。 |
| 影响 | 类型检查工具无法发现拼写错误。没有初始化保证。 |

### TEMP-4: BCICTv4 中硬编码 Q-value 权重

| 字段 | 内容 |
|------|------|
| 位置 | [internalization_control_tutor_v4.py:215-216](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/internalization_control_tutor_v4.py#L215-L216) |
| 内容 | `Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05` |
| 说明 | 五个权重 `(1.0, 2.0, 1.5, 1.0, -0.05)` 和两个权重 `(2.0, -1.5, 2.0)` 完全硬编码，没有对应的 config 字段或 dataclass 属性。不可配置、不可比较。 |

### TEMP-5: oracle 修改 ground truth risk 作为 shield 替代

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_helpers.py:366-370](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_helpers.py#L366-L370) |
| 内容 | `s.gridmap.true_risk[br, bc] *= 0.15  # oracle shield effect` |
| 说明 | 注释 "oracle shield effect" 表明这是 inventory 不可用时的临时替代。应该在有 inventory 的正式管线中做 guard。 |

---

## 3. 🟡 缺失 Safety Guard

### GUARD-1: `BayesianRiskHead.update_from_label` 缺少 `np.isfinite` 保护

| 字段 | 内容 |
|------|------|
| 位置 | [risk_model.py:94-96](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/risk_model.py#L94-L96) |
| 参见 | BUG-3 |

### GUARD-2: `BasisRiskHead.update_from_label` 缺少 `np.isfinite` 保护

| 字段 | 内容 |
|------|------|
| 位置 | [structured_basis_head.py:230-233](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py#L230-L233) |
| 参见 | BUG-3 |

### GUARD-3: `preference_posterior.py` normalize 只用 `+ 1e-10` 而没有 check `p.sum() == 0`

| 字段 | 内容 |
|------|------|
| 位置 | [preference_posterior.py:45](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/preference_posterior.py#L45) |
| 问题 | `return p / (p.sum() + 1e-10)` — 如果所有 likelihood 为 0，结果不是 uniform 而是全 0 除以 1e-10，得到一个几乎全 0 的向量。 |
| 影响 | 退化的 posterior 不是合法的概率分布。下游采样或 argmax 会产生未定义行为。 |
| 修复 | 用 `s = p.sum(); return p / s if s > 1e-10 else np.ones_like(p) / len(p)`。 |

---

## 4. ⚪ Dead Code / 冗余

### DEAD-1: `a1mt_observer_shadow_prob.py` L485-487 MAE 计算被 L490-501 覆盖

| 字段 | 内容 |
|------|------|
| 位置 | [a1mt_observer_shadow_prob.py:484-501](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/teachers/a1mt_observer_shadow_prob.py#L484-L501) |
| 问题 | L485 用 `np.mean(np.sqrt(errs_shadow))` 计算 MAE（错误公式：应该是绝对值不是方差的平方根），然后 L490-501 重新用正确的 `abs()` 计算并覆盖同一字段。L485-487 的赋值 **永远不会被读取**。 |

```python
# L484-487 (DEAD — 立即被覆盖)
diag.rmse[dim] = float(np.sqrt(np.mean(errs_shadow)))
diag.mae[dim] = float(np.mean(np.sqrt(errs_shadow)))  # ← 错误公式
diag.rmse_frozen[dim] = float(np.sqrt(np.mean(errs_frozen)))
diag.mae_frozen[dim] = float(np.mean(np.sqrt(errs_frozen)))  # ← 错误公式

# L490-501 (正确 — 覆盖上面的 mae)
diag.mae[dim] = float(np.mean(abs_errs))   # ← 正确
diag.mae_frozen[dim] = float(np.mean(abs_errs_f))  # ← 正确
```

### DEAD-2: `observation_model.py` V0 路径 (`observe_cost_risk`)

| 字段 | 内容 |
|------|------|
| 位置 | [observation_model.py:41-96](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/observation_model.py#L41-L96) |
| 问题 | 文件头部标注 `DEPRECATED`。`observe_cost_risk()` 返回 `(cost_obs, risk_obs, cost_var, risk_var)` 标量，这是 V0 直接观察 cost/risk 的接口，已被 `observe_features()` 替代。 |
| 影响 | 88 行代码无引用。 |

### DEAD-3: `planner_astar.py` V0 函数 (`bounded_astar`, `plan_next_action`)

| 字段 | 内容 |
|------|------|
| 位置 | [planner_astar.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py) 前 200 行 |
| 问题 | V0 的 `cell_cost()`, `bounded_astar()`, `plan_next_action()` 使用 `(belief_cost_mean, belief_risk_mean, belief_cost_var)` 三数组接口。已被 V2 路径完全替代。 |
| 影响 | ~200 行代码无引用，增加维护负担。 |

### DEAD-4: `warned_lane_bias` 在 RSA 路径下永远为空

| 字段 | 内容 |
|------|------|
| 位置 | [lattice_v2_runner.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2_runner.py) state 初始化 |
| 问题 | `warned_lane_bias` 只在 legacy `warning_variant == "legacy_bias"` 路径下被填充。所有 RSA 路径(`rsa_obs_l0`, `rsa_obs_s1`, `rsa_obs_s1_trust`, `rsa_plus_phase10`)使用 `apply_planner_adapter()` 直接写 `warned_cell_extra`。 |
| 影响 | RSA 用户需要看到 `warned_lane_bias` 始终为空。不是 bug 但增加困惑。 |

---

## 5. 修复优先级总结

| 优先级 | ID 列表 | 说明 |
|--------|---------|------|
| **P0 — 立即** | BUG-3, GUARD-1, GUARD-2 | NaN 保护缺失随时可能导致 silent 崩溃 |
| **P0 — 架构** | BUG-1 | runner 不支持 SlowFast 生命周期 — 阻碍 Phase 2B 部署 |
| **P1 — 短期** | BUG-2, BUG-4, BUG-5, BUG-7 | 功能性 bug，影响边际情况 |
| **P1 — 清理** | TEMP-1, TEMP-2, TEMP-3 | 标注 / 猴子补丁正式化 |
| **P2 — 低优** | BUG-6, BUG-8, DEAD-1~4, GUARD-3, TEMP-4, TEMP-5 | 设计妥协、死代码、硬编码 |
