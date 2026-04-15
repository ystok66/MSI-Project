# 场景与 Cell 机制总览报告

> 最后更新：Phase 2B 完成后（2026-04-07）
> 覆盖范围：`src/envs/` 全部场景生成器 + `src/agents/` 预测头和 feature 系统

---

## 1. Cell 类型体系

所有场景共享 `CellType` 枚举（定义于 `map_generator.py`），共 7 种：

| CellType | 值 | 含义 | 默认 cost | 默认 risk |
|----------|---|------|-----------|-----------|
| `NORMAL` | 0 | 可通行普通格 | 1.0 | 0.0 |
| `WALL` | 1 | 不可通行（物理障碍）| ∞ | 0.0 |
| `HIGH_COST` | 2 | 可通行但代价高昂 | 3.0~7.0 | 0.0 |
| `RISKY` | 3 | 可通行但有死亡概率 | 1.0 | 0.15~0.70 |
| `LOCKED_DOOR` | 4 | 需 teacher UNLOCK 才可通行 | ∞→1.0 | 0.0 |
| `TARGET` | 5 | 目标格（delivery point）| 1.0 | 0.0 |
| `OBJECT_SPAWN` | 6 | 物品生成点（pick-up）| 1.0 | 0.0 |

> [!NOTE]
> 在 V2 lattice 系列场景中，cell type 仅决定 **拓扑**（可通行性与门锁状态）。实际的 cost 和 risk 数值由 `WorldWeights` 从 4D feature vector **推导出来** — 这是 feature-driven latent 场景的核心设计。

---

## 2. 4D Feature Vector 体系

定义于 `lattice_v2.py`。所有 V2 场景的每个 cell 携带一个 4 维特征向量 `z = [z₀, z₁, z₂, z₃]`：

| 维度 | 常量名 | 含义 | 值域 | 角色 |
|------|--------|------|------|------|
| `z₀` | `F_LANE_ID` | 车道/分支标识 | 0~1 连续 | 弱 cost 调制 |
| `z₁` | `F_GATE_FLAG` | 门/入口标记 | 0 或 1 | 弱 risk 调制 |
| `z₂` | `F_TEXTURE_1` | 纹理通道 1 | [0, 1] | **强 risk 驱动** |
| `z₃` | `F_TEXTURE_2` | 纹理通道 2 | [0, 1] | **强 risk 驱动** |

### Feature → Cost/Risk 映射（WorldWeights）

`WorldWeights`（`cost_risk_model.py:L117`）定义全局映射：

```
true_cost(z) = max(w_cost · z + b_cost, 0.1)         (线性 + floor)
true_risk(z) = σ(w_risk · z + b_risk)                 (logistic)
```

其中：
- `w_cost` ~ Uniform(-0.3, 0.3)⁴, `b_cost = 1.0`
- `w_risk[0:2]` ~ weak ([-0.5, 0.5]), `w_risk[2:4]` ~ strong ([1.5, 4.0])
- `b_risk` ~ [-3.0, -1.5]（使多数 cell 低风险）

### Feature 生成函数（成熟 / 已冻结）

定义于 `lattice_v2.py:L379-388`，被所有 V2 场景调用：

| 函数 | z₂ 范围 | z₃ 范围 | 语义 |
|------|---------|---------|------|
| `_safe_feature` | [0.0, 0.1] | [0.0, 0.1] | 安全格（低纹理）|
| `_trap_feature` | [0.80, 0.95] | [0.70, 0.90] | 陷阱格（高纹理）|
| `_weak_cue_feature` | [0.30, 0.50] | [0.20, 0.40] | 模糊线索格 |
| `_lane_feature` | [0.0, 0.20] | [0.0, 0.15] | 普通车道格 |

> [!IMPORTANT]
> `harder_baseline_v2` 使用了 **不同的** feature 分布（`_hb_*`），故意缩小 safe/trap 之间的 texture gap，详见 §3.4。

---

## 3. 场景体系

系统有 **三代** 场景实现。按成熟度和编号划分：

### 3.1 成熟 / 主线场景（V2 Lattice Registry, `SCENARIO_REGISTRY`）

#### 3.1.1 `baseline_v2` — 默认基准线

| 项目 | 值 |
|------|---|
| **源文件** | `lattice_v2.py` via `scenario_families.py:L84` |
| **网格** | 7 行 × 可变宽 (3-5 段) |
| **拓扑** | Safe/Risky 双车道 + zigzag detour (每段) |
| **feature 驱动** | ✅ WorldWeights 生成 cost/risk |
| **主要杠杆** | WARN（block risky lane entry gate）|
| **状态** | ✅ **成熟 / Canonical** |
| **用途** | 所有 predictor head 类型的标准基准 |

结构示意（每段 5 列）：
```
Row 0: wall
Row 1: risky lane — 直线路径（短但可能致死）
Row 2: wall separator
Row 3: safe lane — zigzag detour（长但安全）
Row 4-6: detour 空间
```

#### 3.1.2 `fork_trap` — 模糊分支陷阱

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L116` |
| **网格** | 7 行 × 可变宽 |
| **拓扑** | 两分支，近对称 local cues；一支内藏 trap |
| **核心难点** | trap 在 `trap_depth` 步之后才可辨识 |
| **主要杠杆** | WARN |
| **状态** | ✅ 成熟 |

#### 3.1.3 `hazard_belt` — 不可回避风险带

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L379` |
| **网格** | 7 行 × 可变宽 |
| **拓扑** | 走廊中段有一排 risky cell 带，必须穿越 |
| **核心难点** | 风险不可避，只能用 ITEM_DROP 降低伤害 |
| **主要杠杆** | ITEM_DROP |
| **状态** | ✅ 成熟 |

#### 3.1.4 `deadline_gate` — 截止时间 + 门控

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L635` |
| **网格** | 7 行 × 可变宽 |
| **拓扑** | 长绕路 vs 门捷径；deadline 迫使走门 |
| **核心难点** | 不 UNLOCK 则来不及到达 |
| **主要杠杆** | UNLOCK |
| **状态** | ✅ 成熟 |

#### 3.1.5 `delayed_corridor` — 延迟承诺走廊

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L911` / `L2198` |
| **拓扑** | 长走廊 + commitment point；过了 commitment 就不能回退 |
| **核心难点** | commitment 后 backtrack 超过 deadline |
| **主要杠杆** | WARN (early enough) |
| **状态** | ✅ 成熟 |

#### 3.1.6 `distractor_cue` — 干扰线索

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L1117` / `L2406` |
| **拓扑** | 双车道，一侧有诱导性 feature 使 agent 误判 |
| **核心难点** | agent 被虚假 low-texture cue 引诱到实际高风险路 |
| **主要杠杆** | WARN |
| **状态** | ✅ 成熟 |

#### 3.1.7 `funnel_trap` — 漏斗陷阱

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L1302` |
| **拓扑** | 多条路径逐段收窄成 funnel，最终被迫进入 risky |
| **核心难点** | early 干预窗口很短，错过后无退路 |
| **主要杠杆** | WARN + early timing |
| **状态** | ✅ 成熟 |

#### 3.1.8 `elcb` — Epistemic Lane-Choice Benchmark

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L1729` |
| **拓扑** | 多车道选择，每条有不同 cue reliability |
| **核心难点** | 需要 agent belief 更新来区分可用信息 |
| **主要杠杆** | WARN / WAIT |
| **状态** | ✅ 成熟 |

#### 3.1.9 `elcb_po` — Partially Observable ELCB

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L1965` |
| **拓扑** | 与 `elcb` 结构相同，但 `patch_radius` 限制局部观测 |
| **核心难点** | agent 只看到附近 cell，更依赖 belief propagation |
| **主要杠杆** | WARN / WAIT |
| **状态** | ✅ 成熟 |

#### 3.1.10 `temptation_corridor` — 诱惑走廊

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L2616` |
| **拓扑** | 一条"闪亮"捷径（高 gate_flag）vs 安全长路 |
| **核心难点** | agent 可能被 preference-aligned cue 引诱走捷径 |
| **主要杠杆** | WARN |
| **状态** | ✅ 成熟 |

#### 3.1.11 `joint_conflict_corridor` — 联合冲突走廊

| 项目 | 值 |
|------|---|
| **源文件** | `scenario_families.py:L2824` |
| **拓扑** | goal cue 和 temptation cue 指向不同路径 |
| **核心难点** | agent 必须同时解析目标和诱惑的冲突 |
| **主要杠杆** | WARN（区分 goal vs temptation）|
| **状态** | ✅ 成熟 |

---

### 3.2 Complex 场景（多阶段 / 多瓶颈）

#### 3.2.1 `goal_preference_temptation_entanglement_lattice` (GTET-L)

| 项目 | 值 |
|------|---|
| **源文件** | `gtet_lattice.py` (889 行) |
| **网格** | 13-17 行 × 40-60 列 |
| **阶段** | 3 阶段：epistemic fork → temptation fork → terminal belt |
| **分支** | Stage1: 3 支；Stage2: 2-3 支/parent；Stage3: 2 路 + optional 快速锁定道 |
| **特有机制** | **Goal/Preference/Temptation 纠缠**：同一 prefix 行为可解释为多种 (g, θ, z) 假设 |
| **Sidecar metadata** | `GTETMeta`: goal_cue_tags, temptation_cue_tags, preference_cue_tags |
| **Feature 类型** | `_goal_cue_feature`, `_tempt_cue_feature`, `_pref_cue_feature`, `_belt_feature` |
| **主要杠杆** | WARN (mixed) + UNLOCK (Stage 3 fast lane) |
| **Difficulty** | easy/medium/hard → cue_reliability 0.70/0.50/0.35 |
| **状态** | ✅ **成熟 / Canonical complex family** |

3 阶段瓶颈：
1. **Stage 1 — Epistemic**：分支的 goal cue 有 ambiguity overlap → WAIT/WARN
2. **Stage 2 — Temptation**：部分分支有高 lure_strength cue → WARN
3. **Stage 3 — Outcome**：belt zone + optional locked fast lane → UNLOCK/ITEM_DROP

#### 3.2.2 `deep_tree_mixed_bottleneck_lattice` (DTMB-L)

| 项目 | 值 |
|------|---|
| **源文件** | `dtmb_lattice.py` (916 行) |
| **网格** | 13-17 行 × 35-60 列 |
| **阶段** | 3 阶段，各有不同主瓶颈 |
| **分支** | Tree-lattice: 3×2 = 6 终端路由 (medium/hard) |
| **特有机制** | **跨阶段瓶颈漂移**: epistemic → structural → outcome |
| **状态** | ✅ **成熟 / Canonical complex family** |

3 阶段瓶颈：
1. **Stage 1 — Epistemic ambiguity**：weak/misleading cues → WARN/WAIT
2. **Stage 2 — Structural pressure**：LOCKED_DOOR shortcuts → UNLOCK
3. **Stage 3 — Outcome bottleneck**：near-unavoidable hazard belt → ITEM_DROP

---

### 3.3 Transfer 评估场景

#### 3.3.1 `harder_baseline_v2` — Phase 2B

| 项目 | 值 |
|------|---|
| **源文件** | `harder_baseline.py` (309 行) |
| **网格** | 7 行 × 可变宽 (3 段, 3-4 列/段) |
| **拓扑** | 与 baseline_v2 相同的 7 行双车道 |
| **关键修改** | 缩小 safe/trap texture gap (0.15 vs 0.70) |
| **主要用途** | Phase 2B: `basis + slowfast` transfer 实验 |
| **状态** | ✅ **成熟 / 用于 transfer 评估** |

Feature 分布对比（对 baseline_v2）：

| Feature 类型 | baseline_v2 z₂ | harder z₂ | gap |
|-------------|----------------|-----------|-----|
| Safe | [0.0, 0.1] | [0.25, 0.50] | 缩小 |
| Trap | [0.80, 0.95] | [0.45, 0.65] | 大幅缩小 |

其他变化：`time_ratio=1.25`(tighter)、`trap_prob=90%`(higher)

---

### 3.4 V1 旧场景（`map_families.py`）

> [!WARNING]
> V1 场景保留用于向后兼容和特定验证测试。主线实验全部使用 V2 lattice registry。

| 场景 | 网格 | 目的 | 状态 |
|------|------|------|------|
| `semantic_trap` | 10×10 | WARN 验证：右侧 risky 走廊 | 🟡 Legacy / sanity |
| `planning_trap` | 10×10 | UNLOCK 验证：water wall + door | 🟡 Legacy / sanity |
| `exploration_useful` | 10×10 | WAIT 验证：低风险探索区 | 🟡 Legacy / sanity |
| `mixed` | 10×10 | 三阶段混合：WAIT→WARN→UNLOCK | 🟡 Legacy / sanity |
| `door_lattice_sanity` | 9×17 | 门网格验证：UNLOCK/BLOCK 协议 | 🟡 Sanity tool |
| `deceptive_fork` | 6×8 | 最小 MVP：fork + trap + door | 🟡 Sanity tool |

---

### 3.5 V0 手工地图（`map_generator.py`）

| 场景 | 网格 | 状态 |
|------|------|------|
| `generate_default_map` | 8×8 | ⚪ Archived / 不在主实验中使用 |
| `generate_random_map` | 可变 | ⚪ Archived / 不在主实验中使用 |

---

## 4. Predictor Head 体系

Agent 使用 predictor head 从 4D feature 学习 cost/risk。三代实现：

### 4.1 `LatentCostRiskHead` — 4D 线性头

| 项目 | 值 |
|------|---|
| **源文件** | `cost_risk_model.py:L159` |
| **参数** | w_cost ∈ ℝ⁴, b_cost; w_risk ∈ ℝ⁴, b_risk → 共 10 |
| **学习** | Bayesian online update (SGD + gradient clipping) |
| **特点** | 单 episode 内 ~50 cell 样本即可充分训练 |
| **状态** | ✅ **Canonical default** |

### 4.2 `StructuredBasisCostRiskHead` — 结构化基函数头

| 项目 | 值 |
|------|---|
| **源文件** | `structured_basis_head.py` |
| **Cost basis** | φ_c(z) = [1, z₀, z₁, z₂+z₃, z₀z₁, (z₂+z₃)²] → 6D |
| **Risk basis** | φ_r(z) = [1, z₂, z₃, z₂z₃, \|z₂-z₃\|, z₁z₂, z₁z₃] → 7D |
| **参数** | 6 + 7 = 13（vs 线性头的 8）|
| **Uncertainty** | **Jacobian 传播**（Phase 2B 升级，替代旧的 w[:4]² proxy）|
| **状态** | ✅ **成熟 — Phase 2A/2B validated** |
| **效果** | baseline_v2: 1.000 vs linear 0.400; harder: 0.960 vs 0.540 |

### 4.3 `GenericSlowFastPredictor` — 通用双时标包装器

| 项目 | 值 |
|------|---|
| **源文件** | `slow_fast_head.py`（Phase 2B 重写）|
| **架构** | P_slow + P_fast；任意 PredictorProtocol base |
| **Episode 生命周期** | begin: P_fast←copy(P_slow); end: θ_slow←(1-α)θ_slow+αθ_fast |
| **推荐 α** | 0.2（minimum effective value on harder_baseline_v2）|
| **向后兼容** | `SlowFastCostRiskHead()` = factory function wrapping linear head |
| **状态** | ✅ **成熟 — Phase 2B validated** |
| **效果** | basis+slowfast_0.2: 1.000 vs basis_fresh 0.960 vs basis_persist 0.680 |

### 4.4 `PredictorProtocol` — 接口协议

| 项目 | 值 |
|------|---|
| **源文件** | `predictor_protocol.py`（Phase 3A）|
| **方法** | predict_cost, predict_risk, predict_*_uncertainty, update_from_outcome |
| **工具函数** | snapshot_predictor, restore_predictor, extract_theta, extract_theta_components |
| **状态** | ✅ **成熟 / Frozen** |

---

## 5. Teacher 干预机制

### 5.1 四种干预杠杆

| 杠杆 | 含义 | 对应 cell 机制 |
|------|------|---------------|
| **WARN** | 关闭 risky lane entry gate | Gate cell（门锁）|
| **WAIT** | 不干预，让 agent 自主探索 | 无 cell 变化 |
| **UNLOCK** | 打开 LOCKED_DOOR | LOCKED_DOOR → NORMAL |
| **ITEM_DROP** | 投放物品降低 belt 风险 | Hazard belt cell |

### 5.2 门控机制（gate_mode）

| 模式 | 含义 | 使用场景 |
|------|------|---------|
| `block_risky` | 所有门默认开；teacher CLOSES risky-entry 门 | baseline_v2 系列 |
| `unlock_shortcut` | 门默认锁定；teacher OPENS shortcut 门 | deadline_gate, DTMB Stage2 |

---

## 6. Canonical 默认参数（Promotion-Locked）

以下参数已通过实验确认并锁定，不应在不明确原因的情况下修改：

| 参数 | 值 | 确认来源 |
|------|---|---------|
| `warning_variant` | `rsa_obs_s1` | Phase 0 convergence |
| `boredom_weight` (β_bore) | `0.3` | Phase 1B |
| `factor_mode` | `G_THETA` (no-z) | Task 3 GTET selective |
| `patch_radius` | `2` | 默认 |
| `belief_planning_mode` | `True` | 默认 |

---

## 7. 未成熟 / Ablation / 可选组件

### 7.1 Success-Gated Slow Update（未实现）

公式：`g_e = 𝟙[survived_e ∧ goal_e]`；`θ_slow ← (1-αg_e)θ_slow + αg_e·θ_fast`

- **状态**：🔲 **设计已写出，未实现**
- **理由**：Phase 2B 中 α≥0.2 已达天花板，无需 gate 即可工作

### 7.2 Policy Memory (Route C)

- **设计**：让 agent 记住路线选择 pattern 跨 episode
- **状态**：⚪ **未实现 / 高风险**
- **理由**：旧诊断标记为 Route C（最高改造量），仅在 dual-timescale 失败后考虑

### 7.3 Harder 参数变体

Phase 2B 中调试过多种 `harder_baseline` 参数组合：
- 不同 segment 数量 (3-5)
- 不同 texture gap (0.15-0.30)  
- 不同 time_ratio (1.15-1.40)

最终选定当前 canonical config（3 段, gap ≈ 0.15, time_ratio 1.25, trap_prob 90%）。其余实验性参数可归档至 `ablations/phase2b/`。

### 7.4 GTET / DTMB 的 Basis Advantage

Phase 2A 已确认：StructuredBasis 在 GTET/DTMB 上 **无明显优势**（surv 均为 1.000，与 linear 相同），因为这些场景的难点是拓扑而非 feature→risk mapping。

---

## 8. 文件索引

### 核心环境

| 文件 | 内容 | 行数 |
|------|------|------|
| [map_generator.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/map_generator.py) | CellType, GridMap, 默认/随机地图 | 221 |
| [map_families.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/map_families.py) | V1 场景 (A-F) | 726 |
| [lattice_v2.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py) | V2 lattice 核心 + feature 函数 | 412 |
| [scenario_families.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/scenario_families.py) | V2 scenario registry (14 families) | 3057 |
| [gtet_lattice.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/gtet_lattice.py) | GTET-L generator | 889 |
| [dtmb_lattice.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_lattice.py) | DTMB-L generator | 916 |
| [harder_baseline.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/harder_baseline.py) | Harder baseline_v2 | 309 |

### 核心 Agent

| 文件 | 内容 | 行数 |
|------|------|------|
| [cost_risk_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py) | WorldWeights + LatentCostRiskHead | 239 |
| [structured_basis_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/structured_basis_head.py) | 结构化基函数头 | 393 |
| [slow_fast_head.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/slow_fast_head.py) | GenericSlowFastPredictor | ~210 |
| [predictor_protocol.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/predictor_protocol.py) | PredictorProtocol + utilities | ~120 |
