# Scenario / Cell / Map 现状报告

> **参考文档** — 当前 Pedagogical IP 项目中所有场景类型、网格结构、Cell 类型、特征系统和 Agent 决策空间的完整盘点。
> 最后更新：2026-03-30。

---

## 1. Grid 结构总览

项目使用两个层级的网格系统：

| 层级 | 文件 | 作用 |
|------|------|------|
| **低层 GridMap** | `map_generator.py` | 通用网格：cell_types, true_cost, true_risk |
| **高层 Lattice V2** | `lattice_v2.py` | 7-row 分段迫选网格，带 4D 特征向量 |

### 1.1 低层 GridMap

```python
@dataclass
class GridMap:
    height: int                     # 行数
    width: int                      # 列数
    cell_types: np.ndarray          # (H, W) int — CellType 枚举
    true_cost: np.ndarray           # (H, W) float — 真实代价
    true_risk: np.ndarray           # (H, W) float — 真实风险概率
    object_spawn: tuple[int, int]   # 物品生成位置
    target_pos: tuple[int, int]     # 目标位置
    agent_start: tuple[int, int]    # Agent 起始位置
    door_positions: list            # 门的位置列表
```

支持两种生成方式：
- **Hand-designed**: 8×8 固定地图，用于基础测试
- **Procedural**: `generate_random_map()` 参数化随机生成

### 1.2 高层 Lattice V2

**固定 7 行结构，可变列宽：**

```
Row 0: ████████████████████  (wall)
Row 1: ┃ risky lane ────── ┃  (straight, short, has traps)
Row 2: ┃ corridor/merge ── ┃  (entry/exit between segments)
Row 3: ┃ safe lane main ── ┃  (longer, zigzag detour)
Row 4: ┃ wall/detour vert ┃  (detour vertical transitions)
Row 5: ┃ safe lane detour  ┃  (detour horizontal path)
Row 6: ████████████████████  (wall)
```

**设计核心**：
- 两条路径共享相同列范围
- 安全路径通过 row 4-5 的 zigzag 强制增长，比风险路径更长
- 每个 segment 入口有 gate cell，tutor 可以关闭 risky gate 来阻止 agent 进入危险路径

---

## 2. Cell 类型（CellType 枚举）

```python
class CellType(enum.IntEnum):
    NORMAL       = 0    # 普通可通行，cost=1.0, risk=0.0
    WALL         = 1    # 不可通行，cost=∞
    HIGH_COST    = 2    # 高代价可通行，cost=3.0~9.0
    RISKY        = 3    # 有风险可通行，cost=1.0, risk=0.05~0.60
    LOCKED_DOOR  = 4    # 上锁的门，cost=∞ 直到被解锁
    TARGET       = 5    # 目标位置
    OBJECT_SPAWN = 6    # 物品生成位置
```

### 各类型详细属性

| CellType | true_cost | true_risk | 可通行 | 干预相关 |
|----------|:---------:|:---------:|:------:|:--------:|
| NORMAL | 1.0 | 0.0 | ✅ | — |
| WALL | ∞ | 0.0 | ❌ | — |
| HIGH_COST | 3.0~9.0 | 0.0 | ✅ | 可能触发路径偏好 |
| RISKY | 1.0 | 0.05~0.60 | ✅ | WARN / ITEM_DROP 减缓风险 |
| LOCKED_DOOR | ∞→1.0 | 0.0 | ❌→✅ | UNLOCK 解锁后可通行 |
| TARGET | 1.0 | 0.0 | ✅ | 到达即完成 |
| OBJECT_SPAWN | 1.0 | 0.0 | ✅ | 物品拾取点 |

---

## 3. 特征系统（4D Feature Vector）

每个 cell 携带 4 维特征向量 `features[r, c] ∈ ℝ⁴`：

```python
FEATURE_DIM = 4
F_LANE_ID    = 0    # 车道标识：0.0=risky, 1.0=safe, 0.5=corridor
F_GATE_FLAG  = 1    # 是否是 gate/entry cell：1.0=yes, 0.0=no
F_TEXTURE_1  = 2    # 纹理特征 1：越高→越像 trap
F_TEXTURE_2  = 3    # 纹理特征 2：越高→越像 trap
```

### 特征-风险映射

| 特征类型 | F_TEXTURE_1 | F_TEXTURE_2 | 对应风险 |
|----------|:-----------:|:-----------:|:--------:|
| `_safe_feature` | 0.0~0.1 | 0.0~0.1 | 极低 |
| `_lane_feature(mild)` | 0.1~0.2 | 0.05~0.15 | 低 |
| `_weak_cue_feature` | 0.3~0.5 | 0.2~0.4 | 中等（trap 附近） |
| `_trap_feature` | 0.8~0.95 | 0.7~0.9 | 高（trap cell） |

### Latent Mode

当 `latent_mode=True`（默认）时：
- `WorldWeights` 从 feature vector 线性生成 true_cost 和 true_risk
- Agent 只能看到 feature vector，不能直接观察 true_risk
- Agent 必须通过 `CostRiskModel` 来估计 risk → 这是 κ̂ 的信息来源

---

## 4. Branching 与决策空间

### 4.1 BranchAttributes（Agent 观测）

Agent 在每个分叉点看到的是抽象化的 branch 属性：

```python
@dataclass
class BranchAttributes:
    safety_score: float = 0.5       # 安全性（高=更安全）
    temptation_score: float = 0.0   # 诱惑强度
    texture_novelty: float = 0.0    # 纹理新鲜度
    shortcut_bonus: float = 0.0     # 捷径奖励
    risk_penalty: float = 0.0       # 风险惩罚（干预可修改此项）
```

### 4.2 Preference Types（θ）

Agent 的偏好类型决定 utility 权重：

```python
PREF_REWARD = {
    "safe":     [1.0, -0.5,  0.0, 0.0],   # 偏好安全
    "shiny":    [0.0,  1.0,  0.5, 0.0],   # 被诱惑吸引
    "risky":    [0.0,  0.5,  1.0, 0.0],   # 寻求刺激
    "shortcut": [0.0,  0.0,  0.0, 1.5],   # 追求速度
    "neutral":  [0.3,  0.3,  0.3, 0.1],   # 无明显偏好
}
```

### 4.3 Agent Utility 函数

$$U(\pi|\theta) = R_{goal}(\pi) + \lambda_\theta \cdot \langle \vec{w}_\theta, \vec{x}_\pi \rangle - J_{risk}(\pi)$$

选择概率：

$$P_{mix}(\pi|s,\theta) = (1-\epsilon) \cdot \text{softmax}(\beta \cdot U) + \epsilon \cdot \frac{1}{|\Pi|}$$

默认参数：β=4.0（softmax 温度），ε=0.1（lapse rate）。

---

## 5. 场景家族（Scenario Families）

项目包含 **11 个注册场景家族**，各自对应不同的失败机制和干预杠杆：

### 5.1 注册表

```python
SCENARIO_REGISTRY = {
    "baseline_v2":            # 默认 V2 网格，回归锚点
    "fork_trap":              # 模糊岔路陷阱 — WARN
    "hazard_belt":            # 不可避免风险带 — ITEM_DROP
    "deadline_gate":          # 紧迫期限 + 门控捷径 — UNLOCK
    "delayed_corridor":       # 延迟揭示风险 — prefix-aware WARN
    "distractor_cue":         # 误导性局部线索 — WARN + transfer
    "funnel_trap":            # 漏斗陷阱
    "elcb":                   # Explore-Learn-Commit-Burn
    "elcb_po":                # ELCB 部分可观测版
    "temptation_corridor":    # 诱惑走廊
    "joint_conflict_corridor": # 目标-偏好冲突走廊
}
```

### 5.2 核心六大家族详述

#### baseline_v2 — 回归锚点

| 属性 | 值 |
|------|:--:|
| 网格 | 7×(variable), 3 segments |
| 路径结构 | risky (直线, 短) vs safe (zigzag, 长) |
| 风险 | stochastic trap cells |
| 干预 | WARN (关闭 risky gate) |
| 难度梯度 | trap 概率 easy=50% / medium=70% / hard=90% |

#### fork_trap — 模糊岔路

| 属性 | 值 |
|------|:--:|
| 网格 | 7×(~10), 单 fork |
| 核心特征 | 前 `trap_depth` 个 cell 的线索模糊 |
| cue_ambiguity | easy=0.3 / medium=0.6 / hard=0.9 |
| trap_risk | easy=0.30 / medium=0.45 / hard=0.60 |
| risky_row | 随机分配到 row 1 或 row 3 |
| 干预杠杆 | **WARN**（时机：robot-belief-timed） |
| 失败模式 | risk — agent 在 trap 暴露前已 commit |

#### hazard_belt — 不可避免风险带

| 属性 | 值 |
|------|:--:|
| 网格 | 7×(~20), 3 segments (safe + belt + safe) |
| 核心特征 | belt segment 两条 lane 都是 RISKY |
| belt_risk | easy=0.25 / medium=0.30 / hard=0.35 |
| belt_regime | `unavoidable`（无安全路径）或 `near_unavoidable`（bypass 代价=3.0） |
| 干预杠杆 | **ITEM_DROP**（减半 belt 风险） |
| 失败模式 | risk — 穿越 belt 时被风险事件击中 |

#### deadline_gate — 紧迫期限

| 属性 | 值 |
|------|:--:|
| 网格 | 7×(~25), shortcut on row 1 + long path on rows 3-5 |
| 核心特征 | 捷径被 LOCKED_DOOR 门控，默认关闭 |
| time_ratio | easy=1.15 / medium=1.10 / hard=1.05（极紧） |
| 捷径风险 | **0.0**（纯拓扑辅助，不是风险交易） |
| 安全路径风险 | RISKY cells = 1-2/segment, risk=0.15~0.25 |
| 干预杠杆 | **UNLOCK**（打开捷径门） |
| 失败模式 | timeout — 安全路径太长，超时 |

#### delayed_corridor — 延迟揭示

| 属性 | 值 |
|------|:--:|
| 核心特征 | 前 `safe_prefix` 个 cell 风险低，之后急剧升高 |
| deep_risk | easy=0.35 / medium=0.45 / hard=0.55 |
| commitment | 一旦通过前缀区域，回头超过 deadline |
| 干预杠杆 | **WARN**（必须在 agent commit 前发出） |
| 失败模式 | commitment — 进入后回不来 |

#### distractor_cue — 误导线索

| 属性 | 值 |
|------|:--:|
| 核心特征 | 局部 cue 指向错误方向 |
| 干预杠杆 | **WARN** + transfer |
| 失败模式 | cue_error — agent 被错误线索误导 |

### 5.3 难度参数总览

| 家族 | Easy | Medium | Hard |
|------|:----:|:------:|:----:|
| fork_trap | cue_ambig=0.3, trap_risk=0.30 | 0.6, 0.45 | 0.9, 0.60 |
| hazard_belt | belt_risk=0.25, bypass+6 | 0.30, +8 | 0.35, +10 |
| deadline_gate | time=1.15×, 3 seg | 1.10×, 4 seg | 1.05×, 4 seg |
| delayed_corridor | safe_pfx=2, deep=0.35 | 3, 0.45 | 3, 0.55 |

---

## 6. CGC-v2 组合目标走廊

独立于 Scenario Families，CGC-v2 实现组合目标推断：

### 6.1 目标体系

| 类型 | 目标 | reward 权重 [safety, tempt, novelty, speed] |
|------|------|:-------------------------------------------:|
| **Atomic** | collect_red | [0.0, 2.5, 0.5, 0.0] |
| **Atomic** | avoid_blue | [2.0, -1.0, 0.0, 0.0] |
| **Atomic** | use_safe | [3.0, -0.5, 0.0, 0.0] |
| **Atomic** | reach_fast | [0.0, 0.0, 0.0, 3.0] |
| **Composite** | collect_red + avoid_blue | 两者均值 |
| **Composite** | collect_red + use_safe | 两者均值 |
| **Composite** | avoid_blue + use_safe | 两者均值 |
| **Composite** | reach_fast + avoid_blue | 两者均值 |

### 6.2 Episode 类型

| Subtype | 含义 |
|---------|------|
| `goal_aligned` | g 和 θ 自然一致（如 safe θ + safe goal） |
| `goal_conflict` | g 和 θ 冲突（如 shiny θ + safe goal） |
| `goal_boundary` | 模糊——需要观察才能判断 |

### 6.3 诊断窗口

| Step | 观察维度 |
|------|---------|
| Step 1 | preference-driven（fork 处的 lure-sensitive 选择） |
| Step 2 | goal-driven（部分信息后的纠正/承诺） |
| Step 3 | constraint-sensitive（是否尊重约束） |

---

## 7. TIC-v4 教学内化走廊

5 阶段教学内化协议：

| Phase | 名称 | Episodes | 特点 |
|:-----:|------|:--------:|------|
| **A** | Tutor present | 10 | 全量 tutor 干预 |
| **B** | Autonomy transfer | 4 | 无 tutor，测试迁移 |
| **C** | Sparse valid advice | 4 | 偶尔正确建议 |
| **D** | Sparse invalid advice | 4 | 偶尔错误建议 |
| **E** | Beneficial novelty probe | 4 | 有益新奇探测 |

### TIC-v4 Episode Subtypes

| Subtype | 阶段 | 测试维度 |
|---------|:----:|---------|
| temptation_repeat | A, B | 重复诱惑抗性 |
| self_discovery_teach | A | 自发现 vs 教学 |
| warn_rescue | A | 警告后的救援行为 |
| boundary_obs | A | 边界观察 |
| verified_warn | A | 验证后的警告信任 |
| self_discovery_needed | A, B | 需要自发现的场景 |
| false_suppression_cost | A, B | 过度抑制代价 |
| sparse_valid_advice | C | 稀疏正确建议 |
| sparse_invalid_advice | D | 稀疏错误建议 |
| beneficial_novelty | B, E | 有益新奇探索 |

---

## 8. 干预系统与 Cell 的交互

### 8.1 干预→Cell 映射

| 干预类型 | Cell-level 效果 | 语义 |
|:--------:|----------------|------|
| **WAIT** | 无 | 不干预，观察 |
| **WARN** | ↑ risk_penalty on risky branches | 信念证据："这条路有风险" |
| **UNLOCK** | LOCKED_DOOR → NORMAL (cost ∞→1.0) | 打开门控捷径 |
| **ITEM_DROP** | ↓ risk on RISKY cells (belt) | 降低遍历风险 |

### 8.2 干预→BranchAttributes 映射（ConsequenceGroundedRollout）

```python
WARN     →  risk_penalty += 0.15 on risky;  safety_score += 0.05 on safe
UNLOCK   →  risk_penalty -= 0.10 (open paths, reduce uncertainty)
ITEM_DROP → risk_penalty -= 0.08 (hazard mitigation)
```

### 8.3 家族-干预选择性

| 家族 | 最佳干预 | SelGap |
|------|:--------:|:------:|
| fork_trap | WARN | +0.05 |
| hazard_belt | ITEM_DROP | +0.03 |
| deadline_gate | UNLOCK | +0.04 |

---

## 9. 场景配置参数

### 9.1 ScenarioConfig

```python
@dataclass
class ScenarioConfig:
    family_name: str               # 家族名
    difficulty: str                # easy / medium / hard
    primary_intervention: str      # 最自然的干预类型
    cue_reliability: float = 1.0   # 1.0=честный, 0.0=不相关
    hazard_density: float = 0.0    # 危险带占比
    requires_gate: bool = False    # 是否需要 UNLOCK gate
    requires_item: bool = False    # 是否需要 ITEM_DROP
    expected_failure_mode: str     # risk / timeout / cue_error / commitment
    gate_mode: str                 # "block_risky" / "unlock_shortcut"
    belt_regime: str               # "unavoidable" / "near_unavoidable"
    commitment_cells: list         # 过了就回不来的 cell
```

### 9.2 FamilyConfig

```python
@dataclass
class FamilyConfig:
    max_steps: int           # 最大步数（紧迫度控制）
    risk_budget: float       # 风险预算
    prior_risk_mean: float   # Agent 初始风险先验均值
    prior_risk_var: float    # Agent 初始风险先验方差
    search_budget: int       # 搜索预算
    budget_class: int        # 预算类别
```

---

## 10. Map 生成流程

### 10.1 Scenario Family 生成管线

```
generate_scenario(family, seed, difficulty)
  │
  ├── 选择 family 参数表（难度相关）
  ├── 计算网格尺寸 (7 × W)
  ├── 初始化全 WALL
  ├── 开凿 corridor (row 2)
  │
  ├── 对每个 segment:
  │     ├── 开凿 risky lane (row 1)
  │     ├── 开凿 safe lane (row 3) + zigzag detour (rows 4-5)
  │     ├── 设置 entry/exit gates
  │     ├── 放置 trap / weak_cue / features
  │     └── 计算 risk 分布
  │
  ├── BFS 计算最短路径
  ├── 设置时间预算 t_max = ratio × shortest_safe
  │
  └── if latent_mode:
        ├── WorldWeights: features → cost, risk
        └── Family-specific post-processing（保留结构性风险）
```

### 10.2 CGC-v2 生成管线

```
generate_cgc_v2_episode(spec)
  │
  ├── 选择 goal + constraint
  ├── 生成 fork corridor（基于 scenario_families 基础设施）
  ├── 配置 multi-step diagnostic window
  └── 输出 (GridMap, FamilyConfig, LatticeV2Meta, CGCEpisodeSpec)
```

---

## 11. 当前状态与已知问题

### 11.1 活跃使用中

| 组件 | 状态 | 使用者 |
|------|:----:|--------|
| CellType 7-type enum | ✅ Stable | 所有 map generators |
| Lattice V2 7-row structure | ✅ Canonical | scenario_families, CGC-v2 |
| 4D feature vector | ✅ Canonical | latent mode, cost_risk_model |
| BranchAttributes 5D | ✅ Canonical | ActionPredictor, JointPosterior |
| 11 scenario families | ✅ Registered | experiments, curriculum |
| CGC-v2 goal structure | ✅ Canonical | Task 7 compositional goals |
| TIC-v4 5-phase protocol | ✅ Canonical | internalization experiments |

### 11.2 已知局限

1. **GridMap 没有 connectivity 保证**：`generate_random_map()` 不检查可达性。Lattice V2 无此问题（结构保证连通）。

2. **latent_mode post-processing 是 family-specific**：每个家族必须手动 clamp risk/cost 以保留结构性风险（如 hazard_belt 的 belt cells）。这是有意设计——latent 映射不应消除家族特征。

3. **Fixed 7-row structure**：所有 Lattice V2 场景共享 7 行。更复杂的拓扑（如多层楼、非线性网格）暂不支持。

4. **CellType 未扩展**：当前 7 种 CellType 覆盖了所有已有场景。CGC-v2 和 TIC-v4 不引入新 CellType——它们通过 feature vector 和 goal weights 来表达复杂度。

5. **BranchAttributes 是 scenario→posterior 的桥接抽象**：当前由手工构造或 `ConsequenceGroundedRollout` 生成。全管线集成（scenario → BranchAttributes → posterior → curriculum）尚未端到端测试。

### 11.3 冻结与可扩展

| 组件 | 冻结？ | 备注 |
|------|:------:|------|
| CellType enum | 可扩展 | 添加新类型不破坏现有逻辑 |
| Lattice V2 结构 | 冻结 | 7-row 是所有家族的基础 |
| Feature dim=4 | 冻结 | 所有 WorldWeights 和特征函数都依赖 4D |
| BranchAttributes | 冻结 | 5 个字段，intervention 通过修改这些字段起作用 |
| SCENARIO_REGISTRY | 可扩展 | 添加新家族只需注册函数 |
| GOAL_WEIGHTS | 可扩展 | 添加新 atomic goal 需要新的 4D 权重向量 |

---

## Appendix: 完整文件清单

### 环境/场景相关

| 文件 | 大小 | 作用 |
|------|:----:|------|
| `map_generator.py` | 6KB | CellType, GridMap, default/random map |
| `lattice_v2.py` | 14KB | 7-row Lattice V2 结构, SegmentMeta, BFS |
| `lattice_v2_env.py` | 9KB | Lattice V2 gym-style environment |
| `lattice_v2_runner.py` | 29KB | 完整 runner (agent+tutor+observer) |
| `scenario_families.py` | 120KB | 11 个场景家族 + 全部参数表 |
| `map_families.py` | 28KB | 低层 map family 基础工具 |
| `cgc_v2_family.py` | 12KB | CGC-v2 组合目标走廊 |
| `compositional_goal_corridor.py` | 14KB | CGC-v1 (legacy) |
| `compositional_goal_corridor_v2.py` | 9KB | CGC-v2 factor-vector 设计 |
| `teaching_internalization_corridor.py` | 9KB | TIC base |
| `teaching_internalization_corridor_v2.py` | 4KB | TIC-v2 |
| `teaching_internalization_corridor_v3.py` | 6KB | TIC-v3 |
| `teaching_internalization_corridor_v4.py` | 4KB | TIC-v4 5-phase canonical |
| `pedagogical_grid.py` | 15KB | PedagogicalGridEnv 基础 |
| `observation_mask.py` | 4KB | Agent 观测掩码 |
| `semantic_subspace.py` | 4KB | 语义子空间定义 |
| `benchmark_generator.py` | 1KB | 基准生成工具 |
| `persistent_profile_mixed_reveal.py` | 12KB | 持久 profile 混合揭示 |

### Agent 决策相关

| 文件 | 大小 | 作用 |
|------|:----:|------|
| `stochastic_agent_policy.py` | 3KB | BranchAttributes, PREF_REWARD, choice model |
| `cost_risk_model.py` | ~10KB | WorldWeights: features → cost/risk |
| `internalization_state_v3.py` | ~10KB | 5D 真实内化状态 (κ,τ,ν,γ_spec,γ_gen) |
