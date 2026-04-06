# 地图与场景系统详解

> 本报告详细介绍当前系统的环境结构：底层网格地图、场景家族、episode 生成流程、信息时序参数，以及它们与教学决策的关系。

---

## 1. 系统中有几层"地图"

当前系统的"地图"实际上有**三层叠加**：

```
Layer 1: 底层网格 (GridMap)
  ↳ 8×8 或 7×W 的 cell 矩阵，含墙壁、通道、风险区
  
Layer 2: 场景家族 (Scenario Families)
  ↳ 6 种不同的拓扑变体（fork, belt, deadline...）
  ↳ 在底层网格上定义特定的空间结构
  
Layer 3: 教学走廊 (TIC / TIC-v4)
  ↳ 当前 canonical 实验使用的抽象二叉分支结构
  ↳ 只用 Layer 1 的基本网格 + 简化的 fork 结构
```

**当前 canonical 实验主要使用 Layer 3**——教学走廊（TIC）。Layer 2 的 6 种场景家族是更早期的设计，代码存在但不在 canonical 主路径中使用。

---

## 2. 底层网格：GridMap

### 2.1 Cell 类型

每个 cell 有一个类型（`CellType`）：

| 类型 | 值 | 含义 | 代价 | 风险 |
|:---:|:-:|------|:----:|:---:|
| **NORMAL** | 0 | 普通可通行 | 1.0 | 0 |
| **WALL** | 1 | 不可通行 | ∞ | 0 |
| **HIGH_COST** | 2 | 高代价可通行 | 3~7 | 0 |
| **RISKY** | 3 | 有概率发生风险事件 | 1.0 | 0.15~0.6 |
| **LOCKED_DOOR** | 4 | 锁门（Tutor 可 UNLOCK） | ∞ → 1.0 | 0 |
| **TARGET** | 5 | 目标位置 | 1.0 | 0 |
| **OBJECT_SPAWN** | 6 | 物品生成点 | 1.0 | 0 |

### 2.2 手工设计的默认地图（8×8）

```
. . . . H H O .      A = Agent 起点 (0,0)
. W W . H H . .      O = 物品生成 (0,6)
. W W . . . . R      T = 目标 (7,7)
. . . D . . R R      W = 墙
. . . . . . R .      D = 锁门
H H . . . . . .      H = 高代价
H H . W W . . .      R = 有风险
. . . W W . . T      . = 普通
```

### 2.3 随机生成地图

`generate_random_map()` 按比例随机放置不同类型的 cell：
- 墙壁：10%
- 高代价：10%
- 风险：8%
- 锁门：1 个

**但是**：当前 canonical 实验**不直接使用这两种 map**。它们是底层基础设施，实际使用的是 TIC 走廊。

---

## 3. 场景家族（Scenario Families）

`scenario_families.py` 定义了 **6 种场景变体**，每种针对不同的失败模式和干预杠杆：

| 家族 | 拓扑结构 | 核心机制 | 主要干预杠杆 | 失败模式 |
|------|---------|---------|:---------:|---------|
| **baseline_v2** | 标准 V2 lattice | 回归基线 | WARN | 风险 |
| **fork_trap** | 模糊分支陷阱 | 深处藏陷阱，表面看两条路差不多 | WARN | 风险/误判 |
| **hazard_belt** | 不可避免的风险带 | 必须穿越危险区域 | ITEM_DROP | 风险 |
| **deadline_gate** | 紧急 deadline + 门控捷径 | 长路安全但时间可能不够 | UNLOCK | 超时 |
| **delayed_corridor** | 延迟揭示风险 | 走过去才发现有风险 | WARN(提前) | 承诺错误 |
| **distractor_cue** | 误导性线索 | 局部线索指向错误方向 | WARN + 迁移 | 线索误判 |

### fork_trap 示例（最常用的类型之一）

```
Row 0: ████████████████████████       (墙壁)
Row 1: ██ [A1][A2][A3][A4] ████       ← Branch A (可能是陷阱)
Row 2: ██ [F]████████████[M] ██       ← 走廊(Fork → Merge)
Row 3: ██ [B1][B2]██████ [M] ██       ← Branch B (可能安全但绕路)
Row 4: ██████████ [D1][D2] ████       ← 绕路区域
Row 5: ██████████ [D3][D4] ████       ← 绕路区域
Row 6: ████████████████████████       (墙壁)

F = Fork point (Agent 在这里做选择)
M = Merge point (两条路汇合)
A1~A4 = Branch A 的 cells
B1~B2 + D1~D4 = Branch B 的 cells (绕路)
```

**关键设计**：
- trap_depth 决定陷阱在第几个 cell 才出现（之前的 cell 看起来无害）
- cue_ambiguity 决定两条分支的纹理特征有多相似（高模糊 = 更难区分）
- 每次生成随机决定 Row 1 还是 Row 3 是危险的（mirror）

### 这些家族当前的使用状态

| 家族 | Canonical 使用 | 说明 |
|------|:------------:|------|
| baseline_v2 | ❌ | 作为回归基线，不在 lesson catalog 中 |
| fork_trap | ❌ | 早期设计，功能被 TIC 覆盖 |
| hazard_belt | ❌ | 需要 ITEM_DROP，当前未接入 |
| deadline_gate | ❌ | 需要 UNLOCK，当前未接入 |
| delayed_corridor | ❌ | 早期设计 |
| distractor_cue | ❌ | 早期设计 |

> **当前 canonical 实验使用的是下面介绍的 TIC / TIC-v4 走廊**——这些场景家族是更丰富的地图变体，代码已实现但未在主路径中使用。

---

## 4. 教学走廊（TIC / TIC-v4）——Canonical 使用

### 4.1 基本结构

TIC（Teaching Internalization Corridor）是当前实验的核心场景生成器。它生成一个**7 行 × W 列**的简化二叉分支走廊：

```
W = 1 + 1 + 1 + branch_len + 1 + 1 + 1 = branch_len + 6

Row 0: ████████████████████████████████████
Row 1: ████ [A1] [A2] [A3] ... [An] ██████    ← Branch A
Row 2: ██ [S] [F] ████████████ [M] [G] ██     ← 走廊
Row 3: ████ [B1] [B2] [B3] ... [Bn] ██████    ← Branch B
Row 4: ████████████████████████████████████
Row 5: ████████████████████████████████████
Row 6: ████████████████████████████████████

S = Start (Agent 起始位置)
F = Fork point (分支点)
M = Merge point (汇合点)
G = Goal (目标)
A1~An = Branch A 的 cells (branch_len 个)
B1~Bn = Branch B 的 cells (branch_len 个)
```

**默认 `branch_len = 10`**，所以整个地图是 **7 × 16**。

### 4.2 Agent 的行动路线

```
Step 1: Agent 从 S(2,1) 出发 → 走到 F(2,2)
Step 2: 在 Fork point，Agent 选择上(Row 1)或下(Row 3)
Step 3: 沿选择的分支走 10 步
Step 4: 到达 Merge point M(2,13) → 走到 Goal G(2,14)
```

Agent 只在 **Step 2 做一次关键决策**——选 A 还是 B。之后就是走完分支到终点。

### 4.3 两条分支的特征

每次生成场景时，`mirror` 参数随机决定哪条分支是安全的：

```
mirror = 0: Branch A (Row 1) = 安全, Branch B (Row 3) = 危险
mirror = 1: Branch A (Row 1) = 危险, Branch B (Row 3) = 安全
```

**安全分支**的每个 cell：
```python
features[r, c] = [safety≈0.6, tempt≈0.15, texture_1∈[0.3,0.5], 0.0]
risk[r, c] ≈ 0.05  # 低风险
```

**危险分支**的每个 cell，按深度分为两段：

```
前 d_reveal 个 cell（浅处）:
  features = [0.5±0.03] * 4    ← 看起来很"中性"，无法区分
  risk = risk_level ± 0.05     ← 实际有风险，但特征不明显

第 d_reveal 个 cell 之后（深处）：
  features = [safety=低, tempt=高, texture_1∈[0.4,0.6], texture_2∈[0,0.3]]
  risk = risk_level ± 0.05     ← 特征变明显，但可能已经走太深了
```

---

## 5. 信息时序参数——场景的真正"灵魂"

每个 episode 的关键不是地图形状（形状都一样），而是以下参数：

### 5.1 `d_commit` vs `d_reveal`（承诺深度 vs 揭示深度）

$$d_{\text{commit}}: \text{走了几步后"回不了头"}$$
$$d_{\text{reveal}}: \text{走了几步后"才看到真实线索"}$$

```
Branch B (risky):
  [B1] [B2] [B3] [B4] [B5] [B6] [B7] [B8] [B9] [B10]
   ↑         ↑              ↑
  d_reveal=1 d_commit=3     d_reveal 之后特征才变明显

  如果 d_reveal < d_commit:
    → Agent 在被迫 commit 之前就能看到线索 → 有自己的证据 → 可以 self-discovery
    
  如果 d_reveal > d_commit:
    → Agent 在看到线索之前就已经 commit 了 → 必须靠 Tutor 的 WARN 才能避险
```

**这是场景设计中最核心的参数**——它决定了"Agent 能否自己发现危险"：

| 关系 | 含义 | Tutor 该怎么做 |
|------|------|---------------|
| $d_{\text{reveal}} \ll d_{\text{commit}}$ | Agent 有充分的自我证据 | **WAIT 更好**（让 Agent self-discover） |
| $d_{\text{reveal}} \approx d_{\text{commit}}$ | 边界情况 | 需要精确计算 |
| $d_{\text{reveal}} \gg d_{\text{commit}}$ | Agent 看不到就必须选了 | **WARN 更好**（Agent 需要帮助） |

### 5.2 `lure_strength`（诱惑强度）

$$\text{lure} \in [0, 1]: \text{risky 分支的吸引力}$$

- lure = 0.1 → risky 分支没什么吸引力 → safe agent 不太可能被骗
- lure = 0.9 → risky 分支极度诱人 → 连 safe agent 都可能被吸引

### 5.3 `risk_level`（风险等级）

$$\text{risk} \in [0, 1]: \text{risky 分支的实际危险程度}$$

- risk = 0.15 → 低风险（15% 概率出事）
- risk = 0.6 → 高风险（60% 概率出事）

### 5.4 各 Subtype 的参数范围

| Subtype | d_commit | d_reveal | lure | risk | 核心测什么 |
|---------|:--------:|:--------:|:----:|:----:|----------|
| **temptation_repeat** | [3, 5] | [1, 3] | [0.7, 1.0] | [0.3, 0.5] | 高诱惑下能否抵抗 |
| **self_discovery_teach** | [5, 7] | [1, 2] | [0.4, 0.7] | [0.2, 0.4] | d_reveal ≪ d_commit → 应该自己发现 |
| **warn_rescue** | [2, 3] | [3, 5] | [0.6, 0.9] | [0.4, 0.6] | d_reveal > d_commit → 必须 WARN |
| **boundary_obs** | [3, 5] | [3, 5] | [0.3, 0.5] | [0.15, 0.3] | 模糊边界 |

```
temptation_repeat:  "高诱惑 + 有线索" → 训练抵抗力
self_discovery:     "很早就能看到线索" → 鼓励自主发现
warn_rescue:        "看不到就得选了" → Tutor 必须出手
boundary_obs:       "差不多的两条路" → 锻炼精确判断
```

### 5.5 TIC-v4 扩展 Subtype

在 TIC 的基础上，TIC-v4 增加了更丰富的子类型：

| Subtype | 测什么 |
|---------|--------|
| **sparse_valid_advice** | Tutor 给了**正确**但稀疏的建议 → Agent 能否采纳 |
| **sparse_invalid_advice** | Tutor 给了**错误**的建议 → Agent 能否拒绝 |
| **beneficial_novelty** | 新路径其实更好 → Agent 敢不敢探索 |
| **verified_warn** | 可验证的警告 → 建立信任 |
| **false_suppression_cost** | 过度抑制的代价 → 检测 γ_gen 过高的后果 |

---

## 6. Episode 生成流程

当 CurriculumController 选择了一个 lesson（比如 `sparse_invalid_advice`）后，episode 生成的完整流程是：

```
Step 1: Controller 选课 → lesson = "sparse_invalid_advice"
  │
  ▼
Step 2: adaptive_episode_generator_v2 接收
  │  输入: lesson + episode_idx + theta_true + mastery
  │
  ▼
Step 3: 查 subtype 映射 → subtype = "sparse_invalid_advice"
  │  查 SUBTYPE_V4_PARAMS → 得到 d_c, d_r, lure, risk 范围
  │
  ▼
Step 4: Mastery-conditioned 调整
  │  如果 EP mastery < 0.4 → 降低 severity
  │  如果 IA mastery > 0.6 → 增加 severity
  │  如果 IA mastery < 0.4 → 限制 dose_budget
  │
  ▼
Step 5: 随机采样具体参数
  │  d_commit = randint(d_c range)
  │  d_reveal = randint(d_r range)
  │  lure = uniform(lure range)
  │  risk = uniform(risk range)
  │  mirror = randint(0, 1)  → 随机决定哪边安全
  │
  ▼
Step 6: generate_tic_scenario(spec)
  │  → 生成 7×16 的走廊网格
  │  → 填充 cell 特征向量
  │  → 标注 safe/risky cells
  │  → 设置 oracle safe branch
  │
  ▼
输出: (ep_params, spec, gridmap, cfg, meta, scenario_config)
```

### Mastery-conditioned 的含义

同一个 lesson 在不同 mastery 下会生成**不同难度**的 episode：

```
Agent mastery 低（EP=0.3）:
  → severity 降低 0.15
  → d_reveal 减少 1（更早给线索）
  → "对新手友好版"

Agent mastery 高（IA=0.7）:
  → severity 增加 0.1
  → "对老手的挑战版"
```

---

## 7. 观测约束：Agent 能看到什么

### 7.1 观测半径

Agent 的观测半径（Manhattan distance）= **2**：

```
Branch 示例（branch_len=10）:

Fork →  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] → Merge

已观测特征:   ✅  ✅  ✅   ❌  ❌  ❌  ❌  ❌  ❌   ❌
              ↑ dist≤2 ↑
              
未观测 cell: 用中性先验 [0.5, 0.5, 0.25, 0.25] 填充
```

### 7.2 这意味着什么

在一条 10-cell 的分支上：
- Agent 在 fork point 大约只能看到**前 2~3 个 cell**
- 对剩下 7~8 个 cell 完全不了解
- 如果 `d_reveal = 1`（第 1 个 cell 就有线索），Agent 能看到
- 如果 `d_reveal = 5`（第 5 个 cell 才有线索），Agent **看不到**

### 7.3 与 Tutor 的关系

| | Agent | Tutor |
|---|:---:|:---:|
| 前 2-3 个 cell | ✅ 能看到特征 | ✅ 知道 |
| 深处 7-8 个 cell | ❌ 看不到 | ✅ 知道 |
| 哪条是 oracle safe | ❌ 不知道 | ✅ 知道 |

> **Tutor 的 WARN 本质上 = "我替你看了深处的 cell，告诉你别走那边。"**

---

## 8. 特征向量设计

### 8.1 4D 特征的语义分区

每个 cell 的特征 $z = [z_0, z_1, z_2, z_3]$：

| 维度 | 名字 | 在走廊中的使用 | 对 risk 的决定作用 |
|:---:|------|-------------|:-:|
| $z_0$ | identity | safe cell: ~0.6, risky cell: ~0.2-0.5 | 不应该有 |
| $z_1$ | gate_flag | 走廊连接点=1.0，分支内=tempt_strength | 弱 |
| $z_2$ | texture_1 | safe: [0.3,0.5], risky 浅处: ~0.5, risky 深处: [0.4,0.6] | **强** |
| $z_3$ | texture_2 | safe: 0.0, risky 深处: [0,0.3] | **强** |

### 8.2 特征如何暗示安全性

```
安全分支 cell:
  z = [0.6, 0.15, 0.4, 0.0]
  → 高 safety, 低 temptation, 中 texture, 无 texture_2

危险分支 cell（浅处, i < d_reveal）:
  z = [0.5, 0.5, 0.5, 0.5]
  → 看起来"中性"，无法区分 ← 这就是为什么 Agent 可能被骗

危险分支 cell（深处, i ≥ d_reveal）:
  z = [0.2, 0.8, 0.5, 0.2]
  → 低 safety, 高 temptation, 纹理开始不同 ← 线索出现了
```

---

## 9. Transfer 测试场景

教学结束后，系统会在 **4 个 phase 的 transfer 场景** 下测试 Agent：

| Phase | 含义 | 有无 Tutor |
|:-----:|------|:--------:|
| **B** | 同结构、同参数范围 | ❌ 无 |
| **C** | 同结构、更高诱惑 (+0.1~0.15) | ❌ 无 |
| **D** | 更高风险 (+0.05~0.1) | ❌ 无 |
| **E** | 组合：高诱惑 + 高风险 | ❌ 无 |

Transfer 测试的结果产生两个核心指标：
- **C (Competence)**：Agent 在无 Tutor 时选择正确分支的比例
- **E (Exploration)**：Agent 是否仍然愿意尝试 novel 选项

---

## 10. 总结：地图在这个系统中的角色

> **地图不是挑战的来源——信息时序才是。**

```
固定的部分:              变化的部分:
─────────               ──────────
7×16 网格布局             d_commit (何时必须选)
二叉分支结构              d_reveal (何时看到线索)
10 个 cell 的分支长度      lure_strength (诱惑有多强)
obs_radius = 2           risk_level (风险有多高)
4D 语义特征框架            mirror (哪边安全)
                         subtype (测什么能力)
                         mastery-conditioned 调整
```

所有 10 个 lesson 共享同样的空间结构，但通过不同的信息时序参数，产生了从"warn_rescue（必须出手救）"到"self_discovery（应该让学生自己发现）"的完整教学情境光谱。
