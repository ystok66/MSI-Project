# 场景设计审计报告 — Bug / 缺陷 / 不足 / 过于简单

> 审计范围：`src/envs/` 所有场景生成器（14 个注册场景家族）  
> 审计日期：2026-04-07

---

## 目录

| 分类 | 数量 |
|------|------|
| 🔴 Bug / 逻辑错误 | 6 |
| 🟠 设计缺陷 / 不一致 | 7 |
| 🟡 过于简单 / 深度不足 | 6 |
| ⚪ 冗余 / 维护负担 | 3 |

---

## 1. 🔴 Bug / 逻辑错误

### BUG-1: `scenario_families.py` 存在重复函数定义 — 后者覆盖前者

| 字段 | 内容 |
|------|------|
| 位置 | `scenario_families.py` |
| 问题 | 文件中两个函数被定义了两次: |

| 函数名 | 第一次定义 | 第二次定义 | 注册表引用 |
|--------|-----------|-----------|-----------|
| `generate_delayed_corridor` | **L911** (7-row, simple parameters) | **L2198** (commit/reveal parameterized) | L3042 → **第二次** |
| `generate_distractor_cue` | **L1117** (cue_mode=weak/misleading) | **L2406** (重写版) | L3043 → **第二次** |

**影响**：Python 的函数定义是 sequential scope — 第二次 `def` 会覆盖第一次。注册表在文件末尾构建 `SCENARIO_REGISTRY`（L3037），此时引用的是第二次定义。**第一次定义（L911, L1117）是 ~400 行死代码**，永远不会被调用。

> [!CAUTION]
> 第一次和第二次定义的参数结构不同。例如 `generate_delayed_corridor` 第一版使用 `DELAYED_CORRIDOR_PARAMS`(L901)，第二版使用 `DELAYED_COMMITMENT_PARAMS`(L2186)。如果有外部代码直接 `from scenario_families import generate_delayed_corridor` 并传第一版的参数，会得到第二版的行为。

---

### BUG-2: `fork_trap` 的 safe lane detour 在 `safe_row==1` 时物理上不连通

| 字段 | 内容 |
|------|------|
| 位置 | [scenario_families.py:219-225](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/scenario_families.py#L219-L225) |
| 问题 | 当 `risky_row = 3, safe_row = 1` 时（50% 概率随机化），safe lane 在 row 1。detour 逻辑在 row 1 的某列放 WALL，然后开放 **rows 4,5** 作为绕行。但 row 1 和 row 4 之间隔了 rows 2(corridor/wall) 和 row 3(risky)。 |
| 影响 | **Agent 无法从 row 1 到达 row 4**。detour 路径物理上断裂。Safe lane 在这种情况下不可通行。 |
| 代码指示 | L222-225: `# safe_row == 1, detour not needed in current architecture` `# but we still use rows 4, 5 for consistency` — 说明开发者已经意识到了问题，但"for consistency"直接复用了 `safe_row==3` 的逻辑。 |
| 缓解 | fork_trap 是 7 行 grid（H=7），row 0 和 row 6 是墙。safe_row=1 时，detour 应该通过 row 0 以上（不存在）或者不做 detour。目前 **50% 的种子会产生不可通行的 safe lane**。 |

```
H=7 grid 当 safe_row==1:
  Row 0: wall
  Row 1: safe lane (with gap at detour)          ← gap 封死, 无法绕行
  Row 2: corridor (wall in segment)
  Row 3: risky lane
  Row 4: detour cells opened                     ← 无法从 row 1 到达!
  Row 5: detour cells opened                     ← 无法从 row 1 到达!
  Row 6: wall
```

> [!WARNING]
> 这是一个 **静默失败** — BFS 会返回 `shortest_safe = 999`，然后 `base = 20`(fallback)，t_max 按 20 计算。Agent 只能走 risky lane，场景退化成无 safe 选项。

---

### BUG-3: baseline_v2 `latent_mode` 在 BFS 之后才覆盖 risk — 路径长度计算用的是原始 risk

| 字段 | 内容 |
|------|------|
| 位置 | [lattice_v2.py:328-364](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py#L328-L364) |
| 问题 | L331-333 先用 handcrafted risk 数组（trap risk=0.30-0.50）做 BFS 算 `shortest_any` 和 `shortest_safe`。然后 L352-364 在 `latent_mode=True` 时用 `WorldWeights` 覆盖 cost/risk，但 **不重新计算** `shortest_any/shortest_safe`。 |
| 影响 | meta 中的 `shortest_any` 和 `shortest_safe` 是基于 handcrafted risk 的 BFS 结果。但 `t_max` 也是基于这个 BFS 结果计算的（L335-342）。当 `WorldWeights` 碰巧让 trap cell 的 risk 变低或变高时，`t_max` 与真实场景 difficulty 不匹配。 |
| 对比 | `harder_baseline.py` 和 `dtmb_lattice.py` **同样有这个问题**。`hazard_belt` L566-587 做了 post-hoc risk override 但也没重新计算 BFS。 |

> [!NOTE]
> 这不是致命 bug（BFS 不看 risk 值，只看 passable），但 `shortest_safe` 通过 avoid risky gates 来计算，而 latent_mode 可能会改变哪些 cell "真正"危险。

---

### BUG-4: `harder_baseline` 当 `seg_width=3` 时 detour 位置可能越界

| 字段 | 内容 |
|------|------|
| 位置 | [harder_baseline.py:91, 166-167](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/harder_baseline.py#L91) |
| 问题 | `seg_widths = rng.integers(3, 5, ...)` → 可能得到 `sw=3`（只有 3 列宽）。则 `seg_start=col_cursor, seg_end=col_cursor+2`。仅有 `range(seg_start+1, seg_end)` = 1 个内部列。detour 计算 `detour_start = seg_start + (3//2) - 1 = seg_start + 0 = seg_start`。 |
| 影响 | `detour_start == seg_start`，即 detour 从 segment 入口列开始。这会把 safe lane 的入口 cell 也封为 WALL，导致 safe lane 可能入口闭合。后面的 `if dc_down - 1 >= seg_start` 检查通过（因为 dc_down-1 = seg_start-1，不满足），不会开额外通道。 |
| 频率 | `rng.integers(3, 5)` 有 50% 概率得到 3。每 3 segment，67% 概率至少有一个 sw=3。 |

---

### BUG-5: DTMB Stage 2 detour 可能覆盖其他 branch 行

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_lattice.py:712](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_lattice.py#L712) |
| 问题 | detour 行计算 `detour_row = s_row + (1 if s_row < H-3 else -1)`。当两个 sub-branch 的 row 相差正好 1 时（`min_gap=2` 但夹紧后可能只差 1-2），一个 branch 的 detour_row 可能 **与另一个 branch 的主行重叠**。 |
| 影响 | 第一个 branch 的 detour 在 `detour_row` 开 NORMAL cell，然后第二个 branch 在同一 `detour_row` 写 WALL（L718），**互相覆盖**。最终拓扑取决于处理顺序。 |
| 条件 | 当 `H=13` 且 S1 有 3 个 branch（min_gap=3），S2 每个 branch 再分 2 个 sub（min_gap=2），总行数需求 > H-2 时会触发。 |

---

### BUG-6: DTMB Stage 3 `entry_rows.index(e_row)` 可能引用错误的 parent_exit

| 字段 | 内容 |
|------|------|
| 位置 | [dtmb_lattice.py:777](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_lattice.py#L777) |
| 代码 | `e_col = parent_exits[entry_rows.index(e_row)][1] if entry_rows.index(e_row) < len(parent_exits) else c_start` |
| 问题 | `entry_rows` 是 `sorted(set(...))`，去重且排序后索引与 `parent_exits` 的原始顺序不对应。如果两个 parent_exit 在同一 row（去重后变成 1 个 entry_row），index 会取第一个，忽略第二个的列信息。 |
| 影响 | 某些 S2 exit 的 col 信息丢失，Stage 3 可能在错误的列开始连接。 |

---

## 2. 🟠 设计缺陷 / 不一致

### DESIGN-1: Latent mode 与 handcrafted risk 的语义冲突

| 类别 | 行为 |
|------|------|
| `baseline_v2` (latent_mode=True) | `WorldWeights` 完全覆盖 risk — handcrafted trap risk 被丢弃 |
| `hazard_belt` (latent_mode=True) | `WorldWeights` 覆盖后，**post-hoc 强制** belt risk = cfg value (L576-584) |
| `DTMB` (latent_mode=True) | `WorldWeights` 覆盖后，**post-hoc 强制** belt risk ≥ 0.8×cfg (L454-455) |
| `GTET` | 同 DTMB 模式 |
| `fork_trap` (latent_mode=True) | `WorldWeights` 完全覆盖 — **handcrafted trap_risk 被丢弃** |

**问题**：
1. `baseline_v2` 和 `fork_trap` 在 latent_mode 下，feature 设计 **应该** 保证 trap cell 有高 texture → 高 risk。但 `WorldWeights` 是随机生成的（`w_risk[2] ∈ [2,4], w_risk[3] ∈ [1.5,3.5], b_risk ∈ [-3,-1.5]`）。某些种子下，trap feature `[0, 0, 0.85, 0.80]` 经过 WorldWeights 映射后可能只得到 risk=0.15 — 场景的 trap 完全失效。
2. `hazard_belt` / `DTMB` 用 post-hoc override 修补了这个问题，但这意味着 **agent 从 feature 无法学到真实 risk**（feature → WorldWeights 映射的 risk 和实际 risk 不一致）。这违反了 "agent 通过学习 feature→cost/risk 映射来理解环境" 的核心假设。

> [!IMPORTANT]
> **根本矛盾**：Latent mode 假设 risk = f(z, WorldWeights)，但场景设计需要 特定 cell 有特定 risk。当 WorldWeights 不配合时，要么 trap 失效（baseline_v2），要么 risk 被 override（DTMB），让 agent 学到的映射与真实不一致。

---

### DESIGN-2: 所有 7-row 场景（baseline_v2, fork_trap, hazard_belt, deadline_gate）共享完全相同的拓扑框架

所有 7-row 场景的拓扑约束：
- Row 0, 6: 始终是墙
- Row 1: 始终是 risky lane
- Row 2: corridor（segment 内墙，segment 间通道）
- Row 3: 始终是 safe lane main
- Row 4, 5: 始终是 detour space

这意味着：
1. **Agent 实际上只需学会一个启发式**：row 1 = 危险，row 3 = 安全。4D feature 在这种简单拓扑下不是 task-critical 的。
2. **Lane ID (`F_LANE_ID`) 直接泄露了安全信息**：row 1 的 feature[0]=0.0, row 3 的 feature[0]=1.0。Agent 不需要学习 texture 就能区分 risky/safe。
3. **空间复杂度极低**：7 行 × 15-25 列 ≈ 100-175 cells，其中约 50% 是墙。有效决策空间只有 ~50-90 cells。

---

### DESIGN-3: feature 维度 2,3（texture_1, texture_2）之间缺乏结构解耦

WorldWeights 的 risk 权重：`w_risk[2] ∈ [2,4], w_risk[3] ∈ [1.5,3.5]`

所有场景的 feature 生成中，texture_1 和 texture_2 **总是高度正相关**：
- `_safe_feature`: `[0.0-0.1, 0.0-0.1]`
- `_trap_feature`: `[0.80-0.95, 0.70-0.90]`
- `_weak_cue_feature`: `[0.30-0.50, 0.20-0.40]`

**问题**：texture_1 和 texture_2 永远同向变动。这意味着：
1. 4D feature 实际上只有 ~2 个信息维度（lane_id + texture_magnitude）
2. Basis expansion 中的交叉项 `z₂z₃` 和差异项 `|z₂−z₃|` 几乎不携带额外信息
3. Agent 学习效率被高估（只需要学 1 个方向就能掌握 risk 映射）

---

### DESIGN-4: `WorldWeights` 每 episode 重新随机 — 跨 episode transfer 测试不公平

| 字段 | 内容 |
|------|------|
| 位置 | 所有 `latent_mode=True` 场景 |
| 问题 | `generate_world_weights(rng, d=4)` 使用每个 episode 的 seed 生成。如果实验使用不同 seed 跑多 episode，每 episode 得到 **不同的 WorldWeights**。 |
| 影响 | SlowFast transfer 机制（`GenericSlowFastPredictor`）的 slow head 从上一个 episode 积累的 weight→risk 映射，在下一个 episode 完全不适用。Transfer **被定义为不可能的**，除非 WorldWeights 跨 episode 保持一致。 |
| 修复 | 实验脚本中用固定的 `world_weights_seed` 让所有 episode 共享同一个 WorldWeights。但 `generate_scenario()` API 没有这个参数。 |

---

### DESIGN-5: `ScenarioConfig.commitment_cells` 只在 delayed_corridor 和 DTMB 中被填充

| 字段 | 内容 |
|------|------|
| 位置 | `ScenarioConfig` dataclass |
| 问题 | `commitment_cells` 在 `fork_trap` 和 `baseline_v2` 中为空 `[]`，但这些场景也有 commitment（进入 risky lane 后回头代价高）。 |
| 影响 | 依赖 `commitment_cells` 的分析（如 `p_self` 估计）在 fork_trap/baseline_v2 中无法工作。 |

---

### DESIGN-6: 没有 reachability validation — 场景可能不可达

所有生成器都假设拓扑正确，但没有 post-generation validation：
- 没有检查 `agent_start → target_pos` 是否可达（BFS=999 时只会 fallback `t_max`）
- 没有检查 safe lane 是否可通行
- 没有检查 locked doors 是否有可达的替代路径

只在 `_bfs_len` 返回 999 时做 fallback（`base = shortest_safe if shortest_safe < 999 else 40`），但不抛出警告。

---

### DESIGN-7: `hazard_belt` 非 belt segment 的 risky lane feature 使用 `_safe_feature` 而非 `_lane_feature`

| 字段 | 内容 |
|------|------|
| 位置 | [scenario_families.py:491](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/scenario_families.py#L491) |
| 代码 | `features[1, c] = _safe_feature(rng, 0.0)` |
| 问题 | 非 belt segment 的 risky lane（row 1）cell 类型标记为 `CellType.RISKY` (L486)，但 feature 用的是 `_safe_feature`（low texture）。 |
| 影响 | 在 latent_mode 下，`WorldWeights` 根据 safe feature 算出 low risk。然后 L579-581 又 cap 到 `min(risk, 0.05)`。所以 cell 标记为 RISKY 但 risk ≤ 0.05 — 名不副实。Agent 的 risk 学习信号混乱：RISKY 类型 cell 的 risk 只有 0.02-0.05，和 NORMAL cell 一样。 |

---

## 3. 🟡 过于简单 / 深度不足

### SIMPLE-1: 7-row 场景的决策空间过于简单 — 有效二元选择

所有 7-row 场景（baseline_v2, harder_baseline, fork_trap, hazard_belt, deadline_gate, delayed_corridor, distractor_cue）本质上都是 **序列化的二元选择**：

```
每个 segment 进入时: 选 row 1 (risky) 还是 row 3 (safe)?
```

这不是 POMDP 级别的规划问题。Agent 只需要在 ~3 个 decision point 上做 binary choice。真正的 planning（A* 搜索、path evaluation）在大多数情况下是过度设计 — 简单的 "if risk_high then go_safe" 就够了。

**与 Proposal 差距**：Proposal 描述的是 "grid-world where Agent is tasked with fetching objects and delivering them to specified target locations"，暗示更复杂的寻路问题，而非线性走廊 + binary fork。

---

### SIMPLE-2: 场景缺乏真正的部分可观测性

Proposal 强调 POMDP："The Agent receives noisy perceptual cues... The true cost and risk are only revealed upon physical visitation."

当前实现：
- Agent 在 `patch_radius=2` 时，每步能看到 Manhattan 距离 ≤ 2 的 ~12 个 cell
- 7-row 场景宽度 15-25，agent 从 col 1 走到 col W-2 时，大部分 cell 都已被观察过
- 更关键的是：**risky cell 的位置（row 1）本身就是确定的**，只有 risk 程度需要学习

真正的 POMDP 应该有：
- 隐藏的 trap 位置（不仅仅是 trap 程度不确定，而是 **在哪里** 不确定）
- Agent 无法预见的拓扑变化
- 多步决策之间的信息依赖

---

### SIMPLE-3: GTET 和 DTMB 更复杂但未形成标准实验矩阵

| 场景家族 | 独立性 | 实验覆盖度 |
|---------|--------|-----------|
| baseline_v2 | ✅ canonical | ✅ 广泛 |
| harder_baseline_v2 | ✅ canonical | ✅ Phase 2B |
| **fork_trap** | 🟡 和 baseline 高度相似 | 🔲 无独立实验 |
| **hazard_belt** | 🟡 和 baseline 结构相同 | 🔲 无独立实验 |
| **deadline_gate** | 🟡 和 baseline 结构相同 | 🔲 无独立实验 |
| **delayed_corridor** | 🟡 有两个重复定义! | 🔲 无独立实验 |
| **distractor_cue** | 🟡 有两个重复定义! | 🔲 无独立实验 |
| **funnel_trap** | 🔲 | 🔲 无独立实验 |
| **elcb / elcb_po** | 🔲 | 🔲 无独立实验 |
| **temptation_corridor / joint_conflict** | 🔲 | 🔲 无独立实验 |
| GTET | ✅ 独立架构 | ✅ Phase 8+ |
| DTMB | ✅ 独立架构 | ✅ Phase 8+ |
| DTMB hard | ✅ calibrated | ✅ 实验 |

14 个注册场景中，只有 **4 个** 被用于正式实验。其余 10 个场景的设计意图和质量未经实验验证。

---

### SIMPLE-4: 没有跨 episode 的场景进化机制

所有场景都是 **单次生成、单次使用**。没有：
- 课程学习（从 easy → medium → hard 的自动切换）
- 场景记忆（基于 agent 上次表现选择下次场景）
- 渐进难度（同一场景每 episode 增加一点难度）

`profile_state.py` 和 `profile_manager.py` 存在但都是 shadow mode，未接入。

---

### SIMPLE-5: Feature 空间仅 4D — 对于 Basis 和 Transfer 的压力测试不足

4D feature：`[lane_id, gate_flag, texture_1, texture_2]`

其中：
- `lane_id`：0 或 1（二值）
- `gate_flag`：0 或 1（二值）
- `texture_1, texture_2`：连续 [0,1]

有效连续维度只有 2。这对于：
- Basis expansion（6D/7D）来说太简单 — 交叉项几乎不携带信息
- Transfer learning 来说太简单 — 2 个连续维度的映射用几个样本就能学会
- Uncertainty-driven planning 来说太简单 — posterior variance 收敛太快

---

### SIMPLE-6: 不同 scenario family 之间 feature 设计差异不足

| 场景 | safe feature z₂,z₃ | trap feature z₂,z₃ | Δ |
|------|---------------------|---------------------|---|
| baseline_v2 | [0.0-0.1, 0.0-0.1] | [0.80-0.95, 0.70-0.90] | ~0.75 |
| harder_baseline | [0.25-0.50, 0.20-0.45] | [0.45-0.65, 0.40-0.60] | ~0.15 |
| fork_trap | [0.0-0.20, 0.0-0.15] | [0.50-0.80, 0.50-0.70] | ~0.50 |
| GTET goal_cue | [0.25-0.65, 0.20-0.55] | N/A | — |
| DTMB belt | [0.47-1.0, 0.40-0.85] | N/A | — |

`baseline_v2` 的 gap 有 ~0.75 — 太容易区分。`harder_baseline_v2` 缩小到 ~0.15 — 但只是 **参数调整**，不是结构性难度提升。没有场景使用 **反直觉** 的 feature mapping（如 texture_high = safe）。

---

## 4. ⚪ 冗余 / 维护负担

### REDUNDANT-1: `scenario_families.py` 有 3057 行 — 应该拆分

14 个场景生成器全部在同一文件中，总计 3057 行。加上参数字典和 registry，非常难以维护。

建议拆分方案：

| 文件 | 场景 |
|------|------|
| `scenario_families.py` | 只保留 registry + `generate_scenario()` + `ScenarioConfig` |
| `families_7row.py` | baseline_v2, fork_trap, hazard_belt, deadline_gate |
| `families_commitment.py` | delayed_corridor, distractor_cue |
| `families_advanced.py` | funnel_trap, elcb, elcb_po, temptation_corridor, joint_conflict |
| `gtet_lattice.py` | GTET（已拆分 ✅） |
| `dtmb_lattice.py` | DTMB（已拆分 ✅） |
| `harder_baseline.py` | harder_baseline_v2（已拆分 ✅） |

---

### REDUNDANT-2: 7-row 场景之间大量代码复制

每个 7-row 场景都重复了 ~80 行拓扑构建代码：
- corridor 开通（row 2）
- segment 内 wall row 2
- entry/exit gate 开通
- detour 构建（rows 4-5）

这些可以抽取为共享函数 `_build_7row_segment(ct, cost, features, rng, seg_start, seg_end, ...)`.

---

### REDUNDANT-3: 三套独立的 BFS 实现

| 函数 | 文件 | 差异 |
|------|------|------|
| `_bfs_len(gm, start, goal, avoid)` | `lattice_v2.py:391` | 接受 GridMap |
| `_bfs_shortest(ct, start, goal, avoid)` | `dtmb_lattice.py:187` | 接受 cell_types array |
| `_bfs_gtet(ct, start, goal, avoid)` | `gtet_lattice.py:105` | 接受 cell_types array |

三个函数功能完全相同，只是输入格式略有不同。应该统一。

---

## 5. 修复优先级总结

### P0 — 影响结果正确性

| ID | 修复 | 难度 |
|----|------|------|
| BUG-1 | 删除 `scenario_families.py` 中第一版 `generate_delayed_corridor`(L911) 和 `generate_distractor_cue`(L1117) | 🟢 |
| BUG-2 | `fork_trap` 当 `safe_row==1` 时不做 detour 或使用 row 0 空间 | 🟡 |
| DESIGN-1 | 统一 latent_mode 语义 — 要么全场景 post-hoc override，要么设计 feature 保证映射方向一致 | 🟠 设计决策 |

### P1 — 影响实验可信度

| ID | 修复 | 难度 |
|----|------|------|
| BUG-4 | `harder_baseline` 限制 `seg_widths ≥ 4` | 🟢 |
| BUG-5 | DTMB Stage 2 添加 row overlap 检查 | 🟡 |
| DESIGN-4 | 添加 `world_weights_seed` 参数到 `generate_scenario()` API | 🟡 |
| DESIGN-6 | 添加 post-generation reachability assertion | 🟢 |

### P2 — 增强场景深度（Paper quality）

| ID | 改进 | 难度 |
|----|------|------|
| SIMPLE-1 | 设计非线性拓扑场景（grid 而非 corridor） | 🟠 |
| SIMPLE-2 | 增加 trap 位置不确定性（不是固定在 row 1） | 🟡 |
| SIMPLE-5 | 增加 feature 维度至 6D+，让 Basis 有实质差异 | 🟡 |
| DESIGN-3 | 生成 texture_1 和 texture_2 有独立信息（非完全正相关） | 🟡 |
| REDUNDANT-1 | 拆分 scenario_families.py | 🟢 |
