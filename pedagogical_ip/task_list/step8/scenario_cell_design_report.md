# Pedagogical Gridworld: Scenario & Cell Design Report
## Complete Specification of Grid Generation, Cell Types, Features, and World Models

---

## 1. 底层设施：GridMap 与 CellType

### 1.1 CellType 枚举

所有场景共享 7 种 Cell 类型，定义于 [map_generator.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/map_generator.py#L17-L26):

| 值 | 名称 | 语义 | `true_cost` | `true_risk` |
|----|------|------|-------------|-------------|
| 0 | `NORMAL` | 自由通行 | 1.0 | 0.0 |
| 1 | `WALL` | 不可通行 | ∞ | 0.0 |
| 2 | `HIGH_COST` | 可通行但代价高 | 3.0–5.0 | 0.0 |
| 3 | `RISKY` | 可通行，有概率致死 | 1.0 | 0.05–0.50 |
| 4 | `LOCKED_DOOR` | 上锁不可通行，UNLOCK 后变 NORMAL | ∞→1.0 | 0.0 |
| 5 | `TARGET` | 目标格（标记用） | 1.0 | 0.0 |
| 6 | `OBJECT_SPAWN` | 物体生成点 | 1.0 | 0.0 |

### 1.2 GridMap 数据结构

```python
@dataclass
class GridMap:
    height: int                          # 格子行数
    width: int                           # 格子列数
    cell_types: np.ndarray               # (H, W) int — CellType 值
    true_cost: np.ndarray                # (H, W) float — 每格真实通行代价
    true_risk: np.ndarray                # (H, W) float — 每格真实风险概率
    object_spawn: tuple[int, int]        # 物体起始位
    target_pos: tuple[int, int]          # 目标位
    agent_start: tuple[int, int]         # Agent 起始位
    door_positions: list[tuple[int,int]] # LOCKED_DOOR 位置
```

> **Agent 不能直接观测 `true_cost`, `true_risk`。** 它只能通过 `FeatureBeliefMap` 的间接特征观测来推断。

---

## 2. 4D 特征向量系统

### 2.1 特征维度定义

每个非墙壁 cell 都有一个 4D 特征向量 $z \in \mathbb{R}^4$，存储在 `cell_features[r, c, :]`：

| Index | 名称 | 符号 | 语义 | 取值范围 |
|-------|------|------|------|----------|
| 0 | `F_LANE_ID` | $z_0$ | 当前 lane 标识 | 0.0 (risky lane) ~ 1.0 (safe lane) |
| 1 | `F_GATE_FLAG` | $z_1$ | 门/入口标记 | 0.0 (普通) ~ 1.0 (门/入口) |
| 2 | `F_TEXTURE_1` | $z_2$ | 视觉纹理-1 | 0.0 (平滑=安全) ~ 0.95 (粗糙=危险) |
| 3 | `F_TEXTURE_2` | $z_3$ | 视觉纹理-2 | 0.0 (平滑=安全) ~ 0.90 (粗糙=危险) |

**设计意图**：
- **Texture dims ($z_2, z_3$) 是 risk 的主要驱动因子**；lane_id 和 gate_flag 提供结构性线索
- Agent 不能直接读取 $z$，只能通过带噪声的 `FeatureBeliefMap` 间接观测

### 2.2 特征生成函数

定义于 [lattice_v2.py:379–388](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py#L379-L388)：

| 函数 | 用途 | $z_0$ | $z_1$ | $z_2$ | $z_3$ |
|------|------|-------|-------|-------|-------|
| `_safe_feature(rng, lid)` | 安全格 | `lid` | 0.0 | $U(0, 0.1)$ | $U(0, 0.1)$ |
| `_trap_feature(rng, lid)` | 陷阱格 | `lid` | 0.0 | $U(0.80, 0.95)$ | $U(0.70, 0.90)$ |
| `_weak_cue_feature(rng, lid)` | 弱线索 | `lid` | 0.0 | $U(0.30, 0.50)$ | $U(0.20, 0.40)$ |
| `_lane_feature(rng, lid, mild)` | 车道格 | `lid` | 0.0 | $U(0.10, 0.20)$ 或 $U(0, 0.10)$ | $U(0.05, 0.15)$ 或 $U(0, 0.10)$ |

DTMB 扩展函数（[dtmb_lattice.py:155–181](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_lattice.py#L155-L181)）：

| 函数 | 用途 | $z_0$ | $z_1$ | $z_2$ | $z_3$ |
|------|------|-------|-------|-------|-------|
| `_misleading_feature(rng, lid)` | 误导格（看起来安全实则危险） | `lid` | $U(0.3, 0.6)$ | $U(0.05, 0.15)$ | $U(0.05, 0.15)$ |
| `_temptation_feature(rng, lid, s)` | 诱惑格 | `lid` | $U(0.6, 0.9) \times s$ | $U(0.15, 0.35)$ | $U(0.10, 0.25)$ |
| `_belt_feature(rng, lid, risk_level)` | 终端危险带 | `lid` | 0.0 | $\sim 0.7\text{–}0.95 \times \frac{risk}{0.6}$ | $\sim 0.6\text{–}0.85 \times \frac{risk}{0.6}$ |
| `_door_feature(lid)` | 锁门格 | `lid` | 1.0 | 0.0 | 0.0 |

GTET 扩展函数（[gtet_lattice.py:144–179](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/gtet_lattice.py#L144-L179)）：

| 函数 | 用途 | 语义 |
|------|------|------|
| `_goal_cue_feature(rng, lid, reliability)` | 目标线索 | texture 由 reliability 调控，高 reliability → 更可区分 |
| `_tempt_cue_feature(rng, lid, strength)` | 诱惑线索 | 高 gate_flag，中等 texture |
| `_pref_cue_feature(rng, lid, type, strength)` | 偏好线索 | "safe" → 低 texture；"shiny" → 高 texture |

---

## 3. WorldWeights — 从特征到 cost/risk 的映射

### 3.1 定义

在 `latent_mode=True` 时，cost 和 risk **不再直接手动设定**，而是由 `WorldWeights` 从特征向量 $z$ 计算。

定义于 [cost_risk_model.py:116–156](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py#L116-L156)：

```python
@dataclass
class WorldWeights:
    w_cost: np.ndarray    # (4,)
    b_cost: float
    w_risk: np.ndarray    # (4,)
    b_risk: float
```

### 3.2 真值公式

$$c_{\text{true}}(z) = \max(w_c^* \cdot z + b_c^*, \; 0.1)$$

$$\rho_{\text{true}}(z) = \sigma(w_r^* \cdot z + b_r^*)$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 sigmoid 函数。

### 3.3 随机生成规则 (`generate_world_weights`)

| 参数 | 分布 | 语义 |
|------|------|------|
| $w_c^{*[0\text{–}3]}$ | $U(-0.3, 0.3)$ | Cost 权重（各维均匀微弱影响） |
| $b_c^*$ | 1.0（固定） | 基础 cost ≈ 1.0 |
| $w_r^{*[0]}$ (lane_id) | $U(-0.5, 0.5)$ | 轻微影响 risk |
| $w_r^{*[1]}$ (gate_flag) | $U(-0.3, 0.3)$ | 轻微影响 risk |
| **$w_r^{*[2]}$ (texture_1)** | **$U(2.0, 4.0)$** | **强正向驱动 risk** |
| **$w_r^{*[3]}$ (texture_2)** | **$U(1.5, 3.5)$** | **强正向驱动 risk** |
| $b_r^*$ | $U(-3.0, -1.5)$ | 负偏置：大多数 cell 风险低 |

> **核心设计**：texture 维度 ($z_2, z_3$) 是 risk 的强驱动因子。$z_2 = 0.8, z_3 = 0.7$ (trap feature) 配合 $w_r^{*[2]} \approx 3, w_r^{*[3]} \approx 2.5$ 和 $b_r^* \approx -2$，会使 $\rho_{\text{true}} = \sigma(0.8 \times 3 + 0.7 \times 2.5 - 2) \approx \sigma(2.15) \approx 0.90$。

### 3.4 Latent mode 计算流

```
generate_lattice_v2(latent_mode=True):
  1. 先按 CellType 规则布局 cell + 设定默认 features
  2. generate_world_weights(rng) → ww
  3. for every non-wall cell (r, c):
       cost[r,c] = ww.true_cost(features[r,c])
       risk[r,c] = ww.true_risk(features[r,c])
  4. → 真实的 cost/risk 完全由 (features × WorldWeights) 决定
```

在 DTMB 中，belt cells 额外有强制下限：
```python
risk[br, bc] = max(risk[br, bc], cfg["belt_risk"] * 0.8)
```
确保 outcome bottleneck 即使在不利的 WorldWeights 采样下也成立。

---

## 4. Agent 的感知系统

### 4.1 FeatureBeliefMap — 有噪声的特征观测

Agent 不能直接读取 $z$。它维护对每个 cell 的高斯信念 $(μ, σ^2)$：

$$\text{Prior: } \mu_{0} = 0.5, \quad \sigma_0^2 = 0.25 \quad \text{(uninformative)}$$

**观测噪声**（Kalman update）：

| 观测类型 | 观测噪声 $\sigma^2_{\text{obs}}$ | 含义 |
|---------|-----|------|
| 自身所在格 | 0.01 | 近乎精确 |
| 1-hop 邻居 | 0.08 | 模糊但有信息 |

更新公式（per-dimension 独立 Kalman）：

$$K = \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{prior}} + \sigma^2_{\text{obs}}}$$

$$\mu_{\text{post}} = \mu_{\text{prior}} + K \cdot (z_{\text{obs}} - \mu_{\text{prior}})$$

$$\sigma^2_{\text{post}} = \sigma^2_{\text{prior}} \cdot (1 - K)$$

### 4.2 LatentCostRiskHead — 从 belief 到 cost/risk 预测

Agent 用一个 4D 线性模型将观测到的（含噪声）特征映射到 cost 和 risk 预测：

$$\hat{c}(z) = \max(w_c \cdot z + b_c, \; 0.1)$$
$$\hat{\rho}(z) = \sigma(w_r \cdot z + b_r)$$

**学习规则**（Online MAP，SGD on negative log-posterior + L2 prior）：

$$\nabla_w \mathcal{L}_{\text{cost}} = -(c_{\text{true}} - \hat{c}) \cdot z + \frac{w_c}{\sigma^2_{\text{prior}}}$$

$$\nabla_w \mathcal{L}_{\text{risk}} = -(\rho_{\text{true}} - \hat{\rho}) \cdot z + \frac{w_r}{\sigma^2_{\text{prior}}}$$

$$w \leftarrow w - \eta \cdot \text{clip}(\nabla_w, \|\cdot\| \leq 5.0)$$

**默认学习率**：$\eta_c = 0.10$，$\eta_r = 0.30$。

### 4.3 不确定性估计（Laplace 近似）

$$\text{CostVar}(z) \approx z^T \left(\frac{X^TX}{n} + \frac{I}{\sigma^2_{\text{prior}}}\right)^{-1} z$$

$$\text{RiskVar}(z) \approx \hat{\rho}(1-\hat{\rho}) \cdot (1 + z^T H^{-1} z)$$

其中 $H$ 是经验 Hessian 近似。

---

## 5. 场景生成器一览

系统中共有 3 个层级的场景生成器：

### 层级 1：V1 原始家族（10×10 固定结构）

定义于 [map_families.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/map_families.py)：

| 家族 | 格子 | 核心测试点 | 主要 lever |
|------|------|-----------|-----------|
| `semantic_trap` | 10×10 | 误信风险 → 走错路 | WARN |
| `planning_trap` | 10×10 | 有限搜索 → 找不到安全路 | UNLOCK |
| `exploration_useful` | 10×10 | 探索可减少不确定性 | WAIT |
| `mixed` | 10×10 | 三阶段分别需要不同干预 | 混合 |
| `door_lattice_sanity` | 9×17 | BLOCK/UNLOCK 门验证 | UNLOCK |
| `deceptive_fork` | 6×8 | 最小验证场景（两条路 + 陷阱） | WARN+UNLOCK |

### 层级 2：V2 场景家族

定义于 [scenario_families.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/scenario_families.py)：

| 家族 | 格子 | 核心机制 |
|------|------|---------|
| `baseline_v2` | 7×N (3 segment) | 回归锚: risky/safe lane 选择 |
| `fork_trap` | 7×N | 歧义分叉 → WARN |
| `hazard_belt` | 7×N | 不可避免风险带 → ITEM_DROP |
| `deadline_gate` | 7×N | 紧 deadline + 门 → UNLOCK |
| `delayed_corridor` | 7×N | 延迟揭示风险 → WARN |
| `distractor_cue` | 7×N | 误导线索 → WARN + transfer |
| `funnel_trap` | 接近 V2 | 漏斗陷阱 |
| `elcb` / `elcb_po` | 变体 | 扩展走廊变体 |
| `temptation_corridor` | | 诱惑走廊 |
| `joint_conflict_corridor` | | 联合冲突测试 |

### 层级 3：复杂家族（DTMB / GTET）

| 家族 | 格子 | 核心测试点 |
|------|------|-----------|
| DTMB-L | 13–17 × 35–60 | 多阶段分支、混合瓶颈 |
| GTET-L | 13–17 × 40–60 | 目标-偏好-诱惑纠缠 |

---

## 6. V2 基础 Lattice 详细设计

### 6.1 网格结构

定义于 [lattice_v2.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py)：

**固定 7 行**：

```
Row 0: Wall (border)
Row 1: Risky lane — 水平直通路径
Row 2: Corridor / Wall (段间通道 / 段内隔墙)
Row 3: Safe lane — 水平主路 (带 zigzag)
Row 4: Wall (safe lane detour 入口)
Row 5: Safe lane detour — zigzag 绕路层
Row 6: Wall (border)
```

**段间走廊**：Row 2 在段与段之间可通行；在段内部是墙壁（强制选择 lane）。

### 6.2 Segment 设计

每个 segment 宽度 5–7 列（`rng.integers(5, 8)`）。每段包含：

| 元素 | 位置 | 作用 |
|------|------|------|
| Risky entry gate | `(1, seg_start)` | Agent 可从 Row 2 进入 risky lane |
| Safe entry gate | `(3, seg_start)` | Agent 可从 Row 2 进入 safe lane |
| Risky lane cells | `(1, seg_start+1..seg_end-1)` | 直通，短路径 |
| Safe lane + detour | `(3, *) + (5, *)` | 带 zigzag，长路径 |
| Trap cell | Risky lane 中随机位置 | 主要风险源 |

### 6.3 Trap 概率与风险设定

```
difficulty="easy":   P(trap) = 0.50, risk U(0.30, 0.50)
difficulty="medium": P(trap) = 0.70, risk U(0.30, 0.50)
difficulty="hard":   P(trap) = 0.90, risk U(0.30, 0.50)
```

**邻近 trap cell 的格子**（距离 ≤ 1）设为 weak cue：risk $U(0.15, 0.25)$

### 6.4 Safe lane 路径长度差异

Risky lane: $L_r = \text{seg\_width} - 1 + 2$（两个垂直转移）

Safe lane: $L_s = L_r + 2 \times \text{detour\_len} + \text{extra}$（zigzag 增加路径长度）

**这个长度差是 tutor 干预的核心张力**：risky lane 更快但可能致死，safe lane 更安全但时间更紧。

### 6.5 Deadline 设定

$$t_{\max} = \begin{cases} 1.5 \times L_{\text{safe}} & \text{easy} \\ 1.4 \times L_{\text{safe}} & \text{medium} \\ 1.3 \times L_{\text{safe}} & \text{hard} \end{cases}$$

---

## 7. DTMB-L 详细设计

### 7.1 整体结构

定义于 [dtmb_lattice.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/dtmb_lattice.py)：

3 阶段树形格子，列空间切分为约 30%/40%/30%：

```
[Trunk 3col] → [Stage 1: 30%] → [Stage 2: 40%] → [Stage 3: 30%]
```

### 7.2 三个阶段的设计

#### Stage 1: Epistemic root-split (认知歧义)

- 从 trunk 分叉为 3 条 (easy/medium) 或 4 条 (hard) 水平分支
- 每条分支上分配子目标 ($\text{sg}_0, \text{sg}_1, \text{sg}_2$)
- **Feature 策略**:
  - `depth < reveal_depth`：弱线索或误导线索
  - `depth ≥ reveal_depth` 且 is_lure：诱惑特征
  - else：诚实 safe 特征
- **Commitment point**: 过了 `commit_depth` 列后无法回退

| 参数 | Easy | Medium | Hard |
|------|------|--------|------|
| `cue_reliability` | 0.85 | 0.65 | 0.45 |
| `reveal_depth` | 2 | 3 | 4 |
| `commit_depth` | 5 | 4 | 3 |
| `misleading_fraction` | 0.05 | 0.15 | 0.25 |
| `lure_subtree_fraction` | 0.15 | 0.25 | 0.35 |
| `lure_strength` | 0.35 | 0.55 | 0.80 |

#### Stage 2: Structural split (结构瓶颈)

- 每条 S1 分支再分为 2 (easy/medium) 或 2 (hard, 3) 子分支
- 部分子分支含 `LOCKED_DOOR` 快捷路径
- 非门分支增加 detour zigzag（更长）
- **门的概率**: `mid_door_fraction` ∈ {0.15, 0.25, 0.35}
- **门打通的增益**: `door_gain` ∈ {6, 4, 3} 步

#### Stage 3: Terminal outcome (结果瓶颈)

- 所有分支向单一目标 column 收敛
- **Hazard belt**: 宽度 = `belt_fraction × stage_width` 的危险带
  - 覆盖多个 row，近乎不可避免
  - Risk: $U(\text{belt\_risk} \times 0.8, \text{belt\_risk})$

| 参数 | Easy | Medium | Hard |
|------|------|--------|------|
| `belt_fraction` | 0.20 | 0.35 | 0.50 |
| `belt_risk` | 0.30 | 0.45 | 0.40 |

### 7.3 Grid 尺寸与 Deadline

| 难度 | H × W | Deadline ratio |
|------|-------|----------------|
| Easy | 13 × 35 | 1.25 |
| Medium | 15 × 45 | 1.15 |
| Hard | 17 × 60 | 1.16 |

### 7.4 Ground-truth 瓶颈标注

每个阶段标注主导瓶颈类型和推荐 lever：

| Stage | Bottleneck | Lever |
|-------|-----------|-------|
| 1 | Epistemic | WAIT / WARN |
| 2 | Structural | UNLOCK |
| 3 | Outcome | ITEM_DROP |

---

## 8. GTET-L 详细设计

### 8.1 整体结构

定义于 [gtet_lattice.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/gtet_lattice.py)：

3 阶段 + 2 个 merge zone：

```
[Trunk 3col] → [Stage 1] → [Merge 1] → [Stage 2] → [Merge 2] → [Stage 3]
```

### 8.2 三个阶段的设计

#### Stage 1: 弱目标线索分叉

- 3 条分支，每条分配一个子目标 ($\text{sg}_0, \text{sg}_1, \text{sg}_2$)
- **Goal cue zone**（出现在 stage1/3 偏移 + leadlag）：
  - 特征: `_goal_cue_feature(..., reliability)`
  - ambiguity 控制: 有概率给出 wrong subgoal 的标签
- **Temptation cue zone**: 在部分分支出现诱惑特征
- Merge 1: 回到中心行，所有路径汇合

| 参数 | Easy | Medium | Hard |
|------|------|--------|------|
| `goal_cue_reliability` | 0.70 | 0.50 | 0.35 |
| `goal_cue_leadlag` | +2 | 0 | -2 |
| `goal_ambiguity` | 0.35 | 0.55 | 0.75 |
| `lure_strength` | 0.40 | 0.65 | 0.85 |

#### Stage 2: 诱惑分叉

- 2×2 = 4 条子分支 (medium/easy) 或更多
- 奇数分支 = 诱惑路线（高 temptation cue）
- 偶数分支 = 安全路线
- **Preference cue** 出现在 stage2 末段 1/3
  - "safe" 偏好 → 低 texture；"shiny" 偏好 → 高 texture
- **纠缠设计**：诱惑路线同时附带弱 goal cue（agent 不确定是在追 goal 还是被 tempted）

| 参数 | Easy | Medium | Hard |
|------|------|--------|------|
| `preference_cue_strength` | 0.60 | 0.45 | 0.30 |
| `tempt_offset_z` | 0.30 | 0.50 | 0.70 |

#### Stage 3: 终端解析 (belt + 快速通道)

- 两条路径：upper (经过 belt) 和 lower (绕路 + 较弱 belt)
- **Upper belt**: 固定宽度，risk = `belt_risk ± 0.05`
- **Lower belt**: 宽度较窄，risk = `belt_risk × 0.7`（更安全但更长）
- **Fast lane**（仅 medium/hard）：上锁的快速通道，$risk = 0$，需 UNLOCK

| 参数 | Easy | Medium | Hard |
|------|------|--------|------|
| `has_locked_fast_lane` | No | Yes | Yes |
| `belt_width` | 3 | 4 | 5 |
| `belt_risk` | 0.25 | 0.35 | 0.45 |

### 8.3 Merge zone 设计

Merge zone 宽 3 列，所有分支在 merge column 处通过垂直连通汇合。

Merge zone 特征: $z = [\text{row}/H, 0.8, 0.1, 0.1]$（中性 texture，轻微可辨识）。

### 8.4 侧线元数据 (GTETMeta)

```python
@dataclass
class GTETMeta:
    goal_cue_tags: np.ndarray        # (H,W) int: -1=none, 0/1/2=subgoal
    temptation_cue_tags: np.ndarray  # (H,W) float: 0=none, >0=lure intensity
    preference_cue_tags: np.ndarray  # (H,W) int: -1=none, 0=safe, 1=shiny
    goal_consistent_routes: dict     # subgoal_id → [route indices]
    temptation_preferred_routes: dict
    latent_explanation_overlap: list  # per-stage ambiguity
    subgoal_reveal_order: list       # [(col, "sg_i"), ...]
```

---

## 9. PRS Session 中的场景调度

### 9.1 Block 结构

PRS session 使用 4 个 block，每个 block 调度多个 episode：

| Block | 名称 | 场景分布 |
|-------|------|---------|
| A (30 ep) | Training | 基础难度 IID |
| B (15 ep) | IID test | 与 A 同分布 |
| C (15 ep) | Topology shift | 改变结构参数（route_count 等） |
| D (15 ep) | Semantic shift | 改变语义参数（lure_strength 等） |

### 9.2 WeightMode 轴

- `episode_random`: 每个 episode 独立采样 WorldWeights（负控制）
- `session_shared`: 全 session 共享同一组 WorldWeights
- `session_perturbed`: session WorldWeights + 少量扰动

### 9.3 场景 Shift 实现

Topology shift (Block C): 通过 `user_cfg` 修改结构参数：
- `route_count`, `branching_schedule`, `merge layout`
- `has_locked_fast_lane`, `belt placement`

Semantic shift (Block D): 通过 `user_cfg` 修改语义参数：
- `lure_strength`, `goal_cue_reliability`, `temptation offset`

---

## 10. 总参数对照表

### 10.1 DTMB 完整参数表

| 参数 | Easy | Medium | Hard | 影响 |
|------|------|--------|------|------|
| H × W | 13×35 | 15×45 | 17×60 | 格子尺寸 |
| tree_depth | 3 | 3 | 3 | 树深度 |
| branching_schedule | [3,2] | [3,2] | [4,2] | 分支数 |
| stage1_cue_reliability | 0.85 | 0.65 | 0.45 | S1 线索可靠性 |
| stage1_reveal_depth | 2 | 3 | 4 | S1 强线索出现深度 |
| stage1_commit_depth | 5 | 4 | 3 | S1 不可回退点 |
| mid_door_fraction | 0.15 | 0.25 | 0.35 | S2 门出现概率 |
| door_gain | 6 | 4 | 3 | 门打通省的步数 |
| terminal_belt_fraction | 0.20 | 0.35 | 0.50 | S3 belt 宽度比例 |
| belt_risk | 0.30 | 0.45 | 0.40 | S3 belt 风险值 |
| lure_subtree_fraction | 0.15 | 0.25 | 0.35 | S1 诱惑子树比例 |
| lure_strength | 0.35 | 0.55 | 0.80 | S1 诱惑强度 |
| deadline_ratio | 1.25 | 1.15 | 1.16 | 时间预算比 |
| misleading_fraction | 0.05 | 0.15 | 0.25 | S1 误导线索比例 |
| search_budget | 30 | 35 | 40 | A* 搜索预算 |

### 10.2 GTET 完整参数表

| 参数 | Easy | Medium | Hard | 影响 |
|------|------|--------|------|------|
| H × W | 13×40 | 15×50 | 17×60 | 格子尺寸 |
| stage1_branch_count | 3 | 3 | 3 | S1 分支数 |
| stage2_branch_count | 2 | 2 | 3 | S2 每 parent 子分支数 |
| stage3_belt_width | 3 | 4 | 5 | S3 belt 宽度 |
| goal_cue_reliability | 0.70 | 0.50 | 0.35 | 目标线索可靠性 |
| goal_cue_leadlag | +2 | 0 | -2 | 目标线索超前/滞后 |
| lure_strength | 0.40 | 0.65 | 0.85 | 诱惑强度 |
| tempt_offset_z | 0.30 | 0.50 | 0.70 | 诱惑偏移 |
| goal_ambiguity | 0.35 | 0.55 | 0.75 | 目标歧义度 |
| preference_cue_strength | 0.60 | 0.45 | 0.30 | 偏好线索强度 |
| deadline_slack_final | 1.25 | 1.15 | 1.08 | 时间预算比 |
| has_locked_fast_lane | No | Yes | Yes | S3 快速通道 |
| belt_risk | 0.25 | 0.35 | 0.45 | S3 belt 风险 |
| search_budget | 30 | 35 | 40 | A* 搜索预算 |

### 10.3 V2 基础 Lattice 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| H | 7 | 固定 7 行 |
| n_segments | 3 | 默认 3 段 |
| seg_width | $U(5, 7)$ | 每段列数 |
| detour_len | 1 | 固定 |
| P(trap) | 0.50/0.70/0.90 | easy/medium/hard |
| trap_risk | $U(0.30, 0.50)$ | 陷阱 risk |
| weak_cue_risk | $U(0.15, 0.25)$ | 弱线索 risk |
| normal_lane_risk | $U(0.08, 0.15)$ 或 $U(0.05, 0.10)$ | 普通 lane risk |

---

## 11. 附录：场景注册表

[scenario_families.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/scenario_families.py) 中的完整注册表：

```python
SCENARIO_REGISTRY = {
    "baseline_v2":              generate_baseline_v2,
    "fork_trap":                generate_fork_trap,
    "hazard_belt":              generate_hazard_belt,
    "deadline_gate":            generate_deadline_gate,
    "delayed_corridor":         generate_delayed_corridor,
    "distractor_cue":           generate_distractor_cue,
    "funnel_trap":              generate_funnel_trap,
    "elcb":                     generate_elcb,
    "elcb_po":                  generate_elcb_po,
    "temptation_corridor":      generate_temptation_corridor,
    "joint_conflict_corridor":  generate_joint_conflict_corridor,
    "deep_tree_mixed_bottleneck_lattice":               generate_dtmb_lattice,
    "goal_preference_temptation_entanglement_lattice":  generate_gtet_lattice,
}
```

所有场景通过统一入口调用：

```python
gm, cfg, meta, sc = generate_scenario(
    family="...",
    seed=42,
    difficulty="medium",
    latent_mode=True,
    user_cfg={...},  # 可选覆写
)
```
