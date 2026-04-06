# 系统架构 Q&A 报告

---

## Q1: Tutor 如何读取/同步 Agent 的 belief？

### 答案：**直接拷贝，每步同步**

Tutor 通过 `RobotBelief`（"机器人对学生的心智模型"）来获取 agent 的内部状态。它**不是**通过观察 agent 行为来推断，而是**直接读取** agent 的 belief 数组并拷贝一份。

#### 具体流程

```
每一步 step():
  1. observe_features()   ← agent 观察环境、更新 feature_belief
  2. sync_robot_belief()  ← tutor 直接拷贝 agent 的 belief
  3. apply_tutor()        ← tutor 用拷贝的 belief 做决策
  4. plan_and_move()      ← agent 规划并移动
```

#### 同步方式（3种 copy_mode）

| 模式 | 行为 | 失真程度 |
|------|------|----------|
| `exact` | 每步完整拷贝 `belief_mean` + `belief_var` + predictor 权重 | 零失真 |
| `noisy` | 拷贝后额外加高斯噪声 `N(0, 0.05)` | 信噪比可控 |
| `stale` | 每 N 步才同步一次（默认 N=3） | 信息滞后 |

#### 代码位置

- 初始化：`init_robot_belief()` @ `robot_belief.py:60`
  - 读取 `agent_belief_mean`, `agent_belief_var`
  - 拷贝 `latent_predictor` 的线性头权重 (w, b)
- 每步同步：`sync_robot_belief()` @ `robot_belief.py:109`
  - 在 `lattice_v2_runner.py:427` 被调用
- 反事实决策：`build_surrogate_predictor()` @ `robot_belief.py:145`
  - 用快照权重构造只读的 `LatentCostRiskHead`

#### 关键约束

```python
# robot_belief.py 文件头注释
# CRITICAL: The robot-belief tutor must NOT access hidden true trap cells,
# hidden latent vectors, or true future risk values. Only agent-observable
# state and segment topology are allowed.
```

但实际上 tutor 拷贝的是 agent 已经计算好的 belief，不是 ground truth。如果 agent 对某个 cell 的 belief 是错的，tutor 也会继承这个错误。

---

## Q2: Agent 如何知道目的地？如何导航？

### 答案：**目的地是明确给定的。导航用 Bounded A\* 加信念估计。**

#### 目的地

Agent 的目标位置 `goal` 是在 `reset()` 时由 `GridMap.target_pos` 直接设定的，agent**完全知道**目的地在哪里：

```python
# lattice_v2_runner.py:321
_goal = getattr(gm, 'target_pos', (2, W - 2)) or (2, W - 2)
```

在 baseline_v2 中，goal 通常在 `(2, W-2)`（右侧末端）。Agent 始终知道目标坐标。

#### 导航算法：Bounded A\* + 信念代价

Agent 使用 `plan_from_belief()` 做路径规划：

```python
# lattice_v2_runner.py:467
bp = plan_from_belief(
    s.agent_pos, s.goal, s.belief_cost, s.feature_belief.mean,
    s.risk_head, s.passable,
    latent_predictor=s.latent_predictor,
    warned_cell_extra=extra,
    ...
)
next_pos = bp.next_pos  # 下一步走哪
```

#### 代价估算公式

Agent 不知道 true_cost 和 true_risk。它通过 `LatentCostRiskHead` 从 4D feature belief 预测：

```
ĉ(cell) = w_c^T · x̂(cell) + b_c      ← 预测代价
r̂(cell) = σ(w_r^T · x̂(cell) + b_r)   ← 预测风险概率
```

其中 `x̂(cell)` 是 agent 对该 cell 的 4D 特征 belief 均值。

A\* 搜索的 edge cost 是：
```
total_cost(cell) = belief_cost(cell) + λ_risk · r̂ + λ_unc · û + warned_extra(cell)
```

#### Agent 不知道的

- **真实特征向量** — 只有带噪声的观测
- **真实代价/风险** — 只有从 belief 预测的估计
- **哪些 cell 是 trap** — trap 的特征和 risky cell 类似但概率更高
- **Tutor 的决策逻辑** — agent 不知道 tutor 为什么 WARN/UNLOCK

---

## Q3: Tutor 能看到什么？不能看到什么？

### 答案：**Tutor 知道场景拓扑（哪里有风险），但不能看到 true risk scalar 或 true latent。**

#### Tutor 能看到的

| 信息 | 来源 | 说明 |
|------|------|------|
| **场景拓扑** (`SegmentMeta`) | `s.meta.segments` | 知道哪些 cell 属于 risky lane、safe lane |
| **Agent 位置** | `s.agent_pos` | 实时知道 agent 在哪 |
| **Agent belief 拷贝** | `RobotBelief` | 通过 sync 获取 |
| **Agent predictor 权重** | `RobotBelief._predictor_*` | 知道 agent 学到的线性头 |
| **可通行性** | `s.passable` | 知道哪些 door 开着/关着 |
| **时间/步数** | `s.t, s.t_max` | 知道 deadline |

#### Tutor **不能**看到的

| 信息 | 原因 |
|------|------|
| **true_risk[r,c]** — 每个格子的真实风险概率 | 明确被约束禁止 |
| **true_cost[r,c]** — 每个格子的真实代价 | 明确被约束禁止 |
| **cell_features[r,c]** — 真实 4D 特征向量 | 只能通过 agent belief 间接获取 |
| **trap 位置** — 哪个格子是 trap | trap 是 hidden ground truth |
| **Temptation latent z** (GTET) — 诱惑程度 | GTET 的 z 只在 posterior 中估计 |

#### 关键设计意图

Tutor 知道**结构**（"这个 segment 的上方车道是 risky lane"），但不知道**具体数值**（"这个格子风险 0.3"）。这模拟了教师知道"这条路可能危险"但不知道具体危险程度的场景。

Warning 系统（`_apply_segment_warning`）的工作方式是：tutor 知道 `seg.risky_cells`（结构性知识，哪些 cell 在 risky lane），然后发出方向性警告（"左边危险"），但不告诉 agent 具体风险值。

---

## Q4: Cell Type 有哪些？各有什么作用？

### 答案：7 种 CellType，定义在 `map_generator.py:17`

```python
class CellType(IntEnum):
    NORMAL      = 0   # 普通格
    WALL        = 1   # 墙壁
    HIGH_COST   = 2   # 高代价地形
    RISKY       = 3   # 风险地形
    LOCKED_DOOR = 4   # 锁门
    TARGET      = 5   # 目标/终点
    OBJECT_SPAWN= 6   # 物品生成点
```

#### 详细说明

| CellType | true_cost | true_risk | 可通行 | 对 agent 的影响 |
|----------|-----------|-----------|--------|----------------|
| **NORMAL** | 1.0 | 0.0 | ✓ | 普通移动，无特殊效果 |
| **WALL** | ∞ | 0.0 | ✗ | 不可通行，完全阻挡 |
| **HIGH_COST** | 5.0 | 0.0 | ✓ | 可通行但代价高（5倍普通），agent 倾向绕开 |
| **RISKY** | 1.0 | 0.3 | ✓ | 代价正常但有 30% 概率触发风险事件。Agent 需要学习识别。**Warning 的目标就是这类 cell。** |
| **LOCKED_DOOR** | ∞→1.0 | 0.0 | 锁时 ✗ / 开时 ✓ | 初始不可通行。Tutor 可以 UNLOCK 打开它，打开后变成普通格。代表"教师控制的资源"。 |
| **TARGET** | 1.0 | 0.0 | ✓ | 目的地。Agent 到达此格 → episode 成功。 |
| **OBJECT_SPAWN** | 1.0 | 0.0 | ✓ | 物品生成点。Shield 道具会出现在这里。Agent 拾取 shield 可降低后续风险。 |

#### 与 4D Feature Vector 的关系

每个 cell 不是直接暴露 cost/risk 给 agent，而是通过一个 **4D feature vector** 间接表征：

```
f(cell) = [lane_id, gate_flag, texture_1, texture_2]
```

- `lane_id (f[0])`：标识该 cell 属于哪条车道
- `gate_flag (f[1])`：标识是否在 gate 附近
- `texture_1, texture_2 (f[2], f[3])`：纹理特征，由 `WorldWeights` 生成

Agent 通过 `LatentCostRiskHead` 学习 feature → cost/risk 的映射：
```
ĉ = w_c · f + b_c
r̂ = σ(w_r · f + b_r)
```

**同一类 CellType 的 feature 向量会相似但不完全相同**（受 WorldWeights 控制）。这就是 agent 需要"学习"的原因 — 它需要从 noisy feature observations 中学会识别 risky cells。

#### 与 Trap 的关系

`RISKY` cell 中有极少数被标记为 **trap**（`seg.trap_cell`），这些 cell 的 true_risk 更高（通常 0.7-0.9）。Trap 是 RISKY 的极端子集，和普通 RISKY cell 的 feature 向量高度相似，agent 很难仅从 feature 区分。

这构成了 warning 的核心价值：tutor 知道"哪个 segment 有 trap"（结构知识），agent 不知道。Tutor 的 WARN 帮助 agent 避开整个 risky lane，从而绕过 trap。
