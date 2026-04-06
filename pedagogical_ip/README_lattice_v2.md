# Lattice V2 环境技术文档

> **项目:** pedagogical_ip | **更新:** 2026-03-12 | **阶段:** L2C.1 complete

---

## 1. 研究问题

一个有限理性的 agent 在含有陷阱的网格中导航，tutor 可以通过两种方式帮助 agent：

- **结构干预（关门）**：关闭通向危险区域的入口门，强制 agent 走安全路线
- **语言沟通（warning）**：发出特征级别的警告，引导 agent 自己避开危险

核心问题：**沟通能否替代干预？** 在什么条件下 warning 的效果等同于甚至优于直接关门？

### 设计约束

1. **无风险作弊**：Agent 不能直接读到 cell 的 risk 数值，只能观测含噪声的 feature 向量
2. **门初始全开**：Tutor 只能关门（不可逆），不能开门
3. **有限理性**：Agent 用 bounded A*（最多 30 次扩展）做路径规划

---

## 2. 网格几何结构

7 行 × W 列的网格，W 随机（取决于 3 个 segment 的宽度之和）。

```
Row 0: ██████████████████████████████████  (全墙)
Row 1: ██ !!!! █ !!!! █ !!!! ██████████  (risky lane — 直线，路径短)
Row 2: █S ████ · ████ · ████ G█████████  (走廊 — segment 内封死)
Row 3: ██ ··█· █ ··█· █ ··█· ██████████  (safe lane — 有弯道，路径长)
Row 4: ███ ·  ████ ·  ████ ·  █████████  (detour 连接器)
Row 5: ███ ·  ████ ·  ████ ·  █████████  (detour 水平段)
Row 6: ██████████████████████████████████  (全墙)
```

| 元素 | 位置 | 说明 |
|------|------|------|
| S (起点) | (2, 1) | Agent 出发位置 |
| G (终点) | (2, W−2) | 目标位置 |
| 走廊 | Row 2 | segment 之间可通行，segment 内封墙 → 强制 lane choice |
| Risky lane | Row 1 | 直线，路径短 |
| Safe lane | Row 3→4→5→4→3 | 经过 detour 绕弯，路径长 |

---

## 3. Segment 结构

共 **K=3 个 segment**，顺序排列。每个 segment 是一个独立的"双车道"块：

```
         ┌─── segment ───┐
Row 1:   Gate→→→→→→→Gate     ← risky lane（直线，L_r 步）
Row 2:   Entry  墙墙  Exit   ← 走廊入口/出口
Row 3:   Gate→→↓→→→→Gate     ← safe lane 主路
Row 4:        ↓              ← detour 入口
Row 5:        →→→→↑          ← detour 水平段（d_k 列）
```

### Segment 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 宽度 `w_k` | {5, 6, 7}（随机） | 每个 segment 的列数 |
| detour 长度 `d_k` | 固定 = 1 | detour 弯道的列数 |
| 路径差 Δ | **固定 = 6 步** | safe lane 比 risky lane 多走 6 步 |

> 注：L2C.1 之前 d_k ∈ {1,2}，Δ ∈ [6,12]。为了让 warning 能与 detour 代价对抗，现在固定 d_k=1。

### Lane choice 的决策点

Agent 在 Row 2 的 segment 入口处做决策：
- 向上走 → 进入 risky lane（row 1）
- 向下走 → 进入 safe lane（row 3）

每个 segment 有两个 gate：
- `risky_entry_gate` = (1, col_start)
- `safe_entry_gate` = (3, col_start)

---

## 4. Feature 向量

每个 cell 携带一个 **4 维特征向量** x ∈ [0,1]⁴。Agent 不能直接看到 risk，只能通过观测这些 feature 间接推断：

| 维度 | 名称 | 含义 |
|------|------|------|
| x₀ | `lane_id` | 0.0 = risky lane, 1.0 = safe lane |
| x₁ | `gate_flag` | 1.0 = gate cell, 0.0 = 其他 |
| x₂ | `texture_cue_1` | 主要危险纹理（数值越高 → 越像陷阱） |
| x₃ | `texture_cue_2` | 次要危险纹理 |

### Feature 生成规则

| Cell 类型 | x₂ 范围 | x₃ 范围 | 真实风险 ρ |
|-----------|---------|---------|-----------|
| Safe lane | U(0, 0.1) | U(0, 0.1) | 0 |
| Risky（无 cue） | U(0.1, 0.2) | U(0.05, 0.15) | U(0.08, 0.15) |
| Weak cue（trap 邻居） | U(0.3, 0.5) | U(0.2, 0.4) | U(0.15, 0.25) |
| **Trap cell** | **U(0.8, 0.95)** | **U(0.7, 0.9)** | **U(0.3, 0.5)** |

生成函数定义在 [lattice_v2.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/envs/lattice_v2.py) 末尾：

```python
def _trap_feature(rng, lid):
    return [lid, 0.0, rng.uniform(0.80, 0.95), rng.uniform(0.70, 0.90)]

def _weak_cue_feature(rng, lid):
    return [lid, 0.0, rng.uniform(0.30, 0.50), rng.uniform(0.20, 0.40)]

def _safe_feature(rng, lid):
    return [lid, 0.0, rng.uniform(0.0, 0.1), rng.uniform(0.0, 0.1)]
```

---

## 5. Trap 放置

每个 segment 是否含有 trap 是**概率性**的：

| 难度 | P(segment 有 trap) | 时间系数 T_max |
|------|-------------------|---------------|
| Easy | 50% | 1.5 × L_safe |
| **Medium（当前工作点）** | **70%** | **1.4 × L_safe** |
| Hard | 90% | 1.3 × L_safe |

当 segment 有 trap 时：
- Risky lane 中**随机选 1 个 cell** 作为 trap cell
- Trap cell 的**左右相邻 cell** 变成 weak-cue cell（texture 略高，起"线索"作用）
- 其他 risky cell 保持低 texture

---

## 6. 死亡模型

死亡是**概率性的**，不是确定性的：

```python
# Agent 踩到风险为 ρ 的 cell 时
if random() < ρ:
    # 死亡，episode 立即结束
else:
    # 存活，继续移动
    # 但 risk_head 获得一个 "survived risky cell" 的学习信号
```

当前 trap cell 的 ρ ∈ [0.3, 0.5]，即踩上去有 **30–50% 概率死亡**。

---

## 7. 时间限制

```
T_max = time_ratio × L_safe
```

- `L_safe` = 全程走 safe lane 的 BFS 最短路径长度
- `time_ratio` = 1.3（当前工作点），即 agent 有 **1.3 倍安全路径时间** 来完成任务
- 时间用完则 episode 结束（算存活但未达成目标）

---

## 8. Agent 感知系统

### 8.1 观测模型

Agent **每一步**只能观测自己和 1-hop 邻居（上下左右+对角，共 ≤9 个 cell）的 feature：

| 位置 | 观察噪声 σ²_obs | 效果 |
|------|----------------|------|
| 自身 cell | 0.01 | 几乎精确观测 |
| 1-hop 邻居 | 0.08 | 有信息但含噪声 |

### 8.2 Feature Belief Map

Agent 对每个 cell 维护一个 **高斯信念** (μ, σ²)：

- **初始**：μ₀ = [0.5, 0.5, 0.5, 0.5]，σ₀² = 0.25（完全无知）
- **更新**：Kalman filter

```
K = σ²_prior / (σ²_prior + σ²_obs)       # Kalman gain
μ_post = μ_prior + K × (z_obs − μ_prior)  # 均值更新
σ²_post = σ²_prior × (1 − K)              # 方差缩小
```

只有 agent **走到或路过** 一个 cell 时，才会更新其信念。远处的 cell 保持先验。

### 8.3 Bayesian Risk Head

Agent 用一个**共享的线性模型**从 feature 预测风险：

```
ρ_hat(x) = sigmoid(w · x + b)
```

- w ∈ ℝ⁴, b ∈ ℝ，初始为 0
- **Online MAP 更新**，学习率 η = 0.3
- 不同事件的学习权重不同：

| 事件 | 标签 y | 权重 |
|------|--------|------|
| 踩到 trap 死亡 | 1.0 | 4.0 |
| 踩到 risky cell 存活 | ρ_true | 1.5 |
| 走过 safe cell | 0.0 | 0.1 |

### 8.4 Planner: Bounded A*

Agent 用 A* 算法做路径规划，cell cost 公式：

```
c_plan(s) = c_base(s) + λ_r × φ(ρ_hat(x_s)) + λ_u × u_hat(x_s)
```

| 符号 | 含义 | 当前值 |
|------|------|--------|
| c_base | 基础移动代价（normal=1, wall=∞） | — |
| φ(ρ) = −log(1−ρ) | 存活形式的风险惩罚 | — |
| λ_r | 风险权重 | 5.0 |
| λ_u | 不确定性权重 | 0.1 |
| budget | 最大扩展节点数 | 30 |

**关键性质**：初始时所有 cell 的 belief feature 都是 [0.5, 0.5, 0.5, 0.5]，风险预测 ρ_hat = sigmoid(0) = 0.5，所以 planner **一开始不偏好任何 lane** — 它会选最短的（= risky lane）。

---

## 9. 门的机制

- **所有门一开始都是打开的**
- Tutor 可以**关闭** risky_entry_gate → 该 cell 变成 wall → agent 被迫走 safe lane
- 关门**不可逆**（一旦关就不能开）
- 实验中用 `closure_budget` 限制每个 episode 最多关几扇门

---

## 10. Tutor: Time-Aware Door Tutor

Tutor 在 agent 接近 segment 入口（Row 2，距离 ≤1 列）时触发。

### Slack 计算

```
slack = (T_left − L_safe_remaining) / T_left
```

- `T_left` = 剩余步数
- `L_safe_remaining` = 从当前位置到终点（只走 safe lane）的 BFS 距离

### 三种模式

| 模式 | 条件 | 行为 |
|------|------|------|
| **Tight** | slack < 0.3 | **直接关门**（时间紧，保命优先） |
| **Medium** | 0.3 ≤ slack < 0.7 | 有 trap → 关门；无 trap → 发 warning |
| **Loose** | slack ≥ 0.7 | **Warning-first**：先发 warning，不关门 |

Loose mode 是 L2C.1 的关键改动：之前 medium 和 loose 行为相同（都是"有 trap 就关"），现在 loose mode **真正给 communication 留空间**。

---

## 11. Warning 系统（L2C.1）

### 11.1 Utterance 词表

| Utterance | Prototype 向量 | 伪标签 y_u |
|-----------|---------------|-----------|
| RISKY_TEXTURE_AHEAD | [0.5, 0.0, 0.85, 0.80] | 0.8 |
| UPPER_LANE_RISKY | [0.0, 0.0, 0.70, 0.60] | 0.7 |
| SAFE_DETOUR_OPEN | [1.0, 0.0, 0.05, 0.05] | 0.0 |

### 11.2 双层作用机制

Warning 通过**两层**影响 agent：

#### 第一层：伪标签注入（跨回合学习）

对 warned segment 的每个 risky cell j：

```
α_j = exp(−||x̂_j − prototype||² / τ)         # feature 匹配权重（τ=0.3）
risk_head.update(x̂_j, y=0.8, weight=5.0 × α_j) # 注入伪标签到共享 risk head
```

效果：risk head 的权重被推向"高 texture = 高风险"的方向，影响**未来 episode** 的决策。

#### 第二层：Lane-level bias（当回合即时效果）

```
bias = λ_lw × Σ_j (α_j × y_u)    # 聚合 lane 惩罚（λ_lw=5.0）

# 对 warned segment 的每个 risky cell：
planner_cost[cell] += bias         # 直接加到 A* 的代价上
```

效果：整条 risky lane 的 planner 代价大幅增加 → planner 转向 safe lane。

### 11.3 Utterance 选择（Action-Gap 目标）

不再优化"哪个 utterance 最能改变 risk prediction"（旧版），而是优化"哪个 utterance 最能改变 lane choice"：

```
u* = argmax_u  λ_lw × Σ_j α_j(u) × y_u
```

即：选那个能产生最大 lane-level bias 的 utterance。

---

## 12. 当前工作点与核心参数

| 参数 | 值 | 来源 |
|------|----|------|
| n_segments | 3 | lattice_v2.py |
| difficulty | medium | — |
| trap_risk | [0.3, 0.5] | lattice_v2.py L272 |
| trap_prob (per seg) | 70% | lattice_v2.py L261 |
| time_ratio | 1.3 | sweep script |
| detour_len | 1 (Δ=6) | lattice_v2.py L94 |
| feature_dim | 4 | lattice_v2.py L39 |
| obs_noise_self | 0.01 | observation_model.py |
| obs_noise_neighbor | 0.08 | observation_model.py |
| risk_head lr | 0.3 | risk_model.py |
| λ_risk (planner) | 5.0 | planner_astar.py |
| λ_uncertainty | 0.1 | planner_astar.py |
| λ_lane_warn | 5.0 | sweep script |
| warn weight | 5.0 | warning_update.py |
| warn τ | 0.3 | warning_update.py |
| A* budget | 30 | planner_astar.py |

---

## 13. 实验结果摘要

### 主条件矩阵（N=100 seeds）

| 条件 | 存活率 | 关门数 | Warning数 | 进入risky |
|------|--------|--------|-----------|-----------|
| no_tutor | **9%** | 0 | 0 | 5.8 |
| **warning_only** | **80%** | **0** | **2.8** | **1.7** |
| door_budget_2 | 68% | 1.9 | 0.5 | 2.4 |
| door_budget_3 | 99% | 2.3 | 0.7 | 0.3 |
| always_close | 100% | 3.0 | 0 | 0 |

### Lambda lane-warn sweep

| λ_lw | 存活率 |
|------|--------|
| 1 | 9%（无效果） |
| 3 | 46%（开始起作用） |
| 5 | 80% |
| 7 | 100%（饱和） |

> **核心发现**：`warning_only (80%) > door_2 (68%)` — 纯沟通首次超越结构干预。

---

## 14. 代码结构

```
pedagogical_ip/
├── src/
│   ├── envs/
│   │   └── lattice_v2.py              # 网格生成、segment、feature、trap
│   ├── agents/
│   │   ├── feature_belief.py          # FeatureBeliefMap（Kalman，d=4）
│   │   ├── risk_model.py              # BayesianRiskHead（sigmoid linear）
│   │   ├── observation_model.py       # observe_features()
│   │   ├── planner_astar.py           # cell_cost_v2 + warned_cell_extra_cost
│   │   └── warning_update.py          # 3 utterances + lane bias + action-gap select
│   └── teachers/
│       └── time_aware_door_tutor.py   # 三模式 tutor（tight/medium/loose）
├── scripts/
│   ├── _diag_l2c1_sweep.py            # L2C.1 完整 sweep + transfer eval
│   └── _diag_l2b5_sweep.py            # 非饱和调参 sweep
└── results/
    └── l2c1_sweep3.txt                # 最新实验结果
```
