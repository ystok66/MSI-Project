# Tutor 模型详解

> 本报告详细介绍当前系统中 Tutor（教师系统）的完整建模方式。
> Tutor 分为**微观层**（episode 内干预）和**宏观层**（across-episode 课程规划）两部分。

---

## 1. Tutor 是什么

Tutor 是一个**贝叶斯教学决策系统**。它不是一个 RL agent，也不是一个 exact POMDP solver。它通过维护对 learner 状态的信念，在每一步做 constrained greedy 决策，选择对 learner 长期学习最优的干预方式。

Tutor 的核心目标**不是**帮 agent 当前选对——而是：
- 短期：保证 agent 安全
- 长期：让 agent 学会**自主判断**（trust ↑, dependence ↓, exploration ↑）
- 边界：避免过度教学（γ_gen ↓, overteach rate ↓）

```
Tutor 系统
├── 微观层: BC-ICT-v4 (within-episode)
│     "这一节课里，要不要提示学生？"
│     动作: WAIT / SOFT(0.5) / WARN(1.0)
│
└── 宏观层: CurriculumControllerV13 (across-episode)
      "下一节课教什么？要不要考试？要不要结课？"
      动作: TEACH(lesson) / EVAL / STOP
```

---

## 2. 微观层：BC-ICT-v4

### 2.1 输入信息

| 输入 | 来源 | 说明 |
|------|------|------|
| `m` (内化状态) | ⚠️ **直接读取 Agent** | κ, τ, ν, γ_spec, γ_gen |
| `sc` (场景结构) | 环境 | 两条分支的 cell 列表、fork point、oracle safe branch |
| `fb` (cell 特征) | 环境 | 每个 cell 的 4D 语义向量 |
| `lp` (cost/risk 模型) | Agent 的已学模型 | `LatentCostRiskHead` |
| `obs` (观测深度) | 环境 | Agent 能看多远（默认 obs_radius=2） |

### 2.2 决策过程

对每个可能的干预剂量 `dose ∈ {0, 0.5, 1.0}`，Tutor 计算一个 Q 值：

$$Q(\text{dose}) = Q_{\text{online}} + \lambda_{\text{teach}} \cdot V_{\text{full}} - \lambda_{\text{over}} \cdot R_{\text{over}}$$

然后选 Q 最大的 dose。

#### ① $Q_{\text{online}}$：即时场景收益

```python
# dose = 0 (WAIT)
Q_online_wait = 2.0 * p_self * delta_s - 1.5 * p_fail + 2.0

# dose = 1.0 (WARN)
Q_online_warn = 1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self) + 1.0 * tempt - 0.05

# dose = 0.5 (SOFT)
Q_online_soft = 0.5 * Q_online_warn + 0.5 * Q_online_wait
```

其中：
- `p_self` = 估计 agent 依靠自己证据做出正确选择的概率（和 reveal/commit 时序有关）
- `p_fail` = 估计 WAIT 时 agent 失败的概率
- `delta_s` = 完整分支总结和部分可见总结之间的安全评分差距（"Tutor 的额外信息有多大价值"）
- `dvoi` = value of information 差距
- `tempt` = 场景诱惑强度

**直觉**：
- 场景越危险（tempt 高、p_fail 高）→ WARN 更有价值
- Agent 越有可能自己发现正确答案（p_self 高）→ WAIT 更有价值

#### ② $V_{\text{full}}$：教学价值

Tutor **模拟** "如果用这个 dose，agent 的内化状态会怎么变"：

```python
m_next = _predict_m(m, dose, tempt, risk, subtype, has_self_ev)

# 行为 loss 变化
V = L_now - L_next     # 当前 behavior loss 减去预测的下一步 loss
                        # V > 0 意味着"这个干预能改善 agent 的行为"

# 加上路径敏感项
V_full = V + λ_sd · P_sd - λ_dep · P_blind
```

| 项 | 含义 | 公式 |
|---|------|------|
| $L_{\text{now}}$ | 当前行为偏差 | `bridge_behavior_loss(m)` |
| $L_{\text{next}}$ | 预测干预后行为偏差 | `bridge_behavior_loss(m_next)` |
| $P_{\text{sd}}$ | self-discovery 概率 | 高 → WAIT 有更大的 "学生自己发现" 奖励 |
| $P_{\text{blind}}$ | 盲从概率 | 高 → WARN 有更大的 "增加依赖" 惩罚 |

#### ③ $R_{\text{over}}$：过度教学惩罚

```python
R_over = bridge_overteach_penalty(m_next, zones)
```

检测三个过度教学信号：
- **IA 太高**：agent 变得太服从了（连错误建议也照单全收）
- **EP 太低**：agent 的探索能力被过度抑制了
- **TR 太高**：agent 的诱惑抵抗过强（变得刻板了）

### 2.3 前向模拟：`_predict_m()`

这是微观 Tutor 的核心——它用 Agent 的内化动力学模型做**一步模拟**：

```
如果 dose > 0 (WARN / SOFT):
    κ: 更新风险校准（risk 低，说明 warning 帮忙避开了风险）
    τ: ↑ 信任增加（如果 risk > 0.25，saying warning 是准确的）
    ν: ↑ 依赖增加（如果 agent 没有自己证据）
       但 dose=0.5 时效果只有一半: ν_new = ν_old + 0.5·(ν_warn - ν_old)
    γ_gen: ↑ 探索抑制（sustained pressure）
       同样 dose<1.0 时效果减半

如果 dose = 0 (WAIT):
    κ: 根据实际 risk 更新
    ν: ↓ 如果 agent 自己发现了（self_discovery）
    γ_spec: ↑ 如果 agent 被诱惑骗了 + 高诱惑场景
    γ_gen: ↓ 如果 agent 成功自主探索（novel/self_discovery）
```

**关键设计**：SOFT (dose=0.5) 是 WARN 和 WAIT 的折中——信任增加相同，但依赖和探索抑制只增加一半。这提供了一个中间地带。

### 2.4 Behavior Bridge：行为预测

Tutor 使用 `behavior_bridge.py` 来预测 Agent 的行为表现。它是一个**半参数 logistic 映射**：

$$\hat{z}_p(m, c) = \sigma(w^\top \varphi(m, c))$$

其中特征向量 $\varphi$ 包含 13 个维度：

```python
φ = [1,                          # 偏置
     κ, τ, ν, γ_spec, γ_gen,    # 5 个原始状态
     κ·risk,                     # κ × 场景风险 (交互项)
     γ_spec·lure,                # γ_spec × 诱惑强度
     γ_gen·novelty,              # γ_gen × 新奇程度
     τ·(1-self_ev),              # τ × 缺少自我证据
     ν·self_ev,                  # ν × 有自我证据
     κ², γ_gen²]                 # 二次项
```

这些权重是**预先校准**的（从 baseline rollout 数据拟合），不是在线学习的。

### 2.5 Behavior Zones：行为目标区间

Tutor 有一个"理想行为区间"的概念——对每个 probe，定义了最低和最高目标：

| Probe | safe 目标区间 | shiny 目标区间 | 含义 |
|-------|:----------:|:----------:|------|
| RC | [0.55, 0.85] | [0.55, 0.85] | 太低=不会避险，太高=过度谨慎 |
| TR | [0.50, 0.80] | [0.55, 0.85] | 太低=无法抵抗诱惑，太高=刻板 |
| EP | [0.45, 0.70] | [0.40, 0.65] | **最关键**：太低=探索被抑制(过度教学) |
| VA | [0.55, 0.80] | [0.55, 0.80] | 太低=不听正确建议 |
| IA | [0.20, 0.45] | [0.20, 0.45] | **越低越好**：太高=盲从错误建议 |

行为 loss 在区间外按**二次惩罚**计算，权重不同（EP 和 IA 的权重最高 = 2.5）。

---

## 3. 宏观层：CurriculumControllerV13

### 3.1 输入信息

| 输入 | 来源 | 说明 |
|------|------|------|
| `m` (内化状态) | ⚠️ 直接读取 Agent | 当前的 5 维心理状态 |
| `u` (mastery) | Mastery Model | 5 维 Beta-Bernoulli 估计 |
| `h` (教学历史) | Controller 自身记录 | 教了哪些课、用了几节 |
| `B` (剩余预算) | Controller 自身计算 | total_budget - spent |
| `θ` (learner 类型) | 贝叶斯推断 | safe 还是 shiny |

### 3.2 五层决策架构

Controller 的每次决策按 5 层流水线执行：

```
输入: m_t, u_t, h_t, B_t
  │
  ▼
Layer 1: Feasibility Filter (可行性过滤)
  "哪些课 agent 有条件上？预算够不够？"
  │
  ▼
Layer 2: Risk Constraint Filter (风险约束过滤)
  "哪些课的预测风险在可接受范围内？"
  │
  ▼
Layer 3: Scoring (评分排序)
  "在可行的课里，哪个预期收益最高？"
  │
  ▼
Layer 4: Gated STOP Decision (停止判断)
  "是不是该结课了？三个 gate 同时满足才能停。"
  │
  ▼
Layer 5: EVAL Competition (考试决策)
  "是继续教好还是先考一次好？"
  │
  ▼
输出: TEACH(lesson) / EVAL / STOP
```

### 3.3 Layer 1: 可行性过滤

三个条件全满足才通过：

| 条件 | 公式/逻辑 | 说明 |
|------|----------|------|
| **预算可行** | `lesson.cost ≤ remaining_budget` | 没钱了就不能上 |
| **前置条件** | `feasibility(u) ≥ 0.3` | $\sigma(\beta \cdot (u \cdot p - \tau))$，mastery 太低的课不适合 |
| **家族多样性** | 最近 5 课内同 family ≤ 3 | 防止反复上同一类课 |

### 3.4 Layer 2: 风险约束过滤 (Filter+Rank)

对每个通过 Layer 1 的课，预测它的"伤害值"，如果超过了**θ-adaptive 风险预算**就过滤掉：

$$\mu_j(x_t, \ell) + \beta_{\text{pessimism}} \cdot \sigma_j(x_t, \ell) \le \eta_j^{(\theta)}(x_t)$$

检查三个风险维度：

| 风险维度 | 含义 | θ-adaptive 预算 |
|---------|------|----------------|
| **OTR** (overteach rate) | 过度教学率 | safe: 0.55, shiny: **0.75** |
| **ν** (dependence) | 依赖性增长 | safe: 0.40, shiny: **0.60** |
| **γ_gen** (general suppression) | 探索抑制 | safe: 0.25, shiny: **0.40** |

**shiny 的预算更宽**——因为 shiny learner 的理想课程本身就风险更高（高诱惑），如果用和 safe 一样的严格预算，有价值的课会被误过滤。

风险预算还会**动态调整**：
- mastery 低的维度 → 预算放宽（agent 还需要学习）
- ν 已经很高 → ν 预算收紧（保护 agent 独立性）
- γ_gen 已经很高 → γ_gen 预算收紧（保护探索能力）

### 3.5 Layer 3: 评分排序

对通过 risk filter 的课评分：

$$J_t(\ell) = \underbrace{w_{\text{gain}} \cdot G(\ell)}_{\text{预期学习收益}} + \underbrace{\lambda_{\text{unc}}^{\text{eff}} \cdot \sqrt{\text{Var}(G)}}_{\text{探索奖励}} - \underbrace{r_{\text{fid}}}_{\text{基线成本}} - \underbrace{r_{\text{rep}}}_{\text{重复惩罚}} + \underbrace{b_{\text{fam}}^{\text{eff}}}_{\text{家族偏好}}$$

| 项 | 含义 | 来源 |
|---|------|------|
| $G(\ell)$ | 预期增益 | PairwiseResponseModel 的 gain 预测 |
| $\lambda_{\text{unc}}^{\text{eff}} \cdot \sqrt{\text{Var}}$ | 探索奖励 | 预算多时探索更多，经验多时探索更少 |
| $r_{\text{rep}}$ | 重复惩罚 | 最近 5 课中重复选过此课的比例 |
| $b_{\text{fam}}^{\text{eff}}$ | 家族偏好（含衰减） | $b_f(q_t) \cdot \exp(-n_f / \tau_{\text{fam}})$ |

**探索奖励的双重衰减**：

$$\lambda_{\text{unc}}^{\text{eff}} = \lambda_0 \cdot \sigma\left(\frac{B - 0.6 \cdot B_{\text{total}}}{\tau_B}\right) \cdot \exp\left(-\frac{n_{\text{post}}}{\tau_n}\right)$$

- 第一项：预算少于 60% 时快速衰减（省着用）
- 第二项：response model 经验越多，探索值越低（exploitation 更好了）

**家族偏好衰减**：

$$b_f^{\text{eff}} = b_f(q_t) \cdot \exp(-n_f / \tau_{\text{fam}}^{(\theta)})$$

- $\tau_{\text{fam}}^{(\text{safe})} = 3.0$, $\tau_{\text{fam}}^{(\text{shiny})} = 2.0$
- shiny 衰减更快 → 更快地从偏好 family 转向其他 family
- 这解决了 shiny 对 PP-MRB 的过度依赖问题

### 3.6 Layer 4: Gated STOP

**三条件 AND-gate**——只有三个条件同时满足才允许停止教学：

#### Gate 1: Margin ($M_{\text{base}}$)

$$M_{\text{base}} = \epsilon_{\text{stop}}^{(\theta)} - \max_{\ell \in \text{feasible}} J_t(\ell) > 0$$

即"继续教的最大收益已经低于停止阈值"。

阈值本身是 θ-adaptive 的：

$$\epsilon_{\text{stop}} = \epsilon_0^{(\theta)} + a_\nu^{(\theta)} \cdot \nu + b_\gamma^{(\theta)} \cdot \gamma^{\text{gen}} + c_u \cdot \bar{u} + d_B \cdot B$$

| 参数 | safe | shiny | 意义 |
|------|:----:|:-----:|------|
| $\epsilon_0$ | 0.00 | **−0.10** | shiny 的基线阈更低 → 更难触发 STOP |
| $a_\nu$ | 0.04 | **0.005** | shiny 的 ν 对 STOP 的推动力只有 safe 的 1/8 |
| $b_\gamma$ | 0.05 | **0.005** | shiny 的 γ_gen 对 STOP 的推动力只有 safe 的 1/10 |

这解决了旧版本中 shiny 被 premature stop 的问题——因为 shiny 天然 ν 和 γ_gen 更高，如果用 safe 的系数它们会把阈值推得太高。

#### Gate 2: Warm-up ($G_{\text{warm}}$)

$$G_{\text{warm}} = \mathbf{1}[N_{\text{teach}} \ge T_{\min}^{(\theta)}]$$

- $T_{\min}^{(\text{safe})} = 2$：safe 至少教 2 节才能考虑停
- $T_{\min}^{(\text{shiny})} = 3$：shiny 至少教 3 节

#### Gate 3: Plateau ($G_{\text{plateau}}$)

$$\bar{\Delta}_u = \frac{1}{w} \sum_{i} \sum_p |u_t^{(p)} - u_{t-1}^{(p)}|$$

$$G_{\text{plateau}} = \mathbf{1}[\bar{\Delta}_u \le \tau_u^{(\theta)}]$$

- $w = 2$, $\tau_u^{(\text{safe})} = 0.02$, $\tau_u^{(\text{shiny})} = 0.015$
- 只有 mastery 已经不再变化（plateau）才允许停

**STOP 的最终条件**：

$$\text{STOP} \iff (M_{\text{base}} > 0) \;\land\; G_{\text{warm}} \;\land\; G_{\text{plateau}}$$

### 3.7 Layer 5: EVAL (考试)

如果不 STOP，Controller 还要决定"是继续教还是先考一次"：

$$J_t(\text{EVAL}) = \lambda_{\text{info}} \cdot \big(\text{Var}[u_t] + 0.5 \cdot \sqrt{\text{Var}[G]}\big) - c_{\text{eval}}$$

| 参数 | 值 | 含义 |
|------|:--:|------|
| $\lambda_{\text{info}}$ | 0.8 | 信息价值权重 |
| $c_{\text{eval}}$ | 0.3 | 考试的时间成本 |
| max_eval | 3 | 最多考 3 次 |

如果 $J_t(\text{EVAL}) > J_t(\text{best lesson})$，选择 EVAL。

**EVAL 的主要价值不是改变 lesson 排名，而是更新 mastery 估计**：

$$u_{t+1} = \text{BayesUpdate}(u_t, y_t^{\text{probe}})$$

更精确的 mastery → 更好的 STOP 决策 + 更好的 risk constraint 计算。

---

## 4. Pairwise Response Model

Controller 评估 lesson 收益时使用的是 `PairwiseResponseModel`——一个**分层贝叶斯 + 上下文残差 + pairwise 对比**的三层模型：

$$\mu^{\text{gain}}(x, \ell) = \underbrace{\mu^{\text{hier}}(\ell)}_{\text{经验贝叶斯}} + \underbrace{r^{\text{ctx}}(x, \ell)}_{\text{状态条件修正}} + \underbrace{G^{\text{pw}}(x, \ell)}_{\text{pairwise 排名}}$$

| 层 | 含义 | 特点 |
|---|------|------|
| $\mu^{\text{hier}}$ | 跨 lesson 的**平均收益**估计 | 方差低，数据少时也稳定（shrinkage） |
| $r^{\text{ctx}}$ | 基于当前 agent 状态的修正 | 能捕捉 "同一课对不同状态的 agent 效果不同" |
| $G^{\text{pw}}$ | Pairwise 对比排名 | 从 counterfactual replay 中学到的相对排序 |

$G^{\text{pw}}$ 是**真正驱动 lesson argmax 变化的主信号**（PCR = 79-83%）。

---

## 5. 信息流总览

```
Episode N 开始
├── Controller 选动作 → TEACH("sparse_invalid_advice")
│   ├── Layer 1: 过滤不可行课
│   ├── Layer 2: 过滤高风险课
│   ├── Layer 3: 评分排序
│   ├── Layer 4: STOP 三门检查 → 不满足 → 不停
│   └── Layer 5: EVAL vs TEACH → TEACH 胜出
│
├── 生成 episode (adaptive_episode_generator_v2)
│   └── lesson + seed + θ + mastery → (地图, 场景结构, 特征)
│
├── BC-ICT-v4 做微观决策
│   ├── 观察: 分支 summary, reveal/commit 时序, temptation
│   ├── 对 dose=0/0.5/1.0 各模拟一次 _predict_m()
│   └── 选 dose=0.5 (SOFT)
│
├── Agent 做选择
│   ├── 看到分支特征 + SOFT warning
│   ├── 计算 U(branch_A) vs U(branch_B)
│   └── softmax 采样 → 选了 branch_A (正确)
│
├── 更新
│   ├── Agent 内化状态: τ ↑, ν slightly ↑, γ_gen slightly ↑
│   ├── Controller mastery: probe → u_t 更新
│   ├── Response model: 记录这次 lesson 的 gain/harm 结果
│   └── Family counts: TIC-v4 的使用次数 +1
│
Episode N+1 开始...
```

---

## 6. Tutor 知道但 Agent 不知道的

| 信息 | 说明 |
|------|------|
| 哪条分支是 oracle safe | Tutor 知道正确答案 |
| 完整地图的 true cost/risk | Tutor 能看到所有 cell |
| Agent 观测不到的深处分支信息 | Tutor 的 WARN 本质上是在传递这些信息 |
| 教学计划和剩余预算 | Agent 不知道还要上几节课 |
| 之前所有课的 lesson response 记录 | 用于 pairwise 排名 |

## 7. Tutor 不知道但需要推断的

| 信息 | 推断方式 |
|------|---------|
| Agent 的 θ 类型 | ✅ Bayesian posterior $q_t(\theta)$，通过观察选择行为更新 |
| Agent 的 mastery | ✅ 通过 behavior probes 估计（有噪声） |
| Agent 的内化状态 $m_t$ | ⚠️ **当前直接读取**（理想应推断） |

---

## 8. 当前的简化与待改进

### 8.1 微观层

| 简化 | 影响 | 改进方向 |
|------|------|---------|
| 只有 WARN，没用 UNLOCK/ITEM_DROP | 干预手段单一 | 接入已有的 `interventions.py` |
| Tutor 直接读 $m_t$ | 不符合 nested ToM | 应通过行为推断 $\hat{m}_t$ |
| 只做一步前向模拟 | 没有多步规划 | 可做 2-3 步 lookahead |
| Bridge 权重预校准 | 不随 session 更新 | 可做在线 bridge 微调 |

### 8.2 宏观层

| 简化 | 影响 | 改进方向 |
|------|------|---------|
| Greedy 决策（不做多步 planning） | 可能局部最优 | 但 STOP+EVAL 弥补了大部分 |
| 10 个 lesson 的离散选择 | 课程空间有限 | 但当前已足够暴露核心问题 |
| Family prior 手工设定 | 可能不最优 | 但 exp decay 让它自动弱化 |
| Pairwise replay 数据有限 | G_pw 方差可能不够小 | 增加 replay diversity |
