# Agent 模型详解

> 本报告详细介绍当前系统中 Agent（学习者）的完整建模方式。

---

## 1. Agent 是什么

Agent 是一个**模拟的人类学习者**。它不是一个待训练的 RL agent，也不是一个最优规划器。它是一个**有内部心理状态的有界理性决策者**（bounded-rational decision maker），模拟的是"一个正在学习如何在风险环境中做决策的学生"。

Agent 由三个模块组成：

```
┌───────────────────────────────────────────────────────┐
│                      Agent                            │
│                                                       │
│  ① 感知模块 ─── 从环境特征学习 cost/risk              │
│  ② 选择模块 ─── 在 fork point 选择分支                │
│  ③ 内化模块 ─── 心理状态随经历持续变化                 │
└───────────────────────────────────────────────────────┘
```

---

## 2. 感知模块：Agent 如何理解环境

### 2.1 Agent 看到什么

每个地图 cell 有一个 4D 语义特征向量 $z = [z_0, z_1, z_2, z_3]$：

| 维度 | 含义 | 对风险的决定作用 |
|:---:|------|:--------------:|
| $z_0$ | 位置身份标签（lane_id） | ❌ 无关 |
| $z_1$ | 结构标记（gate_flag） | 很弱 |
| $z_2$ | 纹理特征 A（texture_1） | ✅ 强信号 |
| $z_3$ | 纹理特征 B（texture_2） | ✅ 强信号 |

**关键设计**：真实世界的 cost 和 risk 由以下线性模型决定：

$$\text{cost}(z) = w_c^\top z + b_c$$

$$\text{risk}(z) = \sigma(w_r^\top z + b_r)$$

其中 $w_r$ 在 identity 维度上**被置零** ($w_r[0] = 0$)，保证风险只取决于语义特征，不取决于位置编号。

### 2.2 Agent 怎么学

Agent **不知道** $w_c, w_r$ 的真实值。它使用 `LatentCostRiskHead`——两个**在线贝叶斯线性回归模型**来学习：

```
BayesianCostHead:  从 (z, observed_cost) 对学习 w_c
BayesianRiskHead:  从 (z, observed_risk) 对学习 w_r
```

$$\hat{w}_c \leftarrow \hat{w}_c - \eta \cdot \big( -(y_{\text{cost}} - \hat{w}_c^\top z) \cdot z + \hat{w}_c / \sigma^2_{\text{prior}} \big)$$

即 **梯度下降 + 高斯先验正则化**。

**学习数据来源**：
- Agent 只在**访问过的 cell** 上获得 true cost/risk（revelation upon visitation）
- 未访问的 cell 只能基于其特征向量 $z$ 做预测
- 所以 Agent 的学习是 **exploration-dependent** 的

### 2.3 感知 → 分支级总结

Agent 在 fork point 面对两条分支时，不是逐 cell 比较，而是通过 `summarize_branch()` 将一条分支的所有 cell 聚合成一个 **8D 语义总结向量**：

| 维度 | 含义 |
|:---:|------|
| mean_risk | 分支平均预测风险 |
| max_risk | 分支最大预测风险 |
| mean_cost | 分支平均预测代价 |
| risk_unc | 平均风险不确定性 |
| cost_unc | 平均代价不确定性 |
| cue_count | 高纹理 cell 数占比（"线索密度"） |
| cue_var | 纹理多样性 |
| length | 分支长度（归一化） |

从这个总结中，提取出 `BranchAttributes`（safety_score, temptation_score, risk_penalty），作为选择模块的输入。

---

## 3. 选择模块：Agent 如何做决定

### 3.1 效用函数

在 fork point，Agent 对每条分支 $i$ 计算内部效用：

$$U_i = \lambda_\theta \cdot R_{\text{pref}}(\theta, x_i) - \kappa^2 \cdot \text{risk}_i - \gamma^{\text{spec}} \cdot \text{tempt}_i - \gamma^{\text{gen}} \cdot \text{novel}_i + \tau \cdot \text{warn}_i - \nu \cdot \text{obey}_i$$

每一项的含义：

| 项 | 公式 | 含义 |
|---|------|------|
| **偏好奖励** | $\lambda_\theta \cdot w_\theta^\top [s, t, n, c]$ | 不同 $\theta$ 类型对分支属性的先天偏好 |
| **风险惩罚** | $-\kappa^2 \cdot \text{risk}$ | 风险敏感度越高，越回避高风险分支 |
| **诱惑抑制** | $-\gamma^{\text{spec}} \cdot \text{tempt}$ | 已学会的诱惑抑制能力 |
| **探索抑制** | $-\gamma^{\text{gen}} \cdot 0.3$ | 过度教学导致的广泛探索抑制（**有害**） |
| **信任加成** | $+\tau \cdot \text{warn}$ | 如果 Tutor 发了 warning，信任高则更可能听从 |
| **盲从成本** | $-\nu \cdot 0.2$ | 如果 Agent 在没有自己证据的情况下听从，有内在不适感 |

### 3.2 偏好类型 θ

$$R_{\text{pref}}(\theta, x) = w_\theta^\top \begin{bmatrix} \text{safety} \\ \text{temptation} \\ \text{novelty} \\ \text{shortcut} \end{bmatrix}$$

| θ | safety | temptation | 性格描述 |
|:-:|:------:|:----------:|---------|
| **safe** | +2.0 | −1.0 | 保守型：喜欢安全，讨厌诱惑 |
| **shiny** | 0.0 | +3.0 | 冒险型：热爱诱惑，不在意安全 |

**Tutor 不知道 Agent 是哪种类型**，通过观察 Agent 的行为用贝叶斯推断：

$$q_{t+1}(\theta) \propto q_t(\theta) \cdot P(\text{observed choice} \mid \theta)$$

### 3.3 选择概率

$$P(\text{choose branch } i \mid \theta, m) = (1-\varepsilon) \cdot \text{softmax}(\beta \cdot U_i) + \varepsilon \cdot \frac{1}{|\text{branches}|}$$

参数：
- $\beta = 4.0$：理性温度。越高越接近 argmax，越低越随机
- $\varepsilon = 0.1$：lapse rate。10% 概率完全随机选择（"手抖"）

**直觉**：Agent 大部分时候选效用最高的分支，但偶尔会"犯错"。这模拟了人类决策中的噪声和探索行为。

---

## 4. 内化模块：Agent 的心理如何变化

这是整个系统最核心的部分。Agent 有 5 个内部心理状态变量，会随每次经历**持续更新**：

$$m_t = (\kappa_t,\; \tau_t,\; \nu_t,\; \gamma_t^{\text{spec}},\; \gamma_t^{\text{gen}})$$

### 4.1 各状态变量详解

#### κ (Risk Calibration) — 风险校准

$$\kappa_{t+1} = (1-\beta_\kappa) \cdot \kappa_t + \beta_\kappa \cdot \kappa_0 + \alpha_\kappa \cdot (\text{real\_risk} - \text{expected\_risk})$$

- 初始值：$\kappa_0 = 1.0$
- 遇到的风险比预期高 → κ 增加（变得更谨慎）
- 遇到的风险比预期低 → κ 减少（放松警惕）
- 有**均值回归**机制：即使不发生事件，κ 也会慢慢回到 $\kappa_0$

#### τ (Trust) — 对 Tutor 的信任

$$\tau_{t+1} = \begin{cases} \tau_t + 0.25 \cdot (1 - \tau_t) & \text{如果 Tutor 的 warning 确实帮了忙} \\ \tau_t - 0.12 \cdot \tau_t & \text{如果 Tutor 的 warning 是错的} \end{cases}$$

- 初始值：$\tau_0 = 0.3$（中等信任）
- 范围：$[0, 1]$
- 离上界越远，增加越大（指数趋近）

#### ν (Dependence) — 对 Tutor 的依赖 ⚠️

$$\nu_{t+1} = \begin{cases} \nu_t + 0.20 \cdot (1 - \nu_t) & \text{如果 Agent 盲从了（没有自己的证据就听话）} \\ \nu_t - 0.15 \cdot \nu_t & \text{如果 Agent 自己发现了正确路径} \end{cases}$$

- 初始值：$\nu_0 = 0.1$（低依赖）
- 范围：$[0, 0.8]$
- **这是 pedagogically 有害的状态**——ν 高意味着 Agent 失去了独立判断力
- **Tutor 的 WARN 会增加 ν**，这正是 Tutor 不能无限 WARN 的原因

#### γ_spec (Specific Suppression) — 特定诱惑抑制 ✅

$$\gamma^{\text{spec}}_{t+1} = \begin{cases} \gamma^{\text{spec}}_t + 0.22 \cdot (1 - \gamma^{\text{spec}}_t) & \text{如果 Agent 在高诱惑下犯了错} \\ \gamma^{\text{spec}}_t - 0.10 \cdot \gamma^{\text{spec}}_t & \text{如果发生了 false suppression} \end{cases}$$

- 初始值：$\gamma^{\text{spec}}_0 = 0.0$
- 范围：$[0, 0.7]$
- **这是好的**——学会抑制特定类型的诱惑
- 在效用函数中，$\gamma^{\text{spec}}$ 会降低 temptation 的吸引力

#### γ_gen (General Suppression) — 广泛探索抑制 ⚠️

$$\gamma^{\text{gen}}_{t+1} = \begin{cases} \gamma^{\text{gen}}_t + 0.08 \cdot (1 - \gamma^{\text{gen}}_t) & \text{如果 Tutor 持续施压（sustained\_pressure）} \\ \gamma^{\text{gen}}_t - 0.12 \cdot \gamma^{\text{gen}}_t & \text{如果 Agent 成功自主探索} \end{cases}$$

- 初始值：$\gamma^{\text{gen}}_0 = 0.0$
- 范围：$[0, 0.5]$
- **这是过度教学的标志**——Agent 不仅抑制了坏的探索，连好的探索也不敢了
- 在效用函数中，$\gamma^{\text{gen}}$ 会降低 novel 选项的吸引力

### 4.2 更新触发条件

| 事件 | 怎么触发 | 更新哪些变量 |
|------|---------|------------|
| Agent 走了高风险路径 | risk > threshold | κ ↑ |
| Tutor WARN 后 Agent 走了安全路径 | warn_helpful=True | τ ↑ |
| Tutor WARN 但 warning 是错的 | warn_bad=True | τ ↓ |
| Agent 没有自己的证据就听了 Tutor | blind_obey=True | **ν ↑** |
| Agent 在无帮助下自己找到正确路径 | self_discovery=True | **ν ↓** |
| Agent 被高诱惑的 risky 分支骗了 | tempt_error=True | γ_spec ↑ |
| Agent 因抑制而错过了 beneficial option | false_suppression=True | γ_spec ↓ |
| Tutor 持续干预（高 dose） | sustained_pressure=True | **γ_gen ↑** |
| Agent 成功自主探索了 novel 路径 | successful_exploration=True | **γ_gen ↓** |

### 4.3 核心矛盾

内化动力学揭示了教学中的**根本矛盾**：

```
         WARN 多了
        ╱          ╲
    τ ↑ (好)      ν ↑ (坏) + γ_gen ↑ (坏)
    信任建立        依赖增加    探索抑制

         WAIT 多了
        ╱          ╲
    ν ↓ (好)      Agent 可能失败 → 受伤/浪费时间
    自主性增强
```

**最优的教学策略不在两端，而是在中间**——这就是为什么 Tutor 需要精确地选择 WARN 的时机和剂量。

---

## 5. 行为探测（Behavior Probes）

Tutor 通过 5 个 "mastery probes" 来估计 Agent 当前的能力水平：

| Probe | 测什么 | 构造方法 | 主要依赖哪个 $m$ 变量 |
|-------|--------|---------|:------------------:|
| **RC** | 风险判断 | 给 safe (risk=0.05) vs risky (risk=0.4)，看选哪个 | κ |
| **TR** | 诱惑抵抗 | 给 safe (tempt=0.1) vs lure (tempt=0.8)，看选哪个 | γ_spec |
| **EP** | 探索保持 | 给 familiar vs novel_good (is_novel=True)，看选哪个 | γ_gen |
| **VA** | 正确建议采纳 | 给 advised (warn=+0.4) vs other (warn=−0.2)，看选哪个 | τ |
| **IA** | 错误建议拒绝 | 给 bad_advised (warn=+0.3) vs self_good，看选哪个 | ν |

每个 probe 就是**构造一个特定的 2-分支场景，用 Agent 的效用函数算出选择概率**。

例如 RC probe：

$$\text{RC} = P(\text{choose safe}) = (1-\varepsilon) \cdot \sigma\big(\beta \cdot (U_{\text{safe}} - U_{\text{risky}})\big) + \varepsilon \cdot 0.5$$

这些概率值在 $[0, 1]$ 之间，构成了 Tutor 对 Agent mastery 的估计 $u_t = (u^{RC}, u^{TR}, u^{EP}, u^{VA}, u^{IA})$。

---

## 6. Agent 的信息不对称

### Agent 知道但 Tutor 不知道

| 信息 | 说明 |
|------|------|
| 自己的 θ 类型 | Tutor 通过贝叶斯推断 |
| 自己在每条分支上的真实效用 | Tutor 通过预测 m → 行为映射间接推断 |

### Tutor 知道但 Agent 不知道

| 信息 | 说明 |
|------|------|
| 哪条分支是 oracle safe | Agent 只能通过学到的 cost/risk 模型猜测 |
| 真实的 cell cost/risk 值 | Agent 只在访问后才能知道 |
| 整个教学计划 | Agent 不知道 Tutor 的课程安排 |

### 当前实现的简化

| 理想 | 当前实现 |
|------|---------|
| Tutor 推断 Agent 的 $m_t$ | ⚠️ Tutor 直接读取 $m_t$ |
| Agent 通过原始 cell 特征做决策 | ✅ Agent 通过 branch summary 做决策 |

---

## 7. 完整决策流程

```
Step 1: Agent 到达 fork point
         ↓
Step 2: 感知两条分支的 cell 特征 z
         ↓
Step 3: LatentCostRiskHead 预测每个 cell 的 cost/risk
         ↓
Step 4: summarize_branch() → 得到 8D 分支总结
         ↓
Step 5: 提取 BranchAttributes (safety, temptation, risk_penalty)
         ↓
Step 6: 如果 Tutor 发了 WARN → 加入 warn_bonus
         ↓
Step 7: compute_factored_utility() → 用 m_t 和 θ 计算每条分支的 U
         ↓
Step 8: softmax + lapse → 概率性选择
         ↓
Step 9: 根据结果更新 m_t (trust, dependence, suppression...)
         ↓
Step 10: snapshot() 保存历史
```

---

## 8. 需要改进的地方

1. **感知模块过于简化**：Agent 当前直接看到 cell 特征向量，没有真正的"噪声观测" → 可以加入 `FeatureBeliefMap` 的 noisy observation
2. **选择模块是 one-shot 的**：Agent 不做多步规划，只在 fork point 做一次选择 → 和 proposal 里的 POMDP planning agent 有差距
3. **内化状态被 Tutor 偷看**：Tutor 应该通过行为推断 $m_t$，而不是直接读取
4. **缺少长期记忆**：Agent 的 cost/risk 模型每个 episode 重置 → 可以考虑跨 episode 保持
