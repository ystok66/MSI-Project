# Agent 机制与公式总览

> **参考文档** — 项目中 Learner Agent 系统的完整技术说明：决策模型、信念更新、内化状态、规划器、学习机制。
> 覆盖 V0→V2→V3 三代设计，标注当前 canonical 版本。
> 最后更新：2026-03-30。

---

## 1. 系统总览

Agent 系统包含五个核心子系统：

```
┌────────────────────────────────────────────────────────────────────┐
│  子系统 5: 内化状态 m_t = (κ, τ, ν, γ_spec, γ_gen)  [V3 canonical] │
│  FactoredInternalizationState — 5D 教学效果状态                     │
├────────────────────────────────────────────────────────────────────┤
│  子系统 4: Behavior Bridge — m_t → ẑ_probe 映射                    │
│  Semi-parametric logistic bridge + empirical zones                  │
├────────────────────────────────────────────────────────────────────┤
│  子系统 3: 信念系统 (三层)                                          │
│  L1: FeatureBeliefMap (4D Gaussian per cell)                        │
│  L2: BayesianRiskHead + BayesianCostHead (feature → risk/cost)      │
│  L3: BeliefMap (legacy per-cell cost/risk Gaussian)                  │
├────────────────────────────────────────────────────────────────────┤
│  子系统 2: 规划器                                                   │
│  Bounded-budget A* + necessity-aware cost function                   │
├────────────────────────────────────────────────────────────────────┤
│  子系统 1: 决策/选择模型                                            │
│  Bounded-rational softmax + lapse mixture                            │
│  U(π|θ,m) → P(a|s,θ)                                                │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. 决策模型 (stochastic_agent_policy.py)

### 2.1 基础 Utility 函数

$$U(\pi|\theta) = R_{goal}(\pi) + \lambda_\theta \cdot \langle \vec{w}_\theta, \vec{x}_\pi \rangle - J_{risk}(\pi)$$

其中：
- $R_{goal} = \text{safety\_score}$
- $\vec{x}_\pi = [\text{safety}, \text{temptation}, \text{novelty}, \text{shortcut}]$
- $J_{risk} = \text{risk\_penalty}$

**代码位置**：`stochastic_agent_policy.py:58-71`

### 2.2 偏好权重 $\vec{w}_\theta$

```python
PREF_REWARD = {
    "safe":     [1.0, -0.5,  0.0, 0.0],   # 偏好安全，回避诱惑
    "shiny":    [0.0,  1.0,  0.5, 0.0],   # 被诱惑吸引
    "risky":    [0.0,  0.5,  1.0, 0.0],   # 寻求新奇
    "shortcut": [0.0,  0.0,  0.0, 1.5],   # 追求捷径
    "neutral":  [0.3,  0.3,  0.3, 0.1],   # 无偏好
}
```

### 2.3 选择概率

$$P_{mix}(\pi|s,\theta) = (1-\epsilon) \cdot \text{softmax}(\beta \cdot U) + \epsilon \cdot \frac{1}{|\Pi|}$$

$$\text{softmax}_i = \frac{\exp(\beta \cdot U_i)}{\sum_j \exp(\beta \cdot U_j)}$$

| 参数 | 默认值 | 含义 |
|------|:------:|------|
| $\beta$ | 4.0 | softmax 温度（↑=更理性） |
| $\epsilon$ | 0.1 | lapse rate（随机探索概率） |
| $\lambda_\theta$ | 1.0 | 偏好权重缩放 |

**代码位置**：`stochastic_agent_policy.py:74-98`

---

## 3. 内化感知 Utility (internalization_state_v3.py) [V3 CANONICAL]

### 3.1 完整 Utility

$$U(\pi|\theta, m_t) = R_{pref}^{eff} - R_{risk} - R_{tempt} - R_{novel} + R_{warn} - R_{obey} + R_{epi}$$

$$R_{pref}^{eff} = \lambda_\theta \cdot \langle \vec{w}_\theta, \vec{x}_\pi \rangle$$

$$R_{risk} = \kappa^2 \cdot \text{risk\_penalty}$$

$$R_{tempt} = \gamma_{spec} \cdot \text{temptation\_score}$$

$$R_{novel} = \gamma_{gen} \cdot 0.3 \cdot \mathbf{1}[is\_novel]$$

$$R_{warn} = \tau \cdot \text{warn\_bonus}$$

$$R_{obey} = \nu \cdot 0.2 \cdot \mathbf{1}[\text{warn\_bonus} > 0]$$

**代码位置**：`internalization_state_v3.py:134-183`

### 3.2 各项语义

| 项 | 符号 | 效果 | 受控于 |
|----|------|------|--------|
| 偏好收益 | $R_{pref}^{eff}$ | $\theta$-specific branch 偏好 | $\lambda_\theta$ |
| 风险代价 | $R_{risk}$ | κ²放大：κ=1→1, κ=2→4 | $\kappa$ (risk sensitivity) |
| 诱惑抑制 | $R_{tempt}$ | 抑制诱惑分支的吸引力 | $\gamma_{spec}$ |
| 新奇抑制 | $R_{novel}$ | 抑制新奇探索 | $\gamma_{gen}$ |
| 警告信任 | $R_{warn}$ | tutor 警告的信号价值 | $\tau$ |
| 依赖代价 | $R_{obey}$ | 盲目服从的内在代价 | $\nu$ |
| 认知好奇 | $R_{epi}$ | 探索高不确定区域 | B2 flag (默认 OFF) |

### 3.3 B2 认知风险衰减（可选）

当 `use_epistemic_risk=True`：

$$\alpha = \alpha_{min} + (1-\alpha_{min}) \cdot \exp(-\gamma_{epi} \cdot \tilde{u}_r)$$

$$\tilde{u}_r = \text{clip}\left(\frac{u_r}{u_{ref}}, 0, 1\right)$$

$$R_{risk}^{B2} = \kappa^2 \cdot (\rho + (1-\rho)\alpha) \cdot \text{risk\_penalty}$$

| 参数 | 值 | 含义 |
|------|:--:|------|
| $\alpha_{min}$ | 0.25 | 最小衰减比 (risk floor) |
| $\rho$ | 0.35 | 基础风险比例 |
| $\gamma_{epi}$ | 3.0 | 不确定性衰减速率 |
| $u_{ref}$ | 0.5 | 参考不确定性 |

**代码位置**：`internalization_state_v3.py:159-166`

---

## 4. 内化状态 m_t (internalization_state_v3.py)

### 4.1 状态空间

$$m_t = (\kappa_t, \tau_t, \nu_t, \gamma_t^{spec}, \gamma_t^{gen})$$

| 维度 | 语义 | 初始值 | 范围 | ↑方向 |
|------|------|:------:|:----:|:-----:|
| $\kappa$ | 风险敏感度 | 1.0 | [0.3, 3.0] | 放大 risk |
| $\tau$ | Tutor 信任度 | 0.3 | [0.0, 1.0] | 好 |
| $\nu$ | Tutor 依赖度 | 0.1 | [0.0, 0.8] | 坏 |
| $\gamma_{spec}$ | 诱惑抗性 | 0.0 | [0.0, 0.7] | 好 |
| $\gamma_{gen}$ | 一般探索抑制 | 0.0 | [0.0, 0.5] | 坏 |

### 4.2 κ 更新（风险校准）

$$\kappa_{t+1} = (1-\beta_\kappa) \cdot \kappa_t + \beta_\kappa \cdot \kappa_0 + \alpha_\kappa \cdot (r_{real} - r_{expected})$$

$$\kappa_{t+1} = \text{clip}(\kappa_{t+1},\; \kappa_{min},\; \kappa_{max})$$

| 参数 | 值 |
|------|:--:|
| $\kappa_0$ (anchor) | 1.0 |
| $\beta_\kappa$ (reversion) | 0.08 |
| $\alpha_\kappa$ (learning rate) | 0.40 |
| $\kappa_{min}$ | 0.3 |
| $\kappa_{max}$ | 3.0 |

**代码位置**：`internalization_state_v3.py:71-77`

### 4.3 τ 更新（信任）

$$\tau_{t+1} = \tau_t + \begin{cases} \alpha_\tau^+ \cdot (1 - \tau_t) & \text{if warn\_helpful} \\ -\alpha_\tau^- \cdot \tau_t & \text{if warn\_bad} \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_\tau^+$ | 0.25 |
| $\alpha_\tau^-$ | 0.12 |

**代码位置**：`internalization_state_v3.py:79-84`

### 4.4 ν 更新（依赖）

$$\nu_{t+1} = \nu_t + \begin{cases} \alpha_\nu^+ \cdot (1 - \nu_t) & \text{if blind\_obey} \\ -\alpha_\nu^- \cdot \nu_t & \text{if self\_discovery} \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_\nu^+$ | 0.20 |
| $\alpha_\nu^-$ | 0.15 |
| $\nu_{max}$ | 0.8 |

**代码位置**：`internalization_state_v3.py:86-92`

### 4.5 γ_spec 更新（诱惑抗性）

$$\gamma_{spec}^{t+1} = \gamma_{spec}^t + \begin{cases} \alpha_{gs}^+ \cdot (1 - \gamma_{spec}^t) & \text{if tempt\_error} \\ -\alpha_{gs}^- \cdot \gamma_{spec}^t & \text{if false\_suppression} \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_{gs}^+$ | 0.22 |
| $\alpha_{gs}^-$ | 0.10 |
| $\gamma_{spec,max}$ | 0.7 |

**代码位置**：`internalization_state_v3.py:94-100`

### 4.6 γ_gen 更新（探索抑制）

$$\gamma_{gen}^{t+1} = \gamma_{gen}^t + \begin{cases} \alpha_{gg}^+ \cdot (1 - \gamma_{gen}^t) & \text{if sustained\_pressure} \\ -\alpha_{gg}^- \cdot \gamma_{gen}^t & \text{if successful\_exploration} \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_{gg}^+$ | 0.08 |
| $\alpha_{gg}^-$ | 0.12 |
| $\gamma_{gen,max}$ | 0.5 |

**代码位置**：`internalization_state_v3.py:102-108`

---

## 5. V0 内化状态 (internalization_agent.py) [LEGACY]

3D 简化版：$m_t = (\kappa, \eta, \gamma)$

$$U(\pi|\theta,m_t) = (1-\gamma)\lambda_\theta R_{pref} + \eta \cdot B_{warn} - \kappa^2 \cdot J_{risk}$$

$$\kappa_{t+1} = \text{clip}(\kappa_t + \alpha_\kappa \cdot (r_{real} - r_{expected}),\; \kappa_{min},\; \kappa_{max})$$

$$\eta_{t+1} = \eta_t + \alpha_\eta \cdot (z_t - \eta_t), \quad z_t = \mathbf{1}[\text{warn matched truth}]$$

$$\gamma_{t+1} = \text{clip}(\gamma_t + \alpha_\gamma \cdot \mathbf{1}[\text{tempt error}],\; 0,\; \gamma_{max})$$

| 参数 | V0 值 | V3 对应 |
|------|:-----:|---------|
| $\alpha_\kappa$ | 0.50 | 0.40 (+ reversion) |
| $\alpha_\eta$ | 0.35 | → α_τ+/α_τ- 分离 |
| $\alpha_\gamma$ | 0.30 | → α_gs+/α_gs- 分离 |
| $\kappa_0$ | — | 1.0 (V3 added) |
| $\nu$ | — | 0.1 (V3 新增) |
| $\gamma_{gen}$ | — | 0.0 (V3 新增) |

**代码位置**：`internalization_agent.py:28-82`

---

## 6. 信念系统

### 6.1 Level 1: FeatureBeliefMap (feature_belief.py) [V2 CANONICAL]

每个 cell 独立高斯信念，4D 特征空间：

$$b_{r,c,d} \sim \mathcal{N}(\mu_{r,c,d},\; \sigma^2_{r,c,d})$$

**先验**：$\mu_0 = 0.5$，$\sigma^2_0 = 0.25$（uninformative）

**Kalman 更新**（观测 cell feature）：

$$K_d = \frac{\sigma^2_{prior,d}}{\sigma^2_{prior,d} + \sigma^2_{obs}}$$

$$\mu_{d}^{post} = \mu_d^{prior} + K_d \cdot (z_{obs,d} - \mu_d^{prior})$$

$$\sigma^{2,post}_d = \sigma^2_{prior,d} \cdot (1 - K_d)$$

**观测噪声模型**（3-tier 离散）：

| 距离 | $\sigma^2_{obs}$ | 含义 |
|:----:|:----------------:|------|
| 0 (self) | 0.01 | 近乎精确 |
| 1 (neighbor) | 0.08 | 信息量但模糊 |
| 2+ (far) | 0.20 | 远距离粗观 |

**代码位置**：`feature_belief.py:109-124`

### 6.2 Cell 溯源 (CellMemoryMeta)

每个 cell 携带观测历史元数据：

```python
@dataclass
class CellMemoryMeta:
    ever_seen: bool = False       # 是否被观测过
    seen_count: int = 0           # 观测次数
    ever_traversed: bool = False  # 是否被 agent 踩过
    traversed_count: int = 0     
    last_seen_t: int = -1        
    best_view_quality: float = 0.0  # 0=unseen, 0.5=neighbor, 1.0=self
    reachable_since_t: int = -1  # UNLOCK 使之可达的时间
    intervention_tags: set        # {"warned", "unlocked", "item_affected"}
```

### 6.3 干预条件更新

#### WARN 对 FeatureBeliefMap 的效果

$$z_{pseudo} = \mu_{r,c} + \alpha_{warn} \cdot \hat{v}_{warn}$$

$$\sigma^2_{pseudo} = \sigma^2_{r,c} \cdot \text{warn\_confidence}$$

然后做标准 Kalman 更新（pseudo-observation）

| 参数 | 默认值 |
|------|:------:|
| warn_strength | 0.15 |
| warn_confidence | 2.0 |

**代码位置**：`feature_belief.py:157-183`

#### UNLOCK 对 FeatureBeliefMap 的效果

$$\sigma^{2,post}_{r,c} = (1 - \beta_{unlock}) \cdot \sigma^{2}_{r,c}$$

不改变均值。仅降低不确定性。

| 参数 | 默认值 |
|------|:------:|
| $\beta_{unlock}$ | 0.5 |

**代码位置**：`feature_belief.py:143-155`

### 6.4 Level 2: BayesianRiskHead (risk_model.py)

共享线性模型，从 feature vector 预测 risk：

$$\hat\rho = \sigma(w \cdot x + b), \quad \sigma(\cdot) = \text{sigmoid}$$

**在线 MAP 更新**（SGD + L2 prior）：

$$\nabla_w \text{NLL} = -(y - \hat\rho) \cdot x + \frac{w}{\sigma^2_{prior}}$$

$$w \leftarrow w - \eta \cdot \text{clip}(\nabla_w, \|\cdot\| \leq 5)$$

$$\|w\| \leq 10 \quad\text{(norm clamping)}$$

**不确定性估计**（Laplace 近似）：

$$H = \frac{X^T X}{n} + \frac{I}{\sigma^2_{prior}}$$

$$\text{unc}(x) = \hat\rho(1-\hat\rho)(1 + x^T H^{-1} x)$$

| 参数 | 默认值 |
|------|:------:|
| $\eta$ (learning rate) | 0.3 |
| $\sigma^2_{prior}$ | 1.0 |
| $w_0, b_0$ | 0, 0 |

**代码位置**：`risk_model.py:22-127`

### 6.5 Level 2: BayesianCostHead (cost_risk_model.py)

共享线性模型，从 feature vector 预测 cost：

$$\hat{c} = \max(w_c \cdot x + b_c,\; 0.1)$$

更新逻辑同 RiskHead。

| 参数 | 默认值 |
|------|:------:|
| $\eta$ (learning rate) | 0.1 |
| $b_0$ | 1.0 (normal cell prior) |

**代码位置**：`cost_risk_model.py:32-112`

### 6.6 LatentCostRiskHead (cost_risk_model.py)

组合 BayesianCostHead + BayesianRiskHead：

```python
predict_cost(x) → float          # ĉ = w_c·x + b_c
predict_risk(x) → float          # ρ̂ = σ(w_r·x + b_r)
predict_cost_uncertainty(x) → float
predict_risk_uncertainty(x) → float
```

**方向性不确定性**（Phase 10，从 posterior variance 计算）：

$$u_c = w_c^T \cdot \text{diag}(\sigma^2_{x}) \cdot w_c$$

$$u_r = w_r^T \cdot \text{diag}(\sigma^2_{x}) \cdot w_r$$

**代码位置**：`cost_risk_model.py:157-237`

### 6.7 WorldWeights（环境真值生成）

$$c_{true}(z) = \max(w_{cost} \cdot z + b_{cost},\; 0.1)$$

$$\rho_{true}(z) = \sigma(w_{risk} \cdot z + b_{risk})$$

Risk weight 设计：texture dims (2,3) 是主要驱动，其他弱影响。

| 维度 | $w_{risk}$ 范围 | 含义 |
|:----:|:--------------:|------|
| 0 (lane_id) | [-0.5, 0.5] | 弱 |
| 1 (gate_flag) | [-0.3, 0.3] | 弱 |
| 2 (texture_1) | **[2.0, 4.0]** | **强正** |
| 3 (texture_2) | **[1.5, 3.5]** | **强正** |
| bias | [-3.0, -1.5] | 负偏置（多数 cell 低风险） |

**代码位置**：`cost_risk_model.py:114-154`

### 6.8 Level 3: BeliefMap (belief.py) [V0 LEGACY]

每 cell 独立高斯: $b_{cost}(r,c) \sim \mathcal{N}(\mu_c, \sigma^2_c)$，$b_{risk}(r,c) \sim \mathcal{N}(\mu_r, \sigma^2_r)$

**Kalman 更新**：

$$\sigma^{2,post} = \frac{1}{1/\sigma^2_{prior} + 1/\sigma^2_{obs}}$$

$$\mu^{post} = \sigma^{2,post} \cdot \left(\frac{\mu_{prior}}{\sigma^2_{prior}} + \frac{z_{obs}}{\sigma^2_{obs}}\right)$$

**默认先验**：

| 变量 | $\mu_0$ | $\sigma^2_0$ |
|------|:-------:|:----------:|
| cost | 1.5 | 4.0 |
| risk | 0.1 | 0.25 |

**代码位置**：`belief.py:119-137`

### 6.9 RSA 警告更新

精度加权融合（precision-weighted fusion）：

$$p_{old} = 1/\sigma^2_{prior}, \quad p_{obs} = \frac{\text{sensitivity}}{\sigma^2_{eff}}$$

$$p_{new} = p_{old} + p_{obs}$$

$$\mu^{post} = \frac{1}{p_{new}}\Big(p_{old} \cdot \mu_{prior} + p_{obs} \cdot y_{pseudo}\Big)$$

$$\sigma^{2,post} = 1/p_{new}$$

Risky 话语 → $y_{pseudo} = 1.0$；Safe 话语 → $y_{pseudo} = 0.0$。

**代码位置**：`belief.py:240-281`

---

## 7. 规划器 (planner_astar.py)

### 7.1 Bounded-budget A*

共享 A* 引擎，expand 最多 `budget` 个节点：

$$f(n) = g(n) + h(n)$$

$$g(n) = \text{从 start 到 n 的累计 cost}$$

$$h(n) = |n_{row} - goal_{row}| + |n_{col} - goal_{col}| \quad\text{(Manhattan)}$$

Budget exhausted → 返回最接近 goal 的 partial path。

### 7.2 V0 Cell Cost（标量信念）

$$J(i) = \mu_c(i) + \lambda_r \cdot \varphi(\mu_\rho(i)) + \lambda_u \cdot \sigma^2_c(i)$$

$$\varphi(\mu_\rho) = -\ln(1 - \text{clip}(\mu_\rho, \epsilon, 1-\epsilon))$$

| 参数 | 默认值 |
|------|:------:|
| $\lambda_r$ | 3.0 |
| $\lambda_u$ | 0.5 |

**代码位置**：`planner_astar.py:160-189`

### 7.3 V2 Latent Cell Cost（特征信念，CANONICAL）

$$J(i) = \lambda_c \cdot \hat{c}_i + \text{risk\_penalty}_i + \lambda_{uc} \cdot (1-n) \cdot u_c^i + \lambda_{ur} \cdot (1-n) \cdot u_r^i$$

$$\text{risk\_penalty}_i = \lambda_r \cdot \varphi(\hat{\rho}_i) \cdot \Big(f_{learn} + (1 - f_{learn}) \cdot (1-n)\Big)$$

$$f_{learn} = \min\Big(1, \frac{n_{updates}}{10}\Big)$$

$$\varphi(\hat\rho) = -\ln(1 - \text{clip}(\hat\rho, \epsilon, 1-\epsilon))$$

$$\hat{c}_i = \text{LatentPredictor.predict\_cost}(x_{belief}^i)$$

$$\hat\rho_i = \text{LatentPredictor.predict\_risk}(x_{belief}^i)$$

**核心设计原则："Unknown ≠ Dangerous"**

- 当 $n$ (route necessity) 高（无安全替代路径）→ uncertainty penalty 趋零
- 当 $f_{learn}$ 低（模型未训练, $w=0, b=0 \Rightarrow \sigma(0)=0.5$）→ risk penalty 按 necessity 折扣

**Shield 效果**：

$$\text{risk\_penalty}^{shield} = \text{risk\_penalty} \cdot (1 - \gamma_{shield})$$

| 参数 | 默认值 | 含义 |
|------|:------:|------|
| $\lambda_c$ | 1.0 | cost 权重 |
| $\lambda_r$ | 5.0 | risk 权重 |
| $\lambda_{uc}$ | 0.1 | cost 不确定性权重 |
| $\lambda_{ur}$ | 0.1 | risk 不确定性权重 |

**代码位置**：`planner_astar.py:299-387`

### 7.4 搜索预算采样

$$B \sim \text{NegBin}_{discrete}(\text{class})$$

| budget_class | 候选 | 概率 |
|:------------:|:----:|:----:|
| 4 | [3, 4, 5] | [0.25, 0.50, 0.25] |
| 8 | [6, 8, 12] | [0.25, 0.50, 0.25] |
| 16 | [14, 16, 20] | [0.25, 0.50, 0.25] |

**代码位置**：`planner_astar.py:32-52`

---

## 8. 观测模型 (observation_model.py)

### 8.1 Feature 观测（V2 CANONICAL）

Agent 观测 **feature vector**（NOT risk scalars）：

$$z^{obs}_{i,d} = \text{clip}(z^{true}_{i,d} + \epsilon_d, 0, 1), \quad \epsilon_d \sim \mathcal{N}(0, \sigma^2_{obs})$$

**3-tier 噪声**：

| 距离 | $\sigma^2_{obs}$ |
|:----:|:----------------:|
| 0 (self) | 0.01 |
| 1 (neighbor) | 0.08 |
| 2+ (far) | 0.20 |

Wall cells 不产生观测（跳过）。

**代码位置**：`observation_model.py:100-153`

### 8.2 Cost/Risk 观测（V0 LEGACY）

$$c^{obs}_i = \max(0, c^{true}_i + \epsilon_c), \quad \epsilon_c \sim \mathcal{N}(0, \sigma^2_{obs})$$

$$\rho^{obs}_i = \text{clip}(\rho^{true}_i + \epsilon_r, 0, 1), \quad \epsilon_r \sim \mathcal{N}(0, \sigma^2_{obs})$$

**代码位置**：`observation_model.py:28-87`

---

## 9. Warning 处理 (warning_update.py)

### 9.1 双机制警告系统

#### 机制 1：Pseudo-label 注入 RiskHead

$$\alpha_j(u) = \exp\left(-\frac{\|x_{hat}^j - p_u\|^2}{\tau}\right)$$

$$\text{effective\_weight}_j = w \cdot \alpha_j$$

$$\text{RiskHead.update}(x_{hat}^j, y_{label}^u, \text{weight}=\text{effective\_weight}_j)$$

| Utterance | Prototype $p_u$ | y_label |
|-----------|:---------------:|:-------:|
| RISKY_TEXTURE_AHEAD | [0.5, 0.0, 0.85, 0.80] | 0.8 |
| UPPER_LANE_RISKY | [0.0, 0.0, 0.70, 0.60] | 0.7 |
| SAFE_DETOUR_OPEN | [1.0, 0.0, 0.05, 0.05] | 0.0 |

#### 机制 2：Lane-level Warning Bias

$$b_{warn}(\text{lane}) = \sum_j \alpha_j(u) \cdot y_u$$

$$\text{planner\_bias} = \lambda_{lane} \cdot b_{warn}$$

添加到被警告 lane 的总代价中。

| 参数 | 默认值 |
|------|:------:|
| $\tau$ | 0.3 |
| $\lambda_{lane}$ | 5.0 |
| $w$ (pseudo-label weight) | 5.0 |

### 9.2 Utterance 选择（Action-Gap）

$$u^* = \arg\max_u \lambda_{lane} \cdot b_{warn}^u$$

选择使 risky lane 代价增加最大的 utterance。

**代码位置**：`warning_update.py:138-165`

---

## 10. Behavior Bridge (behavior_bridge.py)

### 10.1 Semi-Parametric Bridge

$$\hat{z}_p(m,c) = \sigma\big(\vec{w}_p \cdot \phi(m,c)\big)$$

**特征向量 $\phi$**（13 维）：

$$\phi(m,c) = [1,\; \kappa,\; \tau,\; \nu,\; \gamma_s,\; \gamma_g,\; \kappa \cdot r,\; \gamma_s \cdot l,\; \gamma_g \cdot n,\; \tau(1-se),\; \nu \cdot se,\; \kappa^2,\; \gamma_g^2]$$

### 10.2 预训练权重

| Probe | $w_0$ ($b$) | $\kappa$ | $\tau$ | $\nu$ | $\gamma_s$ | $\gamma_g$ | $\kappa r$ | $\gamma_s l$ | $\gamma_g n$ | $\tau(1-se)$ | $\nu \cdot se$ | $\kappa^2$ | $\gamma_g^2$ |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| RC | 1.8 | 0.9 | 0 | 0 | 0 | -0.3 | 1.2 | 0 | 0 | 0 | 0 | 0 | 0 |
| TR | 1.5 | 0.3 | 0 | 0 | 1.2 | -0.2 | 0 | 1.5 | 0 | 0 | 0 | 0 | 0 |
| EP | 0.3 | -0.2 | 0 | -0.3 | 0 | **-2.5** | 0 | 0 | -1.8 | 0 | 0 | 0 | -1.0 |
| VA | 0.5 | 0 | 1.8 | -1.5 | 0 | 0 | 0 | 0 | 0 | 2.0 | -1.5 | 0 | 0 |
| IA | -1.5 | 0 | -0.3 | **2.5** | 0 | 0.5 | 0 | 0 | 0 | -0.5 | 2.0 | 0 | 0 |

### 10.3 行为损失

$$L_{beh}(m) = \sum_p w_p \cdot \max(lo_p - \hat{z}_p, 0)^2 + \max(\hat{z}_p - hi_p, 0)^2$$

| Probe | weight | 语义 |
|-------|:------:|------|
| RC (Risk Caution) | 1.0 | 风险谨慎程度 |
| TR (Temptation Resist) | 1.2 | 诱惑抗性 |
| EP (Exploration Preserve) | **2.5** | 探索保留（最高权重） |
| VA (Valid Advice) | 1.5 | 有效建议接受度 |
| IA (Invalid Advice) | **2.5** | 无效建议拒绝度 |

### 10.4 过度教学惩罚

$$R_{over} = 2.5 \cdot \max(\hat{z}_{IA} - hi_{IA}, 0)^2 + 2.5 \cdot \max(lo_{EP} - \hat{z}_{EP}, 0)^2 + 1.5 \cdot \max(\hat{z}_{TR} - hi_{TR}, 0)^2$$

**代码位置**：`behavior_bridge.py:96-109`

---

## 11. Agent 信念状态封装

### 11.1 AgentBelief (agent_belief_state.py)

$$b_t^A = (b_{env}, m_t, \theta, h_t)$$

```python
@dataclass
class AgentBelief:
    belief_mean: np.ndarray      # (H, W, d) feature belief
    belief_var: np.ndarray       # (H, W, d) feature variance
    predicted_cost: np.ndarray   # (H, W) from latent predictor
    predicted_risk: np.ndarray   # (H, W) from latent predictor
    observed_mask: np.ndarray    # (H, W) bool
    m_state: dict                # {κ, τ, ν, γ_spec, γ_gen}
    theta: str                   # preference type
    n_steps_taken: int
    n_warnings_received: int
    n_self_discoveries: int
```

### 11.2 WorldState (world_state.py)

$$s_t^{world} = (x_{agent}, G, D, I, T, \Phi_{cost}, \Phi_{risk}, g)$$

```python
@dataclass(frozen=True)
class WorldState:
    agent_pos: tuple[int, int]
    cell_types: np.ndarray     # (H, W)
    passable: np.ndarray       # (H, W) bool
    door_positions: tuple
    doors_unlocked: tuple
    shield_available: bool
    t: int                      # 当前时间步
    t_max: int                  # 总时间预算
    true_cost: np.ndarray      # (H, W) oracle
    true_risk: np.ndarray      # (H, W) oracle
    goal_pos: tuple[int, int]
```

**核心不变量**：Agent 永远不直接读取 WorldState。所有信息通过 observation model 间接观测。

---

## 12. Agent 循环总结

```
每步 t:
  1. observe_features(pos, true_features, noise)
     → 生成 FeatureObservation (pos → noisy z)
  
  2. FeatureBeliefMap.update(pos, obs_mean, obs_var)
     → Kalman 更新 belief(pos) ← obs
  
  3. LatentCostRiskHead.update_from_outcome(z, cost, risk)
     → SGD 更新 w_cost, w_risk (如果有 outcome)
  
  4. process_teacher_action(warning)
     → 如果 WARN: FeatureBeliefMap.apply_warn_update + RiskHead pseudo-label
     → 如果 UNLOCK: FeatureBeliefMap.apply_unlock_update
     → 置 plan_invalidated = True
  
  5. plan_next_action_v2(pos, goal, feature_belief, latent_predictor, ...)
     → cell_cost_v2_latent → bounded A* → path → next action
     → 如果 plan_invalidated → 重新搜索
  
  6. execute(action)
     → 移动到 new_pos，处理 risk event / item pickup
     → m_t.update_risk/trust/dependence/gamma_spec/gamma_gen
  
  7. m_t.snapshot()
     → 记录 5D 内化状态历史
```

---

## 13. 完整模块清单

| 文件 | 大小 | 当前状态 | 核心公式 |
|------|:----:|:--------:|---------|
| `stochastic_agent_policy.py` | 3KB | ✅ Canonical | §2: $U(\pi|\theta)$, $P_{mix}$ |
| `internalization_state_v3.py` | 9KB | ✅ **V3 Canonical** | §3-4: $U(m_t)$, 5D update |
| `internalization_agent.py` | 6KB | ⚠️ V0 Legacy | §5: 3D $(κ,η,γ)$ |
| `feature_belief.py` | 8KB | ✅ Canonical | §6.1: 4D Kalman, provenance |
| `risk_model.py` | 4KB | ✅ Canonical | §6.4: Bayesian logistic |
| `cost_risk_model.py` | 8KB | ✅ Canonical | §6.5-6.7: LatentCostRiskHead |
| `belief.py` | 10KB | ⚠️ V0 Legacy | §6.8: scalar Kalman |
| `planner_astar.py` | 18KB | ✅ Canonical | §7: bounded A*, cell_cost |
| `observation_model.py` | 7KB | ✅ Canonical (V2 part) | §8: feature observation |
| `warning_update.py` | 8KB | ✅ Canonical | §9: pseudo-label + lane bias |
| `behavior_bridge.py` | 5KB | ✅ Canonical | §10: $m \to \hat{z}$ bridge |
| `agent_belief_state.py` | 3KB | ✅ POMDP shell | §11.1: $b_t^A$ |
| `world_state.py` | 3KB | ✅ POMDP shell | §11.2: $s_t^{world}$ |
| `bounded_agent.py` | 11KB | ⚠️ V0 Deprecated | V0 agent class |
| `behavior_probes.py` | 7KB | ✅ Canonical | behavior zones |
| `branch_summary.py` | 4KB | ✅ Canonical | branch feature summary |
| `branch_concepts.py` | 5KB | ✅ Canonical | concept-level branch features |
| `familiarity.py` | 3KB | ✅ Canonical | cell familiarity tracking |
| `route_necessity.py` | 4KB | ✅ Canonical | route necessity $n \in [0,1]$ |
| `internalization_dynamics_v2.py` | 4KB | ⚠️ V2 Bridge | V2 → V3 transition |
| `mixed_effects_risk_head.py` | 7KB | ✅ Research | per-learner mixed effects |
| `trainable_bridge.py` | 8KB | ✅ Research | learnable $m \to z$ |
| `prefix_prediction.py` | 3KB | ✅ Canonical | action prefix prediction |
| `pragmatic_warning.py` | 2KB | ✅ Canonical | pragmatic listener |

### 版本沿革

| 代 | 内化维度 | Utility 公式 | 信念系统 | 规划器 |
|----|:--------:|:----------:|:--------:|:------:|
| V0 | 3D $(κ,η,γ)$ | $(1-γ)\lambda R + ηB - κ^2J$ | scalar BeliefMap | cell_cost V0 |
| V2 | 3D → 5D | + $\gamma_{spec}$, $\gamma_{gen}$ | FeatureBeliefMap 4D | cell_cost_v2 |
| **V3** | **5D** | + $R_{tempt} + R_{novel} + R_{obey}$ | + LatentCostRiskHead | **cell_cost_v2_latent** |
