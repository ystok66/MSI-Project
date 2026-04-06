# Pedagogical Gridworld: Agent Learner System Design Report
## Complete Specification of Perception, Learning, Planning, and Decision Making

---

## 1. 系统架构总览

Agent 由以下子系统组成，每步按固定顺序执行：

```
Observe → Learn → Plan → Move → Outcome feedback
```

```mermaid
graph TB
    subgraph "1. Perception"
        OBS[Observation Model<br>带噪声的 4D 特征观测]
        FB[FeatureBeliefMap<br>Kalman 滤波后验]
    end
    
    subgraph "2. World Model (Learning)"
        CRH[LatentCostRiskHead<br>4D→cost + 4D→risk]
        BRH[BayesianRiskHead<br>MAP 线性模型]
        BCH[BayesianCostHead<br>MAP 线性模型]
    end
    
    subgraph "3. Planning"
        ASTAR[Bounded A*<br>budget=30 nodes]
        CCF[Cell Cost Function<br>J = λ_c·ĉ + λ_r·φ(r̂) + ...]
        RN[Route Necessity<br>Unknown ≠ Dangerous]
        BP[BeliefPlan<br>结构化规划结果]
    end
    
    subgraph "4. Warning Processing"
        WU[Warning Update<br>pseudo-label + lane bias]
        RSA[RSA Channel<br>L0 / S1 belief update]
    end
    
    OBS --> FB
    FB --> CRH
    FB --> ASTAR
    CRH --> CCF
    RN --> CCF
    CCF --> ASTAR
    ASTAR --> BP
    WU --> FB
    WU --> BRH
    RSA --> FB
```

---

## 2. 感知系统 (Perception)

### 2.1 特征观测模型

定义于 [observation_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/observation_model.py)。

每步 agent 对当前位置及邻居观测 4D 特征向量 $z \in \mathbb{R}^4$，加入高斯噪声：

$$z_{\text{obs}} = \text{clip}(z_{\text{true}} + \mathcal{N}(0, \sigma^2_{\text{obs}} \cdot I_d), \; 0, \; 1)$$

#### 3 层噪声模型（Phase 5 extended patch）

| 距离 (Manhattan) | 噪声 $\sigma^2_{\text{obs}}$ | 含义 |
|:---:|:---:|------|
| 0 (自身) | 0.01 | 近乎精确 |
| 1 (邻居) | 0.08 | 模糊但有信息 |
| 2+ (远处) | 0.20 | 很模糊 |

**观测半径** `patch_radius`：
- `patch_radius=1`（默认）：自身 + 4 个邻居
- `patch_radius=2`：自身 + 4 邻居 + 8 远处 = 最多 13 个 cell
- 墙壁 cell 跳过不观测

### 2.2 FeatureBeliefMap — Kalman 后验

定义于 [feature_belief.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/feature_belief.py)。

每个 cell 的每个特征维度独立维护高斯信念 $(\mu, \sigma^2)$。

#### 先验

$$\mu_0 = 0.5, \quad \sigma^2_0 = 0.25 \quad \text{(uninformative)}$$

#### Kalman 更新

$$K = \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{prior}} + \sigma^2_{\text{obs}}}$$

$$\mu_{\text{post}} = \mu_{\text{prior}} + K \cdot (z_{\text{obs}} - \mu_{\text{prior}})$$

$$\sigma^2_{\text{post}} = \sigma^2_{\text{prior}} \cdot (1 - K)$$

#### 关键性质

- **每维独立更新**：4D 向量的各维度分别做 scalar Kalman 更新
- **多次观测递减**：第一次观测（距离 0）后 $\sigma^2 \approx 0.01$，几乎确定；反复观测仅微弱减少
- **Latent predictor 消费的是 belief mean $\mu$**，不是真实 $z$

#### Cell 级元数据 (CellMemoryMeta)

每个 cell 还维护 provenance 信息：

```python
@dataclass
class CellMemoryMeta:
    ever_seen: bool = False         # 是否被观测过
    seen_count: int = 0             # 观测次数
    ever_traversed: bool = False    # 是否被实际走过
    traversed_count: int = 0
    last_seen_t: int = -1           # 最后一次被观测的 timestep
    best_view_quality: float = 0.0  # 0=unseen, 0.5=neighbor, 1.0=self
    reachable_since_t: int = -1     # UNLOCK 后可达时间
    intervention_tags: set          # {"warned", "unlocked", "item_affected"}
```

#### 干预条件下的特殊更新

1. **UNLOCK update**: 降低新开路径的 uncertainty（不改变 mean）

$$\sigma^2_{\text{post}} = \sigma^2_{\text{prior}} \times (1 - \beta_{\text{unlock}}) \quad (\beta_{\text{unlock}} = 0.5)$$

2. **WARN update**: 沿警告方向推动 belief mean

$$\mu_{\text{post}} = \mu_{\text{prior}} + w_{\text{str}} \cdot d_{\text{warn}} \cdot \sigma^2_{\text{obs\_equiv}}$$

其中 $d_{\text{warn}}$ 是警告方向向量，$w_{\text{str}}$ 是 warn_strength。

---

## 3. 学习系统 (World Model)

### 3.1 LatentCostRiskHead — 联合 cost/risk 学习器

定义于 [cost_risk_model.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/cost_risk_model.py)。

组合两个独立的贝叶斯线性头：

```python
class LatentCostRiskHead:
    cost_head: BayesianCostHead    # 4D → cost (Gaussian likelihood)
    risk_head: BayesianRiskHead    # 4D → risk (Bernoulli likelihood)
```

**参数总数**: $2 \times (4 + 1) = 10$（两组 4 维权重 + 1 偏置）

### 3.2 BayesianCostHead

#### 预测

$$\hat{c}(z) = \max(w_c \cdot z + b_c, \; 0.1)$$

- **先验**: $w_c = \mathbf{0}$, $b_c = 1.0$（初始预测所有 cell cost ≈ 1）
- **prior_var**: 1.0（$L_2$ prior 标准差 = 1）

#### 在线 MAP 更新

$$\nabla_w \mathcal{L}_{\text{cost}} = -(c_{\text{true}} - \hat{c}) \cdot z \cdot w_{\text{sample}} + \frac{w_c}{\sigma^2_{\text{prior}}}$$

$$w_c \leftarrow w_c - \eta_c \cdot \text{clip}(\nabla_w, \|\cdot\| \leq 5.0)$$

$$b_c \leftarrow b_c - \eta_c \cdot \text{clip}(\nabla_b, ||\leq 5.0)$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\eta_c$ | 0.1 | cost 学习率 |
| `prior_var` | 1.0 | L2 正则化强度 |
| `max_grad_norm` | 5.0 | 梯度裁剪阈值 |
| `max_w_norm` | 10.0 | 权重范数上限 |

#### 不确定性估计 (Laplace 近似)

$$\text{CostVar}(z) \approx z^T \left(\frac{X^TX}{n} + \frac{I}{\sigma^2_{\text{prior}}}\right)^{-1} z$$

未训练时 ($n < 2$) 返回 1.0（高不确定性）。

### 3.3 BayesianRiskHead

#### 预测

$$\hat{\rho}(z) = \sigma(w_r \cdot z + b_r)$$

$\sigma(x) = \frac{1}{1 + e^{-x}}$（数值稳定 sigmoid）

- **先验**: $w_r = \mathbf{0}$, $b_r = 0.0$
- 初始预测: $\hat{\rho} = \sigma(0) = 0.5$（最大不确定性）

#### 在线 MAP 更新

$$\nabla_w \mathcal{L}_{\text{risk}} = -(\rho_{\text{label}} - \hat{\rho}) \cdot z \cdot w_{\text{sample}} + \frac{w_r}{\sigma^2_{\text{prior}}}$$

$$w_r \leftarrow w_r - \eta_r \cdot \text{clip}(\nabla_w, \|\cdot\| \leq 5.0)$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\eta_r$ | 0.3 | risk 学习率（比 cost 高 3×） |
| `prior_var` | 1.0 | L2 正则化强度 |

#### 不确定性估计 (Laplace 近似)

$$\text{RiskVar}(z) \approx \hat{\rho}(1-\hat{\rho}) \cdot (1 + z^T H^{-1} z)$$

其中 $H \approx \frac{X^TX}{n} + \frac{I}{\sigma^2_{\text{prior}}}$

未训练时 ($n < 2$) 返回 0.25（均匀分布方差）

### 3.4 Directional Uncertainty（Phase 10）

当 FeatureBeliefMap 的 posterior variance 可用时，使用**方向性**不确定性估计替代 Hessian-based 估计：

$$u_c(z) = w_c^T \cdot \text{diag}(\sigma^2_{\text{belief}}) \cdot w_c$$

$$u_r(z) = w_r^T \cdot \text{diag}(\sigma^2_{\text{belief}}) \cdot w_r$$

直觉：如果 belief variance 在 texture 维度很高($\sigma^2_{z_2}, \sigma^2_{z_3}$ 大)，且 risk 权重在这些维度也高($w_{r[2]}, w_{r[3]}$ 大)，则 risk 不确定性高。

### 3.5 学习数据来源与权重

Agent 在每步移动后从**真实环境**获取反馈，用于更新 LatentCostRiskHead：

| 事件 | cost_label | risk_label | weight | 含义 |
|------|:---:|:---:|:---:|------|
| 进入 RISKY cell 并死亡 | `true_cost` | 1.0 | **4.0** | 强正样本 |
| 进入 RISKY cell 并存活 | `true_cost` | `true_risk` 或 0.0 | 1.5 | 中等正样本 |
| 进入非 RISKY cell | `true_cost` | 0.0 | 0.1 | 弱负样本 |

**Risk supervision 模式**：
- `oracle_visited`: 存活后使用 true risk 值（$y = \rho_{\text{true}}$）— 直接监督
- `binary_outcome`: 存活后使用 0.0（$y = 0$）— 仅从死亡事件学习

**关键设计**：学习使用的 $z$ 是 **belief mean**（$\mu_{\text{belief}}$），不是 true features。这确保 agent 学到的映射是"观测到的特征→risk"，与规划时使用的特征一致。

### 3.6 学习动力学分析

> **诊断结论**: 10 个参数的线性模型在约 10–50 个样本后就已收敛，单个 episode 提供足够的训练数据。这使得跨 episode 的先验知识在新 episode 开始时迅速被覆盖——这是 PRS-2 transfer failure 的根因。

收敛速度估算：

- Episode 典型步数: 15–60
- 每步 1 个训练样本
- 4D 线性模型 effective rank ≈ 4
- $\eta_r = 0.3$ + 无大 prior decay → 约 10 步 risk head 已从先验偏离
- 梯度裁剪 + 权重裁剪防止发散但不阻止快速适应

---

## 4. 规划系统 (Planning)

### 4.1 Bounded A* 搜索

定义于 [planner_astar.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/planner_astar.py)。

```python
_astar_core(start, goal, cost_fn, H, W, budget=30, passable_mask=None)
```

- **预算限制**: 最多展开 `budget` 个节点（默认 30）
- **启发函数**: Manhattan distance $h(n) = |r_n - r_g| + |c_n - c_g|$
- **动作空间**: 4 方向 — UP, DOWN, LEFT, RIGHT
- **预算用尽**: 返回离目标最近的部分路径

#### 搜索预算的随机性（agent 能力建模）

```python
BUDGET_CLASSES = {
    4:  [3, 4, 5],    # low competence
    8:  [6, 8, 12],   # medium
    16: [14, 16, 20], # high
}
BUDGET_CLASS_PROBS = [0.25, 0.50, 0.25]  # peaked at center
```

Agent 的搜索预算从离散 NegBin 近似中采样，模拟有限理性。

### 4.2 Cell Cost Function — J(r, c)

**Latent path（Phase 4+，主线）**：

$$J(r,c) = \lambda_c \cdot \hat{c}_i + \lambda_r \cdot \phi(\hat{\rho}_i) \cdot \gamma_{\text{shield}} \cdot \gamma_{\text{learn}} + \lambda_{uc} \cdot (1-n) \cdot u_c + \lambda_{ur} \cdot (1-n) \cdot u_r$$

| 项 | 符号 | 默认权重 | 含义 |
|----|------|---------|------|
| Predicted cost | $\hat{c}_i$ | $\lambda_c = 1.0$ | 通行代价 |
| Risk penalty | $\phi(\hat{\rho}_i)$ | $\lambda_r = 5.0$ | $\phi(\rho) = -\ln(1 - \rho)$，survival 形式 |
| Cost uncertainty | $u_c$ | $\lambda_{uc} = 0.1$ | Directional cost uncertainty |
| Risk uncertainty | $u_r$ | $\lambda_{ur} = 0.1$ | Directional risk uncertainty |

#### Risk penalty transform

$$\phi(\hat{\rho}) = -\ln(1 - \text{clip}(\hat{\rho}, \epsilon, 1-\epsilon))$$

| $\hat{\rho}$ | $\phi(\hat{\rho})$ | 含义 |
|:---:|:---:|------|
| 0.0 | 0.0 | 无风险 |
| 0.1 | 0.105 | 低风险，轻微惩罚 |
| 0.3 | 0.357 | 中风险 |
| 0.5 | 0.693 | 高风险 |
| 0.8 | 1.609 | 极高风险 |
| 0.95 | 2.996 | 几乎致命 |

乘以 $\lambda_r = 5.0$ 后，$\hat{\rho} = 0.5$ 的 cell 增加 3.47 cost，相当于 3.5 步正常移动。

#### Shield 降低

$$\gamma_{\text{shield}} = 1 - r_{\text{shield}} \quad (r_{\text{shield}} = 0.5)$$

有 shield 时 risk penalty 减半。

#### Learning factor

$$\gamma_{\text{learn}} = \text{learning\_factor} + (1 - \text{learning\_factor}) \cdot (1 - n)$$

$$\text{learning\_factor} = \min(1, n_{\text{updates}} / 10)$$

- 未训练时 ($n_{\text{updates}} = 0$): $\gamma_{\text{learn}} = 1 - n$（route necessity 折扣全部 risk）
- 训练充分时 ($n_{\text{updates}} \geq 10$): $\gamma_{\text{learn}} = 1$（完全信任 prediction）

原理：$\sigma(0) = 0.5$ 不是真实 risk 预测，是先验不确定性。未训练时不应惩罚。

#### Route Necessity 折扣

$$n \in [0, 1]: \quad n = e^{-\Delta / \tau}$$

$\Delta = L_{\text{avoid}} - L_{\text{best}}$（避开该路线后路径增加的步数）

- $n \to 1$: 没有替代路线，uncertainty penalty 趋近 0（必须走）
- $n \to 0$: 有同样好的替代路线，full uncertainty penalty

**核心原则: Unknown ≠ Dangerous**

### 4.3 Warning Extra Cost

Tutor 发出 WARN 后，在对应 cell 上添加额外 cost：

```python
warned_cell_extra_cost[(r, c)] = 5.0  # per cell
```

Planner 的 cost_fn 变为 $J(r,c) + \text{extra}(r,c)$。

### 4.4 BeliefPlan — 结构化规划结果

定义于 [belief_planning.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/belief_planning.py)。

```python
@dataclass
class BeliefPlan:
    action: str                     # "UP" / "DOWN" / "LEFT" / "RIGHT" / "STAY"
    next_pos: tuple[int, int]
    planned_prefix: list[tuple]     # 前 horizon 步的计划路径
    full_path: list[tuple]          # 完整 A* 路径
    expected_cost: float            # 累计 cost 预测
    expected_risk: float            # 累计 risk (independence 近似)
    uncertainty: float              # 平均 uncertainty
    runner_up_gap: float            # 最优 vs 次优路径分差
    action_confidence: float        # 归一化置信度
    dominant_reason: str            # 选择原因分类
    score_breakdown: ScoreBreakdown # cost/risk/uncertainty 分项分数
```

#### 置信度计算

$$\text{confidence} = \frac{\text{gap}}{\text{gap} + \tau_{\text{conf}}}$$

$\tau_{\text{conf}} = 1.0$（confidence_temperature）

#### Dominant reason 分类

1. 计算 cost, risk, uncertainty 三项的相对贡献
2. 如果第一和第二的差 < 20%: 返回 "mixed"
3. 否则 deadline 检查: 如果路径长度 > 0.8 × 剩余时间: "deadline_pressure"
4. 否则返回贡献最大的项: "lower_cost" / "lower_risk" / "lower_uncertainty"

### 4.5 PrefixPrediction — 前缀诊断

定义于 [prefix_prediction.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/prefix_prediction.py)。

对计划路径的前 $H$ 个 cell 做**只读**诊断：

$$\text{cumulative\_risk} = 1 - \prod_{i \in P} (1 - \min(\hat{\rho}_i, 0.999))$$

（independence 近似：至少踩到一个 trap 的概率）

```python
@dataclass
class PrefixPrediction:
    prefix_cells: list[tuple]
    cost_predictions: list[float]       # per-cell ĉ
    risk_predictions: list[float]       # per-cell ρ̂
    cost_uncertainties: list[float]     # per-cell u_c
    risk_uncertainties: list[float]     # per-cell u_r
    cumulative_cost: float              # sum(ĉ)
    cumulative_risk: float              # 1 - ∏(1-ρ̂)
    risky_prefix_cells: list[tuple]     # ρ̂ > threshold
```

### 4.6 FailureModeEstimate

```python
@dataclass
class FailureModeEstimate:
    high_cumulative_risk: float     # 沿路径的累计 risk
    high_uncertainty: float         # 平均 uncertainty
    deadline_miss: float            # (path_len - remaining) / remaining
    no_safe_route: float            # 高 cost 候选路径比例
    warning_insufficient: float     # risky 路径中未被 warning 影响的比例
```

---

## 5. Stochastic Agent Policy

定义于 [stochastic_agent_policy.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/stochastic_agent_policy.py)。

用于**分支选择**（GTET/DTMB 的 multi-branch 场景）的 bounded-rational policy。

### 5.1 Utility 函数

$$U(\pi \mid s, \theta) = R_{\text{goal}}(\pi) + \lambda_\theta \cdot R_{\text{pref}}(\pi; \theta) - J_{\text{risk}}(\pi)$$

- $R_{\text{goal}}$ = safety_score（越安全越好）
- $R_{\text{pref}}(\pi;\theta) = w_\theta^T \cdot a(\pi)$ 其中 $a(\pi) = [\text{safety}, \text{tempt}, \text{novelty}, \text{shortcut}]$

### 5.2 偏好类型权重

| $\theta$ | safety | tempt | novelty | shortcut |
|---------|:---:|:---:|:---:|:---:|
| safe | 2.0 | -1.0 | 0.0 | 0.0 |
| risky | -0.5 | 0.5 | 0.0 | 0.0 |
| shiny | 0.0 | 3.0 | 0.0 | 0.0 |
| shortcut | 0.0 | 0.0 | 0.0 | 2.0 |
| neutral | 0.3 | 0.0 | 0.0 | 0.0 |

### 5.3 Softmax + Lapse 混合

$$P_{\text{mix}}(\pi \mid s, \theta) = (1-\epsilon) \cdot \text{softmax}(\beta \cdot U) + \epsilon \cdot \text{Uniform}$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\beta$ | 4.0 | 理性温度 (越高越 rational) |
| $\epsilon$ | 0.1 | lapse rate (随机探索概率) |
| $\lambda_\theta$ | 1.0 | 偏好权重 |

---

## 6. Route Necessity

定义于 [route_necessity.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/route_necessity.py)。

### 6.1 计算公式

$$n_{\text{route}} = \begin{cases} 0.0 & \text{if } \Delta \leq 0 \text{ (替代路线同样好)} \\ e^{-\Delta/\tau} & \text{if } \Delta > 0 \\ 1.0 & \text{if avoid path 不可达或超过 deadline} \\ 0.8 & \text{if 即使 best path 也超过 deadline} \end{cases}$$

其中 $\Delta = L_{\text{avoid}} - L_{\text{best}}$, $\tau = 3.0$

### 6.2 BFS 实现

- **best path**: 使用所有 passable cells 的 BFS 最短路径
- **avoid path**: 排除 route_cells 后的 BFS 最短路径
- 不可达返回 999

---

## 7. RSA Warning Channel — Agent 如何理解警告

定义于 [rsa_warning_channel.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/rsa_warning_channel.py)。

### 7.1 假说空间

$$R = \{\text{left\_risky}, \text{right\_risky}, \text{both\_safe}, \text{hazard\_ahead}\}$$

### 7.2 Utterance 空间

$$U = \{\text{warn\_left}, \text{warn\_right}, \text{warn\_ahead}, \text{generic\_warn}\}$$

Utterance costs: warn_left/right = 0.0, warn_ahead = 0.2, generic_warn = 0.5

### 7.3 Literal Listener (L0)

$$L_0(r \mid u, c) \propto \exp(\lambda_{\text{sem}} \cdot \text{match}(u, r, c)) \cdot P(r \mid c)$$

$\lambda_{\text{sem}} = 3.0$（semantic sharpness）

match 函数基于 side alignment（精确匹配=1.0, 部分=0.5, 矛盾=0.05）

### 7.4 Pragmatic Speaker (S1)

$$S_1(u \mid r, c) \propto \exp(\alpha_{\text{RSA}} \cdot [\log L_0(r \mid u, c) - \lambda_C \cdot C(u)])$$

$\alpha_{\text{RSA}} = 2.0$（speaker rationality）, $\lambda_C = 1.0$（utterance cost weight）

### 7.5 S1 Belief Update

$$b^+(r) \propto S_1(u \mid r, c) \cdot b^-(r)$$

### 7.6 Trust-Gated Variant (S1_trust)

$$b^+(r) \propto [S_1(u \mid r, c)]^{\eta_\tau} \cdot b^-(r)$$

$$\eta_\tau = \text{clip}(\hat{\tau}, 0.3, 2.0)$$

高 trust → 更强的证据更新；低 trust → 警告被打折。

### 7.7 Planner 适配

$$\Delta\rho = \mathbb{E}_{r \sim b^+}[\rho(r)] - \mathbb{E}_{r \sim \text{uniform}}[\rho(r)]$$

返回 $\Delta\rho \in [-0.5, 0.5]$，加到 planner 的 risk 预测上。

---

## 8. Legacy Warning System — 双通道

定义于 [warning_update.py](file:///f:/SCAI/Learning-agent/pedagogical_ip/src/agents/warning_update.py)。

### 通道 1: Pseudo-label 注入

对 upcoming cell $j$，向 risk_head 注入合成样本：

$$\text{risk\_head.update\_from\_label}(z_{\text{proto}}, y_{\text{label}}, w_{\text{eff}})$$

$$w_{\text{eff}} = w_{\text{base}} \cdot \alpha_j, \quad \alpha_j = \exp\left(-\frac{\|z_j - z_{\text{proto}}\|^2}{\tau}\right)$$

| Utterance | Prototype | Label |
|-----------|-----------|-------|
| RISKY_TEXTURE_AHEAD | [0.5, 0.0, 0.85, 0.80] | 0.8 |
| UPPER_LANE_RISKY | [0.0, 0.0, 0.70, 0.60] | 0.7 |
| SAFE_DETOUR_OPEN | [1.0, 0.0, 0.05, 0.05] | 0.0 |

### 通道 2: Lane-level bias

聚合 bias 加到整条 lane 上：

$$b_{\text{warn}} = \lambda_{\text{lane}} \cdot \sum_j \alpha_j \cdot y_\text{label} \quad (\lambda_{\text{lane}} = 5.0)$$

---

## 9. 每步执行流完整细节

```
Step t:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] OBSERVE
  ├── observe_features_patch(pos, patch_radius)
  │     → 生成带噪声的 4D 特征观测
  ├── for each observed cell:
  │     feature_belief.update(r, c, z_obs, σ²_obs, t)
  │     → Kalman 更新后验 (μ, σ²)
  └── cue_cells_seen 计数器 ++

[2] APPLY_TUTOR  (external, see tutor report)
  └── 可能修改: warned_cell_extra, passable, inventory

[3] PLAN_AND_MOVE
  ├── compute_route_necessity(...)
  │     → n ∈ [0,1]: 路径结构必要性
  │
  ├── plan_from_belief(...)  [belief_planning_mode]
  │     ├── plan_with_alternatives_v2(...)
  │     │     ├── 构建 cost_fn(r,c) using cell_cost_v2_latent
  │     │     ├── _astar_core(budget=30) → best path
  │     │     └── 对每个 first-step 候选做 short A* → candidate_scores
  │     ├── compute_prefix_predictions(path, horizon=5)
  │     └── estimate_failure_modes(plan, t, t_max)
  │
  ├── agent_pos = next_pos  (移动)
  ├── mark_traversed(r, c, t)
  │
  └── OUTCOME RESOLUTION
        ├── 获取 x_belief = feature_belief.get_mean(r, c)
        ├── 获取 true_cost, true_risk
        │
        ├── IF RISKY cell:
        │     ├── risky_entered++
        │     ├── effective_risk = true_risk × (1 - shield_reduction) if shield
        │     ├── IF rng.random() < effective_risk:
        │     │     ├── DEATH: update(x, cost, risk=1.0, weight=4.0)
        │     │     └── done = True
        │     └── ELSE (survived):
        │           └── update(x, cost, risk=true_risk or 0.0, weight=1.5)
        │
        └── ELSE (safe cell):
              └── update(x, cost, risk=0.0, weight=0.1)
```

---

## 10. 跨 Episode 状态持久化 (PRS Session)

| 状态 | 跨 episode 持久? | 说明 |
|------|:---:|------|
| `LatentCostRiskHead` (weights) | ✓ (stateful 模式) | 这是 transfer 的载体 |
| `FeatureBeliefMap` | ✗ | 每个 episode 全新 (因为 map 不同) |
| `belief_cost` | ✗ | 每个 episode 重置 |
| `passable` | ✗ | 每个 episode 重置 |
| `BayesianRiskHead` (standalone) | ✗ | 被 LatentCostRiskHead 替代 |
| `warned_cell_extra` | ✗ | 每个 episode 重置 |
| `inventory` | ✗ | 每个 episode 重置 |

---

## 11. 参数总汇表

### 11.1 学习参数

| 参数 | 值 | 模块 |
|------|-----|------|
| Feature dim $d$ | 4 | all |
| Cost learning rate $\eta_c$ | 0.1 | BayesianCostHead |
| Risk learning rate $\eta_r$ | 0.3 | BayesianRiskHead |
| Prior variance $\sigma^2_{\text{prior}}$ | 1.0 | both heads |
| Gradient clip norm | 5.0 | both heads |
| Weight norm clamp | 10.0 | both heads |
| Cost bias prior $b_{c,0}$ | 1.0 | BayesianCostHead |
| Risk bias prior $b_{r,0}$ | 0.0 | BayesianRiskHead |

### 11.2 观测参数

| 参数 | 值 | 模块 |
|------|-----|------|
| Self noise $\sigma^2_{\text{self}}$ | 0.01 | observation_model |
| Neighbor noise $\sigma^2_{\text{nbr}}$ | 0.08 | observation_model |
| Far noise $\sigma^2_{\text{far}}$ | 0.20 | observation_model |
| Belief prior mean $\mu_0$ | 0.5 | FeatureBeliefMap |
| Belief prior var $\sigma^2_0$ | 0.25 | FeatureBeliefMap |

### 11.3 规划参数

| 参数 | 值 | 模块 |
|------|-----|------|
| Search budget | 30 | planner_astar |
| $\lambda_c$ (cost weight) | 1.0 | cell_cost_v2_latent |
| $\lambda_r$ (risk weight) | 5.0 | cell_cost_v2_latent |
| $\lambda_{uc}$ (cost unc) | 0.1 | cell_cost_v2_latent |
| $\lambda_{ur}$ (risk unc) | 0.1 | cell_cost_v2_latent |
| Necessity $\tau$ | 3.0 | route_necessity |
| Confidence $\tau_{\text{conf}}$ | 1.0 | belief_planning |
| Prefix horizon | 5 | prefix_prediction |
| Risk threshold | 0.3 | prefix_prediction |

### 11.4 Warning 参数

| 参数 | 值 | 模块 |
|------|-----|------|
| $\lambda_{\text{lane}}$ (lane bias) | 5.0 | warning_update |
| $w_{\text{base}}$ (pseudo-label) | 5.0 | warning_update |
| $\tau$ (feature match) | 0.3 | warning_update |
| $\lambda_{\text{sem}}$ (RSA sharpness) | 3.0 | rsa_warning_channel |
| $\alpha_{\text{RSA}}$ (speaker rationality) | 2.0 | rsa_warning_channel |
| $\lambda_C$ (utterance cost) | 1.0 | rsa_warning_channel |

### 11.5 Stochastic Policy 参数

| 参数 | 值 | 模块 |
|------|-----|------|
| $\beta$ (temperature) | 4.0 | stochastic_agent_policy |
| $\epsilon$ (lapse rate) | 0.1 | stochastic_agent_policy |
| $\lambda_\theta$ (pref weight) | 1.0 | stochastic_agent_policy |
