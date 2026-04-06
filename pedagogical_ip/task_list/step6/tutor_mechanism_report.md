# Tutor 机制与公式总览

> **参考文档** — 当前项目中的 Tutor 系统完整技术说明：所有公式、参数、决策流程。
> 涵盖微观决策(Micro)、宏观干预(Macro)、观测器(Observer)、后验推断(Posterior)、后果推演(Consequence Rollout)。
> 最后更新：2026-03-30。

---

## 1. 系统总览

Tutor 系统分为三个决策层和两个推断层：

```
┌─────────────────────────────────────────────────────────────────────┐
│  推断层 2: 联合后验 q(g, θ, z)                                       │
│  JointGoalPrefPosterior — 组合目标 × 偏好 × 诱惑                      │
│  + CompositeGoalCompatibility 结构先验                                │
├─────────────────────────────────────────────────────────────────────┤
│  推断层 1: 5D Observer m̂_t                                          │
│  A1MtObserver — (τ̂, ν̂, γ̂_gen, γ̂_spec, κ̂)                         │
├─────────────────────────────────────────────────────────────────────┤
│  决策层 3: Goal-Conditional Curriculum (宏观)                        │
│  S(ℓ) = E_q[lift] + λ·V_teach - λ_infl·R_infl + β_κ·g_κ(κ̂)       │
├─────────────────────────────────────────────────────────────────────┤
│  决策层 2: Option Intervention Controller (中观)                     │
│  Q_opt = Q_base + λ_teach·V_teach + λ_time·U_time                  │
│         - λ_infl·R_infl - λ_res·C_res                              │
├─────────────────────────────────────────────────────────────────────┤
│  决策层 1: BCICTv4 Micro Tutor (微观) [FROZEN]                       │
│  Q(a|m̂) = Q_online + λ_teach·V_full - λ_over·R_over                │
│  A_micro = {WAIT, WARN}                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 微观决策：BCICTv4 (internalization_control_tutor_v4.py)

### 2.1 动作空间

$$\mathcal{A}_{micro} = \{WAIT,\; WARN\}$$

SOFT (dose=0.5) 被证明结构冗余（V_SOFT ≈ 0），保留为 research-only。

### 2.2 Q 函数

对每个 dose $\omega \in \{0, 0.5, 1.0\}$：

$$Q(\omega) = Q_{online}(\omega) + \lambda_{teach} \cdot V_{full}(\omega) - \lambda_{over} \cdot R_{over}(\omega)$$

最终动作：

$$a^* = \arg\max_\omega Q(\omega)$$

**代码位置**：`internalization_control_tutor_v4.py:243-249`

### 2.3 Q_online 组件

$$Q_{online}^{WARN} = 1.0 \cdot \Delta_s + 2.0 \cdot \text{dVOI} + 1.5 \cdot (1 - p_{self}) + 1.0 \cdot \text{tempt} - 0.05$$

$$Q_{online}^{WAIT} = 2.0 \cdot p_{self} \cdot \Delta_s - 1.5 \cdot p_{fail} + 2.0$$

$$Q_{online}^{SOFT} = 0.5 \cdot Q_{online}^{WARN} + 0.5 \cdot Q_{online}^{WAIT}$$

**代码位置**：`internalization_control_tutor_v4.py:206-207`

其中：
- $\Delta_s = \max\big(|s_{A,full} - s_{B,full}| - |s_{A,vis} - s_{B,vis}|, 0\big)$ — 信息不对称
- $\text{dVOI} = \max\big(\sigma(|s_{full}|) - \sigma(|s_{vis}|), 0\big)$ — Value of Information 增量
- $p_{self}$ — 自发现概率（见 §8）
- $p_{fail}$ — 不干预失败概率（见 §8）

### 2.4 V_full 组件：教学价值

前向模拟 $\hat m' = \text{predict\_m}(m, \omega, ...)$ 后：

$$V = L_{now} - L_{next}$$

路径敏感调整：

$$p_{blind} = \begin{cases} 0.7 \cdot \omega & \text{if no self-evidence} \\ 0.2 \cdot \omega & \text{if has self-evidence} \end{cases}$$

$$p_{sd} = p_{self} \cdot s_{sd} \cdot (1 - \omega), \quad s_{sd} = \begin{cases} 0.8 & \text{self-discovery subtype} \\ 0.4 & \text{otherwise} \end{cases}$$

$$V_{full} = V + \lambda_{sd} \cdot p_{sd} - \lambda_{dep} \cdot p_{blind}$$

**代码位置**：`internalization_control_tutor_v4.py:232-239`

### 2.5 参数表

| 参数 | 值 | 含义 |
|------|:--:|------|
| $\lambda_{teach}$ | 3.5 | 教学价值权重 |
| $\lambda_{over}$ | 4.0 | 过度教学惩罚权重 |
| $\lambda_{sd}$ | 1.5 | 自发现意愿权重 |
| $\lambda_{dep}$ | 2.0 | 依赖性惩罚权重 |
| $\beta$ (agent) | 4.0 | Agent softmax 温度 |
| $\epsilon$ (agent) | 0.1 | Agent lapse rate |

### 2.6 前向模拟 _predict_m

$$\hat m' = \text{predict\_m}(m, \omega, tempt, risk, subtype)$$

| 条件 | 更新 |
|------|------|
| dose > 0 | `update_risk(0.05, 0.15)`, `update_trust(warn_helpful=(risk>0.25))` |
| dose > 0, no self_evidence | `update_dependence(blind_obey=True)` |
| dose < 1.0 | `ν' = ν + dose · (ν_{full} - ν)` (dose-scaled) |
| dose > 0 | `update_gamma_gen(sustained_pressure=True)` |
| dose < 1.0 | `γ' = γ + dose · (γ_{full} - γ)` (dose-scaled) |
| dose = 0, self-disc | `update_dependence(self_discovery=True)` |
| dose = 0, tempt>0.5, risk>0.3 | `update_gamma_spec(tempt_error=True)` |
| dose = 0, novelty | `update_gamma_gen(successful_exploration=True)` |

**代码位置**：`internalization_control_tutor_v4.py:76-100`

### 2.7 Phase 7 Shadow 扩展（默认 OFF）

#### EPU Shadow (Expected Pedagogical Utility)

$$EU(\omega) = 3.0 \cdot \text{survival} + 2.0 \cdot \text{learning} - 1.5 \cdot \text{OTR} - 0.05 \cdot \omega$$

$$\text{survival} = 1 - \hat\kappa \cdot risk \cdot \max(1 - 0.5\omega, 0.3)$$

$$\text{learning} = 0.4(1-\hat\nu) + 0.4\hat\tau + 0.2(1-\hat\gamma_{gen})$$

$$\text{OTR} = \max(\hat\nu + \hat\gamma_{gen} - 1, 0)$$

**代码位置**：`internalization_control_tutor_v4.py:104-121`

#### Belief-Horizon p_self

$$p_{self}^{new} = (1-\eta) \cdot p_{geom} + \eta \cdot p_{belief}$$

$$p_{belief} = \min\big(\text{risk\_aware} \cdot \text{update\_gain} \cdot \text{info\_window}, 1\big)$$

$$\text{risk\_aware} = \min(2\hat\kappa, 1), \quad \text{update\_gain} = \max(1-\hat\nu, 0.1)$$

**代码位置**：`internalization_control_tutor_v4.py:123-137`

#### EIG Observation Value

$$I(A; \theta) = \sum_{\theta} \sum_{a} P(a,\theta) \cdot \log\frac{P(a,\theta)}{P(a) \cdot P(\theta)}$$

$$P(a|\theta) = \sigma\big(\beta \cdot (U_0^{(\theta)} - U_1^{(\theta)})\big)$$

**代码位置**：`internalization_control_tutor_v4.py:139-163`

---

## 3. 5D 观测器：A1MtObserver (internalization_observer.py)

### 3.1 状态空间

$$\hat m_t = (\hat\tau_t,\; \hat\nu_t,\; \hat\gamma_t^{gen},\; \hat\gamma_t^{spec},\; \hat\kappa_t)$$

| 维度 | 语义 | Layer | 方向 |
|------|------|:-----:|:----:|
| $\hat\tau$ | 信任度（valid-advice uptake） | L2 micro | ↑=好 |
| $\hat\nu$ | 依赖度（blind obedience） | L2 micro | ↑=差 |
| $\hat\gamma^{gen}$ | 一般抑制（exploration inhibition） | L2 micro | ↑=差 |
| $\hat\gamma^{spec}$ | 诱惑抗性（behavioral state） | L1+L3 diag | ↑=好 |
| $\hat\kappa$ | 风险校准（risk prediction accuracy） | L1+L3 macro | ↑=cautious |

### 3.2 事件提取

从观测事件 `ObsEvent` 提取软信号：

$$e_{trust+} = \mathbf{1}[warned \land follow \land correct]$$

$$e_{trust-} = \mathbf{1}[warned \land follow \land wrong]$$

$$e_{blind} = \mathbf{1}[warned \land follow] \cdot (1 - p_{self})$$

$$e_{selfdisc} = \mathbf{1}[self\_discovery] \cdot p_{self}$$

$$e_{pressure} = (1-\alpha_p) \cdot e_{pressure}^{prev} + \alpha_p \cdot dose, \quad \alpha_p = 0.3$$

$$e_{explore+} = \mathbf{1}[beneficial\_novelty \lor self\_discovery]$$

**代码位置**：`internalization_observer.py:367-375`

### 3.3 τ̂ 更新（Trust）

$$\hat\tau' = \hat\tau + \alpha_\tau^+ \cdot e_{trust+} \cdot (1 - \hat\tau) - \alpha_\tau^- \cdot e_{trust-} \cdot \hat\tau$$

条件回归（仅当无近期事件 AND 低置信度）：

$$\text{if } recent\_events_\tau = 0: \quad \lambda_{eff} = \lambda_\tau \cdot (1 - conf_\tau)$$

$$\hat\tau' \mathrel{+}= \lambda_{eff} \cdot (\tau_0 - \hat\tau')$$

$$\hat\tau' = \text{clip}(\hat\tau', 0, 1)$$

| 参数 | A0 值 | A1 值 |
|------|:-----:|:-----:|
| $\alpha_\tau^+$ | 0.22 | 0.22 |
| $\alpha_\tau^-$ | 0.10 | 0.10 |
| $\beta_\tau^{probe}$ | 0.15 | **0.0** (OFF) |
| $\lambda_\tau$ | 0.02 | **0.005** |
| $\tau_0$ | 0.3 | 0.3 |

**代码位置**：`internalization_observer.py:402-415`

### 3.4 ν̂ 更新（Dependence）

$$\hat\nu' = \hat\nu + \alpha_\nu^+ \cdot e_{blind} \cdot (\nu_{max} - \hat\nu) - \alpha_\nu^- \cdot e_{selfdisc} \cdot \hat\nu$$

条件回归：

$$\text{if } recent\_events_\nu = 0: \quad \hat\nu' \mathrel{+}= \lambda_\nu(1-conf_\nu) \cdot (\nu_0 - \hat\nu')$$

$$\hat\nu' = \text{clip}(\hat\nu', 0, \nu_{max})$$

| 参数 | A0 值 | A1 值 |
|------|:-----:|:-----:|
| $\alpha_\nu^+$ | 0.18 | 0.18 |
| $\alpha_\nu^-$ | 0.13 | 0.13 |
| $\beta_\nu^{probe}$ | 0.10 | **0.0** (OFF) |
| $\lambda_\nu$ | 0.02 | **0.005** |
| $\nu_0$ | 0.1 | 0.1 |
| $\nu_{max}$ | 0.8 | 0.8 |

**代码位置**：`internalization_observer.py:417-427`

### 3.5 γ̂_gen 更新（General Suppression）

$$\hat\gamma_{gen}' = \hat\gamma_{gen} + \alpha_\gamma^+ \cdot e_{pressure} \cdot (\gamma_{max} - \hat\gamma_{gen}) - \alpha_\gamma^- \cdot e_{explore+} \cdot \hat\gamma_{gen}$$

条件回归同上。

$$\hat\gamma_{gen}' = \text{clip}(\hat\gamma_{gen}', 0, \gamma_{max})$$

| 参数 | A0 值 | A1 值 |
|------|:-----:|:-----:|
| $\alpha_\gamma^+$ | 0.07 | 0.07 |
| $\alpha_\gamma^-$ | 0.10 | 0.10 |
| $\beta_\gamma^{probe}$ | 0.10 | **0.0** (OFF) |
| $\lambda_\gamma$ | 0.02 | **0.005** |
| $\gamma_0$ | 0.0 | 0.0 |
| $\gamma_{max}$ | 0.5 | 0.5 |

**代码位置**：`internalization_observer.py:429-439`

### 3.6 γ̂_spec 更新（Temptation Resistance, P4-A）

仅在 $lure \geq 0.3$ 时触发：

$$\hat\gamma_{spec}' = \hat\gamma_{spec} + \begin{cases} \alpha_{gs}^+ \cdot lure \cdot (1 - \hat\gamma_{spec}) & \text{if resisted (correct)} \\ -\alpha_{gs}^- \cdot lure \cdot \hat\gamma_{spec} & \text{if followed (incorrect)} \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_{gs}^+$ (resist) | 0.03 |
| $\alpha_{gs}^-$ (follow) | 0.025 |
| lure threshold | 0.3 |
| $\gamma_{spec,max}$ | 1.0 |

**代码位置**：`internalization_observer.py:441-453`

### 3.7 κ̂ 更新（Risk Calibration, P5）

仅在 $risk \geq 0.1$ 且 $risk\_hat \neq$ None 时触发：

$$\delta_t^{risk} = risk - risk\_hat \quad\text{(signed error)}$$

$$\hat\kappa' = (1-\lambda_\kappa)\hat\kappa + \lambda_\kappa \kappa_0 + \begin{cases} \alpha_\kappa^+ \cdot \delta^{risk} \cdot (\kappa_{max} - \hat\kappa) & \text{if } \delta>0 \text{ (underestimated)} \\ \alpha_\kappa^- \cdot \delta^{risk} \cdot (\hat\kappa - \kappa_{min}) & \text{if } \delta<0 \text{ (overestimated)} \end{cases}$$

$$\hat\kappa' = \text{clip}(\hat\kappa', \kappa_{min}, \kappa_{max})$$

| 参数 | 值 | 含义 |
|------|:--:|------|
| $\kappa_0$ | 0.3 | 回归锚点 |
| $\lambda_\kappa$ | 0.02 | 回归速率（慢） |
| $\alpha_\kappa^+$ | 0.015 | 低估风险→更谨慎 |
| $\alpha_\kappa^-$ | 0.012 | 高估风险→放松 |
| risk gate | 0.1 | 最小风险触发阈值 |

**代码位置**：`internalization_observer.py:455-469`

### 3.8 置信度更新

$$conf_\tau' = (1-\rho) \cdot conf_\tau + \rho \cdot q_\tau$$

$$conf_\nu' = (1-\rho) \cdot conf_\nu + \rho \cdot q_\nu$$

$$conf_\gamma' = (1-\rho) \cdot conf_\gamma + \rho \cdot q_\gamma$$

其中 $\rho = 0.15$（EMA rate），$q_*$ 为预测一致性信号。

**代码位置**：`internalization_observer.py:499-504`

---

## 4. 中观决策：Option Intervention Controller (option_intervention_controller.py)

### 4.1 动作空间

$$\mathcal{O}_{macro} = \{NONE,\; WARN,\; UNLOCK,\; ITEM\_DROP\}$$

### 4.2 评分函数

$$Q_{opt}(o) = Q_{base}(o) + \lambda_{teach} \cdot V_{teach}(o) + \lambda_{time} \cdot U_{time}(o) - \lambda_{infl} \cdot R_{infl}(o) - \lambda_{res} \cdot C_{res}(o)$$

**代码位置**：`option_intervention_controller.py:186-194`

### 4.3 各组件

#### V_teach — 教学价值

| Option | $V_{teach}$ |
|--------|:-----------:|
| NONE | 0.0 |
| WARN | 1.0 |
| UNLOCK | 0.5 |
| ITEM_DROP | 0.1 |

#### U_time — 时机紧迫度

$$U_{time}^{WARN} = p_{blind} \cdot 1.0$$

$$U_{time}^{UNLOCK} = p_{timeout} \cdot 1.0$$

$$U_{time}^{ITEM\_DROP} = \max(p_{blind}, p_{timeout}) \cdot 0.5$$

#### R_infl — 膨胀惩罚

$$R_{infl}(o) = a_o \cdot \nu_{pressure} + b_o \cdot \gamma_{pressure}$$

$$\nu_{pressure} = \max(\hat\nu + \Delta\hat\nu, 0), \quad \gamma_{pressure} = \max(\hat\gamma_{gen} + \Delta\hat\gamma_{gen}, 0)$$

| Option | $a_o$ (ν weight) | $b_o$ (γ weight) |
|--------|:-----------------:|:-----------------:|
| WARN | 1.0 | 0.5 |
| UNLOCK | 0.3 | 0.5 |
| ITEM_DROP | 0.8 | 0.0 |

#### C_res — 资源代价

$$C_{res}^{WARN} = 0.3 \cdot (1 + 0.2 \cdot n_{warn})$$

$$C_{res}^{UNLOCK} = 1.0 \cdot (1 + 2.0 \cdot \max(0, n_{unlock} - 2))$$

$$C_{res}^{ITEM\_DROP} = \begin{cases} 1.5 & \text{no shield} \\ 4.5 & \text{already shielded} \end{cases}$$

### 4.4 权重参数

| 参数 | 值 |
|------|:--:|
| $\lambda_{teach}$ | 1.5 |
| $\lambda_{time}$ | 1.0 |
| $\lambda_{infl}$ | 4.0 |
| $\lambda_{res}$ | 1.0 |

**代码位置**：`option_intervention_controller.py:33-58`

---

## 5. 干预语义 (intervention_semantics.py)

### 5.1 WARN — 信念证据

$$\mu_i^+ = \mu_i + \alpha_{warn} \cdot \hat{v}_{warn}$$

$$\Sigma_i^+ = (1 - \beta_{warn}) \cdot \Sigma_i$$

| 参数 | 值 |
|------|:--:|
| $\alpha_{warn}$ | 0.3 |
| $\beta_{warn}$ | 0.2 |

**不变量**：WARN 不改变世界拓扑。

**代码位置**：`intervention_semantics.py:32-80`

### 5.2 UNLOCK — 可达性揭示

$$s_{t+1}^{world} = \text{Unlock}(s_t^{world})$$

$$b_{t+1}^{A,env} = \text{AffordanceReveal}(b_t^{A,env}, s_{t+1}^{world})$$

| 参数 | 值 |
|------|:--:|
| uncertainty_reduction | 0.3 |

**不变量**：UNLOCK 不改变风险值。

**代码位置**：`intervention_semantics.py:94-141`

### 5.3 ITEM_DROP — 遍历缓解

$$\text{TraversalCost}^{shield}(i) = \lambda_r \cdot (1 - \gamma_{shield}) \cdot \varphi(\hat r_i)$$

$$\varphi(r) = -\ln(1 - r)$$

| 参数 | 值 |
|------|:--:|
| $\gamma_{shield}$ | 0.5 |
| $\lambda_r$ | 3.0 |

**不变量**：ITEM_DROP 不改变 agent 信念或世界拓扑。

**代码位置**：`intervention_semantics.py:156-201`

---

## 6. 后果推演 (consequence_grounded_option_rollout.py)

### 6.1 干预→BranchAttributes 映射

对每个选项 $o$，修改 branch 属性后通过 ActionPredictor 计算反事实动作分布：

| Option | 效果 |
|--------|------|
| **WARN** | if risk_penalty > 0.15: `risk_penalty += α_warn`; if safety > 0.5: `safety += α_warn·0.5` |
| **UNLOCK** | `shortcut_bonus += α_unlock` |
| **ITEM_DROP** | `risk_penalty *= (1 - γ_shield)` |

$$\text{success\_lift}(o) = P_{safe}^{cf}(o) - P_{safe}^{orig}$$

| 参数 | 值 |
|------|:--:|
| $\alpha_{warn}$ | 0.15 |
| $\alpha_{unlock}$ | 0.5 |
| $\gamma_{shield}$ | 0.5 |

**代码位置**：`consequence_grounded_option_rollout.py:106-155`

---

## 7. Agent 动作模型 (stochastic_agent_policy.py + action_predictor.py)

### 7.1 Agent Utility 函数

$$U(\pi|\theta) = R_{goal}(\pi) + \lambda_\theta \cdot \langle \vec{w}_\theta, \vec{x}_\pi \rangle - J_{risk}(\pi)$$

$$R_{goal} = \text{safety\_score}, \quad J_{risk} = \text{risk\_penalty}$$

$$\vec{x}_\pi = [\text{safety}, \text{temptation}, \text{texture\_novelty}, \text{shortcut\_bonus}]$$

**代码位置**：`stochastic_agent_policy.py:58-71`

### 7.2 偏好权重表

$$\vec{w}_\theta =$$

| θ | safety | temptation | novelty | shortcut |
|:-:|:------:|:----------:|:-------:|:--------:|
| safe | 1.0 | -0.5 | 0.0 | 0.0 |
| shiny | 0.0 | 1.0 | 0.5 | 0.0 |
| risky | 0.0 | 0.5 | 1.0 | 0.0 |
| shortcut | 0.0 | 0.0 | 0.0 | 1.5 |
| neutral | 0.3 | 0.3 | 0.3 | 0.1 |

### 7.3 选择概率

$$P_{mix}(\pi|s,\theta) = (1-\epsilon)\cdot\frac{\exp(\beta \cdot U(\pi))}{\sum_{\pi'}\exp(\beta \cdot U(\pi'))} + \epsilon \cdot \frac{1}{|\Pi|}$$

| 参数 | 值 |
|------|:--:|
| $\beta$ | 4.0 |
| $\epsilon$ | 0.1 |
| $\lambda_\theta$ | 1.0 |

**代码位置**：`stochastic_agent_policy.py:74-98`

### 7.4 ActionPredictor 接口

$$\log P(a_{obs}|s,b_A) = \text{score}(world\_state, agent\_belief, branches, a_{obs})$$

用于逆规划的核心似然项。

**代码位置**：`action_predictor.py:95-105`

---

## 8. 自发现概率估计 (self_discovery.py)

### 8.1 p_self

$$p_{self} = \sigma\left(\frac{d_{commit} - d_{reveal} - m}{\tau_v}\right)$$

当 $d_{commit} \gg d_{reveal}$：$p_{self} \to 1$（agent 先看到线索）。
当 $d_{commit} \ll d_{reveal}$：$p_{self} \to 0$（盲目承诺）。

| 参数 | 默认值 |
|------|:------:|
| margin $m$ | 0.0 |
| $\tau_v$ | 1.0 |

### 8.2 p_fail

$$p_{fail} = 1 - p_{self}(d_c, d_r, \tau_f=1.5)$$

$$p_{fail} = 1 - \sigma\left(\frac{d_c - d_r}{1.5}\right)$$

**代码位置**：`self_discovery.py:24-73`

---

## 9. 后验推断

### 9.1 q(g,z) — GoalTemptationPosterior (goal_temptation_posterior.py)

$$q_t(g,z) \propto q_{t-1}(g,z) \cdot P(a_t^{obs}|s_t, g, z)$$

- 目标：$g \in \{true\_goal, decoy\_goal\}$
- 诱惑：$z \in \{0.0, 0.3, 0.6, 0.9\}$
- 先验：$q_0(z) = (0.4, 0.3, 0.2, 0.1)$（低诱惑偏置）
- 诱惑对 BranchAttributes 的效果：$tempt' = tempt + z$ on risky branch
- 目标对 BranchAttributes 的效果：decoy_goal → `safety' = 1 - safety`

**代码位置**：`goal_temptation_posterior.py:103-147`

### 9.2 q(g,θ,z) — JointGoalPrefPosterior (joint_goal_pref_posterior.py)

$$q_t(g,\theta,z) \propto q_{t-1}(g,\theta,z) \cdot P(a_t^{obs}|s_t, g, \theta, z) \cdot \exp(\beta_C \cdot C_t(g))$$

- 目标：$g \in \mathcal{G}$（8 个：4 atomic + 4 composite）
- 偏好：$\theta \in \Theta$ （$\Theta_2 = \{safe, shiny\}$ 或 $\Theta_K = \{safe, shiny, risky, shortcut, neutral\}$）
- 诱惑：$z \in \{0.0, 0.3, 0.6, 0.9\}$（可选）

**目标条件选择概率**（来自 GoalHypothesisSpace）：

$$U(a;g,\theta) = \langle \vec{w}_g, \vec{x}_a \rangle \cdot f_{scale} + \lambda_\theta \cdot \langle \vec{w}_\theta, \vec{x}_a \rangle - J_{risk}(a)$$

$$P(a|g,\theta) = (1-\epsilon)\cdot\text{softmax}(\beta \cdot U) + \epsilon \cdot \text{Unif}$$

**代码位置**：`joint_goal_pref_posterior.py`，`compositional_goal_hypotheses.py`

### 9.3 Composite Goal Compatibility (composite_goal_compatibility.py)

$$C_t(g) = \bar L_g - \lambda_{red} \cdot \text{redundancy}(g) - \lambda_{comp} \cdot (|g|-1)$$

其中：
- $\bar L_g = \frac{1}{|g|}\sum_{u \in g} L_u$ — 各子目标平均对数似然
- $\text{redundancy}(g) = \max_{u_i, u_j \in g} D_{KL}(P^{u_i} \| P^{u_j})$（行为分布 KL 散度）
- $(|g|-1)$ — 复杂度惩罚（原子=0，二元组合=1）

| 参数 | 值 |
|------|:--:|
| $\beta_C$ (compat strength) | 0.5 |
| $\lambda_{comp}$ | 0.3 |
| $\lambda_{red}$ | 0.2 |

**子目标边际**（canonical reporting metric）：

$$q_t(u) = \sum_{g \ni u}\sum_\theta\sum_z q_t(g,\theta,z)$$

**代码位置**：`composite_goal_compatibility.py`

---

## 10. 宏观课程：Goal-Conditional Curriculum Hook (goal_conditional_curriculum_hook.py)

### 10.1 评分函数

$$S(\ell) = \mathbb{E}_{q(g,\theta)}\big[\text{lift}(\ell|g,\theta)\big] + \lambda_{teach} \cdot V_{teach} \cdot \mathbf{1}[o \neq NONE] - \lambda_{infl} \cdot R_{infl} + \beta_\kappa \cdot g_\kappa(\hat\kappa) \cdot w_o$$

$$\text{lift}(\ell|g,\theta) = P_{safe}^{cf}(o|g,\theta) - P_{safe}^{orig}(g,\theta)$$

$$V_{teach} = \frac{H(q)}{H_{max}(q)}$$

$$R_{infl} = \max(0, \hat\nu) \cdot 0.1 \quad \text{(仅 } o \neq NONE\text{)}$$

$$g_\kappa(\hat\kappa) = \max(0, \hat\kappa), \quad w_o = \begin{cases} 1.0 & o \neq NONE \\ 0.5 & o = NONE \end{cases}$$

| 参数 | 值 |
|------|:--:|
| $\lambda_{teach}$ | 0.5 |
| $\lambda_{infl}$ | 4.0 |
| $\beta_\kappa$ | 0.02 |
| min_confidence | 0.15 |

**代码位置**：`goal_conditional_curriculum_hook.py:67-167`

---

## 11. κ̂ 宏观奖励 (Stage 5)

$$S_{teach}^{5d}(\ell|x_t) = S_{teach}^{base}(\ell|x_t) + \beta_\kappa \cdot \mathbf{1}[\ell \in \mathcal{L}_{risk}] \cdot \frac{|\hat\kappa_t - \kappa_0|}{\kappa_{max} - \kappa_{min}}$$

Centered-deviation 设计：使用 $|\hat\kappa - \kappa_0|$（距锚点绝对偏差），不区分方向。

| 参数 | 值 |
|------|:--:|
| $\beta_\kappa$ | 0.02 (minimum effective dose) |
| $\kappa_0$ | 0.3 |
| $\mathcal{L}_{risk}$ | {tic_rescue_heavy, blind_corridor, warn_rescue} |

---

## 12. 三层分离原则

### Layer 2 Micro（只看 3D）

$$Q_{micro}(a) = f(\hat\tau, \hat\nu, \hat\gamma_{gen}, \text{scene})$$

$\hat\gamma_{spec}$ 和 $\hat\kappa$ **不进入** micro Q。已证明：P4-C attribution ablation 599/600 step identity。

### Layer 3 Macro（看 5D）

$$x_t^{macro} = \big(q_t(g,\theta,z),\; \hat\tau_t,\hat\nu_t,\hat\gamma_t^{gen},\hat\gamma_t^{spec},\hat\kappa_t\big)$$

$\hat\kappa$ 以 **additive macro state** 形式进入，不是后验隐变量。

### 核心设计原则

> **"Estimate first, consume later."** — 每个新维度先进入 Layer 1 状态估计器，只有通过信号审计、非冗余证明、OOD 鲁棒性后才能进入 Layer 2/3 评分。

---

## 13. 完整模块清单

| 文件 | 大小 | 核心公式 |
|------|:----:|---------|
| `internalization_control_tutor_v4.py` | 14KB | §2: $Q_{micro}$, V_full, Q_online |
| `internalization_observer.py` | 32KB | §3: 5D update (τ̂,ν̂,γ̂_gen,γ̂_spec,κ̂) |
| `option_intervention_controller.py` | 10KB | §4: $Q_{opt}$ option scoring |
| `intervention_semantics.py` | 9KB | §5: WARN/UNLOCK/ITEM_DROP formal update |
| `consequence_grounded_option_rollout.py` | 7KB | §6: intervention → BranchAttributes |
| `action_predictor.py` | 4KB | §7: $P(a|s,b)$ bounded-rational |
| `stochastic_agent_policy.py` | 3KB | §7: $U(\pi|\theta)$ utility |
| `self_discovery.py` | 2KB | §8: $p_{self}$, $p_{fail}$ |
| `goal_temptation_posterior.py` | 9KB | §9.1: $q(g,z)$ Bayesian update |
| `joint_goal_pref_posterior.py` | 11KB | §9.2: $q(g,\theta,z)$ + compat |
| `composite_goal_compatibility.py` | 8KB | §9.3: $C_t(g)$ structural prior |
| `compositional_goal_hypotheses.py` | 6KB | §9.2: goal-conditioned utility |
| `compositional_goal_bridge.py` | 6KB | CGC-v2 → POMDP bridge |
| `goal_conditional_curriculum_hook.py` | 6KB | §10: $S(\ell)$ macro scoring |
| `intervention_risk_head.py` | 5KB | $p_{blind}$, $p_{timeout}$ estimation |
| `intervention_policy.py` | 16KB | Base intervention policy |
| `robot_belief_over_agent.py` | 6KB | Robot belief POMDP shell |
| `shadow_bridge.py` | 7KB | Shadow parity adapter |
