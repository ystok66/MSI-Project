# RSA Pragmatic Communication Research Project

A Bayesian concept learning and pragmatic communication model based on the Rational Speech Acts (RSA) framework.

## Overview

This project implements a computational model for:

1. **Concept Learning**: Learning word meanings from ambiguous visual scenes
2. **Pragmatic Inference**: Using RSA to infer speaker intent beyond literal meaning
3. **Multi-Intent Utterances**: Handling utterances like "2 blue" or "1 red, 1 solid"

---

## Project Structure

```
project/
├── templates.py        # Shape and color definitions (29 shapes, 12 colors)
├── world.py            # Scene and object representations
├── encoders.py         # Feature encoding (Lab color + shape)
├── concepts.py         # Gaussian concept representations
├── gaussian.py         # Diagonal Gaussian math utilities
├── scoring.py          # Inclusion scoring (KL-based)
├── learner.py          # Incremental concept learning
├── rsa.py              # RSA inference (L0, S1, L1, Multi-Intent)
├── rsa_helper.py       # 简化的 Jupyter Notebook 接口
├── active_learner.py   # 主动学习 (Ask/Answer/SelfTrain)
├── rsa_action.py       # Physics Engine (条件规则学习)
├── rsa_action_pixel.py # Pixel Motor System (36D RGBI 运动组合)
├── rsa_plan.py         # Planning Agent (具身规划 + 概念组合)
├── USAGE_GUIDE.md      # 使用指南
└── tests/              # Comprehensive test suite (125+ tests)
```

---

## Core Modules

### 1. `templates.py` - Visual Templates

Defines the visual world vocabulary:

- **Shapes** (29 种): 基础 (`box`, `solid`, `dot`, `donut`), 条形 (`hbar`, `vbar`), L 形 (`l`, `l_90/180/270`), T 形 (`t`, `t_90/180/270`), 十字 (`cross`), S/Z 形 (`s`, `s_90`, `z`, `z_90`), 角 (`corner`, `corner_90/180/270`), 对角 (`diag`, `diag_90`), U 形 (`u`, `u_90/180/270`)
- **Colors** (12 种): `red`, `blue`, `green`, `yellow`, `orange`, `purple`, `cyan`, `magenta`, `brown`, `pink`, `gray`, `white`

Each shape is a 9-dimensional binary occupancy vector (3×3 grid).

### 2. `world.py` - Scene Representation

```python
@dataclass
class Obj:
    shape_name: str                # e.g., "box"
    color_rgb: Tuple[int, int, int] # RGB (0-255)
    occ: np.ndarray                # Shape occupancy (3,3)

@dataclass
class Scene:
    regions: List[Optional[Obj]]  # Up to 4 regions
```

### 3. `encoders.py` - Feature Encoding

Converts objects to 12-dimensional feature vectors:

```
x = [Lab_L, Lab_a, Lab_b, occ_0, occ_1, ..., occ_8]
     ├─ 3 color dims ─┤ ├───── 9 shape dims ──────┤
```

**Gaussian Shape Encoding (默认启用):**

形状维度默认使用高斯平滑 (Gaussian Smoothing) 替代硬编码的 0/1：

- 引入**空间邻域**概念，解决稀疏性问题
- 避免**零方差陷阱**，提升数值稳定性
- 适合 Zero-Shot Learning 的概念合成

```python
# 切换编码模式
import encoders
encoders.USE_GAUSSIAN_SHAPE = True   # 默认: 高斯平滑
encoders.USE_GAUSSIAN_SHAPE = False  # 回退: 硬编码 0/1
encoders.GAUSSIAN_SIGMA = 0.5        # 模糊程度 (0.5 适合 3x3)
```

**Key functions:**

- `rgb_to_lab()`: Perceptually uniform color space
- `shape_to_vec()`: 3×3 matrix → 9D vector (支持高斯平滑)
- `encode_shape_gaussian()`: 高斯平滑核心实现
- `encode_scene()`: Scene → (4, 12) feature matrix + mask

### 4. `concepts.py` - Gaussian Concepts

Each word/token is represented as a diagonal Gaussian:

```python
@dataclass
class Concept:
    token: str                          # Word string
    mu: np.ndarray                      # Mean, shape (d,)
    var: np.ndarray                     # Diagonal variance, shape (d,)
    kappa: float = 0.0                  # Observation count
    embedding: Optional[np.ndarray]     # 语义向量 (ZSL 用)
    decay_count: int = 0                # 记忆年龄 (Jost's Law)
```

**ConceptTable**: Lazy-initialized concept dictionary with prior:

- `mu0 = zeros(d)` (默认), 或 centered prior (`use_centered_prior=True`, d=12 时)
- `var0 = ones(d)` (broad initial uncertainty)
- `kappa0 = 0.5` (weak prior)

### 5. `gaussian.py` - Gaussian Math

**KL Divergence** (diagonal Gaussians):

```
KL(A || B) = 0.5 * Σᵢ [σ²ₐᵢ/σ²ᵦᵢ + (μᵦᵢ - μₐᵢ)²/σ²ᵦᵢ - 1 + ln(σ²ᵦᵢ/σ²ₐᵢ)]
```

**Log-Determinant** (for volume penalty):

```
log|Σ| = Σᵢ log(σ²ᵢ)
```

### 6. `scoring.py` - Inclusion Scoring

**Log-Inclusion Score**:

```
log_inc(t, u) = -KL(Aₜ || Bᵤ) / τ
```

Where:

- `Aₜ ~ N(xₜ, εᵒᵇʲ · I)` is the object distribution (point-like)
- `Bᵤ ~ N(μᵤ, diag(σ²ᵤ))` is the concept distribution

**Default Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `eps_obj` | 1e-4 | Object variance (point-like observation) |
| `tau` | 1.0 | Temperature for softening KL |
| `min_var` | 1e-8 | Minimum variance for numerical stability |

### 7. `learner.py` - Incremental Learning

**Learning Pipeline:**

1. **Infer Posterior**: `P(t | U, scene)` via RSA
2. **Scale by Cardinality**: `w = k · p` (soft counts)
3. **Update Concepts**: Weighted batch merge

**Update Formula** (numerically stable):

```
κ' = κ + W
μ' = μ + (W/κ') · (μ_batch - μ)
M₂' = M₂ + M₂_batch + (κ·W/κ') · (μ_batch - μ)²
σ² = M₂'/κ' + var_floor
```

---

## RSA Framework (`rsa.py`)

### Semantic Potential — Literal Listener $L_0$

$L_0$ 是**纯语义解释器**——只关心"这句话是否描述了这个对象"，不考虑说话人的努力成本。

**Semantic Score** (每个 token 的 log-inclusion 之和)：

$$S_{L_0}(t, U) = \sum_{u \in U} \text{log\_inc}(t, u)$$

**L0 Posterior** (在候选目标上归一化，消除歧义)：

$$P_{L_0}(t | U) = \frac{\exp(S_{L_0}(t, U))}{\sum_{t'} \exp(S_{L_0}(t', U))}$$

> **注意**: Cost ($\beta \cdot \text{vol}$, $\lambda |U|$) 不在 $L_0$ 中。听话人不会因为"这句话太长了"而改变语义理解。Cost 是说话人 ($S_1$) 的考量。

### Rational Speaker $S_1$ (Utility-Based)

$S_1$ 建模为最大化**效用函数**的理性说话者 (Goodman & Frank, 2016):

$$U_{\text{raw}}(U, t) = \underbrace{\text{log\_inc}(t, U)}_{\text{Informativeness}} - \underbrace{\beta \cdot \text{vol}(U) + \lambda |U|}_{\text{Cost}}$$

$$P_{S_1}(U | t) = \text{softmax}_{U' \in \text{Alt}}\big(\alpha \cdot U_{\text{raw}}(U', t)\big)$$

| 组件            | 含义                     | 公式                                      |
| --------------- | ------------------------ | ----------------------------------------- | -------- | --- |
| Informativeness | 话语与对象的语义匹配度   | $\text{log\_inc}(t, U) = \sum_u -KL/\tau$ |
| Volume Cost     | 偏好精确（窄方差）的概念 | $\beta \cdot \sum_u \log                  | \Sigma_u | $   |
| Length Cost     | 偏好简短话语             | $\lambda \cdot                            | U        | $   |
| α (Rationality) | 选择噪声：α 越大越确定   | 缩放整个效用 $U_{\text{raw}}$             |

**Key Features:**

- **Auto Alt from Table**: 自动将所有已知 token 作为 alternatives
- **Dynamic Empty Alt**: ME / Scalar Implicature 场景自动禁用空话语

**Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 5.0 | Speaker rationality (越高越确定) |
| `beta` | 0.1 | 精度偏好 (Volume penalty in Cost) |
| `lam` | 0.0 | 长度代价 (Length penalty in Cost) |
| `auto_alt_from_table` | True\* | 用所有已知 token 做 alternatives |
| `include_empty_alt` | Dynamic | ME/Scalar 场景自动禁用 |

> **Note**: `auto_alt_from_table` 在 `RSAHelper.infer()` 中被禁用以防无关概念干扰。学习 (`learn_step`) 中保持启用。

### Pragmatic Listener $L_1$

$$P_{L_1}(t | U) \propto P(t) \cdot P_{S_1}(U | t)$$

Where $P(t)$ is uniform over non-empty regions.

---

## Multi-Intent RSA

Handles utterances with multiple cardinalities, e.g., "1 blue, 1 solid" or "2 blue".

### Intent Representation

```python
intents = [(['blue'], 1), (['solid'], 1)]  # "1 blue, 1 solid"
intents = [(['blue'], 2)]                  # "2 blue"
```

### Dual Exactness Modes

The system supports **two modes** for enforcing cardinality constraints, selected via `exactness_mode`:

| Mode                  | 语义                 | "exactly k" 来源               | 适用场景            |
| --------------------- | -------------------- | ------------------------------ | ------------------- |
| `"soft_count"` (默认) | 固定 $\|T_j\| = k_j$ | Sigmoid 软计数 + $\gamma$ 惩罚 | 训练稳定 / 可微优化 |
| `"pure_rsa"`          | $\|T_j\| \geq k_j$   | 标量蕴涵 (数词 Alt 竞赛)       | 理论正确 / 语用涌现 |

---

### Mode A: Soft Count (默认)

**L0 分数** (固定大小分配 $|T_j| = k_j$)：

$$S_{L_0}(A, \text{intents}) = \sum_j \left[ \sum_{t \in T_j} \text{log\_inc}(t, W_j) - \beta \cdot k_j \cdot \text{vol}(W_j) - \lambda |W_j| \right] - \gamma \cdot \text{exactness}$$

**Exactness Constraint** (可微计数松弛)：

$$\text{soft\_count}(u, T) = \sum_{t \in T} \sigma\big(0.2 \cdot (\text{log\_inc}(t, u) + 38)\big)$$

$$\text{exactness} = \sum_j \sum_{u \in W_j} \big(\text{soft\_count}(u, T_j) - k_j\big)^2$$

> **评估 (7/10)**: 有效的工程 relax。$\sigma$ 门控将连续 log_inc 二值化为"是/否匹配"。但它计算的是**期望基数** $\mathbb{E}[|T|]$，不是基数概率——两个 0.5 的对象 ($\sum = 1.0$) 与一个确定对象 ($\sum = 1.0$) 被视为等价，忽略了不确定性结构。

---

### Mode B: Pure RSA Competition (标量蕴涵)

"exactly k" 不需要显式惩罚，而是通过 **RSA 说话者竞争自然涌现**。

**核心思想**: 如果有 3 个蓝色对象，理性说话者会倾向说 "3 blue" 而非 "2 blue"（因为更信息化）。因此，听到 "2 blue" 就意味着"不是 3"——这就是**标量蕴涵 (Scalar Implicature)**。

**1. L0 使用 "≥k" 语义** (不做 "exactly k")：

$$S_{L_0}(T; u) = \begin{cases} \sum_{t \in T} s(t, W) - \eta |T| & \text{if } |T| \geq k \\ -\infty & \text{otherwise} \end{cases}$$

其中 $\eta$ 是弱集合大小先验 (默认 0.02)，防止 L0 偏好大集合。

**2. Alt 包含不同数词** (数词竞赛)：

$$\text{Alt}(W) = \{(1, W), (2, W), \dots, (n, W)\}$$

对于多意图还包括跨 token 的组合，如 "(1, blue), (1, solid)"。

**3. S1 通过效用函数选择话语**：

$$\text{Utility}(u, A) = \alpha \cdot \ln P_{L_0}(A | u) - \text{Cost}(u)$$

$$P_{S_1}(u | A) = \text{softmax}_{u' \in \text{Alt}} \ \text{Utility}(u', A)$$

**4. 软计数来自后验边缘化** (不是 sigmoid 门控)：

$$p_i^{(j)} = \Pr(i \in T_j | u) = \sum_{A: i \in T_j} P_{L_1}(A | u)$$

$$\mathbb{E}[|T_j|] = \sum_{i \in \mathcal{O}} p_i^{(j)}$$

> **优势**: 软计数是 RSA 推断的直接产物，具有明确的概率语义。场景 $n \leq 4$，枚举成本可忽略。

---

### 两种模式的评估与比较

| 维度        | soft_count            | pure_rsa             |
| ----------- | --------------------- | -------------------- | ---------- | --- |
| 理论正确性  | 6/10 — 启发式 relax   | 9/10 — 标准 RSA 语用 |
| 训练稳定性  | 9/10 — 可微、梯度友好 | 7/10 — 枚举空间较大  |
| "exactly k" | 显式惩罚 (辅助损失)   | 语用涌现 (标量蕴涵)  |
| 计算复杂度  | $O(C(n, k))$          | $O(2^n \cdot         | \text{Alt} | )$  |
| 不确定性    | 忽略 (只看期望)       | 完整建模             |

**"黄金标准"替代方案**: 如果需要严格的概率论保证，可用 **Poisson-Binomial 分布** 显式计算恰好 $k$ 个成功的概率。对 $n \leq 4$ 计算可行。

**推荐路线**: 训练初期用 `soft_count` 稳定学习，测试/推理阶段切换到 `pure_rsa` 让蕴涵自然工作。

---

### Multi-Intent Hyperparameters

| Parameter         | Default        | Mode       | Description                    |
| ----------------- | -------------- | ---------- | ------------------------------ |
| `exactness_gamma` | 2.0            | soft_count | Exactness constraint strength  |
| `exactness_mode`  | `"soft_count"` | both       | `"soft_count"` or `"pure_rsa"` |
| `eta`             | 0.02           | pure_rsa   | 弱集合大小先验                 |

---

## Usage Examples

### Basic Concept Learning

```python
from templates import SHAPES, COLORS_RGB
from world import Obj, Scene
from encoders import encode_scene
from concepts import ConceptTable
from learner import learn_step

# Create object
obj = Obj(shape_name='box', color_rgb=COLORS_RGB['blue'],
          occ=np.array(SHAPES['box'], dtype=np.float32))

# Encode scene
X, mask = encode_scene(Scene(regions=[obj, None, None, None]))

# Learn from utterance
table = ConceptTable(d=12)
learn_step(X, mask, k=1, tokens=['blue', 'box'], table=table)
```

### RSA Inference

```python
from rsa import infer_posterior

# Test scene with 2 objects
X, mask = encode_scene(Scene(regions=[blue_obj, red_obj, None, None]))

# Infer: who is "blue"?
posterior = infer_posterior(X, mask, ['blue'], table)
# posterior[0] = P(region 0 | "blue")
# posterior[1] = P(region 1 | "blue")
```

### Multi-Intent Inference

```python
from rsa import infer_posterior_multi_intent

# "1 blue, 1 solid" in scene with 4 objects
intents = [(['blue'], 1), (['solid'], 1)]
result = infer_posterior_multi_intent(X, mask, intents, table, use_rsa=True)

# result: Dict[assignment, probability]
# e.g., ((0,), (3,)): 0.85 means obj0 is blue, obj3 is solid
```

---

## Key Theoretical Contributions

### 1. Informativeness in Multi-Intent RSA

Standard RSA S1 uses raw L0 scores, but this fails for multi-intent because:

- It doesn't account for utterance ambiguity
- Overlap objects (matching multiple intents) get incorrectly boosted

**Solution**: Use normalized L0 posterior `P_L0(T | U)` instead of raw scores.

### 2. Automatic Alternative Generation

Single-token utterances like "green" have trivial alternatives `Alt = {('green',)}`, causing S1 = 1 for all targets.

**Solution**:

- `auto_alt_from_table=True`: Add all known tokens as alternatives
- **Dynamic `include_empty_alt`**: Automatically disabled when:
  - ME scenario: novel word + known concepts
  - Scalar scenario: ≥2 known concepts competing

### 3. Scalar Implicature (v3.3)

When speaker says "blue" but could have said "solid", listener infers target doesn't match "solid".

**Example:**

```
Scene: red_box, blue_box (no name), blue_solid (has "solid" name)
Query: "blue"
L0:  blue_box=68.5%, blue_solid=31.5%
RSA: blue_box=98.2%, blue_solid=1.8%  ← Scalar Implicature!
```

### 4. Exactness Constraint

Ensures "2 blue" means exactly 2 blue objects, not 3 or more.

```
exactness_penalty = γ · Σᵤ (soft_count - expected_k)²
```

---

## Active Learning (`active_learner.py`)

主动学习模块实现了**元认知**能力：模型不仅能判断对象是什么，还能识别自己"不知道什么"。

### 熟知度分数 (Domain-Aware Z-score)

基于合法随机物体采样的 Z-score 判定：

$$Z(x) = \frac{\max_{c} \text{Score}(x, c) - \mu_{bg}}{\sigma_{bg}}$$

| 符号                 | 含义                                     |
| -------------------- | ---------------------------------------- |
| $\text{Score}(x, c)$ | log_inc 包含度分数（取所有概念中最大值） |
| $\mu_{bg}$           | 随机合法物体的平均最大得分               |
| $\sigma_{bg}$        | 随机合法物体的得分标准差                 |

**逻辑**: Z-score 衡量“当前对象与概念的匹配度比随机物体好多少个标准差”。

### 判定机制

同时检查 **区分度** 和 **证据强度**（默认 $\theta = 3.0$，$\kappa_{\min} = 2.0$）：

- **Unknown**: 如果 $Z(x) \leq \theta$ **或** $\kappa < \kappa_{\min}$，生成临时概念 `concept_xxxxx`
- **Known**: 如果 $Z(x) > \theta$ **且** $\kappa \geq \kappa_{\min}$，返回最佳匹配概念

### Ask-Answer 协议

```python
# 1. Ask: 模型判断对象
result = rsa.ask(["red box", "pink solid", "", ""], position=1)
# result.is_known = False
# result.provisional_token = "concept_a1b2c"

# 2. Answer: 用户反馈
rsa.answer(["red box", "pink solid", "", ""], position=1, utterance="1 pink")
# → 创建新概念 'pink'，替换 'concept_a1b2c'
```

### Self-Train: 自监督学习

```python
stats = rsa.self_train([
    ["pink solid", "pink box", "", ""],
    ["cyan l", "cyan t", "", ""]
])
```

模型用 Ask 结果作为伪标签训练自己，实现无监督聚类。

### Reflect: 概念合并

```python
result = rsa.reflect(z_threshold=1.0, allow_merge_trained=False, verbose=True)
```

**反思机制**：检测概念表中相似的概念并进行合并。基于 KL 散度的 Z-score 判断概念相似性：

$$Z = \frac{\mu_{baseline} - KL_{pair}}{\sigma_{baseline}}$$

如果 $Z > \theta$，说明两个概念非常相似（KL 散度远低于随机概念对的平均值），应该合并。

**合并规则:**

| 情况                | `allow_merge_trained=False` | `allow_merge_trained=True`          |
| ------------------- | --------------------------- | ----------------------------------- |
| 临时概念 + 训练概念 | 用训练概念名                | 用训练概念名                        |
| 两个训练概念        | 不合并                      | 合并 → `frozenset({'red', 'blue'})` |
| 两个临时概念        | 随机选一个                  | 随机选一个                          |

**返回值:**

```python
{
    "pairs_checked": 6,      # 检查的概念对数量
    "merges": [              # 合并记录
        ("concept_a1b2c", "concept_d3e4f", "concept_a1b2c"),  # (token_a, token_b, kept)
    ],
    "skipped_both_trained": 0  # 跳过的训练概念对数量
}
```

**Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `novelty_threshold` | 3.0 | Z-score 高于此值视为已知 |
| `min_kappa_known` | 2.0 | κ 至少达到此值才判为已知 |
| `z_threshold` | 1.0 | Reflect Z-score 阈值 |
| `allow_merge_trained` | False | 是否允许合并训练概念 |

---

## Zero-Shot Learning (零样本学习)

基于**语义-视觉同构性假设**：如果语言空间中 $\vec{Orange} \approx 0.5 \cdot \vec{Red} + 0.5 \cdot \vec{Yellow}$，那么感知空间中 $\mu_{Orange}$ 也应该在 $\mu_{Red}$ 和 $\mu_{Yellow}$ 之间。

### 核心方法：Gaussian Process Regression

用 GP 回归从 embedding 空间到感知空间的映射，每个感知维度一个独立 GP，共享 RBF 核超参：

$$\mu_d(v) \sim \mathcal{GP}(0, k(v, v')), \quad k(v, v') = \sigma^2_f \exp\left(-\frac{\|v - v'\|^2}{2l^2}\right)$$

给定锚点 $\{(v_j, \mu_{j,d})\}$，对新点 $v^*$ 的预测：

$$m^*_d = \mathbf{k}_*^\top (K + \sigma^2_n I)^{-1} \mathbf{y}_d$$
$$s^2_{\text{epi}} = k_{**} - \mathbf{k}_*^\top (K + \sigma^2_n I)^{-1} \mathbf{k}_*$$

**最终方差 (epistemic + aleatoric 可分解)**：

$$\sigma^2_{\text{new},d} = \underbrace{s^2_{\text{epi}}(v^*)}_{\text{GP预测方差}} + \underbrace{\sigma^2_{\text{ale},d}(v^*)}_{\text{锚点方差插值}} + \epsilon_{\min}$$

| 项                      | 含义                 | 行为                            |
| ----------------------- | -------------------- | ------------------------------- |
| $s^2_{\text{epi}}$      | Epistemic 不确定性   | 远离锚点时自动增大              |
| $\sigma^2_{\text{ale}}$ | Aleatoric 不确定性   | 锚点自身方差的 GP-weighted 插值 |
| $\epsilon_{\min}$       | 方差下限 (var_floor) | 防止方差坍缩                    |

> **vs 原方法 (softmax 核插值)**：原方法 `synthesize_concept` 用 softmax 权重做线性插值，方差仅乘以固定缩放系数 $\gamma$，缺少 between-means 项，且外推时不确定性不增长。GP 版本解决了这两个问题。

### 使用流程

```python
import numpy as np
from rsa_helper import RSAHelper

rsa = RSAHelper()

# 1. 训练基础概念 (视觉接地)
rsa.train(["red box", "", "", ""], "1 red")
rsa.train(["yellow box", "", "", ""], "1 yellow")
rsa.train(["blue box", "", "", ""], "1 blue")

# 2. 注入语义向量
rsa.add_embedding("red", np.array([1.0, 0.0, 0.0]))
rsa.add_embedding("yellow", np.array([0.0, 1.0, 0.0]))
rsa.add_embedding("blue", np.array([0.0, 0.0, 1.0]))

# 3. GP 零样本合成 (推荐)
vec_orange = np.array([0.7, 0.7, 0.0])
rsa.synthesize_concept_gp("orange", vec_orange)
# 输出:
# GP epistemic σ²: 0.0099
# Aleatoric σ² (mean): 0.4194
# Total σ² (mean): 0.4393
# +0.495 × red, +0.495 × yellow, +0.010 × blue

# 4. 直接用新概念推理!
probs = rsa.infer(["orange box", "red box", "yellow box", ""], "1 orange")
# probs[0] 应该最高
```

### 核心方法

| 方法                                   | 描述                                 |
| -------------------------------------- | ------------------------------------ |
| `add_embedding(token, vec)`            | 给已学概念注入语义向量               |
| `synthesize_concept_gp(token, vec)`    | **GP 回归合成** (epistemic 不确定性) |
| `synthesize_concept(token, vec, temp)` | 原方法: softmax 核插值 (轻量)        |
| `grounded_concepts()`                  | 返回所有有 embedding 的概念列表      |

### Hyperparameters

**GP 方法** (`synthesize_concept_gp`):

| Parameter      | Default | Description      |
| -------------- | ------- | ---------------- |
| `length_scale` | 0.5     | RBF 核长度尺度 l |
| `signal_var`   | 1.0     | 信号方差 σ²_f    |
| `noise_var`    | 0.01    | 观测噪声 σ²_n    |
| `var_floor`    | 0.01    | 最小方差 ε_min   |

**原方法** (`synthesize_concept`):

| Parameter           | Default | Description                      |
| ------------------- | ------- | -------------------------------- |
| `temp`              | 0.1     | Softmax 温度 (0.01=NN, 1.0=平均) |
| `uncertainty_scale` | 1.2     | 方差放大系数 γ                   |

---

## Sleep (记忆衰减)

基于 **Jost's Law (乔斯定律)**：_如果两个记忆强度相同，较老的那个遗忘得更慢。_

实现短时记忆 (STM) 与长时记忆 (LTM) 的动态区分。

### 使用方法

```python
# 训练一批数据
rsa.train(...)
rsa.self_train(...)

# 睡觉整理记忆 (建议每个 Episode 结束后调用)
rsa.sleep(base_rate=0.3)

# 噪音概念被遗忘，核心概念保留
```

### 衰减公式

$$\lambda(t) = \frac{\alpha}{1 + \beta \cdot t}$$

- `t` = decay_count (记忆年龄)
- `α` = base_rate (初始衰减率)
- `β` = stability (稳固系数)

### 衰减过程

| 过程                | 公式               | 效果                      |
| ------------------- | ------------------ | ------------------------- |
| 精度丢失 (Blurring) | `var *= (1 + λ)`   | 记忆变模糊                |
| κ 保持不变          | —                  | 观测计数是事实，不会遗忘  |
| 年龄递增            | `decay_count += 1` | 越老衰减越慢 (Jost's Law) |

> **设计决策**: κ 是“见过多少次”的统计量，不应随时间消失。遗忘体现为方差膨胀（记忆变模糊），
> 当不确定性超过阈值时概念被彻底清除 (pruning)。

### Hyperparameters

| Parameter         | Default | Description                               |
| ----------------- | ------- | ----------------------------------------- |
| `base_rate` (α)   | 0.3     | 新概念每次方差膨胀 30%                    |
| `stability` (β)   | 1.0     | 越大→概念越快进入长时记忆                 |
| `prune_threshold` | 50.0    | 平均方差超过此值时删除概念 (方差初始≈1.0) |

---

## Physics Engine (物理引擎)

基于 RSA 的**神经符号世界模型**，学习条件物理规则。

### 核心思想

将物理规则编码为 **72D 联合高斯概念** (RGBI, 与 Pixel Motor 统一)：

- **前36维 (Context)**: 当前状态 RGBI - 什么条件下规则生效
- **后36维 (Effect)**: 状态变化 RGBI - 规则的效果

$$\mathbf{x}_{rule} = [S_t, \Delta S]$$

预测时，通过 **后验加权混合 (Bayesian Model Averaging)** 或 **MAP/采样** 生成下一状态：

$$w_k = \text{softmax}\big(S_k / \tau\big), \quad S_k = \text{log\_inc}(\mathbf{s}_t, \mathbf{C}^{(k)}_{\text{ctx}})$$

> **注意**: 这是后验加权混合 (BMA)，不是精度加权混合 (PoE)。精度加权组合参见 Pixel Motor System。

### 推理模式

| mode       | 公式                                                | 适用场景                 |
| ---------- | --------------------------------------------------- | ------------------------ |
| `"mean"`   | $\Delta\hat{s} = \sum_k w_k \mu^{(k)}_{\text{eff}}$ | 平滑预测，多步模拟       |
| `"map"`    | $\Delta\hat{s} = \mu^{(k^*)}_{\text{eff}}$          | 确定性预测，避免混合鬼畜 |
| `"sample"` | $\Delta \sim \mathcal{N}(\mu^{(k)}, \Sigma^{(k)})$  | 随机模拟，不确定性探索   |

### 使用方法

```python
from rsa_action import PhysicsEngine, PhysicsGrid

engine = PhysicsEngine(sigma=0.5, temp=0.1)

# 学习重力规则：红+绿 → 红下落
grid_t = PhysicsGrid.from_colors({(0,1): 'red', (2,1): 'green'})
grid_next = PhysicsGrid.from_colors({(1,1): 'red', (2,1): 'green'})
engine.learn(grid_t, grid_next, "gravity")

# 学习悬浮规则：红+蓝 → 红上升
grid_t = PhysicsGrid.from_colors({(2,1): 'red', (1,2): 'blue'})
grid_next = PhysicsGrid.from_colors({(1,1): 'red', (1,2): 'blue'})
engine.learn(grid_t, grid_next, "levitate")

# 预测：三种模式
pred, info = engine.predict(new_grid, mode="mean")   # BMA
pred, info = engine.predict(new_grid, mode="map")    # Top-1
pred, info = engine.predict(new_grid, mode="sample") # 采样
# info['weights'] 显示每个规则的权重
# info['chosen_rule'] 显示被选中的规则 (map/sample 模式)
```

### 核心方法

| 方法                                  | 描述                          |
| ------------------------------------- | ----------------------------- |
| `learn(grid_t, grid_next, rule_name)` | 学习一个物理规则              |
| `predict(grid_t, mode)`               | 预测下一状态 (BMA/MAP/Sample) |
| `simulate(grid_t, steps)`             | 多步模拟                      |
| `laws()`                              | 列出已学规则                  |

### Hyperparameters

| Parameter | Default | Description                         |
| --------- | ------- | ----------------------------------- |
| `sigma`   | 0.5     | 状态编码的高斯模糊                  |
| `temp`    | 0.1     | 规则选择锐度 (越小越确定)           |
| `rho`     | 0.0     | Effect不确定性正则 (偏好低方差规则) |
| `mode`    | `mean`  | 推理模式: `mean`/`map`/`sample`     |

---

## Pixel Motor System (像素运动系统)

36维 RGBI 像素空间的**零样本运动组合**系统。每个 cell 用 4 通道表示：`[R, G, B, I]`，其中 `I = mean(R, G, B)` 是颜色无关的亮度/结构信息。

### 核心原理：高斯专家乘积

$$\mu_{new} = \frac{\sum \mu_i \cdot prec_i}{\sum prec_i}, \quad prec = 1/\sigma^2$$

- **方差小** = 精度高 = 话语权大 = "否决权"
- 颜色概念学习"什么颜色"，位置概念学习"哪个位置"
- 组合时自动形成"约束的交集"

### 使用方法

```python
from rsa_action_pixel import PixelMotorSystem

agent = PixelMotorSystem()

# 牙牙学语
agent.babble()

# 零样本组合
grid = agent.imagine(['cmd_red', 'cmd_pos_4'], visualize=True)

# 检查内部表示
agent.inspect('cmd_red', amplify_factor=5.0)
```

### 核心方法

| 方法                | 描述             |
| ------------------- | ---------------- |
| `babble()`          | 运动牙牙学语训练 |
| `imagine(commands)` | 高斯专家乘积合成 |
| `inspect(token)`    | 检查概念内部表示 |

---

## Planning Agent (规划代理)

基于模型的**具身规划**系统，使用层级贝叶斯实现跨颜色泛化。

### 核心机制：层级贝叶斯 (Hierarchical Bayes)

将每种颜色视为一个 task，共享结构先验：

$$\mu_{\text{eff},c}^{(k)} = \mu_{\text{shared}}^{(k)} + \delta_c^{(k)}, \quad \delta_c \sim \mathcal{N}(0, \sigma^2_\delta I)$$

| 组件                  | 作用                               | 更新方式                            |
| --------------------- | ---------------------------------- | ----------------------------------- |
| $\mu_{\text{shared}}$ | 跨颜色共享的结构原型（"哪里变亮"） | 所有颜色样本 Welford 更新           |
| $\delta_c$            | 颜色特定残差（通常很小）           | 残差 = raw - μ_shared，强 shrinkage |
| $\sigma^2_\delta$     | 残差先验方差（默认 0.01）          | 越小 = shrinkage 越强               |

**优势（vs 认知白化）：**

- 不丢失颜色信息 — 残差捕获颜色相关细微差异
- 明确概率语义 — 等价于 mixed-effects 模型
- few-shot 泛化 — 新颜色 1-3 个样本即可适配

### 使用方法

```python
from rsa_plan import PlanningAgent, make_scene

agent = PlanningAgent(sigma_delta=0.01)
agent.babble()

# 用红/黄/绿教 "grow" (学习 μ_shared + δ_red/δ_yellow/δ_green)
for _ in range(20):
    for color in [[1,0,0], [1,1,0], [0,1,0]]:
        agent.learn_dynamic_concept(
            make_scene(color, 'dot'),
            make_scene(color, 'cross'), "grow")

# Few-shot: 蓝色 3 个样本 → 学习 δ_blue
for _ in range(3):
    agent.learn_dynamic_concept(
        make_scene([0,0,1], 'dot'),
        make_scene([0,0,1], 'cross'), "grow")

# 规划 (自动选择 few-shot 或 zero-shot via shared prior)
history = agent.ask_to_show(make_scene([0,0,1], 'dot'), "grow")
agent.visualize_plan(history)
```

### 核心方法

| 方法                                      | 描述                              |
| ----------------------------------------- | --------------------------------- |
| `learn_dynamic_concept(start, end, name)` | 层级贝叶斯学习 (shared + δ_c)     |
| `learn_visual_concept(grid, name)`        | 学习视觉概念                      |
| `compose_concepts(["a", "b"], "ab")`      | 高斯乘积组合                      |
| `generate(concept, color_cmd)`            | 从空白生成                        |
| `ask_to_show(start, concept)`             | 逆向规划 (uses effective concept) |

### Hyperparameters

| Parameter     | Default | Description                        |
| ------------- | ------- | ---------------------------------- |
| `sigma_delta` | 0.01    | 残差先验方差 σ²_δ (shrinkage 强度) |

---

## Generative Concept Composition (生成式概念组合)

通过**高斯乘积**实现零样本生成：从未见过的概念组合。

### 核心原理

$$\mu_{new} = \frac{\sum \mu_i \cdot prec_i}{\sum prec_i}, \quad prec = 1/\sigma^2$$

- 分别学习 "blue" (各种形状) 和 "box" (各种颜色)
- 组合时，**方差小的维度主导**
- "blue" 主导颜色，"box" 主导形状
- 结果：从未见过的 "blue box"

### 使用方法

```python
from rsa_plan import PlanningAgent, make_scene

agent = PlanningAgent()
agent.babble()

# 学习 "blue": 多种形状，只用蓝色
for shape in ['cross', 'l', 'corner', 'hbar', 'vbar']:
    agent.learn_visual_concept(make_scene([0,0,1], shape), 'blue')

# 学习 "box": 多种颜色，只用 box 形状
for color in [[1,0,0], [0,1,0], [1,1,0]]:
    agent.learn_visual_concept(make_scene(color, 'box'), 'box', whitening=True)

# 零样本组合
agent.compose_concepts(['blue', 'box'], 'blue_box')

# 从空白画布生成
history = agent.generate('blue_box', 'cmd_blue')
agent.visualize_plan(history)
```

## 使用指南

详见 [USAGE_GUIDE.md](USAGE_GUIDE.md)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_rsa.py::TestSingleConceptGeneralization -v
```

**Test Coverage:**

- 137+ tests covering all modules
- Concept learning convergence
- RSA mutual exclusivity
- Multi-intent overlap detection
- Cross-color generalization

---

## Hyperparameter Summary

| Module       | Parameter             | Default        | Description                          |
| ------------ | --------------------- | -------------- | ------------------------------------ |
| scoring      | `eps_obj`             | 1e-4           | Object observation variance          |
| scoring      | `tau`                 | 1.0            | KL temperature                       |
| rsa          | `alpha`               | 5.0            | Speaker rationality                  |
| rsa          | `beta`                | 0.1            | Volume penalty (size principle)      |
| rsa          | `lam`                 | 0.0            | Length cost                          |
| rsa          | `include_empty_alt`   | True           | Include ∅ in alternatives            |
| rsa          | `auto_alt_from_table` | True           | Use all known tokens as alternatives |
| multi-intent | `exactness_gamma`     | 2.0            | Exactness constraint strength        |
| multi-intent | `exactness_mode`      | `"soft_count"` | `"soft_count"` or `"pure_rsa"`       |
| multi-intent | `eta`                 | 0.02           | Weak set-size prior (pure_rsa only)  |
| learner      | `var_floor`           | 1e-6           | Variance floor to prevent collapse   |
| concepts     | `kappa0`              | 0.5            | Initial pseudo-count                 |

---

## MLC: Neuro-Symbolic Compositional Learner (`ns_learner/`)

基于 **Bayesian Program Learning (BPL)** 的组合泛化模型，用于解决 Meta-Learning Compositional (MLC) 任务。模型通过少量示例（SUPPORT）学习词到颜色的映射规则，然后泛化到未见过的组合（QUERY）。

### 项目结构

```
ns_learner/
├── ns_concept.py      # NeuroConcept: 每词概率模型 (Role + Repeat + Emission)
├── ns_primitives.py   # 栈机器引擎: 5 种原语操作 + 不可变栈
├── ns_inference.py    # Beam Search 推理: Soft-EM E-step
├── ns_ast.py          # AST-Hybrid 解码器: 层次化作用域处理
├── ns_learner.py      # NSLearner: BPL Agent (两层架构)
├── ns_colors.py       # 颜色空间: RGB / CIELAB 调色板 + 噪声工具
└── gaussian.py        # 高斯数学: KL 散度, log-det
```

---

### 1. NeuroConcept — 每词概率模型 (`ns_concept.py`)

每个词 $w$ 携带三个独立的概率分布：

#### 1.1 角色分布 (Role Prior)

每个词属于哪种语法角色，由 **Dirichlet-Multinomial** 后验建模：

$$P(\text{role} \mid w) = \frac{\alpha[\text{role}] + \text{count}_w[\text{role}]}{\sum_{r \in \text{ROLES}} (\alpha[r] + \text{count}_w[r])}$$

| 角色           | 语义                     | 示例                     |
| -------------- | ------------------------ | ------------------------ |
| `EMIT`         | 叶节点名词，发射一个颜色 | `DAX → YELLOW`           |
| `REPEAT`       | 后缀一元操作，重复 k 次  | `thrice → ×3`            |
| `SWAP_INFIX`   | 中缀二元操作，交换顺序   | `A after B → B, A`       |
| `CONCAT_INFIX` | 中缀二元操作，拼接       | `A lug B → A, B`         |
| `OVER_INFIX`   | 中缀二元操作，包围       | `A surround B → A, B, A` |

先验 $\alpha$ 是可元学习的超参数（默认 EMIT 偏高）。

#### 1.2 重复次数分布 (Repeat Count)

$$P(k \mid w) = \frac{\gamma[k] + \text{count}_w[k]}{\sum_{k' \in \{1,2,3,4\}} (\gamma[k'] + \text{count}_w[k'])}$$

仅在 `role = REPEAT` 时使用。先验 $\gamma$ 同样可元学习。

#### 1.3 发射分布 (Emission Model)

这是模型的核心创新点。支持三种模式：

**模式 A: 离散 Dirichlet（`delta ≠ None`）**

```
P(color | w) = (δ[c] + count_w[c]) / Σ(δ[c'] + count_w[c'])
```

输入向量先通过 `vec_to_color()` 量化为离散颜色名，再查计数表。

**模式 B: NIG/KL 连续（`delta = None, gauss = False`）**

基于 Normal-Inverse-Gamma 后验的 KL 散度评分：

$$\text{score}(x, w) = -\frac{1}{\tau} D_{KL}\big(\mathcal{N}(x, \varepsilon I) \;\|\; \mathcal{N}(\mu_{\text{post}}, \Sigma_{\text{post}})\big)$$

其中后验参数由 NIG 充分统计量计算：

$$\mu_{\text{post}} = \frac{\kappa_0 \mu_0 + \sum w_i x_i}{\kappa_0 + \sum w_i}, \quad \sigma^2_{\text{post}} = \frac{2\beta_0 + \text{SSE}}{2\alpha_0 + \sum w_i}$$

> **问题**: KL 中的 $\log(\sigma^2_{\text{post}} / \varepsilon)$ 项在高维或大方差空间（如 CIELAB）中过大，
> 导致发射评分（~-6）远低于角色先验（~-0.7），使所有词被错误分配为 REPEAT。

**模式 C: 高斯对数似然（`gauss = True`，推荐用于连续空间）**

$$\text{score}(x, w) = \frac{1}{\tau} \log \mathcal{N}(x \mid \mu_{\text{post}}, \sigma^2_{\text{post}})$$

$$= -\frac{1}{2\tau} \sum_d \left[\log(2\pi \sigma^2_d) + \frac{(x_d - \mu_d)^2}{\sigma^2_d}\right]$$

> **优势**: 评分在 ~-0.2（匹配）到 ~-3（不匹配）范围，与角色先验量级兼容。

#### 1.4 充分统计量更新 (Soft-EM M-step)

每次 trace 步骤按后验权重 $w$ 累加：

```python
role_counts[role]    += w          # Dirichlet 角色计数
emit_stats['sum_w']  += w          # NIG: Σw
emit_stats['sum_wx'] += w * vec    # NIG: Σw·x
emit_stats['sum_wx2']+= w * vec²   # NIG: Σw·x²
color_counts[color]  += w          # Dirichlet 颜色计数
```

---

### 2. 栈机器执行引擎 (`ns_primitives.py`)

程序 trace 是一系列操作（每个词对应一个），在不可变栈上执行：

| 操作              | 栈变化                            | 输出                 |
| ----------------- | --------------------------------- | -------------------- |
| `EMIT(μ)`         | push `[μ]`                        | 名词发射一个颜色向量 |
| `REPEAT(k, a)`    | pop `a` items → X, push X·k       | 重复 top-of-stack    |
| `SWAP_INFIX(B)`   | pop A, consume B → push `B, A`    | 后项在前             |
| `CONCAT_INFIX(B)` | pop A, consume B → push `A, B`    | 前项在前             |
| `OVER_INFIX(B)`   | pop A, consume B → push `A, B, A` | 包围                 |

其中 `a`（arity）是**潜在变量**——决定从栈上取多少项作为操作数。这使得变长表达式绑定成为可能。

---

### 3. NIG 先验参数 (`NIGParams`)

| 参数     | 默认值     | 含义                             |
| -------- | ---------- | -------------------------------- |
| `mu0`    | `1/d` 向量 | 先验均值（均匀）                 |
| `kappa0` | 0.1        | 先验强度（越小越容易被数据覆盖） |
| `alpha0` | 1.0        | 方差先验自由度                   |
| `beta0`  | 1.0        | 方差先验尺度                     |

---

### 4. 两种解码器

#### 4.1 Stack-machine 解码器 (`ns_inference.py`)

从左到右扫描输入词序列，每个词选择角色并在栈上执行。Beam Search 维护 top-K 部分 trace。

**评分函数**:

$$\text{score}(\text{trace}) = \sum_{i} \big[\log P(\text{role}_i \mid w_i) + \log P(\text{emit}_i \mid w_i) + \log P(\text{target match})\big]$$

**限制**: 无法正确处理嵌套作用域，如 `DAX surround 3 after 1 thrice`，其中 `after` 的操作数 `1 thrice` 跨越多个词。

#### 4.2 AST-Hybrid 解码器 (`ns_ast.py`)

构建层次化 AST (Abstract Syntax Tree)，正确处理作用域绑定。

**结构**:

```
Input: "DAX surround 3 after 1 thrice"

AST:
  OVER_INFIX(surround)
  ├── left: EMIT(DAX)
  └── right: SWAP_INFIX(after)
              ├── left: EMIT(3)
              └── right: REPEAT(thrice, k=3)
                          └── arg: EMIT(1)
```

每个 INFIX 节点递归处理右子树，精确分割作用域。

**性能**: AST 89% vs Stack 70% （σ=0 无噪声，10 个 query）。

---

### 5. 两层元学习架构 (`NSLearner`)

#### 5.1 内层循环 (Inner Loop): Few-shot Soft-EM

给定一个 episode 的 SUPPORT，通过 3 轮 Soft-EM 学习词-角色-颜色映射：

1. **Bootstrap**: 识别简单 1:1 名词映射（单词→单颜色）
2. **E-step**: 对每个例子 beam search top-K traces
3. **M-step**: 按 softmax 权重累加充分统计量
4. **衰减**: 每轮将旧计数衰减 50%，防止早期错误固化

#### 5.2 外层循环 (Outer Loop): Empirical Bayes 元训练

在 100+ 背景 episode 上学习全局先验 $\Phi$：

$$\max_\Phi \sum_{\text{episodes}} \log P_\Phi(\text{QUERY} \mid \text{SUPPORT})$$

元学习的参数包括：

| 参数        | 含义           | 默认值                          |
| ----------- | -------------- | ------------------------------- |
| `alpha`     | Role 先验权重  | `{EMIT: 3.0, REPEAT: 1.0, ...}` |
| `nig`       | 颜色先验       | `mu0, kappa0, alpha0, beta0`    |
| `lam`       | 动作概率惩罚   | 0.3                             |
| `beta`      | 角色评分温度   | 2.0                             |
| `tau_span`  | Arity 跨度先验 | 0.5                             |
| `eps_obj`   | 观测方差       | 0.1                             |
| `rsa_alpha` | RSA 语用权重   | 0.5                             |
| `gauss`     | 是否用高斯似然 | False                           |

---

### 6. 颜色空间与噪声 (`ns_colors.py`)

#### Lab 编码

CIELAB 颜色空间归一化到 [0, 1]：

$$L^* \in [0, 100] \to [0, 1], \quad a^* \in [-128, 127] \to [0, 1], \quad b^* \in [-128, 127] \to [0, 1]$$

| SCAN 符号 | 颜色名 | RGB           | 归一化 Lab      |
| --------- | ------ | ------------- | --------------- |
| 1         | BLUE   | (0, 0, 255)   | [0.1, 0.9, 0.0] |
| 2         | RED    | (255, 0, 0)   | [0.4, 0.9, 0.8] |
| 3         | GREEN  | (0, 128, 0)   | [0.3, 0.1, 0.8] |
| DAX       | YELLOW | (255, 255, 0) | [0.9, 0.3, 1.0] |

#### 噪声模型

$$\text{noisy\_vec} = \text{clean\_lab} + \mathcal{N}(0, \sigma^2 I_3)$$

在归一化 Lab 空间中加入各向同性高斯噪声，模拟感知不确定性。

---

### 7. 发射模型对比：连续 vs 离散

两种模型的前提假设和能力存在根本差异：

|                | 连续模型 (Gaussian)             | 离散模型 (Dirichlet)                    |
| -------------- | ------------------------------- | --------------------------------------- |
| **需要知道**   | 向量维度 (d=3)                  | 完整的颜色词表 + 每个颜色的精确坐标     |
| **学的是**     | 每个词的 μ 和 σ²                | 每个词产生各颜色名的计数                |
| **面对新颜色** | ✅ 可以泛化（只要在连续空间中） | ❌ 无法处理（不在词表中的颜色无法识别） |
| **噪声处理**   | 软退化：噪声被后验方差吸收      | 硬判决：`nearest_color()` 量化后查表    |

> **关键洞察**: 离散模型的噪声鲁棒性来源于 Voronoi 边界——只要噪声不超过颜色间最小距离（~15-20 Lab 单位），`nearest_color()` 量化完全正确。一旦超过，产生不可恢复的硬错误。连续模型则平滑退化。

---

### 8. 噪声鲁棒性实验结果

**实验设置**: Mini-SCAN 任务，14 support + 10 query，CIELAB 3D，RSA OFF (α=0)，每 σ 重复 5 次取平均。

| σ   | Cont+Stack | Cont+AST | Disc+Stack | Disc+AST | Δ(C-D)   |
| --- | ---------- | -------- | ---------- | -------- | -------- |
| 0   | 70%        | **90%**  | 70%        | **90%**  | 0%       |
| 5   | 70%        | **90%**  | 70%        | **90%**  | 0%       |
| 10  | 70%        | **90%**  | 70%        | **90%**  | 0%       |
| 15  | 70%        | **90%**  | 70%        | **90%**  | 0%       |
| 20  | 56%        | 72%      | 56%        | 72%      | 0%       |
| 25  | 44%        | 56%      | 44%        | 56%      | 0%       |
| 30  | 24%        | 30%      | 36%        | 46%      | -16%     |
| 40  | 24%        | 30%      | 16%        | 18%      | **+12%** |
| 50  | 22%        | 18%      | 14%        | 14%      | **+4%**  |

**关键发现**:

1. **AST >> Stack**: 所有条件下 AST 解码器均优于 Stack (+20pp)
2. **σ ≤ 15**: 两种发射模型完全等价（90% 精度）
3. **σ = 20~25**: 下降区间，两者仍然一致
4. **σ ≥ 40**: 连续模型展现优势（+12%），因为高斯后验平滑吸收噪声
5. 唯一失败 query (`DAX surround 3 after 1 thrice`) 是三层嵌套的作用域解析问题，与发射模型无关

### 9. 拟人程度评估

确定性模型匹配人类行为的指标：

| 指标                               | Stack | AST   | 变化    |
| ---------------------------------- | ----- | ----- | ------- |
| M1. 准确率（vs 语法金标准）        | 70%   | 90%   | +20pp   |
| M2. 众数一致率（匹配人类多数回答） | 7/10  | 9/10  | +2      |
| M3. 人类一致率（匹配模型的人占比） | 60.8% | 74.7% | +13.9pp |
| M4. 难度相关性 (Pearson r)         | 0.628 | 0.464 | -0.16   |

**解读**: AST 模型 90% 精度略超人类整体 81%。Stack 难度相关性更高，因为它在人类也觉得难的题上出错（65-70% 人类正确率）。AST 消除了全部 substitution 错误（30%→0%），仅剩 1 个 order_error。

---

## References

- Frank, M. C., & Goodman, N. D. (2012). Predicting pragmatic reasoning in language games. _Science_.
- Goodman, N. D., & Frank, M. C. (2016). Pragmatic language interpretation as probabilistic inference. _Trends in Cognitive Sciences_.
- Lake, B. M. (2019). Compositional generalization through meta-sequence-to-sequence learning. _NeurIPS_.
- Lake, B. M., & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. _Nature_.
