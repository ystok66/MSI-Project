# 元认知 (Metacognition) 流程详解

## 概述

元认知能力使模型能够判断自己"知道什么"和"不知道什么"。

---

## 1. Known/Unknown 判定流程

当一个新对象输入时，系统执行以下步骤：

```
输入对象 x
    ↓
┌─────────────────────────────────────┐
│ Step 1: 编码                         │
│ x → 12维特征向量 [Lab(3) + Shape(9)] │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 2: 计算最佳匹配分数              │
│ 遍历所有概念 c ∈ C:                   │
│   score(x, c) = log_inc(x, μ_c, σ²_c)│
│ best_score = max(score)              │
│ best_token = argmax(score)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 3: 计算背景基准线                │
│ 采样 50 个随机合法物体                │
│ 计算它们在概念表中的最佳分数           │
│ μ_bg = mean(random_scores)           │
│ σ_bg = std(random_scores)            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 4: 计算 Z-score                 │
│ Z = (best_score - μ_bg) / σ_bg       │
│                                      │
│ 解释: Z 表示当前对象的得分比           │
│ "随机物体"高出多少个标准差             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 5: 判定                         │
│ if Z > novelty_threshold:            │
│     → KNOWN (返回 best_token)        │
│ else:                                │
│     → UNKNOWN (创建临时概念)          │
└─────────────────────────────────────┘
```

### 参数说明

| 参数                | 默认值 | 作用             |
| ------------------- | ------ | ---------------- |
| `novelty_threshold` | 0.2    | Z-score 阈值     |
| `match_threshold`   | -44.0  | (旧方法，已弃用) |

**novelty_threshold 越大**：越难被判定为 known → 更多新概念
**novelty_threshold 越小**：越容易被判定为 known → 概念合并更多

---

## 2. Train 的概念变化

`rsa.train(scene, description)` 执行**监督学习**。

### 流程

```python
rsa.train(["red box", "", "", ""], "1 red")
```

1. **编码** → 提取 "red box" 的 12 维特征 `x`
2. **解析标签** → `tokens=["red"]`, `k=1`
3. **更新概念表**：
   - 如果 "red" 不存在 → 创建新概念，初始化 `μ=x`, `κ=1`
   - 如果 "red" 已存在 → 在线更新 `μ`, `σ²`, `κ`

### 概念更新公式

```
κ' = κ + 1
μ' = μ + (x - μ) / κ'
M₂' = M₂ + (x - μ)(x - μ')
σ² = M₂' / κ' + ε
```

### 效果

- **κ (kappa)**: 观测计数增加
- **μ (mu)**: 均值向新样本方向移动
- **σ² (var)**: 方差根据样本分布调整

---

## 3. Self-Train 的概念变化

`rsa.self_train(scenes)` 执行**自监督学习**。

### 流程

```python
rsa.self_train([["red box", "", "", ""]])
```

对每个场景的每个对象：

```
┌─────────────────────────────────────┐
│ Step 1: Ask (自问)                   │
│ result = ask(scene, position)        │
│                                      │
│ 如果 KNOWN → token = best_token      │
│ 如果 UNKNOWN → token = concept_xxxxx │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 2: Answer (自答)                │
│ answer(scene, position, f"1 {token}")│
│                                      │
│ 用 Ask 的结果作为伪标签进行训练        │
└─────────────────────────────────────┘
```

### 与 Train 的区别

| 方面     | train()              | self_train()           |
| -------- | -------------------- | ---------------------- |
| 标签来源 | 用户提供             | 模型自己生成           |
| 概念命名 | 用户指定（如 "red"） | 可能是 "concept_xxxxx" |
| 适用场景 | 有监督学习           | 无监督聚类             |

### Self-Train 返回值

```python
stats = rsa.self_train(scenes)
# stats = {"known": 3, "new": 2, "empty": 4}
```

- `known`: 被判定为已知概念的数量
- `new`: 被判定为未知（创建临时概念）的数量
- `empty`: 空位置数量

---

## 4. Reflect 的概念变化

`rsa.reflect()` 执行**概念合并**。

### 流程

```
遍历所有概念对 (A, B):
    ↓
计算对称 KL 散度:
    KL = (KL(A||B) + KL(B||A)) / 2
    ↓
计算 Z-score:
    Z = (μ_baseline - KL) / σ_baseline
    ↓
如果 Z > z_threshold (概念相似):
    → 合并概念
```

### 合并规则

| 情况        | 结果                                   |
| ----------- | -------------------------------------- |
| 临时 + 训练 | 用训练概念名                           |
| 临时 + 临时 | 随机选一个                             |
| 训练 + 训练 | 不合并 (除非 allow_merge_trained=True) |

---

## 5. 完整示例

```python
from rsa_helper import RSAHelper

# 初始化
rsa = RSAHelper(novelty_threshold=1.0)

# 1. 训练基础概念
rsa.train(["red box", "", "", ""], "1 red")
rsa.train(["blue box", "", "", ""], "1 blue")
# 概念表: {"red": κ=1, "blue": κ=1}

# 2. 自监督学习
rsa.self_train([["red solid", "", "", ""]])
# Ask: Z > 1.0? → KNOWN (red)
# Answer: 巩固 red
# 概念表: {"red": κ=2, "blue": κ=1}

rsa.self_train([["pink box", "", "", ""]])
# Ask: Z < 1.0? → UNKNOWN
# Answer: 创建 concept_a1b2c
# 概念表: {"red": κ=2, "blue": κ=1, "concept_a1b2c": κ=1}

# 3. 反思合并
rsa.reflect(z_threshold=0.5)
# 如果 concept_a1b2c 与 red 相似 → 合并
# 概念表: {"red": κ=3, "blue": κ=1}
```

---

## 6. 调参建议

### 想要更多新概念（更细粒度的聚类）

```python
rsa = RSAHelper(novelty_threshold=2.0)  # 更高
```

### 想要更少新概念（更粗粒度的聚类）

```python
rsa = RSAHelper(novelty_threshold=0.0)  # 更低
```

### 颜色 vs 形状 的影响

- **颜色差异大**：容易产生不同概念
- **只有形状差异**：可能被合并到同一概念（因为颜色相同）

建议：如果想区分形状，使用不同颜色或使用 `train()` 明确标注。

---

## 7. Sleep (记忆衰减)

`rsa.sleep()` 执行**记忆衰减**，模拟 Jost's Law。

### 核心原理

**Jost's Law (乔斯定律)**：_如果两个记忆强度相同，较老的那个遗忘得更慢。_

```
衰减率: λ(t) = α / (1 + β * t)

t = decay_count (记忆年龄)
α = base_rate (初始衰减率)
β = stability (稳固系数)
```

### 双重衰减过程

1. **强度流失 (Fading)**: `κ *= (1 - λ)` — 概念变弱
2. **精度丢失 (Blurring)**: `var *= (1 + λ)` — 记忆变模糊

### 使用示例

```python
# 训练一批数据
rsa.train(...)
rsa.self_train(...)

# 睡觉整理记忆
rsa.sleep(base_rate=0.3)

# 噪音概念被遗忘，核心概念保留
```

### 参数说明

| 参数              | 默认值 | 作用                      |
| ----------------- | ------ | ------------------------- |
| `base_rate` (α)   | 0.3    | 新概念每次衰减 30%        |
| `stability` (β)   | 1.0    | 越大→概念越快进入长时记忆 |
| `prune_threshold` | 0.1    | κ 低于此值时删除概念      |

### 效果

| 场景                  | 结果                       |
| --------------------- | -------------------------- |
| 新噪音概念 (t=0)      | 剧烈衰减 (30%)，可能被删除 |
| 反复巩固的概念 (t=50) | 几乎不衰减 (<1%)，稳如磐石 |

这实现了 **短时记忆 (STM)** 和 **长时记忆 (LTM)** 的动态区分！
