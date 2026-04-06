# RSA Helper 使用指南

## 快速开始

```python
from rsa_helper import RSAHelper

# 初始化
rsa = RSAHelper(color_perturbation=0.15, alpha=5.0)
```

---

## 1. 训练概念

### 单个训练

```python
rsa.train(["blue box", "", "", ""], "1 blue")
```

### 批量训练

```python
rsa.train_batch([
    (["blue box", "", "", ""], "1 blue"),
    (["blue solid", "", "", ""], "1 blue"),
    (["red box", "", "", ""], "1 red"),
])
```

### 参数说明

- **scene**: `["颜色 形状", "", "", ""]` - 4个区域
- **description**: `"数量 词1 词2..."` (如 `"1 blue"`)
- **show**: 是否显示训练图 (默认 True)
- **perturb**: 是否添加颜色扰动 (默认使用初始化设置)

---

## 2. 推理

### 单意图

```python
probs = rsa.infer(["blue box", "red box", "", ""], "1 blue")
# probs = [0.998, 0.002, 0.0, 0.0]
```

### 多意图

```python
probs = rsa.infer(["blue box", "red solid", "", ""], "1 blue, 1 solid")
```

### 2x2 网格输出

```python
rsa.show_grid(["blue box", "red box", "", ""], "1 blue")
```

### L0 vs RSA 对比

```python
results = rsa.compare(["blue box", "red box", "", ""], "1 blue")
print(results["L0"])   # L0 概率
print(results["RSA"])  # RSA 概率
```

---

## 3. 可视化

```python
# 可视化场景
rsa.visualize(["blue box", "red solid", "green l", "yellow t"])

# 可视化推理概率
rsa.visualize_probs(["blue box", "red box", "", ""], "1 blue")
```

---

## 4. 主动学习 (Active Learning)

主动学习模块让模型能够：

- 区分已知 vs 未知对象
- 自动生成临时概念
- 通过用户反馈或自监督学习

### Ask - 询问对象是什么

使用 **Domain-Aware Z-score** 判断对象是否已知：

```python
result = rsa.ask(["red box", "pink solid", "", ""], position=1)

# result.is_known      - 是否是已知概念 (Z > 0.3 即为已知)
# result.best_token    - 最佳匹配概念
# result.familiarity   - Z-score (标准化显著性分数)
# result.provisional_token - 临时生成的概念 (如果未知)
# result.message       - 人类可读响应
```

**Z-score 解释**：

- `Z > 0.3`: 显著高于随机基准 → 已知
- `Z ≤ 0.3`: 与随机基准相差不大 → 未知

### Answer - 处理用户反馈

```python
# 确认已知概念
rsa.answer(["red box", "", "", ""], position=0, utterance="1 red")

# 命名新概念
rsa.answer(["red box", "pink solid", "", ""], position=1, utterance="1 pink")
```

### Self-Train - 自监督学习

模型用自己的 Ask 结果作为伪标签训练：

```python
stats = rsa.self_train([
    ["pink solid", "pink box", "", ""],
    ["cyan l", "cyan t", "", ""]
])
# stats = {"known": 2, "new": 2, "empty": 4}
```

**重要说明**：

- 每个位置独立处理，不会互相污染
- 已知对象会巩固对应概念
- 未知对象会自动创建新概念 (`concept_xxxxx`)
- 推荐用于探索包含未知颜色/形状的新场景

---

## 5. 状态查看

```python
rsa.status()      # 打印状态
rsa.concepts()    # 返回概念列表
rsa.reset()       # 重置模型
```

---

## 6. 可用颜色和形状

### 颜色

`red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `orange`, `purple`, `pink`, `brown`, `gray`, `white`, `black`

### 形状

`box`, `solid`, `l`, `l_90`, `l_180`, `l_270`, `t`, `t_90`, `t_180`, `t_270`, `s`, `s_90`, `z`, `z_90`, `hbar`, `vbar`, `cross`, `donut`

---

## 完整示例

```python
from rsa_helper import RSAHelper

# 初始化
rsa = RSAHelper(color_perturbation=0)

# 训练
rsa.train_batch([
    (["blue box", "", "", ""], "1 blue"),
    (["blue l", "", "", ""], "1 blue"),
    (["red box", "", "", ""], "1 red"),
    (["red solid", "", "", ""], "1 red"),
], show=False)

# 推理
probs = rsa.infer(["blue box", "red box", "", ""], "1 blue")
print(f"P(blue_box) = {probs[0]:.2%}")  # 99.8%

# 主动学习
result = rsa.ask(["blue box", "pink solid", "", ""], position=1)
if not result.is_known:
    rsa.answer(["blue box", "pink solid", "", ""], position=1, utterance="1 pink")
    print("新概念 'pink' 已添加")
```
