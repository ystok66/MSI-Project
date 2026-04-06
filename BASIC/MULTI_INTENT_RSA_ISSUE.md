# Multi-Intent RSA 问题报告

## 测试案例: 案例 6 变体 (vbar/hbar 版本)

### 1. 输入

#### 1.1 训练数据

```
概念 "blue" 训练:
  - blue vbar → "1 blue"
  - blue hbar → "1 blue"
  - blue t → "1 blue"
  - blue l → "1 blue"

概念 "hbar" 训练:
  - blue hbar → "1 hbar"
  - red hbar → "1 hbar"
  - green hbar → "1 hbar"
  - yellow hbar → "1 hbar"
```

#### 1.2 测试场景

```
Region 0: blue vbar  (颜色=蓝, 形状=vbar)
Region 1: red hbar   (颜色=红, 形状=hbar)
Region 2: blue hbar  (颜色=蓝, 形状=hbar) ← 重叠对象!
Region 3: red vbar   (颜色=红, 形状=vbar)
```

#### 1.3 查询

```
"1 blue, 1 hbar"
意图: [(['blue'], 1), (['hbar'], 1)]
```

---

### 2. 预期输出

**最佳分配**: `blue_vbar + red_hbar`

**理由**:

- blue_vbar 是 "blue" 的最佳匹配 (蓝色 + vbar形状)
- red_hbar 是 "hbar" 的最佳匹配 (红色 + hbar形状)
- 两者不重叠，完美满足 "1 blue, 1 hbar"

**次优分配**: `blue_hbar + red_hbar` (使用重叠对象满足 blue)

**错误分配**: `red_vbar + red_hbar` (red_vbar 不是蓝色!)

**预期概率分布**:

| 分配                 | 预期 L0 | 预期 RSA        |
| -------------------- | ------- | --------------- |
| blue_vbar + red_hbar | >90%    | >90% (应该更高) |
| blue_hbar + red_hbar | ~5%     | <5% (惩罚重叠)  |
| red_vbar + red_hbar  | ~0%     | ~0% (颜色错误)  |

---

### 3. 实际输出

#### 3.1 L0 单对象分数

| 对象      | L0(blue)   | L0(hbar)   | 分析                     |
| --------- | ---------- | ---------- | ------------------------ |
| blue_vbar | **-41.03** | -50.35     | blue 匹配好, hbar 匹配差 |
| red_hbar  | -45.70     | **-36.91** | blue 匹配差, hbar 匹配好 |
| blue_hbar | -42.51     | -37.65     | 两者都中等 (重叠对象)    |
| red_vbar  | -44.22     | -49.61     | 两者都匹配差             |

#### 3.2 分配概率对比

| 分配 (blue + hbar)       | L0           | RSA          | 问题?             |
| ------------------------ | ------------ | ------------ | ----------------- |
| **blue_vbar + red_hbar** | **95.4%** ✅ | 43.2%        | RSA 降低了!       |
| blue_vbar + blue_hbar    | 4.3%         | 0%           | ✅                |
| blue_hbar + red_hbar     | 0.3%         | 0%           | ✅                |
| **red_vbar + red_hbar**  | **0%**       | **43.2%** ⚠️ | RSA 给了错误分配! |
| red_vbar + blue_hbar     | 0%           | 13.7%        | ⚠️                |

---

### 4. 问题分析

#### 4.1 核心问题

**L0 表现正确，RSA 表现错误！**

- L0 给 `blue_vbar + red_hbar` 95.4% - 正确!
- RSA 给 `blue_vbar + red_hbar` 只有 43.2%
- RSA 给 `red_vbar + red_hbar` 43.2% - 这是错误的!

#### 4.2 S1 分数分析

```
S1 scores (说话者使用 "1 blue, 1 hbar" 的概率):
  blue_vbar + red_hbar: S1=0.9697  ← 正确分配
  red_vbar + red_hbar:  S1=0.9697  ← 错误分配, 但 S1 相同!
```

**问题**: 两个分配的 S1 分数完全相同!

#### 4.3 根本原因

Multi-Intent RSA 的 S1 计算方式有问题:

```python
# 当前 S1 逻辑:
# S1(U | T) = P_L0(T | U) normalized across utterances
#
# 对于 red_vbar + red_hbar:
#   - "1 blue, 1 hbar": L0 很低, 但比其他话语好
#   - "1 hbar, 1 hbar": 不能用 (需要2个hbar)
#   - "2 blue": 不匹配
#   - "2 hbar": 不匹配
#
# 结果: 即使 L0 很低, S1 softmax 后还是高!
```

**S1 的问题**:

S1 比较的是 **"不同话语对同一个分配的描述能力"**，而不是 **"同一个话语对不同分配的区分能力"**。

当 alternative 话语都不适用时，即使当前话语 L0 很低，S1 也会给高概率。

#### 4.4 数学表述

```
L1(T | U) ∝ P(T) × S1(U | T)

问题:
  S1(U | T_correct) ≈ S1(U | T_wrong) = 0.97

  因此:
  L1(T_correct | U) ≈ L1(T_wrong | U)

  结果: RSA 无法区分正确和错误的分配!
```

---

### 5. 对比: 单意图 RSA vs 多意图 RSA

| 特性             | 单意图 RSA                       | 多意图 RSA                         |
| ---------------- | -------------------------------- | ---------------------------------- |
| S1 含义          | "说话者会用这个词描述这个对象吗" | "说话者会用这个话语描述这个分配吗" |
| Alternative 竞争 | 不同单词竞争描述同一对象         | 不同多词话语竞争描述同一分配       |
| 问题场景         | 有多个概念时工作良好             | 当 alternatives 都不适用时失效     |

**单意图 RSA 成功案例 (Scalar Implicature)**:

```
Scene: blue_box, blue_solid
Query: "blue"
Alternative: "solid"

S1(blue | blue_box) >> S1(blue | blue_solid)
因为 blue_solid 有更好的词 "solid" 可用
→ RSA 正确增强 blue_box
```

**多意图 RSA 失败案例**:

```
Scene: blue_vbar, red_hbar, blue_hbar, red_vbar
Query: "1 blue, 1 hbar"
Alternatives: "1 hbar, 1 blue", "2 blue", "2 hbar"

S1("1 blue, 1 hbar" | blue_vbar+red_hbar) ≈
S1("1 blue, 1 hbar" | red_vbar+red_hbar)

因为 alternatives 对两者都不适用
→ RSA 无法区分!
```

---

### 6. 潜在解决方案

#### 方案 A: 直接使用 L0 而非 RSA

在 multi-intent 场景中，L0 已经足够好 (95.4%)。可以：

```python
use_rsa=False  # 对于 multi-intent 场景
```

#### 方案 B: 修改 S1 normalization

不跨话语 normalize，而是跨分配 normalize：

```python
# 当前: S1(U | T) = P_L0(T | U) / Σ_U' P_L0(T | U')
# 改为: S1(U | T) = P_L0(T | U) / Σ_T' P_L0(T' | U)
```

#### 方案 C: 添加 L0 先验约束

在 L1 中保留更多 L0 信号：

```python
L1(T | U) ∝ P(T) × S1(U | T) × exp(γ × L0(T, U))
```

#### 方案 D: 改进 alternatives 生成

生成更多相关的 alternative 话语，使 S1 有更好的对比。

---

### 7. 结论

| 方面         | 评估                                            |
| ------------ | ----------------------------------------------- |
| **L0 表现**  | ✅ 优秀 (95.4% 给正确分配)                      |
| **RSA 表现** | ❌ 有问题 (给错误分配 43.2%)                    |
| **问题根因** | S1 normalization 在 alternatives 都不适用时失效 |
| **建议**     | 对于 multi-intent 场景，考虑直接使用 L0         |

---

_报告生成时间: 2026-02-01_
