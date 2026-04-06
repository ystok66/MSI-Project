# 新颖性检测方法报告

## 问题背景

在 Active Learning 中，我们需要判断一个新观测对象是"已知"还是"未知"。

---

## 当前方法：基准线比较

### 算法

1. **生成基准线**：随机采样 n=50 个特征向量，计算每个与概念表的最大 log 重合度，取平均

   ```
   baseline = mean([max_c(log_inc(x_random, c)) for _ in range(50)])
   ```

2. **计算对象分数**：当前对象与概念表的最大 log 重合度

   ```
   obj_score = max_c(log_inc(x, c))
   ```

3. **计算熟知度**：

   ```
   familiarity = obj_score - baseline
   ```

4. **判断**：
   - `familiarity > threshold` → **已知**
   - `familiarity <= threshold` → **未知**

### 测试结果

| 对象              | obj_score | familiarity | 状态 |
| ----------------- | --------- | ----------- | ---- |
| red_box (训练过)  | -43.90    | 27.25       | 已知 |
| pink_box (未训练) | -45.89    | 25.58       | ?    |
| random vector     | -56.15    | 13.60       | ?    |

### 问题

- 单个随机样本的分数可能**偶然高于**基准线平均值
- 导致 `random` 也得到正的 familiarity (13.60)
- 如果 threshold=0，`random` 也会被判定为"已知"（误判）

---

## 改进方法：基于标准差的阈值

### 算法

1. 生成 n 个随机样本的分数：`scores = [max_c(log_inc(x_i, c)) for i in range(n)]`
2. 计算均值和标准差：`μ = mean(scores)`, `σ = std(scores)`
3. 计算 z-score：
   ```
   z = (obj_score - μ) / σ
   ```
4. 判断：
   - `z > 2` (或 1.5) → **已知** (在 95% 置信区间外)
   - `z <= 2` → **未知**

### 优势

- **统计意义明确**：z > 2 意味着"比 97.5% 的随机对象都好"
- **自动适应**：阈值会随着概念表的丰富度自动调整
- **减少误判**：随机对象的 z-score 分布在 0 附近

### 数学表达

$$z = \frac{S(x) - \mu_{random}}{\sigma_{random}}$$

其中：

- $S(x) = \max_{c \in \mathcal{C}} \log P(x | c)$ 是对象的最大匹配分数
- $\mu_{random}, \sigma_{random}$ 是随机样本的统计量

---

## 代码实现建议

```python
def familiarity_score_zscore(self, x, n_samples=50):
    # 收集随机样本分数
    random_scores = []
    for _ in range(n_samples):
        x_rand = np.random.randn(self.table.d)
        score = max(log_inc(x_rand, c) for c in concepts)
        random_scores.append(score)

    # 计算统计量
    mu = np.mean(random_scores)
    sigma = np.std(random_scores)

    # 计算对象分数
    obj_score = max(log_inc(x, c) for c in concepts)

    # z-score
    z = (obj_score - mu) / max(sigma, 1e-8)

    return z, obj_score
```

---

## 选择建议

| 方法                | 优点                         | 缺点               |
| ------------------- | ---------------------------- | ------------------ |
| **当前方法** (差值) | 简单，计算快                 | 阈值需手动调整     |
| **z-score 方法**    | 统计意义明确，无需手动调阈值 | 需计算标准差，略慢 |

**推荐**：如果追求理论严谨性，使用 z-score 方法。如果追求简单，当前方法配合 `novelty_threshold=15` 也可以工作。
