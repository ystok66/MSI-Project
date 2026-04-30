# Goal B 下一步实施指南（给 Antigravity）

## 0. 文档目标

这份文档的目标不是重复 Goal B 的负结果，而是把下一步工作收紧成：

1. 基于当前代码与公式，解释为什么现有机制更容易产生 `anti-synergy`
2. 给出 2 到 3 条可实现、可比较、低冗余的机制路线
3. 指明哪些模块应该保留、重构、迁移、或暂时不要碰
4. 设计最小实验矩阵与测试清单
5. 尽量减少超参数、公式和机制冗余
6. 兼顾鲁棒性、可泛化性，以及未来框架整合

本指南默认读者会直接修改 `cls_color_selection` 仓库代码。

---

## 1. 当前代码与公式：已经实现了什么

这一节只陈述现状，不做过强因果判断。

### 1.1 当前 `wrong update` 的真实公式

代码位置：

- `cls_color_selection/cls_color_selection/learner/feedback_update.py`

当前失败 confirm 的基本流程是：

1. 取 unconstrained beam：

```text
beam = {(s_k, π_k, Y_k)}_{k=1..K}
```

2. 由分数做 posterior：

```text
q_k = exp(s_k) / Σ_j exp(s_j)
```

3. 根据反馈构造 likelihood：

#### `wrong_only`

若 learner 提交为 `Ŷ`，则：

```text
P(F_wrong | Ŷ, Y_k) =
  ε_wrong,       if Y_k = Ŷ
  1 - ε_wrong,   otherwise
```

#### `wrong_positions`

对每个位置 `ℓ`：

```text
s_{k,ℓ} =
  1 - ε_eq,  if Y_{k,ℓ} = Ŷ_ℓ
  ε_eq,      otherwise
```

若 `mask_ℓ=True` 表示该位置被标记为正确，`mask_ℓ=False` 表示错误，则：

```text
log P(F_mask | Ŷ, Y_k, mask)
= Σ_{ℓ: mask_ℓ=True}  log s_{k,ℓ}
 + Σ_{ℓ: mask_ℓ=False} log (1 - s_{k,ℓ})
```

若位置来自 tutor assist，还会乘折扣 `rho_assist`：

```text
log P(...)
= Σ_ℓ w_ℓ · [mask_ℓ log s_{k,ℓ} + (1-mask_ℓ) log(1-s_{k,ℓ})]

w_ℓ =
  rho_assist, if assist_mask_ℓ=True
  1.0,        otherwise
```

4. 重加权 posterior：

```text
q'_k ∝ q_k · P(F | Ŷ, Y_k)
```

5. 对 concept library 做 differential M-step：

```text
Δθ ∝ Σ_k (q'_k - q_k) · stats(π_k)
```

更具体一点：

```text
Δ n_{w,r} = η_fb · Σ_k (q'_k - q_k) · C_{w,r}(π_k)
```

其中：

- `n_{w,r}` 是 word-role 的计数
- `C_{w,r}(π_k)` 是 trace `π_k` 中词 `w` 以 role `r` 出现的计数

EMIT / REPEAT 的统计也会一起更新。

### 1.2 当前 `correct update` 的真实公式

代码位置：

- `cls_color_selection/cls_color_selection/learner/correct_update.py`

当前最小版本 `apply_correct_answer(...)` 已经存在。
它的语义是：

1. 用 GT output 做 constrained beam search

```text
T_GT = {(s_k^+, π_k^+)}_{k=1..M}
```

其中每个 `π_k^+` 都是 GT-compatible trace。

2. 在 constrained trace 集合里做 posterior：

```text
q_k^+ = exp(s_k^+) / Σ_j exp(s_j^+)
```

3. 直接做 weighted positive M-step：

```text
Δθ_correct ∝ Σ_k q_k^+ · stats(π_k^+)
```

它和 `TutorLearnerModel.update_from_output(...)` 的语义一致：

- constrained E-step
- weighted M-step

但注意：当前版本仍然是“纯 GT positive update”，还没有显式读入 `wrong history`。

### 1.3 当前 grammar-only Goal B runner 的真实协议

代码位置：

- `cls_color_selection/tmp/run_goal_b.py`

它的条件集合是：

- `correct_only`
- `wrong1_correct`
- `wrong2_correct`
- `wrong3_correct`
- `correct_x2`
- `correct_x3`
- `correct_x4`
- `wrong1_only`
- `wrong2_only`
- `wrong3_only`

每个条件都从相同初始 learner 深拷贝开始。

当前 `wrong submissions` 的来源是：

- 直接从初始 beam 中取 top wrong outputs

也就是说，现有协议默认比较的是：

> “用最容易获得的 beam-top wrongs 做几次负更新，再做一次正更新”

而不是：

> “用结构上互补的信息型 wrong 序列做 conditioning”

这点很重要，因为它限制了 Goal B 当前 protocol 的可表达能力。

### 1.4 当前 Goal B 指标

代码位置：

- `cls_color_selection/cls_color_selection/tutor_api/goal_b_metrics.py`

当前 snapshot 指标是：

- `H_beam`
- `gt_rank`
- `gt_mass`
- `top1`
- `top1_score`
- `margin`
- `probe_acc`
- `probe_ll`

这些指标足够支持：

- narrowing 曲线
- matched-budget 比较
- correct marginal gain

但还不够支持：

- wrong sequence type 分析
- GT-compatible family 内部的结构收缩分析
- parameter-level blame 与 commit 诊断

---

## 2. 当前负结果应该怎样解释：稳妥版本

### 2.1 已经被结果支持的事实

以下事实可以直接从现有 grammar-only 结果支持：

1. matched-budget 下，`correct×N` 优于 `wrong×k + correct`
2. correct update 的边际增益随着 prior wrong 次数增加而下降
3. wrong 1 和 wrong 2 有一定 narrowing
   - `gt_mass` 上升
   - `H_beam` 下降
4. wrong 3 开始出现回退
5. probe 侧增益很弱，说明当前更新多为局部改进，而非稳定泛化

### 2.2 目前最合理、但仍需进一步验证的机制解释

当前结果最符合以下解释：

1. `wrong update` 与 `correct update` 目前都直接作用于同一个长期 concept library
2. `wrong update` 提供的大多是 output-level negative evidence，而不是 GT-compatible family 内的附加结构信息
3. 因此，future correct 往往需要先纠正 prior wrong 的一部分偏移，再进行正向 consolidation
4. 更关键的是，prior wrong history 目前没有显式进入 correct posterior

也就是说，现有机制下：

```text
wrong×k + correct
≈
先做几次弱但带偏移的 library update
再做一次 GT positive update
```

而不是：

```text
wrong×k + correct
≈
先积累结构约束
再让 GT-compatible posterior 在这些约束下更聚焦
```

这个区别就是 Goal B 成败的核心。

### 2.3 还不能说成“已证实”的地方

以下命题目前还只是工作假说，不应写成已证明：

- “wrong 只是污染长期 grammar”
- “wrong 完全没有为 correct 预处理空间”
- “QueryMemory 一定是正确的 state 落点”
- “trace_analysis.py 直接就能做 blame”

更稳的说法应该是：

- wrong 似乎同时带来一部分 narrowing 和一部分会削弱 later correct 的偏移
- 目前缺的是一种机制，让 wrong history 能以 GT-compatible family 内部可见的方式作用到 correct update

---

## 3. Goal B 要成立，必要条件是什么

这里给出最核心的一条判断：

> `wrong×k + correct` 想优于 `direct-correct`，前面的 wrong 必须提供 **correct 自身不包含的附加结构信息**，并且这些信息要在 GT-compatible trace family 内部起作用。

否则：

- direct-correct 已经用 GT 直接做 constrained posterior
- 它本身已经很强
- wrong 只是更低效地绕一圈

### 3.1 一个形式化写法

设 unconstrained posterior 为：

```text
q(π | x) ∝ exp(score(π; x, θ))
```

设 GT-compatible trace 集合为：

```text
Π_GT = {π : Y(π)=GT}
```

那么 direct-correct 的 posterior 可以写为：

```text
q_dir(π | x, GT)
∝ q(π | x) · 1[π ∈ Π_GT]
```

Goal B 成立要求 wrong history `D^-` 引入一个额外因子 `C(π; D^-)`，使得：

```text
q_goalB(π | x, GT, D^-)
∝ q(π | x) · 1[π ∈ Π_GT] · C(π; D^-)
```

并且 `C(π; D^-)` 不是常数，也不是 GT 本身已经完全提供的信息。

如果对所有 `π ∈ Π_GT` 都有：

```text
C(π; D^-) ≈ const
```

那么：

```text
q_goalB ≈ q_dir
```

Goal B 基本不可能赢。

### 3.2 因此需要满足的三条条件

1. `wrong` 与 `correct` 的角色要分离
2. `wrong history` 必须进入 final correct update
3. narrowed posterior 必须被转成长期 grammar 改善

第三条尤其关键。
如果 narrowing 只停留在 query-local state，不被 commit 到长期 grammar，eval 未必会提升。

---

## 4. 推荐的机制路线

本节给出 3 条路线，按优先级排序。

---

## 4.A 路线 A：Query-Local Wrong History + History-Conditioned Correct Commit

### 4.A.1 推荐级别

**推荐优先级：最高**

这是最符合 Goal B 精神、且对现有代码侵入可控的路线。

### 4.A.2 核心思想

把 `wrong` 的主要作用改成：

- 不直接强写长期 library
- 而是在 query-local state 里积累结构约束

然后 `correct` 不再是简单 GT positive update，而是：

- 先跑 GT-compatible constrained beam
- 再用 wrong history 对这些 GT-compatible traces 做 reweight
- 最后一次性 commit 到长期 grammar

### 4.A.3 建议的状态对象

不要直接往现有 `QueryMemory` 里塞大量 Goal-B-specific grammar state。

推荐新增：

`cls_color_selection/cls_color_selection/learner/goal_b_state.py`

定义：

```python
@dataclass
class WrongObservation:
    submitted: List[str]
    wrong_mask: List[bool]
    top1_before: List[str]
    beam_stats_before: dict
    trace_features: Optional[dict] = None

@dataclass
class GoalBProtocolState:
    words: List[str]
    ground_truth: List[str]
    wrong_history: List[WrongObservation] = field(default_factory=list)
```

理由：

- 更干净
- 不污染现有 online environment memory 抽象
- 适合 grammar-only runner
- 将来如果要接入在线 query loop，也可以再决定是否融合到 `QueryMemory`

### 4.A.4 建议的公式

#### 第一步：wrong history accumulation

wrong 不直接做长期参数更新，而是积累 query-local evidence：

```text
D^- = {(Ŷ_t, mask_t, aux_t)}_{t=1..k}
```

其中 `aux_t` 可选，包括：

- beam snapshot
- trace salience features
- top1 trace role map

#### 第二步：GT-compatible posterior with wrong conditioning

先取 GT-compatible traces：

```text
Π_GT = {π : Y(π)=GT}
```

定义一个 query-local compatibility score：

```text
C(π; D^-) = exp( λ_hist · S_hist(π; D^-) )
```

其中最小可做版本：

```text
S_hist(π; D^-)
= Σ_t [ λ_pos · A_pos(π; mask_t, Ŷ_t)
      - λ_neg · B_neg(π; mask_t, Ŷ_t) ]
```

为了减少超参数，建议第一版直接把 `λ_pos = λ_neg = 1`，只保留一个全局 `λ_hist`。

#### 第三步：第一版可行的 `A_pos / B_neg`

第一版不要上复杂 blame。
先用 position-level consistency：

对 wrong 观测 `t` 的每个位置 `ℓ`：

- 若 `mask_{t,ℓ}=True`，说明 learner 当时该位置已经对了
- 若 `mask_{t,ℓ}=False`，说明 learner 当时该位置错了

对 GT-compatible trace `π`，定义其在位置 `ℓ` 的解释颜色为 `Y_π,ℓ = GT_ℓ`，但我们真正关心的是该位置由哪个 structural choice 造成。

第一版可以只定义一个轻量 proxy：

```text
A_pos(π; mask_t, Ŷ_t)
= Σ_{ℓ: mask_{t,ℓ}=True} 1[Ŷ_{t,ℓ} = GT_ℓ]
```

```text
B_neg(π; mask_t, Ŷ_t)
= Σ_{ℓ: mask_{t,ℓ}=False} blame_proxy(π, ℓ, t)
```

其中 `blame_proxy` 的第一版不要太复杂。
推荐利用 `trace_analysis.py` 提供的 alignment / salience 作为特征来源，但不要把它当作现成 blame 模块。

可行第一版：

```text
blame_proxy(π, ℓ, t)
= T_ℓ(π) · 1[Ŷ_{t,ℓ} ≠ GT_ℓ]
```

其中 `T_ℓ` 来自 trace salience 或 position uncertainty 的近似。

如果想更保守，第一版甚至可以先不用 `trace_analysis.py`，
只做：

```text
blame_proxy(π, ℓ, t) = 1[Ŷ_{t,ℓ} ≠ GT_ℓ]
```

然后在 ablation 中再加入 structural term。

#### 第四步：history-conditioned correct posterior

```text
q^*(π | x, GT, D^-)
∝ exp(score(π)) · 1[π ∈ Π_GT] · C(π; D^-)
```

#### 第五步：一次性 commit

```text
Δθ_commit ∝ η_commit · Σ_{π ∈ Π_GT} q^*(π) · stats(π)
```

这一步是 Goal B 中的真正长期 grammar 学习。

### 4.A.5 路线 A 的优点

- wrong 与 correct 角色清晰分离
- 最符合 Goal B 假设
- 避免 wrong 过早写长期 grammar
- 便于 matched-budget 比较

### 4.A.6 路线 A 的风险

- 如果 `C(π; D^-)` 太弱，则退化成 direct-correct
- 如果 `C(π; D^-)` 太强且不准，会把 GT-compatible posterior 压进错误局部 basin
- 如果没有 commit/replay，跨题泛化可能仍然不动

---

## 4.B 路线 B：弱长期 wrong + 强 query-local history + final commit

### 4.B.1 推荐级别

**推荐优先级：第二**

如果路线 A 完全 local 导致 eval 不动，可以试这条。

### 4.B.2 核心思想

wrong 仍允许对长期 grammar 做一点点改动，但力度显著弱于现在：

```text
η_wrong_long << η_fb
```

同时仍保留 query-local history，final correct 读入该 history 做 commit。

### 4.B.3 最简实现

对当前 `apply_feedback(...)` 增加一个模式：

- `mode='decomposed_wrong'`

执行：

1. 正常计算 `q'_k`
2. 只把 `δq = q'_k - q_k` 的一小部分写入长期库

```text
Δθ_wrong_long
= η_wrong_long · Σ_k (q'_k - q_k) · stats(π_k)
```

3. 同时把当前 wrong 记录进 `GoalBProtocolState`

### 4.B.4 什么时候用路线 B

如果路线 A 出现：

- 当前 query narrowing 好了
- 但 probe / held-out 仍然几乎不动

那说明完全 local 的 wrong 可能太弱，可以试路线 B。

---

## 4.C 路线 C：Wrong Shadow Buffer / Delta Buffer，correct 时统一 merge

### 4.C.1 推荐级别

**推荐优先级：第三**

这条路线更工程化，但实现成本更高。

### 4.C.2 核心思想

wrong 不直接写主 library，而是写一个 shadow delta buffer：

```text
Δ^- = Σ_t Δ^-_t
```

correct 时再根据 GT posterior 决定 merge：

```text
Δ_final = α · Δ^- + β · Δ^+
```

这相当于把 wrong history 的影响保留为“待确认草稿”，而不是立即生效。

### 4.C.3 为什么不是第一优先级

- 机制更重
- 参数更多
- 不如路线 A 清晰

只有当前两条路线都不够用时再上。

---

## 5. 超参数最小化原则

为了避免机制和公式冗余，建议严格控制第一版参数数目。

### 5.1 第一版只允许 3 个新自由度

路线 A 第一版建议只引入：

1. `lambda_hist`
   - wrong history 对 GT-compatible posterior 的作用强度
2. `eta_commit`
   - final correct commit 强度
3. `n_replay`
   - 若需要 replay，则 replay 次数

其他全部先固定：

- `lambda_pos = 1`
- `lambda_neg = 1`
- 不做 near-GT soft target
- 不做复杂 sequence-type scheduler
- 不做多重温度

### 5.2 第二版才考虑的参数

只有当第一版看到了正协同迹象，才考虑加：

- `lambda_struct`
- `eta_wrong_long`
- `near_gt_radius`
- `history_decay`

---

## 6. 模块审计：哪些该改，哪些该保留，哪些暂时不碰

### 6.1 `learner/correct_update.py`

**结论：保留，但必须修改**

当前优点：

- 已有 clean learner-side correct update
- 语义与 `TutorLearnerModel.update_from_output(...)` 一致

当前问题：

1. 目前是纯 GT constrained update，没有 wrong-history conditioning
2. 只显式调用 `infer_top_k_ast`，没有处理 `stack` 模式
3. 没有 protocol-local diagnostics 输出
4. 没有和 `TargetPredictor` 的统一 cache 约定

建议修改：

- 改成支持：
  - `mode='direct_correct'`
  - `mode='history_conditioned_correct'`
- 增加 `history_state: Optional[GoalBProtocolState]`
- 增加 `cls_mode` 兼容
- 输出更丰富的 diagnostics：
  - `n_gt_traces`
  - `mass_before_history`
  - `mass_after_history`
  - `history_effective_entropy`

### 6.2 `learner/feedback_update.py`

**结论：保留，但不建议继续让它承担 Goal B 全部逻辑**

当前作用：

- 真实 wrong update 核心实现

对 Goal B 的建议：

- 保持当前 baseline 路径
- 不要把 query-local history accumulation 也硬塞进这个类

建议新增：

- `accumulate_wrong_history(...)`
- 或单独的新模块 `goal_b_history.py`

这样可以避免 `FeedbackUpdater` 同时承担：

- online wrong update
- Goal-B local narrowing
- history-conditioned commit

造成职责过重。

### 6.3 `tutor_api/goal_b_metrics.py`

**结论：保留功能，但建议迁移位置**

它不属于 tutor API。
更合理的位置应是：

- `cls_color_selection/cls_color_selection/analysis/goal_b_metrics.py`
- 或 `experiments/goal_b_metrics.py`

保留理由：

- 指标设计本身是对的
- 现在已经能复现 negative result

建议增强：

- `best_near_gt_rank`
- `gt_family_entropy`
- `parameter_drift_l1`
- `history_condition_effect`

### 6.4 `tmp/run_goal_b.py`

**结论：原型有价值，但必须迁出 `tmp/`**

当前作用：

- 成功产出第一轮 negative result
- 已经是非常好的实验原型

当前问题：

- 在 `tmp/` 下，不利于正式迭代
- 把实验逻辑、报告逻辑、聚合逻辑混在一个文件里

建议：

迁移为：

- `cls_color_selection/cls_color_selection/experiments/run_goal_b.py`
- `cls_color_selection/cls_color_selection/experiments/registry_goal_b.py`

并拆分：

- runner
- condition spec
- aggregation/report

### 6.5 `learner/memory.py`

**结论：暂时不要改**

理由：

- 当前 `QueryMemory` 服务的是 online task memory
- Goal B grammar-only protocol 需要的是实验态 / protocol 态
- 直接塞进去会污染抽象

建议：

- 先新增 `GoalBProtocolState`
- 等将来要整合到 online learner，再决定是否 merge

### 6.6 `tutor_api/beam_analysis.py`

**结论：保留，不改核心公式**

它对 Goal B 非常有价值，尤其是：

- `H_beam`
- `margin`
- `E_wrong`

但它是分析工具，不是 update 机制。
不要把它误当成 learner-side blame pipeline。

### 6.7 `tutor_api/trace_analysis.py`

**结论：保留，作为特征源，不作为现成解法**

它适合提供：

- position-word alignment
- structural uncertainty
- trace salience

但真正的 blame-to-update 机制需要你另行设计。

### 6.8 `tutor_api/counterfactual.py` / `hint_policy.py` / `tutor_inverse.py`

**结论：Goal B 第一阶段暂时不要改**

理由：

- 这些模块主要服务 tutor/hint 线
- Goal B 当前要先解决 grammar-only 机制
- 过早把 hint/risk/tutor 再混回来会降低因果可解释性

---

## 7. 推荐实验矩阵

### 7.1 第一阶段：复现 + 稳固 baseline

目的：

- 确认当前 negative result 稳定
- 给后续方案提供对照基线

条件：

- 保持现有 10 个条件
- 增加更多 seeds
- query 不只用 top wrong，还加：
  - rank-2 wrong
  - rank-3 wrong
  - diversity-selected wrong

成功标准：

- negative result 在更多 task/seed 上稳定复现

### 7.2 第二阶段：路线 A，local-only history

条件：

- `direct_correct`
- `wrong1_hist + correct_commit`
- `wrong2_hist + correct_commit`
- `wrong3_hist + correct_commit`
- matched-budget `correct_x2/x3/x4`

注意：

- wrong 只积累 history，不写长期 grammar
- correct 读取 history 后再 commit

关键看：

- `Synergy_k(gt_mass)` 是否由负转零或转正
- matched-budget 是否出现至少局部翻转

### 7.3 第三阶段：路线 A + replay/distillation

当第二阶段出现：

- 当前 query 有正协同
- probe 侧仍然不动

再加最小 replay：

```text
对 commit 后的 learner，在 2~4 个 probe 上做 constrained distill
```

比较：

- without replay
- with replay

看 held-out 指标是否才开始拉开。

### 7.4 第四阶段：路线 B

如果路线 A 完全没有协同，再试：

- 弱长期 wrong
- local history
- final commit

这里主要回答：

> 完全 local 是否太弱？少量长期 wrong prior 是否更合适？

### 7.5 第五阶段：wrong sequence type 实验

必须分组：

- `near_local_wrong`
- `repeated_same_wrong`
- `diverse_wrong`
- `structural_wrong`

因为 Goal B 很可能不是 “错几次” 本身有用，而是 “错得是否互补” 有用。

---

## 8. 推荐的测试清单

### 8.1 单元测试

建议新增：

- `tests/test_correct_update.py`
- `tests/test_goal_b_metrics.py`
- `tests/test_goal_b_protocol.py`

#### `test_correct_update.py`

至少覆盖：

1. `apply_correct_answer` 能找到 GT-compatible traces
2. `direct_correct` 会提高 `gt_mass`
3. `history_conditioned_correct(history=None)` 等价于 `direct_correct`
4. `ast` / `stack` 两种 mode 行为一致性
5. `update_depth` 的作用边界清晰

#### `test_goal_b_metrics.py`

至少覆盖：

1. `gt_rank`
2. `gt_mass`
3. `margin`
4. 空 beam 处理
5. probe log-likelihood 计算

#### `test_goal_b_protocol.py`

至少覆盖：

1. 同一个初始 predictor 深拷贝后，各条件互不污染
2. local-only history 不应直接改变长期 grammar
3. final commit 后 grammar 才改变
4. matched-budget 聚合逻辑正确

### 8.2 回归测试

Goal B 新模块不应破坏原有 Phase 1-4：

- `tests/test_feedback_update.py`
- `tests/test_phase4_inverse.py`

都应继续通过。

### 8.3 统计测试

建议新增一个轻量 aggregate check：

如果路线 A 成功，至少应满足：

```text
mean(gt_mass wrong1_hist+correct) >= mean(gt_mass direct_correct)
```

更强目标：

```text
mean(gt_mass wrong1_hist+correct) > mean(gt_mass correct_x2)
```

这可以先做成非严格 CI 阈值报告，不必第一天就写死成 pytest hard assert。

---

## 9. 预期结果与判定标准

### 9.1 对路线 A 的预期

我认为最合理的预期不是“一上来就大胜”，而是三阶段：

#### 预期 1：先消除 anti-synergy

如果路线 A 是对的，第一步最可能看到：

- `wrong1_hist+correct` 不再显著差于 `correct_x2`
- `Gain_correct(S1)` 不再低于 `Gain_correct(S0)`

#### 预期 2：小规模正协同

进一步可能看到：

- `wrong1_hist+correct` 在 `gt_mass` 上略优于 `direct_correct`
- `wrong2_hist+correct` 仍然可接受
- `wrong3_hist+correct` 未必继续变好

#### 预期 3：泛化提升要靠 replay/commit 设计

若只做 local narrowing + final commit，可能：

- 当前 query 指标提升明显
- probe 仍然较平

如果加 replay 后：

- `probe_acc`
- `probe_ll`

才开始改善，那是合理现象，不算机制失败。

### 9.2 失败判据

若出现以下情况，应视为路线失败或需要收缩：

1. `history_conditioned_correct` 与 `direct_correct` 几乎完全相同
   - 说明 history 因子没有提供额外信息
2. `wrong1_hist+correct < direct_correct` 仍稳定为负
   - 说明 history 仍主要在制造偏移，而不是提供有效约束
3. 当前 query 指标变好但 probe 永远不动
   - 说明没有把 narrowing 转成长期 grammar 学习

---

## 10. 建议的实施顺序

### 第 1 步：把原型变成正式模块

1. 把 `tmp/run_goal_b.py` 迁到正式 experiments
2. 把 `goal_b_metrics.py` 迁出 `tutor_api`
3. 为 `correct_update.py` 增加：
   - `stack` 支持
   - history-conditioned mode

### 第 2 步：新增 protocol-local state

新增：

- `learner/goal_b_state.py`

不要先碰 `QueryMemory`。

### 第 3 步：实现路线 A 的第一版

第一版只做：

- local wrong history
- history-conditioned GT posterior
- single final commit

不要一开始就加 replay、near-GT、weak-long-term wrong。

### 第 4 步：做最小实验矩阵

先跑：

- `direct_correct`
- `wrong1_hist+correct`
- `wrong2_hist+correct`
- `correct_x2`
- `correct_x3`

如果没有任何正面迹象，再决定是否进入路线 B。

### 第 5 步：只有在需要时才加 replay

若当前 query 指标改善、但 probe 不动，再加 replay。

---

## 11. 最短结论

对 Antigravity 的最重要指令只有三句：

1. **不要把当前 negative result 解释成 wrong 完全没价值，而要解释成：当前 wrong history 没有被以对 GT-compatible family 有效的方式注入 final correct update。**
2. **优先实现“query-local wrong history + history-conditioned correct commit”，不要继续直接强化长期 wrong update。**
3. **先追求消除 anti-synergy，再追求正协同；如果当前 query 指标改善但泛化不动，再用最小 replay 把 narrowing 转成长期 grammar 学习。**

---

## 12. Route A v2 已知结果与收紧后的判断

这一节记录已经跑出的 `Route A v2` 结果，避免重复试验同一类机制。

对应代码：

- `cls_color_selection/cls_color_selection/learner/goal_b_state.py`
- `cls_color_selection/cls_color_selection/learner/correct_update.py`
- `cls_color_selection/tmp/run_goal_b_v2.py`

### 12.1 已知事实

1. `hist{k}_lam{λ}` 条件的最终表现几乎退化为 `correct_only`
2. 更强的 `lambda_hist` 不会带来增益，反而轻微变差
3. 诊断脚本已经证明 `compute_history_factor(...)` 不再是常数
4. 但即使 factor 有方差，`history_conditioned_correct` 仍未优于 `direct_correct`

这说明当前 `Route A v2` 的失败，不是因为实现根本没生效，而是因为它生效的方向没有提供有用的新信息。

### 12.2 更稳妥的机制解释

当前结果最符合下面这个解释：

1. `apply_correct_answer(...)` 仍然先在 `Π_GT` 上做 constrained posterior
2. `compute_history_factor(...)` 只是对这些已经 GT-compatible 的 traces 再乘一个 `C(π; D^-)`
3. 当前 `C(π; D^-)` 虽然有变化，但和原始 constrained score 高度同向
4. 因此新 posterior 更像是 `q_dir` 的 sharpened 版本，而不是一个真正不同的 posterior

形式上可以写成：

```text
q_goalB(π) ∝ exp(s(π)) · 1[π ∈ Π_GT] · C(π; D^-)
```

如果 `log C(π; D^-)` 与 `s(π)` 在 `Π_GT` 内部强正相关，那么：

```text
q_goalB ≈ sharpen(q_dir)
```

这通常只会减少多样性，不会注入 direct-correct 不具备的额外结构信息。

### 12.3 这条负结果说明了什么

可以认为当前已经得到一个比较干净的结论：

- **单纯的 query-local wrong history reweighting，不足以让 Goal B 成立**

但还不能把它扩大成：

- “所有 local-history 路线都失败”

更准确地说，这次证伪的是：

- **在 GT-compatible trace 集合固定不变的前提下，仅靠后验重加权，且该重加权与原始 score 同向时，不足以超过 direct-correct**

### 12.4 对下一步优先级的影响

因此，下一步不应继续在 `lambda_hist`、长度距离、pattern 距离上做更多小修补。优先级应改为：

1. Route B：弱长期 wrong + local history + final correct commit
2. Route A-support variant：不是只 reweight `Π_GT`，而是改变 final correct 可见的 candidate support
3. commit/replay 最小版：验证 query-local narrowing 能否转成跨题 grammar 改进

### 12.5 下一步必须补的两个诊断

在进入 Route B 前，先补两个低成本诊断：

1. `corr(score(π), log C(π; D^-))`
   - 若长期接近 1，说明 history factor 主要在重复原始 ranking
2. posterior ESS / top-mass concentration before-vs-after reweight
   - 若 history 只是在压缩 posterior 而不改变 family 内排序来源，说明它更像 sharpening，不是新证据

这两个诊断应进入正式实验日志。
