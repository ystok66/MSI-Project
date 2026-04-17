# `cls_color_selection` Goal B 调查报告

## 0. 结论先行

这次代码调查的核心结论是：

1. 当前真实 learner 的 grammar 学习机制是单边的。
   只有 failed confirm 会触发 grammar update；successful confirm 不会触发 learner-side 的显式正向 consolidation。
2. 因此，当前仓库还不能直接回答 Goal B。
   它能研究的是 “wrong-confirm 有没有用”，但还不能研究 “wrong x k + correct 是否优于 direct-correct”。
3. `hint` 不是 clean correct-answer learning。
   现有设计明确把 hint 当成 tutor assistance，并通过 `assist_mask` + `rho_assist` 折扣其 grammar 证据强度。
4. 如果要推进 Goal B，最优先的不是继续挖旧日志，而是新增一个 grammar-only 的可控协议。
   这个协议要显式构造：
   `direct-correct`、`wrong x k + correct`、`correct x (k+1)`、`wrong x k only`
   并记录每一步的 narrowing 与 correct-update gain。

下面按你最关心的 7 个调查点展开。

---

## 1. 现有代码里，真实 learner 到底怎样学 grammar

### 1.1 failed confirm 是唯一的 grammar update 入口

真实 learner 的 confirm 失败更新路径是：

1. 环境返回 `submitted + wrong-position mask`
2. `FeedbackUpdater.apply_feedback(...)` 读取当前 beam posterior
3. 用 wrong feedback likelihood 重加权 posterior
4. 用 differential M-step 直接改 concept library
5. invalidate cache，并重新预测 `target_output`

对应代码：

- `cls_color_selection/cls_color_selection/environment/transition.py`
  - `confirm(...)`
- `cls_color_selection/cls_color_selection/environment/grammar_task_env.py`
  - `step_confirm(...)`
- `cls_color_selection/cls_color_selection/learner/feedback_update.py`
  - `reweight_beam_posterior(...)`
  - `differential_m_step(...)`
  - `apply_feedback(...)`
- `cls_color_selection/cls_color_selection/experiments/run_phase1.py`
  - failed confirm 后调用 `feedback_updater.apply_feedback(...)`
- `cls_color_selection/cls_color_selection/experiments/run_phase3.py`
- `cls_color_selection/cls_color_selection/experiments/run_phase4.py`

### 1.2 success confirm 不会让真实 learner 学 grammar

在 `run_phase1.py` / `run_phase3.py` / `run_phase4.py` 的 query loop 里，
`if success: break`，不会进入任何 learner-side positive update。

这意味着：

- 当前真实 learner 没有 `apply_correct_answer(...)`
- success 只是结束题目，不是一次正向 consolidation
- 当前 “wrong 有学习作用，success 没有”

### 1.3 wrong update 会改变后续 query 内部动态

failed confirm 后不仅 grammar 被更新，`target_output` 还会重新预测。
如果新 `target_output` 改了，代码会：

- 覆写 `state.target_output`
- 把 `state.completion` 重置为全空

同时候选池生成也是围绕 `state.target_output` 进行的，而不是围绕 GT。
所以 wrong update 不只是“记录一次反馈”，而是会改 query 内的后续搜索目标。

这点很重要，因为它解释了为什么当前系统天然偏向 wrong-side learning：

- wrong 会推动 grammar 状态变化
- success 不会推动 grammar 状态变化

---

## 2. 对问题 1：`apply_correct_answer(...)` 应该怎样定义才公平

### 2.1 我认为最公平的对象不是 completion，而是 grammar posterior / trace posterior

如果 Goal B 的语义是：

> learner 学到了正确答案

那 update 的作用对象应该是：

- latent trace posterior
- beam posterior
- concept library 统计量

而不应该是：

- 直接把 `state.completion` 设成 GT
- 直接把 `state.target_output` 设成 GT

后两者只是 “系统替 learner 做对了”，不是 “learner 的 grammar 被教会了”。

### 2.2 最自然的公平实现，是复用 CLS 原生的 constrained inference 语义

仓库里其实已经有一个非常接近 `apply_correct_answer(...)` 的现成原型：

- `cls_color_selection/cls_color_selection/tutor_api/learner_model.py`
  - `update_from_output(words, Y_submit)`

它做的事情是：

1. 把目标输出转成 `target_vecs`
2. 运行 constrained beam search
   - `infer_top_k_ast(words, target_vecs, library, priors)`
   - 或 `infer_top_k_stack(...)`
3. 对这些 “能生成该输出的 trace” 做 weighted M-step

这和 BASIC 里的 `CLSAgent.study(...)` 非常一致。
`CLSAgent.study(...)` 在 support learning 里本来就是 “给定正确输出 -> constrained E-step -> M-step”。

所以从“公平代表 learner 学到了正确答案”的角度看，
**最推荐的实现不是手写一个 ad-hoc positive reward，而是复用 constrained E-step + M-step。**

### 2.3 推荐的最小实现

我建议的最小实现是新增：

`cls_color_selection/cls_color_selection/learner/correct_update.py`

或者先加到 `feedback_update.py`，接口类似：

```python
def apply_correct_answer(
    predictor,
    words: List[str],
    ground_truth: List[str],
    *,
    mode: str = "constrained_exact",
    eta_correct: float = 1.0,
    update_depth: str = "full_trace",
) -> dict:
    ...
```

推荐默认模式：

`constrained_exact`

语义：

1. 用 GT 做 constrained beam
2. 得到 GT-compatible traces
3. 用这些 traces 做 weighted M-step
4. 更新 concept library
5. 调用方负责 invalidate cache

### 2.4 为什么不建议只奖励 “当前 beam 里 Y_k == GT 的候选”

因为这会有一个明显缺陷：

- 如果 GT 当前不在 beam 里，update 会接近 0
- 这会把 “current beam 没覆盖 GT” 和 “learner 不该学 correct answer” 混在一起

而 Goal B 恰恰最需要在 learner 还没学会时，也能给一次 clean correct supervision。

所以：

- 只奖励 `Y_k == GT` 的 unconstrained beam 候选，不够稳
- 至少要有 GT-compatible constrained search 作为 rescue path

### 2.5 是否要奖励 near-GT

建议分两层：

1. 主协议默认先做 `exact GT-compatible`
   这样最干净，语义最清楚
2. 另做一个 near-GT ablation
   例如按 edit distance / position match 给 soft 权重

原因：

- exact 模式更能代表 “teacher 给了正确答案”
- near-GT 模式更像 “teacher 给了比较接近的正向示范”
- 两者都值得测，但 exact 应该是主线 baseline

### 2.6 update 强度如何和 wrong update 可比

建议不要强行让它和 `eta_fb` 完全同值，而是让实验显式控制：

- `eta_wrong`
- `eta_correct`

推荐默认：

- 先设 `eta_correct = eta_fb`
- 然后额外做一个 matched-strength sweep

也就是说，可比性应该由 protocol 保障，而不是靠在语义上把 correct update 人为弱化到和 wrong 完全同形。

### 2.7 我对正确实现的排序

从最推荐到最不推荐：

1. `constrained_exact`
   - GT 做 constrained beam
   - weighted M-step
2. `constrained_soft`
   - GT / near-GT 做 constrained or semi-constrained posterior
3. `positive_diff_on_unconstrained_beam`
   - 只奖励当前 beam 中 GT / near-GT 候选
4. `force completion / force target_output`
   - 不推荐，不算 learner learning

---

## 3. 对问题 2：Goal B 的协同要怎样被隔离

### 3.1 最关键的不是最终总分，而是 “同一份 correct update 的边际增益”

应该固定同一份 correct supervision，
比较它作用在不同 learner 状态上的效果：

- `S0`: 初始 learner
- `S1`: 经过 1 次 wrong update
- `S2`: 经过 2 次 wrong update
- `S3`: 经过 3 次 wrong update

然后测：

- `Gain_correct(Si)`

也就是同一份 correct update 在 `Si` 上带来的增益。

### 3.2 建议的协同定义

对某个指标 `M`，定义：

```text
Synergy_k(M) = [M(after wrong^k + correct) - M(after wrong^k)]
               - [M(after correct) - M(initial)]
```

如果 `Synergy_k > 0`，才说明：

前面的 wrong 让最后那次 correct 更有效。

### 3.3 最该测的 gain 指标

建议至少记录：

- GT rank gain
- GT exact posterior mass gain
- GT-near mass gain
- top-1 flip to GT 概率
- top-1 flip to near-GT 概率
- next-query probe gain
- held-out probe gain

这比只看最终 EvalSuccessRate 更直接，因为它测的是 “correct update 本身学得更好没有”。

---

## 4. 对问题 3：wrong x k 是否真的在 narrowing

### 4.1 当前代码已经有一半现成工具

`cls_color_selection/cls_color_selection/tutor_api/beam_analysis.py`
已经能从 beam 里提取：

- `p_exact`
- `H_beam`
- `H_beam_norm`
- `margin`
- `E_wrong`
- per-position `p_wrong`

所以 Goal B 所需的 narrowing 诊断并不是从零开始。

### 4.2 但还缺 4 个关键指标

我建议新增一个专用 metrics 模块，例如：

`cls_color_selection/cls_color_selection/tutor_api/goal_b_metrics.py`

补上：

- `gt_rank`
- `gt_mass`
- `best_near_gt_rank`
- `top1_vs_best_gt_gap`

其中：

- `gt_rank`: beam 中第一个 `Y_k == GT` 的排名；若缺失则记 `inf`
- `gt_mass`: 所有 `Y_k == GT` 的 posterior mass
- `best_near_gt_rank`: 满足 near-GT 阈值的最佳排名
- `top1_vs_best_gt_gap`: `score(top1) - score(best_gt_like)`

### 4.3 应该记录成一条时间曲线

每次 wrong 之后都要记一次：

- before wrong
- after wrong 1
- after wrong 2
- after wrong 3
- after final correct

而不是只记 teach 结束时的总结果。

### 4.4 我对 narrowing 的判断

Goal B 成立的前提不是 “wrong 有更新”，而是：

- `H_beam` 下降
- `gt_rank` 上升
- `gt_mass` 上升
- top-1 与更好候选的 gap 缩小

如果这些曲线不动，最后 correct update 就很难体现真正协同。

---

## 5. 对问题 4：不同 wrong 序列的信息价值不同

### 5.1 现有代码没有这个分类，但很容易加

建议每次 wrong 记录一条 submission record，字段至少包括：

- `submitted`
- `ground_truth`
- `mask`
- `n_wrong_positions`
- `edit_distance_to_gt`
- `length_delta`
- `top1_before_update`
- `top1_after_update`

### 5.2 我建议的 4 类序列标签

1. `local_near_wrong`
   - edit distance 小
   - 只错 1 到 2 个位置
2. `repeated_same_wrong`
   - 连续两次 wrong 的 `submitted` 很像
3. `diverse_wrong`
   - 每次 wrong 暴露不同位置或不同长度偏差
4. `structural_wrong`
   - 长度偏差大，或 top trace role/repeat 结构偏差大

### 5.3 最小可行判据

不需要一开始就做复杂 trace clustering。
第一版可以先用 output-level 判据：

- Hamming / edit distance
- wrong mask overlap
- length mismatch

如果后面要更强，再接 trace-level salience / role map 差异。

---

## 6. 对问题 5：matched-budget 对照为什么是必须的

### 6.1 当前仓库还没有 Goal B 需要的 matched-budget condition

现有 registry 只区分：

- `none`
- `wrong_only`
- `wrong_positions`

没有：

- `direct_correct`
- `wrong_k_plus_correct`
- `correct_x_n`

所以要做 Goal B，必须新增独立 runner 或独立实验集。

### 6.2 最少要有的 5 组条件

我建议主协议固定为：

1. `correct_only`
   - 一次 correct update
2. `wrong_k_only`
   - k 次 wrong update，无 correct
3. `wrong_k_plus_correct`
   - k 次 wrong，再一次 correct
4. `correct_x_(k+1)`
   - 做 k+1 次 correct update
5. `direct_correct_then_probe`
   - 从初始状态直接 correct，再立刻 probe

### 6.3 为什么 `correct_x_(k+1)` 很重要

因为如果：

- `wrong_k + correct` 优于 `correct_only`

还不能说明 wrong 有特殊 pedagogical value。
它也可能只是因为更新次数更多。

只有当：

- `wrong_k + correct` 也优于 matched-budget 的 `correct_x_(k+1)`

你才能比较有把握地说：

wrong 提供了结构化 narrowing 信息，而不只是额外步数。

---

## 7. 对问题 6：为什么 `hint` 绝不能替代 correct-answer learning

### 7.1 hint 走的是 tutor assistance，不是 learner grammar supervision

`apply_hint_to_state(...)` 会把正确球直接放进 `completion`，并且打上 `assist_mask`。

之后 learner-side feedback update 会对这些位置乘以 `rho_assist` 折扣。
默认配置里：

- `rho_assist = 0.3`

这说明系统明确把 hint 视为：

- “被 tutor 帮过的位置”
- 其 grammar evidence 应该降权

### 7.2 现有 counterfactual 逻辑本身就在强调 “hint 会污染学习”

`counterfactual.py` 的注释写得非常明确：

- WAIT branch:
  learner learns from its own mistakes, uncontaminated
- HINT branch:
  feedback update with assist-aware discounting

所以对 Goal B 而言，`hint` 不能作为 clean correct-answer protocol。

### 7.3 我对这个问题的结论

如果要研究 Goal B，必须新增一个不经过 hint 的 learner-side correct update。

也就是：

- correct answer 被当作 grammar evidence 吸收
- 不是把正确球塞进 state
- 也不是 tutor assist discount 版本

---

## 8. 对问题 7：为什么不该继续挖旧日志，而该新造协议

### 8.1 现有自然样本确实太少

本次本地解析已有结果后发现：

- `phase2/raw_results.jsonl`
  - teach 成功 query 999 条
  - `confirm_count > 1` 的成功样本 0 条
- `phase3/raw_results.jsonl`
  - teach 成功 query 1284 条
  - `confirm_count > 1` 的成功样本 70 条
- `phase3_t3_hint/raw_results.jsonl`
  - teach 成功 query 779 条
  - `confirm_count > 1` 的成功样本 14 条

也就是说，自然发生的 “先错几次再成功” 很稀少。

### 8.2 稀少是机制决定的，不只是数据不够

当前策略 `should_confirm(...)` 只有在：

- 所有位置填满
- 或 fill ratio 达阈值

时才 confirm。
默认阈值又是：

- `confirm_fill_threshold = 1.0`

这意味着 learner 通常是 “填满自己当前的 `target_output` 再提交”。
再结合：

- candidate pool 是围绕当前 `target_output` 采样
- wrong 后才会改 `target_output`

自然就更容易出现两种结局：

- 一次 confirm 就成功
- 连续 wrong 直到 timeout

而不是丰富的 `wrong -> wrong -> success` 轨迹。

### 8.3 所以 Goal B 应该用人为协议构造

推荐新增一个独立 runner：

`cls_color_selection/cls_color_selection/experiments/run_goal_b.py`

它不要依赖自然 query loop 产生样本，而是显式构造教学协议。

---

## 9. 推荐的最小 Goal B 协议

### 9.1 我建议先做 grammar-only protocol

因为 `cls_color_selection` 还混有：

- risk belief
- candidate pool availability
- death / warning / courage

这些都会污染 Goal B 的因果解释。

最小协议应该先只测 grammar 学习：

1. 固定 learner 初始状态
2. 固定一个 query `(words, GT)`
3. 用当前 grammar 得到 beam
4. 人为抽取 learner 的 top-1 或指定 wrong submission
5. 应用 wrong update 若干次
6. 应用一次 clean correct update
7. 立刻测 probe / held-out grammar 指标

这个 protocol 可以完全绕开：

- candidate pool
- risk/danger
- hint injection
- mid-query material constraints

### 9.2 三层 runner 设计

建议拆成：

1. `build_protocol_state(...)`
   - 初始化 learner / query / GT / probe set
2. `apply_wrong_step(...)`
   - 给定提交序列，执行一轮 wrong update + log
3. `apply_correct_step(...)`
   - 执行一次 clean correct update + log

### 9.3 必跑条件

主实验至少包含：

- `direct_correct`
- `wrong1_plus_correct`
- `wrong2_plus_correct`
- `wrong3_plus_correct`
- `correct_x2`
- `correct_x3`
- `correct_x4`
- `wrong1_only`
- `wrong2_only`
- `wrong3_only`

---

## 10. 我建议新增的日志规范

每一步 wrong / correct 后，至少记录：

- `query_words`
- `ground_truth`
- `submitted`
- `step_type`
  - `wrong`
  - `correct`
- `step_index`
- `beam_entropy_before`
- `beam_entropy_after`
- `gt_rank_before`
- `gt_rank_after`
- `gt_mass_before`
- `gt_mass_after`
- `best_near_gt_rank_before`
- `best_near_gt_rank_after`
- `top1_before`
- `top1_after`
- `top1_margin_before`
- `top1_margin_after`
- `update_l1_role_delta`
- `update_l1_emit_delta`
- `probe_acc_before`
- `probe_acc_after`
- `probe_ll_before`
- `probe_ll_after`

如果只允许第一版做最小日志，我建议先保留：

- `H_beam`
- `gt_rank`
- `gt_mass`
- `top1`
- `margin`
- `probe_acc`
- `probe_ll`

---

## 11. 推荐的代码落点

### 11.1 可复用的现有模块

- `learner/feedback_update.py`
  - differential M-step 和 beam posterior reweighting 框架
- `learner/cls_wrapper.py`
  - unconstrained beam posterior
- `tutor_api/learner_model.py`
  - `update_from_output(...)` 已经是 correct update 的最好原型
- `tutor_api/beam_analysis.py`
  - narrowing 指标基础框架
- `BASIC/cls_learner/agent.py`
  - `study(...)` 的原生 constrained E-step / M-step 语义
- `BASIC/cls_learner/layer1_cortex/cortex.py`
  - `m_step_from_traces(...)`

### 11.2 建议新增的模块

- `cls_color_selection/cls_color_selection/learner/correct_update.py`
  - clean correct-answer learner update
- `cls_color_selection/cls_color_selection/tutor_api/goal_b_metrics.py`
  - GT rank / GT mass / near-GT metrics
- `cls_color_selection/cls_color_selection/experiments/run_goal_b.py`
  - 协议化实验 runner

### 11.3 我不建议第一刀就改的地方

- `policy.py`
- `hint_policy.py`
- `counterfactual.py`
- `run_phase1.py` 的自然 query loop

这些都不是 Goal B 的最短路径。
第一刀应该先建立可控 grammar-only protocol。

---

## 12. 最后给 Antigravity 的执行重点

如果把这份调查转成执行单，我会要求它优先交这 3 样：

### A. `apply_correct_answer(...)` 设计草案

必须包含：

- exact GT-compatible 方案
- near-GT ablation 方案
- 是否复用 `cortex.m_step_from_traces(...)`
- 如何与 `eta_fb` 对齐

### B. Goal B 实验协议

必须包含：

- `direct_correct`
- `wrong x k + correct`
- `correct x (k+1)`
- `wrong x k only`
- probe / held-out 设计

### C. 指标与日志规范

必须包含：

- narrowing 曲线
- correct update 边际 gain
- matched-budget 结果
- wrong 序列类型标签

---

## 13. 一句话总结

对 `cls_color_selection` 来说，Goal B 的真正第一刀不是继续强化 wrong-confirm，
而是新增一个 **clean learner-side correct update**，并用一个 **grammar-only、可控、matched-budget 的协议**
去测：

> 同样一次 correct update，放在 wrong x k 之后，是否真的更有效。
