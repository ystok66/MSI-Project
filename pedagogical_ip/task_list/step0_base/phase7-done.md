# Phase 7 完成总结：Approximate Robot Belief Over Agent Belief

## 结果

- **209/209 tests pass**（+32 新增：8 robot_belief + 8 agent_predictor + 9 intervention_policy + 10 integration - 3 overlap with fixes）
- **Legacy 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 核心变化

#### 新建 `src/teachers/robot_belief.py`（~145 行）

| 类/函数 | 语义 |
|---------|------|
| `RobotBelief` | 近似 agent 内部状态：belief mean/var + competence params + copy_mode |
| `init_robot_belief()` | 从 agent 状态创建 surrogate（exact/noisy/stale） |
| `sync_robot_belief()` | 每步同步，respecting copy_mode |
| `build_surrogate_predictor()` | 从 snapshot 构建独立 LatentCostRiskHead |

**Competence mismatch 与 belief mismatch 分离**：
- `copy_mode`（exact/noisy/stale）控制 belief 精度
- `budget_mismatch` / `risk_weight_mismatch` 控制 competence 偏差

#### 新建 `src/teachers/agent_predictor.py`（~150 行）

| 函数 | 语义 |
|------|------|
| `predict_agent_prefix()` | 从 surrogate rollout agent 的 BeliefPlan |
| `predict_agent_prefix_after_warn()` | **反事实**：如果发 warning 后 agent 会怎么走 |
| `predict_agent_prefix_after_unlock()` | **反事实**：如果 unlock 后 agent 会怎么走 |
| `estimate_learning_gain()` | 启发式：prefix 沿线的 uncertainty reduction |

#### 新建 `src/teachers/intervention_policy.py`（~175 行）

| 类/函数 | 语义 |
|---------|------|
| `InterventionDecision` | 结构化决策：action + scores + reason + predicted_prefix + counterfactual_scores + decision_margin |
| `InterventionConfig` | 可配置权重：catastrophe / learning_gain / autonomy_penalty / deadline |
| `score_interventions()` | 三路反事实 rollout → 加权评分 WAIT/WARN/UNLOCK |

#### 修改 `src/envs/lattice_v2_runner.py`（+50 行）

- `V2EpisodeState` 新增：`robot_belief_mode`, `robot_belief`, `last_intervention`, `belief_copy_mode`, `budget_mismatch`
- `observe()`：同步 `sync_robot_belief()` after feature_belief update
- `_apply_tutor_dispatch()`：robot-belief 分支 → `score_interventions()` → 执行 WARN/UNLOCK/WAIT

---

### 设计要点

| 要点 | 实现 |
|------|------|
| Intervention 评分 | **反事实 surrogate rollout**，不是拍静态常数 |
| WAIT | rollout 当前 surrogate |
| WARN | 在 surrogate 上施加 warning effect → 再 rollout |
| UNLOCK | 在 surrogate topology 上打开门 → 再 rollout |
| Robot 不偷看真值 | 只读 agent 的 feature_belief + latent_predictor snapshot |
| Competence vs belief 分离 | `copy_mode` 管 belief；`budget_mismatch` 管 competence |
| learning_gain | 局部启发式：prefix 沿线平均 uncertainty |
| InterventionDecision | action + scores + reason + predicted_prefix + counterfactual_scores + decision_margin |
| 不碰 particle_teacher | 保留旧 teacher path 不变 |

---

### 累积进度

| Phase | 内容 | 测试 |
|-------|------|------|
| 0 | 基线冻结 | 77 |
| 1 | 协议层 | 91 |
| 2 | Runner 平台化 | 97 |
| 3 | 环境接口 | 105 |
| 4 | Latent world 语义 | 124 |
| 5 | Patch 观测 + Prefix 预测 | 150 |
| 6 | Belief-conditioned bounded planning | 177 |
| **7** | **Approximate robot belief over agent belief** | **209 (+32)** |
