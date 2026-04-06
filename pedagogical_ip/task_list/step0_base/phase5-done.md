# Phase 5 完成总结：Patch 观测 + Prefix 预测

## 结果

- **150/150 tests pass**（+26 新增：7 patch_obs + 6 belief + 9 prefix + 4 integration）
- **Legacy 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 核心变化

#### 修改 `src/agents/observation_model.py`

| 函数 | 语义 |
|---|---|
| `observe_features_patch()` | 可配置 `patch_radius` 的局部 patch 观测 |
| 噪声模型 | 离散 3 档：d=0→0.01, d=1→0.08, d≥2→0.20 |
| Legacy 兼容 | `patch_radius<=1` 直接委托 `observe_features()`，RNG 序列不变 |

#### 修改 `src/agents/feature_belief.py`

- `visit_count`：观测计数（非遍历计数）
- `last_observed_t`：最后观测时间步
- `update()` 新增 `t` 参数

#### 新建 `src/agents/prefix_prediction.py`

| 类/函数 | 语义 |
|---|---|
| `PrefixPrediction` | path prefix 上 cost/risk/uncertainty 聚合 |
| `compute_prefix_predictions()` | 只读诊断，不改 planner/belief/env 状态 |
| `cumulative_risk` | 独立近似：1 − ∏(1 − ρ_i) |
| `risky_prefix_cells` | 超过阈值的高风险前缀格 |

#### 修改 `src/agents/planner_astar.py`

- `plan_next_action_v2()` 返回 3-tuple：`(action, next_pos, path)`
- 不改搜索算法

#### 修改 `src/envs/lattice_v2_runner.py`

- `V2EpisodeState` 新增 `patch_radius`, `prefix_horizon`, `last_prefix`
- `observe()` 支持 `patch_radius>1` 路径
- `plan_and_move()` 在 `prefix_horizon>0 + latent_mode` 下计算 prefix

---

### 累积进度

| Phase | 内容 | 测试 |
|-------|------|------|
| 0 | 基线冻结 | 77 |
| 1 | 协议层 | 91 |
| 2 | Runner 平台化 | 97 |
| 3 | 环境接口 | 105 |
| 4 | Latent world 语义 | 124 |
| **5** | **Patch 观测 + Prefix 预测** | **150 (+26)** |

细节见 `task_list/step0-phase5.md`。
