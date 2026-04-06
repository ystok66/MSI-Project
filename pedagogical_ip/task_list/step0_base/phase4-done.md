# Phase 4 完成总结：统一 Latent World 语义

## 结果

- **124/124 tests pass**（+21 新增：8 cost_risk + 5 belief + 8 latent_path）
- **Legacy 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%

---

### 核心变化

#### 新建 `src/agents/cost_risk_model.py`

| 类 | 语义 |
|---|---|
| `BayesianCostHead` | 线性 Bayesian：feature → cost（Gaussian likelihood） |
| `WorldWeights` | 固定世界参数，从 z 导出 true_cost / true_risk |
| `generate_world_weights()` | seed 可复现的 world weight 生成 |
| `LatentCostRiskHead` | 组合 cost_head + risk_head，实现 latent_predictor 协议 |

- supervision mode: `"oracle_visited"` / `"binary_outcome"`
- 组合复用 `BayesianRiskHead`，不替换旧文件

#### 修改 `src/envs/lattice_v2.py`

- `latent_mode=True` 时从 features 导出 cost/risk（通过 `WorldWeights`）
- `LatticeV2Meta` 新增 `world_weights` + `latent_mode`
- legacy mode 完全不变

#### 修改 `src/agents/planner_astar.py`

- 新函数 `cell_cost_v2_latent()` — 4 项分离评分：
  - `λ_c * cost_hat + λ_r * risk_penalty + λ_uc * cost_unc + λ_ur * risk_unc`
- `plan_next_action_v2()` 新增 `latent_predictor` 参数
- legacy path（`latent_predictor=None`）行为不变

#### 修改 `src/envs/lattice_v2_runner.py`

- `V2EpisodeState` 新增 `latent_mode` / `latent_predictor`
- `reset()` 根据 `latent_mode` 创建 `LatentCostRiskHead` 或 `BayesianRiskHead`
- `plan_and_move()` outcome 更新双路径分发

### 设计决策

- **Feature-as-latent 语义**：z = 当前 4 维 feature vector，不引入第二层隐变量
- **Additive migration**：`latent_mode` config 控制，默认 False = legacy
- **Uncertainty 分离**：`λ_uc`, `λ_ur` 独立可调
- **Supervision 可配**：oracle_visited vs binary_outcome
- **组合不替换**：`LatentCostRiskHead` 组合 `BayesianRiskHead`，旧文件原封不动
