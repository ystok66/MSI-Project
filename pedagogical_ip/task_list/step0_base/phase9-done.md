# Phase 9 完成总结：Evaluation System + Experiment Matrix

## 结果

- **279/279 tests pass**（+34 新增）
- **Legacy 基线不变**：9/80/68/99/100%
- **Smoke matrix 完成**：3 jobs (no_tutor / warning / robot_belief × medium)
- **Export 管线验证**：online_table.csv, transfer_table.csv, tradeoff_data.json, intervention_usage.json

---

### 核心新增

#### `src/metrics/phase9_metrics.py`（~320 行）

| 层级 | 类型 | schema 区分 |
|------|------|------------|
| Episode-level | `EpisodeSummary` | `success: bool`, `death: bool`, `timeout: bool` |
| Episode-level | `TransferSummary` | 独立类型，不与 online 混 |
| Aggregate-level | `AggregateMetrics` | `success_rate: float`, `mean/std/n/sem` |

三组指标：
- **Task**: success/death/timeout/cost/risk/intervention_count
- **Learning**: cost_prediction_error, risk_calibration_gap, uncertainty_reduction (visited+nearby)
- **Pedagogical**: boredom_proxy, frustration_proxy, timing_quality, information_gain

#### `src/metrics/transfer_eval.py`（~100 行）

Transfer 协议：
- **复制**: learned predictor weights (cost_w/b, risk_w/b)
- **重置**: episodic belief, position, inventory, runner state
- **关闭**: tutor (tutor_mode="none", robot_belief_mode=False)

#### `configs/phase9_eval.yaml`（~75 行）

- Agent: weak(budget=4) / medium(16) / strong(30)
- Teacher: no_tutor / warning_only / unlock_only / item_only / heuristic_mixed / robot_belief
- **每个 teacher 显式声明 `allowed_interventions`**
- Smoke subset 配置内置

#### `scripts/run_phase9_matrix.py`（~160 行）

- `--smoke`：3 条件 × 3 seed × 2 transfer
- `--filter agent=medium,teacher=robot_belief`
- 输出: `online_results.json`, `transfer_results.json`

#### `scripts/plot_phase9_results.py`（~120 行）

- CSV: online_table.csv, transfer_table.csv
- JSON: tradeoff_data.json, intervention_usage.json
- 只导出数据，不追求复杂作图

#### `src/envs/lattice_v2_runner.py` 新增 `get_extended_metrics()`

---

### 你的 6 条修正全部落地

| 修正 | 状态 |
|------|------|
| 1. Episode `success:bool` vs aggregate `success_rate:float` | ✅ |
| 2. Teacher allowed_interventions 显式声明 | ✅ |
| 3. Transfer: 复制 learned params / 重置 episode state | ✅ |
| 4. `risk_calibration_gap` 不叫 ECE | ✅ |
| 5. Aggregate 带 mean/std/n/sem | ✅ |
| 6. Report 先 CSV/JSON 导出 | ✅ |

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
| 7 | Approximate robot belief | 209 |
| 8 | Unified intervention family + shield | 245 |
| **9** | **Evaluation system + experiment matrix** | **279 (+34)** |
