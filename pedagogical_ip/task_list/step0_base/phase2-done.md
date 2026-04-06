# Phase 2 完成总结：V2 平台化

> 相关文件见 [step0-phase1.md](./step0-phase1.md)（Phase 1 任务说明）

## 结果

- **97/97 tests pass**（+6 runner tests）
- **V2 基线不变**：no_tutor=9%, warn=80%, door_2=68%, door_3=99%, close=100%
- `_diag_l2c1_sweep.py`：297 → 115 行

---

### 改了什么

#### 新建 `src/envs/lattice_v2_runner.py`

- `V2EpisodeState` dataclass：episode 全部可变状态集中管理
- `LatticeV2Runner.reset()` / `.step()` / `.get_metrics()`
- teacher dispatch 从脚本提取出来，cadence 明确

#### 瘦身 `scripts/_diag_l2c1_sweep.py`

- `run_episode()` 从 180 行 → 4 行（委托给 runner）
- sweep / print / condition 逻辑不变

#### 新建 `tests/test_v2_runner.py`（6 tests）

- 可复现 reset
- state schema 完整性
- step 推进
- episode 终止
- teacher mode 覆盖
- no_tutor baseline 合理性

### 设计决策

- 选 **Runner 而非 Gym wrapper**
  - Teacher cadence 不是标准 action space
  - risk_head 跨 episode 持久化
  - 不过度抽象
