# CaP-X vs Current Furniture Assembly Eval Pipeline Report

## 1. CaP-X 架构概览

CaP-X (Code-as-Policies for Manipulation) 是一个基于 Interactive Gymnasium 的在线代码执行与评测框架。其核心思想是 **LLM → Python Code → 物理模拟器 (Robosuite/OmniGibson)**。

- **工作流**：接收任务指令 (prompt) 及通过 Perception 模型 (SAM3等) 抓取的场景信息 -> LMM 生成调用机器人 API 的代码 -> 在基于 `exec()` 封装的沙箱中运行 -> 根据模拟器的状态返回 Success 或利用报错/视觉差异 (Visual Differencing) 进行多轮 (Multi-turn) try-retry。
- **底层支撑**：它极度依赖如 OpenAI 兼容接口的模型代理模块 (`openrouter_server.py`) 以及一系列专门抽象的机器人 API (例如 `FrankaControlApi`)。

## 2. 我们当前 Pipeline (Furniture) 概览

我们当前的系统主要是 **静态文件 -> 视觉解析 -> 离线文本/JSON 规划** 为主导。

- **工作流**：在 Trimesh 中离线跑完了 `100+` 个 RRT 轨迹，渲染为 MP4 及 `info.json`。Agent 看这套预处理的 MP4 与 prompt，输出对动作序列的理解 (e.g. 运动轨迹 JSON 的还原或部件组装顺序)。
- **核心逻辑**：基于 Few-Shot / In-context learning，高度定制的 `all_test_data_demo` 做样例对比，通过 `evaluate_gemini_motion.py` 在文本/数据层面对其输出的 JSON 结构跟真值进行正则提取及对比，属于 **Offline JSON Parsing Eval** 范畴。

## 3. 二者逐层对比

| 对比维度 | CaP-X | 当前 Furniture Pipeline |
| --- | --- | --- |
| **测试目标** | Code-as-Policies 控制能力与物理交互自纠错 | LMM 纯视觉空间推理、动作理解与 JSON 遵循能力 |
| **交互性质** | 在线闭环 (Online closed-loop) | 离线评估 (Offline batch evaluation) |
| **Agent 输出** | 纯 Python 代码 (含循环、感知判断) | 描述性文本或按格式填写的 JSON 数据 |
| **输入方式** | 实时的模拟器相机帧、状态与 API Docs 动态拼接 | 预渲染的全过程 MP4 动画与静态 Prompt + Few-shots |
| **成功判据** | 模拟器 `compute_reward() == 1.0` | `evaluate_gemini_motion` 里的文本/JSON正则匹配命中率 |

## 4. 缺失功能清单 (CaP-X 无法直接支持的我们需求)

完全可以说，直接把我们的任务强套到它头上是极度不合适的：

- **完全不支持预先录制的多视频变体对比** (`from_scratch / half_assembled_distractors`)：CaP-X 完全围绕 Gym 初始化展开，没有针对静态离线数据集循环跑测试的 harness。
- **没有 Offline Few-Shot Prompt 管理**：它没有自带像我们通过读取 demo 文件夹内的视频喂给 Gemini 做上下文示范的优雅组织形式。
- **评测维度的不同**：不支持通过正则或结构体对比检查 LLM 的规划准确率，它唯一的评测是“代码在这个环境下跑不报错且改变了标志位”。

## 5. 可桥接功能清单 (最值得借鉴的 5 个点)

1. **`SimpleExecutor` 代码沙箱与 `Tee` 截获机制**: 位于 `capx/envs/tasks/base.py`，非常优雅地使用 `sys.stdout` 截获与 `exec()` 来执行模型输出的代码，即便发生 Crash 也能把 trace 甩回给 LLM。
2. **Visual Differencing Module (VDM)**: 在 Multi-turn 里，如果首轮动作失败，它会通过 VDM 将“预期状态”与“当前图片”的差别总结成文本，再喂给 LLM 重写，这非常适合我们在纠错任务里的思路。
3. **OpenRouter 兼容代理思想**: `capx.serving.openrouter_server` 起了一个轻量 HTTP Server，统一接收标准 OpenAI payload 并分发。能解耦掉重型 SDK 的依赖。
4. **动态 API 文档拼装**: 不去手动写长串 Prompt，而是根据载入的 API Object 的 `__doc__` 自动收集可用的 Python 接口给 Agent 查阅。
5. **感知服务的独立守护进程管理**: 在 YAML 配置里利用 `api_servers` 区块，自动起子进程管理底层的几何或视觉模型，规避环境冲突。

## 6. 最小 Prototype 方案建议

为了借用它的思想而不破坏我们主线，建议开发一个名为 **`pipeline/run_code_exec_eval.py`** 的原型测试：

- **目标**：不要求 Agent 预测长长的 JSON 数组，而是改成要求它根据看 Furniture MP4，仅仅输出“几行调用 Trimesh 抽象指令”的 Python 伪代码 (e.g. `pick_part("leg_1")`, `attach_to("base")`)。
- **复用部分**：原有的 `PlanningEnv` 作为底层的 Sandbox `env`，提供一个 `run_code()` 方法；现有 JSON 数据仅用作环境初始化状态。
- **新增部分**：借用并精简一段 `SimpleExecutor` 跑 Agent 的代码。给 Agent 提供可调用的 API `['check_collision', 'snap_part', 'is_assembled']`。
- **评测标准**：评估标准不再是查 JSON 文本，而是这段 Python 代码能否顺利在 `PlanningEnv` 中执行不报错，并且模拟拼接完成后 `is_assembled()` 返回 `True`。
- **限制**：完全独立于现有的 `run_3cases_testing.py`，不干扰当前的 Batch 工作流。

## 7. 兼容性与 Go/No-go 最终结论

| 决策点 | 评级 | 结论说明 |
| --- | --- | --- |
| 直接替代现有 Pipeline | **基本不可行** | 逻辑范式是正交的，修改成本超过重写一遍，风险极高。 |
| 作为思想/机制参考 | **高** | 代码沙箱、动态多轮重试、API Docs动态下发等思路极富启发。 |
| 做轻量 Bridge Prototype | **中** | 若你想验证 "LLM写代码指挥拼接" 会不会比 "LLM写JSON轨迹" 准确率更高，这是值得实验的下一个切入点。 |
| 深度集成其仓库 | **低** | 依赖过重 (Robosuite/OmniGibson)，绝不建议污染现有 Trimesh 清爽生态。 |

### 🏁 下一步建议：No-go for direct integration; Go for conceptual extraction.
保持本仓库隔离，仅把 `envs.base` 的执行引擎思想手动用几个几十行的函数 “移植” 回我们 `pedagogical_ip` 的一个新分支做 prototype 即可。
