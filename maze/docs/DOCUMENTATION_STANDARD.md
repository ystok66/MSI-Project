# 文档架构规范

这份文档定义 `maze/` 项目的文档应该怎么分层，避免以后出现：

- 所有说明都写在根目录
- 规格、结论、临时想法混在一起
- 旧结论和新结论互相覆盖
- 代码改了但没人知道该更新哪份文档

## 1. 文档总原则

- 文档按用途分层，不按“想到什么写什么”分层。
- 稳定规范和临时笔记分开放。
- 一份文档只负责一种主要用途。
- 文档标题要让人一眼知道它是“规范 / 架构 / 规格 / 记录”中的哪一种。

## 2. 文档目录约定

```text
docs/
├─ README.md
├─ CODEBASE_STRUCTURE.md
├─ DOCUMENTATION_STANDARD.md
├─ architecture/
├─ specs/
├─ adr/
└─ notes/
```

## 3. 每类文档的职责

### 根目录 `README.md`

只做入口，不做所有事情。

应该包含：

- 项目是什么
- 当前能跑什么
- 核心目录结构
- 关键文档链接
- 最短运行方式

不应该包含：

- 长篇设计细节
- 多轮实验日志
- 尚未落地的长规格

### `docs/architecture/`

放相对稳定的系统设计。

适合放：

- 场景机制概览
- env / learner / tutor 交互关系
- eval split 的设计理由
- 数据流和状态边界

不适合放：

- 今天调参时的零碎观察
- 尚未决定的具体 TODO 列表

### `docs/specs/`

放“准备做”的规格，而不是“已经实现”的事实。

适合放：

- `WAYPOINT` 机制规格
- richer inverse rollout 设计
- 新 eval 指标定义

每份 spec 最少要写清：

- 目标
- 范围
- 非目标
- 代码落点
- 验收标准

### `docs/adr/`

放关键架构决策记录。

适合放：

- 为什么当前先用启发式 inverse warning，而不是 full POMDP
- 为什么保持 `env / learner / tutor / runner` 四层
- 为什么 eval 要分 same-map 和 new-map

建议每份 ADR 包含：

- 背景
- 决策
- 替代方案
- 影响

### `docs/notes/`

放临时记录和实验笔记。

适合放：

- 某个 seed/参数区间的现象
- 调参记录
- 快速观察
- 待验证猜想

这里允许更轻量，但仍然要写日期和主题。

## 4. 文件命名规范

统一使用大写标题风格或明确前缀，避免含糊文件名。

推荐：

- `SCENARIO_OVERVIEW.md`
- `WAYPOINT_SPEC.md`
- `ADR_001_WARNING_BEFORE_WAYPOINT.md`
- `2026-04-20_seed22_observations.md`

不推荐：

- `newidea.md`
- `temp.md`
- `todo2.md`
- `notes_final_v3.md`

## 5. 什么时候必须更新文档

出现下面情况时，代码改动应同时更新至少一份文档：

- 目录结构变化
- 新增核心模块
- tutor action space 改变
- eval split 改变
- 关键 metric 含义改变
- 默认实验配置改变

简化规则：

- 改结构，更新 `CODEBASE_STRUCTURE.md`
- 改机制，更新 `architecture/` 或新增 `specs/`
- 改关键决策，新增 `adr/`
- 调参观察，写到 `notes/`

## 6. 文档写作格式规范

每份正式文档建议按这个顺序：

1. 一句话说明目的
2. 当前适用范围
3. 核心定义或规则
4. 与代码的对应位置
5. 后续扩展点或已知边界

正式文档尽量避免：

- 过多口语化重复
- 大段未分类想法堆叠
- 同一概念在不同章节反复改名

## 7. 当前推荐的维护方式

现在这个项目还在早期，所以文档不要一开始就膨胀成几十份。

当前推荐节奏：

1. 用 `README.md` 保持入口清楚
2. 用 `architecture/` 记录稳定机制
3. 用 `specs/` 承接下一步要实现的功能
4. 用 `notes/` 承接调参和观察
5. 重要决策再写入 `adr/`

这样能保持：

- 主线清楚
- 临时内容有地方放
- 以后不会再次回到“所有内容都堆在根目录”
