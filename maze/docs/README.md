# Docs Index

这个目录用于区分“代码怎么组织”和“研究内容怎么记录”，避免以后把设计、实验结论、临时想法都堆在根目录。

## 文档分类

- [代码结构规范](./CODEBASE_STRUCTURE.md)
  说明包结构、模块职责、命名和 import 规则。
- [文档架构规范](./DOCUMENTATION_STANDARD.md)
  说明哪些内容该写到哪里，哪些文档是长期规范，哪些只是临时记录。
- [下一步实现指南](./specs/ANTIGRAVITY_NEXT_STEP_GUIDE.md)
  给 Antigravity 的下一阶段 maze 实现规范，覆盖 inverse tutor、metrics、ablation、实验与验收计划。
- [场景概览](./architecture/SCENARIO_OVERVIEW.md)
  说明当前 `risky_maze` 原型的机制边界和最小版本范围。
- `risky_maze/scenarios/`
  放固定地图 spec、后续 `v1/v2` 变体和通用 validator。
- [architecture/](./architecture/README.md)
  放稳定的系统设计文档。
- [specs/](./specs/README.md)
  放准备落代码的功能规格。
- [adr/](./adr/README.md)
  放关键架构决策记录。
- [notes/](./notes/README.md)
  放短期实验记录、调参笔记、临时观察。

## 当前推荐阅读顺序

1. 先看根目录 [README.md](../README.md)
2. 再看 [场景概览](./architecture/SCENARIO_OVERVIEW.md)
3. 然后看 [代码结构规范](./CODEBASE_STRUCTURE.md)
4. 如果要继续扩机制，再看 [文档架构规范](./DOCUMENTATION_STANDARD.md)
