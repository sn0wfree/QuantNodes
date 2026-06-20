# QuantNodes 设计文档索引

> **当前架构版本**：v1.0  
> **最新更新**：2026-05-08

---

## 文档导航

| 序号 | 文档 | 定位 | 状态 |
|------|------|------|------|
| 04 | [架构设计.md](./04-架构设计.md) | 三层架构 + Expression DSL + 序列化 | ✅ 已确认 |
| 07 | [执行清单.md](./07-执行清单.md) | 9阶段执行追踪 + 基础设施修复 | ✅ 全部完成 |
| 13 | [Agent架构设计.md](./13-Agent架构设计.md) | Agent系统 + Config-Driven配置驱动 | ✅ 已完成 |
| 22 | [算子系统设计与规范.md](./22-算子系统设计与规范.md) | 算子扩展机制 + 重构计划 + TA-Lib集成 | ✅ 已完成 |
| 24 | [核心功能框架设计.md](./24-核心功能框架设计.md) | 三大核心功能详细设计 | ✅ 已完成 |
| 规范 | [大型项目开发测试规范.md](./大型项目开发测试规范.md) | 开发测试流程规范 | ✅ 已完成 |
| 操作 | [QuantNodes-操作手册.md](./QuantNodes-操作手册.md) | 完整操作指南 | ✅ 新增 |
| Agent | [Agent-策略构建操作手册.md](./Agent-策略构建操作手册.md) | 小白 Agent 使用指南 | ✅ 新增 |
| QuickStart | [QuickStart.md](./QuickStart.md) | 5 分钟快速入门 | ✅ 新增 |

---

## 阅读建议

### 新人快速上手

1. ⭐ **[QuickStart.md](./QuickStart.md)** — 5 分钟快速入门
2. **[Agent-策略构建操作手册.md](./Agent-策略构建操作手册.md)** — 通过自然语言让 Agent 帮你构建策略
3. **[04-架构设计.md](./04-架构设计.md)** — 总览三层架构、BaseNode、Pipeline、Expression DSL

### 功能开发

- **功能1 (Config-Driven回测)**: 13 (Config-Driven章节) + 24 (功能1详细章节)
- **功能2 (策略监控)**: 24 (功能2章节)
- **功能3 (量化研究)**:
  - 3A: 因子库 (Wiki代理层) — 24 (功能3A章节)
  - 3B: 研报复现 — 24 (功能3B章节)
  - 3C: AutoResearch — 24 (功能3C章节)

### 架构深入

- **节点体系**: 04 (Expression DSL + 序列化)
- **因子算子**: 22 (扩展机制 + 重构)
- **Agent系统**: 13 (完整Agent架构)

---

## 文档结构

```
04-架构设计.md ─────────────────────────┐
                                          ├─► 核心架构
07-执行清单.md ──────────────────────────┤
                                          │  + 项目进度

13-Agent架构设计.md ─────────────────────┤
                                          ├─► Agent层
24-核心功能框架设计.md ───────────────────┤
                                          ├─► 产品功能

22-算子系统设计与规范.md ────────────────┘
                                          └─► 算子层
```

---

## 归档文档

落地后的过程文档（实施计划 / 重构方案 / 修复记录）已迁入 [`archived/`](./archived/)。

`archived/` 包含两类:
1. **2026-05-08 之前**的历史文档快照
2. **B6 (2026-06-20) 整理**的过程文档:
   - AgentChat-* 系列重构方案 (Layout/Phase0/Refactor/UI-Enhancement)
   - AgentMemory-Persistence-Plan / AgentWebUI-Enhancement-Plan
   - AgentPhase3 / AgentPhase4 实施计划
   - Feature3A / 3B / 3C 实施计划
   - Fix-DoubleSave / Fix-ModelSwitch / Fix-P1-AgentLoop
   - QuantNodes-Frontend-Implementation-Plan / Architecture-Analysis-2026-05-13 / ARCHITECTURE_CHANGE
   - 代码质量修复计划

   注: Feature3D（用户友好自定义算子 API）保留在主目录, 因相关 API 仍在演进。

---

## 项目当前状态 (2026-05-08)

| 指标 | 数值 |
|------|------|
| 测试用例 | 2574+ |
| 算子数量 | 317+ |
| ruff 错误 | 148 (已修复 518 个) |
| mypy 错误 | 已修复主要问题 |
| 完成度 | ~95% |

---

**最后更新**：2026-05-08
