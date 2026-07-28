---
id: L-322
title: 4 步 OOS 流程（L-201 重申）
severity: CRITICAL
auto_checkable: manual
category: oos_validation
related_lessons: [L-201]
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-322: 4 步 OOS 流程

## 一句话总结
4 步 OOS 流程从流程维度重申。

## 4 步骤

```
① 验证过拟合严重程度 (CV% / Bootstrap / 单起点)
② 修复 off-by-one bug (索引/标签/执行周期)
③ expanding-window 消除 look-ahead (严格 X[t]→Y[t+1])
④ 起点依赖 CV% < 25% PASS
```

## 正确做法

详见 L-201 完整流程。

## 历史教训来源
- 与 L-201 重复但强调"流程"维度