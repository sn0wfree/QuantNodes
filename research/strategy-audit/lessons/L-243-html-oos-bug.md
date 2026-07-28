---
id: L-243
title: HTML 表格 OOS 显示 bug 是工程易踩点
severity: HIGH
auto_checkable: agent
category: engineering
related_lessons: []
related_daily: [L-20260710-1]
source: 05_LESSONS_LIBRARY.md
---

# L-243: HTML OOS 显示 bug

## 一句话总结
OOS 表格必须用 `navs_A_with_bench.loc[OOS_START:]` 而非 `navs_A.loc[OOS_START:]`。

## 问题描述
```python
navs_A = ...  # 不含 v6
oos = navs_A.loc[OOS_START:]  # v6 在 navs_A 里没有列
# → L640 字典推导走 fallback metrics(full_metrics[col])
#   全期 Calmar 填进 OOS 格
# → v6 OOS 显示 0.748 (全期) 而非真实 OOS 0.662
```

## 检测 prompt (给 Agent 的检查清单)

1. **OOS 切窗口是否包含所有策略**:
   - 新增策略后, 检查 OOS 切窗口是否包含

2. **fallback 路径**:
   - 避免 silent fallback (全期指标伪装 OOS)

## 正确做法

```python
# 用 navs_A_with_bench.loc[OOS_START:] 重新切 OOS
# (v6 已在 navs_A_with_bench 中)
oos_navs = navs_A_with_bench.loc[OOS_START:]
oos_metrics = compute_metrics(oos_navs, oos_start=OOS_START)
```

## 历史教训来源
- 首次发现: `a60589c` (2026-07-10)