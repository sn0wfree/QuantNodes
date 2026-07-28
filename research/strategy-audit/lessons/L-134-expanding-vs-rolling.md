---
id: L-134
title: 时变 LASSO（rolling 156w）实测退步
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-101, L-133]
related_daily: [L-20260714-2]
source: 05_LESSONS_LIBRARY.md
---

# L-134: 时变 LASSO rolling 退步

## 一句话总结
expanding 估计比 rolling 156w 更稳定。

## 问题描述
| 窗口 | Calmar |
|------|--------|
| expanding | 0.981 |
| rolling 156w | **0.333 (-66%)** ❌ |

根因:
- rolling 窗口引入"老数据丢弃", LASSO 学到的模式是短暂的
- expanding 累积所有历史, β 估计更稳定
- 156 周样本量不足以稳定估计 36+ 因子

## 检测 prompt (给 Agent 的检查清单)

1. **时变估计窗口**:
   - 优先 expanding
   - rolling 仅做"稳健性测试", 不放生产

2. **窗口大小**:
   - 必须 ≥ 8x 因子数 (36 因子 → 至少 288 周)

## 正确做法

```python
# 推荐: expanding
beta = estimate_lasso(X_expanding, Y_expanding)

# 不推荐: rolling (除非因子数 ≥ 288)
beta = estimate_lasso(X_rolling_288w, Y_rolling_288w)
```

## 历史教训来源
- 首次发现: v7.5 Step 3 (`5f44fbf`, 2026-07-14)