---
id: L-201
title: OOS 验证 4 步标准化流程 ⭐⭐⭐
severity: CRITICAL
auto_checkable: manual
category: lookahead
related_lessons: [L-202, L-203, L-322]
related_daily: [L-20260715-1, L-20260719-2, L-20260720-1]
source: 05_LESSONS_LIBRARY.md
---

# L-201: OOS 验证 4 步标准化流程

## 一句话总结
任何"好结果"必须经过 4 步, 缺一步都是危险的。

## 问题描述
v7.10 OOS 4 commit 链:
```
1. 验证过拟合严重程度 (688862d)
   ↓
2. 修复 off-by-one bug (eaa6c9b)
   ↓
3. expanding-window 彻底消除 look-ahead (ead005c)
   ↓
4. 起点依赖 CV% < 25% PASS (cad8654)
```

实证对比:
| 步骤 | Calmar | 解释 |
|------|--------|------|
| 初始显示 OOS | **0.671** | 看起来太好了 ⚠️ |
| (1) 过拟合验证 | **0.241** | -64%, 严重过拟合 |
| (2) off-by-one 修复 | **0.486** | -28%, bug 不只是过拟合 |
| (3) expanding OOS (2022+) | **0.662** | 真实 OOS |
| (4) expanding OOS (2023+) | **1.121** | 单段稳定性 |
| **final 真实 OOS (全期)** | **0.466** ⭐ | 这是最终真相 |

## 检测 prompt (给 Agent 的检查清单)

任何 OOS "好结果" 报告必须含 4 步骤, 否则标记为 INCOMPLETE。

## 正确做法

```python
# 步骤 1: 验证过拟合严重程度
# 故意破坏 look-ahead, 看指标退化多少
broken_sharpe = test_with_intentional_leakage()
real_sharpe = test_without_leakage()
overfit_severity = (broken_sharpe - real_sharpe) / broken_sharpe
assert overfit_severity < 0.30, f"Overfit {overfit_severity:.1%} > 30%"

# 步骤 2: 修复 off-by-one bug
for shift in [-1, 0, 1]:
    metrics = test_with_shift(shift)
# 不同 shift 给出显著不同 metrics → 有 off-by-one bug

# 步骤 3: expanding-window 彻底消除 look-ahead
beta = expanding_window_tvpr(X, Y, window=252)
# 不使用 full_sample

# 步骤 4: 起点依赖 CV% < 25% PASS
calmars = [test_with_start(s) for s in [2018, 2020, 2022]]
cv_pct = np.std(calmars) / abs(np.mean(calmars))
assert cv_pct < 0.25, f"CV% {cv_pct:.1%} >= 25%"
```

## 历史教训来源
- 首次发现: v7.10 OOS 4 commit 链 (`688862d` → `eaa6c9b` → `ead005c`)