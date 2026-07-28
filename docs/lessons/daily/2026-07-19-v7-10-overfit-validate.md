# 2026-07-19 — V7.10 过拟合验证 + V7.7 look-ahead 修复

> **本日 commit 数**：2 个
> **主题**：V7.10 验证 TV-PR 过拟合 -64% + V7.7 修复 look-ahead
> **阶段**：V7 关键修正期

---

## 今日 commits

| hash | 类型 | 描述 |
|------|------|------|
| `688862d` | **fix(v7.10)**: 验证 TV-PR 过拟合 - Calmar 0.671→0.241 (-64%) |
| `f22c407` | fix(v7.7): 修复 look-ahead bias + Phase 1 重新验证 |

---

## 当日教训

### L-20260719-1: V7.7 Y[t] 和 f18_mom_short[t] 完全重叠，corr=0.96 [CRITICAL]

**问题**：`f22c407` V7.7 修复 look-ahead bias。看似"高预测力"的因子 (corr=0.96)，可能跟 Y 是同口径。

**反推**：
```python
# f18_mom_short[t] = nav[t] / nav[t-1] - 1   ← t-1 → t 收益
# Y[t]              = nav[t] / nav[t-1] - 1   ← t-1 → t 收益
# → 必然 corr=1.0 (or 0.96 with epsilon)
```

**修复**：
```python
# Y[t] = (nav[t+1] / nav[t]) - 1  # t → t+1
# f18_mom_short[t] = (nav[t] / nav[t-1]) - 1  # t-1 → t
# → corr 应该为 0 (independent tests)
```

**应用**：
1. **任何"高 IC 因子"**：先做"Y 重叠检查"
2. **corr=1.0 必有 bug**

**关联**：[05_LESSONS_LIBRARY §L-205](../research_history/05_LESSONS_LIBRARY.md) v7.7 Y[t] 和 f18_mom_short[t] 完全重叠

---

### L-20260719-2: "好结果" 必须经过 4 步 OOS 验证（第一步：严重程度） [CRITICAL]

**问题**：`688862d` V7.10 验证 TV-PR 过拟合：
- 初始显示 OOS Calmar: **0.671**（看起来太好了 ⚠️）
- 步骤 1: 过拟合验证 → **0.241**（**-64%, 严重过拟合**）

**4 步骤（完整版）**：
```
1. 验证过拟合严重程度 (688862d)    ← 本日
2. 修复 off-by-one bug (eaa6c9b)   ← 07-20
3. expanding-window 彻底消除 look-ahead (ead005c)  ← 07-20
4. 起点依赖 CV% < 25% PASS (cad8654)  ← 07-20
```

**应用**：
1. **任何 OOS "好结果"必须经此 4 步**
2. **缺一步都是危险的**

**关联**：[05_LESSONS_LIBRARY §L-201](../research_history/05_LESSONS_LIBRARY.md) OOS 验证 4 步标准化流程

---

## 第二天的防范清单（07-20）

1. **off-by-one 修复**：步骤 2
2. **expanding-window 实施**：步骤 3
3. **起点依赖 CV% 验证**：步骤 4
4. **诚实归因**：每步都更新文档