---
id: L-202
title: full-sample 含前视偏差反致过拟合
severity: CRITICAL
auto_checkable: agent
category: lookahead
related_lessons: [L-201, L-223]
related_daily: [L-20260728-1]
source: 05_LESSONS_LIBRARY.md
---

# L-202: full-sample 含前视偏差反致过拟合

## 一句话总结
expanding OOS 优于 full_sample 是真理: Sharpe 1.11 → 1.57 (+41%), Calmar 0.84 → 2.25 (+168%)。

## 问题描述
| 指标 | full_sample | expanding | expanding 优势 |
|------|------------|-----------|------------|
| Sharpe | 1.11 | **1.57** | **+41%** ⭐ |
| Calmar | 0.84 | 2.25 | **+168%** ⭐ |

## 检测 prompt (给 Agent 的检查清单)

审查标准化代码时, 系统性地检查以下问题:

### 1. 找出所有标准化调用
- `X.mean()` / `X.std()` / `X.var()` (全样本统计)
- `StandardScaler()` / `MinMaxScaler()` (sklearn)
- `quantile()` / `rank()` (可能含未来)

### 2. 评估标准化的"方向"
- **时间维标准化**: 沿 axis=0 (默认), 用于宏观因子 ✅
- **截面标准化**: 沿 axis=1 (用 .T), 用于 PV 因子 ✅
- **滚动标准化**: `.rolling(N).mean()`, 用于任何时间序列 ✅

### 3. 判断是否违规
如果满足以下条件, 则判定 VIOLATED:

A. 在策略**回测主循环内**调用了全样本统计
```python
for date in dates:
    mean = X.mean()              # ❌ CRITICAL
    std = X.std()                # ❌ CRITICAL
    weights = (X - mean) / std
```

B. 在 **fit / train** 函数中用了全样本统计 (而非 rolling)
```python
def standardize(X):
    return (X - X.mean()) / X.std()  # ❌ CRITICAL
```

C. 标准化函数被多次调用, 每次**未指定窗口** (rolling/expanding)

### 4. 排除合法用法
以下情况**不算违规**:
- 使用 `.rolling(N).mean()` / `.expanding().mean()`
- sklearn `StandardScaler` 只用于 train set (没 transform test)
- 标准化在 `__init__` / `setup` 阶段一次性完成, 且用于 visualization (非训练)
- 计算"因子 IC" 等分析用途 (非策略训练)

## 正确做法

```python
# 错误: full_sample
def standardize(X):
    return (X - X.mean()) / X.std()

# 正确 1: rolling
def standardize_rolling(X, window=252):
    mean = X.rolling(window).mean()
    std = X.rolling(window).std()
    return (X - mean) / std

# 正确 2: expanding
def standardize_expanding(X, min_periods=252):
    mean = X.expanding(min_periods).mean()
    std = X.expanding(min_periods).std()
    return (X - mean) / std

# 正确 3: 区分 macro / PV
def standardize_smart(X, is_macro=False):
    if is_macro:
        return (X - X.expanding().mean()) / X.expanding().std()
    else:
        return ((X.T - X.T.mean()) / X.T.std()).T
```

## 关联代码案例

### 错误 (v7.10 早期)
```python
# v7/data_loader_v7_6.py:142
def standardize_v7_10(X, factor_names):
    """标准化 v7.10 因子数据 (含未来 bug)"""
    mean = X.mean(axis=(0, 2))  # ❌ 全样本均值
    std = X.std(axis=(0, 2))    # ❌ 全样本标准差
    return (X - mean) / std
```

## 历史教训来源
- 首次发现: v7.6 未来函数修复 (`4be2fa3`)
- v7.10 DEPRECATED: 全样本标准化 (`847d6ec` + `9a07ca3`)