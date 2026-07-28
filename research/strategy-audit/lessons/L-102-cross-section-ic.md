---
id: L-102
title: 截面因子 vs 时序因子用不同 IC 计算方法
severity: HIGH
auto_checkable: agent
category: methodology
related_lessons: [L-222]
related_daily: [L-20260716-1]
source: 05_LESSONS_LIBRARY.md
---

# L-102: 截面因子 vs 时序因子用不同 IC 计算方法

## 一句话总结
宏观因子用时序 IC, PV 因子用截面 IC, 不能用通用 `compute_ic(factor_panel)`。

## 问题描述
宏观因子 (k≤16) 对所有资产截面值相同, 用截面 IC 会返回 NaN;
PV 因子 (k≥17) 衡量选股能力, 必须用截面 IC。

## 检测 prompt (给 Agent 的检查清单)

审查 IC 计算代码时, 检查:

1. **因子类型识别**:
   - 因子名包含 `macro` / `regime` / `vol_` / `cpi` / `gdp` → 宏观因子 (时序 IC)
   - 因子名包含 `pv` / `price` / `volume` / `momentum` → PV 因子 (截面 IC)

2. **IC 计算方法是否匹配**:
   ```python
   # 宏观因子 (时序 IC):
   ic = spearmanr(beta_k * X[t, 0, k], mean(Y[t+1, :]))

   # PV 因子 (截面 IC):
   ic = spearmanr(contrib[t, :], Y[t+1, :])
   ```

3. **是否有通用 IC 函数**:
   - 若 `compute_ic(factor_panel)` 一个函数处理所有因子 → 警告
   - 应拆分为 `compute_ic_cross_section()` + `compute_ic_time_series()`

## 正确做法

```python
def compute_ic_cross_section(factor_panel: np.ndarray, returns: np.ndarray) -> float:
    """PV 因子截面 IC: spearmanr(contrib, returns)"""
    return spearmanr(factor_panel.flatten(), returns.flatten()).correlation

def compute_ic_time_series(beta: np.ndarray, factor: np.ndarray, returns: np.ndarray) -> float:
    """宏观因子时序 IC: spearmanr(beta * factor, mean(returns))"""
    signal = (beta * factor).flatten()
    target = returns.mean(axis=1).flatten()
    return spearmanr(signal, target).correlation
```

## 关联代码案例

- v7.6 修复前 (`5c30172`): 宏观因子 IC 全为 NaN
- v7.6 修复后: 宏观时序 IC 0.14-0.30 (alpha 主力)

## 历史教训来源
- 首次发现: v7.11 (`5c30172`, 2026-07-20)