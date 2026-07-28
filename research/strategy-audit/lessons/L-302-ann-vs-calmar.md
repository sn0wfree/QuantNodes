---
id: L-302
title: 高 Ann ≠ 高 Calmar
severity: MEDIUM
auto_checkable: manual
category: decision
related_lessons: [L-125]
related_daily: [L-20260713-3, L-20260724-3]
source: 05_LESSONS_LIBRARY.md
---

# L-302: 高 Ann ≠ 高 Calmar

## 一句话总结
真正看 OOS Calmar / Sharpe / MaxDD 组合, 不要被"高年化"迷惑。

## 问题描述
| 策略 | Ann | Sharpe | Calmar |
|------|-----|--------|--------|
| Stage 12A (hybrid 单用) | **14.84%** | - | 1.17 |
| Stage 12A + VT | 6.28% | 1.51 | 1.60 ⭐ |
| v1.0 locked OOS | - | **1.51** | **1.79** ⭐ |

## 检测 prompt (给 Agent 的检查清单)

1. **是否只看年化收益**:
   - 应综合看 Calmar / Sharpe / MaxDD

2. **n_years 计算**:
   - 必须用日期差 (`(end - start).days / 365.25`)
   - 不要用 `len(returns) / 252`

## 正确做法

```python
# 错误: n_years 用数据点数
n_years = len(returns) / 252

# 正确: n_years 用日期差
n_years = (returns.index[-1] - returns.index[0]).days / 365.25
```

## 历史教训来源
- 首次发现: Stage 12A (`07956ca`) vs V1.0 OOS