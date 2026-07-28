---
id: L-203
title: 起点依赖 < 25% PASS / 25-50% PROMISING / > 50% DEPRECATED
severity: CRITICAL
auto_checkable: manual
category: oos_validation
related_lessons: [L-201, L-304]
related_daily: [L-20260714-3]
source: 05_LESSONS_LIBRARY.md
---

# L-203: 起点依赖 CV% 阈值

## 一句话总结
CV% 测试是 overfit gold standard, 阈值固定: 25%。

## 问题描述
| 策略 | CV% | 状态 |
|------|------|------|
| v6.2 5-fold OOS | 4/5 胜 | 看起来 PROMISING |
| v6.2 **起点依赖** | **56.9%** | **DEPRECATED** ⚠️ |
| v7.10 起点依赖 | 16.6% | PASS ⭐ |

## 检测 prompt (给 Agent 的检查清单)

任何策略上 P0 任务前, 必须做 CV% 测试:
1. 选择 3-5 个不同起点 (如 2018, 2020, 2022)
2. 每个起点独立加载原始数据 + 独立标准化
3. 计算每个起点的 OOS Calmar
4. CV% = std / |mean|

阈值:
- < 25% → PASS (维持 RECOMMENDED)
- 25-50% → PROMISING (降级, 需调参)
- > 50% → DEPRECATED (复刻 v6.2 命运)

## 正确做法

```python
# 起点依赖 CV% 测试
START_DATES = ["2018-01-01", "2020-01-01", "2022-01-01"]
calmars = []

for start in START_DATES:
    # 每个起点独立加载原始数据 + 独立标准化
    raw_data = load_raw_data()
    truncated = raw_data.loc[start:]
    X_std = standardize_independent(truncated)  # 不共享标准化

    calmar = run_backtest(X_std)
    calmars.append(calmar)

cv_pct = np.std(calmars) / abs(np.mean(calmars))
status = "PASS" if cv_pct < 0.25 else "PROMISING" if cv_pct < 0.50 else "DEPRECATED"
```

## 历史教训来源
- 首次发现: v6.2 CV% 56.9% FAIL (`223ef65`)