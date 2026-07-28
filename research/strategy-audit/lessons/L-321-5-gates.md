---
id: L-321
title: 5 道闸门（P0 任务清单）
severity: CRITICAL
auto_checkable: manual
category: integration
related_lessons: [L-201, L-203, L-211, L-213, L-322]
related_daily: [L-20260718-2]
source: 05_LESSONS_LIBRARY.md
---

# L-321: 5 道闸门

## 一句话总结
任何策略上 P0 必须经过 5 道闸门。

## 问题描述
| 闸门 | 内容 | 阈值 |
|------|------|------|
| 1. 数据闸门 | OHLCV 前复权 / 动态资产池 / 缺数据审计 | 50% 阈值、min_assets=10 |
| 2. IC 闸门 | 单因子 IC / 滚动 IC / 去重 / 阈值过滤 | |IC|>0.05 |
| 3. 因子闸门 | 宏观时序 vs PV 截面 / ADMM 正交化慎用 | 区分 + 残差化 |
| 4. OOS 闸门 | single → 5-fold → walk-forward → expanding + CV% | **CV% < 25%** |
| 5. 硬化闸门 | 起点依赖测试 / 死代码清理 / 文档 / 工厂函数 | 见各条 |

## 检测 prompt (给 Agent 的检查清单)

任何策略上 P0 任务前, 必须通过 5 道闸门。

## 正确做法

```python
# 5 道闸门 P0 任务清单
def five_gates_check(strategy):
    # Gate 1: 数据
    assert ohlcv_adjusted(strategy.data)
    assert strategy.min_assets >= 10
    assert gap_audit(strategy.data)

    # Gate 2: IC
    for factor in strategy.factors:
        ic = compute_ic(factor, returns)
        assert abs(ic) > 0.05, f"Factor {factor} IC {ic} < 0.05"

    # Gate 3: 因子
    assert macro_time_series_ic(strategy.macro_factors)
    assert no_symmetry_orthogonalization(strategy)

    # Gate 4: OOS
    cv_pct = compute_cv_pct(strategy)
    assert cv_pct < 0.25, f"CV% {cv_pct} >= 25%"

    # Gate 5: 硬化
    assert dead_code_clean(strategy)
    assert docs_updated(strategy)
    assert factory_function_exists(strategy)

    return True
```

## 历史教训来源
- 首次发现: v7.10 Stage 32 硬化 (`bbcaf86`)