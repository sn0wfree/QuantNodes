---
id: L-215
title: 缺失处理保持成块/连续缺失，独立随机缺失不够
severity: MEDIUM
auto_checkable: manual
category: oos_validation
related_lessons: [L-203]
related_daily: [L-20260715-2]
source: 05_LESSONS_LIBRARY.md
---

# L-215: 成块缺失 vs 随机缺失

## 一句话总结
敏感性测试必须有"成块缺失"模式, 独立随机缺失不模拟真实上市/停牌。

## 问题描述
- 20% 随机缺失: 性能退化 -101%
- 20% 上市/退市型成块缺失: 性能退化更大

## 检测 prompt (给 Agent 的检查清单)

1. **敏感性测试是否有"成块缺失"模式**:
   - 不是简单的 `random_na(0.2)`
   - 应使用 `block_missing(0.2)`

## 正确做法

```python
# 错误: 随机缺失 (不符合实际)
def random_missing(nav, frac=0.2):
    """随机设置 20% 为 NaN"""
    mask = np.random.random(nav.shape) < frac
    return nav.where(~mask)

# 正确: 成块缺失 (模拟上市/退市)
def block_missing(nav, frac=0.2):
    """成块缺失: 模拟 20% 资产在某段连续时间内缺失"""
    n_assets = nav.shape[1]
    n_blocks = int(frac * n_assets)
    block_assets = np.random.choice(n_assets, n_blocks, replace=False)

    for asset in block_assets:
        # 随机选一段连续区间
        block_len = np.random.randint(20, 60)
        start = np.random.randint(0, len(nav) - block_len)
        nav.iloc[start:start+block_len, asset] = np.nan
    return nav
```

## 历史教训来源
- 首次发现: v7.6 Sensitivity Phase 4 (`adb7cda`)