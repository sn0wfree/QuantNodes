---
id: L-232
title: YAML 配置驱动让"策略配置"与"策略实现"解耦
severity: MEDIUM
auto_checkable: manual
category: engineering
related_lessons: [L-231]
related_daily: [L-20260720-5]
source: 05_LESSONS_LIBRARY.md
---

# L-232: YAML 配置驱动

## 一句话总结
6 个 YAML 模板 (v1.0/v2/v3/v4/v6/v7.10), 用户零代码改动即可切换策略。

## 问题描述
```yaml
# v7.10.yaml
lambda_tv: 0.15
lambda_l1: 0.05
method: expanding
```

## 检测 prompt (给 Agent 的检查清单)

1. **策略配置是否硬编码在代码中**:
   - 应改为 YAML 配置

2. **是否有 6 个 YAML 模板**:
   - v1.0/v2/v3/v4/v6/v7.10

## 正确做法

```python
# run_from_yaml('v7.10.yaml')
config = load_yaml('v7.10.yaml')
strategy = V7_10Strategy(config)
result = backtest(strategy)
```

## 历史教训来源
- 首次发现: YAML 期 (`c9eb84c`, 2026-07-20)