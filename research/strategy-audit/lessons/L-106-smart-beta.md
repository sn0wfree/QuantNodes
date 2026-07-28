---
id: L-106
title: Smart β 是低 beta 工具不是 alpha 工具
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-121]
related_daily: [L-20260709-4]
source: 05_LESSONS_LIBRARY.md
---

# L-106: Smart β 是低 beta 工具不是 alpha 工具

## 一句话总结
Smart β 本质是 risk premium, 集中 2 ETF > 7 ETF 等权。

## 问题描述
- β=0.60, Up capture 0.66, Down capture 0.59
- 年化 alpha +7.79%/y 但 regime-conditional
- 2024-09 科创 50 起飞 = 失效起点
- 集中 2 ETF > 7 ETF: 512040 (价值) + 515100 (红利低波 100), Sharpe 0.83+

## 检测 prompt (给 Agent 的检查清单)

1. **Smart β 是否被定位为独立策略**:
   - 若是, 警惕"低波动 + 稳健上行"陷阱
   - 牛市失效, 熊市防御

2. **Smart β 与 regime detection 配合**:
   - 是否有 HMM / 趋势过滤器?
   - 若无, 在牛市中会被高 beta 资产抛离

3. **Smart β 数量**:
   - 7 只等权 Smart β 反而拖累
   - 2-3 只精选效果最佳

## 正确做法

```python
# 错误: Smart β 作为独立策略
weights = smart_beta_7etf_equal_weight()

# 正确: Smart β 作为"风险平价底仓"
weights_base = smart_beta_top2() * 0.30  # 30% 底仓
weights_alpha = momentum_weights * 0.70    # 70% 进攻层
final_weights = weights_base + weights_alpha
```

## 历史教训来源
- 首次发现: v4 Smart β 深度研究 (`181bf5a`, 2026-07-09)