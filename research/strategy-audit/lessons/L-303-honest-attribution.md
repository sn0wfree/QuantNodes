---
id: L-303
title: 诚实归因 + 状态降级（不要掩饰）
severity: HIGH
auto_checkable: manual
category: decision
related_lessons: [L-301]
related_daily: [L-20260710-3, L-20260714-3, L-20260728-3]
source: 05_LESSONS_LIBRARY.md
---

# L-303: 诚实归因 + 状态降级

## 一句话总结
任何"看起来像胜利"必须经过完整审计, 该降级就降级。

## 问题描述
实证:
- v3 Calmar 退化 → 报告 5.1 节明确列改进方向 (没有掩饰)
- v6.2 4/5 胜 → CV% 56.9% FAIL → 从 `PROMISING` 降为**研究版本**
- v7.10 OOS 0.671 → 修复 off-by-one → expanding → 0.466 (没有掩饰)

## 检测 prompt (给 Agent 的检查清单)

1. **是否有"看起来好"但未审计的结果**:
   - 必须经过 4 步 OOS 流程 (L-201)
   - 必须经过 CV% 测试 (L-203)

2. **状态机是否严格执行**:
   - < 25% PASS, 25-50% PROMISING, > 50% DEPRECATED
   - 该降级就降级

## 正确做法

```python
# 诚实归因示例
if metrics.calmar < expected:
    # 列出改进方向, 不要掩饰
    improvements = [
        "信号源需要切换 (反转 → 动量)",
        "加权方法需要优化 (1/N → Vol-parity)",
        "OOS 验证需要重做 (4 步流程)",
    ]
    status = "DEPRECATED"  # 该降级就降级
```

## 历史教训来源
- 首次发现: v3 5 项改进方向 + v6.2 状态降级 (`8be00ae`)