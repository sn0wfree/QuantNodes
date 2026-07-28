---
id: L-101
title: 简单规则常胜复杂 IC 择时
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-103, L-133]
related_daily: [L-20260709-2, L-20260714-1]
source: 05_LESSONS_LIBRARY.md
---

# L-101: 简单规则常胜复杂 IC 择时

## 一句话总结
静态 value 单因子等权 baseline (Calmar 0.638) 远胜复杂 IC 择时模型。

## 问题描述
6 因子 IC 诊断发现：
- 仅 `value` 稳定正 IC (mean +0.044, hit 60%, IR 0.17)
- `reversal` / `dividend` 在 A 股是稳定负 alpha
- `low_vol` 是反指因子 (IC -0.454)
- 短期 IC 记忆 4-13 周最佳

## 检测 prompt (给 Agent 的检查清单)

当审查多因子策略时, 系统性地检查:

1. **是否有"简单 baseline 对照"**:
   - 在引入复杂加权前, 是否跑过"单因子等权" baseline?
   - 是否有 ablation: 移除最复杂部分后是否反而提升?

2. **IC 稳定性检查**:
   - 因子 IC mean 是否 > 0.02 (弱信号阈值)
   - Hit rate 是否 > 55%
   - IR (IC mean / IC std) 是否 > 0.10
   - 若 3 项中 2 项不达标, 该因子应被排除

3. **奥卡姆剃刀原则**:
   - 因子数 > 10 时, 复杂加权通常退步
   - 因子数 < 5 时, 等权优于 IC 加权
   - 因子间相关性 > 0.5 时, 加权效果会被稀释

## 正确做法

```python
# 错误: 复杂 6 因子 IC 加权
weights = compute_ic_weights(factor_panel)  # 过拟合

# 正确 1: 单因子等权 baseline
weights = equal_weight(top_n=10)

# 正确 2: 消融实验
for n_factors in [1, 3, 5, 10]:
    weights = top_n_equal_weight(n_factors)
    test_oos(weights)
# 选择 OOS Calmar 最高的 n_factors
```

## 关联代码案例

- v4 IC² 加权 (Calmar 0.613)
- v5.1 等权 + 逆波动 (Calmar 0.589, 反而更高)

## 历史教训来源
- 首次发现: v4 6 因子深度研究 (`e983294`, 2026-07-09)
- 实证数据: `docs/research_history/05_LESSONS_LIBRARY.md §L-101`