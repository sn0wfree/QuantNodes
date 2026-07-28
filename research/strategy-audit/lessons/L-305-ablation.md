---
id: L-305
title: 共享代码改动要小心，消融实验是必要的
severity: MEDIUM
auto_checkable: manual
category: decision
related_lessons: []
related_daily: [L-20260709-12]
source: 05_LESSONS_LIBRARY.md
---

# L-305: 共享代码改动 + 消融实验

## 一句话总结
v5.1 共享 `cross_section_zscore` 改一处影响两策略, 必须 4 项 S 单独测试。

## 问题描述
| S 组合 | OOS Calmar | 改善 |
|--------|-----------|------|
| v5.1 baseline | 0.586 | - |
| S2 单独 ❌ | 0.516 | -11.9% 严重拖累 |
| S1+S3+S4 (无 S2) ⭐⭐ | 0.604 | **+3.1%** |
| S1+S2+S3+S4 ❌ | 0.509 | -13.1% |

根因: S2 失败的根因是 rank-based 信息损失。

## 检测 prompt (给 Agent 的检查清单)

1. **共享代码改动是否有 ab 测试**:
   - 必须有消融实验

2. **"消融直觉"是否验证**:
   - 不能凭直觉做改动

## 正确做法

```python
# 共享代码改动前: 消融实验
ablation_results = {}
for ablation in ['S1', 'S2', 'S3', 'S4']:
    test_oos(ablation)
    ablation_results[ablation] = compute_metrics()

# 选择改善最大的组合
best_combo = max(ablation_results, key=ablation_results.get)
```

## 历史教训来源
- 首次发现: v5.1 共享 `cross_section_zscore` 改一处影响两策略