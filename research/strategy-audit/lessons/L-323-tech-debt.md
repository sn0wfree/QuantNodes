---
id: L-323
title: 工程债的"识别 + 修复 + 预防"
severity: LOW
auto_checkable: manual
category: engineering
related_lessons: []
related_daily: [L-20260727-1, L-20260727-2, L-20260727-3]
source: 05_LESSONS_LIBRARY.md
---

# L-323: 工程债管理

## 一句话总结
所有已发现工程债应在下一个 P0 任务时修复, 不要累积。

## 已知工程债 (跨 V7-V10 整理)

- 子策略 NAV 跟踪 (V3 → V10 仍未完全修)
- FI+ stub (v2/fi_plus_v2.py 只有 64 行占位)
- 双口径 (5bp 成本 / 无成本) 需要持续维护
- 接口抽象 (`pc_raise NotImplementedError` 用于 ir_full)

## 检测 prompt (给 Agent 的检查清单)

1. **下一阶段 P0 任务时主动修复已发现债**:
   - 不要让债累积

## 正确做法

```python
# 工程债清单
tech_debt = {
    'sub_nav_tracking': 'V3 → V10 仍未完全修',
    'fi_plus_stub': '64 行占位',
    'dual_cost_model': '5bp vs 无成本',
    'interface_ir_full': 'NotImplementedError',
}

# P0 任务时: 至少修复 1 个工程债
for debt_name, description in tech_debt.items():
    fix_one_debt(debt_name, description)
```

## 历史教训来源
- 首次发现: 跨 V7-V10 整理