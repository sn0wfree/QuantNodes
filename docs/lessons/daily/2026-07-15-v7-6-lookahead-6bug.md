# 2026-07-15 — V7.6 未来函数 6 Bug + Sensitivity 全部 7 Phase

> **本日 commit 数**：21 个（**最密集技术修复日**）
> **主题**：V7.6 未来函数 6 个 Bug 修复 + Sensitivity 全部 Phase（1-7）+ 业绩曲线对比
> **阶段**：V7 关键修正期

---

## 今日 commits（按主题分组）

### Group 1: Sensitivity 测试执行（7 commit）
- `92a3961` — test: v7.6 sensitivity - Phase 1 完成（单参数 19 组）
- `64437b5` — test: v7.6 sensitivity - Phase 2（Hold-out 多段）
- `950420b` — test: v7.6 sensitivity - Phase 3（Bootstrap 稳定性）
- `adb7cda` — test: v7.6 sensitivity - Phase 4（缺失数据扰动）
- `f0a1083` — test: v7.6 sensitivity - Phase 5（构造层扰动）
- `c16252d` — test: v7.6 sensitivity - Phase 6（β_path 断点分析）
- `58f71eb` — test: v7.6 sensitivity - Phase 7（综合报告）

### Group 2: Top-N + TF/Regime 加固（3 commit）
- `b6e1679` — test: v7.6 top_n=5 起点 CV% 验证（5 combos × 5 起点）
- `5f67738` — test: v7.6 TF + Regime 加固测试（8 combos）
- `6488ada` — docs: 记录 top_n=5 验证 + TF/Regime 加固实验结果

### Group 3: 失效根因 + SL + 防御（3 commit）
- `f6cf429` — test: v7.6 2021/2022 失效根因深度分析（4 维度）
- `adec207` — test: v7.6 regime_combo 防御验证（9 combos）
- `71389b9` — test: v7.6 硬止损（SL）验证（8 combos）

### Group 4: 业绩曲线对比（2 commit）
- `d4b2d21` — test: v7.6 关键测试业绩曲线对比（7 个策略）
- `b0131a7` — chore: 保存修复前状态（v7.6 未来函数审计完成）

### Group 5: V7.6 未来函数修复（4 commit）⭐
- `0c1c6a4` — **fix: v7.6 未来函数修复（6个Bug）**
- `5d0ea88` — test: v7.6 修复后全量5起点测试结果
- `105b4b3` — wip: v7.6 X[t]→Y[t+1] 信号重设计（待验证，仍有同期Bug）
- `4be2ba3` — **fix: v7.6 彻底修复未来函数（信号-执行同期 + 日频NAV映射）**

### Group 6: TV-PR 标准实现（1 commit）
- `9d56a0b` — fix(v7.6): TV-PR ADMM 标准实现 + 全量样本估计

### Group 7: TV-PR 验证文档（1 commit）
- `a9d5908` — docs(v7.6): 更新 TV-PR 验证结果

---

## 当日教训（最关键的一天）

### L-20260715-1: X[t]→Y[t+1] 信号-执行同期是常见陷阱 [CRITICAL]

**问题**：`0c1c6a4` 修复 V7.6 未来函数 6 个 Bug。最关键 bug：

```python
# 错误: 用 t 周收益训练，预测 t+1 周
Y[t] = (NAV[t] / NAV[t-1]) - 1  # 默认 t-1 到 t
# 但 (X[t], Y[t]) → 这是同期训练, look-ahead!

# 正确: 用 t-1 周收益训练，预测 t 周
Y[t] = (NAV[t] / NAV[t-1]) - 1  # t-1 → t
X[t-1] → Y[t]  # 严格下一期
```

**根因**：
- 默认 `pct_change()` 计算的是 t-1 → t 的收益
- 如果用 (X[t], Y[t]) 训练 = 用 t 期信号预测 t 期收益 = 同期
- 必须 X[t-1] → Y[t]（X 用 t-1 之前的，Y 是 t-1 → t）

**正确做法**：
1. **训练时必须有明确的"t 定义"**
2. **写代码注释**：`X[t] → Y[t+1]`
3. **每个模型**：用 last-out-of-sample 测试一遍

**关联**：[05_LESSONS_LIBRARY §L-204](../research_history/05_LESSONS_LIBRARY.md) X[t]→Y[t+1] 信号-执行同期是常见陷阱

---

### L-20260715-2: Sensitivity 测试是 P0 任务，10 阶段覆盖 [HIGH]

**问题**：本日完成 Sensitivity 全部 7 Phase + 综合报告（Phase 8 在 07-14）：
- Phase 1：单参数 19 组（19 个超参数）
- Phase 2：Hold-out 多段（时间分段）
- Phase 3：Bootstrap 稳定性（165.95% 严重过拟合）
- Phase 4：缺失数据扰动（20% 缺失退化 -101%）
- Phase 5：构造层扰动（特征/标签构造）
- Phase 6：β_path 断点分析（β 估计稳定性）
- Phase 7：综合报告

**教训**：
1. **单测过 ≠ 稳健**：必须 Sensitivity 7+ 阶段
2. **Phase 3 Bootstrap CV 165.95%**：严重过拟合信号
3. **Phase 4 缺失 -101%**：必须成块缺失而非随机缺失

**正确做法**：
```python
# Sensitivity Phase 1: 单参数扫描
for param_name in ['lambda_l1', 'lambda_tv', 'top_n']:
    for value in [0.01, 0.05, 0.1, 0.15, 0.2]:
        test_oos({param_name: value})

# Phase 4: 成块缺失扰动（不是随机）
def block_missing(nav, frac=0.2):
    """模拟 20% 上市/退市型缺失"""
    blocks = random_block_split(nav, frac)
    return nav_with_blocks

# 不要用: random_na(frac=0.2)  # 不符合实际
```

**关联**：[05_LESSONS_LIBRARY §L-215](../research_history/05_LESSONS_LIBRARY.md) 缺失处理保持成块/连续缺失

---

### L-20260715-3: 未来函数修复要彻底，包含日频 NAV 映射 [CRITICAL]

**问题**：`4be2ba3` V7.6 彻底修复未来函数（信号-执行同期 + 日频 NAV 映射）。

**教训**：
1. **修复一期**：6 个 Bug 修复（`0c1c6a4`）
2. **仍有同期 Bug**：发现修复不彻底（`105b4b3`）
3. **最终修复**：信号-执行同期 + 日频 NAV 映射（`4be2ba3`）

**正确做法**：
1. **未来函数修复**：必须经过 4 步（详见 L-201）
2. **日频 NAV 映射**：周频信号 → 日频权重时，必须用 t-1 周的信号
3. **修复后必跑 expanding-window OOS**

**关联**：[05_LESSONS_LIBRARY §L-201](../research_history/05_LESSONS_LIBRARY.md) OOS 验证 4 步标准化流程

---

### L-20260715-4: 失败案例归档（2021/2022 失效根因深度分析） [MEDIUM]

**问题**：`f6cf429` v7.6 2021/2022 失效根因深度分析（4 维度）。

**教训**：
1. 任何策略必须做"失效根因分析"
2. 4 维度：宏观环境 + 信号质量 + 风险事件 + 组合配置
3. 不是为了"修复"，而是为了"理解失败模式"

**正确做法**：
```python
# 失效根因分析 4 维度
dimensions = {
    'macro': '2021 政策调控 + 2022 疫情',
    'signal': '动量信号在震荡市失效',
    'risk': '回撤集中爆发，无预警',
    'portfolio': '仓位过于集中',
}
```

**关联**：[05_LESSONS_LIBRARY §L-303](../research_history/05_LESSONS_LIBRARY.md) 诚实归因 + 状态降级

---

## 第二天的防范清单（07-16）

1. **因子 IC 评估**：区分截面 vs 时序
2. **增强因子库**：用 L-222 原则（macro 时序 vs PV 截面）
3. **未来函数彻底修复后**：保留 expanding-window 验证