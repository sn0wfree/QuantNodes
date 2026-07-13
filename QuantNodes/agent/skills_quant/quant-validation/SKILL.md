---
name: quant-validation
description: 策略抗过拟合检验 — 起点依赖 / 调仓日偏移 / 参数扰动 / 消融, 适用于任何可回测策略.
---

# Quant Validation

对任意可回测策略运行 4 个抗过拟合检验, 输出 `validation_report.md` (含红黄绿结论).

## 工作流

1. **起点依赖测试** (`validate_starting_points`) — 在 [2018, 2020, 2022] 三个起点重跑, 检查 Calmar 波动率
2. **调仓日偏移测试** (`validate_rebalance_offsets`) — 调仓日 ±3/±5 个交易日偏移, 检查 Calmar 稳定性
3. **参数扰动测试** (`validate_parameter_perturbation`) — lookback ±15 天、corr_threshold ±0.05、上限 ±1, 检查 Calmar 始终 > 0.4
4. **消融实验** (`ablation`) — 关闭 4 条规则各做一次, 检查每关一项 Calmar 退化 ≥ 5%

## 工具集

| 工具 | 用途 |
|------|------|
| `run_strategy_backtest` | 跑一次回测, 返回 metrics (供 4 个检验内部调用) |
| `validate_starting_points` | 起点依赖 |
| `validate_rebalance_offsets` | 调仓日偏移 |
| `validate_parameter_perturbation` | 参数扰动 |
| `ablation` | 消融实验 |

## 验收标准 (默认)

- 起点依赖 Calmar 波动 ≤ 25%
- 调仓日偏移 Calmar 波动 ≤ 15%
- 参数扰动 Calmar 始终 > 0.4
- 消融: 每关一项 Calmar 退化 ≥ 5% (即每条规则都有贡献)

## 输出格式

`validation_report.md`:

```markdown
# 策略抗过拟合检验报告 — {strategy_name}

## 起点依赖
| 起点 | Calmar | 最大回撤 | 年化 |
|------|--------|----------|------|
| 2018-01-01 | 0.78 | -15% | 12% |
| 2020-01-01 | 0.85 | -18% | 14% |
| 2022-01-01 | 0.71 | -20% | 11% |
波动: 8.2% (< 25%, ✅ PASS)

## 调仓日偏移
...

## 总结
✅ 4/4 检验通过
```

## 反模式

- 不要用样本内最优点作为参数 (过拟合)
- 不要在检验过程中再调参 (二次过拟合)
- 不要忽视调仓日偏移导致的交易成本
- 消融实验中"关掉所有规则反而更好" = 严重过拟合警告
