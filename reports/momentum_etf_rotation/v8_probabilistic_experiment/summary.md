# v8 概率化 Jump Model 实验报告 (含交易成本)

**日期**: 2026-07-22

## 1. 实验设计

- **测试资产**: 510300 / 511260 / 518880 / 159915 / 512760
- **Walk-Forward 起点**: ['2018-01-01', '2019-01-01', '2020-01-01']
- **训练窗口**: 1000 天, **测试窗口**: 252 天, **步长**: 60 天
- **测试成本**: [0, 10, 20] bp/单边 (含 0/10/20 三档)
- **成本模型**: NAV[t] = NAV[t-1] × (1+adj_ret) × max(1-turnover×cost_bp/10000, 0)

## 2. 4 个对比版本

| 版本 | 状态输出 | 仓位计算 | 调参 |
|------|----------|----------|------|
| v8_method_b | 硬分类 0/1 | 阈值 0.25 段 | bear_threshold=0.25 |
| v8_prob_2state | 概率 P(bull), P(bear) | P·[1.0, 0.0] | **零调参** |
| v8_prob_3state | 概率 P(bull), P(neutral), P(bear) | P·[1.0, 0.6, 0.0] | **零调参** |
| v8_uniform | 等权 100% | 1.0 | **零调参** |

## 3. 成本 = 0 bp/单边

| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar |
|------|-----------|---------|-----------|-----------|------------|
| v8_method_b | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |
| v8_prob_2state | 12.33% | 11.04% | **1.284** | -19.28% | **1.142** |
| v8_prob_3state | 13.16% | 11.78% | **1.285** | -20.34% | **1.146** |
| v8_uniform | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |

**关键对比 (vs v8_method_b)**:

- v8_prob_2state: Sharpe -0.013, Calmar -0.060
- v8_prob_3state: Sharpe -0.012, Calmar -0.056
- v8_uniform: Sharpe +0.000, Calmar +0.000

## 3. 成本 = 10 bp/单边

| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar |
|------|-----------|---------|-----------|-----------|------------|
| v8_method_b | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |
| v8_prob_2state | 12.33% | 11.04% | **1.284** | -19.28% | **1.142** |
| v8_prob_3state | 13.16% | 11.78% | **1.285** | -20.34% | **1.146** |
| v8_uniform | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |

**关键对比 (vs v8_method_b)**:

- v8_prob_2state: Sharpe -0.013, Calmar -0.060
- v8_prob_3state: Sharpe -0.012, Calmar -0.056
- v8_uniform: Sharpe +0.000, Calmar +0.000

## 3. 成本 = 20 bp/单边

| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar |
|------|-----------|---------|-----------|-----------|------------|
| v8_method_b | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |
| v8_prob_2state | 12.33% | 11.04% | **1.284** | -19.28% | **1.142** |
| v8_prob_3state | 13.16% | 11.78% | **1.285** | -20.34% | **1.146** |
| v8_uniform | 24.81% | 22.09% | **1.297** | -33.18% | **1.202** |

**关键对比 (vs v8_method_b)**:

- v8_prob_2state: Sharpe -0.013, Calmar -0.060
- v8_prob_3state: Sharpe -0.012, Calmar -0.056
- v8_uniform: Sharpe +0.000, Calmar +0.000

## 4. 成本敏感性分析 (同一版本 Sharpe vs 成本)

| 版本 | 0bp Sharpe | 10bp Sharpe | 20bp Sharpe | 10bp 损失 | 20bp 损失 |
|------|-----------|-------------|-------------|-----------|----------|
| v8_method_b | 1.297 | 1.297 | 1.297 | +0.000 | +0.000 |
| v8_prob_2state | 1.284 | 1.284 | 1.284 | +0.000 | +0.000 |
| v8_prob_3state | 1.285 | 1.285 | 1.285 | +0.000 | +0.000 |
| v8_uniform | 1.297 | 1.297 | 1.297 | +0.000 | +0.000 |

## 5. 最终判定

**标准成本 (10bp) 下排序**:

1. **v8_method_b**: Sharpe=1.297
2. **v8_uniform**: Sharpe=1.297
3. **v8_prob_3state**: Sharpe=1.285
4. **v8_prob_2state**: Sharpe=1.284

**结论**: v8_method_b 在 10bp 成本下最优 (Sharpe=1.297, vs v8_method_b: +0.000)

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| `comparison_walkforward.csv` | walk-forward 详细结果 (含 cost_bp 维度) |
| `stability_summary.md` | 跨起点/跨资产稳定性 (按成本分层) |
| `equity_curves.png` | 4 版本指标对比图 |
| `state_probability.png` | 状态概率时间线示例 |
| `summary.md` | 本报告 |
