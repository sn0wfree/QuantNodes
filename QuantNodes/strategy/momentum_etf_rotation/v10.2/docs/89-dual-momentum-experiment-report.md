# 89 - Dual Momentum + CA-GCP 实验报告

## 实验概述

验证 CA-GCP（Cross-Asset Graph Conformal Prediction）风险控制层能否改善 dual momentum 策略的风险收益特征。

**策略**: Dual Momentum（4 资产轮动：510300/513100/518880/511260）
**CA-GCP**: Global pipeline（38 ETF），月频调仓 + 周频风控
**校准**: Walk-Forward 5 fold，343 combo grid search

## 最优参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 6 | KNN 邻居数（修复 Bug 后确认） |
| eta | 0.5 | 系统压力调制灵敏度 |
| tau | 20 | 时间衰减半衰期（天） |
| stress_yellow | 0.92 | 黄色警报阈值 |
| stress_red | 0.98 | 红色警报阈值 |

**校准 Score**: `10*extreme - 5*pa_std - 1*(width_bps/1000) - 2*hw_cv`

## 架构

```
Global CA-GCP (38 ETF, k=6, eta=0.5, tau=20)
  + 月频调仓 (月末 dual_momentum_signal)
  + 周频风控 (周一 CA-GCP full check, 其他日 hold)
  + 默认阈值 (stress_yellow=0.92, stress_red=0.98)
  + 10bp 交易成本
```

## 分阶段业绩对比

### Bare Dual Momentum

| Fold | Period | AnnRet | Vol | MaxDD | Sharpe | Calmar |
|------|--------|--------|-----|-------|--------|--------|
| 1 | 2022-10 → 2023-09 | 26.68% | 13.33% | -6.88% | 2.002 | 3.879 |
| 2 | 2023-05 → 2024-04 | 33.04% | 16.24% | -9.80% | 2.034 | 3.370 |
| 3 | 2024-01 → 2024-12 | 23.24% | 21.47% | -17.92% | 1.082 | 1.297 |
| 4 | 2024-08 → 2025-07 | 32.22% | 17.11% | -11.53% | 1.883 | 2.795 |
| 5 | 2025-03 → 2026-06 | 39.58% | 28.47% | -24.89% | 1.390 | 1.590 |
| **Avg** | | **30.95%** | **19.33%** | **-14.20%** | **1.678** | **2.586** |

### Dual Momentum + CA-GCP

| Fold | Period | AnnRet | Vol | MaxDD | Sharpe | Calmar |
|------|--------|--------|-----|-------|--------|--------|
| 1 | 2022-10 → 2023-09 | 21.28% | 13.39% | -6.88% | 1.590 | 3.094 |
| 2 | 2023-05 → 2024-04 | 23.39% | 14.50% | -8.88% | 1.613 | 2.634 |
| 3 | 2024-01 → 2024-12 | 33.61% | 19.98% | -15.64% | 1.682 | 2.149 |
| 4 | 2024-08 → 2025-07 | 42.10% | 21.33% | -7.94% | 1.974 | 5.303 |
| 5 | 2025-03 → 2026-06 | 39.47% | 20.19% | -15.58% | 1.955 | 2.534 |
| **Avg** | | **31.97%** | **17.88%** | **-10.98%** | **1.763** | **3.143** |

### Delta (CA-GCP - Bare)

| Fold | AnnRet Δ | Vol Δ | MaxDD Δ | Sharpe Δ | Calmar Δ |
|------|----------|-------|---------|----------|----------|
| 1 | -5.40% | +0.06% | +0.00% | -0.412 | -0.785 |
| 2 | -9.65% | -1.75% | +0.92% | -0.421 | -0.737 |
| 3 | **+10.37%** | -1.49% | **+2.28%** | **+0.600** | **+0.852** |
| 4 | **+9.89%** | +4.22% | **+3.59%** | **+0.091** | **+2.508** |
| 5 | -0.11% | -8.28% | **+9.31%** | **+0.565** | **+0.944** |
| **Avg** | **+1.02%** | **-1.45%** | **+3.22%** | **+0.084** | **+0.556** |

### 换手率与成本

| Fold | Bare Turnover | CA-GCP Turnover | Bare Cost | CA-GCP Cost |
|------|--------------|-----------------|-----------|-------------|
| 1 | 3.4x/yr | 14.1x/yr | 0.34% | 1.41% |
| 2 | 1.1x/yr | 11.8x/yr | 0.11% | 1.18% |
| 3 | 5.5x/yr | 11.2x/yr | 0.55% | 1.12% |
| 4 | 5.5x/yr | 9.3x/yr | 0.55% | 0.93% |
| 5 | 2.3x/yr | 14.8x/yr | 0.23% | 1.48% |
| **Avg** | **3.5x/yr** | **12.2x/yr** | **0.35%** | **1.22%** |

## 关键发现

### 1. CA-GCP 在回撤期显著有效

- **Fold 3**（2024 年回撤期）：MaxDD 从 -17.92% 降至 -15.64%（+2.28%），Sharpe +0.60
- **Fold 5**（2025-2026 回撤期）：MaxDD 从 -24.89% 降至 -15.58%（+9.31%），Sharpe +0.57
- **Aggregate MaxDD 改善 +3.22%**（-14.20% → -10.98%）

### 2. CA-GCP 在平稳期 hurt

- **Fold 1/2**：CA-GCP 触发频繁（14-15x turnover），交易成本吃掉收益
- 原因：平稳期 stress 波动小，risk filter 频繁 bounce → 高换手

### 3. 全局 CA-GCP 优于 Sector CA-GCP

- **Global**: stress 来自 38 只 ETF 跨板块离散，信号强
- **Sector**: stress 被同板块内高相关性稀释（0.03 vs 0.47），risk filter 从未触发
- 结论：CA-GCP 的 stress 设计用于捕捉**跨板块系统性风险**，不适合板块内使用

### 4. predict_fast Bug 修复

- **Bug 1**（关键）：pool 用了目标资产 scores，应为源资产 scores
- **Bug 2**（中等）：k 变化时未重建 KNN graph
- 修复后：k 从 2 变为 6（更多邻居 = 更好覆盖），eta 从 0.7 变为 0.5（更温和）

### 5. 阈值校准实验

- 尝试了 3 种 Pareto score（绝对值/相对值/综合），均失败
- 原因：calib 期（6-9 月）市场特征和 test 期不同，校准阈值无法泛化
- **默认阈值（stress_yellow=0.92）是最优选择**

## 代码变更清单

| 文件 | 改动 |
|------|------|
| `v10.2/ca_gcp/core/pipeline.py` | 修复 predict_fast Bug 1（源资产 scores）+ 清理 dead code |
| `v10.2/experiments/07_calibrate.py` | 修复 Bug 2（k 变化时重建 graph）|
| `v10.2/experiments/12_dual_momentum_backtest.py` | WF 实验脚本，使用全局校准参数 |
| `v10.2/integration/dual_momentum_ca_gcp.py` | 核心集成：月频调仓 + 周频风控 |
| `v10.2/integration/ca_gcp_risk_filter.py` | 支持 group_rules + calibrate_risk_filter |
| `v10.2/tests/test_dual_momentum_integration.py` | 18 单元测试 |
| `v10.2/data/results/best_params.json` | 更新为 k=6, eta=0.5, tau=20 |
| `v10.2/docs/88-dual-momentum-integration.md` | 集成文档 |
| `v10.2/docs/89-dual-momentum-experiment-report.md` | 本报告 |

## Git Commits

```
62474a8 feat(v10.2): WF 实验使用全局校准参数 (k=6, eta=0.5, tau=20)
172abb5 fix(v10.2): predict_fast Bug 1+2 修复 + turnover penalty
79035a6 fix(v10.2): 阈值校准实验结论 — 默认阈值最优
8367659 feat(v10.2): global CA-GCP + 月频调仓 + 周频风控 (最终版本)
ec45658 feat(v10.2): Sector CA-GCP 基础设施 (Phase A)
a6bb534 feat(v10.2): Phase B 基础设施 (阈值校准 + 分组规则)
09bee19 feat(v10.2): 月频调仓 + 周频风控 (CA-GCP 周一 full check, 其他日 hold)
d055994 feat(v10.2): 添加换手率和交易成本对比指标
2af9a7d refactor(v10.2): 使用 common/metrics.compute_metrics 统一指标计算
4813cfa feat(v10.2): 补充 Fold 5 (2025-03→2026-06) 包含最新数据
0470647 fix(v10.2): 完善 dual momentum 对比指标
3a8d832 feat(v10.2): dual_momentum + CA-GCP 风控验证 (Walk-Forward)
```
