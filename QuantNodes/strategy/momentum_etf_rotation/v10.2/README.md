# v10.2 — v10 + CA-GCP 独立风控层

## 是什么

v10.2 = v10 主体策略 + Cross-Asset Graph Conformal Prediction (CA-GCP) 独立风控层。

CA-GCP 是 Parker & Zhang (2026) 提出的不确定性量化框架，**不替代 v10 的因子选股**，而是：
1. **预警**：监控市场状态（系统性压力 + 区间宽度突变）
2. **风控**：在预警触发时调整 v10 仓位

## 与 v10 的关系

```
v10 主体（不改）        v10.2 新增
─────────────         ──────────────
[因子打分]            [因子打分]
[TV-PR]               [TV-PR]
[Vol-parity]     +    [CA-GCP 风险层]    ← 新增
[持仓]                [调整后持仓]
```

## 实证结果（38 ETF, 2022-04 ~ 2023-04 测试期）

| 指标 | v10 mock | v10.2 (+ CA-GCP) |
|------|----------|------------------|
| 年化收益 | 1.67% | 9.30% |
| Sharpe | 0.08 | **0.49** |
| MaxDD | -20.7% | **-17.6%** |
| Calmar | 0.08 | **0.53** |

详见 [docs/82-ca-gcp-pool-size-test.md](docs/82-ca-gcp-pool-size-test.md)

## 快速开始

```python
from v10.2 import CAGCPipeline, CAGCPConfig, RiskFilterRules, ca_gcp_risk_filter

# 1. 训练 CA-GCP
pipe = CAGCPipeline(CAGCPConfig(k=8))
pipe.fit(returns_train)

# 2. 预测区间
intervals = pipe.predict(returns_calib, returns_test)

# 3. 作为风控 hook
adjusted_weights, diag = ca_gcp_risk_filter(v10_weights, intervals)
print(f"Alert level: {diag['alert_level']}, scale: {diag['applied_scale']}")
```

## 依赖

- numpy, pandas, scipy（基础）
- 无 networkx（避免 v7.13 教训）
- 无 PyTorch（保持轻量）

## 实验

8 个实验脚本位于 `experiments/`，按顺序运行：

1. `01_build_graph.py` — 构建 KNN 图
2. `02_run_baselines.py` — 跑 5 baselines + CA-GCP
3. `03_coverage_compare.py` — 覆盖率对比
4. `04_width_compare.py` — 区间宽度对比
5. `05_crisis_day_compare.py` — 危机日表现
6. `06_early_warning.py` — 预警信号评估
7. `07_calibrate.py` — 超参 grid（慢，仅 6 组合）
8. `08_v10_2_backtest.py` — Phase B mock 回测

## 单元测试

```bash
PYTHONPATH=. python -m pytest v10.2/tests/ -v
```

12 tests, all passing.

## 论文参考

Parker, E. & Zhang, R. (2026). *Graph-Based Uncertainty-Aware Financial Forecasting via Cross-Asset Conformal Prediction*. Computer Life, 14(3), 21-29.

## 历史

- 2026-07-28: v10.2.0 — Phase A (CA-GCP 独立验证) + Phase B (v10 mock 集成)
- 来源论文: arXiv 2026 / Computer Life 14(3)

## 局限

1. 默认超参下过保守（marginal cov 99.9% vs 95% 目标）
2. 宽度比 Vol-CP +42%（论文 +10%），因 ETF 池相关性弱
3. 测试期仅 252 天，未覆盖 2024 年 A 股雪球事件
4. Mock momentum 信号，真 v10 主体未接入