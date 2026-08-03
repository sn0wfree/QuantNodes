# CA-GCP v6 Utility-Aligned 实验报告

## 1. Stage A — Downside Probability

- F1: 0.542
- AUC: 0.489
- PR-AUC: 0.3720774737606156
- ECE: 0.188
- Brier: 0.275

## 2. Stage B — Action Calibration

- Risk budget mean/max/min: 0.051 / 0.140 / 0.000
- α: 0.15, β: 0.05, max: 0.2

## 3. 评估指标 (含新指标)

| 指标 | v4 baseline | v6 2-stage |
|---|---|---|
| F1 | 0.546 | 0.548 |
| AUC | 0.489 | 0.489 |
| PR-AUC | - | 0.372 |
| ECE | - | 0.188 |
| Brier | - | 0.275 |
| net_utility | - | -0.014 |

## 4. is_downside 事件 Recall

- 总事件: 13
- 捕获: 3 / 13 = 23.1%
