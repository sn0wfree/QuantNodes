# CA-GCP v6 Overlay v10-DynD 报告

## 1. OOS 业绩对比 (2022-01-01 ~ 2026-06-30)

| 指标 | Baseline (v10-DynD) | Overlay (+ v6 risk_budget) | Delta |
|---|---|---|---|
| Sharpe | 1.414 | 1.288 | -0.126 |
| Calmar | 1.745 | 1.531 | -0.214 |
| AnnRet | 8.32% | 7.17% | -1.15% |
| MaxDD | -4.77% | -4.68% | +0.08% |

## 2. 决定

**不采纳**

- Sharpe 改善: -0.126
- Calmar 改善: -0.214
- MaxDD 改善:  +0.08%

## 3. 限制

- v6 AUC 接近 0.5 (随机), 因此 risk_budget 信号弱
- Phase 1.6 walk-forward 0% is_downside recall, 模型过于保守
- Overlay 改善可能来自纯降仓 (暴露降低) 而非 v6 信号