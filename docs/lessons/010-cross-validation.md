# 教训 010: 回测结果的交叉验证

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

v7.3 的 Sharpe 在修复前是 1.820，修复后是 0.827。差异来自 `freq="W"` 的年化错误（2.2x）。但 Calmar/MaxDD/AnnRet 在修复前后一致，说明 NAV 计算本身是正确的。

## 教训

1. **单一指标不可靠**：Sharpe 高但 Calmar 低 → 可能是 Vol 被低估
2. **多指标交叉验证**：AnnRet / Vol / Sharpe / MaxDD / Calmar 应该自洽
3. **频率一致性**：所有指标使用相同的 `freq` 参数

## 验证清单

```
□ Sharpe ≈ AnnRet / Vol
□ Calmar ≈ AnnRet / |MaxDD|
□ Sortino ≈ AnnRet / DownsideVol
□ MaxDD 发生日期与 NAV 最低点一致
□ WinRate + LossRate ≈ 1
□ 逐年收益之和 ≈ 总收益（近似）
```

## 防范流程

1. **输出完整指标表**：不只是 Sharpe
2. **检查自洽性**：用公式反算验证
3. **对比已知基准**：与同类策略的结果对比
