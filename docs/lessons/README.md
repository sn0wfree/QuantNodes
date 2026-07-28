# QuantNodes 教训库

本目录记录项目开发中踩过的坑和总结的教训，每个教训一个文件。

## 教训列表

| 编号 | 文件 | 标题 | 严重度 | 来源 |
|---|---|---|---|---|
| 001 | 001-data-exploration-mistake.md | 数据探索中的指标选择错误 | HIGH | v7.3.2 NaN 分析 |
| 002 | 002-resample-on-returns.md | 对收益数据做 resample.pct_change | CRITICAL | v7.3 数据管道 |
| 003 | 003-mixed-return-types.md | 混合 simple return 和 log return | HIGH | v7.3 数据管道 |
| 004 | 004-nav-calculation-wrong.md | NAV 用 (1+log_return).cumprod() | HIGH | v7.3 数据管道 |
| 005 | 005-sharpe-annualization-bug.md | compute_metrics freq 参数错误 | HIGH | v7.3 指标计算 |
| 006 | 006-1day-lookahead.md | 调仓日当天生效的前视偏差 | MODERATE | v7.3 回测引擎 |
| 007 | 007-lasso-sparsity.md | Lasso 在高维场景下的稀疏解 | MODERATE | v7.3.2 设计 |
| 008 | 008-data-pipeline-principle.md | 数据管道设计原则 | HIGH | v7.3 架构 |
| 009 | 009-cache-consistency.md | 缓存一致性 | MODERATE | v7.3 缓存管理 |
| 010 | 010-cross-validation.md | 回测结果的交叉验证 | HIGH | v7.3 验证 |

## 使用方法

- 开发新功能前：浏览相关教训
- 遇到 bug 时：检查是否已有类似教训
- 发现新问题时：新增教训文件（编号递增）
