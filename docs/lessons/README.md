# QuantNodes 教训库

本目录记录项目开发中踩过的坑和总结的教训，分两种粒度：

- **单个教训**：`001-010.md` — 跨阶段的重大 bug 复盘（v7.3 数据管道）
- **逐日教训**：`daily/` — 2026-07-07 → 2026-07-28 共 17 天，每天一个文件

## 1. 单个教训（v7.3 数据管道）

| 编号 | 文件 | 标题 | 严重度 |
|---|---|---|---|
| 001 | [001-data-exploration-mistake.md](001-data-exploration-mistake.md) | 数据探索中的指标选择错误 | HIGH |
| 002 | [002-resample-on-returns.md](002-resample-on-returns.md) | 对收益数据做 resample.pct_change | CRITICAL |
| 003 | [003-mixed-return-types.md](003-mixed-return-types.md) | 混合 simple return 和 log return | HIGH |
| 004 | [004-nav-calculation-wrong.md](004-nav-calculation-wrong.md) | NAV 用 (1+log_return).cumprod() | HIGH |
| 005 | [005-sharpe-annualization-bug.md](005-sharpe-annualization-bug.md) | compute_metrics freq 参数错误 | HIGH |
| 006 | [006-1day-lookahead.md](006-1day-lookahead.md) | 调仓日当天生效的前视偏差 | MODERATE |
| 007 | [007-lasso-sparsity.md](007-lasso-sparsity.md) | Lasso 在高维场景下的稀疏解 | MODERATE |
| 008 | [008-data-pipeline-principle.md](008-data-pipeline-principle.md) | 数据管道设计原则 | HIGH |
| 009 | [009-cache-consistency.md](009-cache-consistency.md) | 缓存一致性 | MODERATE |
| 010 | [010-cross-validation.md](010-cross-validation.md) | 回测结果的交叉验证 | HIGH |

## 2. 逐日教训（daily/）

按日期顺序，每个文件包含当日所有 commit + 教训 + 防范清单。

| 日期 | commit | 文件 | 主题 |
|---|---|---|---|
| 2026-07-07 | 2 | [daily/2026-07-07-stage-10-13.md](daily/2026-07-07-stage-10-13.md) | Stage 10 caps + Stage 13 成本模型 |
| 2026-07-08 | 6 | [daily/2026-07-08-stage-11-rp.md](daily/2026-07-08-stage-11-rp.md) | Stage 11 RP + M4.5 shim |
| **2026-07-09** | **60** | [daily/2026-07-09-v1-v5-peak.md](daily/2026-07-09-v1-v5-peak.md) | **V1.0→V5.1 全链路（峰值日）** |
| 2026-07-10 | 14 | [daily/2026-07-10-v6-v7-0-html.md](daily/2026-07-10-v6-v7-0-html.md) | V6 + V7.0 5 Macro Dynamic |
| 2026-07-11 | 2 | [daily/2026-07-11-v7-3-skeleton.md](daily/2026-07-11-v7-3-skeleton.md) | V7.3 起点（1:1 复刻） |
| 2026-07-13 | 12 | [daily/2026-07-13-v7-3-lock-v7-5.md](daily/2026-07-13-v7-3-lock-v7-5.md) | V7.3 锁定 + V7.4/V7.5 + n_years 修复 |
| 2026-07-14 | 23 | [daily/2026-07-14-v7-6-tv-pr.md](daily/2026-07-14-v7-6-tv-pr.md) | V7.6 TV-PR + V7.5 Step 2/3 失败 + V6.2 DEPRECATED |
| 2026-07-15 | 21 | [daily/2026-07-15-v7-6-lookahead-6bug.md](daily/2026-07-15-v7-6-lookahead-6bug.md) | **V7.6 未来函数 6 Bug + Sensitivity 7 Phase** |
| 2026-07-16 | 11 | [daily/2026-07-16-v7-6-factor-ic.md](daily/2026-07-16-v7-6-factor-ic.md) | V7.6 因子 IC + 增强因子 |
| 2026-07-17 | 2 | [daily/2026-07-17-v7-9-symmetry-fail.md](daily/2026-07-17-v7-9-symmetry-fail.md) | **V7.9 Symmetry 正交化失败（Sharpe -91%）** |
| 2026-07-18 | 4 | [daily/2026-07-18-v7-7-tree-model-fail.md](daily/2026-07-18-v7-7-tree-model-fail.md) | V7.7 树模型失败 + V7.10 硬化 |
| 2026-07-19 | 2 | [daily/2026-07-19-v7-10-overfit-validate.md](daily/2026-07-19-v7-10-overfit-validate.md) | V7.10 4 步 OOS 步骤 1-2 |
| **2026-07-20** | **23** | [daily/2026-07-20-v7-10-oos-unified-engine.md](daily/2026-07-20-v7-10-oos-unified-engine.md) | **V7.10 步骤 3-4 + 统一引擎 + YAML（次高峰）** |
| 2026-07-21 | 2 | [daily/2026-07-21-strategy-research.md](daily/2026-07-21-strategy-research.md) | StrategyResearch 子模块 |
| 2026-07-24 | 18 | [daily/2026-07-24-v10-html-simplify.md](daily/2026-07-24-v10-html-simplify.md) | **V10 诞生 + HTML 24→9 精简** |
| 2026-07-27 | 13 | [daily/2026-07-27-v10-to-v11-migration.md](daily/2026-07-27-v10-to-v11-migration.md) | v10→v11 迁移 + 重复类清理 + Lint |
| 2026-07-28 | 6 | [daily/2026-07-28-v7-3-data-pipeline-audit.md](daily/2026-07-28-v7-3-data-pipeline-audit.md) | **v7 全审计 + v7.3 数据管道修复** |

**总计**：217 commits / 17 天

## 3. 关联文档

- [`../research_history/05_LESSONS_LIBRARY.md`](../research_history/05_LESSONS_LIBRARY.md) — 跨阶段聚合的 48 条主题教训（L-101~L-323）
- [`../research_history/00_TIMELINE.md`](../research_history/00_TIMELINE.md) — V0→V10 完整时间轴
- [`../research_history/04_V7_V10.md`](../research_history/04_V7_V10.md) — V7→V10 详细记录
- [`./daily/summary.md`](daily/summary.md) — 17 天总结（待生成）

## 4. 使用方法

- **遇到 bug**：先查 daily/ 同日 + 相邻日的教训
- **设计新策略**：查 05_LESSONS_LIBRARY 主题教训（L-1xx~3xx）
- **数据管道问题**：查 001-010 单教训文件
- **发现新问题**：新增教训文件（001- 编号递增 / daily/ 日期文件）