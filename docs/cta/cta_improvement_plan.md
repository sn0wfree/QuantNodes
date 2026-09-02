# CTA框架改进计划 — Phase 1-2

> 日期: 2026-09-02
> 范围: 修复已知bug + 性能优化（Phase 3-4后续再做）

---

## Baseline基准（修改前，2026-09-02保存）

**配置**: `tvpr × inv_vol`, target_vol=0.30, max_leverage=3.0, 无交易成本

| 指标 | 值 |
|------|-----|
| Sharpe | **1.878** |
| AnnRet | 12.5% |
| Vol | 6.7% |
| MaxDD | -12.5% |
| Calmar | 1.000 |
| 品种数 | 66 |
| 区间 | 2018-01-03 ~ 2025-12-31 |

**年度收益**: 2019 +30.9% / 2020 +12.4% / 2021 +26.6% / 2022 +13.2% / 2023 +11.1% / 2024 +3.3% / 2025 +5.9%

结果文件: `reports/cta/baseline_{nav,summary,yearly}_pre.*`

---

## Phase 1: 修复已知bug

### 1.1 module_weights bug（严重）
- **问题**: `ComboConfig.module_weights` 5个权重对应6个模块，`zip`静默丢弃IndustrialChainModuleStrategy
- **修复**: 6个权重 `(1/6,)×6`；同时保持默认行为向后兼容说明
- **验证**: 修复后IndustrialChain模块信号参与加权

### 1.2 交易成本默认为0
- **问题**: `slippage_bps=None`，所有回测零成本，Sharpe虚高
- **修复**: baseline保持无成本（可比性）；新增 `--slippage` 参数跑成本敏感性
- **验证**: 报告成本拖累估算

### 1.3 回测报告不保存
- **问题**: 所有结果只打印终端，`reports/cta/`不存在
- **修复**: `scripts/cta_baseline.py`（已建）保存NAV/指标/年度收益；诊断脚本结果落盘
- **验证**: `reports/cta/` 文件存在且可复现

### 1.4 集群定义不完整
- **问题**: `CLUSTERS` 只覆盖10个品种（5组），DEFAULT_PRODUCTS有66个，集群限制对多数品种失效
- **修复**: 补全全部66个品种的板块分组
- **验证**: 每个品种有归属集群

### 1.5 Basis数据硬截断
- **问题**: `end=min(self._end, "2025-07-31")` 期限结构因子最后5个月无数据
- **修复**: 移除硬截断，跟随策略end
- **验证**: basis因子最新日期=数据最新日期

---

## Phase 2: 性能优化

### 2.1 XS因子向量化
- **问题**: `for date in ret.index` per-row循环，O(T×N)且慢
- **修复**: `df.rank(axis=1, pct=True)` 向量化
- **验证**: 输出与原实现一致（随机抽样对比）

### 2.2 仓位EMA平滑向量化
- **问题**: per-row loop（factor_modules.py:589-595）
- **修复**: 保持循环（EMA天然递推）但用numpy加速；或等价lfilter
- **验证**: 输出一致

### 2.3 pct_change FutureWarning修复
- **问题**: pandas新版本`fill_method`弃用警告（momentum_xs.py / short_term_reversal.py等）
- **修复**: 显式`fill_method=None`
- **验证**: 无FutureWarning

---

## 验收标准

1. 修复后回测（无成本）vs baseline: 指标差异可解释（module_weights修复应提升IndustrialChain贡献）
2. 成本敏感性: 1bp/5bp/10bp滑点下Sharpe衰减量化
3. 全部ruff check通过
4. 报告文件落盘 `reports/cta/`
