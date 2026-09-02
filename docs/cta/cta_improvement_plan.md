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

### 1.1 module_weights "bug" — 排查后无bug
- **实情**: 探索报告误报。`self._modules`实际5个模块（OscillatorAlpha不在combo内），5权重对应5模块，`zip`无丢弃
- **行动**: 无行为修改；仅加防御性长度校验（未来加模块时报错而非静默丢弃）

### 1.2 交易成本默认为0
- **问题**: `slippage_bps=None`，所有回测零成本，Sharpe虚高
- **修复**: `scripts/cta_baseline.py --sensitivity` 跑 0/1/5/10bps 敏感性
- **结果**: 策略对成本极敏感（高换手）

| 滑点 (bps) | Sharpe | AnnRet | MaxDD | Calmar |
|-----------|--------|--------|-------|--------|
| 0 | 1.688 | 12.5% | -10.3% | 1.214 |
| 1 | 1.254 | 10.2% | -11.7% | 0.871 |
| 5 | 0.617 | 6.0% | -15.4% | 0.386 |
| 10 | 0.347 | 3.7% | -18.7% | 0.197 |

**结论**: 每1bp滑点≈-0.04 Sharpe。零成本Sharpe严重虚高，实盘评估必须带成本。

### 1.3 回测报告不保存
- **问题**: 所有结果只打印终端，`reports/cta/`不存在
- **修复**: `scripts/cta_baseline.py`（已建）保存NAV/指标/年度收益；诊断脚本结果落盘
- **验证**: `reports/cta/` 文件存在且可复现

### 1.4 集群定义不完整
- **问题**: `CLUSTERS` 只覆盖10个品种（5组），DEFAULT_PRODUCTS有66个，集群限制对多数品种失效
- **修复**: 补全全部66个品种的板块分组
- **验证**: 每个品种有归属集群

### 1.5 Basis数据硬截断 — 无需修复（数据滞后非代码bug）
- **实情**: `spot_basis_daily`表数据本身只到2025-07-21，截断`min(end, "2025-07-31")`与数据现实一致
- **行动**: 代码不改；缺口在数据刷新（spot basis数据源需更新），非代码问题

---

## 修复后结果（2026-09-02）

### 归因（三层隔离回测）

| 配置 | Sharpe | AnnRet | Vol | MaxDD | Calmar |
|------|--------|--------|-----|-------|--------|
| baseline（修复前） | 1.878 | 12.5% | 6.7% | -12.5% | 1.000 |
| 仅fill_method+向量化 | 1.851 | 12.4% | 6.7% | -12.5% | 0.989 |
| 全部修复（含CLUSTERS统一） | **1.688** | 12.5% | 7.4% | **-10.3%** | **1.214** |

**归因结论**:
- fill_method=None修复: Sharpe -0.027（LU停牌104天幻影价格消除，影响极小）
- XS因子向量化: 输出与旧实现完全一致（合成数据验证4因子equals），仅速度提升24x
- 仓位EMA向量化: 与旧实现等价（最大差1.3e-15浮点噪声）
- **CLUSTERS统一: 主要行为变化**（Sharpe -0.163 但 MaxDD改善2.2pp, Calmar +21%）
  - 原定义仅覆盖10品种 → 现覆盖全部66品种（7板块）
  - cluster冲突过滤/tail risk降仓/敞口上限从"基本失效"变为实际生效
  - 板块敞口上限约束reshape风险结构：波动↑回撤↓

### 判定

- Sharpe 1.69 vs 1.88: 旧值部分来自幻影价格+集群约束失效的虚高
- 新版本语义更正确（无幻影价格、全覆盖集群约束），Calmar/MaxDD更优
- 零成本回测不可用于实盘评估（见§1.2敏感性）

---

## Phase 2: 性能优化

### 2.1 XS因子向量化
- **问题**: `for date in ret.index` per-row循环，O(T×N)且慢
- **修复**: 统一`_quantile_signals()`辅助函数（`quantile(axis=1)` + numpy布尔广播）
  - 注意: `DataFrame & Series`按列对齐的坑 → 用`.to_numpy()[:, None]`显式行广播
- **验证**: 4因子（MomentumXS/VolXS/BasisXS/AmihudXS）输出与旧实现`equals`一致
- **效果**: 24x加速（526ms→22ms @ 500×30面板）

### 2.2 仓位EMA平滑向量化
- **问题**: per-row loop（factor_modules.py `_run_with_lots`）
- **修复**: 前置零行 + `ewm(alpha, adjust=False)`（匹配循环初始条件prev=0）
- **验证**: 最大差1.3e-15（浮点噪声），行为等价

### 2.3 pct_change FutureWarning修复
- **问题**: pandas `fill_method='pad'`默认弃用警告，54处调用
- **修复**: 显式`fill_method=None`（面板无内部NaN时行为等价；LU停牌段消除幻影价格，语义更正确）
- **验证**: 全tree 0处未修复，回测可跑无警告

---

## 验收标准

1. 修复后回测（无成本）vs baseline: 指标差异可解释（module_weights修复应提升IndustrialChain贡献）
2. 成本敏感性: 1bp/5bp/10bp滑点下Sharpe衰减量化
3. 全部ruff check通过
4. 报告文件落盘 `reports/cta/`
