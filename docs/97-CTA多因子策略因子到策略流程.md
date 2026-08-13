# 97. CTA 多因子策略：因子到策略的完整流程

> 状态: 定稿 (2026-08-13)
> 关联: docs/94-多因子CTA策略设计.md (设计迭代), docs/95-CTA多因子策略规格.md (规格),
>        docs/96-CTA多因子策略现状总结.md (现状快照)
> 代码: `QuantNodes/strategy/cta/strategies/factor_modules.py`

## 一、本文目的

本文逐层梳理「因子 → 策略」的完整数据流：从原始行情数据出发，经过
因子信号合成、模块聚合、组合加权、方向化、仓位计算，最终产出回测损益。
本文基于当前代码实现（含最新连续信号模式），与 docs/95 的规格互为补充。

## 二、整体架构：三层嵌套

策略 `FactorModuleComboStrategy` 继承 `_ModuleStrategy`，内部聚合 7 个模块子策略
（每个模块又是一个 `_ModuleStrategy`）。因此是**两层信号合成**：

```
数据层 load_panel
  → 信号层 _signals_raw (模块内聚合 → 模块间加权)
  → 方向化 _signals
  → 仓位层 _signal_to_lots / _run_with_lots
```

```
┌───────────────────────────────────────────────┐
│  FactorModuleComboStrategy                     │
│  ├─ TrendMomentumModuleStrategy   (趋势动量 22) │
│  ├─ MeanReversionModuleStrategy   (均值回归 15) │
│  ├─ VolumeFlowModuleStrategy      (量能资金流 24)│
│  ├─ EventCalendarModuleStrategy   (事件日历 6)  │
│  ├─ TermBasisModuleStrategy       (期限基差 2)  │
│  ├─ MacroModuleStrategy           (宏观联动 3)  │
│  └─ IndustrialChainModuleStrategy (产业链 5)    │
└───────────────────────────────────────────────┘
```

## 三、第 1 步：数据层（`load_panel`）

每个模块通过 `CtaDataLoader` 加载主力连续合约，产出：

| 数据 | 字段 | 来源 |
|------|------|------|
| `_price_panel` | 收盘价 (T×N) | `load_continuous_main` |
| `_ohlc_map` | 每品种 OHLC | 同上 |
| `_vols` / `_ois` | 成交量 / 持仓量 | 同上 |
| `_spread_df` | 期限结构 (近远月价差) | `load_term_structure` (懒加载) |
| `_basis_df` | 现货基差 | `load_spot_basis` (懒加载) |
| `_vix` / `_dxy` | 宏观序列 | `load_macro_series` (懒加载) |

关键点：
- 新上市品种早期无数据 → NaN 列，回测时自动跳过（`reindex(columns=self._products)`）
- 外部数据（期限/基差/宏观）通过 `_ensure_external` 懒加载，仅用到时才取

## 四、第 2 步：因子 → 模块信号（`_signals_raw`）

每个模块遍历自己的 `FACTORS` 列表 `[(因子名, 因子类, 输入类型)]`，对每个因子
二选一生成信号面板，全部存入 `panels[name]`。

### 4.1 两种信号模式

**路径 A — 连续模式**（`continuous=True` 且因子有 `raw_value` 且 `raw != compute`）：
1. `_raw_value_panel` 取因子**连续值**（如 MACD 柱值、CCI、stochastic）
2. `_cont_zscore`（:400）逐品种滚动 z-score（win 252, min 20）→ clip ±3
   → **除以 3 归一化到 [-1,1]**

**路径 B — 离散模式**（其余因子）：
`_signal_panel` 取因子 `compute()` 的 **-1/0/+1** 方向信号。

> 判据：`use_raw = (raw_value.__func__ is not compute.__func__)`。
> 仅当 `raw_value` 是真连续值（与 `compute` 不同实现）才连续化；
> 事件型因子（如 Chandelier/Squeeze，`raw == compute`）保持离散，避免 z-score 损坏。

**当前连续覆盖（11 个核心因子）**：

| 模块 | 连续因子 |
|------|----------|
| 趋势动量 (6) | aroon, linear_slope, trix, macd, vortex, chandelier |
| 均值回归 (5) | half_life, stochastic, cci, williams_r, squeeze |
| 其余模块 | 全离散 |

### 4.2 组内聚合（`aggregation`）

所有因子面板合成**单个模块信号**：

| 类型 | 机制 | 说明 |
|------|------|------|
| `equal` | 等权平均 `sum/len` | 简单，但噪声因子占比高 |
| `inv_vol` | 滚动信号波动率倒数加权 | `w_k ∝ 1/σ_k` |
| `regression` | 滚动 OLS 学权重 | 信号 → 未来收益的系数 |
| `tvpr` ★ | expanding TV-PR 学时变权重 | 推荐，见下 |

**TV-PR 聚合**（`_agg_tvpr`，:409）：
- X (T,N,K) = 组内因子信号；Y (T,N) = 品种未来 1 日收益
- β_t = ADMM 解的 expanding TV-PR：`min Σ(r-Xβ)² + λ_tv Σ|Δβ| + λ_l1 Σ|β|`
- 模块信号 = `Σ_k β_t,k × X_t,i,k`；β 正负自由（负 β = 因子反用）
- 关键：非预测性因子 β → 0，信号由少数好因子主导 → 这是比 equal 强很多的根因

### 4.3 为什么 equal 与 tvpr 差距巨大

Sharpe 对比（离散基线）：`equal` 0.110 vs `tvpr` 1.462。
- **equal**：所有因子等权 1/K，若 20 因子中 15 个噪声，噪声占 75% 权重，sign 方向随机化
- **tvpr**：学习时变权重，坏因子 β→0，sign 由预测因子主导

## 五、第 3 步：模块信号 → 组合信号

`FactorModuleComboStrategy._signals_raw`（:884）让 7 个模块各自 `_signals_raw()`，
再按 `module_weighting` 组间加权：

| 类型 | 机制 |
|------|------|
| `fixed` | 预置 `module_weights`（默认等权 1/7） |
| `inv_vol` ★ | 各模块信号波动率倒数加权 |

## 六、第 4 步：方向化 + 过滤（`_signals`）

- `np.sign(raw)` → 组合方向 -1/0/+1
- 可选 `momentum_filter`：组合信号 N 日均 < 0 → 降仓（`momentum_filter_coef`）
  - `momentum_filter_window=0` 关闭；`=60, coef=0.0` 时动量为负全空仓

## 七、第 5 步：仓位与回测

1. `_exec_sig`（:515）：执行滞后，T 收盘算信号 → T+1 成交（生产口径）
2. `vol_target_positions`：方向 × 波动率目标 → 组合权重（`target_vol`, `vol_lookback`, `max_leverage`）
3. `_weights_to_lots`（:859）：权重 × 资金/手数名义 → 目标手数
   - 当日活跃品种**等分资金**，按合约乘数折算手数后取整
4. `_run_with_lots`（:529）：`_PanelStrategy.target_positions` **只在方向变化时调仓**
   → `run_cta_backtest` 跑损益（含 commission / margin_call）

可选杠杆硬限（`leverage_hard_cap`）：两遍法迭代，逐日 gross/equity ≤ cap 等比缩放权重。

## 八、完整链路图

```
load_panel
  │ 收盘/OHLC/量/持仓/期限/基差/宏观
  ▼
每模块 _signals_raw (7 模块并行)
  ├─ raw_value 因子 → z-score → [-1,1]  (连续)
  ├─ compute 因子 → -1/0/1             (离散)
  └─ tvpr 组内聚合 → 模块信号
  ▼
组合 _signals_raw: 模块信号 × inv_vol 组间加权
  ▼
_signals: sign() → 方向 -1/0/1  (+ 动量过滤)
  ▼
_exec_sig: T+1 执行滞后
  ▼
vol_target_positions → 权重 → 手数
  ▼
_run_with_lots: 方向变化才调仓 → run_cta_backtest
```

## 九、当前默认配置与业绩

```python
FactorModuleComboStrategy(config=ComboConfig(
    aggregation="tvpr",            # 组内: 因子级 expanding TV-PR
    module_weighting="inv_vol",    # 组间: 模块滚动波动率倒数
    continuous=True,               # 连续信号模式 (默认开启)
    cont_z_window=252, cont_z_clip=3.0,  # z-score 参数
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
))
```

| 信号模式 | 组内聚合 | Sharpe | 年化 | MaxDD | OOS(2022-25) |
|----------|----------|--------|------|-------|--------------|
| 离散 | tvpr | 1.462 | 0.168 | -0.171 | 1.015 |
| 连续 | tvpr | **2.006** | **0.197** | **-0.111** | **1.507** |

关键发现：连续信号 z-score 归一化到 [-1,1] 与离散同尺度，TV-PR 学到的权重更稳定，
全维度领先（Sharpe +37%、MaxDD 改善 35%、OOS +48%）。

## 十、版本体系：fut_v1 / fut_v2

> 状态: 规划 (2026-08-13)
> 代码: `factor_modules.py` (fut_v1) / `factor_modules_v2.py` (fut_v2)

### 10.1 版本划分

| 版本 | 类 | 方向化 | 说明 |
|------|-----|--------|------|
| fut_v1 | `FactorModuleComboStrategy` | `np.sign(raw)` 丢量级 | 现行版本, Sharpe 2.006 |
| fut_v2 | `FactorModuleComboStrategyV2` (新子类) | 保留量级 × conviction 缩放 | 升级候选, 见 10.2 |

fut_v1 为基线, 行为冻结; fut_v2 为超集 (继承全部信号/聚合/回测逻辑, 仅覆盖
方向化与调仓触发器)。

### 10.2 为什么保留量级 (IC 证据)

| 指标 | 值 |
|------|-----|
| sign IC (方向) | 0.054 |
| raw IC (连续) | 0.076 (+0.022) |
| rank IC (Spearman) | 0.069 |

**方向内强度单调性** (强信号 vs 弱信号未来收益):
LONG 强/弱 = 4.3×, SHORT 强/弱 = 7.6× (q=0.75 档)。
量级在**方向内**携带置信度信息, 但 |raw| 独立排序不单调 → 用法是
`sign(raw) × conviction(|raw|)`, 只缩仓不反用。

### 10.3 fut_v2 设计 (两个递进方案)

**conviction 计算**: 逐品种滚动 252 天对 |raw| 做 pct-rank → [0,1], 无前视,
免疫 β 尺度漂移 (raw std≈0.0007, 不可直接当仓位)。

| 方案 | 公式 | 效果 | 风险预算 |
|------|------|------|----------|
| v2.1 只缩仓 | `conv = rank_mag` | 弱信号缩仓, 强信号满仓 | 总杠杆↓ |
| v2.2 预算再分配 | `conv = rank_mag / mean(rank_mag)` | 弱→强转移 | 杠杆≈不变, 需 clip max_leverage |

**调仓触发器** (必改): fut_v1 仅 sign 变化调仓, 量级变化不触发。
fut_v2 改为 **符号变 OR \|Δlots\| > 20%** 才调仓 (基于 lots, 无需改签名)。

### 10.4 实验计划 (1 → 2 递进)

1. fut_v2.1 A/B vs fut_v1 基线: Sharpe / AnnRet / MaxDD / OOS(2022-25) / 换手率
2. v2.1 不劣化 → fut_v2.2 (预算再分配 + max_leverage clip)
3. 记录对比表, 达标则设为新默认, 否则保留 fut_v1