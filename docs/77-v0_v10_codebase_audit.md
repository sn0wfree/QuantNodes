# v0-v10 代码全面审计报告

> **日期**: 2026-07-27
> **范围**: v0.0 ~ v10 所有策略代码、回测脚本、指标计算
> **方法**: 逐版本代码审查 + 复现验证 + 统一指标重算

---

## 一、发现的问题

### CRITICAL（3 个）

#### BUG-1: v10 `load_navs()` resample('D') 零收益稀释

**文件**: `strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py:48`
**问题**: `nav.resample('D').ffill()` 把交易日数据扩展到日历日，插入 28.5% 零收益日
**影响**: Sharpe 虚增 ~18-23%，Static 方案从 1.216 虚增至 1.930
**修复**: 删除 resample('D')，改为 `pd.DataFrame(navs).dropna()`
**状态**: ✅ 已修复

#### BUG-2: v4 `_performance_metrics()` 硬编码 252

**文件**: `QuantNodes/strategy/momentum_etf_rotation/v4/multi_strategy_v4.py:164-184`
**问题**: 硬编码 `252/n` 和 `sqrt(252)`，对周频数据年化错误
**影响**: V4Result.metrics 在 43ETF 周频面板上 Sharpe 虚增 ~2.2x
**修复**: 自动检测频率（中位数间隔 >4 天 → 周频 52，否则日频 252）
**状态**: ✅ 已修复

#### BUG-3: v4 `v4_full_backtest.py` freq="W" 用在日频数据

**文件**: `scripts/v4/v4_full_backtest.py:117`
**问题**: 日频 smartbeta 数据传 `freq="W"`，N_Years 算成 39.6（实际 8.2）
**影响**: 公开发布的 Sharpe 被压缩 ~2.2x，年化收益严重低估
**修复**: `freq="W"` → `freq="D"`
**状态**: ✅ 已修复

### HIGH（2 个）

#### BUG-4: v9 `v9_factor_galaxy.py` 日频数据默认 freq='W'

**文件**: `scripts/v9/v9_factor_galaxy.py:139`
**问题**: `compute_factor_metrics(w, returns_daily)` 不传 freq，默认 freq='W'
**影响**: Sharpe 虚增 ~2.2x
**修复**: 添加 `freq='D'` 参数
**状态**: ✅ 已修复

#### BUG-5: v0 lookback 错配

**文件**: `combo/unified_v1v5_compare.py:154,171,189`
**问题**: combo 对比 v0.0/v0.1/v0.2 用 lookback=144，版本定义用 lookback=90
**影响**: v0.0/v0.1/v0.2 的 NAV 不是它们的规范配置
**修复**: lookback=144 → lookback=90
**状态**: ✅ 已修复

### MODERATE（2 个）

#### BUG-6: v7.10 `calculate_daily_nav` 格式不匹配

**文件**: `v7/macro_substrategy_v7_6.py:442`
**问题**: `calculate_daily_nav()` 期望长格式，`construct_portfolio()` 返回宽格式
**影响**: `v7_10_gen_nav.py` 和 `v7_10_step_test.py` 运行时崩溃
**缓解**: HTML 从 `v7_10_nav_v56.parquet` 加载，由正确转换的脚本生成
**状态**: ⚠️ 待修复

#### BUG-7: v4 factor 策略 warmup 期权重归零

**文件**: `combo/unified_v1v5_compare.py` v4_factor_patched()
**问题**: 因子 IC warmup 期全为 0 → 权重归零 → NAV 平坦
**影响**: v4 factor Sharpe = -3.322，NAV 近乎平坦
**状态**: ⚠️ 设计问题，需重构

### LOW（2 个）

#### BUG-8: v9 `compute_factor_metrics()` 死代码

**文件**: `v9/factor_galaxy.py:294-305`
**问题**: return 后有不可达代码，其中硬编码 sqrt(252)
**影响**: 无运行时影响
**状态**: 待清理

#### BUG-9: v10 `rrg_rotation.py` 缺少 rebalance guard

**文件**: `v10/rrg_rotation.py:205-207`
**问题**: 换手成本每日计算（非调仓日为 0）
**影响**: 无数值影响，代码质量问题
**状态**: 待清理

---

## 二、复现性验证结果

### 统一方法

- 年化因子: sqrt(252)
- 年化收益: `(1 + total_return) ^ (1 / calendar_years) - 1`
- 日历年: `(index[-1] - index[0]).days / 365.25`

### 全期指标 (2018-2026)

| 策略 | 年化收益 | 波动率 | Sharpe | 最大回撤 | Calmar |
|------|---------|--------|--------|---------|--------|
| v7.10 TV-PR | +17.89% | 19.82% | 0.903 | -20.54% | 0.871 |
| v5.1 量价 (逆波动) | +15.48% | 16.33% | 0.948 | -18.15% | 0.853 |
| v10:DualMom | +15.07% | 21.20% | 0.711 | -44.70% | 0.337 |
| v5 量价 | +14.45% | 17.22% | 0.839 | -19.41% | 0.745 |
| v0.0 baseline | +10.09% | 8.25% | 1.223 | -10.43% | 0.967 |
| v10:Vol-parity | +9.90% | 8.14% | 1.216 | -9.57% | 1.035 |
| v10:DynD | +9.05% | 6.70% | 1.350 | -8.68% | 1.043 |
| v1.0 locked | +4.96% | 4.42% | 1.122 | -5.81% | 0.853 |

### OOS 指标 (2022-2026)

| 策略 | 年化收益 | Sharpe | Calmar | 最大回撤 |
|------|---------|--------|--------|---------|
| v7.10 TV-PR | +20.87% | 1.017 | 1.353 | -15.42% |
| v10:DualMom | +17.77% | 0.794 | 0.553 | -32.13% |
| v0.0 baseline | +10.67% | 1.366 | 1.523 | -7.01% |
| v10:Vol-parity | +10.27% | 1.434 | 1.426 | -7.20% |
| v10:DynD | +8.61% | 1.464 | 1.806 | -4.77% |
| v1.0 locked | +3.47% | 1.459 | 1.791 | -1.94% |

---

## 三、修复前后对比

| 策略 | 修复前 Sharpe | 修复后 Sharpe | 变化 | 原因 |
|------|-------------|-------------|------|------|
| v10 Vol-parity | 1.930 | 1.216 | -0.714 | resample('D') 零收益稀释 |
| v10 DynD | 1.849 | 1.350 | -0.499 | 同上 |
| v0.0 baseline | 0.827 | 1.223 | +0.396 | lookback 144→90 |
| v0.1 +VT | 0.880 | 1.055 | +0.175 | lookback 144→90 |
| v0.2 +TF | 0.791 | 1.090 | +0.299 | lookback 144→90 |

---

## 四、代码质量观察

### 各版本指标计算方式不统一

| 版本 | 年化收益 | 波动率年化 | 数据频率检测 |
|------|---------|-----------|------------|
| v0-v3 (combo) | 日历日/365.25 | sqrt(252) | 无（假设日频） |
| v4 `_performance_metrics` | 硬编码 252/n | sqrt(252) | 无（已修复） |
| v5/v5.1 | 日历日/365.25 | sqrt(252) | 无 |
| v7.10 | len/252 或 52 | sqrt(252) 或 sqrt(52) | 手动判断 |
| v9 `compute_metrics` | len/freq | sqrt(freq) | freq 参数 |
| v10 `metrics()` | 日历日/365.25 | sqrt(252) | 无 |
| `nav_curves_html.py` | 252/(len-1) | sqrt(252) | 无 |

**结论**: 至少 6 种不同的指标计算方式散落在代码库中，是 bug 的根源。

### 数据频率使用

| 版本 | 数据频率 | resample 调用 |
|------|---------|-------------|
| v0-v3 | 日频 | 无 |
| v4 | 日频/周频（两种脚本） | 无 |
| v5/v5.1 | 日频 OHLCV | `resample("M")` 月度 |
| v7.10 | 日频→周频→日频 | `resample("W")` 周频 |
| v9 | 周频 | `resample("D")` 权重展开 |
| v10 | 日频 | `resample('D')` ← 已删除 |

---

## 五、重构建议（详见 78-refactoring-plan.md）

核心问题：指标计算、回测循环、数据加载在每个版本中独立实现，导致：
1. 同一个 bug 在不同版本重复出现
2. 修复一处需要手动同步其他版本
3. 指标口径不统一，对比困难

**建议**: 提取公共模块，所有版本共用。

---

*文档版本: 1.0*
*日期: 2026-07-27*
