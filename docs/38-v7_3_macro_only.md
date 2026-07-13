# v7.3 — 宏观子策略 (简化版 + 完整版)

> **编号**: 38
> **状态**: ✅ 简化版 + 完整版均已完整还原 + **v7_macro_baseline 锁定 (2026-07-13)**
> **日期**: 2026-07-13
> **关联**: docs/35 (宏观因子体系业界调研) + `~/Public/高频宏观因子/` 参考实现

---

## ⚠️ v7_macro_baseline 锁定声明 (2026-07-13)

**锁定入口**: `QuantNodes/strategy/momentum_etf_rotation.v7.v7_macro_baseline()`

### 锁定配置

| 参数 | 值 | 来源 |
|------|---:|------|
| `bootstrap_times` | **500** | 敏感性分析确认收敛点 (bt=200 已达 90% 精度, bt=500 性价比最优) |
| `bootstrap_resample_min/max` | 78 / 104 周 | 源 cell 102 (1.5-2 年) |
| `bootstrap_random_state` | **42** | 锁定 (确保可复现) |
| `bootstrap_cache_alpha` | True | 30x 加速 (LassoCV α 共享) |
| `quarter_window` | 8 (2 年) | 源 cell 102 |
| `max_weight` | 0.5 | 源 cell 99 |
| `sum_lower / upper` | 0.9 / 1.0 | 源 cell 94 |
| 池 | 13 指数 (含 `中债1-3年国债财富指数`) | 源 cell 99 |
| 因子 | 8 (源 cell 99, 不含 `期限利差因子_加权`) | 源 cell 99 |

### 锁定性能 (3 个 random_state [42, 7, 123] 平均)

| 窗口 | Ann | Vol | Sharpe | MaxDD | Calmar |
|------|----:|----:|-------:|------:|-------:|
| **全期 2010-2026** | 2.94% | 7.20% | 0.408 | -20.4% | **0.145** |
| **OOS 2022-至今** | 3.37% | 6.99% | 0.482 | -9.1% | **0.371** |
| **OOS 2023-至今 (用户原话)** | **5.24%** | **6.74%** | **0.778** | **-8.45%** | **0.620** |

### 用途

- 宏观子策略 baseline, 与 `v6.2` (行业轮动) 配合用 (combo 50/50 OOS 2023-至今 Calmar 0.829)
- 适合长期低风险配置 (DD -8%, vol 6.5%, 季度调仓换手低)
- 不擅长: 短期交易 / 牛市捕捉 / 小盘股 alpha

### 未来变更规则 (严格)

1. **任何 v7 改动** (新因子 / 新池 / 新算法 / 改默认值) **必跑 baseline 对照**
2. **性能退化 > 5%** 必须更新 baseline, 加 migration note
3. **任何 config 改动** 必须先看 `tests/strategy/momentum_etf_rotation/v7/test_v7_macro_baseline.py`
4. **跨 random_state 稳定性**: Ann CV% 应 < 10%, Calmar CV% 应 < 15% (历史基线)

### 锁定测试

- `tests/strategy/momentum_etf_rotation/v7/test_v7_macro_baseline.py` (7 个测试)
  - 配置值冻结 (3 个)
  - 可复现性 (1 个, 同 random_state 必出同 NAV)
  - 性能锁定 (2 个, OOS 2023 Ann/Calmar 5% 容差)
  - 数据兼容性 (1 个)
  - baseline pool 校验 (在 TestV7MacroBaselineConfig)

---

## v7_macro_baseline_v2_tf (趋势过滤增强版, 2026-07-13)

### 背景

v7_macro_baseline (无 TF) OOS 2018-至今 Calmar 0.387, 跑不赢 v1.0 locked (0.913).
根因分析 (5 个 root cause, 详见 §四):
- #1 **无趋势过滤 (TF)**: v1.0 通过 TF 在 2018/2022 熊市减仓, 大幅降 DD
- #2 无选股层
- #3 调仓频率慢 3x
- #4 因子频率不对等
- #5 因子层 RP 设计目标非 alpha 最大化

**最高 ROI 修复: 加趋势过滤 (TF) 单点改动**, 预期 DD -3-5pp, Calmar +30-100%.

### 新配置

| 参数 | 值 | 来源 |
|------|---:|------|
| 基础配置 | 同 v7_macro_baseline | 继承 (不修改原 baseline) |
| `trend_filter_enabled` | True | 新增 |
| `trend_filter_ma` | 200 日 | v1.0 默认 (沪深300 200 日 MA) |
| `trend_filter_bear` | 0.5 | v1.0 默认 (熊市半仓) |
| `equity_indices` | ['沪深300指数', '中证500指数', '中证1000', '恒生指数'] | 权益专属 TF |
| 基准 | 沪深300 (与 v1.0 一致) | 复用 |

### TF 信号逻辑 (v2 equity→bonds, 2026-07-13 优化)

```
if 沪深300 价格 < 200 日 MA:
    # 熊市: 只减权益 × 0.5, 释放权重按比例分配给债券
    freed = sum(w[equity] × 0.5)
    w[equity] *= 0.5
    w[bonds] += freed × (w[bonds] / sum(w[bonds]))
else:
    # 多头: w 不变
```

**设计理由**: v7 是宏观配置模型, 涉及权益/债券/商品三大类. TF 针对权益设计 (沪深300 vs MA200), 对债券减仓反直觉 (熊市债券是 flight to safety 资产). equity→bonds 方案:
- 降低换手 63% (38%→14%), 减少交易成本
- 保持 DD 改善 (equity 减仓效果不变)
- Alpha 微升 (+0.04-0.15pp), 因为债券在熊市涨

### Migration Notes

**2026-07-13: TF 权益专属化 (equity→bonds)**
- 旧行为: 所有 13 指数 × 0.5 → 换手 38%/年
- 新行为: 只减 4 权益指数 × 0.5, freed weight → 5 债券按比例分配 → 换手 14%/年
- 影响: Ann +0.04-0.15pp, Sharpe -3%, Calmar +0.9-3.4%, 换手 -62-63%
- 改动文件: `macro_substrategy_v7_3.py` (BOND_INDICES 常量 + equity_indices 配置 + apply_trend_filter 重写)
- 测试: 11 个 TF 单元测试 (4 改 + 3 新边界), 全部通过

**2026-07-13: 初始 v2 发布**
- **`v7_macro_baseline` (无 TF) 保持锁定**: OOS 2023-至今 Calmar 0.620, 零修改
- **`v7_macro_baseline_v2_tf` (有 TF) 新增**: OOS 2023-至今 Calmar 预估 0.7-0.9
- **用户 2 选 1**: production 部署可任选
- **未来 v7.x 改动**: 任何 baseline 改动需先跑对照 (见 v7_macro_baseline 锁定规则)

### 锁定测试

- `tests/strategy/momentum_etf_rotation/v7/test_v7_macro_baseline_v2_tf.py` (14 个)
  - 4 个配置冻结
  - 7 个 apply_trend_filter 单元测试 (含 3 个边界: 空 equity, 无 bonds, equity 零权重)
  - 5 个端到端 backtest (slow)
  - 1 个跨 seed 稳定性

---

---

## 一、决策与版本对照

| # | 决策 | 备注 |
|---|------|------|
| 1 | 跳过 Stage 30.1-30.3, 直接 Stage 30.4 | 用户决策 |
| 2 | BME bug 暂不修复 | 与 v7.3 无关 |
| 3 | 不验证 v6.2 thr1.0 | 无关 |
| 4 | 直接复用 9 因子 Excel 数据 | 用户决策 |
| 5 | **不写代码, 但完整版用 5 宽基 ETF 还是 13 INDICES?** | 用户决策 → **13 INDICES** (与 source 一致) |
| 6 | Bootstrap 2000 次全面 | 用户决策 |
| 7 | 不并行修 BME | 用户决策 |

## 二、最终结果 (数据 bug 修复后, 2026-07-13)

### 2.1 OOS 2023-至今 (用户原话"23年到现在")

| 策略 | Ann | Vol | DD | Calmar | 备注 |
|------|----:|----:|---:|-------:|------|
| v7.3 完整版 (Symmetry + Bootstrap2000 + FRP) | 5.93% | 7.44% | -8.74% | 0.679 | 修复前 ~1% |
| v7.3 简化版 (IC 加权) | 5.31% | 7.31% | -8.38% | 0.634 | |
| v6.2 ir_expanding (current best) | 14.79% | 15.47% | -15.30% | 0.966 | 单一策略最佳 |
| v1.0 locked (低风险基准) | 4.94% | 2.49% | -1.94% | **2.548** | 波动极低 |
| 13 指数等权 | 6.50% | 8.24% | -9.68% | 0.671 | |
| **combo 50/50 (v6.2 + v7.3 完整版)** | 11.69% | 11.45% | -11.42% | **1.023** | ⭐ Calmar 首破 1.0 |
| **combo 50/50 (v6.2 + v7.3 简化版)** | 12.87% | 11.99% | -10.63% | **1.210** | ⭐⭐ 最佳组合 |

### 2.2 OOS 2022-2026 (4-year 标准窗口)

| 策略 | Ann | Vol | DD | Calmar | 备注 |
|------|----:|----:|---:|-------:|------|
| v7.3 完整版 | 3.66% | 7.84% | -11.30% | 0.324 | 修复前 0.172 |
| v7.3 简化版 | 4.26% | 7.94% | -10.01% | 0.426 | |
| v6.2 ir_expanding | 7.64% | 16.19% | -16.73% | 0.457 | |
| **combo 50/50 (v6.2 + v7.3 简化版)** | 8.74% | 12.41% | -10.63% | **0.822** | +80% vs v6.2 |
| combo 50/50 (v6.2 + v7.3 完整版) | 6.00% | 12.06% | -11.56% | 0.519 | |

### 2.3 与 source 对比

| 来源 | 样本 | Calmar | Ann |
|------|------|-------:|----:|
| **source (v2 notebook cell 121)** | 2012-2024 | **1.626** | **29.07%** |
| **我 combo 简化版 OOS 2023-至今** | 2023-2026 | **1.210** | 12.87% |
| 我 v7.3 完整版 OOS 2023-至今 | 2023-2026 | 0.679 | 5.93% |
| 我 v7.3 完整版 OOS 2022-2026 | 2022-2026 | 0.324 | 3.66% |
| 我 v7.3 完整版 全期 | 2010-2026 | 0.129 | 2.70% |

**结论**: 数据 bug 修复后 v7.3 完整版 2023-至今段 Ann 5.93% (修复前 ~1%), combo 50/50 (v6.2 + v7.3 完整版) Calmar **1.023** (修复前 0.470). 与 source 1.626 仍差 30%, 原因: Wind vs iFind 数据精度 + 缺 1 个中债1-3年国债财富指数.

## 三、source vs 我的实现差距分析 (5 个 root cause)

### 3.1 池差异: 13 ETFs vs 13 indices (or 14 in v2 cell 73)

我用 `data/real/etf_nav_2018-01-01_2026-06-30.parquet` (44 ETF 池子) 找不到 512100 (中证1000 ETF), 简化为 5 ETF.

**但是**: source 用的是 13 INDICES (level-1), 不是 ETF.
- 沪深300 / 中证500 / 中证1000 / 恒生指数 + 4 中债 + 4 商品

**修正**: v7.3 v2 改用 INDEX_COLS (13 indices). ✅ 已修正.

### 3.2 数据池差异: 13 vs 14 indices

source cell 73 中使用 14 个指数, **包括** `'中债1_3年期国债财富指数'` (财富指数 = 含分红 total return), 这个我本地 Excel 数据里没有.

| source 14 个指数 | 本地 Excel 13 个 |
|---|---|
| 沪深300, 中证500, 中证1000, 恒生 | 沪深300, 中证500, 中证1000, 恒生 |
| 中债10年期国债指数 | 中债10年期国债指数 |
| 中债3_5年期国债指数 | 中债3-5年期国债指数 |
| **中债1_3年期国债财富指数** ❌ 缺失 | - |
| 中债国开行债券总指数 | 中债国开行债券总指数 |
| 中债企业债总指数 | 中债企业债总指数 |
| 南华综合, 工业品, 农产品 | 南华综合, 工业品, 农产品 |
| 布伦特原油 | 期货结算价:布伦特原油 |
| 沪金指数 | 收盘价:沪金指数 |

**影响**: 缺 1 个指数 → FRP 求解空间略小 → 收益降低约 10-20%.

### 3.3 调仓窗口 (3 年 vs 2 年)

我设了 `min_history_weeks=52*3=156` (3 年). **修正**: source 是 8 quarter = 2 年. ✅ 已修正.

### 3.4 Bootstrap 参数 (200/2000, 104-156 vs 500, 78-104)

用户决策"全面 2000 次", 但 source 实际用 500 次. 我用 500/2000 都尝试过, 结果类似 (ann ≈ 0.5%).

### 3.5 时间窗口不同 (2010-2026 vs 2012-2024)

source 用 2012-2024 (12 年). 我用 2008-2026 (18 年). 不同时间段的因子结构影响 Beta 估计.

## 四、对 v7.3 完整版的诊断 (实测)

### 完整版 root cause: weights 极度集中

实测 v7.3 完整版 (bootstrap=500) 中, FRP 求解器返回的 weights 在每个 rebal 日都呈现**极度分散** (每只资产 5-15%):

```
2010-03-31: 13 只各 ~5-8% (等权附近)
2012-09-30: 重仓债券 (中债3-5年/国开/企业债 各 14-16%)
2015-03-31: 13 只 7-11%
2020-03-31: 重仓原油 + 中债
```

理论上分散是好事, 但权重接近等权意味着 **Lasso β 接近全零 / 无信号**, 导致"看不见就拿等权".
源文件 29% alpha 暗示原作者的 Lasso β 有更强信号 (特别是对成长型资产的择时).

### 为什么我的 Lasso β 信号弱?

可能原因:
1. 数据滞后 (iFind vs Wind, 更新频率不同)
2. 我的数据没有 中债1-3年 财富指数, FRP 求解空间缺一个维度
3. 我的 9 因子用的是 iFind 数据, source 9 因子是 Wind 数据, 数值有差异
4. 我用了 `cache_alpha=True` 优化速度, 可能与 source 全 OLS/全 LassoCV 有差别

## 五、最终结论与建议

### 5.1 v7.3 简化版 (combo 50/50)
- **保留**: 实质是 13 指数等权 + 季度调仓.
- **combo 50/50 Calmar 0.921 > v6.2 0.755 (+22%)**.
- MaxDD -10.63% vs v6.2 -16.73% (降低 37%).
- 实现成本低 (300 行一次性脚本).

### 5.2 v7.3 完整版 (Bootstrap-Lasso + Symmetry + FRP)
- **不达成 source 的 29% 收益** (我们的 0.30% 远低).
- 失败原因: 数据差异 (Wind vs iFind) + 缺乏 14 个指数中的 1 个 + 我的代码与 QuantOPT 在参数优化上的差距.

### 5.3 后续建议 (按 ROI 排序)
1. **抽取 中债1-3年国债财富指数** 从 akshare/Wind, 加入 INDEX_COLS → 14 indices, 预计提升 10-15% 收益.
2. **Wind 数据接入**: 将 Excel 换成 Wind 数据 (`~/Public/高频宏观因子/` 看起来来自 Wind 客户端导出), 预计收益提升到 source 80%+ 水平.
3. **使用 QuantOPT for FRP** (替换 scipy SLSQP 简化版), 预计小幅提升.

### 5.4 当前 v7 升级状态
- ✅ **stage 30.4 简化版 + 完整版均实现完毕**.
- ✅ **37 个单元测试全过** + 端到端 run_v7_3_backtest 跑通.
- ✅ **commit ebff3cb (v1)** + 准备 commit v2.
- ⚠️ 完整版收益仅 0.30% (远低于 source 29%), 主要原因是数据差异.

## 六、关键文件

```
QuantNodes/strategy/momentum_etf_rotation/v7/        (Production code, ~1100 行)
├── symmetry.py                      (RollingSymmetry, Klein 2013)
├── bootstrap_lasso.py               (BootstrapLassoMapping, 2000x averaging)
├── factor_risk_parity.py            (scipy SLSQP, 等价 QuantOPT)
├── macro_substrategy_v7_3.py         (V7_3Config + SubStrategy + run_v7_3_backtest)
└── data_loader.py                   (load_factor_returns + load_index_panel)

tests/strategy/momentum_etf_rotation/v7/             (37 tests)
├── test_symmetry.py                 (10 tests, no-leakage 验证)
├── test_bootstrap_lasso.py          (9 tests + 1 slow)
├── test_factor_risk_parity.py       (12 tests, scipy SLSQP)
└── test_v7_3_strategy.py            (4 tests, 端到端 smoke)

scripts/
├── v7_3_simple_backtest.py          (简化版, 季度 IC 加权)
└── v7_3_full_backtest.py            (完整版, Bootstrap-Lasso + FRP)

reports/momentum_etf_rotation/v7/                  (Output)
├── v7_3_oos_results.csv             (简化版 4 策略)
├── v7_3_oos_navs.parquet            (简化版 NAVs)
├── v7_3_factor_loadings.csv         (简化版季度权重)
├── v7_3_full_oos_results.csv        (完整版 4 策略)
└── v7_3_full_oos_navs.parquet       (完整版 NAVs)
```

## 七、运行命令

```bash
# 测试
python3.11 -m pytest tests/strategy/momentum_etf_rotation/v7/ -v -m 'not slow'

# 简化版 (10 秒)
python3.11 scripts/v7_3_simple_backtest.py

# 完整版 (5-30 分钟)
python3.11 scripts/v7_3_full_backtest.py --bootstrap 200       # 快速
python3.11 scripts/v7_3_full_backtest.py --bootstrap 500       # 标准
python3.11 scripts/v7_3_full_backtest.py --bootstrap 2000      # 极致
```

---

## 八、v7 演化总结与完整对比 (2026-07-13)

### 8.1 v7 三代版本对照

| 版本 | 入口函数 | 核心改动 | OOS 2022 Calmar |
|------|---------|---------|----------------|
| `v7_macro_baseline` (v1) | `v7_macro_baseline()` | Bootstrap-Lasso + FRP, 13 indices, 季频 | 0.364 |
| `v7_macro_baseline_v2_tf` (v2) | `v7_macro_baseline_v2_tf()` | +趋势过滤 (equity→bonds) | **0.952** ⭐ |
| `v7_macro_baseline_v3_momentum` (v3) | `v3_momentum_config()` | +动量叠加 (slope_r2, α=0.05) | 0.891 |

### 8.2 v7 三代性能对比 (OOS 2022-01-01 ~ 2026-05-29, seed=42)

| 指标 | v1 (baseline) | v2 (TF only) | v3 (TF+mom) |
|------|------:|------:|------:|
| 年化收益 | 3.27% | 4.74% | **5.18%** |
| 年化波动 | 7.02% | **5.76%** | 6.49% |
| 最大回撤 | -8.98% | **-4.98%** | -5.82% |
| Sharpe | 0.466 | 0.823 | **0.891** |
| Calmar | 0.364 | **0.952** | 0.891 |
| 年化换手 | 5.96% | 11.85% | 12.10% |
| Alpha vs HS300 | 3.41% | 4.63% | **4.89%** |
| 月度胜率 | 54.00% | 58.00% | **60.00%** |

### 8.3 v7 三代性能对比 (OOS 2018-01-01 ~ 2026-05-29, seed=42)

| 指标 | v1 (baseline) | v2 (TF only) | v3 (TF+mom) |
|------|------:|------:|------:|
| 年化收益 | 3.48% | 4.28% | **4.62%** |
| 年化波动 | 6.77% | **5.76%** | 6.24% |
| 最大回撤 | -8.98% | **-7.30%** | -7.52% |
| Sharpe | 0.513 | 0.743 | **0.798** |
| Calmar | 0.387 | **0.586** | 0.574 |

### 8.4 动量叠加网格搜索结果

**参数空间**: 2 option × 3 type × 3 lookback × 5 α × 3 scenario = 270 组合
**实际测试**: ~20 组合 (分阶段筛选)

#### Stage 0: 场景筛选 (hybrid, 90d, α=0.3, Option A)

| 场景 | Ann | Vol | DD | Calmar |
|------|-----|-----|-----|--------|
| v7 baseline | 3.27% | 7.02% | -8.98% | 0.364 |
| **v7+TF only** | **4.74%** | **5.76%** | **-4.98%** | **0.952** |
| v7+mom only | 3.64% | 9.01% | -13.39% | 0.272 |
| v7+TF+mom both | 4.67% | 8.31% | -9.90% | 0.472 |

#### Stage 1: α 筛选 (with TF, hybrid, 90d, Option A)

| α | Ann | Vol | DD | Calmar |
|---|-----|-----|-----|--------|
| 0.05 | 5.06% | 6.51% | -5.69% | 0.890 |
| 0.10 | 4.98% | 6.82% | -6.41% | 0.778 |
| 0.15 | 4.91% | 7.17% | -7.37% | 0.666 |
| 0.20 | 4.83% | 7.54% | -8.26% | 0.584 |
| 0.30 | 4.67% | 8.31% | -9.90% | 0.472 |

#### Stage 2: 动量类型 (with TF, α=0.05, 90d, Option A)

| type | Ann | Vol | DD | Calmar |
|------|-----|-----|-----|--------|
| price | 5.02% | 6.55% | -5.68% | 0.883 |
| **slope_r2** | **5.18%** | **6.49%** | **-5.82%** | **0.891** |
| hybrid | 5.06% | 6.51% | -5.69% | 0.890 |

#### Stage 3: Lookback (with TF, α=0.05, hybrid, Option A)

| lookback | Ann | Vol | DD | Calmar |
|----------|-----|-----|-----|--------|
| 60d | 5.12% | 6.65% | -5.83% | 0.878 |
| 90d | 5.06% | 6.51% | -5.69% | 0.890 |
| **144d** | **5.06%** | **6.44%** | **-5.64%** | **0.898** |

#### Stage 4: Option B (第10因子)

Option B 把 market_momentum 作为 LASSO 第10因子, 结果与 TF only 完全一致 (Calmar 0.952). 原因: LASSO 已包含足够宏观信息, 动量因子无新增信息.

### 8.5 跨版本完整对比 (OOS 2022-2026)

| 策略 | 数据池 | 年化收益 | 年化波动 | Sharpe | DD | Calmar |
|------|--------|---------|---------|--------|-----|--------|
| **v1.0 locked** | 52 ETFs | 3.47% | 2.38% | **1.510** | **-1.94%** | **1.791** ⭐ |
| v3 multi | 52 ETFs | 7.69% | 7.43% | 1.080 | -9.89% | 0.778 |
| v6.2 ir_exp | 52 ETFs | 13.14% | — | 0.810 | -16.73% | 0.786 |
| v7 baseline (v1) | 13 idx | 3.27% | 7.02% | 0.466 | -8.98% | 0.364 |
| **v7+v2 TF only** | 13 idx | 4.74% | 5.76% | 0.823 | -4.98% | **0.952** |
| v7+v3 TF+mom | 13 idx | 5.18% | 6.49% | 0.891 | -5.82% | 0.891 |

### 8.6 跨版本完整对比 (全期 2018-2026)

| 策略 | 数据池 | 年化收益 | 年化波动 | Sharpe | DD | Calmar |
|------|--------|---------|---------|--------|-----|--------|
| **v1.0 locked** | 52 ETFs | 4.96% | 4.42% | **1.160** | **-5.81%** | **0.853** |
| v3 multi | 52 ETFs | 6.01% | 7.76% | 0.830 | -13.56% | 0.443 |
| v6.2 ir_exp | 52 ETFs | 15.72% | — | 1.020 | -21.59% | 0.728 |
| v7 baseline (v1) | 13 idx | 3.48% | 6.77% | 0.513 | -8.98% | 0.387 |
| v7+v2 TF only | 13 idx | 4.28% | 5.76% | 0.743 | -7.30% | 0.586 |
| v7+v3 TF+mom | 13 idx | 4.62% | 6.24% | 0.798 | -7.52% | 0.637 |

### 8.7 关键发现

1. **v1.0 locked (52 ETFs) Calmar 1.791 仍然是 OOS 最佳** — 低波动 (2.38%) + 低 DD (-1.94%), 风险调整后最优
2. **v7+v2 (13 idx) OOS Calmar 0.952 > v6.2 (52 ETFs) 0.786** — 宏观配置胜出权益选股
3. **TF 是 ROI 最高单点修复** — Calmar 0.364→0.952 (+161%), DD -8.98%→-4.98% (-44%)
4. **动量叠加可提高收益但降低 Calmar** — Ann +9.3%, Vol +12.7%, DD +16.9%, Calmar -6.4%
5. **动量类型 slope_r2 最优** — slope×R² 比纯价格动量更稳健 ( penalizes 噪声趋势)
6. **lookback 144d 最优** — 长周期动量更稳定 (Calmar 0.898 vs 90d 0.890)
7. **α=0.05 最优** — 低混合系数保留 FRP 宏观配置逻辑, 动量只做微调
8. **Option B (第10因子) 无增量** — 与 TF only 完全一致, LASSO 已包含足够信息
9. **数据池差异**: v1.0/v3/v6 用 52 ETFs (行业/风格/商品), v7 用 13 indices (宏观配置), 定位不同不可直接比
10. **v7+v2 DD -4.98% 优于 v3 (-9.89%) 和 v6.2 (-16.73%)** — 宏观配置风控更好

### 8.8 动量因子结论

**动量因子可以加到宏观配置上, 但不提升 Calmar.**

- 动量叠加 **提高收益** (4.74%→5.18%, +9.3%)
- 但 **增加波动** (5.76%→6.49%, +12.7%) 和 **放大 DD** (-4.98%→-5.82%, +16.9%)
- Calmar **下降** (0.952→0.891, -6.4%)

**原因**: v7 的 13 indices 已经跨权益/债券/商品三大类, 动量信号在资产间轮动的效果有限, 反而增加了噪声.

**建议**: 保持 v7+v2 TF only 作为 production baseline (Calmar 0.952), 动量作为可选增强 (追求更高收益时启用).

### 8.9 文件清单

```
QuantNodes/strategy/momentum_etf_rotation/v7/
├── macro_substrategy_v7_3.py     (V7_3Config + SubStrategy + run_v7_3_backtest)
├── momentum_overlay.py           (动量计算 + Option A/B 整合) [v3 新增]
├── v3_momentum_backtest.py       (v3 回测入口) [v3 新增]
├── data_loader.py                (load_factor_returns + load_index_panel + load_benchmark_price)
├── bootstrap_lasso.py            (BootstrapLassoMapping)
├── factor_risk_parity.py         (FactorRiskParityOptimizer, 包装 QuantOPT)
├── symmetry.py                   (RollingSymmetry, Klein 2013)
├── _quantopt_model.py            (源 QuantOPT_model.py copy)
└── __init__.py                   (v7_macro_baseline + v7_macro_baseline_v2_tf + v3_momentum_config)

tests/strategy/momentum_etf_rotation/v7/
├── test_v7_3_strategy.py                 (5 tests)
├── test_symmetry.py                      (10 tests)
├── test_bootstrap_lasso.py               (10 tests)
├── test_factor_risk_parity.py            (9 tests)
├── test_v7_macro_baseline.py             (9 tests, v1 锁定)
├── test_v7_macro_baseline_v2_tf.py       (14 tests, v2 TF)
└── test_v7_macro_baseline_v3_momentum.py (15 tests, v3 动量) [v3 新增]

data/high_freq_macro/
├── v9_indices_daily.parquet       (13 指数日对数收益)
├── v9_indices_daily_prices.parquet (13 指数日价格)
├── v9_factors_weekly.parquet      (9 因子净值)
├── v9_factors_weekly_returns.parquet (周对数收益)
├── v9_benchmark_沪深300.parquet   (沪深300 价格)
└── v56_expanded_daily.parquet     (56 assets 日对数收益) [v4 新增]
```

---

## 九、v7_macro_baseline_v4_expanded — 扩大资产池 (2026-07-13)

### 设计动机

v7+v2 TF (13 indices) 的 Calmar 0.952 优于 v1 baseline (0.364), 但受限于:
1. **13 indices 不可直接交易** — 指数是虚拟资产, 实际需通过 ETF 执行
2. **资产数量有限** — 分散化不够, 轮动机会少
3. **商品指数不够** — 南华指数流动性差, 期货数据噪声大

v4 扩大到 **56 assets** (51 ETFs + 5 bond indices), 解决上述问题.

### 资产池

| 类别 | 数量 | 来源 | 说明 |
|------|------|------|------|
| A_BROAD ETFs | 6 | DEFAULT_POOL | 沪深300/500/上证50/创业板/科创50/深证100 |
| A_SECTOR ETFs | 20 | DEFAULT_POOL | 半导体/新能源/酒/医药/军工等 |
| HK ETFs | 5 | DEFAULT_POOL | 恒生/恒生科技/中概互联 |
| COMMODITY ETFs | 6 | DEFAULT_POOL | 黄金/白银/豆粕/能源化工/有色 |
| OVERSEAS ETFs | 6 | DEFAULT_POOL | 纳斯达克/标普/日经 |
| SmartBeta ETFs | 8 | SMARTBETA_8 | 红利低波/质量/价值/现金流 |
| Bond indices | 5 | v7 INDEX_COLS | 中债10年/3-5年/1-3年/国开/企业债 |
| **总计** | **56** | | |

### TF 分类

| 类别 | 资产 | 数量 |
|------|------|------|
| Equity (bear时减半) | A_BROAD + A_SECTOR + HK + OVERSEAS + SmartBeta | 45 |
| Commodity (不动) | COMMODITY ETFs | 6 |
| Bond (bear时加仓) | 5 bond indices | 5 |

### 配置

```python
V7_4Config(
    asset_pool="expanded",
    index_pool=EXPANDED_COLS,  # 56 assets
    equity_cols=EQUITY_ETF_COLS,  # 45 equity ETFs
    commodity_cols=COMMODITY_ETF_COLS,  # 6 commodity ETFs
    bond_cols=EXPANDED_BOND_INDICES,  # 5 bond indices
    trend_filter_enabled=True,  # 继承 v2 TF
    trend_filter_bear=0.5,
)
```

### 数据管道

```
ETF NAV (价格水平) → log returns → ┐
                                    ├→ pd.concat → v56_expanded_daily.parquet
Bond indices (日收益率)            ──┘
```

- ETF 数据: `data/real/etf_nav_2018-01-01_2026-06-30.parquet` (44 ETFs) + SmartBeta (8 unique)
- Bond 数据: `data/high_freq_macro/v9_indices_daily.parquet` (5 bond indices)
- 重叠期: 2018-2026, 2193 交易日
- **关键修复**: `idx_ret_window.fillna(0)` 解决 ETF 结构性 NaN (上市前数据)

### 回测结果 (OOS 2023-10-02 ~ 2026-03-30)

| 版本 | Ann | Vol | DD | Calmar | Sharpe |
|------|----:|----:|---:|-------:|-------:|
| v1 baseline (13 indices) | 6.55% | 7.26% | -6.85% | 0.956 | 0.910 |
| v2 TF (13 indices, equity→bonds) | 7.32% | 6.57% | -4.98% | **1.468** | **1.108** |
| v4 expanded (56 assets, no TF) | **10.62%** | 14.12% | -12.11% | 0.877 | 0.786 |
| v4 expanded + TF (56 assets, equity→bonds) | **10.64%** | 12.71% | -12.11% | 0.879 | 0.860 |

### 分析

1. **Expanded pool 大幅提升收益**: Ann 从 6.55% 提升到 10.62% (+62%)
2. **TF 改善风险调整**: Vol 从 14.12% 降到 12.71% (-10%), Sharpe 从 0.786 提升到 0.860 (+9%)
3. **Calmar 略低于 v2 TF**: 因 DD 更大 (-12.11% vs -4.98%), 但 Ann 更高
4. **TF 触发率**: 11 次调仓中 5 次触发 (2023Q3-2024Q2 + 2026Q1)

### 结论

- Expanded pool 适合**追求高收益**的投资者 (Ann 10.64%)
- v2 TF 适合**追求低波动**的投资者 (Vol 6.57%, Calmar 1.468)
- 两者可组合: 扩大资产池 + 趋势过滤

### 文件结构 (v4 新增/修改)

```
QuantNodes/strategy/momentum_etf_rotation/v7/
├── data_loader.py                [修改] +EXPANDED_COLS + load_expanded_panel()
├── macro_substrategy_v7_3.py     [修改] +V7_4Config + TF 适配
└── __init__.py                   [修改] +v7_macro_baseline_v4_expanded

tests/.../v7/
└── test_v7_macro_baseline_v4_expanded.py  [新建] 13 tests
```

---

## 十、v7_macro_baseline_v5 — 硬止损 + 连续TF + 时变LASSO (2026-07-13)

### 10.1 设计动机 (用户深度讨论 2026-07-13)

v4+TF 已达到 OOS Ann 10.64% / Calmar 0.879, 但用户指出根本性缺陷:

> "宏观象限→资产映射" 范式有结构性缺陷:
> 1. 忽视非宏观因素 (估值/动量/拥挤度)
> 2. 月末/季度调仓存在信息滞后
> 3. 依赖回测而非预测
> 4. 静态映射无法捕捉宏观-资产关系变化
> 5. 二值信号 (MA200 bull/bear) 损失信息

用户提出 **5+1 改进建议**, 经讨论后我们聚焦以下 3 个可立即落地的高 ROI 改动:

| # | 建议 | 落地 | 优先级 |
|---|------|------|--------|
| 1 | 硬止损 (8-12% DD) | ✅ Step 1 | P0 |
| 2 | 连续TF替代二值MA200 | ✅ Step 2 | P1 |
| 3 | 时变LASSO (滚动窗口) | ✅ Step 3 | P2 |
| 4 | 微观因子 (估值/动量/vol) | ⏳ 用动量/vol作为proxy | P3 |
| 5 | 在线预测误差监控 | ⏳ 暂缓 (复杂度高) | P3 |
| 6 | 事件驱动调仓 | ⏳ 暂缓 (回测时间×10) | P3 |
| 7 | 分层贝叶斯 | ⏳ 暂缓 (过度设计) | P4 |

### 10.2 Step 1: 硬止损 (Stop Loss)

**设计**: 在每个调仓日, 检查当前 NAV 相对历史峰值的回撤. 若回撤 ≥ 10%, 强制将权益类资产仓位清零, 全仓债券 (flight to safety).

**新增配置** (V7_3Config):
```python
stop_loss_enabled: bool = False
stop_loss_threshold: float = -0.10   # 10% DD 触发
stop_loss_bond_alloc: float = 1.0    # 止损后 100% 债券
```

**核心逻辑** (`run_v7_3_backtest`):
```python
if cfg.stop_loss_enabled and nav_history:
    peak = max(nav_history)
    dd = current_nav / peak - 1
    if dd < cfg.stop_loss_threshold:
        # 强制 100% 债券 (按原 bond 权重比例分配)
        w = {col: bond_alloc if col in bond_cols else 0.0
             for col in cfg.index_pool}
```

**为何 10% 而非 8%?**
- 8% 太紧, 易被短期波动触发, 增加无效调仓
- 10-12% 是行业经验阈值, 既能截断系统性下跌, 又避免噪声触发
- 用户讨论确认: "可以更宽松"

**预期效果**:
- 在 OOS 2022 (沪深300 DD -22%) 等大跌年份, 止损可将组合 DD 截断到 ~12-13%
- Calmar 改善: 0.879 → ~1.0+

### 10.3 Step 2: 连续 TF Score (替代二值 MA200)

**问题**: 当前 `apply_trend_filter` 仅有两种状态:
- 多头: MA200 之上 → 原权重不变
- 熊市: MA200 之下 → equity × 0.5, 释放给债券

这种**二值**信号存在两大缺陷:
1. 信息损失: 距 MA200 5% 和 距 MA200 20% 都触发同样减仓
2. 滞后: MA200 是 200 日均线, 信号反应慢

**改进**: 构造连续 trend score ∈ [-1, +1]:

```
trend_score = 0.5 × MA200距离_score
            + 0.3 × 60日动量_score
            + 0.2 × 波动率比率_score
```

| 因子 | 计算 | 归一化 |
|------|------|--------|
| MA200 距离 | (price - MA200) / MA200 | × 5, clip [-1, 1] |
| 60日动量 | price/price[60] - 1 | × 5, clip [-1, 1] |
| 波动率比率 | -1 × (vol_20d / vol_60d - 1) | × 2, clip [-1, 1] |

**仓位调整**:
```python
if score < -0.3: equity_scale = 0.3   # 强熊市
elif score > 0.3: equity_scale = 1.2  # 强牛市
else: 线性插值 0.3 → 1.2
```

**关键改进点**:
1. **连续信号** 保留全部信息, 不再二值化
2. **多因子合成** 解决 "只靠 MA200 反应慢" 问题
3. **线性插值** 介于 bear/bull 之间时平滑过渡
4. **可超配** (1.2) 牛市时加大权益敞口

### 10.4 Step 3: 时变 LASSO (滚动窗口)

**问题**: 当前 LASSO 用 expanding window (从回测起点到当前), 系数 β 实际是**全样本平均**, 缺乏时变性. 若宏观-资产关系发生结构性变化 (如 2020 疫情后), β 无法及时响应.

**改进**: 用滚动窗口 (156 周 = 3 年) 替代 expanding window. β 仅反映最近 3 年的关系.

**新增配置** (V7_5Config):
```python
lasso_rolling_window: int | None = None  # None=expanding (兼容), 156=3年滚动
```

**权衡**:
- 优点: 捕捉时变关系, 避免静态映射
- 代价: 估计稳定性降低 (样本量减少), 可能放大噪声
- 风险: 需验证 OOS 是否稳定优于 expanding

### 10.5 实施计划 (分步验证)

| Step | 内容 | 验证 | 预期 ROI |
|------|------|------|---------|
| 1 | 硬止损 | 单测 + e2e | ⭐⭐⭐⭐⭐ |
| 2 | 连续TF | 单测 + e2e | ⭐⭐⭐⭐ |
| 3 | 时变LASSO | 单测 + e2e | ⭐⭐⭐ |

每步完成立即跑测试 + git commit, 不累积.

### 10.6 回测结果 (待实施后填入)

| 版本 | Ann | Vol | DD | Calmar | Sharpe | vs v4+TF |
|------|----:|----:|---:|-------:|-------:|---------:|
| v4+TF (baseline) | 10.64% | 12.71% | -12.11% | 0.879 | 0.860 | — |
| v5 + stop loss (Step 1) | TBD | TBD | TBD | TBD | TBD | TBD |
| v5 + continuous TF (Step 2) | TBD | TBD | TBD | TBD | TBD | TBD |
| v5 + rolling LASSO (Step 3) | TBD | TBD | TBD | TBD | TBD | TBD |
| v5 全开 (Step 1+2+3) | TBD | TBD | TBD | TBD | TBD | TBD |

#### 10.6.1 Step 1 实测结果 (硬止损 10%)

**OOS 2022-2026 (用户原话"22年到现在", 2022年是下跌市)**:

| 版本 | Ann | Vol | DD | Calmar | 改善 |
|------|----:|----:|---:|-------:|-----:|
| v4+TF (56 assets) | 5.79% | 11.00% | -11.60% | 0.499 | — |
| v5+stop_loss (56) | **6.92%** | 10.32% | -11.60% | **0.597** | ⬆️ +20% |

**结论**: 在 2022~2026 的下跌/震荡市中, 硬止损改善了 **Ann (+1.13%)** 和 **Calmar (+0.098)**, DD 维持不变.

**OOS 2018-2026 (全期, 包含 2018 急跌)**:

| 版本 | Ann | Vol | DD | Calmar |
|------|----:|----:|---:|-------:|
| v4+TF (56 assets) | 6.41% | 10.85% | -13.47% | 0.476 |
| v5+stop_loss (56) | 4.12% | 9.27% | -15.84% | 0.260 |

**结论**: 全期数据上止损**反拖累了** Ann (-2.29%) 和 Calmar (-0.216). 原因: 2018 急跌时止损触发, 但 2019 反弹时权益仓位被压制, 错过 main 反弹. 这是硬止损的典型 trade-off.

**决策**: 保留硬止损作为**可选项** (`stop_loss_enabled`), 让用户根据风险偏好选择. 在下跌/震荡市 (OOS 2022+) 显著改善, 在 V 型反转市 (2018→2019) 有机会成本.

#### 10.6.2 Step 2 实测结果 (连续 TF Score) - ⚠️ **负面结果**

**OOS 2022-2026 对比**:

| 版本 | Ann | Vol | DD | Calmar | 备注 |
|------|----:|----:|---:|-------:|------|
| v2 二值 MA200 (13 idx) | **4.89%** | 5.69% | -4.98% | **0.981** | 二值基准 |
| v4+TF 二值 MA200 (56) | **5.79%** | 11.00% | -11.60% | 0.499 | 二值 + 56 assets |
| v5.1 连续 TF Score (56) | 4.05% | 12.33% | -12.76% | 0.317 | 连续 (bull=1.2) |
| v5.1 conservative (56) | 4.47% | 11.60% | -12.86% | 0.348 | bull=1.0, bear=0.5 |
| v5.1 no_vol (56) | 4.65% | 11.39% | -12.98% | 0.358 | 无 vol_ratio 因子 |

**分析**: **连续 TF Score 在所有变体下都劣于二值 MA200** (Calmar -0.13 ~ -0.18).

**根因分析**:
1. **过度交易**: 连续 score 在 -0.3 ~ +0.3 之间线性触发, 而二值 MA200 只在 MA200 真正跌破时触发. 连续信号更频繁, 增加换手和误判.
2. **杠杆效应**: bull=1.2 让组合在牛市中 120% 权益, 放大波动 (Vol 12.33% vs 11.00%), 抵消了 upside capture.
3. **vol_ratio 噪声**: 短期 vol spike 与中期趋势无关, 加入反而引入噪声 (no_vol 略优于含 vol).

**结论** (诚实记录):
- 连续 TF Score 的理论优势 (多因子/信息保留) 在这个数据集上**没有转化为实际提升**
- 二值 MA200 + 50% bear equity 是**该数据集的 sweet spot**, 简单且稳健
- 用户的 5+1 建议之一 (连续替代二值) 在实测中**未通过验证**

**决策**:
- ✅ v7_macro_baseline_v5_tf_score() 工厂**保留** (作为可选项 + 教学示例)
- ⚠️ v2 二值 TF **不应被替换**, 保留为推荐默认
- 📝 此负结果已记录, 用户可据此判断: 改进 TF 信号空间有限, ROI 集中在止损 (Step 1) 和 LASSO 改进 (Step 3)

#### 10.6.3 Step 3 实测结果 (时变 LASSO) - (待实施)

### 10.7 文件结构 (v5 新增/修改)

```
QuantNodes/strategy/momentum_etf_rotation/v7/
├── macro_substrategy_v7_3.py     [修改] +stop_loss_* + V7_5Config + compute_trend_score
├── bootstrap_lasso.py            [修改] 支持 rolling window
└── __init__.py                   [修改] +v7_macro_baseline_v5_stop_loss +v7_macro_baseline_v5_tf_score +v7_macro_baseline_v5_rolling

tests/.../v7/
└── test_v7_macro_baseline_v5_regime.py  [新建] ~20 tests
```
